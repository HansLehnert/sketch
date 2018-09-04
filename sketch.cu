/**
 * @brief Countmin-CU sketch
 *
 * CUDA implementation
 *
 * @file sketch.cpp
 * @author Hans Lehnert
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <limits>
#include <thread>
#include <unordered_set>
#include <unordered_map>

#include "fasta.hpp"
#include "MappedFile.hpp"
#include "cuda_error.h"


struct SketchSettings {
    int min_length;
    int max_length;
    int n_length;

    std::vector<int> threshold;

    float growth;
};


const unsigned int MAX_LENGTH = 28;

const unsigned int N_HASH = 4;
const unsigned int HASH_BITS = 14;

const float GROWTH = 2;


unsigned short h_seeds[N_HASH * MAX_LENGTH * 2];
__constant__ unsigned short d_seeds[N_HASH * MAX_LENGTH * 2];


/**
 * @brief Compute H3 hash
 */
unsigned int hashH3(unsigned long key, unsigned short* seeds, int bits) {
    unsigned int result = 0;
    for (int i = 0; i < bits; i++) {
        if (key & 1)
            result ^= seeds[i];
        key >>= 1;
    }
    return result;
}


/**
 * @brief Compute H3 hash
 */
__global__ void hashH3(
        unsigned int n,
        unsigned long* keys,
        unsigned short* hashes,
        int seed_offset,
        int bits) {
    unsigned int start_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = start_index; i < n; i += stride) {
        unsigned long mask = 1;
        hashes[i] = 0;

        for (int j = 0; j < bits; j++) {
            if (keys[i] & mask)
                hashes[i] ^= d_seeds[j + seed_offset * MAX_LENGTH * 2];
            mask <<= 1;
        }
    }
}


void hashWorker(
        SketchSettings settings,
        MappedFile* test_file,
        MappedFile* control_file,
        std::unordered_set<unsigned long>* heavy_hitters,
        int sequence_length,
        unsigned long threshold,
        int device = 0) {

    cudaSetDevice(device);

    // Parse data sets
    std::vector<unsigned long> data_vectors = parseFasta(
        test_file->data(), test_file->size(), sequence_length);
    std::vector<unsigned long> control_vectors = parseFasta(
        control_file->data(), control_file->size(), sequence_length);

    // Create stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Transfer data to device
    unsigned long n_data = data_vectors.size();
    // unsigned long* h_data;
    unsigned long* d_data;
    // cudaHostAlloc(
    //     &h_data, n_data * sizeof(unsigned long), cudaHostAllocWriteCombined);
    // cudaMemcpyAsync(
    //     h_data,
    //     data_vectors.data(),
    //     n_data * sizeof(unsigned long),
    //     cudaMemcpyHostToHost,
    //     stream);

    cudaMalloc(&d_data, n_data * sizeof(unsigned long));
    cudaMemcpy(
        d_data,
        data_vectors.data(),
        n_data * sizeof(unsigned long),
        cudaMemcpyHostToDevice);

    // Hash values
    unsigned short* d_hashes;
    unsigned short* h_hashes = new unsigned short[N_HASH * n_data];
    cudaMalloc(&d_hashes, N_HASH * n_data * sizeof(unsigned short));
    // gpuErrchk(cudaHostAlloc(
    //     &h_hashes,
    //     N_HASH * n_data * sizeof(unsigned short),
    //     cudaHostAllocDefault));

    int block_size = 128;
    int num_blocks = (n_data + block_size - 1) / block_size;

    for (unsigned int i = 0; i < N_HASH; i++) {
        hashH3<<<num_blocks, block_size, 0, stream>>>(
            n_data, d_data, d_hashes + n_data * i, i, sequence_length * 2);
    }

    cudaMemcpyAsync(
        h_hashes,
        d_hashes,
        N_HASH * n_data * sizeof(unsigned short),
        cudaMemcpyDeviceToHost);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    // Find heavy-hitters
    unsigned int sketch[N_HASH * (1 << HASH_BITS)];

    for (unsigned int i = 0; i < n_data; i++) {
        unsigned int min_hits = std::numeric_limits<unsigned int>::max();

        for (unsigned int j = 0; j < N_HASH; j++) {
            if (sketch[h_hashes[i + n_data * j] + (j << HASH_BITS)] < min_hits) {
                min_hits = sketch[h_hashes[i + n_data * j] + (j << HASH_BITS)];
            }
        }

        for (unsigned int j = 0; j < N_HASH; j++) {
            if (sketch[h_hashes[i + n_data * j] + (j << HASH_BITS)] == min_hits) {
                sketch[h_hashes[i + n_data * j] + (j << HASH_BITS)]++;
            }
        }

        if (min_hits + 1 >= threshold) {
            heavy_hitters->insert(data_vectors[i]);
        }
    }

    // Get frequencies for heavy-hitters
    std::unordered_map<unsigned long, int> frequencies;

    for (auto i : *heavy_hitters) {
        frequencies[i] = std::numeric_limits<int>::max();

        for (unsigned int j = 0; j < N_HASH; j++) {
            unsigned int hash = hashH3(i, h_seeds + j * (MAX_LENGTH * 2), sequence_length * 2);
            if (sketch[hash + (j << HASH_BITS)] < frequencies[i]) {
                frequencies[i] = sketch[hash + (j << HASH_BITS)];
            }
        }

        frequencies[i] /= GROWTH;
    }

    // Control step
    for (unsigned int i = 0; i < control_vectors.size(); i++) {
        std::unordered_map<unsigned long, int>::iterator counter;
        counter = frequencies.find(control_vectors[i]);
        if (counter != frequencies.end()) {
            counter->second--;
        }
    }

    // Select only the heavy-hitters not in the control set
    for (auto i : frequencies) {
        if (i.second <= 0) {
            heavy_hitters->erase(heavy_hitters->find(i.first));
        }
    }

    cudaFree(d_data);
    cudaFree(d_hashes);
    // cudaFreeHost(h_data);
    // cudaFreeHost(h_hashes);
    delete[] h_hashes;
}


int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr
            << "Usage:" << std::endl
            << '\t' << argv[0]
            << " test_set control_set min_length max_length threshold_1 ..."
            << std::endl;
        return 1;
    }

    // Configure sketch settings
    SketchSettings settings;
    settings.min_length = atoi(argv[3]);
    settings.max_length = atoi(argv[4]);
    settings.n_length = settings.max_length - settings.min_length + 1;
    settings.growth = 2.0;

    if (argc - 5 < settings.n_length) {
        std::cerr
            << "Missing threshold values. Got "
            << argc - 5
            << ", expected "
            << settings.n_length
            << std::endl;
        return 1;
    }

    for (int i = 5; i < argc; i++) {
        settings.threshold.push_back(atoi(argv[i]));
    }

    // Generate seeds
    for (unsigned int i = 0; i < N_HASH * MAX_LENGTH * 2; i++)
        h_seeds[i] = rand() & ((1 << HASH_BITS) - 1);
    cudaMemcpyToSymbol(d_seeds, h_seeds, sizeof(d_seeds));
    cudaDeviceSynchronize();

    // Load memory mapped files
    MappedFile test_file = MappedFile::load(argv[1]);
    MappedFile control_file = MappedFile::load(argv[2]);

    // Start time measurement
    auto start_time = std::chrono::steady_clock::now();

    std::vector<std::unordered_set<unsigned long>> heavy_hitters;
    heavy_hitters.resize(settings.n_length);

    int n_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> worker_threads;
    worker_threads.reserve(n_threads);

    int device_count;
    cudaGetDeviceCount(&device_count);

    for (int n = 0; n < n_threads; n++) {
        worker_threads.emplace_back(
            hashWorker,
            settings,
            &test_file,
            &control_file,
            &heavy_hitters[n],
            settings.min_length + n,
            settings.threshold[n],
            0);
    }

    for (int n = 0; n < worker_threads.size(); n++) {
        worker_threads[n].join();
    }

    // End time measurement
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff_total = end_time - start_time;

    std::clog << "Execution time: " << diff_total.count() << " s" << std::endl;

    // Print heavy-hitters
    int heavy_hitters_count = 0;

    for (int n = 0; n < settings.n_length; n++) {
        heavy_hitters_count += heavy_hitters[n].size();
        std::clog
            << "Heavy-hitters (length " << settings.min_length + n << "): "
            << heavy_hitters[n].size() << std::endl;


        for (auto x : heavy_hitters[n]) {
            std::cout
                << sequenceToString(x, settings.min_length + n)
                << std::endl;
        }
    }

    std::clog << "Heavy-hitters (total): " << heavy_hitters_count << std::endl;
    std::clog << "CUDA Devices: " << device_count << std::endl;

    return 0;
}
