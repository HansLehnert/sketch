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
#include <mutex>
#include <unordered_map>

#include "cuda_error.h"
#include "fasta.hpp"
#include "MappedFile.hpp"
#include "Sketch.hpp"
#include "PackedArray.cuh"
#include "HashTable.cuh"

const unsigned int MAX_LENGTH = 32;

const unsigned int N_HASH = 4;
const unsigned int HASH_BITS = 14;

const unsigned int MAX_BUFFER_SIZE = 1 << 25;

union HashSet {
    uint64_t vec;
    uint16_t val[N_HASH];
};

// Seeds
__constant__ HashSet d_seeds[MAX_LENGTH][4];
__constant__ int d_thresholds[MAX_LENGTH];

struct SketchSettings {
    int min_length;
    int max_length;
    int n_length;

    std::vector<int> threshold;

    float growth;
};


// Populates sketch using the countmin-cu strategy and extract heavy-hitters
//
// Must be called with only 1 block due to heavy dependence on thread
// synchronization.
__global__ void countmincu(
        char* data,
        size_t data_length,
        Sketch<int32_t, N_HASH, HASH_BITS>* sketches,
        int min_length,
        int max_length,
        HashTable<HASH_BITS>* heavy_hitters
) {
    const uint32_t start_index = threadIdx.x;
    const uint32_t stride = blockDim.x;

    const int32_t offset = threadIdx.x;
    const uint32_t last_start_pos = data_length - min_length + 1;

    for (uint32_t start_pos = start_index; start_pos < last_start_pos; start_pos += stride) {
        bool sequence_end = false;

        int i;
        HashSet hashes = {0};
        uint64_t encoded_kmer = 0;

        // Hashing of the first symbols, which don't yet generate a k-mer of
        // the wanted length
        for (i = 0; i < max_length - 1; i++) {
            uint8_t symbol;

            switch (data[start_pos + i]) {
            case 'A':
                symbol = 0;
                break;
            case 'C':
                symbol = 1;
                break;
            case 'T':
                symbol = 2;
                break;
            case 'G':
                symbol = 3;
                break;
            default:
                sequence_end = true;
                break;
            }

            if (sequence_end)
                break;

            packedArraySet<2, uint64_t>(&encoded_kmer, i, symbol);

            hashes.vec ^= d_seeds[i][symbol].vec;
        }

        if (sequence_end) {
            continue;
        }

        // Hashes for relevant lengths
        for (; i < max_length; i++) {
            uint8_t symbol;

            switch (data[start_pos + i]) {
            case 'A':
                symbol = 0;
                break;
            case 'C':
                symbol = 1;
                break;
            case 'T':
                symbol = 2;
                break;
            case 'G':
                symbol = 3;
                break;
            default:
                sequence_end = true;
                break;
            }

            packedArraySet<2, uint64_t>(&encoded_kmer, i, symbol);

            // Add to sketch
            /*__shared__ extern uint32_t min_hits[];
            if (hash_index == 0)
                min_hits[start_index] = std::numeric_limits<uint32_t>::max();*/

            __syncthreads();

            hashes.vec ^= d_seeds[i][symbol].vec;

            int32_t counters[N_HASH];
            for (int j = 0; j < N_HASH; j++) {
                counters[j] = sketches[i - min_length + 1][j][hashes.val[j]];
            }

            int32_t min_hits = counters[0];
            for (int j = 1; j < N_HASH; j++)
                min_hits = min(min_hits, counters[j]);

            //atomicMin(&min_hits[start_index], counter);

            __syncthreads();

            for (int j = 0; j < N_HASH; j++) {
                if (counters[j] == min_hits) {
                    atomicAdd(&sketches[i - min_length + 1][j][hashes.val[j]], 1);
                }
            }

            if (min_hits >= d_thresholds[i - min_length + 1]) {
                hashTableInsert<HASH_BITS>(
                    &heavy_hitters[i - min_length + 1],
                    encoded_kmer,
                    min_hits
                );
            }
        }
    }
}


void sketchWorker(
        const SketchSettings& settings,
        std::mutex* mutex,
        int start,
        int stride,
        const uint16_t* d_hashes,
        const std::vector<unsigned long>* test_data,
        const std::vector<unsigned long>* control_data,
        const std::vector<unsigned char>* test_lengths,
        const std::vector<unsigned char>* control_lengths,
        std::vector<std::unordered_map<uint64_t, int>>* heavy_hitters_vec) {

    int test_data_size = test_data->size();

    uint16_t* h_hashes;
    cudaMallocHost(&h_hashes, N_HASH * MAX_BUFFER_SIZE * sizeof(uint16_t));

    for (int n = start; n < settings.n_length; n += stride) {
        uint64_t mask = ~(~0UL << ((settings.min_length + n) * 2));

        Sketch<uint16_t, N_HASH, HASH_BITS> sketch = {0};
        int length = settings.min_length + n;

        std::unordered_map<uint64_t, int> heavy_hitters;

        for (int m = 0; m < test_data_size; m += MAX_BUFFER_SIZE) {
            unsigned long batch_size;
            if (test_data_size - m > MAX_BUFFER_SIZE) {
                batch_size = MAX_BUFFER_SIZE;
            }
            else {
                batch_size = test_data_size - m;
            }
            mutex[0].lock();

            // Copy hashes from device
            gpuErrchk(cudaMemcpy(
                h_hashes,
                &d_hashes[N_HASH * MAX_BUFFER_SIZE * n],
                N_HASH * batch_size * sizeof(uint16_t),
                cudaMemcpyDeviceToHost
            ));

            // Hash values
            for (int i = 0; i < batch_size; i++) {
                if ((*test_lengths)[m + i] < length)
                    continue;

                int min_hits = std::numeric_limits<int>::max();
                uint16_t* hashes = &h_hashes[i * N_HASH];

                for (int j = 0; j < N_HASH; j++) {
                    if (sketch[j][hashes[j]] < min_hits) {
                        min_hits = sketch[j][hashes[j]];
                    }
                }

                for (int j = 0; j < N_HASH; j++) {
                    if (sketch[j][hashes[j]] == min_hits) {
                        sketch[j][hashes[j]]++;
                    }
                }

                min_hits++;

                if (min_hits >= settings.threshold[n]) {
                    uint64_t sequence = (*test_data)[m + i] & mask;
                    heavy_hitters[sequence] = min_hits;
                }
            }

            mutex[1].unlock();
        }

        for (auto& i : heavy_hitters) {
            i.second /= settings.growth;
        }

        // Control step
        int control_data_size = control_data->size();
        for (int i = 0; i < control_data_size; i++) {
            if ((*control_lengths)[i] < length)
                continue;

            std::unordered_map<uint64_t, int>::iterator counter;
            counter = heavy_hitters.find((*control_data)[i] & mask);
            if (counter != heavy_hitters.end()) {
                counter->second--;
            }
        }

        // Select only the heavy-hitters not in the control set
        auto i = heavy_hitters.begin();
        while (i != heavy_hitters.end()) {
            if (i->second <= 0) {
                i = heavy_hitters.erase(i);
            }
            else {
                i++;
            }
        }

        (*heavy_hitters_vec)[n] = heavy_hitters;
    }

    cudaFreeHost(h_hashes);
}


// Move pointer to the postion after the last new line
void seekLastNewline(char** pos) {
    while(*(*pos - 1) != '\n') {
        (*pos)--;
    }
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
    const size_t n_seeds = sizeof(d_seeds) / sizeof(uint16_t);
    uint16_t h_seeds[sizeof(d_seeds) / sizeof(uint16_t)];
    for (unsigned int i = 0; i < n_seeds; i++) {
        h_seeds[i] = rand() & ~(~0UL << HASH_BITS);
    }
    gpuErrchk(cudaMemcpyToSymbol(d_seeds, h_seeds, sizeof(d_seeds)));
    gpuErrchk(cudaDeviceSynchronize());

    // Copy thresholds
    gpuErrchk(cudaMemcpyToSymbol(
        d_thresholds,
        settings.threshold.data(),
        sizeof(int) * settings.threshold.size()
    ));

    // Load memory mapped files
    MappedFile test_file = MappedFile::load(argv[1]);
    MappedFile control_file = MappedFile::load(argv[2]);

    // Heavy-hitters containers
    HashTable<HASH_BITS>* h_heavyhitters =
        new HashTable<HASH_BITS>[settings.n_length];
    HashTable<HASH_BITS>* d_heavyhitters;
    gpuErrchk(cudaMalloc(
        &d_heavyhitters,
        sizeof(HashTable<HASH_BITS>) * settings.n_length
    ));
    gpuErrchk(cudaMemset(
        d_heavyhitters,
        0,
        sizeof(HashTable<HASH_BITS>) * settings.n_length
    ));

    // Allocate gpu memory for data transfer
    char* d_data_test;
    uint16_t* d_hashes;
    gpuErrchk(cudaMalloc(&d_data_test, MAX_BUFFER_SIZE));
    gpuErrchk(cudaMalloc(
        &d_hashes,
        MAX_BUFFER_SIZE * settings.n_length * N_HASH* sizeof(uint16_t)
    ));

    // Sketches data structures
    Sketch<int32_t, N_HASH, HASH_BITS>* d_sketches;
    gpuErrchk(cudaMalloc(
        &d_sketches,
        sizeof(Sketch<int32_t, N_HASH, HASH_BITS>) * settings.n_length
    ));

    // Start time measurement
    auto start_time = std::chrono::steady_clock::now();

    int batch_size = test_file.size() < MAX_BUFFER_SIZE ? test_file.size() : MAX_BUFFER_SIZE;

    gpuErrchk(cudaMemcpy(
        d_data_test,
        test_file.data(),
        batch_size,
        cudaMemcpyHostToDevice
    ));

    int block_size = 4 * settings.max_length;//256;
    int num_blocks = 1;//batch_size / block_size;
    countmincu<<<num_blocks, block_size>>>(
        d_data_test,
        batch_size,
        d_sketches,
        settings.min_length,
        settings.max_length,
        d_heavyhitters
    );

    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(
        h_heavyhitters,
        d_heavyhitters,
        sizeof(HashTable<HASH_BITS>) * settings.n_length,
        cudaMemcpyDeviceToHost
    ));

    // End time measurement
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> total_diff = end_time - start_time;

    std::clog << "Execution time: " << total_diff.count() << " s" << std::endl;

    // Print heavy-hitters
    int heavy_hitters_count = 0;

    for (int n = 0; n < settings.n_length; n++) {
        for (int i = 0; i < h_heavyhitters->n_slots; i++) {
            if (h_heavyhitters[n].slots[i].used) {
                heavy_hitters_count++;
                std::cout
                    << sequenceToString(
                        h_heavyhitters[n].slots[i].key, settings.min_length + n)
                    << std::endl;
            }
        }
    }

    /*for (int n = 0; n < settings.n_length; n++) {
        heavy_hitters_count += heavy_hitters[n].size();
        std::clog
            << "Heavy-hitters (length " << settings.min_length + n << "): "
            << heavy_hitters[n].size() << std::endl;

        for (auto x : heavy_hitters[n]) {
            std::cout
                << sequenceToString(x.first, settings.min_length + n)
                << std::endl;
        }
    }*/

    std::clog << "Heavy-hitters (total): " << heavy_hitters_count << std::endl;

    // Free shared memory
    // cudaFree(d_data);
    // cudaFree(d_sketch);
    // cudaFree(d_hashes);
    // cudaFree(d_heavy_hitters);
    // cudaFree(heavy_hitters_count);
    // cudaFreeHost(h_data);

    return 0;
}
