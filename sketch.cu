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
#include <future>
#include <unordered_map>

#include "fasta.hpp"
#include "MappedFile.hpp"


const unsigned int MAX_LENGTH = 28;

const unsigned int N_HASH = 4;
const unsigned int HASH_BITS = 14;

// Seeds
__constant__ uint16_t d_seeds[N_HASH * MAX_LENGTH * 2];


struct SketchSettings {
    int min_length;
    int max_length;
    int n_length;

    std::vector<int> threshold;

    float growth;
};


struct Sketch {
    int32_t count[N_HASH][1 << HASH_BITS];
};


/**
 * @brief
 * Compute H3 hash
 *
 * Compute the H3 hash on a set of keys using constant memory seeds. Keys are
 * shifted by the offset, to start the hash.
 */
template <int n_hash>
__global__ void hashH3(
        int n,
        int bits,
        uint64_t* keys,
        uint16_t* src,
        uint16_t* dst,
        int offset) {
    unsigned int start_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = start_index; i < n; i += stride) {
        for (int j = 0; j < n_hash; j++)
            dst[i * n_hash + j] = src[i * n_hash + j];

        unsigned long key = keys[i] >> offset;
        for (int j = 0; j < bits; j++) {
            if (key & 1) {
                for (int k = 0; k < n_hash; k++) {
                    dst[i * n_hash + k] ^=
                        d_seeds[(j + offset) * n_hash + k];
                }
            }
            key >>= 1;
        }
    }
}


void sketchWorker(
        const SketchSettings& settings,
        int start,
        int stride,
        const uint16_t* d_hashes,
        const std::vector<unsigned long>& test_data,
        const std::vector<unsigned long>& control_data,
        const std::vector<unsigned char>& test_lengths,
        const std::vector<unsigned char>& control_lengths,
        std::vector<std::unordered_map<uint64_t, int>>* heavy_hitters_vec) {

    int test_data_size = test_data.size();
    uint16_t* h_hashes = new uint16_t[N_HASH * test_data_size];

    for (int n = start; n < settings.n_length; n += stride) {
        uint64_t mask = ~(~0UL << ((settings.min_length + n) * 2));

        Sketch sketch = {0};
        int length = settings.min_length + n;

        std::unordered_map<uint64_t, int> heavy_hitters;

        // Copy hashes from device
        cudaMemcpy(
            h_hashes,
            &d_hashes[N_HASH * test_data_size * n],
            N_HASH * test_data_size * sizeof(uint16_t),
            cudaMemcpyDeviceToHost
        );

        // Hash values
        for (int i = 0; i < test_data_size; i++) {
            if (test_lengths[i] < length)
                continue;

            int min_hits = std::numeric_limits<int>::max();
            uint16_t* hashes = &h_hashes[i * N_HASH];

            for (int j = 0; j < N_HASH; j++) {
                if (sketch.count[j][hashes[j]] < min_hits) {
                    min_hits = sketch.count[j][hashes[j]];
                }
            }

            for (int j = 0; j < N_HASH; j++) {
                if (sketch.count[j][hashes[j]] == min_hits) {
                    sketch.count[j][hashes[j]]++;
                }
            }

            min_hits++;

            if (min_hits >= settings.threshold[n]) {
                uint64_t sequence = test_data[i] & mask;
                heavy_hitters[sequence] = min_hits;
            }
        }

        for (auto& i : heavy_hitters) {
            i.second /= settings.growth;
        }

        // Control step
        for (int i = 0; i < control_data.size(); i++) {
            if (control_lengths[i] < length)
                continue;

            std::unordered_map<uint64_t, int>::iterator counter;
            counter = heavy_hitters.find(control_data[i] & mask);
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
    uint16_t* h_seeds = new uint16_t[N_HASH * MAX_LENGTH * 2];
    for (int i = 0; i < N_HASH * MAX_LENGTH * 2; i++)
        h_seeds[i] = rand() & ~(~0UL << HASH_BITS);
    cudaMemcpyToSymbol(d_seeds, h_seeds, sizeof(d_seeds));
    cudaDeviceSynchronize();
    delete[] h_seeds;

    // Load memory mapped files
    MappedFile test_file = MappedFile::load(argv[1]);
    MappedFile control_file = MappedFile::load(argv[2]);

    // Heavy-hitters containers
    std::vector<std::unordered_map<uint64_t, int>> heavy_hitters;
    heavy_hitters.resize(settings.n_length);

    // Start time measurement
    auto start_time = std::chrono::steady_clock::now();

    // Parse data set and transfer to device
    std::vector<unsigned char> test_lengths;
    std::vector<unsigned long> test_data = parseFasta(
        test_file.data(),
        test_file.size(),
        settings.min_length,
        ~(~0UL << (settings.max_length * 2)),
        &test_lengths);

    unsigned long n_data_test = test_data.size();
    unsigned long* d_data_test;

    auto preprocessing_time = std::chrono::steady_clock::now();

    // Copy sequences to device
    cudaMalloc(&d_data_test, n_data_test * sizeof(unsigned long));
    cudaMemcpyAsync(
        d_data_test,
        test_data.data(),
        n_data_test * sizeof(unsigned long),
        cudaMemcpyHostToDevice);

    // Allocate memory for hashes and sketches
    uint16_t* d_hashes;
    size_t hash_data_size =
        settings.n_length * n_data_test * N_HASH * sizeof(uint16_t);

    cudaMalloc(&d_hashes, hash_data_size);

    // Calculate hashes for the first length.
    // The first is a special case since it needs to hash over MIN_LENGTH
    // symbols instead of only one
    int block_size = 256;
    int num_blocks = 16;

    hashH3<N_HASH><<<block_size, num_blocks, 0>>>(
        n_data_test,
        settings.min_length * 2,
        d_data_test,
        &d_hashes[0],
        &d_hashes[0],
        0);

    // Compute for the rest of the k-mers lengths
    for (int i = 1; i < settings.n_length; i++) {
        hashH3<N_HASH><<<num_blocks, block_size, 0>>>(
            n_data_test,
            2,
            d_data_test,
            &d_hashes[n_data_test * N_HASH * (i - 1)],
            &d_hashes[n_data_test * N_HASH * i],
            (settings.min_length + i - 1) * 2);
    }

    // Sync device in separate thread to measure total hashing time
    std::chrono::time_point<std::chrono::steady_clock> hash_time;
    auto cuda_sync = std::async(
        [&] {
            cudaDeviceSynchronize();
            hash_time = std::chrono::steady_clock::now();
        }
    );

    // Parse control file during hash calculation
    std::vector<unsigned char> control_lengths;
    std::vector<unsigned long> control_data = parseFasta(
        control_file.data(),
        control_file.size(),
        settings.min_length * 2,
        ~(~0UL << (settings.max_length * 2)),
        &control_lengths);

    cuda_sync.wait();

    // Create threads
    int n_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    threads.reserve(n_threads);

    for (int i = 0; i < n_threads; i++) {
        threads.emplace_back(
            sketchWorker,
            settings,
            i,
            n_threads,
            d_hashes,
            test_data,
            control_data,
            test_lengths,
            control_lengths,
            &heavy_hitters
        );
    }

    for (int i = 0; i < threads.size(); i++) {
        threads[i].join();
    }

    // End time measurement
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> preprocessing_diff = preprocessing_time - start_time;
    std::chrono::duration<double> hash_diff = hash_time - preprocessing_time;
    std::chrono::duration<double> total_diff = end_time - start_time;

    std::clog << "Preprocessing time: " << preprocessing_diff.count() << " s" << std::endl;
    std::clog << "Hashing time: " << hash_diff.count() << " s" << std::endl;
    std::clog << "Execution time: " << total_diff.count() << " s" << std::endl;
    std::clog << "Data vectors: " << n_data_test << std::endl;

    // Print heavy-hitters
    int heavy_hitters_count = 0;

    for (int n = 0; n < settings.n_length; n++) {
        heavy_hitters_count += heavy_hitters[n].size();
        std::clog
            << "Heavy-hitters (length " << settings.min_length + n << "): "
            << heavy_hitters[n].size() << std::endl;

        for (auto x : heavy_hitters[n]) {
            std::cout
                << sequenceToString(x.first, settings.min_length + n)
                << std::endl;
        }
    }

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
