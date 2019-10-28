/**
 * @brief Countmin-CU sketch
 *
 * CUDA implementation
 *
 * @file sketch.cpp
 * @author Hans Lehnert
 */

#include <iostream>
#include <string>
//#include <vector>
#include <chrono>
#include <limits>
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
const unsigned int HASH_TABLE_BITS = 10;

const unsigned int MAX_BUFFER_SIZE = 1 << 22;  // 4 MB

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
// synchronization. Block size should be equal or less to the max_length.
// If complete_sequences is true, only starting points where all lengths can be
// extracted will be used. This is used to handle processing in chunks
// correctly
__global__ void countmincu(
        char* data,
        const size_t data_length,
        const int min_length,
        const int max_length,
        const bool complete_sequences,
        Sketch<int32_t, N_HASH, HASH_BITS>* sketches,
        HashTable<HASH_TABLE_BITS>* heavy_hitters
) {
    // A delay is added to prevent threads from updating the same sketch on
    // concurrent updates
    // const int32_t delay = threadIdx.x % max_length;

    const uint32_t last_pos =
        data_length - (complete_sequences ? max_length : min_length);

    uint32_t start_pos = threadIdx.x; // Start position refers to the start of the string
    for (; start_pos < last_pos; start_pos += blockDim.x) {
        bool sequence_end = false;

        HashSet hashes = {0};
        uint64_t encoded_kmer = 0;

        for (int i = 0; i < max_length + blockDim.x - 1; i++) {
            int pos = i;// - delay;

            if (pos < 0)
                continue;
            if (pos >= max_length)
                break;
            if (start_pos + pos >= data_length)
                break;

            uint8_t symbol;

            switch (data[start_pos + pos]) {
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

            packedArraySet<2, uint64_t>(&encoded_kmer, pos, symbol);

            hashes.vec ^= d_seeds[pos][symbol].vec;

            __syncthreads();

            // Add to sketch
            int32_t counters[N_HASH];
            int32_t min_hits;

            //hashTableInsert(&heavy_hitters[0], pos, max_length);

            if (pos >= min_length - 1) {
                for (int j = 0; j < N_HASH; j++) {
                    counters[j] = sketches[pos - min_length + 1][j][hashes.val[j]];
                }

                min_hits = counters[0];

                for (int j = 1; j < N_HASH; j++)
                    min_hits = min(min_hits, counters[j]);

                __syncthreads();

                for (int j = 0; j < N_HASH; j++) {
                    if (counters[j] == min_hits) {
                        atomicAdd(&sketches[pos - min_length + 1][j][hashes.val[j]], 1);
                    }
                }

                if (min_hits >= d_thresholds[pos - min_length + 1]) {
                    hashTableInsert<HASH_TABLE_BITS>(
                        &heavy_hitters[pos - min_length + 1],
                        encoded_kmer,
                        min_hits
                    );
                }
            }
        }
    }
}

__global__ void controlStage(
        char* data,
        const size_t data_length,
        const int32_t min_length,
        const int32_t max_length,
        const bool complete_sequences,
        const int growth,
        HashTable<HASH_TABLE_BITS>* heavy_hitters
) {
    // Copy hash table to shared memory
    __shared__ HashTable<HASH_TABLE_BITS> s_heavy_hitters;
    for (int i = threadIdx.x; i < s_heavy_hitters.n_slots; i += blockDim.x) {
        s_heavy_hitters.slots[i].used = heavy_hitters[blockIdx.x].slots[i].used;
        s_heavy_hitters.slots[i].key = heavy_hitters[blockIdx.x].slots[i].key;
        s_heavy_hitters.slots[i].value =
            heavy_hitters[blockIdx.x].slots[i].value / growth;
    }

    const uint32_t last_pos =
        data_length - (complete_sequences ? max_length : min_length);

    for (int32_t start_pos = threadIdx.x; start_pos < last_pos; start_pos += blockDim.x) {
        bool sequence_end = false;

        uint64_t encoded_kmer = 0;

        for (int i = 0; i < max_length; i++) {
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

            if (i == blockIdx.x + min_length - 1) {
                int32_t* counter;
                bool found = hashTableGet<HASH_TABLE_BITS>(
                    &s_heavy_hitters,
                    encoded_kmer,
                    &counter
                );

                if (found) {
                    atomicAdd(counter, -1);
                }
            }
        }
    }

    __syncthreads();

    // Copy table back to global memory
    for (int i = threadIdx.x; i < sizeof(s_heavy_hitters); i += blockDim.x) {
        ((char*)&heavy_hitters[blockIdx.x])[i] = ((char*)&s_heavy_hitters)[i];
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
    HashTable<HASH_TABLE_BITS>* h_heavyhitters =
        new HashTable<HASH_TABLE_BITS>[settings.n_length];
    HashTable<HASH_TABLE_BITS>* d_heavyhitters;
    gpuErrchk(cudaMalloc(
        &d_heavyhitters,
        sizeof(HashTable<HASH_TABLE_BITS>) * settings.n_length
    ));
    gpuErrchk(cudaMemset(
        d_heavyhitters,
        0,
        sizeof(HashTable<HASH_TABLE_BITS>) * settings.n_length
    ));

    // Allocate gpu memory for data transfer
    char* d_data_test;
    gpuErrchk(cudaMalloc(&d_data_test, MAX_BUFFER_SIZE));

    char* d_data_control;
    gpuErrchk(cudaMalloc(&d_data_control, MAX_BUFFER_SIZE));

    // Sketches data structures
    Sketch<int32_t, N_HASH, HASH_BITS>* d_sketches;
    gpuErrchk(cudaMalloc(
        &d_sketches,
        sizeof(Sketch<int32_t, N_HASH, HASH_BITS>) * settings.n_length
    ));

    // Start time measurement
    auto start_time = std::chrono::steady_clock::now();

    int i = 0;
    int active_buffer = 0;
    int final_chunk = false;

    while (!final_chunk) {
        uint64_t batch_size;
        uint64_t bytes_left = test_file.size() - i;

        if (bytes_left <= MAX_BUFFER_SIZE) {
            batch_size = bytes_left;
            final_chunk = true;
        }
        else {
            batch_size = MAX_BUFFER_SIZE;
        }

        // There are 2 buffers used for transfering data to GPU an we
        // alternate using them
        char* buffer = d_data_test + MAX_BUFFER_SIZE * active_buffer;

        gpuErrchk(cudaMemcpy(
            buffer, test_file.data() + i, batch_size, cudaMemcpyHostToDevice));

        int num_blocks = 1;  // Must be 1
        int block_size = 256;
        countmincu<<<num_blocks, block_size>>>(
            buffer,
            batch_size,
            settings.min_length,
            settings.max_length,
            !final_chunk,
            d_sketches,
            d_heavyhitters
        );

        std::clog
            << "countminCU" << std::endl
            << '\t' << reinterpret_cast<size_t>(buffer) << std::endl
            << '\t' << batch_size << std::endl
            << '\t' << settings.min_length << std::endl
            << '\t' << settings.max_length << std::endl
            << '\t' << !final_chunk << std::endl
            << '\t' << reinterpret_cast<size_t>(d_sketches) << std::endl
            << '\t' << reinterpret_cast<size_t>(d_heavyhitters) << std::endl;

        i += batch_size - settings.max_length + 1;
        active_buffer = active_buffer ? 0 : 1;
    }

    gpuErrchk(cudaDeviceSynchronize());

    i = 0;
    final_chunk = false;
    while (!final_chunk) {
        uint64_t batch_size;
        uint64_t bytes_left = control_file.size() - i;

        if (bytes_left <= MAX_BUFFER_SIZE) {
            batch_size = bytes_left;
            final_chunk = true;
        }
        else {
            batch_size = MAX_BUFFER_SIZE;
        }

        // There are 2 buffers used for transfering data to GPU an we
        // alternate using them
        char* buffer = d_data_test + MAX_BUFFER_SIZE * active_buffer;

        gpuErrchk(cudaMemcpy(
            buffer, control_file.data(), batch_size, cudaMemcpyHostToDevice));

        int num_blocks = settings.n_length;  // Blocks process a single length
        int block_size = 512;
        controlStage<<<num_blocks, block_size>>>(
            buffer,
            batch_size,
            settings.min_length,
            settings.max_length,
            !final_chunk,
            settings.growth,
            d_heavyhitters
        );

        std::clog
            << "controlStage" << std::endl
            << '\t' << reinterpret_cast<size_t>(buffer) << std::endl
            << '\t' << batch_size << std::endl
            << '\t' << settings.min_length << std::endl
            << '\t' << settings.max_length << std::endl
            << '\t' << !final_chunk << std::endl
            << '\t' << settings.growth << std::endl
            << '\t' << reinterpret_cast<size_t>(d_heavyhitters) << std::endl;

        i += batch_size - settings.max_length + 1;
        active_buffer = active_buffer ? 0 : 1;
    }

    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(
        h_heavyhitters,
        d_heavyhitters,
        sizeof(HashTable<HASH_TABLE_BITS>) * settings.n_length,
        cudaMemcpyDeviceToHost
    ));

    gpuErrchk(cudaDeviceSynchronize());

    // End time measurement
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> total_diff = end_time - start_time;

    std::clog << "Execution time: " << total_diff.count() << " s" << std::endl;

    // Print heavy-hitters
    int heavy_hitters_count = 0;

    int count = 0;
    for (int n = 0; n < settings.n_length; n++) {
        for (int i = 0; i < h_heavyhitters->n_slots; i++) {
            if (h_heavyhitters[n].slots[i].used
                    && h_heavyhitters[n].slots[i].value > 0) {
                heavy_hitters_count++;
                std::cout
                    << sequenceToString(
                        h_heavyhitters[n].slots[i].key, settings.min_length + n)
                    << " "
                    << h_heavyhitters[n].slots[i].value
                    << std::endl;
                count += h_heavyhitters[n].slots[i].value;
            }
        }
    }


    Sketch<int, N_HASH, HASH_BITS>* h_sketches;
    h_sketches = new Sketch<int, N_HASH, HASH_BITS>[settings.n_length];
    gpuErrchk(cudaMemcpy(
        h_sketches,
        d_sketches,
        sizeof(Sketch<int, N_HASH, HASH_BITS>) * settings.n_length,
        cudaMemcpyDeviceToHost
    ));

    count = 0;
    for (int i = 0; i < N_HASH; i++) {
        for (int j = 0; j < (1 << HASH_BITS); j++) {
            count += h_sketches[0][i][j];
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
    std::clog << "Count total: " << count << std::endl;

    // Free shared memory
    // cudaFree(d_data);
    // cudaFree(d_sketch);
    // cudaFree(d_hashes);
    // cudaFree(d_heavy_hitters);
    // cudaFree(heavy_hitters_count);
    // cudaFreeHost(h_data);

    return 0;
}
