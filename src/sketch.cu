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
#include <chrono>
#include <limits>
#include <unordered_map>
#include <cooperative_groups.h>

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

using namespace cooperative_groups;

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
// Grid size should be less than max_length for synchronization to be effective
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

    grid_group grid = this_grid();
    thread_block block = this_thread_block();

    const uint32_t last_pos =
        data_length - (complete_sequences ? max_length : min_length);
    const uint32_t stride = blockDim.x * gridDim.x;

    for (int i = 0; i < blockIdx.x; i++) {
        grid.sync();
    }

    // We need to do the same amount of iterations in every thread in order
    // to not miss a grid sync
    for (int i = 0; i < (data_length + stride - 1) / stride; i++) {
        // Start position refers to the start of the sequence
        const uint32_t start_pos = stride * i + blockDim.x * blockIdx.x + threadIdx.x;

        bool sequence_end = false;

        HashSet hashes = {0};
        uint64_t encoded_kmer = 0;

        for (int i = 0; i < max_length; i++) {
            grid.sync();

            int pos = i;

            if (start_pos > last_pos || start_pos + pos >= data_length) {
                block.sync();
                continue;
            }

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

            if (sequence_end) {
                block.sync();
                continue;
            }

            packedArraySet<2, uint64_t>(&encoded_kmer, pos, symbol);

            hashes.vec ^= d_seeds[pos][symbol].vec;

            // Add to sketch
            int32_t counters[N_HASH];
            int32_t min_hits;

            if (pos >= min_length - 1) {
                for (int j = 0; j < N_HASH; j++) {
                    counters[j] = sketches[pos - min_length + 1][j][hashes.val[j]];
                }

                min_hits = counters[0];

                for (int j = 1; j < N_HASH; j++)
                    min_hits = min(min_hits, counters[j]);

                block.sync();

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
            else {
                block.sync();
            }
        }
    }

    for (int i = 0; i < gridDim.x - blockIdx.x - 1; i++) {
        grid.sync();
    }
}

// Execute the control stage for the k-mer extracting process
//
// Grid size should be equal to the amount of k-mer lengths to evaluate
__global__ void controlStage(
        char* data,
        const size_t data_length,
        const int32_t min_length,
        const int32_t max_length,
        const bool complete_sequences,
        const float growth,
        HashTable<HASH_TABLE_BITS>* heavy_hitters
) {
    // Copy hash table to shared memory
    __shared__ HashTable<HASH_TABLE_BITS> s_heavy_hitters;
    for (int i = threadIdx.x; i < s_heavy_hitters.n_slots; i += blockDim.x) {
        s_heavy_hitters.slots[i] = heavy_hitters[blockIdx.x].slots[i];
        s_heavy_hitters.slots[i].value /= growth;
    }

    const uint32_t last_pos =
        data_length - (complete_sequences ? max_length : min_length);
    const uint32_t length = min_length + blockIdx.x;

    for (int32_t start_pos = threadIdx.x; start_pos < last_pos; start_pos += blockDim.x) {
        bool sequence_end = false;

        uint64_t encoded_kmer = 0;

        for (int i = 0; i < length; i++) {
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
        }

        if (sequence_end)
            continue;

        // Search for the sequence in the heavy-hitters hash table
        int32_t* counter;
        bool found = hashTableGet<HASH_TABLE_BITS>(
            &s_heavy_hitters,
            encoded_kmer,
            &counter
        );

        if (found) {
            atomicSub(counter, 1);
        }
    }

    __syncthreads();

    // Copy table back to global memory
    for (int i = threadIdx.x; i < s_heavy_hitters.n_slots; i += blockDim.x) {
        heavy_hitters[blockIdx.x].slots[i] = s_heavy_hitters.slots[i];
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

    // Get device attributes
    int cuda_device;
    gpuErrchk(cudaGetDevice(&cuda_device));
    int max_threads_per_block;
    gpuErrchk(cudaDeviceGetAttribute(
        &max_threads_per_block,
        cudaDevAttrMaxBlockDimX,
        cuda_device
    ));

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
    char* d_transfer_area;
    gpuErrchk(cudaMalloc(&d_transfer_area, MAX_BUFFER_SIZE * 2));

    // Sketches data structures
    Sketch<int32_t, N_HASH, HASH_BITS>* d_sketches;
    gpuErrchk(cudaMalloc(
        &d_sketches,
        sizeof(Sketch<int32_t, N_HASH, HASH_BITS>) * settings.n_length
    ));

    // Create a stream to avoid the default stream and allow transfer and
    // execution overlap
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Start time measurement
    auto start_time = std::chrono::steady_clock::now();

    // Test stage
    int i = 0;
    int active_buffer = 0;
    int final_chunk = false;

    while (!final_chunk) {
        size_t batch_size;
        size_t bytes_left = test_file.size() - i;

        if (bytes_left <= MAX_BUFFER_SIZE) {
            batch_size = bytes_left;
            final_chunk = true;
        }
        else {
            batch_size = MAX_BUFFER_SIZE;
        }

        // There are 2 buffers used for transfering data to GPU an we
        // alternate using them
        char* buffer = d_transfer_area + MAX_BUFFER_SIZE * active_buffer;

        gpuErrchk(cudaMemcpyAsync(
            buffer,
            test_file.data() + i,
            batch_size,
            cudaMemcpyHostToDevice,
            stream
        ));

        uint32_t num_blocks = settings.max_length;
        uint32_t block_size = 512;
        bool complete_sequences = !final_chunk;
        void* args[] = {
            &buffer,
            &batch_size,
            &settings.min_length,
            &settings.max_length,
            &complete_sequences,
            &d_sketches,
            &d_heavyhitters
        };
        gpuErrchk(cudaLaunchCooperativeKernel(
            (void*)countmincu,
            {num_blocks, 1, 1},
            {block_size, 1, 1},
            args,
            0,  // No shared memory use
            stream
        ));

#ifdef DEBUG
        std::clog
            << "countminCU" << std::endl
            << '\t' << reinterpret_cast<size_t>(buffer) << std::endl
            << '\t' << batch_size << std::endl
            << '\t' << settings.min_length << std::endl
            << '\t' << settings.max_length << std::endl
            << '\t' << !final_chunk << std::endl
            << '\t' << reinterpret_cast<size_t>(d_sketches) << std::endl
            << '\t' << reinterpret_cast<size_t>(d_heavyhitters) << std::endl;
#endif

        i += batch_size - settings.max_length + 1;
        active_buffer = active_buffer ? 0 : 1;
    }

    gpuErrchk(cudaDeviceSynchronize());

    // Control stage
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
        char* buffer = d_transfer_area + MAX_BUFFER_SIZE * active_buffer;

        gpuErrchk(cudaMemcpyAsync(
            buffer,
            control_file.data() + i,
            batch_size,
            cudaMemcpyHostToDevice,
            stream
        ));

        // We only need to scale the counters once, so the growth is set to 1
        // beyond the first kernel call
        // TODO: Move scaling to separate kernel
        float growth = i == 0 ? settings.growth : 1;

        int num_blocks = settings.n_length;  // Blocks process a single length
        int block_size = max_threads_per_block;
        controlStage<<<num_blocks, block_size, 0, stream>>>(
            buffer,
            batch_size,
            settings.min_length,
            settings.max_length,
            !final_chunk,
            growth,
            d_heavyhitters
        );

#ifdef DEBUG
        std::clog
            << "controlStage" << std::endl
            << '\t' << reinterpret_cast<size_t>(buffer) << std::endl
            << '\t' << batch_size << std::endl
            << '\t' << settings.min_length << std::endl
            << '\t' << settings.max_length << std::endl
            << '\t' << !final_chunk << std::endl
            << '\t' << settings.growth << std::endl
            << '\t' << reinterpret_cast<size_t>(d_heavyhitters) << std::endl;
#endif

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

    for (int n = 0; n < settings.n_length; n++) {
        int partial_count = 0;

        for (int i = 0; i < h_heavyhitters->n_slots; i++) {
            if (!h_heavyhitters[n].slots[i].used
                    || h_heavyhitters[n].slots[i].value <= 0) {
                continue;
            }

                std::cout
                    << sequenceToString(
                        h_heavyhitters[n].slots[i].key, settings.min_length + n)
                    << std::endl;

            partial_count++;
            }

        heavy_hitters_count += partial_count;

        std::clog
            << "Heavy-hitters (" << settings.min_length + n << "): "
            << partial_count << std::endl;
    }

    std::clog << "Heavy-hitters (total): " << heavy_hitters_count << std::endl;

    Sketch<int, N_HASH, HASH_BITS>* h_sketches;
    h_sketches = new Sketch<int, N_HASH, HASH_BITS>[settings.n_length];
    gpuErrchk(cudaMemcpy(
        h_sketches,
        d_sketches,
        sizeof(Sketch<int, N_HASH, HASH_BITS>) * settings.n_length,
        cudaMemcpyDeviceToHost
    ));

    int count = 0;
    for (int i = 0; i < N_HASH; i++) {
        for (int j = 0; j < (1 << HASH_BITS); j++) {
            count += h_sketches[0][i][j];
        }
    }
    delete[] h_sketches;

    std::clog << "Count total: " << count << std::endl;

    // Free memory
    cudaFree(d_transfer_area);
    cudaFree(d_sketches);
    cudaFree(d_heavyhitters);
    delete[] h_heavyhitters;

    return 0;
}
