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
#include <unordered_set>

#include "fasta.hpp"
#include "MappedFile.hpp"


// K-mers lengths
const unsigned int MIN_LENGTH = 10;
const unsigned int MAX_LENGTH = 20;
const unsigned int N_LENGTH = MAX_LENGTH - MIN_LENGTH + 1;

// Number of hashes to use in the sketch
const unsigned int N_HASH = 4;

// Number of bits for the hashing seeds. Also determines the sketch size.
const unsigned int HASH_BITS = 14;

// Thresholds for use in heavy-hitters detection in sketch frequencies
constexpr unsigned int THRESHOLD[] = {
    365, 308, 257, 161, 150, 145, 145, 145, 145, 145, 145};

// Growth parameter for control step
const float GROWTH = 2;

// Seeds
__constant__ unsigned short d_seeds[N_HASH * MAX_LENGTH * 2];


/**
 * @brief
 * Compute H3 hash
 *
 * Compute the H3 hash on a set of keys using constant memory seeds. Keys are
 * shifted by the offset, to start the hash.
 */
template <int bits, int n_hash>
__global__ void hashH3(
        unsigned int n,
        unsigned long* keys,
        unsigned short* src,
        unsigned short* dst,
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


/**
 * @brief Extract heavy-hitters using a Countmin-CU sketch
 */
template <int n_hash>
__global__ void countminCu(
        unsigned int n,
        unsigned int threshold,
        unsigned int* sketch,
        unsigned short* hashes,
        unsigned long* heavy_hitters,
        int* frequencies,
        unsigned int* heavy_hitters_count) {
    unsigned int threads = blockDim.x * gridDim.x;
    unsigned int stride = (n + threads - 1) / threads;
    unsigned int start = stride * (blockIdx.x * blockDim.x + threadIdx.x);
    unsigned int end = start + stride;
    end = end < n ? end : n;

    for (unsigned int i = start; i < end; i++) {
        unsigned int min_hits = ~0U;
        unsigned int* hits[n_hash];
        unsigned short current_hashes[n_hash];

        for (unsigned int j = 0; j < n_hash; j++) {
            current_hashes[j] = hashes[i * n_hash + j];

            hits[j] = sketch + current_hashes[j] + (j << HASH_BITS);

            if (*hits[j] < min_hits) {
                min_hits = *hits[j];
            }
        }

        for (unsigned int j = 0; j < n_hash; j++) {
            if (*hits[j] == min_hits) {
                atomicAdd(hits[j], 1);
            }
        }

        if (min_hits + 1 == threshold) {
            unsigned long pos = atomicAdd(heavy_hitters_count, 1);
            heavy_hitters[pos] = i;
        }
    }

    __syncthreads();

    if (start == 0) {
        // Remove duplicates
        int k = 0;
        for (int i = 1; i < *heavy_hitters_count; i++) {
            if (heavy_hitters[i] == heavy_hitters[k])
                continue;


            heavy_hitters[++k] = heavy_hitters[i];
        }

        *heavy_hitters_count = k + 1;

        // Gather frequencies
        for (int i = 0; i < *heavy_hitters_count; i++) {
            int* frequency = frequencies + i;
            *frequency = (~0U) >> 1;
            for (unsigned int j = 0; j < n_hash; j++) {
                unsigned short hash = hashes[heavy_hitters[i] * n_hash + j];
                unsigned int hits = sketch[hash + (j << HASH_BITS)];

                if (hits < *frequency) {
                    *frequency = hits;
                }
            }
        }
    }
}


/**
 * @brief Executes the control step for emerging heavy-hitter extarction
 *
 * @param n Amount of keys in the control group
 * @param bits Number of bits in the sequences
 * @param test_keys Keys used for heavy-hitter search
 * @param control_keys Keys in the control group
 * @param heavy_hitters Indexes (position) of the heavy-hitter sequences on the
 * test group
 * @param frequencies Frequencies of the heavy-hitters
 * @param heavy_hitters_count Number of heavy-hitters (in the 'heavy-hitters'
 * array)
 */
__global__ void control(
        unsigned int n,
        int bits,
        unsigned long* test_keys,
        unsigned long* control_keys,
        unsigned long* heavy_hitters,
        int* frequencies,
        unsigned int* heavy_hitters_count) {

    unsigned int threads = blockDim.x * gridDim.x;
    unsigned int stride = (n + threads - 1) / threads;
    unsigned int start = stride * (blockIdx.x * blockDim.x + threadIdx.x);
    unsigned int end = start + stride;
    end = end < n ? end : n;

    unsigned long mask = (~0UL >> (64 - bits));

    __shared__ unsigned long heavy_hitters_keys[1 << 8];
    __shared__ int target_frequencies[1 << 8];

    if (start == 0) {
        for (int i = 0; i < *heavy_hitters_count; i++) {
            heavy_hitters_keys[i] = test_keys[heavy_hitters[i]] & mask;
            target_frequencies[i] = frequencies[i] / GROWTH;
        }
    }

    __syncthreads();


    // Decrement counters for matched sequences
    for (unsigned int i = start; i < end; i++) {
        for (unsigned int j = 0; j < *heavy_hitters_count; j++) {
            if ((control_keys[i] & mask) == heavy_hitters_keys[j]) {
                atomicSub(&target_frequencies[j], 1);
                break;
            }
        }
    }

    __syncthreads();

    // Get only non-negative counters
    if (start == 0) {
        int k = 0;
        for (int i = 0; i < *heavy_hitters_count; i++) {
            // heavy_hitters[i] = frequencies[i];
            if (target_frequencies[i] > 0) {
                heavy_hitters[k] = heavy_hitters[i];
                k++;
            }
        }
        *heavy_hitters_count = k;
    }
}


int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout
            << "Usage:" << std::endl
            << '\t' << argv[0] << " test_set control_set" << std::endl;
        return 1;
    }

    // Generate seeds
    unsigned int* h_seeds = new unsigned int[N_HASH * MAX_LENGTH * 2];
    for (unsigned int i = 0; i < N_HASH * MAX_LENGTH * 2; i++)
        h_seeds[i] = rand() & (~0UL >> (64 - HASH_BITS * 2));
    cudaMemcpyToSymbol(d_seeds, h_seeds, sizeof(d_seeds));
    cudaDeviceSynchronize();
    delete h_seeds;

    // Load memory mapped files
    MappedFile test_file = MappedFile::load(argv[1]);
    MappedFile control_file = MappedFile::load(argv[2]);

    // Start time measurement
    auto start = std::chrono::steady_clock::now();

    // Parse data set and transfer to device
    std::vector<unsigned long> test_data = parseFasta(
        test_file.data(), test_file.size(), 20);
    std::vector<unsigned long> control_data = parseFasta(
        control_file.data(), control_file.size(), 20);

    unsigned long n_data_test = test_data.size();
    unsigned long n_data_control = control_data.size();
    unsigned long* d_data_test;
    unsigned long* d_data_control;

    cudaMalloc(&d_data_test, n_data_test * sizeof(unsigned long));
    cudaMemcpyAsync(
        d_data_test,
        test_data.data(),
        n_data_test * sizeof(unsigned long),
        cudaMemcpyHostToDevice);

    cudaMalloc(&d_data_control, n_data_control * sizeof(unsigned long));
    cudaMemcpyAsync(
        d_data_control,
        control_data.data(),
        n_data_control * sizeof(unsigned long),
        cudaMemcpyHostToDevice);

    // Allocate memory for hashes and sketches
    unsigned int* d_sketch[N_LENGTH];
    unsigned short* d_hashes[N_LENGTH];
    unsigned long* d_heavy_hitters[N_LENGTH];
    int* d_frequencies[N_LENGTH];
    unsigned int* heavy_hitters_count;
    cudaStream_t stream[N_LENGTH];

    for (int i = 0; i < N_LENGTH; i++) {
        cudaMalloc(d_sketch + i, (N_HASH << HASH_BITS) * sizeof(unsigned int));
        cudaMalloc(d_hashes + i, n_data_test * N_HASH * sizeof(unsigned short));
        cudaMalloc(d_heavy_hitters + i, (1 << 10) * sizeof(unsigned long));
        cudaMalloc(d_frequencies + i, (1 << 10) * sizeof(int));

        cudaStreamCreate(stream + i);

        cudaMemset(
            d_sketch[i], 0, (N_HASH << HASH_BITS) * sizeof(unsigned int));
    }

    cudaMemset(d_hashes[0], 0, n_data_test * N_HASH * sizeof(unsigned short));

    cudaMallocManaged(&heavy_hitters_count, N_LENGTH * sizeof(unsigned int));
    cudaMemset(heavy_hitters_count, 0, N_LENGTH * sizeof(unsigned int));

    int block_size = 256;
    int num_blocks = 16;
    // Calculate hashes for the first length.
    // The first is a special case since it needs to hash over MIN_LENGTH
    // symbols instead of only one

    hashH3<MIN_LENGTH * 2, N_HASH><<<block_size, num_blocks, 0, stream[0]>>>(
        n_data_test, d_data_test, d_hashes[0], d_hashes[0], 0);

    cudaStreamSynchronize(stream[0]);

    // Compute heavy-hitters for the first length
    countminCu<N_HASH><<<1, 64, 0, stream[0]>>>(
        n_data_test,
        THRESHOLD[0],
        d_sketch[0],
        d_hashes[0],
        d_heavy_hitters[0],
        d_frequencies[0],
        &heavy_hitters_count[0]);

    // Execute the control step for the first length
    control<<<1, block_size, 0, stream[0]>>>(
        n_data_control,
        MIN_LENGTH * 2,
        d_data_test,
        d_data_control,
        d_heavy_hitters[0],
        d_frequencies[0],
        &heavy_hitters_count[0]);

    // Compute for the rest of the k-mers lengths
    for (int i = 1; i < N_LENGTH; i++) {
        hashH3<2, N_HASH><<<num_blocks, block_size, 0, stream[i]>>>(
            n_data_test,
            d_data_test,
            d_hashes[i - 1],
            d_hashes[i],
            (MIN_LENGTH + i - 1) * 2);

        cudaStreamSynchronize(stream[i]);

        countminCu<N_HASH><<<1, 64, 0, stream[i]>>>(
            n_data_test,
            THRESHOLD[i],
            d_sketch[i],
            d_hashes[i],
            d_heavy_hitters[i],
            d_frequencies[i],
            &heavy_hitters_count[i]);

        control<<<1, block_size, 0, stream[i]>>>(
            n_data_control,
            (MIN_LENGTH + i - 1) * 2,
            d_data_test,
            d_data_control,
            d_heavy_hitters[i],
            d_frequencies[i],
            &heavy_hitters_count[i]);
    }


    cudaDeviceSynchronize();

    // Copy heavy hitters from device
    unsigned long* h_heavy_hitters[N_LENGTH];
    for (int i = 0; i < N_LENGTH; i++) {
        cudaStreamSynchronize(stream[i]);

        h_heavy_hitters[i] = new unsigned long[heavy_hitters_count[i]];
        cudaMemcpy(
            h_heavy_hitters[i],
            d_heavy_hitters[i],
            heavy_hitters_count[i] * sizeof(unsigned long),
            cudaMemcpyDeviceToHost);
    }

    // End time measurement
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "Execution time: " << diff.count() << " s" << std::endl;
    std::cout << "Data vectors: " << n_data_test << std::endl;

    // Write heavy-hitters to output file
    std::ofstream heavy_hitters_file("heavy-hitters_cu-pipelined.txt");
    int heavy_hitters_total = 0;

    for (int n = 0; n < N_LENGTH; n++) {
        heavy_hitters_total += heavy_hitters_count[n];
        std::cout
            << "Heavy-hitters (length " << MIN_LENGTH + n << "): "
            << heavy_hitters_count[n] << std::endl;

        for (int m = 0; m < heavy_hitters_count[n]; m++) {
            heavy_hitters_file
                << h_heavy_hitters[n][m]
                // << sequenceToString(test_data[h_heavy_hitters[n][m]], MIN_LENGTH + n)
                << std::endl;
        }
    }

    heavy_hitters_file.close();

    std::cout << "Heavy-hitters (total): " << heavy_hitters_total << std::endl;

    // Free shared memory
    // cudaFree(d_data);
    // cudaFree(d_sketch);
    // cudaFree(d_hashes);
    // cudaFree(d_heavy_hitters);
    // cudaFree(heavy_hitters_count);
    // cudaFreeHost(h_data);

    return 0;
}
