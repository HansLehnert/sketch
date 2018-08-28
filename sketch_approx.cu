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


const unsigned int N_HASH = 4;
const unsigned int M = 14;
const unsigned int RHO = 145;


// __constant__ unsigned short d_seeds[N_HASH * 32];


/**
 * @brief Compute H3 hash
 */
template <int bits>
__device__ unsigned int hashH3(unsigned long key, unsigned short* seeds) {
    unsigned int result = 0;
    for (int i = 0; i < bits; i++) {
        if (key & 1)
            result ^= seeds[i];
        key >>= 1;
    }
    return result;
}


template <int n_hash, int bits>
__global__ void countminCu(
        unsigned int n,
        unsigned long* keys,
        unsigned short* seeds,
        unsigned int* sketch,
        unsigned long* heavy_hitters,
        unsigned int* count) {
    unsigned int start_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = start_index; i < n; i += stride) {
        unsigned int min_hits = ~0;
        unsigned short hashes[n_hash];

        for (unsigned int j = 0; j < n_hash; j++) {
            hashes[j] = hashH3<32>(keys[i], seeds + j * bits);
            if (sketch[hashes[j] + (j << M)] < min_hits) {
                min_hits = sketch[hashes[j] + (j << M)];
            }
        }

        for (unsigned int j = 0; j < n_hash; j++) {
            if (sketch[hashes[j] + (j << M)] == min_hits) {
                atomicAdd(&sketch[hashes[j] + (j << M)], 1);
            }
        }

        if (min_hits + 1 == RHO) {
            unsigned long pos = atomicAdd(count, 1);
            heavy_hitters[pos] = keys[i];
        }
    }
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Missing dataset file." << std::endl;
        return 1;
    }

    // Generate hash vectors
    unsigned short* d_seeds;
    unsigned short h_seeds[N_HASH * 32];
    for (unsigned int i = 0; i < N_HASH * 32; i++)
        h_seeds[i] = rand() & ((1 << M) - 1);
    cudaMalloc(&d_seeds, sizeof(h_seeds));
    cudaMemcpy(d_seeds, h_seeds, sizeof(h_seeds), cudaMemcpyHostToDevice);

    // Start time measurement
    auto start = std::chrono::steady_clock::now();

    // Parse data set and transfer to device
    std::ifstream dataset_file(argv[1]);
    std::vector<unsigned long> data_vector = parseFasta(dataset_file, 16);
    dataset_file.close();

    unsigned long n_data = data_vector.size();
    unsigned long* h_data;
    unsigned long* d_data;
    cudaHostAlloc(
        &h_data, n_data * sizeof(unsigned long), cudaHostAllocWriteCombined);
    cudaMemcpyAsync(
        h_data,
        data_vector.data(),
        n_data * sizeof(unsigned long),
        cudaMemcpyHostToHost);

    cudaMalloc(&d_data, n_data * sizeof(unsigned long));
    cudaMemcpyAsync(
        d_data,
        h_data,
        n_data * sizeof(unsigned long),
        cudaMemcpyHostToDevice);

    // Hash values
    unsigned int* sketch;
    unsigned long* d_heavy_hitters;
    unsigned int* heavy_hitters_count;

    cudaMalloc(&sketch, (N_HASH << M) * sizeof(unsigned int));
    cudaMalloc(&d_heavy_hitters, (1 << 10) * sizeof(unsigned long));
    cudaMallocManaged(&heavy_hitters_count, sizeof(unsigned int));

    cudaDeviceSynchronize();

    int block_size = 256;
    int num_blocks = 16;

    countminCu<N_HASH, 32><<<num_blocks, block_size>>>(
        n_data,
        d_data,
        d_seeds,
        sketch,
        d_heavy_hitters,
        heavy_hitters_count);

    cudaPeekAtLastError();
    cudaDeviceSynchronize();

    unsigned long* h_heavy_hitters = new unsigned long[*heavy_hitters_count];
    cudaMemcpy(
        h_heavy_hitters,
        d_heavy_hitters,
        *heavy_hitters_count * sizeof(unsigned long),
        cudaMemcpyDeviceToHost);

    // End time measurement
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::clog << "Execution time: " << diff.count() << " s" << std::endl;
    std::clog << "Data vectors: " << n_data << std::endl;
    std::clog << "Heavy-hitters: " << *heavy_hitters_count << std::endl;

    // Print heavy-hitters
    for (int i = 0; i < *heavy_hitters_count; i++) {
        std::cout << sequenceToString(h_heavy_hitters[i], 16) << std::endl;
    }

    // Free shared memory
    cudaFree(d_data);
    cudaFreeHost(h_data);

    return 0;
}
