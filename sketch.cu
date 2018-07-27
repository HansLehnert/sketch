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


__constant__ unsigned short d_seeds[N_HASH * 32];


/**
 * @brief Compute H3 hash
 */
__global__ void hashH3(
        unsigned int n,
        unsigned long* keys,
        unsigned short* hashes,
        int seed_offset) {
    unsigned int start_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = start_index; i < n; i += stride) {
        unsigned int mask = 1;
        hashes[i] = 0;

        for (int j = 0; j < 32; j++) {
            if (keys[i] & mask)
                hashes[i] ^= d_seeds[j + seed_offset * 32];
            mask <<= 1;
        }
    }
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Missing dataset file." << std::endl;
        return 1;
    }

    // Generate hash vectors
    unsigned short h_seeds[N_HASH * 32];
    for (unsigned int i = 0; i < N_HASH * 32; i++)
        h_seeds[i] = rand() & ((1 << M) - 1);
    cudaMemcpyToSymbol(d_seeds, h_seeds, sizeof(d_seeds));

    // Create sketch
    unsigned int sketch[N_HASH * (1 << M)];
    //cudaMallocManaged(&sketch, N_HASH * (1 << M) * sizeof(unsigned int));

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
    unsigned short* d_hashes;
    unsigned short* h_hashes;
    cudaMalloc(&d_hashes, N_HASH * n_data * sizeof(unsigned short));
    cudaHostAlloc(
        &h_hashes,
        N_HASH * n_data * sizeof(unsigned short),
        cudaHostAllocDefault);

    int block_size = 128;
    int num_blocks = (n_data + block_size - 1) / block_size;

    for (unsigned int i = 0; i < N_HASH; i++) {
        hashH3<<<num_blocks, block_size>>>(
            n_data, d_data, d_hashes + n_data * i, i);
    }

    cudaMemcpyAsync(
        h_hashes,
        d_hashes,
        N_HASH * n_data * sizeof(unsigned short),
        cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    // Find heaavy-hitters
    std::unordered_set<unsigned long> heavy_hitters;
    heavy_hitters.reserve(n_data / 10);

    for (unsigned int i = 0; i < n_data; i++) {
        unsigned int min_hits = std::numeric_limits<unsigned int>::max();

        for (unsigned int j = 0; j < N_HASH; j++) {
            if (sketch[h_hashes[i + n_data * j] + (j << M)] < min_hits) {
                min_hits = sketch[h_hashes[i + n_data * j] + (j << M)];
            }
        }

        for (unsigned int j = 0; j < N_HASH; j++) {
            if (sketch[h_hashes[i + n_data * j] + (j << M)] == min_hits) {
                sketch[h_hashes[i + n_data * j] + (j << M)]++;
            }
        }

        if (min_hits + 1 >= RHO) {
            heavy_hitters.insert(data_vector[i]);
        }
    }

    // End time measurement
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "Execution time: " << diff.count() << " s" << std::endl;
    std::cout << "Data vectors: " << n_data << std::endl;
    std::cout << "Heavy-hitters: " << heavy_hitters.size() << std::endl;

    // Write heavy-hitters to output file
    std::ofstream heavy_hitters_file("heavy-hitters_cu.txt");
    for (auto x : heavy_hitters) {
        heavy_hitters_file << sequenceToString(x, 16) << std::endl;
    }
    heavy_hitters_file.close();

    // Free shared memory
    cudaFree(d_data);
    cudaFree(d_hashes);
    cudaFreeHost(h_data);
    cudaFreeHost(h_hashes);
    // cudaFree(sketch);

    return 0;
}
