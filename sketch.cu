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


/**
 * @brief Compute H3 hash
 */
__global__ void hashH3(
        unsigned int n,
        unsigned int* keys,
        unsigned short* seeds,
        unsigned short* hashes) {
    unsigned int start_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = start_index; i < n; i += stride) {
        unsigned int mask = 1;
        hashes[i] = 0;

        for (int j = 0; j < 32; j++) {
            if (keys[i] & mask)
                hashes[i] ^= seeds[j];
            mask <<= 1;
        }
    }
}


int main(int argc, char* argv[]) {
    // Generate hash vectors
    unsigned short* seeds;
    cudaMallocManaged(&seeds, N_HASH * 32 * sizeof(unsigned short));

    for (unsigned int i = 0; i < N_HASH * 32; i++) {
        seeds[i] = rand() & ((1 << M) - 1);
    }

    // Create sketch
    unsigned int* sketch;
    cudaMallocManaged(&sketch, N_HASH * (1 << M) * sizeof(unsigned int));

    // Start time measurement
    auto start = std::chrono::steady_clock::now();

    // Parse data set
    std::ifstream dataset_file("data/test.fasta");
    std::vector<unsigned int> data_vector = parseFasta(dataset_file, 16);
    dataset_file.close();

    unsigned int n_data = data_vector.size();
    unsigned int* data;
    cudaMallocManaged(&data, n_data * sizeof(unsigned int));
    for (int i = 0; i < n_data; i++) {
        data[i] = data_vector[i];
    }

    std::unordered_set<unsigned int> heavy_hitters;
    heavy_hitters.reserve(n_data / 10);

    // Hash values
    unsigned short* hashes;
    cudaMallocManaged(&hashes, N_HASH * n_data * sizeof(unsigned short));

    int block_size = 128;
    int num_blocks = (n_data + block_size - 1) / block_size;

    for (unsigned int i = 0; i < N_HASH; i++) {
        hashH3<<<num_blocks, block_size>>>(
            n_data, data, seeds + 32 * i, hashes + n_data * i);
    }
    cudaDeviceSynchronize();

    for (unsigned int i = 0; i < n_data; i++) {
        unsigned int min_hits = std::numeric_limits<unsigned int>::max();

        for (unsigned int j = 0; j < N_HASH; j++) {
            if (sketch[hashes[i + n_data * j] + (j << M)] < min_hits) {
                min_hits = sketch[hashes[i + n_data * j] + (j << M)];
            }
        }

        for (unsigned int j = 0; j < N_HASH; j++) {
            if (sketch[hashes[i + n_data * j] + (j << M)] == min_hits) {
                sketch[hashes[i + n_data * j] + (j << M)]++;
            }
        }

        if (min_hits + 1 >= RHO) {
            heavy_hitters.insert(data[i]);
        }
    }

    // End time measurement
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "Data vectors: " << n_data << std::endl;
    std::cout << "Execution time: " << diff.count() << " s" << std::endl;

    // Write heavy-hitters to output file
    std::ofstream heavy_hitters_file("heavy-hitters_cu.txt");
    for (auto x : heavy_hitters) {
        std::string sequence;

        for (int i = 0; i < 16; i++) {
            switch (x << (i * 2) >> 30) {
            case 0:
                sequence += 'A';
                break;
            case 1:
                sequence += 'C';
                break;
            case 2:
                sequence += 'T';
                break;
            case 3:
                sequence += 'G';
                break;
            }
        }
        heavy_hitters_file << sequence << std::endl;
    }
    heavy_hitters_file.close();

    // Free shared memory
    cudaFree(seeds);
    cudaFree(sketch);
    cudaFree(hashes);

    return 0;
}
