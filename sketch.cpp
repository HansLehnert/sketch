/**
 * @brief Countmin-CU sketch implementation
 *
 * @file sketch.cpp
 * @author Hans Lehnert
 * @date 2018-07-18
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
unsigned int hashH3(unsigned int key, unsigned short* seeds) {
    unsigned int result = 0;
    for (int i = 0; i < 32; i++) {
        if (key & 1)
            result ^= seeds[i];
        key >>= 1;
    }
    return result;
}


int main(int argc, char* argv[]) {
    // Generate hash vectors
    unsigned short seeds[N_HASH][32];
    for (unsigned int i = 0; i < N_HASH; i++) {
        for (unsigned int j = 0; j < 32; j++) {
            seeds[i][j] = rand() & ((1 << M) - 1);
        }
    }

    // Create sketch
    unsigned int sketch[N_HASH][1 << M] = {0};

    // Start time measurement
    auto start = std::chrono::steady_clock::now();

    // Parse data set
    std::ifstream dataset_file("data/test.fasta");
    std::vector<unsigned int> data_vectors = parseFasta(dataset_file, 16);
    dataset_file.close();

    std::unordered_set<unsigned int> heavy_hitters;
    heavy_hitters.reserve(data_vectors.size() / 10);

    // Hash values
    for (unsigned int i = 0; i < data_vectors.size(); i++) {
        unsigned int min_hits = std::numeric_limits<unsigned int>::max();
        unsigned int hashes[N_HASH];

        for (unsigned int j = 0; j < N_HASH; j++) {
            hashes[j] = hashH3(data_vectors[i], seeds[j]);
            if (sketch[j][hashes[j]] < min_hits) {
                min_hits = sketch[j][hashes[j]];
            }
        }

        for (unsigned int j = 0; j < N_HASH; j++) {
            if (sketch[j][hashes[j]] == min_hits) {
                sketch[j][hashes[j]]++;
            }
        }

        if (min_hits + 1 >= RHO) {
            heavy_hitters.insert(data_vectors[i]);
        }
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "Data vectors: " << data_vectors.size() << std::endl;
    std::cout << "Execution time: " << diff.count() << " s" << std::endl;

    // Write heavy-hitters to output file
    std::ofstream heavy_hitters_file("heavy-hitters.txt");
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

    return 0;
}
