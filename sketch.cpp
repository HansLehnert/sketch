/**
 * @brief Countmin-CU sketch
 *
 * Base implementation
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
template <int bits>
unsigned int hashH3(unsigned long key, unsigned short* seeds) {
    unsigned int result = 0;
    for (int i = 0; i < bits; i++) {
        if (key & 1)
            result ^= seeds[i];
        key >>= 1;
    }
    return result;
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Missing dataset file." << std::endl;
        return 1;
    }

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
    std::ifstream dataset_file(argv[1]);
    std::vector<unsigned long> data_vectors = parseFasta(dataset_file, 16);
    dataset_file.close();

    std::unordered_set<unsigned long> heavy_hitters;
    // heavy_hitters.reserve(data_vectors.size() / 10);

    // Hash values
    for (unsigned int i = 0; i < data_vectors.size(); i++) {
        unsigned int min_hits = std::numeric_limits<unsigned int>::max();
        unsigned int hashes[N_HASH];

        for (unsigned int j = 0; j < N_HASH; j++) {
            hashes[j] = hashH3<32>(data_vectors[i], seeds[j]);
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

    std::clog << "Execution time: " << diff.count() << " s" << std::endl;
    std::clog << "Data vectors: " << data_vectors.size() << std::endl;
    std::clog << "Heavy-hitters: " << heavy_hitters.size() << std::endl;

    // Print heavy-hitters
    for (auto x : heavy_hitters) {
        std::cout << sequenceToString(x, 16) << std::endl;
    }

    return 0;
}
