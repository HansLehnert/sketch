/**
 * @brief Countmin-CU sketch
 *
 * Implementation using vector (AVX) instructions
 *
 * @file sketch_avx.cpp
 * @author Hans Lehnert
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <limits>
#include <unordered_set>
#include <immintrin.h>

#include "fasta.hpp"


const unsigned int N_HASH = 4;
const unsigned int M = 14;
const unsigned int RHO = 145;


/**
 * @brief Compute H3 hash
 *
 * Uses AVX2
 */
__m256i hashH3_vec(__m256i keys, unsigned short* seeds) {
    __m256i result = _mm256_setzero_si256();
    __m256i zero = _mm256_setzero_si256();

    for (int i = 0; i < 32; i++) {
        // Load a seed into a vector
        __m256i seed = _mm256_set1_epi32(seeds[i]);

        // Mask the seed by the vector keys bits
        seed = _mm256_castps_si256(_mm256_blendv_ps(
            _mm256_castsi256_ps(seed),
            _mm256_castsi256_ps(zero),
            _mm256_castsi256_ps(keys)));

        // Update the results
        result = _mm256_xor_si256(result, seed);

        // Shift the keys to change the masking values
        keys = _mm256_slli_epi32(keys, 1);
    }

    return result;
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Missing dataset file." << std::endl;
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
    std::vector<unsigned int> data_vectors = parseFasta(dataset_file, 16);
    dataset_file.close();

    std::unordered_set<unsigned int> heavy_hitters;
    heavy_hitters.reserve(data_vectors.size() / 10);

    // Hash values
    for (unsigned int i = 0; i < data_vectors.size() - 8; i += 8) {
        __m256i keys = _mm256_loadu_si256((__m256i*)(data_vectors.data() + i));

        __m256i min_hits = _mm256_set1_epi32(
            std::numeric_limits<unsigned int>::max());
        __m256i hashes[N_HASH];

        // Hash and find min counters
        for (unsigned int j = 0; j < N_HASH; j++) {
            hashes[j] = hashH3_vec(keys, seeds[j]);

            __m256i hits = _mm256_i32gather_epi32(
                (int*)sketch[j], hashes[j], sizeof(unsigned int));
            min_hits = _mm256_min_epu32(min_hits, hits);
        }

        // Update counters
        for (unsigned int j = 0; j < N_HASH; j++) {
            __m256i counter = _mm256_i32gather_epi32(
                (int*)sketch[j], hashes[j], sizeof(unsigned int));
            __m256i cmp_mask = _mm256_cmpeq_epi32(counter, min_hits);

            for (int k = 0; k < 8; k++) {
                if (((unsigned int*)&cmp_mask)[k]) {
                    sketch[j][((unsigned int*)&hashes[j])[k]]++;
                }
            }
        }

        // Update heavy-hitters
        for (int k = 0; k < 8; k++) {
            if (((unsigned int*)&min_hits)[k] + 1 >= RHO) {
                heavy_hitters.insert(((unsigned int*)&keys)[k]);
            }
        }
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "Execution time: " << diff.count() << " s" << std::endl;
    std::cout << "Data vectors: " << data_vectors.size() << std::endl;
    std::cout << "Heavy-hitters: " << heavy_hitters.size() << std::endl;

    // Write heavy-hitters to output file
    std::ofstream heavy_hitters_file("heavy-hitters_avx.txt");
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
