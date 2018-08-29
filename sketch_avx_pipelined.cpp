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
#include <thread>
#include <unordered_set>
#include <unordered_map>
#include <immintrin.h>

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


struct Seeds {
    uint16_t values[4][MAX_LENGTH + 6][N_HASH];
    // 4 symbols (ACTG) and 3 pre and post padding seeds;
};


struct Sketch {
    int32_t count[N_HASH][1 << HASH_BITS];
};


union vec128 {
    __m128i v;
    uint16_t s[8];
    uint32_t i[4];
    uint64_t l[2];
};

union vec256 {
    __m256i v;
    uint16_t s[16];
    uint32_t i[8];
    uint64_t l[4];
    struct {
        vec128 low;
        vec128 high;
    };
};


void hashWorker(
        const Seeds& seeds,
        Sketch* sketch,
        MappedFile* test_file,
        MappedFile* control_file,
        std::unordered_set<unsigned long>* heavy_hitters) {

    const char* data = test_file->data();

    vec256 hash_mask;
    hash_mask.v = _mm256_set1_epi64x(
        std::numeric_limits<unsigned short>::max());

    int n = 0;
    int end = test_file->size();

    int count = 0;

    while (n < end) {
        // Skip comment lines
        if (data[n] == '>') {
            while (data[n] != '\n' && n < end) {
                n++;
            }
            n++;
            continue;
        }

        vec256 hash_vec;
        hash_vec.v = _mm256_setzero_si256();

        uint64_t sequence = 0;

        for (int m = 0; m < MAX_LENGTH + 3; m++) {
            uint_fast8_t symbol;

            // Convert symbol to binary representation
            if (data[n + m] == 'A') {
                symbol = 0;
            }
            else if (data[n + m] == 'C') {
                symbol = 1;
            }
            else if (data[n + m] == 'T') {
                symbol = 2;
            }
            else if (data[n + m] == 'G') {
                symbol = 3;
            }
            else {
                break;
            }

            sequence = sequence << 2 | symbol;

            // Update hashes
            vec256 seed_vec;
            seed_vec.v = _mm256_lddqu_si256(
                (__m256i*)&seeds.values[symbol][m][0]);
            hash_vec.v = _mm256_xor_si256(hash_vec.v, seed_vec.v);

            // Add to sketch
            if (m + 1 >= MIN_LENGTH) {
                vec256 hash[N_HASH];
                vec128 hits[N_HASH];
                vec128 min_hits;
                int length[4];
                min_hits.v = _mm_set1_epi32(std::numeric_limits<int>::max());

                vec256 hash_tmp = hash_vec;

                count++;

                // Calculate sequence length of each of the parallel hashes
                uint_fast8_t write_flag[4];
                for (int i = 0; i < 4; i++) {
                    length[i] = m - 2 + i - MIN_LENGTH;
                    write_flag[i] = length[i] >= 0 && length[i] < N_LENGTH;
                }

                // Find the minimum counts
                for (int i = 0; i < N_HASH; i++) {
                    hash[i].v = _mm256_and_si256(hash_tmp.v, hash_mask.v);
                    hits[i].v = _mm256_i64gather_epi32(
                        &(sketch->count[i][0]), hash[i].v, 4);
                    hash_tmp.v = _mm256_srli_epi64(hash_tmp.v, 16);
                    min_hits.v = _mm_min_epi32(min_hits.v, hits[i].v);
                }

                // Update the counts
                for (int i = 0; i < N_HASH; i++) {
                    vec128 cmp;
                    cmp.v = _mm_cmpeq_epi32(hits[i].v, min_hits.v);

                    for (int j = 0; j < 4; j++) {
                        if (write_flag[j] && cmp.i[j]) {
                            sketch->count[i][hash[i].l[j]]++;
                        }
                    }
                }

                // Add sequences which go over the threshold to the results
                for (int i = 0; i < 4; i++) {
                    if (write_flag[i] &&
                            min_hits.i[i] == THRESHOLD[length[i]]) {
                        // Mask to extract the correct length sequence
                        uint64_t mask;
                        mask = ~(~0UL << ((length[i] + MIN_LENGTH) * 2));

                        heavy_hitters[length[i]].insert(sequence & mask);
                    }
                }
            }
        }

        n += 4;
    }

    std::clog << "Sequences checked:" << count << std::endl;
}


int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout
            << "Usage:" << std::endl
            << '\t' << argv[0] << " test_set control_set" << std::endl;
        return 1;
    }

    // Generate random seeds
    unsigned short base_seeds[N_HASH][MAX_LENGTH * 2];
    for (int i = 0; i < N_HASH; i++) {
        for (int j = 0; j < MAX_LENGTH * 2; j++) {
            base_seeds[i][j] = rand() & ~(~0U << HASH_BITS);
        }
    }

    Seeds sym_seeds = {0};
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < MAX_LENGTH; j++) {
            for (int k = 0; k < N_HASH; k++) {
                sym_seeds.values[i][j + 3][k] =
                    (i & 1 ? base_seeds[k][j * 2] : 0) ^
                    (i & 2 ? base_seeds[k][j * 2 + 1] : 0);
            }
        }
    }

    // Create sketches
    Sketch sketch[N_LENGTH] = {0};

    // Load memory mapped files
    MappedFile test_file = MappedFile::load(argv[1]);
    MappedFile control_file = MappedFile::load(argv[2]);

    // Start time measurement
    auto start = std::chrono::steady_clock::now();

    std::unordered_set<unsigned long> heavy_hitters[N_LENGTH];
    std::thread worker_threads[N_LENGTH];

    // for (int n = 0; n < 1; n++) {
    //     worker_threads[n] = std::thread(
    //         hashWorker,
    //         &test_file,
    //         &control_file,
    //         sym_seeds,
    //         &heavy_hitters[0]
    //     );
    // }

    // for (int n = 0; n < 1; n++) {
    //     worker_threads[n].join();
    // }
    hashWorker(
        sym_seeds,
        &sketch[0],
        &test_file,
        &control_file,
        &heavy_hitters[0]
    );

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::clog << "Execution time: " << diff.count() << " s" << std::endl;

    // Print heavy-hitters
    int heavy_hitters_count = 0;

    for (int n = 0; n < N_LENGTH; n++) {
        heavy_hitters_count += heavy_hitters[n].size();
        std::clog
            << "Heavy-hitters (length " << MIN_LENGTH + n << "): "
            << heavy_hitters[n].size() << std::endl;

        for (auto x : heavy_hitters[n]) {
            std::cout
                << sequenceToString(x, MIN_LENGTH + n) << std::endl;
        }
    }

    std::clog << "Heavy-hitters (total): " << heavy_hitters_count << std::endl;

    return 0;
}
