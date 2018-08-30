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
        vec128 lo;
        vec128 hi;
    };
};


void hashWorker(
        const Seeds& seeds,
        Sketch* sketch,
        MappedFile* test_file,
        MappedFile* control_file,
        std::unordered_set<unsigned long>* heavy_hitters) {

    const char* data = test_file->data();

    vec128 sketch_offset = { .i = {
        0, 1 << HASH_BITS, 2 << HASH_BITS, 3 << HASH_BITS} };

    int n = 0;
    int end = test_file->size();

    int count = 0;

    while (n < end) {
        // Skip comment lines
        if (data[n] == '>') {
            while (n < end && data[n] != '\n') {
                n++;
            }
            n++;
            continue;
        }

        vec256 hash_vec;
        hash_vec.v = _mm256_setzero_si256();

        uint64_t sequence = 0;

        int m;
        for (m = 0; m < MAX_LENGTH + 3; m++) {
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
            if (symbol != 0) {  // Symbol 0 hashes with all zeros so we skip it
                vec256 seed_vec;
                seed_vec.v = _mm256_lddqu_si256(
                    (__m256i*)&seeds.values[symbol][m]);
                hash_vec.v = _mm256_xor_si256(hash_vec.v, seed_vec.v);
            }

            // Add to sketch
            if (m + 1 >= MIN_LENGTH) {
                vec128 hash[4];
                vec128 hits[4];
                int length[4];
                vec128 min_hits[4];

                count++;

                // Calculate sequence length of each of the parallel hashes
                uint_fast8_t write_flag[4];
                for (int i = 0; i < 4; i++) {
                    length[i] = m - 2 + i - MIN_LENGTH;
                    write_flag[i] = length[i] >= 0 && length[i] < N_LENGTH;
                }

                // Find the minimum counts
                for (int i = 0; i < 4; i++) {
                    if (write_flag[i]) {
                        hash[i].v = _mm_unpacklo_epi16(
                            hash_vec.lo.v, _mm_setzero_si128());
                        hash[i].v = _mm_or_si128(hash[i].v, sketch_offset.v);
                        hits[i].v = _mm_i32gather_epi32(
                            &(sketch[length[i]].count[0][0]), hash[i].v, 4);

                        // Compute the minimum counter value
                        vec128 min_tmp1, min_tmp2;
                        min_tmp1.v = _mm_shuffle_epi32(hits[i].v, 0b01001110);
                        min_tmp1.v = _mm_min_epi32(hits[i].v, min_tmp1.v);
                        min_tmp2.v = _mm_shuffle_epi32(min_tmp1.v, 0b10110001);
                        min_hits[i].v = _mm_min_epi32(min_tmp1.v, min_tmp2.v);
                    }

                    hash_vec.v = _mm256_permute4x64_epi64(hash_vec.v, 0b111001);
                }

                // Update the counts
                for (int i = 0; i < 4; i++) {
                    if (write_flag[i]) {
                        vec128 cmp;
                        cmp.v = _mm_cmpeq_epi32(hits[i].v, min_hits[i].v);

                        for (int j = 0; j < N_HASH; j++) {
                            if (cmp.i[j]) {
                                sketch[length[i]].count[0][hash[i].i[j]]++;
                            }
                        }
                    }
                }

                // Add sequences which go over the threshold to the results
                for (int i = 0; i < 4; i++) {
                    if (write_flag[i] &&
                            min_hits[i].i[0] + 1 == THRESHOLD[length[i]]) {
                        // Mask to extract the correct length sequence
                        uint64_t mask;
                        mask = ~(~0UL << ((length[i] + MIN_LENGTH) * 2));

                        heavy_hitters[length[i]].insert(sequence & mask);
                    }
                }
            }
        }

        // Check if no more sequences fit on the remainder of the line
        if (m < MIN_LENGTH) {
            n += m + 1;
        }
        else {
            n += 4;
        }
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
