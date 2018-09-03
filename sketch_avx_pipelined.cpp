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
// constexpr unsigned int THRESHOLD[] = {
//     365, 308, 257, 161, 150, 145, 145, 145, 145, 145, 145};

// Growth parameter for control step
const float GROWTH = 2;


struct SketchSettings {
    int min_length;
    int max_length;
    int n_length;

    std::vector<int> threshold;

    float growth;
};


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


void countminCu(
        const Seeds& seeds,
        Sketch* sketch,
        MappedFile* test_file,
        std::vector<int>& threshold,
        int data_start,
        int data_end,
        std::unordered_map<uint64_t, int>* heavy_hitters) {

    vec128 sketch_offset = { .i = {
        0, 1 << HASH_BITS, 2 << HASH_BITS, 3 << HASH_BITS} };

    // Adjust start and end positions to line endings
    const char* data = test_file->data();
    int n = data_start;
    while (n > 0 && data[n - 1] != '\n')
        n++;

    int end = data_end < test_file->size() ? data_end : test_file->size();
    while (n < test_file->size() && data[n] != '\n')
        n++;

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
                            sketch[length[i]].count[0], hash[i].v, 4);

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
                            min_hits[i].i[0] + 1 >= threshold[length[i]]) {
                        // Mask to extract the correct length sequence
                        uint64_t mask;
                        mask = ~(~0UL << ((length[i] + MIN_LENGTH) * 2));

                        heavy_hitters[length[i]][sequence & mask] = min_hits[i].i[0];
                    }
                }
            }
        }

        // Check if no more sequences fit on the remainder of the line
        if (m - 4 < MIN_LENGTH) {
            n += m + 1;
        }
        else {
            n += 4;
        }
    }
}


void control(
        MappedFile* control_file,
        int data_start,
        int data_end,
        std::unordered_map<uint64_t, int>* heavy_hitters) {

    // Adjust start and end positions to line endings
    const char* data = control_file->data();
    int n = data_start;
    while (n > 0 && data[n - 1] != '\n')
        n++;

    int end = data_end < control_file->size() ? data_end : control_file->size();
    while (n < control_file->size() && data[n] != '\n')
        n++;

    uint64_t sequence = 0;
    int length = 0;

    for (; n < end; n++) {
        // Skip comment lines
        if (data[n] == '>') {
            while (n < end && data[n] != '\n') {
                n++;
            }
            continue;
        }

        sequence <<= 2;
        length++;

        // Convert symbol to binary representation
        if (data[n] == 'A') {
            sequence |= 0;
        }
        else if (data[n] == 'C') {
            sequence |= 1;
        }
        else if (data[n] == 'T') {
            sequence |= 2;
        }
        else if (data[n] == 'G') {
            sequence |= 3;
        }
        else {
            length = 0;
            sequence = 0;
            continue;
        }

        if (length >= MIN_LENGTH) {
            int max_l = length - MIN_LENGTH + 1;
            max_l = max_l < N_LENGTH ? max_l : N_LENGTH;

            uint64_t mask = ~0UL << (MIN_LENGTH * 2);
            for (int i = 0; i < max_l; i++) {
                auto hh = heavy_hitters[i].find(sequence & ~mask);
                if (hh != heavy_hitters[i].end()) {
                    hh->second--;
                }

                mask <<= 2;
            }
        }
    }
}


int main(int argc, char* argv[]) {
    if (argc < 14) {
        std::cout
            << "Usage:" << std::endl
            << '\t' << argv[0] << " test_set control_set threshold_1 ..." << std::endl;
        return 1;
    }

    // Read thresholds from arguments
    std::vector<int> thresholds;
    for (int i = 3; i < argc; i++) {
        thresholds.push_back(atoi(argv[i]));
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
    auto start_time = std::chrono::steady_clock::now();

    std::unordered_map<uint64_t, int> heavy_hitters[N_LENGTH];

    countminCu(
        sym_seeds,
        &sketch[0],
        &test_file,
        thresholds,
        0,
        test_file.size(),
        &heavy_hitters[0]
    );

    // Scale frequencies by the growth factor
    for (int i = 0; i < N_LENGTH; i++) {
        for (auto& j : heavy_hitters[i]) {
            j.second /= GROWTH;
        }
    }

    auto test_time = std::chrono::steady_clock::now();

    // Execute control step
    int n_threads = std::thread::hardware_concurrency();
    std::thread* control_threads = new std::thread[n_threads];

    int data_stride = (control_file.size() + n_threads - 1) / n_threads;

    for (int i = 0; i < n_threads; i++) {
        control_threads[i] = std::thread(
            control,
            &control_file,
            i * data_stride,
            (i + 1) * data_stride,
            &heavy_hitters[0]);
    }

    for (int i = 0; i < n_threads; i++) {
        control_threads[i].join();
    }

    // Erase heavy-hitters in control set
    for (int i = 0; i < N_LENGTH; i++) {
        auto j = heavy_hitters[i].begin();
        while (j != heavy_hitters[i].end()) {
            if (j->second <= 0)
                j = heavy_hitters[i].erase(j);
            else
                j++;
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> test_diff = test_time - start_time;
    std::chrono::duration<double> control_diff = end_time - test_time;
    std::chrono::duration<double> total_diff = end_time - start_time;

    std::clog << "Test time: " << test_diff.count() << " s" << std::endl;
    std::clog << "Control time: " << control_diff.count() << " s" << std::endl;
    std::clog << "Total time: " << total_diff.count() << " s" << std::endl;

    // Print heavy-hitters
    int heavy_hitters_count = 0;

    for (int n = 0; n < N_LENGTH; n++) {
        heavy_hitters_count += heavy_hitters[n].size();
        std::clog
            << "Heavy-hitters (length " << MIN_LENGTH + n << "): "
            << heavy_hitters[n].size() << std::endl;

        for (auto x : heavy_hitters[n]) {
            std::cout
                << sequenceToString(x.first, MIN_LENGTH + n) << std::endl;
        }
    }

    std::clog << "Heavy-hitters (total): " << heavy_hitters_count << std::endl;

    return 0;
}
