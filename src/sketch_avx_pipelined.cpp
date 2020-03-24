/**
 * @brief Countmin-CU sketch
 *
 * Implementation using AVX2
 *
 * @file sketch.cpp
 * @author Hans Lehnert
 */

#include <iostream>
#include <fstream>
#include <cstring>
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
#include "util.hpp"


// Number of hashes to use in the sketch
const unsigned int N_HASH = 4;

// Amount of slots that will be used effectively for storing hashes, for
// alignment purpose
constexpr unsigned int N_HASH_SLOTS = ceilToPowerOf2(N_HASH);
static_assert(N_HASH_SLOTS == 4 || N_HASH_SLOTS == 8, "Unsupported value for N_HASH");

// Number of sequences processed in each vector register
constexpr unsigned int SEQ_PER_VEC = 16 / N_HASH_SLOTS;
// 16 is the number of 2 byte ints that fit in a 256-bit register

// Number of leading and trailing zeros in the seeds arrays
constexpr unsigned int SEED_PADDING = SEQ_PER_VEC - 1;

// Number of bits for the hashing seeds. Also determines the sketch size.
const unsigned int HASH_BITS = 14;

// Growth parameter for control step
const float GROWTH = 2;

// Mask used for unused values in min operation
constexpr vec256 MIN_MASK = generateMask(N_HASH);


struct SketchSettings {
    int min_length;
    int max_length;
    int n_length;

    std::vector<int> threshold;

    float growth;
};


struct Sketch {
    int32_t count[N_HASH_SLOTS][1 << HASH_BITS];
};


void countminCu(
        const SketchSettings settings,
        const uint16_t* seeds,
        Sketch* sketch,
        MappedFile* test_file,
        int data_start,
        int data_end,
        std::unordered_map<uint64_t, int>* heavy_hitters) {

    vec256 sketch_offset = { .i = {
        0,
        1 << HASH_BITS,
        2 << HASH_BITS,
        3 << HASH_BITS,
        4 << HASH_BITS,
        5 << HASH_BITS,
        6 << HASH_BITS,
        7 << HASH_BITS
    }};

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
        for (m = 0; m < settings.max_length + 3; m++) {
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
                (__m256i*)&seeds[symbol * N_HASH_SLOTS * (settings.max_length + 2 * SEED_PADDING) + m * N_HASH_SLOTS]);
            hash_vec.v = _mm256_xor_si256(hash_vec.v, seed_vec.v);

            // Add to sketch
            if (m + 1 < settings.min_length)
                continue;

            vec256 hash[SEQ_PER_VEC];
            vec256 hits[SEQ_PER_VEC];
            int length[SEQ_PER_VEC];
            vec256 min_hits[SEQ_PER_VEC];

            // Calculate sequence length of each of the parallel hashes
            uint_fast8_t write_flag[SEQ_PER_VEC];
            for (int i = 0; i < SEQ_PER_VEC; i++) {
                length[i] = m - 2 + i - settings.min_length;
                write_flag[i] = length[i] >= 0 && length[i] < settings.n_length;
            }

            // Find the minimum counts
            for (int i = 0; i < SEQ_PER_VEC; i++) {
                if (write_flag[i]) {
                    if constexpr(N_HASH_SLOTS == 4) {
                        hash[i].lo.v = _mm_unpacklo_epi16(
                            hash_vec.lo.v, _mm_setzero_si128());
                        hash[i].lo.v = _mm_or_si128(hash[i].lo.v, sketch_offset.lo.v);
                        hits[i].lo.v = _mm_i32gather_epi32(
                            sketch[length[i]].count[0], hash[i].lo.v, 4);

                        if constexpr (N_HASH != N_HASH_SLOTS) {
                            hits[i].lo.v = _mm_or_si128(hits[i].lo.v, MIN_MASK.lo.v);
                        }

                        // Compute the minimum counter value
                        vec128 min_tmp1, min_tmp2;
                        min_tmp1.v = _mm_shuffle_epi32(hits[i].lo.v, 0b01001110);
                        min_tmp1.v = _mm_min_epu32(hits[i].lo.v, min_tmp1.v);
                        min_tmp2.v = _mm_shuffle_epi32(min_tmp1.v, 0b10110001);
                        min_hits[i].lo.v = _mm_min_epu32(min_tmp1.v, min_tmp2.v);
                    }
                    /* else if constexpr(N_HASH_SLOTS == 8) {
                        hash[i].v = _mm_unpacklo_epi16(
                            hash_vec.lo.v, _mm_setzero_si128());
                        hash[i].lo.v = _mm_or_si128(hash[i].lo.v, sketch_offset.v);
                        hits[i].lo.v = _mm_i32gather_epi32(
                            sketch[length[i]].count[0], hash[i].lo.v, 4);

                        if constexpr (N_HASH != N_HASH_SLOTS) {
                            hits[i].lo.v = _mm_or_si128(hits[i].lo.v, MIN_MASK.lo.v);
                        }

                        // Compute the minimum counter value
                        vec128 min_tmp1, min_tmp2;
                        min_tmp1.v = _mm_shuffle_epi32(hits[i].lo.v, 0b01001110);
                        min_tmp1.v = _mm_min_epu32(hits[i].lo.v, min_tmp1.v);
                        min_tmp2.v = _mm_shuffle_epi32(min_tmp1.v, 0b10110001);
                        min_hits[i].lo.v = _mm_min_epu32(min_tmp1.v, min_tmp2.v);
                    }*/
                }

                hash_vec.v = _mm256_permute4x64_epi64(hash_vec.v, 0b111001);
            }

            // Update the counts
            for (int i = 0; i < SEQ_PER_VEC; i++) {
                if (write_flag[i]) {
                    vec128 cmp;
                    cmp.v = _mm_cmpeq_epi32(hits[i].lo.v, min_hits[i].lo.v);

                    for (int j = 0; j < N_HASH; j++) {
                        if (cmp.i[j]) {
                            sketch[length[i]].count[0][hash[i].i[j]]++;
                        }
                    }
                }
            }

            // Add sequences which go over the threshold to the results
            for (int i = 0; i < SEQ_PER_VEC; i++) {
                if (write_flag[i] &&
                        min_hits[i].i[0] + 1 >= settings.threshold[length[i]]) {
                    // Mask to extract the correct length sequence
                    uint64_t mask;
                    mask = ~(~0UL << ((length[i] + settings.min_length) * 2));

                    heavy_hitters[length[i]][sequence & mask] = min_hits[i].i[0] + 1;
                }
            }
        }

        // Check if no more sequences fit on the remainder of the line
        if (m < settings.max_length + SEQ_PER_VEC - 1
            && m - SEQ_PER_VEC < settings.min_length)
        {
            n += m + 1;
        }
        else {
            n += SEQ_PER_VEC;
        }
    }
}


void control(
        const SketchSettings& settings,
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

        if (length >= settings.min_length) {
            int max_l = length - settings.min_length + 1;
            max_l = max_l < settings.n_length ? max_l : settings.n_length;

            uint64_t mask = ~0UL << (settings.min_length * 2);
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
    if (argc < 5) {
        std::cerr
            << "Usage:" << std::endl
            << '\t' << argv[0]
            << " test_set control_set min_length max_length threshold_1 ..."
            << std::endl;
        return 1;
    }

    // Configure sketch settings
    SketchSettings settings;
    settings.min_length = atoi(argv[3]);
    settings.max_length = atoi(argv[4]);
    settings.n_length = settings.max_length - settings.min_length + 1;
    settings.growth = 2.0;

    if (argc - 5 < settings.n_length) {
        std::cerr
            << "Missing threshold values. Got "
            << argc - 5
            << ", expected "
            << settings.n_length
            << std::endl;
        return 1;
    }

    for (int i = 5; i < argc; i++) {
        settings.threshold.push_back(atoi(argv[i]));
    }

    // Generate random seeds

    // Number of leading and trailing zeros in seeds arrays
    uint16_t* seeds = new uint16_t[4 * (settings.max_length + 2 * SEED_PADDING) * N_HASH_SLOTS];
    memset(seeds, 0, 4 * (settings.max_length + 2 * SEED_PADDING) * N_HASH_SLOTS * sizeof(uint16_t));
    for (int i = 0; i < settings.max_length; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < N_HASH; k++) {
                seeds[(settings.max_length + 2 * SEED_PADDING) * N_HASH_SLOTS * j + N_HASH_SLOTS * (i + SEED_PADDING) + k] =
                    rand() & ~(~0U << HASH_BITS);
            }
        }
    }

    // Create sketches
    std::vector<Sketch> sketch;
    sketch.resize(settings.n_length);
    for (int i = 0; i < sketch.size(); i++)
        memset(&sketch[i], 0, sizeof(Sketch));

    // Load memory mapped files
    MappedFile test_file(argv[1]);
    MappedFile control_file(argv[2]);

    // Start time measurement
    auto start_time = std::chrono::steady_clock::now();

    std::vector<std::unordered_map<uint64_t, int>> heavy_hitters;
    heavy_hitters.resize(settings.n_length);

    countminCu(
        settings,
        seeds,
        &sketch[0],
        &test_file,
        0,
        test_file.size(),
        &heavy_hitters[0]
    );

    auto test_time = std::chrono::steady_clock::now();

    // Copy frequencies to later use for real frequencies
    std::vector<std::unordered_map<uint64_t, int>> frequencies = heavy_hitters;

    // Scale frequencies by the growth factor
    for (int i = 0; i < settings.n_length; i++) {
        for (auto& j : heavy_hitters[i]) {
            j.second /= GROWTH;
        }
    }

    // Execute control step
    int n_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> control_threads;
    control_threads.reserve(n_threads);

    int data_stride = (control_file.size() + n_threads - 1) / n_threads;

    for (int i = 0; i < n_threads; i++) {
        control_threads.emplace_back(
            control,
            settings,
            &control_file,
            i * data_stride,
            (i + 1) * data_stride,
            &heavy_hitters[0]);
    }

    for (int i = 0; i < n_threads; i++) {
        control_threads[i].join();
    }

    // Erase heavy-hitters in control set
    for (int i = 0; i < settings.n_length; i++) {
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

    for (int n = 0; n < settings.n_length; n++) {
        heavy_hitters_count += heavy_hitters[n].size();
        std::clog
            << "Heavy-hitters (length " << settings.min_length + n << "): "
            << heavy_hitters[n].size() << std::endl;

        for (auto& x : heavy_hitters[n]) {
            std::cout
                << sequenceToString(x.first, settings.min_length + n) << " "
                << frequencies[n][x.first] << std::endl;
        }
    }

    std::clog << "Heavy-hitters (total): " << heavy_hitters_count << std::endl;

    // Free allocated memory
    delete[] seeds;

    return 0;
}
