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
#include <unordered_map>
#include <cstring>

#include "fasta.hpp"
#include "MappedFile.hpp"
#include "Sketch.hpp"
#include "PackedArray.hpp"

typedef PackedArray<2, 32> Sequence;

// Number of hashes to use in the sketch
const unsigned int N_HASH = 4;

// Number of bits for the hashing seeds. Also determines the sketch size.
const unsigned int HASH_BITS = 14;

// Growth parameter for control step
const float GROWTH = 2;


struct SketchSettings {
    int min_length;
    int max_length;
    int n_length;

    std::vector<int> threshold;

    float growth;
};


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

    // Generate seeds
    size_t n_seeds = N_HASH * settings.max_length * 4;
    unsigned short* seeds = new unsigned short[n_seeds];
    for (unsigned int i = 0; i < n_seeds; i++) {
        seeds[i] = rand() & ~(~0UL << HASH_BITS);
    }

    // Load memory mapped files
    MappedFile test_file(argv[1]);
    MappedFile control_file(argv[2]);
    const char* test_data = test_file.data();
    const char* control_data = control_file.data();

    std::vector<std::unordered_set<Sequence>> heavy_hitters(settings.n_length);

    // Start time measurement
    auto start_time = std::chrono::steady_clock::now();

    int kmer_start = 0;

    // Sketch definition
    SketchSet<int, N_HASH, HASH_BITS> sketch(settings.n_length);

    while (kmer_start < test_file.size()) {
        bool sequence_end = false;

        int i;
        unsigned short hashes[N_HASH] = {0};
        Sequence encoded_kmer;

        for (i = 0; i < settings.max_length; i++) {
            int symbol;

            switch (test_data[kmer_start + i]) {
            case 'A':
                symbol = 0;
                break;
            case 'C':
                symbol = 1;
                break;
            case 'T':
                symbol = 2;
                break;
            case 'G':
                symbol = 3;
                break;
            default:
                sequence_end = true;
                break;
            }

            if (sequence_end)
                break;

            encoded_kmer.set(i, symbol);

            for (int j = 0; j < N_HASH; j++) {
                hashes[j] ^= seeds[4 * N_HASH * i + 4 * j + symbol];
            }

            if (i < settings.min_length - 1)
                continue;


            // Add to sketch
            unsigned int min_hits = std::numeric_limits<unsigned int>::max();

            for (int j = 0; j < N_HASH; j++) {
                int counter = sketch[i - settings.min_length + 1][j][hashes[j]];
                if (counter < min_hits) {
                    min_hits = counter;
                }
            }

            for (unsigned int j = 0; j < N_HASH; j++) {
                if (sketch[i - settings.min_length + 1][j][hashes[j]] == min_hits) {
                    sketch[i - settings.min_length + 1][j][hashes[j]]++;
                }
            }

            if (min_hits + 1 >= settings.threshold[i - settings.min_length + 1]) {
                heavy_hitters[i - settings.min_length + 1].insert(encoded_kmer);
            }
        }

        if (sequence_end && i < settings.min_length) {
            kmer_start += i + 1;
            continue;
        }

        kmer_start++;
    }

    auto test_end_time = std::chrono::steady_clock::now();

    // Get frequencies for heavy-hitters
    std::vector<std::unordered_map<Sequence, int>> frequencies(settings.n_length);
    for (int n = 0; n < settings.n_length; n++) {
        for (auto& encoded_kmer : heavy_hitters[n]) {
            frequencies[n][encoded_kmer] = std::numeric_limits<int>::max();

            for (int j = 0; j < N_HASH; j++) {
                unsigned short hash = 0;
                for (int i = 0; i < settings.min_length + n; i++)
                    hash ^= seeds[4 * N_HASH * i + 4 * j + encoded_kmer.get(i)];

                if (sketch[n][j][hash] < frequencies[n][encoded_kmer]) {
                    frequencies[n][encoded_kmer] = sketch[n][j][hash];
                }
            }

            frequencies[n][encoded_kmer] /= GROWTH;
        }
    }

    // Control step
    kmer_start = 0;
    while (kmer_start < control_file.size()) {
        bool sequence_end = false;

        int i;
        Sequence encoded_kmer;

        for (i = 0; i < settings.max_length; i++) {
            int symbol;

            switch (control_data[kmer_start + i]) {
            case 'A':
                symbol = 0;
                break;
            case 'C':
                symbol = 1;
                break;
            case 'T':
                symbol = 2;
                break;
            case 'G':
                symbol = 3;
                break;
            default:
                sequence_end = true;
                break;
            }

            if (sequence_end)
                break;

            encoded_kmer.set(i, symbol);

            if (i < settings.min_length - 1)
                continue;

            std::unordered_map<Sequence, int>::iterator counter;
            counter = frequencies[i - settings.min_length + 1].find(encoded_kmer);
            if (counter != frequencies[i - settings.min_length + 1].end()) {
                counter->second--;
            }
        }

        if (sequence_end && i < settings.min_length) {
            kmer_start += i + 1;
            continue;
        }

        kmer_start++;
    }

    // Select only the heavy-hitters not in the control set
    for (int n = 0; n < settings.n_length; n++) {
        for (auto i : frequencies[n]) {
            if (i.second <= 0) {
                heavy_hitters[n].erase(heavy_hitters[n].find(i.first));
            }
        }
    }

    auto end_time = std::chrono::steady_clock::now();

    // Report times
    std::clog
        << "Test time: "
        << std::chrono::duration<double>(test_end_time - start_time).count()
        << " s"
        << std::endl;
    std::clog
        << "Control time: "
        << std::chrono::duration<double>(end_time - test_end_time).count()
        << " s"
        << std::endl;
    std::clog
        << "Total time: "
        << std::chrono::duration<double>(end_time - start_time).count()
        << " s"
        << std::endl;

    // Print heavy-hitters
    int heavy_hitters_count = 0;

    for (int n = 0; n < settings.n_length; n++) {
        heavy_hitters_count += heavy_hitters[n].size();
        std::clog
            << "Heavy-hitters (length " << settings.min_length + n << "): "
            << heavy_hitters[n].size() << std::endl;

        for (auto x : heavy_hitters[n]) {
            std::cout
                << sequenceToString(x.data[0], settings.min_length + n) << std::endl;
        }
    }

    std::clog << "Heavy-hitters (total): " << heavy_hitters_count << std::endl;

    delete[] seeds;

    return 0;
}
