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

#include "fasta.hpp"
#include "MappedFile.hpp"


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


/**
 * @brief Compute H3 hash
 */
unsigned int hashH3(unsigned long key, unsigned short* seeds, int bits) {
    unsigned int result = 0;
    for (int i = 0; i < bits; i++) {
        if (key & 1)
            result ^= seeds[i];
        key >>= 1;
    }
    return result;
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

    // Generate seeds
    unsigned short* seeds[N_HASH];
    unsigned short* seeds_values;
    seeds_values = new unsigned short[N_HASH * settings.max_length * 2];
    for (unsigned int i = 0; i < N_HASH; i++) {
        seeds[i] = &seeds_values[settings.max_length * 2 * i];
        for (unsigned int j = 0; j < settings.max_length * 2; j++) {
            seeds[i][j] = rand() & ((1 << HASH_BITS) - 1);
        }
    }

    // Load memory mapped files
    MappedFile test_file = MappedFile::load(argv[1]);
    MappedFile control_file = MappedFile::load(argv[2]);

    // Start time measurement
    auto start_time = std::chrono::steady_clock::now();

    std::unordered_set<unsigned long>* heavy_hitters;
    heavy_hitters = new std::unordered_set<unsigned long>[settings.n_length];

    for (int n = 0; n < settings.n_length; n++) {
        int length = settings.min_length + n;

        // Create sketch
        unsigned int sketch[N_HASH][1 << HASH_BITS] = {0};

        // Parse data sets
        std::vector<unsigned long> data_vectors = parseFasta(
            test_file.data(), test_file.size(), length);
        std::vector<unsigned long> control_vectors = parseFasta(
            control_file.data(), control_file.size(), length);

        // Hash values
        for (unsigned int i = 0; i < data_vectors.size(); i++) {
            unsigned int min_hits = std::numeric_limits<unsigned int>::max();
            unsigned int hashes[N_HASH];

            for (unsigned int j = 0; j < N_HASH; j++) {
                hashes[j] = hashH3(data_vectors[i], seeds[j], length * 2);
                if (sketch[j][hashes[j]] < min_hits) {
                    min_hits = sketch[j][hashes[j]];
                }
            }

            for (unsigned int j = 0; j < N_HASH; j++) {
                if (sketch[j][hashes[j]] == min_hits) {
                    sketch[j][hashes[j]]++;
                }
            }

            if (min_hits + 1 == settings.threshold[n]) {
                heavy_hitters[n].insert(data_vectors[i]);
            }
        }

        // Get frequencies for heavy-hitters
        std::unordered_map<unsigned long, int> frequencies;

        for (auto i : heavy_hitters[n]) {
            frequencies[i] = std::numeric_limits<int>::max();

            for (unsigned int j = 0; j < N_HASH; j++) {
                unsigned int hash = hashH3(i, seeds[j], length * 2);
                if (sketch[j][hash] < frequencies[i]) {
                    frequencies[i] = sketch[j][hash];
                }
            }

            frequencies[i] /= GROWTH;
        }

        // Control step
        for (unsigned int i = 0; i < control_vectors.size(); i++) {
            std::unordered_map<unsigned long, int>::iterator counter;
            counter = frequencies.find(control_vectors[i]);
            if (counter != frequencies.end()) {
                counter->second--;
            }
        }

        // Select only the heavy-hitters not in the control set
        for (auto i : frequencies) {
            if (i.second <= 0) {
                heavy_hitters[n].erase(heavy_hitters[n].find(i.first));
            }
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> total_diff = end_time - start_time;

    std::clog << "Total time: " << total_diff.count() << " s" << std::endl;

    // Print heavy-hitters
    int heavy_hitters_count = 0;

    for (int n = 0; n < settings.n_length; n++) {
        heavy_hitters_count += heavy_hitters[n].size();
        std::clog
            << "Heavy-hitters (length " << settings.min_length + n << "): "
            << heavy_hitters[n].size() << std::endl;

        for (auto x : heavy_hitters[n]) {
            std::cout
                << sequenceToString(x, settings.min_length + n) << std::endl;
        }
    }

    std::clog << "Heavy-hitters (total): " << heavy_hitters_count << std::endl;

    return 0;
}
