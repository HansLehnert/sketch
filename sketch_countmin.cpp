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
const unsigned int N_HASH = 7;

// Number of bits for the hashing seeds. Also determines the sketch size.
const unsigned int HASH_BITS = 15;

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
    for (int i = 0; i < N_HASH; i++) {
        seeds[i] = &seeds_values[settings.max_length * 2 * i];
        for (int j = 0; j < settings.max_length * 2; j++) {
            seeds[i][j] = rand() & ((1 << HASH_BITS) - 1);
        }
    }

    // Load memory mapped files
    MappedFile test_file = MappedFile::load(argv[1]);
    MappedFile control_file = MappedFile::load(argv[2]);

    // Start time measurement
    std::chrono::duration<double> preprocessing_total;
    std::chrono::duration<double> test_total;
    std::chrono::duration<double> control_total;
    std::chrono::time_point<std::chrono::steady_clock> mid_time;

    auto start_time = std::chrono::steady_clock::now();

    std::vector<std::unordered_map<unsigned long, int>> heavy_hitters;
    heavy_hitters.resize(settings.n_length);

    for (int n = 0; n < settings.n_length; n++) {
        int length = settings.min_length + n;

        // Create sketch
        unsigned int sketch[N_HASH][1 << HASH_BITS] = {0};

        mid_time = std::chrono::steady_clock::now();

        // Parse data sets
        std::vector<unsigned long> data_vectors = parseFasta(
            test_file.data(), test_file.size(), length);
        std::vector<unsigned long> control_vectors = parseFasta(
            control_file.data(), control_file.size(), length);

        preprocessing_total += std::chrono::steady_clock::now() - mid_time;
        mid_time = std::chrono::steady_clock::now();

        // Hash values
        for (int i = 0; i < data_vectors.size(); i++) {
            unsigned int min_hits = std::numeric_limits<unsigned int>::max();

            for (int j = 0; j < N_HASH; j++) {
                unsigned short hash = hashH3(
                    data_vectors[i], seeds[j], length * 2);

                if (++sketch[j][hash] < min_hits) {
                    min_hits = sketch[j][hash];
                }
            }

            if (min_hits + 1 >= settings.threshold[n]) {
                heavy_hitters[n][data_vectors[i]] = min_hits;
            }
        }

        test_total += std::chrono::steady_clock::now() - mid_time;
        mid_time = std::chrono::steady_clock::now();

        for (auto& i : heavy_hitters[n]) {
            i.second /= GROWTH;
        }

        // Control step
        for (unsigned int i = 0; i < control_vectors.size(); i++) {
            std::unordered_map<unsigned long, int>::iterator counter;
            counter = heavy_hitters[n].find(control_vectors[i]);
            if (counter != heavy_hitters[n].end()) {
                counter->second--;
            }
        }

        // Select only the heavy-hitters not in the control set
        auto j = heavy_hitters[n].begin();
        while (j != heavy_hitters[n].end()) {
            if (j->second <= 0)
                j = heavy_hitters[n].erase(j);
            else
                j++;
        }

        control_total += std::chrono::steady_clock::now() - mid_time;
        mid_time = std::chrono::steady_clock::now();
    }

    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> total_diff = end_time - start_time;

    // Report times
    std::clog << "Preprocessing time: " << preprocessing_total.count()
        << " s" << std::endl;
    std::clog << "Test time: " << test_total.count() << " s" << std::endl;
    std::clog << "Control time: " << control_total.count() << " s" << std::endl;
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
                << sequenceToString(x.second, settings.min_length + n) << std::endl;
        }
    }

    std::clog << "Heavy-hitters (total): " << heavy_hitters_count << std::endl;

    delete[] seeds_values;

    return 0;
}
