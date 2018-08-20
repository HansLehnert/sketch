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


const unsigned int MIN_LENGTH = 10;
const unsigned int MAX_LENGTH = 20;
const unsigned int N_LENGTH = MAX_LENGTH - MIN_LENGTH + 1;

const unsigned int N_HASH = 4;
const unsigned int HASH_BITS = 14;

const unsigned int THRESHOLD[] = {
    365, 308, 257, 161, 150, 145, 145, 145, 145, 145, 145};
const float GROWTH = 2;


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
    if (argc < 3) {
        std::cout
            << "Usage:" << std::endl
            << '\t' << argv[0] << " test_set control_set" << std::endl;
        return 1;
    }

    // Generate seeds
    unsigned short seeds[N_HASH][MAX_LENGTH * 2];
    for (unsigned int i = 0; i < N_HASH; i++) {
        for (unsigned int j = 0; j < MAX_LENGTH * 2; j++) {
            seeds[i][j] = rand() & ((1 << HASH_BITS) - 1);
        }
    }

    // Load memory mapped files
    MappedFile test_file = MappedFile::load(argv[1]);
    MappedFile control_file = MappedFile::load(argv[2]);

    // Start time measurement
    auto start = std::chrono::steady_clock::now();

    std::unordered_set<unsigned long> heavy_hitters[N_LENGTH];

    for (int n = 0; n < N_LENGTH; n++) {
        int length = MIN_LENGTH + n;

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

            if (min_hits + 1 == THRESHOLD[n]) {
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

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "Execution time: " << diff.count() << " s" << std::endl;

    // Write heavy-hitters to output file
    std::ofstream heavy_hitters_file("heavy-hitters.txt");

    int heavy_hitters_count = 0;

    for (int n = 0; n < N_LENGTH; n++) {
        heavy_hitters_count += heavy_hitters[n].size();

        for (auto x : heavy_hitters[n]) {
            heavy_hitters_file
                << sequenceToString(x, MIN_LENGTH + n) << std::endl;
        }
    }

    heavy_hitters_file.close();

    std::cout << "Heavy-hitters: " << heavy_hitters_count << std::endl;

    return 0;
}
