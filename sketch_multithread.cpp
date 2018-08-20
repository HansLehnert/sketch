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

#include "fasta.hpp"
#include "MappedFile.hpp"


const unsigned int MIN_LENGTH = 10;
const unsigned int MAX_LENGTH = 20;
const unsigned int N_LENGTH = MAX_LENGTH - MIN_LENGTH + 1;

const unsigned int N_HASH = 4;
const unsigned int HASH_BITS = 14;

constexpr unsigned int THRESHOLD[N_LENGTH] = {
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


template <int length, int threshold>
void hashWorker(
        MappedFile* test_file,
        MappedFile* control_file,
        std::unordered_set<unsigned long>* heavy_hitters) {

    // Generate seeds
    unsigned short seeds[N_HASH][length * 2];
    for (unsigned int i = 0; i < N_HASH; i++) {
        for (unsigned int j = 0; j < length * 2; j++) {
            seeds[i][j] = rand() & ((1 << HASH_BITS) - 1);
        }
    }

    // Create sketch
    unsigned int sketch[N_HASH][1 << HASH_BITS] = {0};

    // Parse data sets
    std::vector<unsigned long> data_vectors = parseFasta(
        test_file->data(), test_file->size(), length);
    std::vector<unsigned long> control_vectors = parseFasta(
        control_file->data(), control_file->size(), length);

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

        if (min_hits + 1 == threshold) {
            heavy_hitters->insert(data_vectors[i]);
        }
    }

    // Get frequencies for heavy-hitters
    std::unordered_map<unsigned long, int> frequencies;

    for (auto i : *heavy_hitters) {
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
            heavy_hitters->erase(heavy_hitters->find(i.first));
        }
    }
}


int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout
            << "Usage:" << std::endl
            << '\t' << argv[0] << " test_set control_set" << std::endl;
        return 1;
    }

    // Load memory mapped files
    MappedFile test_file = MappedFile::load(argv[1]);
    MappedFile control_file = MappedFile::load(argv[2]);

    // Start time measurement
    auto start = std::chrono::steady_clock::now();

    std::unordered_set<unsigned long> heavy_hitters[N_LENGTH];
    std::thread worker_threads[N_LENGTH];

    worker_threads[0] = std::thread(hashWorker<10, THRESHOLD[0]>, &test_file, &control_file, &heavy_hitters[0]);
    worker_threads[1] = std::thread(hashWorker<11, THRESHOLD[1]>, &test_file, &control_file, &heavy_hitters[1]);
    worker_threads[2] = std::thread(hashWorker<12, THRESHOLD[2]>, &test_file, &control_file, &heavy_hitters[2]);
    worker_threads[3] = std::thread(hashWorker<13, THRESHOLD[3]>, &test_file, &control_file, &heavy_hitters[3]);
    worker_threads[4] = std::thread(hashWorker<14, THRESHOLD[4]>, &test_file, &control_file, &heavy_hitters[4]);
    worker_threads[5] = std::thread(hashWorker<15, THRESHOLD[5]>, &test_file, &control_file, &heavy_hitters[5]);
    worker_threads[6] = std::thread(hashWorker<16, THRESHOLD[6]>, &test_file, &control_file, &heavy_hitters[6]);
    worker_threads[7] = std::thread(hashWorker<17, THRESHOLD[7]>, &test_file, &control_file, &heavy_hitters[7]);
    worker_threads[8] = std::thread(hashWorker<18, THRESHOLD[8]>, &test_file, &control_file, &heavy_hitters[8]);
    worker_threads[9] = std::thread(hashWorker<19, THRESHOLD[9]>, &test_file, &control_file, &heavy_hitters[9]);
    worker_threads[10] = std::thread(hashWorker<20, THRESHOLD[10]>, &test_file, &control_file, &heavy_hitters[10]);

    for (int n = 0; n < N_LENGTH; n++) {
        worker_threads[n].join();
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
