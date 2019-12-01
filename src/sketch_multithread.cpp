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
#include <thread>
#include <memory>
#include <mutex>
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


void sketchWorker(
    const unsigned short* seeds,
    const int length,
    const int threshold,
    const float growth,
    std::shared_ptr<MappedFile> test_file,
    std::shared_ptr<MappedFile> control_file,
    std::mutex* control_file_load_mutex,
    std::unordered_map<Sequence, int>* heavy_hitters
) {
    Sketch<int, N_HASH, HASH_BITS> sketch;
    memset(&sketch, 0, sizeof(sketch));

    // Test step
    unsigned long test_size = test_file->size();
    const char* test_data = test_file->data();

    int kmer_start = 0;

    while (kmer_start < test_size) {
        bool sequence_end = false;

        unsigned short hashes[N_HASH] = {0};
        Sequence encoded_kmer;

        int i = 0;
        for (; i < length; i++) {
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
        }

        if (sequence_end) {
            kmer_start += i + 1;
            continue;
        }

        // Add to sketch
        unsigned int min_hits = std::numeric_limits<unsigned int>::max();

        for (int j = 0; j < N_HASH; j++) {
            int counter = sketch[j][hashes[j]];
            if (counter < min_hits) {
                min_hits = counter;
            }
        }

        for (int j = 0; j < N_HASH; j++) {
            if (sketch[j][hashes[j]] == min_hits) {
                sketch[j][hashes[j]]++;
            }
        }

        if (min_hits + 1 >= threshold) {
            (*heavy_hitters)[encoded_kmer] = min_hits;
        }

        kmer_start++;
    }

    test_file = nullptr;

    // Scale frequencies
    for (auto& heavy_hitter : *heavy_hitters) {
        heavy_hitter.second /= growth;
    }

    // Load control file
    control_file_load_mutex->lock();

    if (!control_file->isLoaded())
        control_file->load();

    control_file_load_mutex->unlock();

    // Control step
    unsigned long control_size = control_file->size();
    const char* control_data = control_file->data();

    kmer_start = 0;

    while (kmer_start < control_size) {
        bool sequence_end = false;

        Sequence encoded_kmer;

        int i = 0;
        for (; i < length; i++) {
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
        }

        if (sequence_end) {
            kmer_start += i + 1;
            continue;
        }

        std::unordered_map<Sequence, int>::iterator counter;
        counter = heavy_hitters->find(encoded_kmer);
        if (counter != heavy_hitters->end()) {
            counter->second--;
        }

        kmer_start++;
    }

    // Remove kmers whose frequencies are too low after control step
    auto next = heavy_hitters->begin();
    while (next != heavy_hitters->end()) {
        auto current = next++;
        if (current->second <= 0)
            heavy_hitters->erase(current);
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

    // Generate seeds
    size_t n_seeds = N_HASH * settings.max_length * 4;
    unsigned short* seeds = new unsigned short[n_seeds];
    for (unsigned int i = 0; i < n_seeds; i++) {
        seeds[i] = rand() & ((1 << HASH_BITS) - 1);
    }

    // Load memory mapped files
    std::shared_ptr<MappedFile> test_file =
        std::make_shared<MappedFile>(argv[1]);
    std::shared_ptr<MappedFile> control_file =
        std::make_shared<MappedFile>(argv[2], false);

    // Start time measurement
    auto start_time = std::chrono::steady_clock::now();

    std::vector<std::unordered_map<Sequence, int>> heavy_hitters(
        settings.n_length);
    std::mutex control_file_load_mutex;

    // Spawn threads
    std::vector<std::thread> threads;
    threads.reserve(settings.n_length);
    for (int i = 0; i < settings.n_length; i++) {
        threads.emplace_back(
            sketchWorker,
            seeds,
            settings.min_length + i,
            settings.threshold[i],
            settings.growth,
            test_file,
            control_file,
            &control_file_load_mutex,
            &heavy_hitters[i]
        );
    }

    for (int i = 0; i < settings.n_length; i++) {
        threads[i].join();
    }

    auto end_time = std::chrono::steady_clock::now();

    // Report times
    // std::clog
    //     << "Test time: "
    //     << std::chrono::duration<double>(test_end_time - start_time).count()
    //     << " s"
    //     << std::endl;
    // std::clog
    //     << "Control time: "
    //     << std::chrono::duration<double>(end_time - test_end_time).count()
    //     << " s"
    //     << std::endl;
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

        for (auto& x : heavy_hitters[n]) {
            std::cout
                << sequenceToString(x.first.data[0], settings.min_length + n)
                << std::endl;
        }
    }

    std::clog << "Heavy-hitters (total): " << heavy_hitters_count << std::endl;

    delete[] seeds;

    return 0;
}
