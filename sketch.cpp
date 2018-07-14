#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <limits>
#include <unordered_set>


const unsigned int N_HASH = 4;
const unsigned int M = 14;
const unsigned int RHO = 145;


/**
 * @brief Compute H3 hash
 */
unsigned int hashH3(unsigned int key, unsigned short* seeds) {
    unsigned int result = 0;
    for (int i = 0; i < 32; i++) {
        if (key & 1)
            result ^= seeds[i];
        key >>= 1;
    }
    return result;
}


/**
 * @brief Extract k-mers from DNA sequences in a .fasta file
 * 
 * @param input Input stream from where the .fasta file will be read
 * @param length Length of the k-mers
 * @return std::vector<unsigned int> 
 */
std::vector<unsigned int> parseFasta(std::istream& input, int length) {
    std::vector<unsigned int> data_vectors;

    while (!input.eof()) {
        std::string line;
        input >> line;

        // Skip line
        if (line[0] == '>') {
            continue;
        }

        unsigned int sequence = 0;
        for (unsigned int i = 0; i < line.length(); i++) {
            sequence <<= 2;
            switch(line[i]) {
            case 'A':
                sequence |= 0;
                break;
            case 'C':
                sequence |= 1;
                break;
            case 'T':
                sequence |= 2;
                break;
            case 'G':
                sequence |= 3;
                break;
            }

            if (i >= length - 1)
                data_vectors.push_back(sequence);
        }
    }

    return data_vectors;
}


int main(int argc, char* argv[]) {
    // Generate hash vectors
    unsigned short seeds[N_HASH][32];
    for (unsigned int i = 0; i < N_HASH; i++) {
        for (unsigned int j = 0; j < 32; j++) {
            seeds[i][j] = rand() & ((1 << M) - 1);
        }
    }

    // Create sketch
    unsigned int sketch[N_HASH][1 << M] = {0};
    
    // Start time measurement
    auto start = std::chrono::steady_clock::now();
    
    // Parse data set
    std::ifstream dataset_file("data/test.fasta");
    std::vector<unsigned int> data_vectors = parseFasta(dataset_file, 16);
    dataset_file.close();

    std::unordered_set<unsigned int> heavy_hitters;
    heavy_hitters.reserve(data_vectors.size() / 10);

    // Hash values
    for (unsigned int i = 0; i < data_vectors.size(); i++) {
        unsigned int min_hits = std::numeric_limits<unsigned int>::max();
        unsigned int hashes[N_HASH];

        for (unsigned int j = 0; j < N_HASH; j++) {
            hashes[j] = hashH3(data_vectors[i], seeds[j]);
            if (sketch[j][hashes[j]] < min_hits) {
                min_hits = sketch[j][hashes[j]];
            }
        }

        for (unsigned int j = 0; j < N_HASH; j++) {
            if (sketch[j][hashes[j]] == min_hits) {
                sketch[j][hashes[j]]++;
            }
        }

        if (min_hits + 1 >= RHO) {
            heavy_hitters.insert(data_vectors[i]);
        }
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "Data vectors: " << data_vectors.size() << std::endl;
    std::cout << "Execution time: " << diff.count() << " s" << std::endl;

    // Write heavy-hitters to output file
    std::ofstream heavy_hitters_file("heavy-hitters.txt");
    for (auto x : heavy_hitters) {
        std::string sequence;

        for (int i = 0; i < 16; i++) {
            switch (x << (i * 2) >> 30) {
            case 0:
                sequence += 'A';
                break;
            case 1:
                sequence += 'C';
                break;
            case 2:
                sequence += 'T';
                break;
            case 3:
                sequence += 'G';
                break;
            }
        }
        heavy_hitters_file << sequence << std::endl;
    }
    heavy_hitters_file.close();

    return 0;
}
