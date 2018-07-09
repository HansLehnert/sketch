#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <limits>


const unsigned int N_HASH = 4;
const unsigned int M = 14;


unsigned int hash(unsigned int key, unsigned int* seeds) {
    unsigned int result = 0;
    while (key) {
        if (key & 1)
            result ^= seeds[0];
        key >>= 1;
        seeds++;
    }
    return result;
}


int main(int argc, char* argv[]) {
    // Parse data set
    std::ifstream dataset_file("control.fasta");
    std::vector<int> data_vectors;

    while (!dataset_file.eof()) {
        std::string line;
        dataset_file >> line;

        if (line[0] == '>') {
            continue;
        }

        int sequence = 0;
        for (unsigned int i = 0; i < line.length(); i++) {
            sequence << 2;
            switch(line[i]) {
            case 'A':
                sequence |= 0;;
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

            if (i >= 15)
                data_vectors.push_back(sequence);
        }
    }

    std::cout << "Data vectors: " << data_vectors.size() << std::endl;

    // Generate hash vectors
    unsigned int seeds[N_HASH][32];
    for (unsigned int i = 0; i < N_HASH; i++) {
        for (unsigned int j = 0; j < 32; j++) {
            seeds[i][j] = rand() & ((1 << M) - 1);
        }
    }

    // Create sketch and hash
    unsigned int sketch[N_HASH][1 << M] = {0};

    auto start = std::chrono::steady_clock::now();

    for (unsigned int i = 0; i < data_vectors.size(); i++) {
        unsigned int min_hits = std::numeric_limits<unsigned int>::max();
        unsigned int min_func = 0;
        unsigned int min_hash = 0;
        for (unsigned int j = 0; j < N_HASH; j++) {
            unsigned int hash_result = hash(data_vectors[i], seeds[j]);
            if (sketch[j][hash_result] < min_hits) {
                min_hits = sketch[j][hash_result];
                min_func = j;
                min_hash = hash_result;
            }
            sketch[min_func][min_hash]++;
        }
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Execution time: " << diff.count() << " s" << std::endl;

    std::ofstream output_file("sketch.txt", std::ios::binary);
    output_file.write((char*)&sketch[0][0], N_HASH * (1 << M) * sizeof(int));

    return 0;
}
