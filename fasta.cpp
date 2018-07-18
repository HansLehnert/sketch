#include "fasta.hpp"


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