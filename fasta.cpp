/**
 * @brief .fasta file parsing for k-mers extraction
 *
 * @file fasta.cpp
 * @author your name
 */

#include "fasta.hpp"


std::vector<unsigned long> parseFasta(std::istream& input, int length) {
    std::vector<unsigned long> data_vectors;
    unsigned long mask = ~0UL >> (64 - length * 2);

    while (!input.eof()) {
        std::string line;
        input >> line;

        // Skip line
        if (line[0] == '>') {
            continue;
        }

        unsigned long sequence = 0;
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
                data_vectors.push_back(sequence & mask);
        }
    }

    return data_vectors;
}


std::string sequenceToString(unsigned long sequence, int length) {
    std::string result = "";

    for (int i = length - 1; i >= 0; i--) {
        switch ((sequence >> (i * 2)) & 0b11) {
        case 0:
            result += 'A';
            break;
        case 1:
            result += 'C';
            break;
        case 2:
            result += 'T';
            break;
        case 3:
            result += 'G';
            break;
        }
    }

    return result;
}
