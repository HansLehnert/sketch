/**
 * @brief .fasta file parsing for k-mers extraction
 *
 * @file fasta.cpp
 * @author your name
 */

#include "fasta.hpp"

#include <iostream>


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


std::vector<unsigned long> parseFasta(
        const char* data, int data_length, int sequence_length) {
    std::vector<unsigned long> data_vectors;
    data_vectors.reserve(data_length);

    unsigned long mask = ~0UL >> (64 - sequence_length * 2);
    unsigned long sequence = 0;
    int parsed = 0;
    bool skip_line = false;

    for (int i = 0; i < data_length; i++) {
        if (data[i] == '\n' || data[i] == '\r') {
            sequence = 0;
            parsed = 0;
            skip_line = false;
        }
        else if (!skip_line) {
            if (data[i] == '>') {
                skip_line = true;
                continue;
            }

            sequence <<= 2;

            switch(data[i]) {
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

            sequence &= mask;

            if (++parsed >= sequence_length) {
                data_vectors.push_back(sequence);
            }
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
