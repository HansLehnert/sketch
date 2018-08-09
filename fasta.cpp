/**
 * @brief .fasta file parsing for k-mers extraction
 *
 * @file fasta.cpp
 * @author your name
 */

#include "fasta.hpp"

#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>


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


std::vector<unsigned long> parseFasta(std::string filename, int length) {
    // Open memory-mapped file
    int file = open(filename.c_str(), O_RDONLY);

    struct stat file_stat;
    fstat(file, &file_stat);

    char* data = (char*)mmap(
        NULL, file_stat.st_size, PROT_READ, MAP_PRIVATE, file, 0);

    // Extract k-mers
    std::vector<unsigned long> data_vectors;
    data_vectors.reserve(file_stat.st_size);

    unsigned long mask = ~0UL >> (64 - length * 2);
    unsigned long sequence = 0;
    int sequence_length = 0;
    bool skip_line = false;
    for (int i = 0; i < file_stat.st_size; i++) {
        if (data[i] == '\n' || data[i] == '\r') {
            sequence = 0;
            sequence_length = 0;
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

            if (sequence_length++ >= length) {
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
