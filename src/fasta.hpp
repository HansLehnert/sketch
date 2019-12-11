/**
 * @brief .fasta file parsing for k-mers extraction
 *
 * @file fasta.hpp
 * @author Hans Lehnert
 */
#pragma once

#include <vector>
#include <string>
#include <istream>


/**
 * @brief Extract k-mers from DNA sequences in a .fasta file
 *
 * @param input Input stream from where the .fasta file will be read
 * @param length Length of the k-mers
 * @return std::vector<unsigned int>
 */
std::vector<unsigned long> parseFasta(std::istream& input, int length);

std::vector<unsigned long> parseFasta(
    const char* data, int data_length, int sequence_length);

std::vector<unsigned long> parseFasta(
    const char* data,
    int data_length,
    int sequence_length,
    unsigned long mask,
    std::vector<unsigned char>* lengths = nullptr);


/**
 * @brief Convert a k-mer stored as binary to it's string representation
 *
 * @param sequence Binary representation
 * @param length Amount of bases to extract from the binary representation
 * @param reverse If true, encoded sequence starts from LSB to MSB
 * @return std::string
 */
std::string sequenceToString(
    unsigned long sequence,
    int length,
    bool reverse = false
);
