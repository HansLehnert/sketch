#include <vector>
#include <istream>

/**
 * @brief Extract k-mers from DNA sequences in a .fasta file
 * 
 * @param input Input stream from where the .fasta file will be read
 * @param length Length of the k-mers
 * @return std::vector<unsigned int> 
 */
std::vector<unsigned int> parseFasta(std::istream& input, int length);
