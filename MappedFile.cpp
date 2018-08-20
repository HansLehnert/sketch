#include "MappedFile.hpp"

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>


MappedFile::~MappedFile() {
    munmap(m_data, m_size);
    close(m_fd);
}


MappedFile MappedFile::load(std::string filename) {
    MappedFile file;

    file.m_fd = open(filename.c_str(), O_RDONLY);

    struct stat file_stat;
    fstat(file.m_fd, &file_stat);

    file.m_size = file_stat.st_size;

    file.m_data = (char*)mmap(
        NULL,
        file.m_size,
        PROT_READ,
        MAP_PRIVATE,
        file.m_fd,
        0);

    return file;
}


const char* MappedFile::data() {
    return m_data;
}


int MappedFile::size() {
    return m_size;
}
