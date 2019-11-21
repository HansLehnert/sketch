#include "MappedFile.hpp"

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>


MappedFile::MappedFile(std::string filename, bool load) : m_isLoaded(false) {
    m_filename = filename;

    if (load)
        this->load();
}

void MappedFile::load() {
    if (m_isLoaded)
        return;

    m_fd = open(m_filename.c_str(), O_RDONLY);

    struct stat file_stat;
    fstat(m_fd, &file_stat);

    m_size = file_stat.st_size;
    m_data = (char*)mmap(NULL, m_size, PROT_READ, MAP_PRIVATE, m_fd, 0);
    m_isLoaded = true;
}


MappedFile::~MappedFile() {
    if (m_isLoaded)
        munmap(m_data, m_size);
        close(m_fd);
}


bool MappedFile::isLoaded() {
    return m_isLoaded;
}


const char* MappedFile::data() {
    return m_data;
}


int MappedFile::size() {
    return m_size;
}
