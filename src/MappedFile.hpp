#pragma once

#include <string>

class MappedFile {
public:
    ~MappedFile();

    static MappedFile load(std::string filename);

    const char* data();
    int size();

private:
    MappedFile() = default;

    int m_fd;
    int m_size;
    char* m_data;
};
