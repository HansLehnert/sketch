#pragma once

#include <string>

class MappedFile {
public:
    MappedFile(std::string filename, bool load = true);
    ~MappedFile();

    void load();

    bool isLoaded();
    const char* data();
    int size();

private:

    bool m_isLoaded;
    std::string m_filename;
    int m_fd;
    int m_size;
    char* m_data;
};
