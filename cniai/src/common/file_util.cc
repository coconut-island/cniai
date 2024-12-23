#include "common/file_util.h"

#include <fstream>
#include <iostream>

namespace cniai::file_util {

std::ifstream::pos_type fileSize(const char *fileName) {
    std::ifstream in(fileName, std::ifstream::ate | std::ifstream::binary);
    return in.tellg();
}

bool fileExists(const std::string &filename) {
    std::ifstream file(filename);
    return file.good();
}

} // namespace cniai::file_util