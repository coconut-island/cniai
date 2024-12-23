#pragma once

#include <fstream>

namespace cniai::file_util {

std::ifstream::pos_type fileSize(const char *fileName);

bool fileExists(const std::string &filename);

} // namespace cniai::file_util
