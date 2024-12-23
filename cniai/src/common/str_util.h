#pragma once

#include <spdlog/spdlog.h>

#include <cassert>
#include <cstdarg>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace cniai::str_util {

auto constexpr kDefaultDelimiter = ", ";

bool endsWith(const std::string &str, const std::string &suffix);

bool startsWith(const char *str, const char *prefix);

// std::string vformat(char const *fmt, va_list args);

template <typename Format, typename... Args>
std::string fmtstr(const Format format, Args &&...args) {
    return fmt::format(fmt::runtime(format), std::forward<Args>(args)...);
}

template <typename U, typename TStream, typename T>
inline TStream &arr2outCasted(TStream &out, T *arr, size_t size,
                              char const *delim = kDefaultDelimiter) {
    out << "(";
    if (size > 0) {
        for (size_t i = 0; i < size - 1; ++i) {
            out << static_cast<U>(arr[i]) << delim;
        }
        out << static_cast<U>(arr[size - 1]);
    }
    out << ")";
    return out;
};

template <typename TStream, typename T>
inline TStream &arr2out(TStream &out, T *arr, size_t size,
                        char const *delim = kDefaultDelimiter) {
    return arr2outCasted<T>(out, arr, size, delim);
}

template <typename T>
inline std::string arr2str(T *arr, size_t size, char const *delim = kDefaultDelimiter) {
    std::stringstream ss;
    return arr2out(ss, arr, size, delim).str();
};

template <typename T>
inline std::string vec2str(std::vector<T> vec, char const *delim = kDefaultDelimiter) {
    return arr2str(vec.data(), vec.size(), delim);
};

} // namespace cniai::str_util