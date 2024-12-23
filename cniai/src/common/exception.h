#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace cniai::exception {

class CniaiException : public std::runtime_error {
public:
    static auto constexpr MAX_FRAMES = 128;

    explicit CniaiException(char const *file, std::size_t line, std::string const &msg);

    ~CniaiException() noexcept override;

    [[nodiscard]] std::string getTrace() const;

    static std::string demangle(char const *name);

private:
    std::array<void *, MAX_FRAMES> mCallstack{};
    int mNbFrames;
};

} // namespace cniai::exception
