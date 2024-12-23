#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <stdexcept>

namespace cniai::id {

using IdType = std::uint64_t;

class IdOverflowException : public std::runtime_error {
public:
    IdOverflowException();
};

class IdGenerator {
public:
    explicit IdGenerator(IdType start = 0, IdType step = 1);

    IdType getNextId();

    void reset(IdType newStart = 0);

    void setStep(IdType newStep);

private:
    std::atomic<IdType> currentId;
    IdType step;
    std::mutex mutex;
};

} // namespace cniai::id