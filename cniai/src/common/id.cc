#include "id.h"

namespace cniai::id {

IdOverflowException::IdOverflowException() : std::runtime_error("ID overflow detected") {}

IdGenerator::IdGenerator(IdType start, IdType step) : currentId(start), step(step) {}

IdType IdGenerator::getNextId() {
    IdType id = currentId.fetch_add(step, std::memory_order_relaxed);
    if (id + step < id) {
        throw IdOverflowException();
    }
    return id;
}

void IdGenerator::reset(IdType newStart) {
    std::lock_guard<std::mutex> lock(mutex);
    currentId.store(newStart, std::memory_order_relaxed);
}

void IdGenerator::setStep(IdType newStep) {
    std::lock_guard<std::mutex> lock(mutex);
    step = newStep;
}

} // namespace cniai::id