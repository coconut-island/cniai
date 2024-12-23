#include "tensorrt/memory_counters.h"

#include "common/str_util.h"

#include <array>
#include <cmath>

namespace cniai::tensorrt {

auto constexpr kByteUnits = std::array{"B", "KB", "MB", "GB", "TB", "PB", "EB"};

static std::string doubleBytesToString(double bytes, int precision) {
    std::uint32_t unitIdx{0};

    while (std::abs(bytes) >= 1024.0 && unitIdx < kByteUnits.size() - 1) {
        bytes /= 1024.0;
        ++unitIdx;
    }
    auto const format = "%." + std::to_string(precision) + "f %s";
    return cniai::str_util::fmtstr(format.c_str(), bytes, kByteUnits[unitIdx]);
}

std::string MemoryCounters::bytesToString(SizeType bytes, int precision) {
    return doubleBytesToString(static_cast<double>(bytes), precision);
}

std::string MemoryCounters::bytesToString(DiffType bytes, int precision) {
    return doubleBytesToString(static_cast<double>(bytes), precision);
}

std::string MemoryCounters::toString() const {
    return cniai::str_util::fmtstr(
        "[MemUsage] GPU %s, CPU %s, Pinned %s", bytesToString(this->getGpu()).c_str(),
        bytesToString(this->getCpu()).c_str(), bytesToString(this->getPinned()).c_str());
}

void MemoryCounters::allocate(MemoryType memoryType, MemoryCounters::SizeType size) {
    switch (memoryType) {
    case MemoryType::kGPU:
        allocate<MemoryType::kGPU>(size);
        break;
    case MemoryType::kCPU:
        allocate<MemoryType::kCPU>(size);
        break;
    case MemoryType::kPINNED:
        allocate<MemoryType::kPINNED>(size);
        break;
    default:
        CNIAI_THROW("Unknown memory type");
    }
}

void MemoryCounters::deallocate(MemoryType memoryType, MemoryCounters::SizeType size) {
    switch (memoryType) {
    case MemoryType::kGPU:
        deallocate<MemoryType::kGPU>(size);
        break;
    case MemoryType::kCPU:
        deallocate<MemoryType::kCPU>(size);
        break;
    case MemoryType::kPINNED:
        deallocate<MemoryType::kPINNED>(size);
        break;
    default:
        CNIAI_THROW("Unknown memory type");
    }
}

MemoryCounters &MemoryCounters::getInstance() {
    static MemoryCounters mInstance;
    return mInstance;
}

} // namespace cniai::tensorrt
