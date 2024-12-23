#pragma once

#include "tensorrt/ibuffer.h"

#include <atomic>
#include <cstddef>
#include <string>

namespace cniai::tensorrt {

class MemoryCounters {
public:
    using SizeType = std::size_t;
    using DiffType = std::ptrdiff_t;

    MemoryCounters() = default;

    [[nodiscard]] SizeType getGpu() const { return mGpu; }

    [[nodiscard]] SizeType getCpu() const { return mCpu; }

    [[nodiscard]] SizeType getPinned() const { return mPinned; }

    [[nodiscard]] SizeType getUVM() const { return mUVM; }

    [[nodiscard]] DiffType getGpuDiff() const { return mGpuDiff; }

    [[nodiscard]] DiffType getCpuDiff() const { return mCpuDiff; }

    [[nodiscard]] DiffType getPinnedDiff() const { return mPinnedDiff; }

    [[nodiscard]] DiffType getUVMDiff() const { return mUVMDiff; }

    template <MemoryType T>
    void allocate(SizeType size) {
        auto const sizeDiff = static_cast<DiffType>(size);
        if constexpr (T == MemoryType::kGPU) {
            mGpu += size;
            mGpuDiff = sizeDiff;
        } else if constexpr (T == MemoryType::kCPU) {
            mCpu += size;
            mCpuDiff = sizeDiff;
        } else if constexpr (T == MemoryType::kPINNED) {
            mPinned += size;
            mPinnedDiff = sizeDiff;
        } else if constexpr (T == MemoryType::kUVM) {
            mUVM += size;
            mUVMDiff = sizeDiff;
        } else {
            CNIAI_THROW("Unknown memory type: %s", MemoryTypeString<T>::value);
        }
    }

    void allocate(MemoryType memoryType, SizeType size);

    template <MemoryType T>
    void deallocate(SizeType size) {
        auto const sizeDiff = -static_cast<DiffType>(size);
        if constexpr (T == MemoryType::kGPU) {
            mGpu -= size;
            mGpuDiff = sizeDiff;
        } else if constexpr (T == MemoryType::kCPU) {
            mCpu -= size;
            mCpuDiff = sizeDiff;
        } else if constexpr (T == MemoryType::kPINNED) {
            mPinned -= size;
            mPinnedDiff = sizeDiff;
        } else if constexpr (T == MemoryType::kUVM) {
            mUVM -= size;
            mUVMDiff = sizeDiff;
        } else {
            CNIAI_THROW("Unknown memory type: %s", MemoryTypeString<T>::value);
        }
    }

    void deallocate(MemoryType memoryType, SizeType size);

    static MemoryCounters &getInstance();

    static std::string bytesToString(SizeType bytes, int precision = 2);

    static std::string bytesToString(DiffType bytes, int precision = 2);

    [[nodiscard]] std::string toString() const;

private:
    std::atomic<SizeType> mGpu{}, mCpu{}, mPinned{}, mUVM{};
    std::atomic<DiffType> mGpuDiff{}, mCpuDiff{}, mPinnedDiff{}, mUVMDiff{};
};

} // namespace cniai::tensorrt
