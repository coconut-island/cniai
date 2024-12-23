#pragma once

#include "common/assert.h"
#include "nvcommon/cuda_util.h"
#include "tensorrt/cuda_stream.h"
#include "tensorrt/ibuffer.h"
#include "tensorrt/itensor.h"
#include "tensorrt/memory_counters.h"

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdlib>
#include <list>
#include <memory>
#include <mutex>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <vector>

namespace cniai::tensorrt {

// CRTP base class
template <typename TDerived, MemoryType memoryType, bool count = true>
class BaseAllocator {
public:
    using ValueType = void;
    using PointerType = ValueType *;
    using SizeType = std::size_t;
    static auto constexpr kMemoryType = memoryType;

    PointerType allocate(SizeType n) {
        PointerType ptr{};
        static_cast<TDerived *>(this)->allocateImpl(&ptr, n);
        if constexpr (count) {
            MemoryCounters::getInstance().allocate<memoryType>(n);
        }
        return ptr;
    }

    void deallocate(PointerType ptr, SizeType n) {
        if (ptr) {
            static_cast<TDerived *>(this)->deallocateImpl(ptr, n);
            if constexpr (count)
                MemoryCounters::getInstance().deallocate<memoryType>(n);
        }
    }

    [[nodiscard]] MemoryType constexpr getMemoryType() const { return memoryType; }
};

class CudaAllocator : public BaseAllocator<CudaAllocator, MemoryType::kGPU> {
    friend class BaseAllocator<CudaAllocator, MemoryType::kGPU>;

public:
    CudaAllocator() noexcept = default;

protected:
    void
    allocateImpl(PointerType *ptr,
                 SizeType n) // NOLINT(readability-convert-member-functions-to-static)
    {
        CUDA_CHECK(cudaMalloc(ptr, n));
    }

    void deallocateImpl( // NOLINT(readability-convert-member-functions-to-static)
        PointerType ptr, [[maybe_unused]] SizeType n) {
        CUDA_CHECK(cudaFree(ptr));
    }
};

class CudaAllocatorAsync : public BaseAllocator<CudaAllocatorAsync, MemoryType::kGPU> {
    friend class BaseAllocator<CudaAllocatorAsync, MemoryType::kGPU>;

public:
    using CudaStreamPtr = std::shared_ptr<CudaStream>;

    explicit CudaAllocatorAsync(CudaStreamPtr stream) : mCudaStream(std::move(stream)) {
        std::string a = "{}";
        cniai::str_util::fmtstr(a, 1);
        CNIAI_CHECK_WITH_INFO(static_cast<bool>(mCudaStream), "Undefined CUDA stream");
    }

    [[nodiscard]] CudaStreamPtr getCudaStream() const { return mCudaStream; }

protected:
    void allocateImpl(PointerType *ptr, SizeType n) {
        CUDA_CHECK(cudaMallocAsync(ptr, n, mCudaStream->get()));
    }

    void deallocateImpl(PointerType ptr, [[maybe_unused]] SizeType n) {
        CUDA_CHECK(cudaFreeAsync(ptr, mCudaStream->get()));
    }

private:
    CudaStreamPtr mCudaStream;
};

class UVMAllocator : public BaseAllocator<UVMAllocator, MemoryType::kUVM> {
    friend class BaseAllocator<UVMAllocator, MemoryType::kUVM>;

public:
    using Base = BaseAllocator<UVMAllocator, MemoryType::kUVM>;
    UVMAllocator() noexcept = default;

protected:
    void
    allocateImpl(PointerType *ptr,
                 SizeType n) // NOLINT(readability-convert-member-functions-to-static)
    {
        CUDA_CHECK(cudaMallocManaged(ptr, n));
        // TLLM_CUDA_CHECK(::cudaMemAdvise(ptr, n, cudaMemAdviseSetPreferredLocation, 0));
    }

    void deallocateImpl( // NOLINT(readability-convert-member-functions-to-static)
        PointerType ptr, [[maybe_unused]] SizeType n) {
        CUDA_CHECK(cudaFree(ptr));
    }
};

class PinnedAllocator : public BaseAllocator<PinnedAllocator, MemoryType::kPINNED> {
    friend class BaseAllocator<PinnedAllocator, MemoryType::kPINNED>;

public:
    using Base = BaseAllocator<PinnedAllocator, MemoryType::kPINNED>;
    PinnedAllocator() noexcept = default;

protected:
    void
    allocateImpl(PointerType *ptr,
                 SizeType n) // NOLINT(readability-convert-member-functions-to-static)
    {
        CUDA_CHECK(::cudaHostAlloc(ptr, n, cudaHostAllocDefault));
    }

    void deallocateImpl( // NOLINT(readability-convert-member-functions-to-static)
        PointerType ptr, [[maybe_unused]] SizeType n) {
        CUDA_CHECK(::cudaFreeHost(ptr));
    }
};

class HostAllocator : public BaseAllocator<HostAllocator, MemoryType::kCPU> {
    friend class BaseAllocator<HostAllocator, MemoryType::kCPU>;

public:
    HostAllocator() noexcept = default;

protected:
    void
    allocateImpl(PointerType *ptr,
                 SizeType n) // NOLINT(readability-convert-member-functions-to-static)
    {
        *ptr = std::malloc(n);
        if (*ptr == nullptr) {
            throw std::bad_alloc();
        }
    }

    void deallocateImpl( // NOLINT(readability-convert-member-functions-to-static)
        PointerType ptr, [[maybe_unused]] SizeType n) {
        std::free(ptr);
    }
};

template <MemoryType memoryType>
class BorrowingAllocator
    : public BaseAllocator<BorrowingAllocator<memoryType>, memoryType, false> {
    friend class BaseAllocator<BorrowingAllocator<memoryType>, memoryType, false>;

public:
    using Base = BaseAllocator<BorrowingAllocator<memoryType>, memoryType, false>;
    using PointerType = typename Base::PointerType;
    using SizeType = typename Base::SizeType;

    BorrowingAllocator(void *ptr, SizeType capacity) : mPtr(ptr), mCapacity(capacity) {
        CNIAI_CHECK_WITH_INFO(capacity == 0 || static_cast<bool>(mPtr),
                              "Undefined pointer");
        CNIAI_CHECK_WITH_INFO(mCapacity >= 0, "Capacity must be non-negative");
    }

protected:
    void
    allocateImpl(PointerType *ptr,
                 SizeType n) // NOLINT(readability-convert-member-functions-to-static)
    {
        if (n <= mCapacity) {
            *ptr = mPtr;
        } else {
            throw std::bad_alloc();
        }
    }

    void deallocateImpl( // NOLINT(readability-convert-member-functions-to-static)
        [[maybe_unused]] PointerType ptr, [[maybe_unused]] SizeType n) {}

private:
    PointerType mPtr;
    SizeType mCapacity;
};

using CpuBorrowingAllocator = BorrowingAllocator<MemoryType::kCPU>;
using GpuBorrowingAllocator = BorrowingAllocator<MemoryType::kGPU>;
using PinnedBorrowingAllocator = BorrowingAllocator<MemoryType::kPINNED>;
using ManagedBorrowingAllocator = BorrowingAllocator<MemoryType::kUVM>;
using UVMBorrowingAllocator = BorrowingAllocator<MemoryType::kUVM>;

// Adopted from
// https://github.com/NVIDIA/TensorRT/blob/release/8.6/samples/common/buffers.h

//!
//! \brief  The GenericBuffer class is a templated class for buffers.
//!
//! \details This templated RAII (Resource Acquisition Is Initialization) class handles
//! the allocation,
//!          deallocation, querying of buffers on both the device and the host.
//!          It can handle data of arbitrary types because it stores byte buffers.
//!          The template parameters AllocFunc and FreeFunc are used for the
//!          allocation and deallocation of the buffer.
//!          AllocFunc must be a functor that takes in (void** ptr, size_t size)
//!          and returns bool. ptr is a pointer to where the allocated buffer address
//!          should be stored. size is the amount of memory in bytes to allocate. The
//!          boolean indicates whether or not the memory allocation was successful.
//!          FreeFunc must be a functor that takes in (void* ptr) and returns void.
//!          ptr is the allocated buffer address. It must work with nullptr input.
//!
template <typename TAllocator>
class GenericBuffer : virtual public IBuffer {
public:
    using AllocatorType = TAllocator;

    //!
    //! \brief Construct an empty buffer.
    //!
    explicit GenericBuffer(nvinfer1::DataType type,
                           TAllocator allocator = {}) // NOLINT(*-pro-type-member-init)
        : GenericBuffer{0, type, std::move(allocator)} {};

    //!
    //! \brief Construct a buffer with the specified allocation size in number of
    //! elements.
    //!
    explicit GenericBuffer( // NOLINT(*-pro-type-member-init)
        std::size_t size, nvinfer1::DataType type, TAllocator allocator = {})
        : GenericBuffer{size, size, type, std::move(allocator)} {};

    GenericBuffer(GenericBuffer &&buf) noexcept
        : mSize{buf.mSize}, mCapacity{buf.mCapacity}, mType{buf.mType},
          mAllocator{std::move(buf.mAllocator)}, mBuffer{buf.mBuffer} {
        buf.mSize = 0;
        buf.mCapacity = 0;
        buf.mBuffer = nullptr;
    }

    GenericBuffer &operator=(GenericBuffer &&buf) noexcept {
        if (this != &buf) {
            mAllocator.deallocate(mBuffer, toBytes(mCapacity));
            mSize = buf.mSize;
            mCapacity = buf.mCapacity;
            mType = buf.mType;
            mAllocator = std::move(buf.mAllocator);
            mBuffer = buf.mBuffer;
            // Reset buf.
            buf.mSize = 0;
            buf.mCapacity = 0;
            buf.mBuffer = nullptr;
        }
        return *this;
    }

    //!
    //! \brief Returns pointer to underlying array.
    //!
    void *data() override { return mSize > 0 ? mBuffer : nullptr; }

    //!
    //! \brief Returns pointer to underlying array.
    //!
    [[nodiscard]] void const *data() const override {
        return mSize > 0 ? mBuffer : nullptr;
    }

    //!
    //! \brief Returns the size (in number of elements) of the buffer.
    //!
    [[nodiscard]] std::size_t getSize() const override { return mSize; }

    //!
    //! \brief Returns the capacity of the buffer.
    //!
    [[nodiscard]] std::size_t getCapacity() const override { return mCapacity; }

    //!
    //! \brief Returns the type of the buffer.
    //!
    [[nodiscard]] nvinfer1::DataType getDataType() const override { return mType; }

    //!
    //! \brief Returns the memory type of the buffer.
    //!
    [[nodiscard]] MemoryType getMemoryType() const override {
        return mAllocator.getMemoryType();
    }

    //!
    //! \brief Resizes the buffer. This is a no-op if the new size is smaller than or
    //! equal to the current capacity.
    //!
    void resize(std::size_t newSize) override {
        if (mCapacity < newSize) {
            mAllocator.deallocate(mBuffer, toBytes(mCapacity));
            mBuffer = mAllocator.allocate(toBytes(newSize));
            mCapacity = newSize;
        }
        mSize = newSize;
    }

    //!
    //! \brief Releases the buffer.
    //!
    void release() override {
        mAllocator.deallocate(mBuffer, toBytes(mCapacity));
        mSize = 0;
        mCapacity = 0;
        mBuffer = nullptr;
    }

    ~GenericBuffer() override {
        try {
            mAllocator.deallocate(mBuffer, toBytes(mCapacity));
        } catch (std::exception const &e) {
            CNIAI_THROW(e.what());
        }
    }

protected:
    explicit GenericBuffer(std::size_t size, std::size_t capacity,
                           nvinfer1::DataType type, TAllocator allocator = {})
        : mSize{size}, mCapacity{capacity}, mType{type}, mAllocator{std::move(allocator)},
          mBuffer{capacity > 0 ? mAllocator.allocate(toBytes(capacity)) : nullptr} {
        CNIAI_CHECK(size <= capacity);
        CNIAI_CHECK(capacity == 0 || size > 0);
    }

private:
    std::size_t mSize{0}, mCapacity{0};
    nvinfer1::DataType mType;
    TAllocator mAllocator;
    void *mBuffer;
};

using DeviceBuffer = GenericBuffer<CudaAllocatorAsync>;
using StaticDeviceBuffer = GenericBuffer<CudaAllocator>;
using HostBuffer = GenericBuffer<HostAllocator>;
using PinnedBuffer = GenericBuffer<PinnedAllocator>;
using UVMBuffer = GenericBuffer<UVMAllocator>;

template <typename T>
typename std::make_unsigned<T>::type nonNegative(T value) {
    CNIAI_CHECK_WITH_INFO(value >= 0, "Value must be non-negative");
    return static_cast<typename std::make_unsigned<T>::type>(value);
}

template <typename TAllocator>
class GenericTensor : virtual public ITensor, public GenericBuffer<TAllocator> {
public:
    using Base = GenericBuffer<TAllocator>;

    //!
    //! \brief Construct an empty tensor.
    //!
    explicit GenericTensor(nvinfer1::DataType type, TAllocator allocator = {})
        : Base{type, std::move(allocator)} {
        mDims.nbDims = 0;
    }

    //!
    //! \brief Construct a tensor with the specified allocation dimensions.
    //!
    explicit GenericTensor(nvinfer1::Dims const &dims, nvinfer1::DataType type,
                           TAllocator allocator = {})
        : Base{nonNegative(volume(dims)), type, std::move(allocator)}, mDims{dims} {}

    explicit GenericTensor(nvinfer1::Dims const &dims, std::size_t capacity,
                           nvinfer1::DataType type, TAllocator allocator = {})
        : Base{nonNegative(volume(dims)), capacity, type, std::move(allocator)},
          mDims{dims} {}

    GenericTensor(GenericTensor &&tensor) noexcept
        : Base{std::move(tensor)}, mDims{tensor.dims} {
        tensor.mDims.nbDims = 0;
    }

    GenericTensor &operator=(GenericTensor &&tensor) noexcept {
        if (this != &tensor) {
            Base::operator=(std::move(tensor));
            mDims = tensor.dims;
            // Reset tensor.
            tensor.mDims.nbDims = 0;
        }
        return *this;
    }

    [[nodiscard]] nvinfer1::Dims const &getShape() const override { return mDims; }

    void reshape(nvinfer1::Dims const &dims) override {
        Base::resize(nonNegative(volume(dims)));
        mDims = dims;
    }

    void resize(std::size_t newSize) override { ITensor::resize(newSize); }

    void release() override {
        Base::release();
        mDims.nbDims = 0;
    }

private:
    nvinfer1::Dims mDims{};
};

using DeviceTensor = GenericTensor<CudaAllocatorAsync>;
using StaticDeviceTensor = GenericTensor<CudaAllocator>;
using HostTensor = GenericTensor<HostAllocator>;
using PinnedTensor = GenericTensor<PinnedAllocator>;
using UVMTensor = GenericTensor<UVMAllocator>;

} // namespace cniai::tensorrt