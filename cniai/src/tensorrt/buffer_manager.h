#pragma once

#include "common/str_util.h"

#include "tensorrt/cuda_stream.h"
#include "tensorrt/ibuffer.h"
#include "tensorrt/itensor.h"

#include <NvInferRuntime.h>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace cniai::tensorrt {

//! \brief A helper class for managing memory on host and device.
class BufferManager {
public:
    using IBufferPtr = std::shared_ptr<IBuffer>;

    using ITensorPtr = std::shared_ptr<ITensor>;

    using CudaStreamPtr = std::shared_ptr<CudaStream>;

    static auto constexpr kBYTE_TYPE = nvinfer1::DataType::kUINT8;

    //! \brief Allocates an `IBuffer` of the given size on the GPU.
    [[nodiscard]] static IBufferPtr gpu(std::size_t size,
                                        const CudaStreamPtr &cudaStream = nullptr,
                                        nvinfer1::DataType type = kBYTE_TYPE);

    //! \brief Allocates an `ITensor` of the given dimensions on the GPU.
    [[nodiscard]] static ITensorPtr gpu(nvinfer1::Dims dims,
                                        const CudaStreamPtr &cudaStream = nullptr,
                                        nvinfer1::DataType type = kBYTE_TYPE);

    //! \brief Allocates an `IBuffer` of the given size on the CPU.
    [[nodiscard]] static IBufferPtr cpu(std::size_t size,
                                        nvinfer1::DataType type = kBYTE_TYPE);

    //! \brief Allocates an `ITensor` of the given dimensions on the CPU.
    [[nodiscard]] static ITensorPtr cpu(nvinfer1::Dims dims,
                                        nvinfer1::DataType type = kBYTE_TYPE);

    //! \brief Allocates a pinned `IBuffer` of the given size on the CPU.
    [[nodiscard]] static IBufferPtr pinned(std::size_t size,
                                           nvinfer1::DataType type = kBYTE_TYPE);

    //! \brief Allocates a pinned `ITensor` of the given dimensions on the CPU.
    [[nodiscard]] static ITensorPtr pinned(nvinfer1::Dims dims,
                                           nvinfer1::DataType type = kBYTE_TYPE);

    //! \brief Allocates an `IBuffer` of the given size in UVM.
    [[nodiscard]] static IBufferPtr managed(std::size_t size,
                                            nvinfer1::DataType type = kBYTE_TYPE);

    //! \brief Allocates an `ITensor` of the given dimensions in UVM.
    [[nodiscard]] static ITensorPtr managed(nvinfer1::Dims dims,
                                            nvinfer1::DataType type = kBYTE_TYPE);

    //! \brief Set the contents of the given `buffer` to value.
    static void setMem(IBuffer &buffer, int32_t value,
                       const cudaStream_t &cudaStream = nullptr);

    //! \brief Copy `src` to `dst`.
    static void copy(void const *src, IBuffer &dst, MemoryType srcType,
                     const cudaStream_t &cudaStream = nullptr);

    //! \brief Copy `src` to `dst`.
    static void copy(IBuffer const &src, void *dst, MemoryType dstType,
                     const cudaStream_t &cudaStream = nullptr);

    //! \brief Copy `src` to `dst`.
    static void copy(void const *src, IBuffer &dst,
                     const cudaStream_t &cudaStream = nullptr) {
        return copy(src, dst, IBuffer::memoryType(src), cudaStream);
    }

    //! \brief Copy `src` to `dst`.
    static void copy(IBuffer const &src, void *dst,
                     const cudaStream_t &cudaStream = nullptr) {
        return copy(src, dst, IBuffer::memoryType(dst), cudaStream);
    }

    //! \brief Copy `src` to `dst`.
    static void copy(IBuffer const &src, IBuffer &dst,
                     const cudaStream_t &cudaStream = nullptr);

private:
    explicit BufferManager() = default;
    ~BufferManager() = default;
};

} // namespace cniai::tensorrt
