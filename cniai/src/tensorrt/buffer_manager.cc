#include "tensorrt/buffer_manager.h"

#include "tensorrt/buffers.h"

#include "common/str_util.h"
#include "nvcommon/cuda_util.h"

#include <cstring>
#include <cuda_runtime_api.h>
#include <memory>

namespace cniai::tensorrt {

BufferManager::IBufferPtr BufferManager::gpu(std::size_t size,
                                             const CudaStreamPtr &cudaStream,
                                             nvinfer1::DataType type) {
    if (cudaStream) {
        return std::make_unique<DeviceBuffer>(size, type, CudaAllocatorAsync{cudaStream});
    } else {
        return std::make_unique<StaticDeviceBuffer>(size, type, CudaAllocator{});
    }
}

BufferManager::ITensorPtr BufferManager::gpu(nvinfer1::Dims dims,
                                             const CudaStreamPtr &cudaStream,
                                             nvinfer1::DataType type) {
    if (cudaStream) {
        return std::make_unique<DeviceTensor>(dims, type, CudaAllocatorAsync{cudaStream});
    } else {
        return std::make_unique<StaticDeviceTensor>(dims, type, CudaAllocator{});
    }
}

BufferManager::IBufferPtr BufferManager::cpu(std::size_t size, nvinfer1::DataType type) {
    return std::make_unique<HostBuffer>(size, type);
}

BufferManager::ITensorPtr BufferManager::cpu(nvinfer1::Dims dims,
                                             nvinfer1::DataType type) {
    return std::make_unique<HostTensor>(dims, type);
}

BufferManager::IBufferPtr BufferManager::pinned(std::size_t size,
                                                nvinfer1::DataType type) {
    return std::make_unique<PinnedBuffer>(size, type);
}

BufferManager::ITensorPtr BufferManager::pinned(nvinfer1::Dims dims,
                                                nvinfer1::DataType type) {
    return std::make_unique<PinnedTensor>(dims, type);
}

BufferManager::IBufferPtr BufferManager::managed(std::size_t size,
                                                 nvinfer1::DataType type) {
    return std::make_unique<UVMBuffer>(size, type);
}

BufferManager::ITensorPtr BufferManager::managed(nvinfer1::Dims dims,
                                                 nvinfer1::DataType type) {
    return std::make_unique<UVMTensor>(dims, type);
}

void BufferManager::setMem(IBuffer &buffer, int32_t value,
                           const cudaStream_t &cudaStream) {
    if (buffer.getMemoryType() == MemoryType::kGPU) {
        if (cudaStream) {
            CUDA_CHECK(cudaMemsetAsync(buffer.data(), value, buffer.getSizeInBytes(),
                                       cudaStream));
        } else {
            CUDA_CHECK(cudaMemset(buffer.data(), value, buffer.getSizeInBytes()));
        }
    } else {
        std::memset(buffer.data(), value, buffer.getSizeInBytes());
    }
}

void BufferManager::copy(void const *src, IBuffer &dst, MemoryType srcType,
                         const cudaStream_t &cudaStream) {
    if (dst.getSizeInBytes() > 0) {
        if (srcType != MemoryType::kGPU && dst.getMemoryType() != MemoryType::kGPU) {
            std::memcpy(dst.data(), src, dst.getSizeInBytes());
        } else {
            if (cudaStream) {
                CUDA_CHECK(cudaMemcpyAsync(dst.data(), src, dst.getSizeInBytes(),
                                           cudaMemcpyDefault, cudaStream));
            } else {
                CUDA_CHECK(
                    cudaMemcpy(dst.data(), src, dst.getSizeInBytes(), cudaMemcpyDefault));
            }
        }
    }
}

void BufferManager::copy(IBuffer const &src, void *dst, MemoryType dstType,
                         const cudaStream_t &cudaStream) {
    if (src.getSizeInBytes() > 0) {
        if (src.getMemoryType() != MemoryType::kGPU && dstType != MemoryType::kGPU) {
            std::memcpy(dst, src.data(), src.getSizeInBytes());
        } else {
            if (cudaStream) {
                CUDA_CHECK(cudaMemcpyAsync(dst, src.data(), src.getSizeInBytes(),
                                           cudaMemcpyDefault, cudaStream));
            } else {
                CUDA_CHECK(
                    cudaMemcpy(dst, src.data(), src.getSizeInBytes(), cudaMemcpyDefault));
            }
        }
    }
}

void BufferManager::copy(IBuffer const &src, IBuffer &dst,
                         const cudaStream_t &cudaStream) {
    assert(src.getDataType() == dst.getDataType() &&
           cniai::str_util::fmtstr("Incompatible data types: %s != %s",
                                   src.getDataTypeName(), dst.getDataTypeName())
               .c_str());
    assert(src.getSizeInBytes() == dst.getSizeInBytes() &&
           cniai::str_util::fmtstr("Incompatible buffer sizes: %lu != %lu",
                                   src.getSizeInBytes(), dst.getSizeInBytes())
               .c_str());
    copy(src, dst.data(), dst.getMemoryType(), cudaStream);
}

} // namespace cniai::tensorrt