#include "tensorrt/ibuffer.h"
#include "common/assert.h"
#include "tensorrt/buffers.h"

#include <cuda_runtime_api.h>

namespace cniai::tensorrt {

MemoryType IBuffer::memoryType(void const *data) {
    cudaPointerAttributes attributes{};
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, data));
    switch (attributes.type) {
    case cudaMemoryTypeHost:
        return MemoryType::kPINNED;
    case cudaMemoryTypeDevice:
        return MemoryType::kGPU;
    case cudaMemoryTypeManaged:
        return MemoryType::kUVM;
    case cudaMemoryTypeUnregistered:
        return MemoryType::kCPU;
    }

    CNIAI_THROW("Unsupported memory type");
}

char const *IBuffer::getDataTypeName() const {
    switch (getDataType()) {
    case nvinfer1::DataType::kINT64:
        return DataTypeTraits<nvinfer1::DataType::kINT64>::name;
    case nvinfer1::DataType::kINT32:
        return DataTypeTraits<nvinfer1::DataType::kINT32>::name;
    case nvinfer1::DataType::kFLOAT:
        return DataTypeTraits<nvinfer1::DataType::kFLOAT>::name;
#ifdef ENABLE_BF16
    case nvinfer1::DataType::kBF16:
        return DataTypeTraits<nvinfer1::DataType::kBF16>::name;
#endif
    case nvinfer1::DataType::kHALF:
        return DataTypeTraits<nvinfer1::DataType::kHALF>::name;
    case nvinfer1::DataType::kBOOL:
        return DataTypeTraits<nvinfer1::DataType::kBOOL>::name;
    case nvinfer1::DataType::kUINT8:
        return DataTypeTraits<nvinfer1::DataType::kUINT8>::name;
    case nvinfer1::DataType::kINT8:
        return DataTypeTraits<nvinfer1::DataType::kINT8>::name;
#ifdef ENABLE_FP8
    case nvinfer1::DataType::kFP8:
        return DataTypeTraits<nvinfer1::DataType::kFP8>::name;
#endif
    }

    CNIAI_THROW("Unknown data type");
}

char const *IBuffer::getMemoryTypeName() const {
    switch (getMemoryType()) {
    case MemoryType::kPINNED:
        return MemoryTypeString<MemoryType::kPINNED>::value;
    case MemoryType::kCPU:
        return MemoryTypeString<MemoryType::kCPU>::value;
    case MemoryType::kGPU:
        return MemoryTypeString<MemoryType::kGPU>::value;
    case MemoryType::kUVM:
        return MemoryTypeString<MemoryType::kUVM>::value;
    }
    CNIAI_THROW("Unknown memory type");
}

} // namespace cniai::tensorrt