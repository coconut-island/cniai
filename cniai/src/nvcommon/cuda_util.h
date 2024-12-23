#pragma once

#include <cassert>

#include <cuda_runtime_api.h>
#include <iostream>

namespace cniai::cuda_util {

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t _e = call;                                                           \
        if (_e != cudaSuccess) {                                                         \
            std::cerr << "CUDA error " << _e << ": " << cudaGetErrorString(_e) << " at " \
                      << __FILE__ << ":" << __LINE__ << std::endl;                       \
            assert(0);                                                                   \
        }                                                                                \
    } while (0)
#endif // CUDA_CHECK

#ifndef CU_CHECK
#define CU_CHECK(call)                                                                   \
    do {                                                                                 \
        CUresult _e = call;                                                              \
        if (_e != CUDA_SUCCESS) {                                                        \
            std::cerr << "CU error " << _e << ": " << _e << " at " << __FILE__ << ":"    \
                      << __LINE__ << std::endl;                                          \
            assert(0);                                                                   \
        }                                                                                \
    } while (0)
#endif // CU_CHECK

inline int getSMVersion() {
    int device{-1};
    CUDA_CHECK(cudaGetDevice(&device));
    int sm_major = 0;
    int sm_minor = 0;
    CUDA_CHECK(
        cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device));
    CUDA_CHECK(
        cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device));
    return sm_major * 10 + sm_minor;
}

inline int getDevice() {
    int current_dev_id = 0;
    CUDA_CHECK(cudaGetDevice(&current_dev_id));
    return current_dev_id;
}

inline int getDeviceCount() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

} // namespace cniai::cuda_util
