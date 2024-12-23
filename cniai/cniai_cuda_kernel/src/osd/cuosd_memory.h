#pragma once

#include <cstdio>
#include <cuda_runtime.h>

namespace cniai_cuda_kernel::cuosd {

#define checkRuntime(call) check_runtime(call, #call, __LINE__, __FILE__)

#define CUOSD_PRINT_E(f_, ...)                                                           \
    fprintf(stderr, "[cuOSD Error] at %s:%d : " f_, (const char *)__FILE__, __LINE__,    \
            ##__VA_ARGS__)

#define CUOSD_PRINT_W(f_, ...)                                                           \
    printf("[cuOSD Warning] at %s:%d : " f_, (const char *)__FILE__, __LINE__,           \
           ##__VA_ARGS__)

static bool inline check_runtime(cudaError_t e, const char *call, int line,
                                 const char *file) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime error %s # %s, code = %s [ %d ] in file %s:%d\n",
                call, cudaGetErrorString(e), cudaGetErrorName(e), e, file, line);
        return false;
    }
    return true;
}

template <typename T>
class Memory {
public:
    T *host() const { return host_; }

    T *device() const { return device_; }

    size_t size() const { return size_; }

    size_t bytes() const { return size_ * sizeof(T); }

    virtual ~Memory() { free_memory(); }

    void copy_host_to_device(cudaStream_t stream = nullptr) {
        checkRuntime(
            cudaMemcpyAsync(device_, host_, bytes(), cudaMemcpyHostToDevice, stream));
    }

    void copy_device_to_host(cudaStream_t stream = nullptr) {
        checkRuntime(
            cudaMemcpyAsync(host_, device_, bytes(), cudaMemcpyDeviceToHost, stream));
    }

    void alloc_or_resize_to(size_t size) {
        if (capacity_ < size) {
            free_memory();

            checkRuntime(cudaMallocHost(&host_, size * sizeof(T)));
            checkRuntime(cudaMalloc(&device_, size * sizeof(T)));
            capacity_ = size;
        }
        size_ = size;
    }

    void free_memory() {
        if (host_ || device_) {
            checkRuntime(cudaFreeHost(host_));
            checkRuntime(cudaFree(device_));
            host_ = nullptr;
            device_ = nullptr;
            capacity_ = 0;
            size_ = 0;
        }
    }

private:
    T *host_ = nullptr;
    T *device_ = nullptr;
    size_t size_ = 0;
    size_t capacity_ = 0;
};

} // namespace cniai_cuda_kernel::cuosd