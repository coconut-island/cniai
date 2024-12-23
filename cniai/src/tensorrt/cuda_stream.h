#pragma once

#include "nvcommon/cuda_util.h"
#include "tensorrt/cuda_event.h"

#include <cuda_runtime_api.h>

#include <iostream>
#include <memory>

namespace cniai::tensorrt {

class CudaStream {
public:
    explicit CudaStream(unsigned int flags = cudaStreamNonBlocking, int priority = 0)
        : mDevice{cniai::cuda_util::getDevice()} {
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreateWithPriority(&stream, flags, priority));
        bool constexpr ownsStream{true};
        mStream = StreamPtr{stream, Deleter{ownsStream}};
    }

    explicit CudaStream(cudaStream_t stream, int device, bool ownsStream = true)
        : mDevice{device} {
        assert(stream != nullptr && "stream is nullptr");
        mStream = StreamPtr{stream, Deleter{ownsStream}};
    }

    explicit CudaStream(cudaStream_t stream)
        : CudaStream{stream, cniai::cuda_util::getDevice(), false} {}

    int getDevice() const { return mDevice; }

    cudaStream_t get() const { return mStream.get(); }

    void synchronize() const { CUDA_CHECK(cudaStreamSynchronize(get())); }

    void record(CudaEvent::pointer event) const {
        CUDA_CHECK(cudaEventRecord(event, get()));
    }

    void record(CudaEvent const &event) const { record(event.get()); }

    void wait(CudaEvent::pointer event) const {
        CUDA_CHECK(cudaStreamWaitEvent(get(), event));
    }

    void wait(CudaEvent const &event) const { wait(event.get()); }

private:
    class Deleter {
    public:
        explicit Deleter(bool ownsStream) : mOwnsStream{ownsStream} {}

        explicit Deleter() : Deleter{true} {}

        constexpr void operator()(cudaStream_t stream) const {
            if (mOwnsStream && stream != nullptr) {
                CUDA_CHECK(cudaStreamDestroy(stream));
            }
        }

    private:
        bool mOwnsStream;
    };

    using StreamPtr = std::unique_ptr<std::remove_pointer_t<cudaStream_t>, Deleter>;

    StreamPtr mStream;
    int mDevice{-1};
};

} // namespace cniai::tensorrt
