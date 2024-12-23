#pragma once

#include "nvcommon/cuda_util.h"

#include <cuda_runtime.h>

#include <memory>

namespace cniai::tensorrt {

class CudaEvent {
public:
    using pointer = cudaEvent_t;

    explicit CudaEvent(unsigned int flags = cudaEventDisableTiming) {
        pointer event;
        CUDA_CHECK(cudaEventCreate(&event, flags));
        bool constexpr ownsEvent{true};
        mEvent = EventPtr{event, Deleter{ownsEvent}};
    }

    explicit CudaEvent(pointer event, bool ownsEvent = true) {
        assert(event != nullptr && "event is nullptr");
        mEvent = EventPtr{event, Deleter{ownsEvent}};
    }

    pointer get() const { return mEvent.get(); }

    void synchronize() const { CUDA_CHECK(cudaEventSynchronize(get())); }

private:
    class Deleter {
    public:
        explicit Deleter(bool ownsEvent) : mOwnsEvent{ownsEvent} {}

        explicit Deleter() : Deleter{true} {}

        constexpr void operator()(pointer event) const {
            if (mOwnsEvent && event != nullptr) {
                CUDA_CHECK(cudaEventDestroy(event));
            }
        }

    private:
        bool mOwnsEvent;
    };

    using element_type = std::remove_pointer_t<pointer>;
    using EventPtr = std::unique_ptr<element_type, Deleter>;

    EventPtr mEvent;
};

} // namespace cniai::tensorrt
