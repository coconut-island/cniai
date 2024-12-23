#pragma once

#include <nvjpeg.h>

#include "nvcommon/frame.h"

namespace cniai::nvjpeg {

class NvjpegDecoder {

public:
    explicit NvjpegDecoder();
    ~NvjpegDecoder();

    bool
    decode(const void *srcJpeg, size_t length, CniFrame &outFrame,
           nvjpegOutputFormat_t outputFormat = nvjpegOutputFormat_t::NVJPEG_OUTPUT_BGRI);

private:
    nvjpegJpegState_t mJpegState{};
    nvjpegHandle_t mHandle{};
    cudaStream_t mStream{};
};

} // namespace cniai::nvjpeg