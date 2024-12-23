#include "nvjpeg.h"

#include "common/logging.h"
#include "nvcommon/cuda_util.h"
#include "nvjpeg_util.h"

namespace cniai::nvjpeg {

NvjpegDecoder::NvjpegDecoder() {
    NVJPEG_CHECK(nvjpegCreateSimple(&mHandle));
    NVJPEG_CHECK(nvjpegJpegStateCreate(mHandle, &mJpegState));
    CUDA_CHECK(cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking));
}

NvjpegDecoder::~NvjpegDecoder() {
    CUDA_CHECK(cudaStreamDestroy(mStream));
    NVJPEG_CHECK(nvjpegJpegStateDestroy(mJpegState));
    NVJPEG_CHECK(nvjpegDestroy(mHandle));
}

bool NvjpegDecoder::decode(const void *srcJpeg, size_t length, CniFrame &outFrame,
                           nvjpegOutputFormat_t outputFormat) {
    if (outputFormat != NVJPEG_OUTPUT_BGRI && outputFormat != NVJPEG_OUTPUT_RGBI) {
        LOG_WARN("Unsupported format.");
        return false;
    }
    int channels;
    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];
    nvjpegChromaSubsampling_t subsampling;
    NVJPEG_CHECK(nvjpegGetImageInfo(mHandle, (unsigned char *)srcJpeg, length, &channels,
                                    &subsampling, widths, heights));
    int width = widths[0];
    int height = heights[0];

    CniFrame::Format frameFormat = outputFormat == NVJPEG_OUTPUT_BGRI
                                       ? CniFrame::Format::BGR
                                       : CniFrame::Format::RGB;
    outFrame.create(width, height, frameFormat, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    nvjpegImage_t nvjpegImage;
    nvjpegImage.pitch[0] = 3 * width;
    nvjpegImage.channel[0] = (unsigned char *)outFrame.data();
    NVJPEG_CHECK(nvjpegDecode(mHandle, mJpegState, (unsigned char *)srcJpeg, length,
                              outputFormat, &nvjpegImage, mStream));
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    return true;
}

} // namespace cniai::nvjpeg