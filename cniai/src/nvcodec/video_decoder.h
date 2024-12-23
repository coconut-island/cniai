#pragma once

#include <mutex>

#include <nvcuvid.h>

#include "nvcodec.h"

#include "nvcommon/cuda_util.h"

namespace cniai::nvcodec {

inline cudaVideoCodec CodecToNvCodec(Codec codec) {
    switch (codec) {
    case H264:
        return cudaVideoCodec::cudaVideoCodec_H264;
    case HEVC:
        return cudaVideoCodec::cudaVideoCodec_HEVC;
    default:
        break;
    }
    CNIAI_THROW("Unknown codec : {}", (int)codec);
}

inline Codec NvCodecToCodec(cudaVideoCodec codec) {
    switch (codec) {
    case cudaVideoCodec::cudaVideoCodec_H264:
        return H264;
    case cudaVideoCodec::cudaVideoCodec_HEVC:
        return HEVC;
    default:
        break;
    }
    CNIAI_THROW("Unknown codec : {}", (int)codec);
}

class VideoDecoder {
public:
    VideoDecoder(const Codec &codec, const int minNumDecodeSurfaces, Size targetSz,
                 Rect srcRoi, Rect targetRoi, CUcontext ctx, CUvideoctxlock lock)
        : mCUcontext(ctx), mCUvideoctxlock(lock), mDecoder(nullptr) {
        mVideoFormat.codec = codec;
        mVideoFormat.ulNumDecodeSurfaces = minNumDecodeSurfaces;
        // alignment enforced by nvcuvid, likely due to chroma subsampling
        mVideoFormat.targetSz.width = targetSz.width - targetSz.width % 2;
        mVideoFormat.targetSz.height = targetSz.height - targetSz.height % 2;
        mVideoFormat.srcRoi.x = srcRoi.x - srcRoi.x % 4;
        mVideoFormat.srcRoi.width = srcRoi.width - srcRoi.width % 4;
        mVideoFormat.srcRoi.y = srcRoi.y - srcRoi.y % 2;
        mVideoFormat.srcRoi.height = srcRoi.height - srcRoi.height % 2;
        mVideoFormat.targetRoi.x = targetRoi.x - targetRoi.x % 4;
        mVideoFormat.targetRoi.width = targetRoi.width - targetRoi.width % 4;
        mVideoFormat.targetRoi.y = targetRoi.y - targetRoi.y % 2;
        mVideoFormat.targetRoi.height = targetRoi.height - targetRoi.height % 2;
    }

    ~VideoDecoder() { release(); }

    void create(const FormatInfo &videoFormat);
    int reconfigure(const FormatInfo &videoFormat);
    void release();

    bool inited() {
        std::unique_lock<std::mutex> lock(mMtx);
        return mDecoder;
    }

    // Get the codec-type currently used.
    cudaVideoCodec codec() const { return CodecToNvCodec(mVideoFormat.codec); }

    int nDecodeSurfaces() const { return mVideoFormat.ulNumDecodeSurfaces; }

    Size getTargetSz() const { return mVideoFormat.targetSz; }

    Rect getSrcRoi() const { return mVideoFormat.srcRoi; }

    Rect getTargetRoi() const { return mVideoFormat.targetRoi; }

    unsigned long frameWidth() const { return mVideoFormat.ulWidth; }

    unsigned long frameHeight() const { return mVideoFormat.ulHeight; }

    FormatInfo format() {
        std::unique_lock<std::mutex> lock(mMtx);
        return mVideoFormat;
    }

    unsigned long targetWidth() { return mVideoFormat.width; }

    unsigned long targetHeight() { return mVideoFormat.height; }

    cudaVideoChromaFormat chromaFormat() const {
        return static_cast<cudaVideoChromaFormat>(mVideoFormat.chromaFormat);
    }

    int nBitDepthMinus8() const { return mVideoFormat.nBitDepthMinus8; }

    bool decodePicture(CUVIDPICPARAMS *picParams) {
        return cuvidDecodePicture(mDecoder, picParams) == CUDA_SUCCESS;
    }

    CniFrame mapFrame(int picIdx, CUVIDPROCPARAMS &videoProcParams) {
        CUdeviceptr ptr;
        unsigned int pitch;

        CU_CHECK(cuvidMapVideoFrame(mDecoder, picIdx, &ptr, &pitch, &videoProcParams));

        return CniFrame(targetWidth(), targetHeight(), CniFrame::Format::NV12,
                        (void *)ptr, pitch);
    }

    void unmapFrame(CniFrame &frame) {
        CU_CHECK(cuvidUnmapVideoFrame(mDecoder, (CUdeviceptr)frame.data()));
        frame.release();
    }

private:
    CUcontext mCUcontext = nullptr;
    CUvideoctxlock mCUvideoctxlock;
    CUvideodecoder mDecoder = nullptr;
    FormatInfo mVideoFormat = {};
    std::mutex mMtx;
};

} // namespace cniai::nvcodec