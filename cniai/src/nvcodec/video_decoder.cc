#include "video_decoder.h"

#include "frame_queue.h"

#include <cstring>

#include "common/assert.h"
#include "common/logging.h"
#include "nvcommon/cuda_util.h"

namespace cniai::nvcodec {

static const char *GetVideoChromaFormatString(cudaVideoChromaFormat eChromaFormat) {
    static struct {
        cudaVideoChromaFormat eChromaFormat;
        const char *name;
    } aChromaFormatName[] = {
        {cudaVideoChromaFormat_Monochrome, "YUV 400 (Monochrome)"},
        {cudaVideoChromaFormat_420, "YUV 420"},
        {cudaVideoChromaFormat_422, "YUV 422"},
        {cudaVideoChromaFormat_444, "YUV 444"},
    };

    if (eChromaFormat < sizeof(aChromaFormatName) / sizeof(aChromaFormatName[0])) {
        return aChromaFormatName[eChromaFormat].name;
    }
    return "Unknown";
}

void VideoDecoder::create(const FormatInfo &videoFormat) {
    {
        std::unique_lock<std::mutex> lock(mMtx);
        mVideoFormat = videoFormat;
    }
    const cudaVideoCodec codec = CodecToNvCodec(videoFormat.codec);
    const auto chromaFormat =
        static_cast<cudaVideoChromaFormat>(videoFormat.chromaFormat);
    if (videoFormat.nBitDepthMinus8 > 0) {
        std::ostringstream warning;
        warning << "NV12 (8 bit luma, 4 bit chroma) is currently the only supported "
                   "decoder output format. Video input is "
                << videoFormat.nBitDepthMinus8 + 8 << " bit "
                << std::string(GetVideoChromaFormatString(chromaFormat))
                << ".  Truncating luma to 8 bits";
        if (videoFormat.chromaFormat != YUV420)
            warning << " and chroma to 4 bits";
        LOG_WARN(warning.str());
    }
    const cudaVideoCreateFlags videoCreateFlags =
        (codec == cudaVideoCodec_JPEG || codec == cudaVideoCodec_MPEG2)
            ? cudaVideoCreate_PreferCUDA
            : cudaVideoCreate_PreferCUVID;

    // Validate video format.  These are the currently supported formats via NVCUVID
    bool codecSupported =
        cudaVideoCodec_MPEG1 == codec || cudaVideoCodec_MPEG2 == codec ||
        cudaVideoCodec_MPEG4 == codec || cudaVideoCodec_VC1 == codec ||
        cudaVideoCodec_H264 == codec || cudaVideoCodec_JPEG == codec ||
        cudaVideoCodec_H264_SVC == codec || cudaVideoCodec_H264_MVC == codec ||
        cudaVideoCodec_YV12 == codec || cudaVideoCodec_NV12 == codec ||
        cudaVideoCodec_YUYV == codec || cudaVideoCodec_UYVY == codec;

#if (CUDART_VERSION >= 6050)
    codecSupported |= cudaVideoCodec_HEVC == codec;
#endif
#if (CUDART_VERSION >= 7050)
    codecSupported |= cudaVideoCodec_YUV420 == codec;
#endif
#if ((CUDART_VERSION == 7050) || (CUDART_VERSION >= 9000))
    codecSupported |= cudaVideoCodec_VP8 == codec || cudaVideoCodec_VP9 == codec;
#endif
#if (CUDART_VERSION >= 9000)
    codecSupported |= cudaVideoCodec_AV1;
#endif
    CNIAI_CHECK(codecSupported);
    CNIAI_CHECK(cudaVideoChromaFormat_Monochrome == chromaFormat ||
                cudaVideoChromaFormat_420 == chromaFormat ||
                cudaVideoChromaFormat_422 == chromaFormat ||
                cudaVideoChromaFormat_444 == chromaFormat);

    // Check video format is supported by GPU's hardware video decoder
    CUVIDDECODECAPS decodeCaps = {};
    decodeCaps.eCodecType = codec;
    decodeCaps.eChromaFormat = chromaFormat;
    decodeCaps.nBitDepthMinus8 = videoFormat.nBitDepthMinus8;
    CU_CHECK(cuCtxPushCurrent(mCUcontext));
    CU_CHECK(cuvidGetDecoderCaps(&decodeCaps));
    CU_CHECK(cuCtxPopCurrent(nullptr));
    if (!(decodeCaps.bIsSupported &&
          (decodeCaps.nOutputFormatMask & (1 << cudaVideoSurfaceFormat_NV12)))) {
        CNIAI_THROW("Video source is not supported by hardware video decoder refer to "
                    "Nvidia's GPU Support Matrix to confirm your GPU supports hardware "
                    "decoding of the video source's codec.");
    }

    if (decodeCaps.nCounterBitDepth != 32) {
        std::ostringstream error;
        error << "Luma histogram output disabled due to current device using "
              << decodeCaps.nCounterBitDepth
              << " bit bins. Histogram output only supports 32 bit bins.";
        CNIAI_THROW(error.str());
    } else {
        mVideoFormat.nCounterBitDepth = decodeCaps.nCounterBitDepth;
        mVideoFormat.nMaxHistogramBins = decodeCaps.nMaxHistogramBins;
    }

    CNIAI_CHECK(videoFormat.ulWidth >= decodeCaps.nMinWidth &&
                videoFormat.ulHeight >= decodeCaps.nMinHeight &&
                videoFormat.ulWidth <= decodeCaps.nMaxWidth &&
                videoFormat.ulHeight <= decodeCaps.nMaxHeight);

    CNIAI_CHECK((videoFormat.width >> 4) * (videoFormat.height >> 4) <=
                decodeCaps.nMaxMBCount);

    // Create video decoder
    CUVIDDECODECREATEINFO videoDecodeCreateInfo = {};
#if (CUDART_VERSION >= 9000)
    //    createInfo_.enableHistogram = videoFormat.enableHistogram;
    videoDecodeCreateInfo.bitDepthMinus8 = videoFormat.nBitDepthMinus8;
    videoDecodeCreateInfo.ulMaxWidth = videoFormat.ulMaxWidth;
    videoDecodeCreateInfo.ulMaxHeight = videoFormat.ulMaxHeight;
#endif
    videoDecodeCreateInfo.CodecType = codec;
    videoDecodeCreateInfo.ulWidth = videoFormat.ulWidth;
    videoDecodeCreateInfo.ulHeight = videoFormat.ulHeight;
    videoDecodeCreateInfo.ulNumDecodeSurfaces = videoFormat.ulNumDecodeSurfaces;
    videoDecodeCreateInfo.ChromaFormat = chromaFormat;
    videoDecodeCreateInfo.OutputFormat = cudaVideoSurfaceFormat_NV12;
    videoDecodeCreateInfo.DeinterlaceMode =
        static_cast<cudaVideoDeinterlaceMode>(videoFormat.deinterlaceMode);
    videoDecodeCreateInfo.ulTargetWidth = videoFormat.width;
    videoDecodeCreateInfo.ulTargetHeight = videoFormat.height;
    videoDecodeCreateInfo.display_area.left = videoFormat.displayArea.x;
    videoDecodeCreateInfo.display_area.right =
        videoFormat.displayArea.x + videoFormat.displayArea.width;
    videoDecodeCreateInfo.display_area.top = videoFormat.displayArea.y;
    videoDecodeCreateInfo.display_area.bottom =
        videoFormat.displayArea.y + videoFormat.displayArea.height;
    videoDecodeCreateInfo.target_rect.left = videoFormat.targetRoi.x;
    videoDecodeCreateInfo.target_rect.right =
        videoFormat.targetRoi.x + videoFormat.targetRoi.width;
    videoDecodeCreateInfo.target_rect.top = videoFormat.targetRoi.y;
    videoDecodeCreateInfo.target_rect.bottom =
        videoFormat.targetRoi.y + videoFormat.targetRoi.height;
    videoDecodeCreateInfo.ulNumOutputSurfaces = 2;
    videoDecodeCreateInfo.ulCreationFlags = videoCreateFlags;
    videoDecodeCreateInfo.vidLock = mCUvideoctxlock;
    CU_CHECK(cuCtxPushCurrent(mCUcontext));
    {
        std::unique_lock<std::mutex> lock(mMtx);
        CU_CHECK(cuvidCreateDecoder(&mDecoder, &videoDecodeCreateInfo));
    }
    CU_CHECK(cuCtxPopCurrent(nullptr));
}

int VideoDecoder::reconfigure(const FormatInfo &videoFormat) {
    if (videoFormat.nBitDepthMinus8 != mVideoFormat.nBitDepthMinus8 ||
        videoFormat.nBitDepthChromaMinus8 != mVideoFormat.nBitDepthChromaMinus8) {
        CNIAI_THROW("Reconfigure Not supported for bit depth change");
    }

    if (videoFormat.chromaFormat != mVideoFormat.chromaFormat) {
        CNIAI_THROW("Reconfigure Not supported for chroma format change");
    }

    const bool decodeResChange = !(videoFormat.ulWidth == mVideoFormat.ulWidth &&
                                   videoFormat.ulHeight == mVideoFormat.ulHeight);

    //    if ((videoFormat.ulWidth > mVideoFormat.ulMaxWidth) ||
    //        (videoFormat.ulHeight > mVideoFormat.ulMaxHeight)) {
    //        // For VP9, let driver  handle the change if new width/height >
    //        maxwidth/maxheight if (videoFormat.codec != Codec::VP9) {
    //            CNIAI_THROW(
    //                "Reconfigure Not supported when width/height > maxwidth/maxheight");
    //        }
    //    }

    if (!decodeResChange)
        return 1;

    {
        std::unique_lock<std::mutex> lock(mMtx);
        mVideoFormat.ulNumDecodeSurfaces = videoFormat.ulNumDecodeSurfaces;
        mVideoFormat.ulWidth = videoFormat.ulWidth;
        mVideoFormat.ulHeight = videoFormat.ulHeight;
        mVideoFormat.targetRoi = videoFormat.targetRoi;
    }

    CUVIDRECONFIGUREDECODERINFO reconfigParams = {0};
    reconfigParams.ulWidth = mVideoFormat.ulWidth;
    reconfigParams.ulHeight = mVideoFormat.ulHeight;
    reconfigParams.display_area.left = mVideoFormat.displayArea.x;
    reconfigParams.display_area.right =
        mVideoFormat.displayArea.x + mVideoFormat.displayArea.width;
    reconfigParams.display_area.top = mVideoFormat.displayArea.y;
    reconfigParams.display_area.bottom =
        mVideoFormat.displayArea.y + mVideoFormat.displayArea.height;
    reconfigParams.ulTargetWidth = mVideoFormat.width;
    reconfigParams.ulTargetHeight = mVideoFormat.height;
    reconfigParams.target_rect.left = mVideoFormat.targetRoi.x;
    reconfigParams.target_rect.right =
        mVideoFormat.targetRoi.x + mVideoFormat.targetRoi.width;
    reconfigParams.target_rect.top = mVideoFormat.targetRoi.y;
    reconfigParams.target_rect.bottom =
        mVideoFormat.targetRoi.y + mVideoFormat.targetRoi.height;
    reconfigParams.ulNumDecodeSurfaces = mVideoFormat.ulNumDecodeSurfaces;

    CU_CHECK(cuCtxPushCurrent(mCUcontext));
    CU_CHECK(cuvidReconfigureDecoder(mDecoder, &reconfigParams));
    CU_CHECK(cuCtxPopCurrent(nullptr));
    LOG_INFO("Reconfiguring Decoder");
    return mVideoFormat.ulNumDecodeSurfaces;
}

void VideoDecoder::release() {
    CU_CHECK(cuCtxPushCurrent(mCUcontext));
    if (mDecoder) {
        cuvidDestroyDecoder(mDecoder);
        mDecoder = nullptr;
    }
    CU_CHECK(cuCtxPopCurrent(nullptr));
}

} // namespace cniai::nvcodec