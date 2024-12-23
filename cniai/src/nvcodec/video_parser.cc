#include "video_parser.h"

#include "common/logging.h"
#include "nvcommon/cuda_util.h"

namespace cniai::nvcodec {

VideoParser::VideoParser(VideoDecoder *videoDecoder, FrameQueue *frameQueue,
                         bool allowFrameDrop)
    : mVideoDecoder(videoDecoder), mFrameQueue(frameQueue),
      mAllowFrameDrop(allowFrameDrop) {
    CUVIDPARSERPARAMS params;
    std::memset(&params, 0, sizeof(CUVIDPARSERPARAMS));

    params.CodecType = videoDecoder->codec();
    params.ulMaxNumDecodeSurfaces = 1;
    params.ulMaxDisplayDelay = 1; // this flag is needed so the parser will push frames
                                  // out to the decoder as quickly as it can
    params.pUserData = this;
    params.pfnSequenceCallback =
        HandleVideoSequence; // Called before decoding frames and/or whenever there is a
                             // format change
    params.pfnDecodePicture = HandlePictureDecode; // Called when a picture is ready to be
                                                   // decoded (decode order)
    params.pfnDisplayPicture = HandlePictureDisplay; // Called whenever a picture is ready
                                                     // to be displayed (display order)

    CU_CHECK(cuvidCreateVideoParser(&mParser, &params));
}

bool VideoParser::parseVideoData(const unsigned char *data, size_t size,
                                 bool endOfStream) {
    CUVIDSOURCEDATAPACKET packet;
    std::memset(&packet, 0, sizeof(CUVIDSOURCEDATAPACKET));

    packet.flags = CUVID_PKT_TIMESTAMP;
    if (endOfStream)
        packet.flags |= CUVID_PKT_ENDOFSTREAM;

    packet.payload_size = static_cast<unsigned long>(size);
    packet.payload = data;

    CUresult retVal;
    try {
        retVal = cuvidParseVideoData(mParser, &packet);
    } catch (const std::exception &e) {
        LOG_ERROR(e.what());
        mHasError = true;
        mFrameQueue->endDecode();
        return false;
    }

    if (retVal != CUDA_SUCCESS) {
        mHasError = true;
        mFrameQueue->endDecode();
        return false;
    }

    if (endOfStream)
        mFrameQueue->endDecode();

    return !mFrameQueue->isEndOfDecode();
}

int CUDAAPI VideoParser::HandleVideoSequence(void *userData, CUVIDEOFORMAT *format) {
    auto *thiz = static_cast<VideoParser *>(userData);

    FormatInfo newFormat;
    CNIAI_CHECK(format->video_signal_description.video_full_range_flag == 0);
    newFormat.codec = NvCodecToCodec(format->codec);
    newFormat.chromaFormat = static_cast<ChromaFormat>(format->chroma_format);
    newFormat.nBitDepthMinus8 = format->bit_depth_luma_minus8;
    newFormat.nBitDepthChromaMinus8 = format->bit_depth_chroma_minus8;
    newFormat.ulWidth = format->coded_width;
    newFormat.ulHeight = format->coded_height;
    newFormat.fps =
        format->frame_rate.numerator / static_cast<float>(format->frame_rate.denominator);
    newFormat.targetSz = thiz->mVideoDecoder->getTargetSz();
    newFormat.srcRoi = thiz->mVideoDecoder->getSrcRoi();
    if (newFormat.srcRoi.empty()) {
        newFormat.displayArea =
            Rect(format->display_area.left, format->display_area.top,
                 format->display_area.right - format->display_area.left,
                 format->display_area.bottom - format->display_area.top);
        if (newFormat.targetSz.empty())
            newFormat.targetSz =
                Size((format->display_area.right - format->display_area.left),
                     (format->display_area.bottom - format->display_area.top));
    } else
        newFormat.displayArea = newFormat.srcRoi;
    newFormat.width =
        newFormat.targetSz.width ? newFormat.targetSz.width : format->coded_width;
    newFormat.height =
        newFormat.targetSz.height ? newFormat.targetSz.height : format->coded_height;
    newFormat.targetRoi = thiz->mVideoDecoder->getTargetRoi();
    newFormat.ulNumDecodeSurfaces =
        std::min(!thiz->mAllowFrameDrop
                     ? std::max(thiz->mVideoDecoder->nDecodeSurfaces(),
                                static_cast<int>(format->min_num_decode_surfaces))
                     : format->min_num_decode_surfaces * 2,
                 32);
    if (format->progressive_sequence)
        newFormat.deinterlaceMode = Weave;
    else
        newFormat.deinterlaceMode = Adaptive;
    int maxW = 0, maxH = 0;
    // AV1 has max width/height of sequence in sequence header
    if (format->codec == cudaVideoCodec_AV1 && format->seqhdr_data_length > 0) {
        auto *vidFormatEx = (CUVIDEOFORMATEX *)format;
        maxW = vidFormatEx->av1.max_width;
        maxH = vidFormatEx->av1.max_height;
    }
    if (maxW < (int)format->coded_width)
        maxW = format->coded_width;
    if (maxH < (int)format->coded_height)
        maxH = format->coded_height;
    newFormat.ulMaxWidth = maxW;
    newFormat.ulMaxHeight = maxH;
    //    newFormat.enableHistogram = thiz->videoDecoder_->enableHistogram();

    thiz->mFrameQueue->waitUntilEmpty();
    int retVal = newFormat.ulNumDecodeSurfaces;
    if (thiz->mVideoDecoder->inited()) {
        retVal = thiz->mVideoDecoder->reconfigure(newFormat);
        if (retVal > 1 && newFormat.ulNumDecodeSurfaces != thiz->mFrameQueue->getMaxSz())
            thiz->mFrameQueue->resize(newFormat.ulNumDecodeSurfaces);
    } else {
        thiz->mFrameQueue->init(newFormat.ulNumDecodeSurfaces);
        thiz->mVideoDecoder->create(newFormat);
    }
    return retVal;
}

int CUDAAPI VideoParser::HandlePictureDecode(void *userData, CUVIDPICPARAMS *picParams) {
    auto *thiz = static_cast<VideoParser *>(userData);

    bool isFrameAvailable = thiz->mFrameQueue->waitUntilFrameAvailable(
        picParams->CurrPicIdx, thiz->mAllowFrameDrop);
    if (!isFrameAvailable)
        return false;

    if (!thiz->mVideoDecoder->decodePicture(picParams)) {
        LOG_ERROR("Decoding failed!");
        thiz->mHasError = true;
        return false;
    }

    return true;
}

int CUDAAPI VideoParser::HandlePictureDisplay(void *userData,
                                              CUVIDPARSERDISPINFO *picParams) {
    auto *thiz = static_cast<VideoParser *>(userData);
    thiz->mFrameQueue->enqueue(picParams);
    return true;
}

} // namespace cniai::nvcodec