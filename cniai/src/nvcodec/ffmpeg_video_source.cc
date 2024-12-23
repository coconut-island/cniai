#include "ffmpeg_video_source.h"

#include "common/assert.h"
#include "common/logging.h"
#include "common/str_util.h"

namespace cniai::nvcodec {

static Codec AVCodecIDToCodec(AVCodecID codec) {
    switch (codec) {
    case AV_CODEC_ID_H264:
        return H264;
    case AV_CODEC_ID_HEVC:
        return HEVC;
    default:
        break;
    }
    CNIAI_THROW("Unknown codec : {}", (int)codec);
}

static void AVPixelFormatToChromaFormat(const AVPixelFormat pixelFormat,
                                        ChromaFormat &chromaFormat,
                                        int &nBitDepthMinus8) {
    switch (pixelFormat) {
    case AV_PIX_FMT_YUV420P:
        chromaFormat = YUV420;
        nBitDepthMinus8 = 0;
        break;
    default:
        LOG_WARN("ChromaFormat not recognized: {}. Assuming I420", (int)pixelFormat);
        chromaFormat = YUV420;
        nBitDepthMinus8 = 0;
        break;
    }
}

static int StartCodeLen(const unsigned char *data, const size_t sz) {
    if (sz >= 3 && data[0] == 0 && data[1] == 0 && data[2] == 1)
        return 3;
    else if (sz >= 4 && data[0] == 0 && data[1] == 0 && data[2] == 0 && data[3] == 1)
        return 4;
    else
        return 0;
}

bool ParamSetsExist(unsigned char *parameterSets, const int szParameterSets,
                    unsigned char *data, const size_t szData) {
    const int paramSetStartCodeLen = StartCodeLen(parameterSets, szParameterSets);
    const int packetStartCodeLen = StartCodeLen(data, szData);
    // weak test to see if the parameter set has already been included in the RTP stream
    return paramSetStartCodeLen != 0 && packetStartCodeLen != 0 &&
           parameterSets[paramSetStartCodeLen] == data[packetStartCodeLen];
}

FFmpegVideoSource::FFmpegVideoSource(const std::string &filename) {

    ffmpegDemuxer = new FFmpegDemuxer();
    if (!ffmpegDemuxer->open(filename)) {
        CNIAI_THROW("Unsupported video source");
    }

    unsigned char *tmpExtraData;
    int tmpExtraDataSize;
    ffmpegDemuxer->getExtractData(&tmpExtraData, &tmpExtraDataSize);
    if (tmpExtraDataSize) {
        extraData = (uint8_t *)malloc(tmpExtraDataSize);
        memcpy(extraData, tmpExtraData, tmpExtraDataSize);
        extraDataSize = tmpExtraDataSize;
    }

    AVCodecID codec = ffmpegDemuxer->getCodecId();
    mCodec = AVCodecIDToCodec(codec);
    AVPixelFormat pixelFormat = ffmpegDemuxer->getPixelFormat();
    AVPixelFormatToChromaFormat(pixelFormat, mChromaFormat, mBitDepthMinus8);
}

FFmpegVideoSource::~FFmpegVideoSource() {
    if (ffmpegDemuxer) {
        delete ffmpegDemuxer;
        ffmpegDemuxer = nullptr;
    }

    if (extraData) {
        free(extraData);
        extraData = nullptr;
    }

    if (dataWithHeader) {
        free(dataWithHeader);
        dataWithHeader = nullptr;
    }
}

Codec FFmpegVideoSource::getCodec() { return mCodec; }

// FormatInfo FFmpegVideoSource::format() const { return format_; }
//
// void FFmpegVideoSource::updateFormat(const FormatInfo &videoFormat) {
//     format_ = videoFormat;
// }

bool FFmpegVideoSource::getNextPacket(unsigned char **data, size_t *size) {
    int temSize;
    if (!ffmpegDemuxer->retrieve(data, &temSize)) {
        return false;
    }
    *size = temSize;

    if (iFrame++ == 0 && extraDataSize) {
        if (((mCodec == Codec::H264 || mCodec == Codec::HEVC) &&
             !ParamSetsExist(extraData, extraDataSize, *data, *size))) {
            const size_t nBytesToTrimFromData = 0;
            const size_t newSz = extraDataSize + *size - nBytesToTrimFromData;
            dataWithHeader = (uint8_t *)malloc(newSz);
            memcpy(dataWithHeader, extraData, extraDataSize);
            memcpy(dataWithHeader + extraDataSize, (*data) + nBytesToTrimFromData,
                   *size - nBytesToTrimFromData);
            *data = dataWithHeader;
            *size = newSz;
        }
    }

    return *size != 0;
}

} // namespace cniai::nvcodec