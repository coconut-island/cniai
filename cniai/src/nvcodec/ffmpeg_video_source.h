#pragma once

#include "nvcodec.h"

#include "ffmpeg_demuxer.h"

namespace cniai::nvcodec {

class FFmpegVideoSource {
public:
    explicit FFmpegVideoSource(const std::string &fname);
    ~FFmpegVideoSource();

    bool getNextPacket(unsigned char **data, size_t *size);

    //    bool lastPacketContainsKeyFrame() const;

    //    FormatInfo format() const;
    //
    //    void updateFormat(const FormatInfo &videoFormat);

    void getExtraData(uint8_t *_extraData, int *_extraDataSize) const {
        _extraData = extraData;
        *_extraDataSize = extraDataSize;
    }

    Codec getCodec();

private:
    FFmpegDemuxer *ffmpegDemuxer;
    //    FormatInfo format_;
    Codec mCodec;
    ChromaFormat mChromaFormat;
    int mBitDepthMinus8{};
    uint8_t *dataWithHeader = nullptr;
    uint8_t *extraData;
    int extraDataSize;
    int iFrame = 0;
};

} // namespace cniai::nvcodec