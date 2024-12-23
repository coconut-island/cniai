#pragma once

#include <algorithm>
#include <cstring>
#include <limits>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif
#include <libavcodec/avcodec.h>
#include <libavcodec/bsf.h>
#include <libavdevice/avdevice.h>
#include <libswscale/swscale.h>
#ifdef __cplusplus
}
#endif

namespace cniai::nvcodec {

class FFmpegDemuxer {
public:
    explicit FFmpegDemuxer();
    ~FFmpegDemuxer();

    bool open(const std::string &filename);
    bool retrieve(unsigned char **data, int *size);
    bool getExtractData(unsigned char **data, int *size);

    int64_t getTotalFrames() const;
    double getDurationSec() const;
    double getFps() const;
    int64_t getBitrate() const;

    double r2d(AVRational r) const;
    int64_t dtsToFrameNumber(int64_t dts) const;
    double dtsToSec(int64_t dts) const;

    AVCodecID getCodecId();
    AVPixelFormat getPixelFormat();

    int getFrameWidth();
    int getFrameHeight();

    bool getLastPacketContainsKeyFrame();

private:
    void destroy();
    bool processRawPacket();

private:
    std::string mFilename;
    AVFormatContext *mAvFormatContext = nullptr;
    int mVideoStream = -1;
    AVStream *mAvSteam = nullptr;
    AVPacket mPacket{};
    AVPacket mPacketFiltered{};
    double mEpsZero = 0.000025;
    AVDictionary *mAVDictionary = nullptr;
    AVBSFContext *mAVBSFContext = nullptr;
};

} // namespace cniai::nvcodec