#pragma once

#include "ffmpeg_video_source.h"
#include "nvcodec.h"
#include "video_parser.h"

#include <cstring>
#include <thread>

namespace cniai::nvcodec {

class VideoSource {
public:
    virtual ~VideoSource() = default;

    virtual Codec getCodec() const = 0;

    virtual void start() = 0;
    virtual void stop() = 0;
    virtual bool isStarted() const = 0;
    virtual bool hasError() const = 0;

    void setVideoParser(VideoParser *videoParser) { mVideoParser = videoParser; }

    void setExtraData(uint8_t *extraData, int extraDataSize) {
        std::unique_lock<std::mutex> lock(mMtx);
        mExtraData = extraData;
        mExtraDataSize = extraDataSize;
    }

    void getExtraData(uint8_t *extraData, int *extraDataSize) {
        std::unique_lock<std::mutex> lock(mMtx);
        extraData = (uint8_t *)malloc(mExtraDataSize);
        std::memcpy(extraData, mExtraData, mExtraDataSize);
        *extraDataSize = mExtraDataSize;
    }

protected:
    bool parseVideoData(const unsigned char *data, size_t size, bool endOfStream = false);

private:
    VideoParser *mVideoParser = nullptr;
    uint8_t *mExtraData;
    int mExtraDataSize;
    std::mutex mMtx;
};

class FFmpegVideoSourceWrapper : public VideoSource {
public:
    explicit FFmpegVideoSourceWrapper(FFmpegVideoSource *source);

    Codec getCodec() const override;
    void start() override;
    void stop() override;
    bool isStarted() const override;
    bool hasError() const override;

private:
    static void readLoop(void *userData);
    FFmpegVideoSource *mSource_ = nullptr;
    std::thread *mThread_ = nullptr;
    volatile bool mStop = false;
    volatile bool mHasError = false;
};

} // namespace cniai::nvcodec