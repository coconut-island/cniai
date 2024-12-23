#pragma once

#include <sstream>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#ifdef __cplusplus
}
#endif

namespace cniai::nvcodec {

static std::string avErrorToString(int av_error_code) {
    const auto bufSize = 1024U;
    char *errString = (char *)calloc(bufSize, sizeof(*errString));
    if (!errString) {
        return {};
    }

    if (0 != av_strerror(av_error_code, errString, bufSize - 1)) {
        free(errString);
        std::stringstream ss;
        ss << "Unknown error with code " << av_error_code;
        return ss.str();
    }

    std::string str(errString);
    free(errString);
    return str;
}

class FFmpegMuxer {
public:
    FFmpegMuxer(const std::string &filename, AVCodecID avCodecId, int width, int height,
                float fps);
    ~FFmpegMuxer();

    bool write(uint8_t *nalu, int naluLen, int pts = -1);

private:
    bool init(const std::string &filename, std::string extraData = "");
    void destroy();

private:
    bool mSuccessInited = false;
    std::string mFilename;
    std::string mExtractData;

    AVCodecID mCodecId = AV_CODEC_ID_NONE;
    int mWidth = 0;
    int mHeight = 0;
    float mFps = 25.0f;
    int64_t mFrameCount = 0;

    AVFormatContext *mAvformatContext = nullptr;
    AVStream *mVideoStream = nullptr;
};

} // namespace cniai::nvcodec