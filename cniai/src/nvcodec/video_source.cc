#include "video_source.h"

#include "common/assert.h"

namespace cniai::nvcodec {

bool VideoSource::parseVideoData(const unsigned char *data, size_t size,
                                 bool endOfStream) {
    return mVideoParser->parseVideoData(data, size, endOfStream);
}

FFmpegVideoSourceWrapper::FFmpegVideoSourceWrapper(FFmpegVideoSource *source)
    : mSource_(source) {
    CNIAI_CHECK(mSource_);
}

Codec FFmpegVideoSourceWrapper::getCodec() const { return mSource_->getCodec(); }

void FFmpegVideoSourceWrapper::start() {
    mHasError = false;
    mThread_ = new std::thread(readLoop, this);
}

void FFmpegVideoSourceWrapper::stop() {
    mStop = true;
    if (mThread_ != nullptr) {
        if (mThread_->joinable()) {
            mThread_->join();
        }
        delete mThread_;
        mThread_ == nullptr;
    }
}

bool FFmpegVideoSourceWrapper::isStarted() const { return !mStop; }

bool FFmpegVideoSourceWrapper::hasError() const { return mHasError; }

void FFmpegVideoSourceWrapper::readLoop(void *userData) {
    auto *thiz = static_cast<FFmpegVideoSourceWrapper *>(userData);
    for (;;) {
        unsigned char *data;
        size_t size;

        if (!thiz->mSource_->getNextPacket(&data, &size)) {
            thiz->mHasError = false;
            break;
        }
        //        bool containsKeyFrame = thiz->source_->lastPacketContainsKeyFrame();

        uint8_t *extraData = nullptr;
        int extraDataSize;
        thiz->mSource_->getExtraData(extraData, &extraDataSize);
        if (!extraDataSize)
            thiz->setExtraData(extraData, extraDataSize);

        if (!thiz->parseVideoData(data, size)) {
            thiz->mHasError = true;
            break;
        }

        if (thiz->mStop)
            break;
    }

    if (!thiz->mHasError)
        thiz->parseVideoData(nullptr, 0, true);
}

} // namespace cniai::nvcodec