#include "nvcodec.h"

#include <queue>

#include <cuda_runtime_api.h>
#include <nvcuvid.h>

#include "ffmpeg_video_source.h"
#include "frame_queue.h"
#include "video_decoder.h"
#include "video_source.h"

#include "common/assert.h"
#include "common/logging.h"
#include "nvcommon/cuda_util.h"

#include <cniai_cuda_kernel/imgproc.h>

namespace cniai::nvcodec {

void cvtFromNv12(const CniFrame &decodedFrame, CniFrame &outFrame,
                 const ColorFormat colorFormat, CUstream stream) {
    if (colorFormat == ColorFormat::RGB) {
        outFrame.create(decodedFrame.getWidth(), decodedFrame.getHeight(),
                        CniFrame::Format::RGB, stream);
        cniai_cuda_kernel::imgproc::nv12ToRgb(
            (uint8_t *)decodedFrame.data(), decodedFrame.getPitch(),
            (uint8_t *)outFrame.data(), outFrame.getPitch(), decodedFrame.getWidth(),
            decodedFrame.getHeight(), false, stream);
    } else if (colorFormat == ColorFormat::BGR) {
        outFrame.create(decodedFrame.getWidth(), decodedFrame.getHeight(),
                        CniFrame::Format::BGR, stream);
        cniai_cuda_kernel::imgproc::nv12ToRgb(
            (uint8_t *)decodedFrame.data(), decodedFrame.getPitch(),
            (uint8_t *)outFrame.data(), outFrame.getPitch(), decodedFrame.getWidth(),
            decodedFrame.getHeight(), true, stream);
    } else if (colorFormat == ColorFormat::GRAY) {
        outFrame.create(decodedFrame.getWidth(), decodedFrame.getHeight(),
                        CniFrame::Format::GRAY, stream);
        CNIAI_THROW("not support");
    } else if (colorFormat == ColorFormat::NV_NV12) {
        outFrame.create(decodedFrame.getWidth(), decodedFrame.getHeight(),
                        CniFrame::Format::NV12, stream);
        cudaMemcpy2DAsync(outFrame.data(), outFrame.getPitch(), decodedFrame.data(),
                          decodedFrame.getPitch(), decodedFrame.getWidth(),
                          decodedFrame.getHeight() * 3 / 2, cudaMemcpyDeviceToDevice,
                          stream);
    }
}

class VideoReaderImpl : public VideoReader {
public:
    explicit VideoReaderImpl(int deviceId, VideoSource *source, int minNumDecodeSurfaces,
                             bool allowFrameDrop = false, Size targetSz = Size(),
                             Rect srcRoi = Rect(), Rect targetRoi = Rect());
    ~VideoReaderImpl() override;

    bool nextFrame(CniFrame &frame) override;

    Codec getCodec() override;

    int getTargetWidth() override;

    int getTargetHeight() override;

    bool setColorFormat(ColorFormat colorFormat_) override;

    int getPosFrames() const override;

private:
    bool aquireFrameInfo(std::pair<CUVIDPARSERDISPINFO, CUVIDPROCPARAMS> &frameInfo);
    void
    releaseFrameInfo(const std::pair<CUVIDPARSERDISPINFO, CUVIDPROCPARAMS> &frameInfo);
    bool internalGrab(CniFrame &frame);
    void waitForDecoderInit();

    VideoSource *mVideoSource;

    FrameQueue *mFrameQueue = nullptr;
    VideoDecoder *mVideoDecoder = nullptr;
    VideoParser *mVideoParser = nullptr;

    CUcontext mCUcontext = nullptr;
    CUstream mStream = nullptr;
    CUvideoctxlock mCUvideoctxlock;

    std::deque<std::pair<CUVIDPARSERDISPINFO, CUVIDPROCPARAMS>> mFrames;
    ColorFormat mColorFormat = ColorFormat::NV_NV12;
    static const std::string errorMsg;
    int mIFrame = 0;
};

const std::string VideoReaderImpl::errorMsg =
    "Parsing/Decoding video source failed, check GPU memory is available and GPU "
    "supports requested functionality.";

void VideoReaderImpl::waitForDecoderInit() {
    for (;;) {
        if (mVideoDecoder->inited())
            break;
        if (mVideoParser->hasError() || mFrameQueue->isEndOfDecode())
            CNIAI_THROW(errorMsg);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

VideoReaderImpl::VideoReaderImpl(int deviceId, VideoSource *source,
                                 const int minNumDecodeSurfaces,
                                 const bool allowFrameDrop, const Size targetSz,
                                 const Rect srcRoi, const Rect targetRoi)
    : mVideoSource(source), mCUvideoctxlock(nullptr) {
    CU_CHECK(cuInit(0));
    CUdevice device;
    CU_CHECK(cuDeviceGet(&device, deviceId));
    CU_CHECK(cuCtxCreate(&mCUcontext, 0, device));
    CU_CHECK(cuvidCtxLockCreate(&mCUvideoctxlock, mCUcontext));
    CU_CHECK(cuStreamCreate(&mStream, CU_STREAM_NON_BLOCKING));
    mFrameQueue = new FrameQueue();
    mVideoDecoder =
        new VideoDecoder(mVideoSource->getCodec(), minNumDecodeSurfaces, targetSz, srcRoi,
                         targetRoi, mCUcontext, mCUvideoctxlock);
    mVideoParser = new VideoParser(mVideoDecoder, mFrameQueue, allowFrameDrop);
    mVideoSource->setVideoParser(mVideoParser);
    mVideoSource->start();
    waitForDecoderInit();
}

VideoReaderImpl::~VideoReaderImpl() {
    mFrameQueue->endDecode();
    mVideoSource->stop();
    if (mVideoParser) {
        delete mVideoParser;
        mVideoParser = nullptr;
    }
    if (mVideoDecoder) {
        delete mVideoDecoder;
        mVideoDecoder = nullptr;
    }
    if (mFrameQueue) {
        delete mFrameQueue;
        mFrameQueue = nullptr;
    }
    CU_CHECK(cuStreamDestroy(mStream));
    CU_CHECK(cuCtxDestroy(mCUcontext));
}

class VideoCtxAutoLock {
public:
    explicit VideoCtxAutoLock(CUvideoctxlock lock) : mLock(lock) {
        CU_CHECK(cuvidCtxLock(mLock, 0));
    }

    ~VideoCtxAutoLock() { CU_CHECK(cuvidCtxUnlock(mLock, 0)); }

private:
    CUvideoctxlock mLock;
};

bool VideoReaderImpl::aquireFrameInfo(
    std::pair<CUVIDPARSERDISPINFO, CUVIDPROCPARAMS> &frameInfo) {
    if (mFrames.empty()) {
        CUVIDPARSERDISPINFO displayInfo;
        for (;;) {
            if (mFrameQueue->dequeue(displayInfo))
                break;

            if (mVideoParser->hasError())
                CNIAI_THROW(errorMsg);

            if (mFrameQueue->isEndOfDecode())
                return false;

            // Wait a bit
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        bool isProgressive = displayInfo.progressive_frame != 0;
        const int numFields = isProgressive ? 1 : 2 + displayInfo.repeat_first_field;

        for (int activeField = 0; activeField < numFields; ++activeField) {
            CUVIDPROCPARAMS videoProcParams;
            std::memset(&videoProcParams, 0, sizeof(CUVIDPROCPARAMS));

            videoProcParams.progressive_frame = displayInfo.progressive_frame;
            videoProcParams.second_field = activeField;
            videoProcParams.top_field_first = displayInfo.top_field_first;
            videoProcParams.unpaired_field = (numFields == 1);
            videoProcParams.output_stream = mStream;

            mFrames.emplace_back(displayInfo, videoProcParams);
        }
    } else {
        for (auto &frame : mFrames)
            frame.second.output_stream = mStream;
    }

    if (mFrames.empty())
        return false;

    frameInfo = mFrames.front();
    mFrames.pop_front();
    return true;
}

void VideoReaderImpl::releaseFrameInfo(
    const std::pair<CUVIDPARSERDISPINFO, CUVIDPROCPARAMS> &frameInfo) {
    // release the frame, so it can be re-used in decoder
    if (mFrames.empty())
        mFrameQueue->releaseFrame(frameInfo.first);
}

bool VideoReaderImpl::internalGrab(CniFrame &frame) {
    if (mVideoParser->hasError())
        CNIAI_THROW(errorMsg);

    std::pair<CUVIDPARSERDISPINFO, CUVIDPROCPARAMS> frameInfo;
    if (!aquireFrameInfo(frameInfo))
        return false;

    {
        VideoCtxAutoLock autoLock(mCUvideoctxlock);

        // map decoded video frame to CUDA surface
        CniFrame decodedFrame =
            mVideoDecoder->mapFrame(frameInfo.first.picture_index, frameInfo.second);

        cvtFromNv12(decodedFrame, frame, mColorFormat, mStream);
        CU_CHECK(cuStreamSynchronize(mStream));

        // unmap video frame
        // unmapFrame() synchronizes with the VideoDecode API (ensures the frame has
        // finished decoding)
        mVideoDecoder->unmapFrame(decodedFrame);
    }

    releaseFrameInfo(frameInfo);
    mIFrame++;
    return true;
}

bool validColorFormat(const ColorFormat colorFormat) {
    if (colorFormat == ColorFormat::RGB || colorFormat == ColorFormat::BGR ||
        colorFormat == ColorFormat::GRAY || colorFormat == ColorFormat::NV_NV12) {
        return true;
    }
    return false;
}

bool VideoReaderImpl::setColorFormat(ColorFormat colorFormat_) {
    if (!validColorFormat(colorFormat_))
        return false;
    mColorFormat = colorFormat_;
    return true;
}

Codec VideoReaderImpl::getCodec() { return mVideoSource->getCodec(); }

int VideoReaderImpl::getTargetWidth() { return mVideoDecoder->targetWidth(); }

int VideoReaderImpl::getTargetHeight() { return mVideoDecoder->targetHeight(); }

int VideoReaderImpl::getPosFrames() const { return mIFrame; }

bool VideoReaderImpl::nextFrame(CniFrame &frame) {
    if (!internalGrab(frame))
        return false;
    return true;
}

VideoReader *createVideoReader(int deviceId, const std::string &filename,
                               VideoReaderInitParams params) {
    CNIAI_CHECK(!filename.empty());

    VideoSource *videoSource =
        new FFmpegVideoSourceWrapper(new FFmpegVideoSource(filename));

    return new VideoReaderImpl(deviceId, videoSource, params.minNumDecodeSurfaces,
                               params.allowFrameDrop, params.targetSz, params.srcRoi,
                               params.targetRoi);
}

} // namespace cniai::nvcodec