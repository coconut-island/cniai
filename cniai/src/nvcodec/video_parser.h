#pragma once

#include "frame_queue.h"
#include "video_decoder.h"

namespace cniai::nvcodec {

class VideoParser {
public:
    VideoParser(VideoDecoder *videoDecoder, FrameQueue *frameQueue,
                bool allowFrameDrop = false);

    ~VideoParser() { cuvidDestroyVideoParser(mParser); }

    bool parseVideoData(const unsigned char *data, size_t size, bool endOfStream);

    bool hasError() const { return mHasError; }

    bool allowFrameDrops() const { return mAllowFrameDrop; }

private:
    VideoDecoder *mVideoDecoder = nullptr;
    FrameQueue *mFrameQueue = nullptr;
    CUvideoparser mParser = nullptr;
    volatile bool mHasError = false;
    bool mAllowFrameDrop = false;

    // Called when the decoder encounters a video format change (or initial sequence
    // header)
    static int CUDAAPI HandleVideoSequence(void *pUserData, CUVIDEOFORMAT *pFormat);

    // Called by the video parser to decode a single picture
    // Since the parser will deliver data as fast as it can, we need to make sure that the
    // picture index we're attempting to use for decode is no longer used for display
    static int CUDAAPI HandlePictureDecode(void *pUserData, CUVIDPICPARAMS *pPicParams);

    // Called by the video parser to display a video frame (in the case of field pictures,
    // there may be 2 decode calls per 1 display call, since two fields make up one frame)
    static int CUDAAPI HandlePictureDisplay(void *pUserData,
                                            CUVIDPARSERDISPINFO *pPicParams);
};

} // namespace cniai::nvcodec
