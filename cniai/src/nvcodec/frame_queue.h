#pragma once

#include <mutex>
#include <queue>
#include <vector>

#include <nvcuvid.h>

namespace cniai::nvcodec {

// class RawPacket {
// public:
//     RawPacket(const unsigned char *_data, const size_t _size = 0,
//               const bool _containsKeyFrame = false);
//
//     const unsigned char *Data() const noexcept { return data.data(); }
//
//     size_t Size() const noexcept { return data.size(); }
//
//     bool ContainsKeyFrame() const noexcept { return containsKeyFrame; }
//
// private:
//     std::vector<unsigned char> data;
//     bool containsKeyFrame = false;
// };

class FrameQueue {
public:
    ~FrameQueue();
    void init(const int _maxSz);

    // Resize the current frame queue keeping any existing queued values - must only
    // be called in the same thread as enqueue.
    // Parameters:
    //      newSz - new size of the frame queue.
    void resize(const int newSz);

    void endDecode() { endOfDecode_ = true; }

    bool isEndOfDecode() const { return endOfDecode_ != 0; }

    // Spins until frame becomes available or decoding gets canceled.
    // If the requested frame is available the method returns true.
    // If decoding was interrupted before the requested frame becomes
    // available, the method returns false.
    // If allowFrameDrop == true, spin is disabled and n > 0 frames are discarded
    // to ensure a frame is available.
    bool waitUntilFrameAvailable(int pictureIndex, const bool allowFrameDrop = false);

    bool waitUntilEmpty();

    void enqueue(const CUVIDPARSERDISPINFO *picParams);

    // Deque the next frame.
    // Parameters:
    //      displayInfo - New frame info gets placed into this object.
    // Returns:
    //      true, if a new frame was returned,
    //      false, if the queue was empty and no new frame could be returned.
    bool dequeue(CUVIDPARSERDISPINFO &displayInfo);

    // Deque all frames up to and including the frame with index pictureIndex - must only
    // be called in the same thread as enqueue.
    // Parameters:
    //      pictureIndex - Display index of the frame.
    // Returns:
    //      true, if successful,
    //      false, if no frames are dequed.
    bool dequeueUntil(const int pictureIndex);

    void releaseFrame(const CUVIDPARSERDISPINFO &picParams) {
        isFrameInUse_[picParams.picture_index] = 0;
    }

    int getMaxSz() { return maxSz; }

private:
    bool isInUse(int pictureIndex) const { return isFrameInUse_[pictureIndex] != 0; }

    std::mutex mtx_;
    volatile int *isFrameInUse_ = 0;
    volatile int endOfDecode_ = 0;
    int framesInQueue_ = 0;
    int readPosition_ = 0;
    std::vector<CUVIDPARSERDISPINFO> displayQueue_;
    int maxSz = 0;
    //    std::queue<RawPacket> rawPacketQueue;
};

} // namespace cniai::nvcodec
