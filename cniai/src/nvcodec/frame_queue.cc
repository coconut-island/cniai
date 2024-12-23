#include "frame_queue.h"

#include <cstring>
#include <thread>

namespace cniai::nvcodec {

// RawPacket::RawPacket(const unsigned char *data_, const size_t size,
//                      const bool containsKeyFrame_)
//     : data(data_, data_ + size), containsKeyFrame(containsKeyFrame_){};

FrameQueue::~FrameQueue() {
    if (isFrameInUse_)
        delete[] isFrameInUse_;
}

void FrameQueue::init(const int _maxSz) {
    std::unique_lock<std::mutex> lock(mtx_);
    if (isFrameInUse_)
        return;
    maxSz = _maxSz;
    displayQueue_ = std::vector<CUVIDPARSERDISPINFO>(maxSz, CUVIDPARSERDISPINFO());
    isFrameInUse_ = new volatile int[maxSz];
    std::memset((void *)isFrameInUse_, 0, sizeof(*isFrameInUse_) * maxSz);
}

void FrameQueue::resize(const int newSz) {
    if (newSz == maxSz)
        return;
    if (!isFrameInUse_)
        return init(newSz);
    std::unique_lock<std::mutex> lock(mtx_);
    const int maxSzOld = maxSz;
    maxSz = newSz;
    const auto displayQueueOld = displayQueue_;
    displayQueue_ = std::vector<CUVIDPARSERDISPINFO>(maxSz, CUVIDPARSERDISPINFO());
    for (int i = readPosition_; i < readPosition_ + framesInQueue_; i++)
        displayQueue_.at(i % displayQueue_.size()) =
            displayQueueOld.at(i % displayQueueOld.size());
    const volatile int *const isFrameInUseOld = isFrameInUse_;
    isFrameInUse_ = new volatile int[maxSz];
    std::memset((void *)isFrameInUse_, 0, sizeof(*isFrameInUse_) * maxSz);
    std::memcpy((void *)isFrameInUse_, (void *)isFrameInUseOld,
                sizeof(*isFrameInUseOld) * std::min(maxSz, maxSzOld));
    delete[] isFrameInUseOld;
}

bool FrameQueue::waitUntilFrameAvailable(int pictureIndex, const bool allowFrameDrop) {
    while (isInUse(pictureIndex)) {
        if (allowFrameDrop && dequeueUntil(pictureIndex))
            break;
        // Decoder is getting too far ahead from display
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

        if (isEndOfDecode())
            return false;
    }

    return true;
}

bool FrameQueue::waitUntilEmpty() {
    while (framesInQueue_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (isEndOfDecode())
            return false;
    }
    return true;
}

void FrameQueue::enqueue(const CUVIDPARSERDISPINFO *picParams) {
    // Mark the frame as 'in-use' so we don't re-use it for decoding until it is no longer
    // needed for display
    isFrameInUse_[picParams->picture_index] = true;

    // Wait until we have a free entry in the display queue (should never block if we have
    // enough entries)
    do {
        bool isFramePlaced = false;

        {
            std::unique_lock<std::mutex> lock(mtx_);

            if (framesInQueue_ < maxSz) {
                const int writePosition = (readPosition_ + framesInQueue_) % maxSz;
                displayQueue_.at(writePosition) = *picParams;
                framesInQueue_++;
                isFramePlaced = true;
            }
        }

        if (isFramePlaced) // Done
            break;

        // Wait a bit
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (!isEndOfDecode());
}

bool FrameQueue::dequeueUntil(const int pictureIndex) {
    std::unique_lock<std::mutex> lock(mtx_);
    if (isFrameInUse_[pictureIndex] != 1)
        return false;
    for (int i = 0; i < framesInQueue_; i++) {
        const bool found = displayQueue_.at(readPosition_).picture_index == pictureIndex;
        isFrameInUse_[displayQueue_.at(readPosition_).picture_index] = 0;
        framesInQueue_--;
        readPosition_ = (readPosition_ + 1) % maxSz;
        if (found)
            return true;
    }
    return false;
}

bool FrameQueue::dequeue(CUVIDPARSERDISPINFO &displayInfo) {
    std::unique_lock<std::mutex> lock(mtx_);

    if (framesInQueue_ > 0) {
        int entry = readPosition_;
        displayInfo = displayQueue_.at(entry);
        readPosition_ = (entry + 1) % maxSz;
        framesInQueue_--;
        isFrameInUse_[displayInfo.picture_index] = 2;
        return true;
    }

    return false;
}

} // namespace cniai::nvcodec