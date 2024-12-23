
#pragma once

#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>

namespace cniai::blocking_queue {

template <typename T>
class BlockingQueue {
public:
    explicit BlockingQueue(size_t maxCapacity = 0)
        : mMaxCapacity(maxCapacity), mStop(false) {}

    ~BlockingQueue() { stop(); }

    BlockingQueue(const BlockingQueue &) = delete;
    BlockingQueue &operator=(const BlockingQueue &) = delete;

    void enqueue(T item) {
        std::unique_lock<std::mutex> lock(mMtx);
        cvNotFull_.wait(lock, [this] {
            return mQueue.size() < mMaxCapacity || mMaxCapacity == 0 || mStop;
        });

        if (mStop) {
            throw std::runtime_error("Queue is stopped.");
        }

        mQueue.push(std::move(item));
        mCvNotEmpty.notify_one();
    }

    std::optional<T> dequeue() {
        std::unique_lock<std::mutex> lock(mMtx);
        mCvNotEmpty.wait(lock, [this] { return !mQueue.empty() || mStop; });

        if (mQueue.empty()) {
            //            throw std::runtime_error("Queue is stopped and empty.");
            return std::nullopt;
        }

        T item = std::move(mQueue.front());
        mQueue.pop();
        cvNotFull_.notify_one();
        return item;
    }

    std::optional<T> dequeueWithTimeout(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mMtx);
        if (!mCvNotEmpty.wait_for(lock, timeout,
                                  [this] { return !mQueue.empty() || mStop; })) {
            return std::nullopt; // timeout
        }

        if (mQueue.empty()) {
            throw std::runtime_error("Queue is stopped and empty.");
        }

        T item = std::move(mQueue.front());
        mQueue.pop();
        cvNotFull_.notify_one();
        return item;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mMtx);
            mStop = true;
        }
        mCvNotEmpty.notify_all();
        cvNotFull_.notify_all();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mMtx);
        return mQueue.size();
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mMtx);
        return mQueue.empty();
    }

private:
    mutable std::mutex mMtx;
    std::queue<T> mQueue;
    std::condition_variable mCvNotEmpty;
    std::condition_variable cvNotFull_;
    size_t mMaxCapacity;
    bool mStop;
};

} // namespace cniai::blocking_queue