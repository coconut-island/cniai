#pragma once

#include <cuda_runtime_api.h>

#include <atomic>

#include "common/assert.h"
#include "nvcommon/cuda_util.h"

namespace cniai {

class CniFrame {
public:
    enum class Format {
        UNKNOWN = 0,
        NV12 = 1,
        RGB = 2,
        BGR = 3,
        YUV420 = 4,
        GRAY = 5,
    };

    enum class MemoryType { GPU = 0, CPU = 1, PINNED = 2 };

public:
    CniFrame() = default;

    CniFrame(int width, int height, Format format, void *data = nullptr, int pitch = 0,
             MemoryType memoryType = MemoryType::GPU)
        : mWidth(width), mHeight(height), mFormat(format), mPitch(pitch), mData(data),
          mMemoryType(memoryType) {

        if (data != nullptr) {
            extraData = true;
        }
        if (pitch == 0) {
            mPitch = defaultPitch();
        } else {
            mPitch = pitch;
        }
    }

    ~CniFrame() { release(); }

    CniFrame(const CniFrame &cniFrame)
        : mWidth(cniFrame.mWidth), mHeight(cniFrame.mHeight), mFormat(cniFrame.mFormat),
          mPitch(cniFrame.mPitch), mData(cniFrame.mData),
          mMemoryType(cniFrame.mMemoryType) {
        if (mDataRefCount) {
            ++(*mDataRefCount);
        }
    }

    CniFrame &operator=(const CniFrame &cniFrame) {
        if (this == &cniFrame)
            return *this;
        if (cniFrame.mDataRefCount)
            (*(cniFrame.mDataRefCount))++;
        release();
        mWidth = cniFrame.mWidth;
        mHeight = cniFrame.mHeight;
        mFormat = cniFrame.mFormat;
        mPitch = cniFrame.mPitch;
        mData = cniFrame.mData;
        mMemoryType = cniFrame.mMemoryType;
        return *this;
    }

    CniFrame(CniFrame &&cniFrame) noexcept
        : mWidth(cniFrame.mWidth), mHeight(cniFrame.mHeight), mFormat(cniFrame.mFormat),
          mPitch(cniFrame.mPitch), mData(cniFrame.mData) {
        cniFrame.mWidth = 0;
        cniFrame.mHeight = 0;
        cniFrame.mFormat = Format::UNKNOWN;
        cniFrame.mPitch = 0;
        cniFrame.mData = nullptr;
        cniFrame.mMemoryType = MemoryType::GPU;
    }

    CniFrame &operator=(CniFrame &&cniFrame) noexcept {
        if (this == &cniFrame)
            return *this;

        release();
        mWidth = cniFrame.mWidth;
        mHeight = cniFrame.mHeight;
        mFormat = cniFrame.mFormat;
        mPitch = cniFrame.mPitch;
        mData = cniFrame.mData;
        mMemoryType = cniFrame.mMemoryType;

        cniFrame.mWidth = 0;
        cniFrame.mHeight = 0;
        cniFrame.mFormat = Format::UNKNOWN;
        cniFrame.mPitch = 0;
        cniFrame.mData = nullptr;
        cniFrame.mMemoryType = MemoryType::GPU;
        return *this;
    }

public:
    int defaultPitch() {
        switch (mFormat) {
        case Format::NV12:
        case Format::YUV420:
        case Format::GRAY:
            return mWidth;
        case Format::RGB:
        case Format::BGR:
            return mWidth * 3;
        default:
            CNIAI_THROW("Unsupported frame format");
        }
    }

    int size() const {
        switch (mFormat) {
        case Format::NV12:
        case Format::YUV420:
            return mPitch * mHeight * 3 / 2;
        case Format::RGB:
        case Format::BGR:
        case Format::GRAY:
            return mPitch * mHeight;
        default:
            CNIAI_THROW("Unsupported frame format");
        }
    }

    void create(int width, int height, Format format, int pitch = 0,
                cudaStream_t stream = nullptr, MemoryType memoryType = MemoryType::GPU) {
        CNIAI_CHECK(format != Format::UNKNOWN);
        if (mData == nullptr || width != mWidth || height != mHeight ||
            format != mFormat) {
            release();
            mWidth = width;
            mHeight = height;
            mFormat = format;
            mMemoryType = memoryType;
            if (pitch == 0) {
                mPitch = defaultPitch();
            } else {
                mPitch = pitch;
            }

            switch (mMemoryType) {
            case MemoryType::GPU:
                if (stream == nullptr) {
                    CUDA_CHECK(cudaMalloc(&mData, size()));
                } else {
                    CUDA_CHECK(cudaMallocAsync(&mData, size(), stream));
                }
                break;
            case MemoryType::PINNED:
                CUDA_CHECK(cudaMallocHost(&mData, size()));
                break;
            case MemoryType::CPU:
                mData = malloc(size());
            }
            mDataRefCount = new std::atomic<int>(1);
        }
    }

    void create(int width, int height, Format format,
                MemoryType memoryType = MemoryType::GPU, cudaStream_t stream = nullptr) {
        create(width, height, format, 0, stream, memoryType);
    }

    void create(int width, int height, Format format, cudaStream_t stream = nullptr) {
        create(width, height, format, 0, stream);
    }

    void release() {
        if (mDataRefCount) {
            --(*mDataRefCount);
            if (*mDataRefCount == 0) {
                if (mData != nullptr && !extraData) {
                    switch (mMemoryType) {
                    case MemoryType::GPU:
                        CUDA_CHECK(cudaFree(mData));
                        break;
                    case MemoryType::PINNED:
                        CUDA_CHECK(cudaFreeHost(mData));
                        break;
                    case MemoryType::CPU:
                        free(mData);
                    }
                    mData = nullptr;
                }
                delete mDataRefCount;
                mDataRefCount = nullptr;
            }
        }
        mWidth = 0;
        mHeight = 0;
        mFormat = Format::UNKNOWN;
    };

    bool empty() const { return mData == nullptr || size() == 0; }

    void *data() const { return mData; }

    int getWidth() const { return mWidth; }

    int getHeight() const { return mHeight; }

    int getPitch() const { return mPitch; }

    Format getFormat() const { return mFormat; };

    MemoryType getMemoryType() const { return mMemoryType; };

    CniFrame clone(cudaStream_t stream = nullptr) const {
        return clone(getMemoryType(), stream);
    }

    CniFrame clone(MemoryType memoryType, cudaStream_t stream = nullptr) const {
        CNIAI_CHECK(!empty());
        CniFrame frame;
        frame.create(getWidth(), getHeight(), getFormat(), memoryType, stream);
        if (stream == nullptr) {
            CUDA_CHECK(cudaMemcpy(frame.data(), data(), size(), cudaMemcpyDeviceToHost));
        } else {
            CUDA_CHECK(
                cudaMemcpyAsync(frame.data(), data(), size(), cudaMemcpyDefault, stream));
        }
        return frame;
    }

private:
    int mHeight = 0;
    int mWidth = 0;
    Format mFormat = Format::UNKNOWN;
    MemoryType mMemoryType = MemoryType::GPU;
    int mPitch = 0;
    void *mData = nullptr;
    bool extraData = false;
    std::atomic<int> *mDataRefCount = nullptr;
};

} // namespace cniai