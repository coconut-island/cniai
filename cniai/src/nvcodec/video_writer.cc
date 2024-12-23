#include "nvcodec.h"

#include "fstream"
#include <queue>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvEncodeAPI.h>

#include "ffmpeg_video_source.h"
#include "frame_queue.h"
#include "nv_encoder_cuda.h"

#include "common/assert.h"
#include "common/logging.h"
#include "ffmpeg_muxer.h"
#include "nvcommon/cuda_util.h"

#include <cniai_cuda_kernel/imgproc.h>

namespace cniai::nvcodec {

NV_ENC_BUFFER_FORMAT EncBufferFormat(ColorFormat colorFormat);
int NChannels(ColorFormat colorFormat);
GUID CodecGuid(Codec codec);
void FrameRate(float fps, uint32_t &frameRateNum, uint32_t &frameRateDen);
GUID EncodingProfileGuid(EncodeProfile encodingProfile);
GUID EncodingPresetGuid(EncodePreset nvPreset);

bool operator==(const EncoderParams &lhs, const EncoderParams &rhs) {
    return std::tie(lhs.nvPreset, lhs.tuningInfo, lhs.encodingProfile,
                    lhs.rateControlMode, lhs.multiPassEncoding, lhs.constQp.qpInterB,
                    lhs.constQp.qpInterP, lhs.constQp.qpIntra, lhs.averageBitRate,
                    lhs.maxBitRate, lhs.targetQuality, lhs.gopLength) ==
           std::tie(rhs.nvPreset, rhs.tuningInfo, rhs.encodingProfile,
                    rhs.rateControlMode, rhs.multiPassEncoding, rhs.constQp.qpInterB,
                    rhs.constQp.qpInterP, rhs.constQp.qpIntra, rhs.averageBitRate,
                    rhs.maxBitRate, rhs.targetQuality, rhs.gopLength);
};

class FFmpegVideoWriter : public EncoderCallback {
public:
    FFmpegVideoWriter(const std::string &filename, Codec codec, int width, int height,
                      float fps);
    ~FFmpegVideoWriter() override;
    void onEncoded(const std::vector<std::vector<uint8_t>> &vPacket,
                   const std::vector<uint64_t> &pts) override;
    void onEncodingFinished() override;

private:
    FFmpegMuxer *mFFmpegMuxer = nullptr;
};

FFmpegVideoWriter::FFmpegVideoWriter(const std::string &filename, Codec codec, int width,
                                     int height, float fps) {

    AVCodecID avCodecId = codec == Codec::H264 ? AV_CODEC_ID_H264 : AV_CODEC_ID_HEVC;
    mFFmpegMuxer = new FFmpegMuxer(filename, avCodecId, width, height, fps);
}

void FFmpegVideoWriter::onEncodingFinished() {
    delete mFFmpegMuxer;
    mFFmpegMuxer = nullptr;
}

FFmpegVideoWriter::~FFmpegVideoWriter() { onEncodingFinished(); }

void FFmpegVideoWriter::onEncoded(const std::vector<std::vector<uint8_t>> &vPacket,
                                  const std::vector<uint64_t> &pts) {
    CNIAI_CHECK(vPacket.size() == pts.size());
    for (int i = 0; i < vPacket.size(); i++) {
        std::vector<uint8_t> packet = vPacket.at(i);
        mFFmpegMuxer->write(packet.data(), packet.size(), pts.at(i));
    }
}

class VideoWriterImpl : public VideoWriter {
public:
    VideoWriterImpl(int deviceId, EncoderCallback *encoderCallback, Codec codec,
                    int width, int height, float fps, ColorFormat colorFormat,
                    const EncoderParams &encoderParams);
    ~VideoWriterImpl() override;
    void write(CniFrame &frame) override;
    EncoderParams getEncoderParams() const override;
    void release();

private:
    void Init(int deviceId, Codec codec, int width, int height, float fps);
    void InitializeEncoder(GUID codec, float fps);
    void CopyToNvSurface(const CniFrame &frame);

    EncoderCallback *mEncoderCallback;
    ColorFormat mColorFormat = ColorFormat::UNDEFINED;
    NV_ENC_BUFFER_FORMAT mSurfaceFormat =
        NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_UNDEFINED;
    EncoderParams mEncoderParams;
    CUstream mStream = nullptr;
    NvEncoderCuda *mNvEncoderCuda = nullptr;
    std::vector<std::vector<uint8_t>> mPackets;
    int mSrcChannels = 0;
    CUcontext mCUcontext = nullptr;
};

NV_ENC_BUFFER_FORMAT EncBufferFormat(const ColorFormat colorFormat) {
    switch (colorFormat) {
    case ColorFormat::BGR:
        return NV_ENC_BUFFER_FORMAT_ABGR;
    case ColorFormat::RGB:
        return NV_ENC_BUFFER_FORMAT_ARGB;
    case ColorFormat::NV_NV12:
        return NV_ENC_BUFFER_FORMAT_NV12;
    default:
        return NV_ENC_BUFFER_FORMAT_UNDEFINED;
    }
}

int NChannels(const ColorFormat colorFormat) {
    switch (colorFormat) {
    case ColorFormat::BGR:
    case ColorFormat::RGB:
        return 3;
    case ColorFormat::GRAY:
    case ColorFormat::NV_NV12:
        return 1;
    default:
        return 0;
    }
}

VideoWriterImpl::VideoWriterImpl(int deviceId, EncoderCallback *encoderCallback,
                                 Codec codec, int width, int height, float fps,
                                 ColorFormat colorFormat,
                                 const EncoderParams &encoderParams)
    : mEncoderCallback(encoderCallback), mColorFormat(colorFormat),
      mEncoderParams(encoderParams) {
    CNIAI_CHECK(mColorFormat != ColorFormat::UNDEFINED);
    mSurfaceFormat = EncBufferFormat(mColorFormat);
    if (mSurfaceFormat == NV_ENC_BUFFER_FORMAT_UNDEFINED) {
        LOG_WARN("Unsupported input surface format: {}", (int)mColorFormat);
        CNIAI_THROW("Unsupported input surface format: {}", (int)mColorFormat);
    }
    mSrcChannels = NChannels(mColorFormat);
    Init(deviceId, codec, width, height, fps);
}

void VideoWriterImpl::release() {
    std::vector<uint64_t> pts;
    mNvEncoderCuda->EndEncode(mPackets, pts);
    mEncoderCallback->onEncoded(mPackets, pts);
    mEncoderCallback->onEncodingFinished();
    mNvEncoderCuda->DestroyEncoder();
    if (mStream != nullptr) {
        CUDA_CHECK(cudaStreamDestroy(mStream));
        mStream = nullptr;
    }
    if (mCUcontext != nullptr) {
        CU_CHECK(cuCtxDestroy(mCUcontext));
        mCUcontext = nullptr;
    }
}

VideoWriterImpl::~VideoWriterImpl() { release(); }

GUID CodecGuid(const Codec codec) {
    switch (codec) {
    case Codec::H264:
        return NV_ENC_CODEC_H264_GUID;
    case Codec::HEVC:
        return NV_ENC_CODEC_HEVC_GUID;
    default:
        break;
    }
    std::string msg = "Unknown codec: cudacodec::VideoWriter only supports "
                      "CODEC_VW::H264 and CODEC_VW::HEVC";
    LOG_WARN(msg);
    CNIAI_THROW(msg);
}

void VideoWriterImpl::Init(int deviceId, Codec codec, int width, int height, float fps) {
    // init context
    CU_CHECK(cuInit(0));
    CUdevice device;
    CU_CHECK(cuDeviceGet(&device, deviceId));
    CU_CHECK(cuCtxCreate(&mCUcontext, 0, device));
    CNIAI_CHECK(mSrcChannels != 0);
    CU_CHECK(cuStreamCreate(&mStream, CU_STREAM_NON_BLOCKING));
    const GUID codecGuid = CodecGuid(codec);
    try {
        mNvEncoderCuda = new NvEncoderCuda(mCUcontext, width, height, mSurfaceFormat);
        InitializeEncoder(codecGuid, fps);
        mNvEncoderCuda->SetIOCudaStreams((NV_ENC_CUSTREAM_PTR)&mStream,
                                         (NV_ENC_CUSTREAM_PTR)&mStream);
    } catch (std::exception &e) {
        std::string msg =
            std::string(
                "Error initializing Nvidia Encoder. Refer to Nvidia's GPU Support "
                "Matrix to confirm your GPU supports hardware encoding, ") +
            std::string("codec and surface format and check the encoder documentation to "
                        "verify your choice of encoding paramaters are supported.") +
            e.what();
        CNIAI_THROW(msg);
    }
    CNIAI_CHECK(width == mNvEncoderCuda->GetEncodeWidth() &&
                height == mNvEncoderCuda->GetEncodeHeight());
}

void FrameRate(float fps, uint32_t &frameRateNum, uint32_t &frameRateDen) {
    CNIAI_CHECK(fps >= 0);
    int frame_rate = (int)(fps + 0.5);
    int frame_rate_base = 1;
    while (fabs(((double)frame_rate / frame_rate_base) - fps) > 0.001) {
        frame_rate_base *= 10;
        frame_rate = (int)(fps * frame_rate_base + 0.5);
    }
    frameRateNum = frame_rate;
    frameRateDen = frame_rate_base;
}

GUID EncodingProfileGuid(const EncodeProfile encodingProfile) {
    switch (encodingProfile) {
    case (ENC_CODEC_PROFILE_AUTOSELECT):
        return NV_ENC_CODEC_PROFILE_AUTOSELECT_GUID;
    case (ENC_H264_PROFILE_BASELINE):
        return NV_ENC_H264_PROFILE_BASELINE_GUID;
    case (ENC_H264_PROFILE_MAIN):
        return NV_ENC_H264_PROFILE_MAIN_GUID;
    case (ENC_H264_PROFILE_HIGH):
        return NV_ENC_H264_PROFILE_HIGH_GUID;
    case (ENC_H264_PROFILE_HIGH_444):
        return NV_ENC_H264_PROFILE_HIGH_444_GUID;
    case (ENC_H264_PROFILE_STEREO):
        return NV_ENC_H264_PROFILE_STEREO_GUID;
    case (ENC_H264_PROFILE_PROGRESSIVE_HIGH):
        return NV_ENC_H264_PROFILE_PROGRESSIVE_HIGH_GUID;
    case (ENC_H264_PROFILE_CONSTRAINED_HIGH):
        return NV_ENC_H264_PROFILE_CONSTRAINED_HIGH_GUID;
    case (ENC_HEVC_PROFILE_MAIN):
        return NV_ENC_HEVC_PROFILE_MAIN_GUID;
    case (ENC_HEVC_PROFILE_MAIN10):
        return NV_ENC_HEVC_PROFILE_MAIN10_GUID;
    case (ENC_HEVC_PROFILE_FREXT):
        return NV_ENC_HEVC_PROFILE_FREXT_GUID;
    default:
        break;
    }
    std::string msg = "Unknown Encoding Profile.";
    LOG_WARN(msg);
    CNIAI_THROW(msg);
}

GUID EncodingPresetGuid(const EncodePreset nvPreset) {
    switch (nvPreset) {
    case ENC_PRESET_P1:
        return NV_ENC_PRESET_P1_GUID;
    case ENC_PRESET_P2:
        return NV_ENC_PRESET_P2_GUID;
    case ENC_PRESET_P3:
        return NV_ENC_PRESET_P3_GUID;
    case ENC_PRESET_P4:
        return NV_ENC_PRESET_P4_GUID;
    case ENC_PRESET_P5:
        return NV_ENC_PRESET_P5_GUID;
    case ENC_PRESET_P6:
        return NV_ENC_PRESET_P6_GUID;
    case ENC_PRESET_P7:
        return NV_ENC_PRESET_P7_GUID;
    default:
        break;
    }
    std::string msg = "Unknown Nvidia Encoding Preset.";
    LOG_WARN(msg);
    CNIAI_THROW(msg);
}

void VideoWriterImpl::InitializeEncoder(const GUID codec, const float fps) {
    NV_ENC_INITIALIZE_PARAMS initializeParams = {};
    initializeParams.version = NV_ENC_INITIALIZE_PARAMS_VER;
    NV_ENC_CONFIG encodeConfig = {};
    encodeConfig.version = NV_ENC_CONFIG_VER;
    initializeParams.encodeConfig = &encodeConfig;
    mNvEncoderCuda->CreateDefaultEncoderParams(
        &initializeParams, codec, EncodingPresetGuid(mEncoderParams.nvPreset),
        (NV_ENC_TUNING_INFO)mEncoderParams.tuningInfo);
    FrameRate(fps, initializeParams.frameRateNum, initializeParams.frameRateDen);
    initializeParams.encodeConfig->profileGUID =
        EncodingProfileGuid(mEncoderParams.encodingProfile);
    initializeParams.encodeConfig->rcParams.rateControlMode =
        (NV_ENC_PARAMS_RC_MODE)(mEncoderParams.rateControlMode +
                                mEncoderParams.multiPassEncoding);
    initializeParams.encodeConfig->rcParams.constQP = {mEncoderParams.constQp.qpInterB,
                                                       mEncoderParams.constQp.qpInterB,
                                                       mEncoderParams.constQp.qpInterB};
    initializeParams.encodeConfig->rcParams.averageBitRate =
        mEncoderParams.averageBitRate;
    initializeParams.encodeConfig->rcParams.maxBitRate = mEncoderParams.maxBitRate;
    initializeParams.encodeConfig->rcParams.targetQuality = mEncoderParams.targetQuality;
    initializeParams.encodeConfig->gopLength = mEncoderParams.gopLength;
    initializeParams.encodeConfig->frameIntervalP = 1;
    // #if !defined(WIN32_WAIT_FOR_FFMPEG_WRAPPER_UPDATE)
    //     if (initializeParams.encodeConfig->frameIntervalP > 1) {
    //         CNIAI_CHECK(mEncoderCallback->setFrameIntervalP(
    //             initializeParams.encodeConfig->frameIntervalP));
    //     }
    // #endif
    if (codec == NV_ENC_CODEC_H264_GUID)
        initializeParams.encodeConfig->encodeCodecConfig.h264Config.idrPeriod =
            mEncoderParams.idrPeriod;
    else if (codec == NV_ENC_CODEC_HEVC_GUID)
        initializeParams.encodeConfig->encodeCodecConfig.hevcConfig.idrPeriod =
            mEncoderParams.idrPeriod;
    mNvEncoderCuda->CreateEncoder(&initializeParams);
}

#pragma pack(push, 1)

struct BITMAPFILEHEADER {
    uint16_t bfType;      // 文件类型, 0x4D42 ("BM")
    uint32_t bfSize;      // 文件大小
    uint16_t bfReserved1; // 保留，设置为0
    uint16_t bfReserved2; // 保留，设置为0
    uint32_t bfOffBits;   // 从文件开头到像素数据的偏移
};

struct BITMAPINFOHEADER {
    uint32_t biSize;         // 信息头的大小
    int32_t biWidth;         // 图像宽度
    int32_t biHeight;        // 图像高度
    uint16_t biPlanes;       // 颜色平面数
    uint16_t biBitCount;     // 每像素的位数 (24 bits for RGB)
    uint32_t biCompression;  // 压缩类型，0 表示不压缩
    uint32_t biSizeImage;    // 图像数据的大小
    int32_t biXPelsPerMeter; // 水平分辨率
    int32_t biYPelsPerMeter; // 垂直分辨率
    uint32_t biClrUsed;      // 使用的颜色数
    uint32_t biClrImportant; // 重要的颜色数
};

#pragma pack(pop)

inline int rgbaToBmpFile(const char *pFileName, const uint8_t *pRgbaData,
                         const int nWidth, const int nHeight) {
    // BMP 文件头
    BITMAPFILEHEADER fileHeader;
    fileHeader.bfType = 0x4D42; // 'BM'
    fileHeader.bfSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) +
                        nWidth * nHeight * 3; // 24-bit BMP
    fileHeader.bfReserved1 = 0;
    fileHeader.bfReserved2 = 0;
    fileHeader.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);

    // BMP 信息头
    BITMAPINFOHEADER infoHeader;
    infoHeader.biSize = sizeof(BITMAPINFOHEADER);
    infoHeader.biWidth = nWidth;
    infoHeader.biHeight = nHeight;
    infoHeader.biPlanes = 1;
    infoHeader.biBitCount = 24;   // 24-bit BMP (RGB)
    infoHeader.biCompression = 0; // 无压缩
    infoHeader.biSizeImage = nWidth * nHeight * 3;
    infoHeader.biXPelsPerMeter = 0;
    infoHeader.biYPelsPerMeter = 0;
    infoHeader.biClrUsed = 0;
    infoHeader.biClrImportant = 0;

    // 打开文件以写入
    std::ofstream outFile(pFileName, std::ios::binary);
    if (!outFile) {
        std::cerr << "无法打开文件进行写入。" << std::endl;
        return -1;
    }

    // 写入文件头和信息头
    outFile.write(reinterpret_cast<char *>(&fileHeader), sizeof(fileHeader));
    outFile.write(reinterpret_cast<char *>(&infoHeader), sizeof(infoHeader));

    // 写入像素数据（RGBA -> BGR 转换）
    for (int y = nHeight - 1; y >= 0; --y) { // BMP 存储顺序是从底部到顶部
        for (int x = 0; x < nWidth; ++x) {
            int index = (y * nWidth + x) * 4; // RGBA 数据索引
            uint8_t r = pRgbaData[index + 0]; // 红色
            uint8_t g = pRgbaData[index + 1]; // 绿色
            uint8_t b = pRgbaData[index + 2]; // 蓝色
            // 不使用 Alpha 通道
            outFile.put(b); // 蓝色
            outFile.put(g); // 绿色
            outFile.put(r); // 红色
        }
    }

    outFile.close();
    return 0; // 成功
}

inline int rgbToBmpFile(const char *pFileName, const uint8_t *pRgbData, const int nWidth,
                        const int nHeight) {
    // BMP 文件头
    BITMAPFILEHEADER fileHeader;
    fileHeader.bfType = 0x4D42; // 'BM'
    fileHeader.bfSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) +
                        nWidth * nHeight * 3; // 24-bit BMP
    fileHeader.bfReserved1 = 0;
    fileHeader.bfReserved2 = 0;
    fileHeader.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);

    // BMP 信息头
    BITMAPINFOHEADER infoHeader;
    infoHeader.biSize = sizeof(BITMAPINFOHEADER);
    infoHeader.biWidth = nWidth;
    infoHeader.biHeight = nHeight;
    infoHeader.biPlanes = 1;
    infoHeader.biBitCount = 24;   // 24-bit BMP (RGB)
    infoHeader.biCompression = 0; // 无压缩
    infoHeader.biSizeImage = nWidth * nHeight * 3;
    infoHeader.biXPelsPerMeter = 0;
    infoHeader.biYPelsPerMeter = 0;
    infoHeader.biClrUsed = 0;
    infoHeader.biClrImportant = 0;

    // 打开文件以写入
    std::ofstream outFile(pFileName, std::ios::binary);
    if (!outFile) {
        std::cerr << "无法打开文件进行写入。" << std::endl;
        return -1;
    }

    // 写入文件头和信息头
    outFile.write(reinterpret_cast<char *>(&fileHeader), sizeof(fileHeader));
    outFile.write(reinterpret_cast<char *>(&infoHeader), sizeof(infoHeader));

    // 写入像素数据（RGB）
    for (int y = nHeight - 1; y >= 0; --y) { // BMP 存储顺序是从底部到顶部
        for (int x = 0; x < nWidth; ++x) {
            int index = (y * nWidth + x) * 3; // RGB 数据索引
            uint8_t r = pRgbData[index + 0];  // 红色
            uint8_t g = pRgbData[index + 1];  // 绿色
            uint8_t b = pRgbData[index + 2];  // 蓝色
            // 写入像素数据（BGR 顺序）
            outFile.put(b); // 蓝色
            outFile.put(g); // 绿色
            outFile.put(r); // 红色
        }
    }

    outFile.close();
    return 0; // 成功
}

void VideoWriterImpl::CopyToNvSurface(const CniFrame &frame) {
    const NvEncInputFrame *encoderInputFrame = mNvEncoderCuda->GetNextInputFrame();
    if (mColorFormat == ColorFormat::BGR || mColorFormat == ColorFormat::RGB) {
        CU_CHECK(cuCtxPushCurrent(mCUcontext));
        cniai_cuda_kernel::imgproc::rgbToBgra(
            (uint8_t *)frame.data(), (uint8_t *)encoderInputFrame->inputPtr,
            mNvEncoderCuda->GetEncodeWidth(), mNvEncoderCuda->GetEncodeHeight(),
            frame.getPitch(), encoderInputFrame->pitch, 255, mStream);
        cuStreamSynchronize(mStream);
        CU_CHECK(cuCtxPopCurrent(nullptr));
    } else if (mColorFormat == ColorFormat::GRAY) {
        const uint32_t chromaHeight = NvEncoder::GetChromaHeight(
            NV_ENC_BUFFER_FORMAT_NV12, mNvEncoderCuda->GetEncodeHeight());
        cudaMemcpy2DAsync(
            encoderInputFrame->inputPtr, encoderInputFrame->pitch, frame.data(),
            frame.getPitch(), mNvEncoderCuda->GetEncodeWidth(),
            mNvEncoderCuda->GetEncodeHeight(), cudaMemcpyDeviceToDevice, mStream);
        cudaMemset2DAsync(
            &((uint8_t *)encoderInputFrame->inputPtr)[encoderInputFrame->pitch *
                                                      mNvEncoderCuda->GetEncodeHeight()],
            encoderInputFrame->pitch, 128, mNvEncoderCuda->GetEncodeWidth(), chromaHeight,
            mStream);
    } else {
        NvEncoderCuda::CopyToDeviceFrame(
            mCUcontext, frame.data(), static_cast<unsigned>(frame.getPitch()),
            (CUdeviceptr)encoderInputFrame->inputPtr, (int)encoderInputFrame->pitch,
            mNvEncoderCuda->GetEncodeWidth(), mNvEncoderCuda->GetEncodeHeight(),
            CU_MEMORYTYPE_DEVICE, encoderInputFrame->bufferFormat,
            encoderInputFrame->chromaOffsets, encoderInputFrame->numChromaPlanes, false,
            mStream);
    }
}

void VideoWriterImpl::write(CniFrame &frame) {
    CopyToNvSurface(frame);
    std::vector<uint64_t> pts;
    mNvEncoderCuda->EncodeFrame(mPackets, pts);
    mEncoderCallback->onEncoded(mPackets, pts);
};

EncoderParams VideoWriterImpl::getEncoderParams() const { return mEncoderParams; };

VideoWriter *createVideoWriter(int deviceId, const std::string &fileName, Codec codec,
                               int width, int height, float fps, ColorFormat colorFormat,
                               const EncoderParams &params) {
    CNIAI_CHECK(params.idrPeriod >= params.gopLength);
    EncoderCallback *encoderCallback =
        new FFmpegVideoWriter(fileName, codec, width, height, fps);
    return new VideoWriterImpl(deviceId, encoderCallback, codec, width, height, fps,
                               colorFormat, params);
}

} // namespace cniai::nvcodec
