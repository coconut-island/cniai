#pragma once

#include "nvcommon/frame.h"

#include <cuda_runtime_api.h>
#include <string>
#include <vector>

// https://github.com/opencv/opencv_contrib/tree/4.5.3
namespace cniai::nvcodec {

struct Rect {
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;

    bool empty() const { return width <= 0 || height <= 0; }
};

struct Size {
    int width = 0;
    int height = 0;

    bool empty() const { return width <= 0 || height <= 0; }
};

enum Codec {
    H264,
    HEVC,
};

enum class ColorFormat {
    UNDEFINED,
    RGB,
    BGR,
    GRAY,
    NV_NV12,
};

/** @brief Rate Control Modes.
 */
enum EncodeParamsRcMode {
    ENC_PARAMS_RC_CONSTQP = 0x0, //!< Constant QP mode.
    ENC_PARAMS_RC_VBR = 0x1,     //!< Variable bitrate mode.
    ENC_PARAMS_RC_CBR = 0x2      //!< Constant bitrate mode.
};

/** @brief Multi Pass Encoding.
 */
enum EncodeMultiPass {
    ENC_MULTI_PASS_DISABLED = 0x0, //!< Single Pass.
    ENC_TWO_PASS_QUARTER_RESOLUTION =
        0x1, //!< Two Pass encoding is enabled where first Pass is quarter resolution.
    ENC_TWO_PASS_FULL_RESOLUTION =
        0x2, //!< Two Pass encoding is enabled where first Pass is full resolution.
};

/** @brief Supported Encoder Profiles.
 */
enum EncodeProfile {
    ENC_CODEC_PROFILE_AUTOSELECT = 0,
    ENC_H264_PROFILE_BASELINE = 1,
    ENC_H264_PROFILE_MAIN = 2,
    ENC_H264_PROFILE_HIGH = 3,
    ENC_H264_PROFILE_HIGH_444 = 4,
    ENC_H264_PROFILE_STEREO = 5,
    ENC_H264_PROFILE_PROGRESSIVE_HIGH = 6,
    ENC_H264_PROFILE_CONSTRAINED_HIGH = 7,
    ENC_HEVC_PROFILE_MAIN = 8,
    ENC_HEVC_PROFILE_MAIN10 = 9,
    ENC_HEVC_PROFILE_FREXT = 10
};

/** @brief Nvidia Encoding Presets. Performance degrades and quality improves as we move
 * from P1 to P7.
 */
enum EncodePreset {
    ENC_PRESET_P1 = 1,
    ENC_PRESET_P2 = 2,
    ENC_PRESET_P3 = 3,
    ENC_PRESET_P4 = 4,
    ENC_PRESET_P5 = 5,
    ENC_PRESET_P6 = 6,
    ENC_PRESET_P7 = 7
};

/** @brief Tuning information.
 */
enum EncodeTuningInfo {
    ENC_TUNING_INFO_UNDEFINED = 0, //!< Undefined tuningInfo. Invalid value for encoding.
    ENC_TUNING_INFO_HIGH_QUALITY = 1, //!< Tune presets for latency tolerant encoding.
    ENC_TUNING_INFO_LOW_LATENCY = 2,  //!< Tune presets for low latency streaming.
    ENC_TUNING_INFO_ULTRA_LOW_LATENCY =
        3,                        //!< Tune presets for ultra low latency streaming.
    ENC_TUNING_INFO_LOSSLESS = 4, //!< Tune presets for lossless encoding.
    ENC_TUNING_INFO_COUNT
};

/** Quantization Parameter for each type of frame when using
 * ENC_PARAMS_RC_MODE::ENC_PARAMS_RC_CONSTQP.
 */
struct EncodeQp {
    uint32_t qpInterP; //!< Specifies QP value for P-frame.
    uint32_t qpInterB; //!< Specifies QP value for B-frame.
    uint32_t qpIntra;  //!< Specifies QP value for Intra Frame.
};

/** @brief Different parameters for CUDA video encoder.
 */
struct EncoderParams {
public:
    EncoderParams()
        : nvPreset(ENC_PRESET_P3), tuningInfo(ENC_TUNING_INFO_HIGH_QUALITY),
          encodingProfile(ENC_CODEC_PROFILE_AUTOSELECT),
          rateControlMode(ENC_PARAMS_RC_VBR), multiPassEncoding(ENC_MULTI_PASS_DISABLED),
          constQp({0, 0, 0}), averageBitRate(0), maxBitRate(0), targetQuality(30),
          gopLength(250), idrPeriod(250){};
    EncodePreset nvPreset;
    EncodeTuningInfo tuningInfo;
    EncodeProfile encodingProfile;
    EncodeParamsRcMode rateControlMode;
    EncodeMultiPass multiPassEncoding;
    EncodeQp constQp;      //!< QP's for \ref ENC_PARAMS_RC_CONSTQP.
    int averageBitRate;    //!< target bitrate for \ref ENC_PARAMS_RC_VBR and \ref
                           //!< ENC_PARAMS_RC_CBR.
    int maxBitRate;        //!< upper bound on bitrate for \ref ENC_PARAMS_RC_VBR and \ref
                           //!< ENC_PARAMS_RC_CONSTQP.
    uint8_t targetQuality; //!< value 0 - 51 where video quality decreases as
                           //!< targetQuality increases, used with \ref ENC_PARAMS_RC_VBR.
    int gopLength; //!< the number of pictures in one GOP, ensuring \ref idrPeriod >= \ref
                   //!< gopLength.
    int idrPeriod; //!< IDR interval, ensuring \ref idrPeriod >= \ref gopLength.
};

bool operator==(const EncoderParams &lhs, const EncoderParams &rhs);

class EncoderCallback {
public:
    virtual void onEncoded(const std::vector<std::vector<uint8_t>> &vPacket,
                           const std::vector<uint64_t> &pts) = 0;

    virtual void onEncodingFinished() = 0;

    virtual ~EncoderCallback() = default;
};

class VideoWriter {
public:
    virtual ~VideoWriter() = default;

    virtual void write(CniFrame &frame) = 0;

    virtual EncoderParams getEncoderParams() const = 0;
};

VideoWriter *createVideoWriter(int deviceId, const std::string &fileName, Codec codec,
                               int width, int height, float fps, ColorFormat colorFormat,
                               const EncoderParams &params);

/////////// Video Decoding //////////////////

/** @brief Chroma formats supported by cudacodec::VideoReader.
 */
enum ChromaFormat { Monochrome = 0, YUV420, YUV422, YUV444, NumFormats };

/** @brief Deinterlacing mode used by decoder.
 * @param Weave Weave both fields (no deinterlacing). For progressive content and for
 * content that doesn't need deinterlacing.
 * @param Bob Drop one field.
 * @param Adaptive Adaptive deinterlacing needs more video memory than other deinterlacing
 * modes.
 * */
enum DeinterlaceMode { Weave = 0, Bob = 1, Adaptive = 2 };

/** @brief Struct providing information about video file format. :
 */
struct FormatInfo {
    FormatInfo()
        : nBitDepthMinus8(-1), ulWidth(0), ulHeight(0), width(0), height(0),
          ulMaxWidth(0), ulMaxHeight(0), fps(0), ulNumDecodeSurfaces(0),
          nCounterBitDepth(0), nMaxHistogramBins(0){};

    Codec codec;
    ChromaFormat chromaFormat;
    int nBitDepthMinus8;
    int nBitDepthChromaMinus8{};
    int ulWidth;  //!< Coded sequence width in pixels.
    int ulHeight; //!< Coded sequence height in pixels.
    int width;    //!< Width of the decoded frame returned by nextFrame(frame).
    int height;   //!< Height of the decoded frame returned by nextFrame(frame).
    int ulMaxWidth;
    int ulMaxHeight;
    Rect displayArea; //!< ROI inside the decoded frame returned by nextFrame(frame),
                      //!< containing the useable video frame.
    double fps;
    int ulNumDecodeSurfaces; //!< Maximum number of internal decode surfaces.
    DeinterlaceMode deinterlaceMode;
    Size targetSz;         //!< Post-processed size of the output frame.
    Rect srcRoi;           //!< Region of interest decoded from video source.
    Rect targetRoi;        //!< Region of interest in the output frame containing
                           //!< the decoded frame.
    int nCounterBitDepth;  //!< Bit depth of histogram bins if histogram output
                           //!< is requested and supported.
    int nMaxHistogramBins; //!< Max number of histogram bins if histogram
                           //!< output is requested and supported.
};

class VideoReader {
public:
    virtual ~VideoReader() = default;

    virtual bool nextFrame(CniFrame &frame) = 0;

    virtual Codec getCodec() = 0;

    virtual int getTargetWidth() = 0;

    virtual int getTargetHeight() = 0;

    virtual bool setColorFormat(ColorFormat colorFormat) = 0;

    virtual int getPosFrames() const = 0;
};

/** @brief VideoReader initialization parameters
@param allowFrameDrop Allow frames to be dropped when ingesting from a live capture source
to prevent delay and eventual disconnection when calls to nextFrame()/grab() cannot keep
up with the source's fps.  Only use if delay and disconnection are a problem, i.e. not
when decoding from video files where setting this flag will cause frames to be
unnecessarily discarded.
@param minNumDecodeSurfaces Minimum number of internal decode surfaces used by the
hardware decoder.  NVDEC will automatically determine the minimum number of surfaces it
requires for correct functionality and optimal video memory usage but not necessarily for
best performance, which depends on the design of the overall application. The optimal
number of decode surfaces (in terms of performance and memory utilization) should be
decided by experimentation for each application, but it cannot go below the number
determined by NVDEC.
@param targetSz Post-processed size (width/height should be multiples of 2) of the output
frame, defaults to the size of the encoded video source.
@param srcRoi Region of interest (x/width should be multiples of 4 and y/height multiples
of 2) decoded from video source, defaults to the full frame.
@param targetRoi Region of interest (x/width should be multiples of 4 and y/height
multiples of 2) within the output frame to copy and resize the decoded frame to, defaults
to the full frame.
*/
struct VideoReaderInitParams {
    VideoReaderInitParams() : allowFrameDrop(false), minNumDecodeSurfaces(0){};
    bool allowFrameDrop;
    int minNumDecodeSurfaces;
    Size targetSz;
    Rect srcRoi;
    Rect targetRoi;
};

VideoReader *createVideoReader(int deviceId, const std::string &filename,
                               VideoReaderInitParams params = VideoReaderInitParams());

} // namespace cniai::nvcodec
