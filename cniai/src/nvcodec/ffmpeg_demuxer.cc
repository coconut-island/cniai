#include "ffmpeg_demuxer.h"

#include "common/logging.h"

namespace cniai::nvcodec {

inline static std::string ffmpegGetErrorString(int errorCode) {
    char buf[255] = {0};
    const int err = av_strerror(errorCode, buf, 254);
    if (err == 0)
        return buf;
    else
        return "Unknown error";
}

static inline bool h26xContainer(const char *formatLongName) {
    return !strcmp(formatLongName, "QuickTime / MOV") ||
           !strcmp(formatLongName, "FLV (Flash Video)") ||
           !strcmp(formatLongName, "Matroska / WebM");
}

FFmpegDemuxer::FFmpegDemuxer() {
    //    avdevice_register_all();
    memset(&mPacket, 0, sizeof(mPacket));
    av_init_packet(&mPacket);

    memset(&mPacketFiltered, 0, sizeof(mPacketFiltered));
    av_init_packet(&mPacketFiltered);
}

FFmpegDemuxer::~FFmpegDemuxer() { destroy(); }

void FFmpegDemuxer::destroy() {
    if (mAvSteam) {
        mAvSteam = nullptr;
    }

    if (mAvFormatContext) {
        avformat_close_input(&mAvFormatContext);
        mAvFormatContext = nullptr;
    }

    // free last packet if exist
    if (mPacket.data) {
        av_packet_unref(&mPacket);
        mPacket.data = nullptr;
    }

    if (mAVDictionary != nullptr)
        av_dict_free(&mAVDictionary);

    if (mPacketFiltered.data) {
        av_packet_unref(&mPacketFiltered);
        mPacketFiltered.data = nullptr;
    }

    if (mAVBSFContext) {
        av_bsf_free(&mAVBSFContext);
    }
}

bool FFmpegDemuxer::open(const std::string &filename) {
    destroy();

    mFilename = filename;

    mAvFormatContext = avformat_alloc_context();

    av_dict_set(&mAVDictionary, "rtsp_flags", "prefer_tcp", 0);
    av_dict_set(&mAVDictionary, "rtsp_transport", "tcp", 0);

    const AVInputFormat *input_format = nullptr;
    AVDictionaryEntry *entry = av_dict_get(mAVDictionary, "input_format", nullptr, 0);
    if (entry != nullptr) {
        input_format = av_find_input_format(entry->value);
    }

    int err = avformat_open_input(&mAvFormatContext, mFilename.c_str(), input_format,
                                  &mAVDictionary);

    if (err < 0) {
        LOG_WARN("Error opening file");
        LOG_WARN(mFilename);
        goto exit_func;
    }
    err = avformat_find_stream_info(mAvFormatContext, nullptr);
    if (err < 0) {
        LOG_WARN("Unable to read codec parameters from stream ({})",
                 ffmpegGetErrorString(err));
        goto exit_func;
    }

    mVideoStream =
        av_find_best_stream(mAvFormatContext, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (mVideoStream < 0) {
        LOG_WARN("Unable to find codec parameters from stream ({})");
        goto exit_func;
    }
    mAvSteam = mAvFormatContext->streams[mVideoStream];
    return true;

exit_func:

    destroy();

    return false;
}

bool FFmpegDemuxer::retrieve(unsigned char **data, int *size) {
    bool valid = false;

    static const size_t max_read_attempts = 4096;
    size_t cur_read_attempts = 0;

    if (!mAvFormatContext || !mAvSteam)
        return false;

    // get the next frame
    while (true) {

        av_packet_unref(&mPacket);

        int ret = av_read_frame(mAvFormatContext, &mPacket);

        if (ret == AVERROR(EAGAIN))
            continue;

        if (ret == AVERROR_EOF) {
            break;
        }

        if (mPacket.stream_index != mVideoStream) {
            av_packet_unref(&mPacket);
            if (++cur_read_attempts > max_read_attempts) {
                LOG_WARN("packet read max attempts exceeded, if your video have multiple "
                         "streams (video, audio) try to increase attempt "
                         "limit by setting environment variable "
                         "OPENCV_FFMPEG_READ_ATTEMPTS "
                         "(current value is {})",
                         max_read_attempts);
                break;
            }
            continue;
        }

        valid = processRawPacket();
        break;
    }

    if (!mAvSteam)
        return false;

    if (valid) {
        AVPacket &p = mAVBSFContext ? mPacketFiltered : mPacket;
        *data = p.data;
        *size = p.size;
        valid = p.data != nullptr;
    }

    // return if we have a new frame or not
    return valid;
}

bool FFmpegDemuxer::getExtractData(unsigned char **data, int *size) {
    *data = mAvFormatContext->streams[mVideoStream]->codecpar->extradata;
    *size = mAvFormatContext->streams[mVideoStream]->codecpar->extradata_size;
    return true;
}

bool FFmpegDemuxer::processRawPacket() {
    if (mPacket.data == nullptr) // EOF
        return false;

    AVCodecID eVideoCodec = mAvFormatContext->streams[mVideoStream]->codecpar->codec_id;
    const char *filterName = nullptr;
    if (eVideoCodec == AV_CODEC_ID_H264 || eVideoCodec == AV_CODEC_ID_HEVC) {
        if (h26xContainer(mAvFormatContext->iformat->long_name))
            filterName =
                eVideoCodec == AV_CODEC_ID_H264 ? "h264_mp4toannexb" : "hevc_mp4toannexb";
    }

    if (filterName) {
        const AVBitStreamFilter *bsf = av_bsf_get_by_name(filterName);
        if (!bsf) {
            LOG_WARN("Bitstream filter is not available: {}", filterName);
            return false;
        }
        int err = av_bsf_alloc(bsf, &mAVBSFContext);
        if (err < 0) {
            LOG_WARN("Error allocating context for bitstream buffer");
            return false;
        }
        avcodec_parameters_copy(mAVBSFContext->par_in,
                                mAvFormatContext->streams[mVideoStream]->codecpar);
        err = av_bsf_init(mAVBSFContext);
        if (err < 0) {
            LOG_WARN("Error initializing bitstream buffer");
            return false;
        }
    }

    if (mAVBSFContext) {
        if (mPacketFiltered.data) {
            av_packet_unref(&mPacketFiltered);
        }

        int err = av_bsf_send_packet(mAVBSFContext, &mPacket);
        if (err < 0) {
            LOG_WARN("Packet submission for filtering failed");
            return false;
        }
        err = av_bsf_receive_packet(mAVBSFContext, &mPacketFiltered);
        if (err < 0) {
            LOG_WARN("Filtered packet retrieve failed");
            return false;
        }

        return mPacketFiltered.data != nullptr;
    }

    return mPacket.data != nullptr;
}

int64_t FFmpegDemuxer::getTotalFrames() const {
    int64_t nbf = mAvFormatContext->streams[mVideoStream]->nb_frames;

    if (nbf == 0) {
        nbf = (int64_t)floor(getDurationSec() * getFps() + 0.5);
    }
    return nbf;
}

double FFmpegDemuxer::getDurationSec() const {
    double sec = (double)mAvFormatContext->duration / (double)AV_TIME_BASE;

    if (sec < mEpsZero) {
        sec = (double)mAvFormatContext->streams[mVideoStream]->duration *
              r2d(mAvFormatContext->streams[mVideoStream]->time_base);
    }

    return sec;
}

double FFmpegDemuxer::getFps() const {
    double fps = r2d(mAvFormatContext->streams[mVideoStream]->avg_frame_rate);

    if (fps < mEpsZero) {
        fps = r2d(av_guess_frame_rate(mAvFormatContext,
                                      mAvFormatContext->streams[mVideoStream], nullptr));
    }

    if (fps < mEpsZero) {
        fps = 1.0 / r2d(mAvFormatContext->streams[mVideoStream]->time_base);
    }

    return fps;
}

int64_t FFmpegDemuxer::getBitrate() const { return mAvFormatContext->bit_rate / 1000; }

double FFmpegDemuxer::r2d(AVRational r) const {
    return r.num == 0 || r.den == 0 ? 0. : (double)r.num / (double)r.den;
}

int64_t FFmpegDemuxer::dtsToFrameNumber(int64_t dts) const {
    double sec = dtsToSec(dts);
    return (int64_t)(getFps() * sec + 0.5f);
}

double FFmpegDemuxer::dtsToSec(int64_t dts) const {
    return (double)(dts - mAvFormatContext->streams[mVideoStream]->start_time) *
           r2d(mAvFormatContext->streams[mVideoStream]->time_base);
}

AVCodecID FFmpegDemuxer::getCodecId() { return mAvSteam->codecpar->codec_id; }

AVPixelFormat FFmpegDemuxer::getPixelFormat() {
    return (AVPixelFormat)mAvSteam->codecpar->format;
}

int FFmpegDemuxer::getFrameWidth() { return mAvSteam->codecpar->width; }

int FFmpegDemuxer::getFrameHeight() { return mAvSteam->codecpar->height; }

bool FFmpegDemuxer::getLastPacketContainsKeyFrame() {
    const AVPacket &p = mAVBSFContext ? mPacketFiltered : mPacket;
    return (p.flags & AV_PKT_FLAG_KEY) != 0;
}

} // namespace cniai::nvcodec