#include "ffmpeg_muxer.h"

#include "common/logging.h"

namespace cniai::nvcodec {

cniai::nvcodec::FFmpegMuxer::FFmpegMuxer(const std::string &filename, AVCodecID avCodecId,
                                         int width, int height, float fps) {
    mFilename = filename;

    mCodecId = avCodecId;

    mWidth = width;
    mHeight = height;
    mFps = fps;
}

FFmpegMuxer::~FFmpegMuxer() { destroy(); }

bool FFmpegMuxer::init(const std::string &filename, std::string extraData) {
    avformat_network_init();

    char *formatName = nullptr;
    if (filename.starts_with("rtsp")) {
        formatName = strdup("rtsp");
    }

    int ret = avformat_alloc_output_context2(&mAvformatContext, nullptr, formatName,
                                             filename.data());
    if (ret < 0) {
        LOG_ERROR("FFmpeg: failed to allocate an AVFormatContext. Error message: {}",
                  avErrorToString(ret));
        return false;
    }

    mAvformatContext->url = av_strdup(filename.c_str());

    //    m_avformat_context->oformat->flags |= AVFMT_ALLOW_FLUSH;
    mVideoStream = avformat_new_stream(mAvformatContext, nullptr);
    if (mVideoStream == nullptr) {
        LOG_ERROR("alloc video stream err");
        return false;
    }

    AVCodecParameters *vCodecPar = mVideoStream->codecpar;
    vCodecPar->format = AV_PIX_FMT_YUV420P;
    vCodecPar->width = mWidth;
    vCodecPar->height = mHeight;
    vCodecPar->codec_type = AVMEDIA_TYPE_VIDEO;
    vCodecPar->codec_id = mCodecId;
    vCodecPar->codec_tag =
        av_codec_get_tag(mAvformatContext->oformat->codec_tag, mCodecId);

    // store sps, pps
    if (!extraData.empty()) {
        vCodecPar->extradata_size = extraData.size();
        vCodecPar->extradata = (uint8_t *)av_mallocz(vCodecPar->extradata_size +
                                                     AV_INPUT_BUFFER_PADDING_SIZE);
        memcpy(vCodecPar->extradata, extraData.data(), extraData.size());
    }

    if (!(mAvformatContext->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open2(&mAvformatContext->pb, filename.c_str(), AVIO_FLAG_WRITE,
                         nullptr, nullptr);
        if (ret < 0) {
            LOG_ERROR("create oformat failed: ", avErrorToString(ret));
            if (!(mAvformatContext->oformat->flags & AVFMT_NOFILE)) {
                avio_close(mAvformatContext->pb);
            }
            avformat_free_context(mAvformatContext);
            return false;
        }
    }

    ret = avformat_write_header(mAvformatContext, nullptr);
    if (ret < 0) {
        LOG_ERROR("create header failed: ", avErrorToString(ret));
        return false;
    }
    av_dump_format(mAvformatContext, 0, filename.data(), 1);

    return true;
}

bool FFmpegMuxer::write(uint8_t *nalu, int naluLen, int pts) {
    if (nalu == nullptr) {
        LOG_ERROR("nalu is null");
        return false;
    }

    if (mFrameCount == 0) {
        bool is_sps_or_pps =
            (((nalu[4] & 0x1f) == 7) || ((nalu[4] & 0x1f) == 8)); // sps or pps

        if (is_sps_or_pps) {
            mExtractData.resize(naluLen, 0);
            memcpy((void *)mExtractData.data(), nalu, naluLen);
        }
    }

    if (!mSuccessInited) {
        mSuccessInited = init(mFilename, mExtractData);
    }

    if (!mSuccessInited) {
        LOG_ERROR("Failed to open input--> {}", mFilename);
        return false;
    }

    AVPacket *pkt = av_packet_alloc();

    bool keyFrame = false;

    if (mCodecId == AV_CODEC_ID_H264)
        keyFrame = ((nalu[4] & 0x1f) == 5);
    else if (mCodecId == AV_CODEC_ID_HEVC)
        keyFrame = (((nalu[4] & 0x7E) >> 1) == 5);
    else
        LOG_ERROR("failed to detect nalu type");

    if (keyFrame) {
        pkt->flags |= AV_PKT_FLAG_KEY;
    }
    pkt->data = (uint8_t *)nalu;
    pkt->size = naluLen;

    pkt->stream_index = mVideoStream->index;
    pkt->pos = -1;

    static AVRational rational = {1000, (int)(1000 * mFps)};
    if (pts == -1) {
        pkt->pts = av_rescale_q(mFrameCount, rational, mVideoStream->time_base);
    } else {
        pkt->pts = av_rescale_q(pts++, rational, mVideoStream->time_base);
    }
    pkt->duration = av_rescale_q(1, rational, mVideoStream->time_base);
    pkt->dts = pkt->pts;

    LOG_DEBUG("PTS: {}, DTS: {}, Duration: {}", pkt->pts, pkt->dts, pkt->duration);

    mFrameCount++;

    mAvformatContext->duration = pkt->duration;
    int ret = av_interleaved_write_frame(mAvformatContext, pkt);

    if (ret < 0) {
        LOG_ERROR("write video frame err: ", avErrorToString(ret));
    }
    av_packet_free(&pkt);
    return ret == 0;
}

void FFmpegMuxer::destroy() {
    if (mAvformatContext) {
        int ret = av_write_trailer(mAvformatContext);
        if (ret != 0) {
            LOG_ERROR("write trailer failed: ", avErrorToString(ret));
        }
        if (!(mAvformatContext->oformat->flags & AVFMT_NOFILE)) {
            avio_close(mAvformatContext->pb);
        }
        avformat_free_context(mAvformatContext);

        mAvformatContext = nullptr;
        mVideoStream = nullptr;
        mFrameCount = 0;
        mWidth = 0;
        mHeight = 0;
    }
}

} // namespace cniai::nvcodec