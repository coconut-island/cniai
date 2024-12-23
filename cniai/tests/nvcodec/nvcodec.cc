#include "nvcodec/nvcodec.h"

#include "common/logging.h"

#include <gflags/gflags.h>

DEFINE_string(log_level, LOG_LEVEL_DEBUG,
              "Log level, includes [trace, debug, info, warn, err, critical, off]");
DEFINE_string(read_stream_path, "rtsp://localhost:8554/mystream1", "Input stream path.");
DEFINE_string(write_stream_path, "rtsp://localhost:8554/mystream", "Output stream path.");

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    SET_LOG_LEVEL(FLAGS_log_level);

    cniai::nvcodec::VideoReaderInitParams videoReaderInitParams;
    cniai::nvcodec::VideoReader *videoReader = cniai::nvcodec::createVideoReader(
        0, FLAGS_read_stream_path, videoReaderInitParams);
    videoReader->setColorFormat(cniai::nvcodec::ColorFormat::NV_NV12);

    cniai::nvcodec::EncoderParams encoderParams;
    auto *videoWriter = cniai::nvcodec::createVideoWriter(
        0, FLAGS_write_stream_path, cniai::nvcodec::Codec::H264,
        videoReader->getTargetWidth(), videoReader->getTargetHeight(), 25,
        cniai::nvcodec::ColorFormat::NV_NV12, encoderParams);

    int i = 0;
    cniai::CniFrame frame;
    for (;;) {
        if (!videoReader->nextFrame(frame)) {
            break;
        }

        std::cout << i << std::endl;
        i++;
        videoWriter->write(frame);
    }

    delete videoReader;
    delete videoWriter;

    gflags::ShutDownCommandLineFlags();
    return 0;
}