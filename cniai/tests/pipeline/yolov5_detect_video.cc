#include <chrono>
#include <fstream>

#include <cniai_cuda_kernel/cuosd.h>
#include <gflags/gflags.h>

#include "common/logging.h"
#include "nvcodec/nvcodec.h"
#include "nvcommon/frame.h"
#include "pipeline/yolov5/yolov5_detect_pipeline_processor.h"

using namespace cniai::tensorrt;
using namespace cniai::pipeline::yolov5_detect;

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
    CNIAI_CHECK(videoReader->setColorFormat(cniai::nvcodec::ColorFormat::NV_NV12));

    cniai::nvcodec::EncoderParams encoderParams;
    auto *videoWriter = cniai::nvcodec::createVideoWriter(
        0, FLAGS_write_stream_path, cniai::nvcodec::Codec::H264,
        videoReader->getTargetWidth(), videoReader->getTargetHeight(), 25,
        cniai::nvcodec::ColorFormat::NV_NV12, encoderParams);

    Yolov5DetectPipelineProcessor pipelineProcessor;
    pipelineProcessor.initialize();

    auto context = cniai_cuda_kernel::cuosd::cuosd_context_create();
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    int i = 0;
    cniai::CniFrame frame;
    auto tstart = std::chrono::high_resolution_clock::now();
    for (;;) {
        auto start = std::chrono::high_resolution_clock::now();
        if (!videoReader->nextFrame(frame)) {
            break;
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "read frame cost: " << duration.count() << " ms" << std::endl;

        start = std::chrono::high_resolution_clock::now();
        auto request = new Yolov5DetectRequest(frame);
        auto id = pipelineProcessor.enqueue(request);
        auto response = pipelineProcessor.waitResponse(id);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "infer frame cost: " << duration.count() << " ms" << std::endl;

        start = std::chrono::high_resolution_clock::now();
        for (const auto &box : response->detectBoxes) {
            std::string label = box.mClassName + ": " + std::to_string(box.mScore);

            cniai_cuda_kernel::cuosd::cuosd_draw_rectangle(
                context, box.mX, box.mY, box.mX + box.mW, box.mY + box.mH, 3,
                {0, 255, 0, 255}, {0, 0, 0, 0});

            cuosd_draw_text(context, label.c_str(), 13, "resource/simhei.ttf", box.mX,
                            box.mY - 5,
                            cniai_cuda_kernel::cuosd::cuOSDColor{0, 255, 0, 255},
                            cniai_cuda_kernel::cuosd::cuOSDColor{0, 0, 0, 0});
        }

        cniai_cuda_kernel::cuosd::cuosd_apply(
            context, frame.data(),
            ((uint8_t *)frame.data()) + (frame.getPitch() * frame.getHeight()),
            frame.getWidth(), frame.getPitch(), frame.getHeight(),
            cniai_cuda_kernel::cuosd::cuOSDImageFormat::PitchLinearNV12, stream, true);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "draw frame cost: " << duration.count() << " ms" << std::endl;

        start = std::chrono::high_resolution_clock::now();
        videoWriter->write(frame);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "write frame cost: " << duration.count() << " ms" << std::endl;

        delete response;
        delete request;

        std::cout << i << std::endl;
        i++;

        auto tend = std::chrono::high_resolution_clock::now();
        auto tduration =
            std::chrono::duration_cast<std::chrono::milliseconds>(tend - tstart);
        std::cout << "fps: " << i / (tduration.count() / 1000) << "" << std::endl;
    }

    delete videoReader;
    delete videoWriter;
    cuosd_context_destroy(context);
    CUDA_CHECK(cudaStreamDestroy(stream));

    gflags::ShutDownCommandLineFlags();
    return 0;
}