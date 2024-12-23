#include <chrono>
#include <fstream>
#include <iostream>

#include "common/image_util.h"
#include "common/logging.h"
#include "nvjpeg/nvjpeg.h"

#include <gflags/gflags.h>

DEFINE_string(log_level, LOG_LEVEL_DEBUG,
              "Log level, includes [trace, debug, info, warn, err, critical, off]");
DEFINE_string(log_pattern, "[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%t] [%@] %v", "Log pattern");
DEFINE_string(jpeg_path, "resources/yaomeifeng_wallpaper_yuv420p_1920_1080.jpg",
              "Input jpeg path.");
DEFINE_string(output_rgbi_bmp_path,
              "./yaomeifeng_wallpaper_yuv420p_1920_1080_out_rgbi.bmp",
              "Output rgbi bmp path.");
DEFINE_string(output_yuv_bmp_path, "./yaomeifeng_wallpaper_yuv420p_1920_1080_out_yuv.bmp",
              "Output yuv bmp path.");

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    SET_LOG_PATTERN(FLAGS_log_pattern);
    SET_LOG_LEVEL(FLAGS_log_level);

    std::ifstream input(FLAGS_jpeg_path.c_str(),
                        std::ios::in | std::ios::binary | std::ios::ate);
    if (!(input.is_open())) {
        std::cerr << "Cannot open image: " << FLAGS_jpeg_path << std::endl;
        input.close();
        return EXIT_FAILURE;
    }

    std::streamsize fileSize = input.tellg();
    input.seekg(0, std::ios::beg);

    auto jpegData = std::vector<char>(fileSize);
    if (!input.read(jpegData.data(), fileSize)) {
        std::cerr << "Cannot read from file: " << FLAGS_jpeg_path << std::endl;
        input.close();
        return EXIT_FAILURE;
    }

    cniai::CniFrame frame;
    cniai::nvjpeg::NvjpegDecoder nvjpegDecoder;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 100; ++i) {
        auto loop_start_time = std::chrono::high_resolution_clock::now();

        nvjpegDecoder.decode(jpegData.data(), fileSize, frame, NVJPEG_OUTPUT_RGBI);

        auto loop_end_time = std::chrono::high_resolution_clock::now();
        auto loop_duration =
            duration_cast<std::chrono::microseconds>(loop_end_time - loop_start_time);
        std::cout << "Time for iteration " << i + 1 << ": " << loop_duration.count()
                  << " microseconds" << std::endl;
    }

    // 记录结束时间
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Total time for 100 iterations: " << total_duration.count()
              << " milliseconds" << std::endl;

    void *rgbiHostPtr;
    cudaMallocHost(&rgbiHostPtr, frame.size());
    cudaMemcpy(rgbiHostPtr, frame.data(), frame.size(), cudaMemcpyDeviceToHost);
    cniai::image_util::writeImage(
        FLAGS_output_rgbi_bmp_path.c_str(), (unsigned char *)rgbiHostPtr,
        frame.getWidth(), frame.getHeight(), cniai::image_util::ImageFormat::RGBI);
    cudaFreeHost(rgbiHostPtr);

    gflags::ShutDownCommandLineFlags();
    return 0;
}