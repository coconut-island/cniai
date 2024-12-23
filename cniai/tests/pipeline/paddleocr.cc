#include <filesystem>
#include <fstream>

#include <gflags/gflags.h>

#include "common/logging.h"
#include "nvjpeg/nvjpeg.h"
#include "pipeline/paddleocr/paddleocr_pipeline_processor.h"

using namespace cniai::tensorrt;
using namespace cniai::pipeline::paddleocr;

namespace fs = std::filesystem;

std::vector<std::string> getAllJpgFiles(const std::string &folder_path) {
    std::vector<std::string> jpgFiles;

    try {
        if (!fs::exists(folder_path) || !fs::is_directory(folder_path)) {
            std::cerr << "Folder does not exist or is not a directory: " << folder_path
                      << std::endl;
            return jpgFiles;
        }

        for (const auto &entry : fs::directory_iterator(folder_path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".jpg") {
                jpgFiles.push_back(entry.path().string());
            }
        }
    } catch (const fs::filesystem_error &e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
    }

    return jpgFiles;
}

DEFINE_string(image_dir, "", "Input image dir.");

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    SET_LOG_LEVEL(LOG_LEVEL_DEBUG);
    PaddleocrPipelineProcessor pipelineProcessor;
    pipelineProcessor.initialize();

    std::vector<std::string> imagePaths = getAllJpgFiles(FLAGS_image_dir);

    std::vector<cniai::CniFrame> frames(imagePaths.size());
    std::vector<PaddleocrRequest *> requests;
    cniai::nvjpeg::NvjpegDecoder nvjpegDecoder;
    std::vector<cniai::id::IdType> ids;
    for (int i = 0; i < imagePaths.size(); ++i) {
        auto &imagePath = imagePaths[i];
        std::ifstream input(imagePath.c_str(),
                            std::ios::in | std::ios::binary | std::ios::ate);
        if (!(input.is_open())) {
            std::cerr << "Cannot open image: " << imagePath << std::endl;
            input.close();
            return EXIT_FAILURE;
        }

        std::streamsize fileSize = input.tellg();
        input.seekg(0, std::ios::beg);

        auto jpegData = std::vector<char>(fileSize);
        if (!input.read(jpegData.data(), fileSize)) {
            std::cerr << "Cannot read from file: " << imagePath << std::endl;
            input.close();
            return EXIT_FAILURE;
        }

        nvjpegDecoder.decode((uint8_t *)jpegData.data(), fileSize, frames[i],
                             NVJPEG_OUTPUT_RGBI);

        auto *paddleocrRequest = new PaddleocrRequest(frames[i]);
        requests.emplace_back(paddleocrRequest);

        auto id = pipelineProcessor.enqueue(paddleocrRequest);
        ids.emplace_back(id);
    }

    for (const auto &id : ids) {
        auto response = pipelineProcessor.waitResponse(id);
        for (const auto &oCRPredictResult : response->oCRPredictResults) {
            std::cout << oCRPredictResult.score << ":" << oCRPredictResult.text
                      << std::endl;
        }
        delete response;
    }

    for (const auto &request : requests) {
        delete request;
    }

    pipelineProcessor.shutdown();
    gflags::ShutDownCommandLineFlags();
    return 0;
}