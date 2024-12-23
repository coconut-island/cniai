#pragma once

#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

#include "common/json.h"
#include "common/logging.h"
#include "nvcommon/frame.h"
#include "pipeline/data_struct.h"
#include "tensorrt/executor_stat.h"

using namespace cniai::tensorrt;
using namespace cniai::pipeline::data_struct;

namespace cniai::pipeline::yolov5_detect {

using DetectionBox = DetectionBox;

struct Yolov5DetectRequest {

    CniFrame frame;

    explicit Yolov5DetectRequest(const CniFrame &frame) : frame(frame){};
};

struct Yolov5DetectResponse {
    std::vector<DetectionBox> detectBoxes;
};

class Yolov5DetectExecutorState : public ExecutorState {
public:
    explicit Yolov5DetectExecutorState() = default;

public:
    void preprocess(ExecutorStatePreprocessParam *executorStatePreprocessParam) override;
    void
    postprocess(ExecutorStatePostprocessParam *ExecutorStatePostprocessParam) override;

    void initialize() override;

    void initConfig();

private:
    json mJsonConfig;
};

} // namespace cniai::pipeline::yolov5_detect
