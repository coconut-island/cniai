#pragma once

#include <cuda_runtime_api.h>

#include "common/json.h"
#include "common/logging.h"
#include "nvcommon/frame.h"
#include "pipeline/data_struct.h"
#include "pipeline/pipeline_processor.h"
#include "pipeline/simple_pipeline_processor.h"
#include "pipeline/yolov5/yolov5_detect_state.h"

using namespace cniai::tensorrt;
using namespace cniai::pipeline::data_struct;

namespace cniai::pipeline::yolov5_detect {

class Yolov5DetectPipelineProcessor
    : public SimplePipelineProcessor<Yolov5DetectRequest, Yolov5DetectResponse> {
public:
    explicit Yolov5DetectPipelineProcessor() = default;

public:
    void initialize();
};

} // namespace cniai::pipeline::yolov5_detect
