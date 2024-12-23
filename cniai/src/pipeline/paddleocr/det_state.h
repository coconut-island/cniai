#pragma once

#include "common/json.h"
#include "nvcommon/frame.h"
#include "pipeline/data_struct.h"
#include "postprocess_op.h"
#include "tensorrt/executor_stat.h"

using namespace cniai::tensorrt;
using namespace cniai::pipeline::data_struct;

namespace cniai::pipeline::paddleocr {

struct PaddleocrDetRequest {
    CniFrame frame;

    explicit PaddleocrDetRequest(const CniFrame &frame) : frame(frame){};
};

struct PaddleocrDetResponse {
    PaddleocrDetRequest *request;
    std::vector<PaddleOCR::OCRPredictResult> oCRPredictResults;
};

class PaddleocrDetExecutorState : public ExecutorState {
public:
    explicit PaddleocrDetExecutorState() = default;

public:
    void preprocess(ExecutorStatePreprocessParam *executorStatePreprocessParam) override;
    void
    postprocess(ExecutorStatePostprocessParam *ExecutorStatePostprocessParam) override;

    void initialize() override;

    void initConfig();

private:
    PaddleOCR::DBPostProcessor mPostProcessor;
    json mJsonConfig;
};

void showDetResponse(CniFrame &frame, PaddleocrDetResponse *response);

} // namespace cniai::pipeline::paddleocr
