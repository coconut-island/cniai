#pragma once

#include "common/json.h"
#include "nvcommon/frame.h"
#include "pipeline/data_struct.h"
#include "postprocess_op.h"
#include "tensorrt/executor_stat.h"

using namespace cniai::tensorrt;
using namespace cniai::pipeline::data_struct;

namespace cniai::pipeline::paddleocr {

struct PaddleocrClsRequest {
    CniFrame frame;
    PaddleOCR::OCRPredictResult ocrPredictResult;

    explicit PaddleocrClsRequest(const CniFrame &frame,
                                 PaddleOCR::OCRPredictResult &ocrPredictResult)
        : frame(frame), ocrPredictResult(ocrPredictResult){};
};

struct PaddleocrClsResponse {
    PaddleOCR::OCRPredictResult oCRPredictResult;
    int classLabel = 0;
    float clsScores = 0;
};

class PaddleocrClsExecutorState : public ExecutorState {
public:
    explicit PaddleocrClsExecutorState() = default;

public:
    void preprocess(ExecutorStatePreprocessParam *executorStatePreprocessParam) override;
    void
    postprocess(ExecutorStatePostprocessParam *ExecutorStatePostprocessParam) override;

    void initialize() override;

    void initConfig();

private:
    json mJsonConfig;
};

} // namespace cniai::pipeline::paddleocr
