#pragma once

#include "common/json.h"
#include "nvcommon/frame.h"
#include "pipeline/data_struct.h"
#include "postprocess_op.h"
#include "tensorrt/executor_stat.h"

using namespace cniai::tensorrt;
using namespace cniai::pipeline::data_struct;

namespace cniai::pipeline::paddleocr {

struct PaddleocrRecRequest {
    CniFrame frame;

    PaddleOCR::OCRPredictResult ocrPredictResult;

    explicit PaddleocrRecRequest(const CniFrame &frame,
                                 PaddleOCR::OCRPredictResult &ocrPredictResult)
        : frame(frame), ocrPredictResult(ocrPredictResult){};
};

struct PaddleocrRecResponse {
    //    PaddleocrDetRequest *request;
    PaddleOCR::OCRPredictResult oCRPredictResult;
};

class PaddleocrRecExecutorState : public ExecutorState {
public:
    explicit PaddleocrRecExecutorState() = default;

public:
    void preprocess(ExecutorStatePreprocessParam *executorStatePreprocessParam) override;
    void
    postprocess(ExecutorStatePostprocessParam *executorStatePostprocessParam) override;

    void initialize() override;

    void initConfig();

private:
    json mJsonConfig;
    std::vector<std::string> mLabelList;
};

} // namespace cniai::pipeline::paddleocr
