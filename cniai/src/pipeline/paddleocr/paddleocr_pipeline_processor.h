#pragma once

#include "common/image_util.h"
#include "common/json.h"
#include "nvcommon/frame.h"
#include "pipeline/corotine_pipeline_processor.h"
#include "pipeline/data_struct.h"
#include "postprocess_op.h"
#include "tensorrt/buffers.h"
#include "tensorrt/itensor.h"

#include "cls_state.h"
#include "det_state.h"
#include "rec_state.h"

using namespace cniai::tensorrt;
using namespace cniai::pipeline::data_struct;

namespace cniai::pipeline::paddleocr {

struct PaddleocrRequest {
    CniFrame frame;

    explicit PaddleocrRequest(const CniFrame &frame) : frame(frame){};
};

struct PaddleocrResponse {
    std::vector<PaddleOCR::OCRPredictResult> oCRPredictResults;
};

class PaddleocrPipelineProcessor
    : public CoroutinePipelineProcessor<PaddleocrRequest, PaddleocrResponse> {

public:
    explicit PaddleocrPipelineProcessor() = default;

public:
    void initialize();

    void shutdown();

    CoroutinePipelineGenerator process(
        std::shared_ptr<PipelineTask<PaddleocrRequest, PaddleocrResponse>> task) override;

private:
    std::shared_ptr<Executor> mDetExecutor;
    std::shared_ptr<ExecutorState> mDetExecutorState;

    std::shared_ptr<Executor> mClsExecutor;
    std::shared_ptr<ExecutorState> mClsExecutorState;

    std::shared_ptr<Executor> mRecExecutor;
    std::shared_ptr<ExecutorState> mRecExecutorState;
};

} // namespace cniai::pipeline::paddleocr
