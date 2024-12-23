#pragma once

#include "pipeline_processor.h"
#include "tensorrt/executor.h"
#include "tensorrt/executor_stat.h"

using namespace cniai::tensorrt;
using namespace cniai::pipeline::data_struct;

namespace cniai::pipeline {

template <class PipelineRequest, class PipelineResponse>
class SimplePipelineProcessor {
public:
    explicit SimplePipelineProcessor() = default;

    ~SimplePipelineProcessor() { shutdown(); };

    cniai::id::IdType enqueue(PipelineRequest *request) {
        return mExecutor->enqueueRequest(reinterpret_cast<ExecutorRequest *>(request));
    };

    PipelineResponse *waitResponse(cniai::id::IdType const &requestId) {
        return reinterpret_cast<PipelineResponse *>(mExecutor->awaitResponse(requestId));
    };

    bool hasResponse(const cniai::id::IdType &requestId) {
        return mExecutor->hasResponse(requestId);
    }

    void shutdown() { mExecutor->shutdown(); }

protected:
    std::shared_ptr<Executor> mExecutor;
    std::shared_ptr<ExecutorState> mExecutorState;
};

} // namespace cniai::pipeline