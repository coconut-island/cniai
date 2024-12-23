#include "paddleocr_pipeline_processor.h"

#include <memory>

#include "common/id.h"
#include "common/logging.h"

namespace cniai::pipeline::paddleocr {

void PaddleocrPipelineProcessor::initialize() {
    mDetExecutorState = std::make_shared<PaddleocrDetExecutorState>();
    ExecutorConfig executorDetConfig;
    executorDetConfig.mDeviceId = 0;
    executorDetConfig.mEnginePath = "./models/ch_PP-OCRv4_det_infer.trt";
    ExecutorWorkerConfig executorDetWorkerConfig;
    executorDetWorkerConfig.mProfileIndex = 0;
    executorDetConfig.mExecutorWorkerConfigs.emplace_back(executorDetWorkerConfig);
    mDetExecutor = std::make_shared<Executor>(executorDetConfig, mDetExecutorState);

    mDetExecutorState->initialize();

    mClsExecutorState = std::make_shared<PaddleocrClsExecutorState>();
    ExecutorConfig executorClsConfig;
    executorClsConfig.mDeviceId = 0;
    executorClsConfig.mEnginePath = "./models/ch_ppocr_mobile_v2.0_cls_infer.trt";
    ExecutorWorkerConfig executorClsWorkerConfig;
    executorClsWorkerConfig.mProfileIndex = 0;
    executorClsConfig.mExecutorWorkerConfigs.emplace_back(executorClsWorkerConfig);
    mClsExecutor = std::make_shared<Executor>(executorClsConfig, mClsExecutorState);

    mClsExecutorState->initialize();

    mRecExecutorState = std::make_shared<PaddleocrRecExecutorState>();
    ExecutorConfig executorRecConfig;
    executorRecConfig.mDeviceId = 0;
    executorRecConfig.mEnginePath = "./models/ch_PP-OCRv4_rec_infer.trt";
    ExecutorWorkerConfig executorRecWorkerConfig;
    executorRecWorkerConfig.mProfileIndex = 0;
    executorRecConfig.mExecutorWorkerConfigs.emplace_back(executorRecWorkerConfig);
    mRecExecutor = std::make_shared<Executor>(executorRecConfig, mRecExecutorState);

    mRecExecutorState->initialize();
}

void PaddleocrPipelineProcessor::shutdown() {
    mRecExecutor->shutdown();
    mClsExecutor->shutdown();
    mDetExecutor->shutdown();

    mIsRunning = false;
    mTaskQueue->stop();
    if (mThread->joinable()) {
        mThread->join();
    }
}

CoroutinePipelineGenerator PaddleocrPipelineProcessor::process(
    std::shared_ptr<PipelineTask<PaddleocrRequest, PaddleocrResponse>> task) {
    auto paddleocrRequest = task->getRequest();

    auto *paddleocrDetRequest = new PaddleocrDetRequest(paddleocrRequest->frame);
    auto detCoroutineExecutorAwaitable =
        new CoroutineExecutorAwaitable<PaddleocrDetRequest, PaddleocrDetResponse>(
            mDetExecutor, paddleocrDetRequest);
    auto detCoroutineExecutorResponses = co_await *detCoroutineExecutorAwaitable;
    delete detCoroutineExecutorAwaitable;

    std::vector<PaddleocrClsRequest *> paddleocrClsRequests;
    for (auto &detOCRPredictResult :
         detCoroutineExecutorResponses[0]->oCRPredictResults) {
        auto *paddleocrClsRequest =
            new PaddleocrClsRequest(paddleocrRequest->frame, detOCRPredictResult);
        paddleocrClsRequests.emplace_back(paddleocrClsRequest);
    }

    auto clsCoroutineExecutorAwaitable =
        new CoroutineExecutorAwaitable<PaddleocrClsRequest, PaddleocrClsResponse>(
            mClsExecutor, paddleocrClsRequests);
    auto clsCoroutineExecutorResponses = co_await *clsCoroutineExecutorAwaitable;
    delete clsCoroutineExecutorAwaitable;

    std::vector<PaddleocrRecRequest *> paddleocrRecRequests;
    for (const auto &clsCoroutineExecutorResponse : clsCoroutineExecutorResponses) {
        auto *paddleocrRecRequest = new PaddleocrRecRequest(
            paddleocrRequest->frame, clsCoroutineExecutorResponse->oCRPredictResult);
        paddleocrRecRequests.emplace_back(paddleocrRecRequest);
    }

    auto recCoroutineExecutorAwaitable =
        new CoroutineExecutorAwaitable<PaddleocrRecRequest, PaddleocrRecResponse>(
            mRecExecutor, paddleocrRecRequests);
    auto recCoroutineExecutorResponses = co_await *recCoroutineExecutorAwaitable;
    delete recCoroutineExecutorAwaitable;

    auto paddleocrResponse = new PaddleocrResponse();
    paddleocrResponse->oCRPredictResults.resize(0);
    for (const auto &recCoroutineExecutorResponse : recCoroutineExecutorResponses) {
        paddleocrResponse->oCRPredictResults.emplace_back(
            recCoroutineExecutorResponse->oCRPredictResult);
    }

    delete paddleocrDetRequest;
    
    for (auto &detCoroutineExecutorResponse : detCoroutineExecutorResponses) {
        delete detCoroutineExecutorResponse;
    }

    for (auto &paddleocrClsRequest : paddleocrClsRequests) {
        delete paddleocrClsRequest;
    }

    for (auto &clsCoroutineExecutorResponse : clsCoroutineExecutorResponses) {
        delete clsCoroutineExecutorResponse;
    }

    for (auto &paddleocrRecRequest : paddleocrRecRequests) {
        delete paddleocrRecRequest;
    }

    for (auto &recCoroutineExecutorResponse : recCoroutineExecutorResponses) {
        delete recCoroutineExecutorResponse;
    }

    task->setResponse(paddleocrResponse);
    co_return;
}

} // namespace cniai::pipeline::paddleocr