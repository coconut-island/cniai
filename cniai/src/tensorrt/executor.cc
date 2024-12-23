#include "tensorrt/executor.h"
#include "common/assert.h"
#include "common/logging.h"
#include "tensorrt/buffers.h"
#include "tensorrt/ibuffer.h"

#include <fstream>
#include <utility>

namespace cniai::tensorrt {

RoundRobinRuleWorkerDispatcher::RoundRobinRuleWorkerDispatcher(
    ExecutorConfig executorConfig)
    : mExecutorConfig{std::move(executorConfig)} {}

int RoundRobinRuleWorkerDispatcher::getNextWorkerId() {
    mCurrentWorkerIndex++;
    if (mCurrentWorkerIndex + 1 >= mExecutorConfig.mExecutorWorkerConfigs.size()) {
        mCurrentWorkerIndex = 0;
    }
    return mCurrentWorkerIndex;
}

Executor::Executor(ExecutorConfig executorConfig,
                   std::shared_ptr<ExecutorState> &executorState)
    : mExecutorConfig{std::move(executorConfig)}, mExecutorState(executorState) {

    int deviceId = mExecutorConfig.mDeviceId;
    CUDA_CHECK(cudaSetDevice(deviceId));

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    CNIAI_CHECK_WITH_INFO((deviceId >= 0 && deviceId < (deviceCount)),
                          "device id is not correct, device id: {}", deviceId);

    initializeRuntime();

    initializeWorker();

    mExecutorWorkerDispatcher =
        std::make_unique<RoundRobinRuleWorkerDispatcher>(mExecutorConfig);
}

cniai::id::IdType Executor::enqueueRequest(ExecutorRequest *request) {
    auto nextWorkerId = mExecutorWorkerDispatcher->getNextWorkerId();
    LOG_DEBUG("worker size: {}", mWorkers.size());
    LOG_DEBUG("Next worker id: {}", nextWorkerId);
    auto id = mIdGenerator.getNextId();
    auto requestPair = std::make_pair(id, request);
    getWorker(nextWorkerId)->getRequestQueue()->enqueue(requestPair);
    return id;
}

std::vector<cniai::id::IdType>
Executor::enqueueRequests(std::vector<ExecutorRequest *> &requests) {
    std::vector<cniai::id::IdType> ids;
    ids.reserve(requests.size());
    for (const auto &request : requests) {
        ids.emplace_back(enqueueRequest(request));
    }
    return ids;
}

void Executor::enqueueResponse(const cniai::id::IdType &requestId,
                               ExecutorResponse *response) {
    {
        std::lock_guard<std::mutex> lock(mResponseMapMutex);
        mResponseMap[requestId] = response;
    }
    mResponseMapCondVar.notify_one();
}

bool Executor::hasResponses(const std::vector<cniai::id::IdType> &requestIds) {
    bool result = true;
    for (auto &requestId : requestIds) {
        if (mResponseMap.find(requestId) == mResponseMap.end()) {
            result = false;
            break;
        }
    }
    return result;
}

bool Executor::hasResponse(const cniai::id::IdType &requestId) {
    return mResponseMap.find(requestId) != mResponseMap.end();
}

ExecutorResponse *Executor::awaitResponse(const cniai::id::IdType &requestId) {
    std::unique_lock<std::mutex> lock(mResponseMapMutex);
    mResponseMapCondVar.wait(
        lock, [&] { return mResponseMap.find(requestId) != mResponseMap.end(); });

    auto response = mResponseMap[requestId];
    mResponseMap.erase(requestId);
    return response;
}

std::vector<ExecutorResponse *>
Executor::awaitResponses(const std::vector<cniai::id::IdType> &requestIds) {
    std::vector<ExecutorResponse *> executorResponses;
    executorResponses.reserve(requestIds.size());
    for (const auto &requestId : requestIds) {
        executorResponses.emplace_back(awaitResponse(requestId));
    }
    return executorResponses;
}

void Executor::initializeRuntime() {
    std::string &enginePath = mExecutorConfig.mEnginePath;

    std::ifstream engineFile(enginePath, std::ios::in | std::ios::binary);
    CNIAI_CHECK_WITH_INFO(engineFile.is_open(),
                          "Engine path is not correct, engine path: {}",
                          enginePath.c_str());

    engineFile.seekg(0, std::ios::end);
    std::streampos engineSize = engineFile.tellg();
    engineFile.seekg(0, std::ios::beg);

    char *engineBuffer = new char[engineSize];
    engineFile.read(engineBuffer, engineSize);

    mRuntime = std::make_unique<Runtime>(engineBuffer, engineSize);
    engineFile.close();
    delete[] engineBuffer;
}

void Executor::initializeWorker() {
    for (size_t i = 0; i < mExecutorConfig.mExecutorWorkerConfigs.size(); i++) {
        auto worker = std::make_shared<ExecutorWorker>(
            i, mExecutorConfig.mDeviceId, this, mExecutorConfig.mExecutorWorkerConfigs[i],
            mRuntime, mExecutorState);
        mWorkers.emplace_back(worker);
    }
    CNIAI_CHECK(!mWorkers.empty());
}

std::shared_ptr<ExecutorWorker> Executor::getWorker(int workerIndex) {
    return mWorkers[workerIndex];
}

void Executor::shutdown() {
    for (const auto &worker : mWorkers) {
        worker->stop();
    }
}

ExecutorWorker::ExecutorWorker(int workerIndex, int deviceId, Executor *executor,
                               ExecutorWorkerConfig config,
                               std::shared_ptr<Runtime> &runtime,
                               std::shared_ptr<ExecutorState> &executorState)
    : mWorkerIndex{workerIndex}, mDeviceId{deviceId}, mExecutor(executor),
      mConfig(config), mRuntime(runtime), mExecutorState(executorState) {

    mRequestQueue = std::make_shared<cniai::blocking_queue::BlockingQueue<
        std::pair<cniai::id::IdType, ExecutorRequest *>>>();
    mThread = std::make_unique<std::thread>(&ExecutorWorker::run, this);
}

ExecutorWorker::~ExecutorWorker() {
    stop();
    mRuntime->clearContexts();
}

void ExecutorWorker::run() {
    CUDA_CHECK(cudaSetDevice(mDeviceId));

    mCudaStream = std::make_shared<CudaStream>();

    auto profileCount = mRuntime->getNbProfiles();
    if (mConfig.mProfileIndex < 0 || mConfig.mProfileIndex >= profileCount) {
        LOG_WARN("Current profile count: {}, "
                 "executorWorkConfig.mProfileIndex set to 0.",
                 profileCount);
        mConfig.mProfileIndex = 0;
    }

    int contextIndex = mRuntime->addContext(mConfig.mProfileIndex, mCudaStream->get());
    mCudaStream->synchronize();
    auto &context = mRuntime->getContext(contextIndex);

    auto maxPreferredBatchSize = mConfig.mMaxPreferredBatchSize;

    auto profileMaxBatchSize = mRuntime->getProfileMaxBatchSize(mConfig.mProfileIndex);

    if (maxPreferredBatchSize <= 0 || maxPreferredBatchSize > profileMaxBatchSize) {
        LOG_WARN("maxPreferredBatchSize value not support: {}, reset to "
                 "profileMaxBatchSize: {}",
                 maxPreferredBatchSize, profileMaxBatchSize);
        maxPreferredBatchSize = profileMaxBatchSize;
    }

    auto &engine = mRuntime->getEngine();
    int nbIOTensors = engine.getNbIOTensors();
    int profileIndex = mConfig.mProfileIndex;
    for (int ioTensorIndex = 0; ioTensorIndex < nbIOTensors; ++ioTensorIndex) {
        auto tensorName = engine.getIOTensorName(ioTensorIndex);
        if (engine.getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT) {
            auto maxShape = engine.getProfileShape(tensorName, profileIndex,
                                                   nvinfer1::OptProfileSelector::kMAX);
            auto dataType = engine.getTensorDataType(tensorName);
            auto tensor = BufferManager::gpu(maxShape, mCudaStream, dataType);
            mCudaStream->synchronize();
            CUDA_CHECK(cudaMemset(tensor->data(), 0, tensor->getSizeInBytes()));
            mInputTensorMap[tensorName] = tensor;
            context.setInputTensorAddress(tensorName, tensor->data());

            context.setInputShape(tensorName, maxShape);
        }

        CNIAI_CHECK(engine.getTensorIOMode(tensorName) != nvinfer1::TensorIOMode::kNONE);
    }

    mRuntime->inferShapes(contextIndex);

    CNIAI_CHECK(context.allInputDimensionsSpecified());

    for (int ioTensorIndex = 0; ioTensorIndex < nbIOTensors; ++ioTensorIndex) {
        auto tensorName = engine.getIOTensorName(ioTensorIndex);

        if (engine.getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kOUTPUT) {
            auto shape = context.getTensorShape(tensorName);
            auto dataType = engine.getTensorDataType(tensorName);
            auto output = BufferManager::gpu(shape, mCudaStream, dataType);
            mCudaStream->synchronize();
            CUDA_CHECK(cudaMemset(output->data(), 0, output->getSizeInBytes()));
            mOutputTensorMap[tensorName] = output;
            context.setOutputTensorAddress(tensorName, output->data());

            if (mIsAllocHostOutputTensor) {
                auto hostOutput = BufferManager::pinned(shape, dataType);
                CUDA_CHECK(
                    cudaMemset(hostOutput->data(), 0, hostOutput->getSizeInBytes()));
                mOutputHostTensorMap[tensorName] = hostOutput;
            }
        }

        CNIAI_CHECK(engine.getTensorIOMode(tensorName) != nvinfer1::TensorIOMode::kNONE);
    }

    mCudaStream->synchronize();

    uint32_t batchingIndex = 0;
    std::vector<std::pair<cniai::id::IdType, ExecutorRequest *>> willInferenceRequests;
    while (mIsRunning) {
        LOG_DEBUG("Dequeue.");
        auto executeRequest =
            mConfig.mDequeueWithTimeout <= 0
                ? mRequestQueue->dequeue()
                : mRequestQueue->dequeueWithTimeout(
                      std::chrono::milliseconds(mConfig.mDequeueWithTimeout));

        LOG_DEBUG("Dequeue get one.");

        if (!executeRequest && willInferenceRequests.empty()) {
            LOG_DEBUG("1");
            continue;
        }

        bool isBatchingDone = false;

        if (executeRequest) {
            LOG_DEBUG("2");
            batchingIndex++;
            willInferenceRequests.emplace_back(executeRequest.value());
        }

        if (batchingIndex == (maxPreferredBatchSize - 1)) {
            LOG_DEBUG("3");
            isBatchingDone = true;
            batchingIndex = 0;
        }

        if (executeRequest && mRequestQueue->empty()) {
            LOG_DEBUG("4");
            isBatchingDone = true;
            batchingIndex = 0;
        }

        if (isBatchingDone) {
            LOG_DEBUG("Batch Done, start infer.");

            std::vector<ExecutorRequest *> requests;
            requests.reserve(willInferenceRequests.size());
            for (const auto &willInferenceRequest : willInferenceRequests) {
                requests.emplace_back(willInferenceRequest.second);
            }

            ExecutorStatePreprocessParam executorStatePreprocessParam{
                &requests,    mRuntime,         contextIndex,
                profileIndex, &mInputTensorMap, mCudaStream};
            mExecutorState->preprocess(&executorStatePreprocessParam);

            for (auto &inputTensorPair : mInputTensorMap) {
                context.setInputShape(inputTensorPair.first.c_str(),
                                      inputTensorPair.second->getShape());

                auto &shape = inputTensorPair.second->getShape();
            }

            mRuntime->inferShapes(contextIndex);

            CNIAI_CHECK_WITH_INFO(context.allInputDimensionsSpecified(),
                                  "Input dimensions not specified");

            for (auto &outputTensorPair : mOutputTensorMap) {
                auto outputShape = context.getTensorShape(outputTensorPair.first.c_str());
                outputTensorPair.second->reshape(outputShape);

                if (mIsCopyHostOutputTensor) {
                    mOutputHostTensorMap[outputTensorPair.first]->reshape(outputShape);
                }
            }

            CNIAI_CHECK(mRuntime->executeContext(contextIndex, mCudaStream->get()));
            if (mIsCopyHostOutputTensor) {
                std::vector<std::string> outputCopyTensorNames;
                if (mCopyOutputHostTensorNames.empty()) {
                    for (const auto &outputTensorPair : mOutputTensorMap) {
                        outputCopyTensorNames.push_back(outputTensorPair.first);
                    }
                } else {
                    outputCopyTensorNames = mCopyOutputHostTensorNames;
                }

                for (const auto &outputCopyTensorName : outputCopyTensorNames) {
                    BufferManager::copy(*mOutputTensorMap[outputCopyTensorName],
                                        *mOutputHostTensorMap[outputCopyTensorName],
                                        mCudaStream->get());
                }
            }

            std::vector<ExecutorResponse *> responses;
            ExecutorStatePostprocessParam executorStatePostprocessParam{
                &requests,
                &responses,
                mRuntime,
                contextIndex,
                profileIndex,
                &mOutputTensorMap,
                &mOutputHostTensorMap,
                mCudaStream};
            mExecutorState->postprocess(&executorStatePostprocessParam);

            CNIAI_CHECK(responses.size() == requests.size());

            for (int i = 0; i < responses.size(); ++i) {
                mExecutor->enqueueResponse(willInferenceRequests[i].first, responses[i]);
            }

            willInferenceRequests.clear();
        }
    }
}

void ExecutorWorker::stop() {
    mIsRunning = false;
    mRequestQueue->stop();
    if (mThread->joinable()) {
        mThread->join();
    }
}

std::shared_ptr<
    cniai::blocking_queue::BlockingQueue<std::pair<cniai::id::IdType, ExecutorRequest *>>>
ExecutorWorker::getRequestQueue() {
    return mRequestQueue;
}

} // namespace cniai::tensorrt