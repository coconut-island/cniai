#pragma once

#include <chrono>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/blocking_queue.h"
#include "common/id.h"
#include "common/thread_pool.h"
#include "tensorrt/cuda_stream.h"
#include "tensorrt/executor_stat.h"
#include "tensorrt/runtime.h"

namespace cniai::tensorrt {

class Executor;
class ExecutorWorkerConfig;

class ExecutorConfig {
public:
    explicit ExecutorConfig() = default;

public:
    int mDeviceId{0};
    std::string mEnginePath;
    std::vector<ExecutorWorkerConfig> mExecutorWorkerConfigs;
};

class ExecutorWorkerConfig {
public:
    explicit ExecutorWorkerConfig() = default;

public:
    int mMaxPreferredBatchSize{0};
    int mDequeueWithTimeout{0};
    int mProfileIndex{-1};
};

class IExecutorWorkerDispatcher {

public:
    virtual ~IExecutorWorkerDispatcher() = default;

public:
    virtual int getNextWorkerId() = 0;
};

class RoundRobinRuleWorkerDispatcher : public IExecutorWorkerDispatcher {
public:
    explicit RoundRobinRuleWorkerDispatcher(ExecutorConfig executorConfig);

public:
    int getNextWorkerId() override;

private:
    ExecutorConfig mExecutorConfig;
    int mCurrentWorkerIndex{-1};
};

struct ExecutorRequest;

struct ExecutorResponse;

class ExecutorWorker {
public:
    explicit ExecutorWorker(int workerIndex, int deviceId, Executor *executor,
                            ExecutorWorkerConfig config,
                            std::shared_ptr<Runtime> &runtime,
                            std::shared_ptr<ExecutorState> &executorState);
    ~ExecutorWorker();

    std::shared_ptr<cniai::blocking_queue::BlockingQueue<
        std::pair<cniai::id::IdType, ExecutorRequest *>>>
    getRequestQueue();

    void stop();

private:
    void run();

private:
    int mWorkerIndex;
    int mDeviceId;
    Executor *mExecutor;
    ExecutorWorkerConfig mConfig;
    std::shared_ptr<CudaStream> mCudaStream;
    std::unique_ptr<std::thread> mThread;
    std::shared_ptr<Runtime> mRuntime;
    std::unordered_map<std::string, std::shared_ptr<ITensor>> mInputTensorMap;
    std::unordered_map<std::string, std::shared_ptr<ITensor>> mOutputTensorMap;
    std::unordered_map<std::string, std::shared_ptr<ITensor>> mOutputHostTensorMap;
    std::shared_ptr<cniai::blocking_queue::BlockingQueue<
        std::pair<cniai::id::IdType, ExecutorRequest *>>>
        mRequestQueue;
    std::atomic<bool> mIsRunning{true};
    std::shared_ptr<ExecutorState> mExecutorState;
    bool mIsAllocHostOutputTensor = true;
    bool mIsCopyHostOutputTensor = true;
    std::vector<std::string> mCopyOutputHostTensorNames{};
};

class Executor {

public:
    explicit Executor(ExecutorConfig executorConfig,
                      std::shared_ptr<ExecutorState> &executorState);

    ~Executor() = default;

private:
public:
    cniai::id::IdType enqueueRequest(ExecutorRequest *request);

    std::vector<cniai::id::IdType>
    enqueueRequests(std::vector<ExecutorRequest *> &requests);

    bool hasResponse(const cniai::id::IdType &requestId);

    bool hasResponses(const std::vector<cniai::id::IdType> &requestIds);

    ExecutorResponse *awaitResponse(cniai::id::IdType const &requestId);

    std::vector<ExecutorResponse *>
    awaitResponses(const std::vector<cniai::id::IdType> &requestIds);

    void shutdown();

private:
    void initializeRuntime();

    void initializeWorker();

    std::shared_ptr<ExecutorWorker> getWorker(int workerIndex);

    void enqueueResponse(const cniai::id::IdType &requestId, ExecutorResponse *response);

private:
    ExecutorConfig mExecutorConfig;
    std::unique_ptr<IExecutorWorkerDispatcher> mExecutorWorkerDispatcher;
    cniai::id::IdGenerator mIdGenerator;
    std::shared_ptr<Runtime> mRuntime;
    std::vector<std::shared_ptr<ExecutorWorker>> mWorkers;
    std::shared_ptr<ExecutorState> mExecutorState;
    std::unordered_map<cniai::id::IdType, ExecutorResponse *> mResponseMap;
    std::mutex mResponseMapMutex;
    std::condition_variable mResponseMapCondVar;

    friend class ExecutorWorker;
};

} // namespace cniai::tensorrt