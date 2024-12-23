#pragma once

#include <list>

#include "tensorrt/executor.h"
#include "tensorrt/executor_stat.h"

namespace cniai::pipeline {

template <class PipelineRequest, class PipelineResponse>
class PipelineTask {
public:
    explicit PipelineTask(cniai::id::IdType id, PipelineRequest *request)
        : mId(id), mRequest(request) {}

    cniai::id::IdType getId() { return mId; }

    PipelineRequest *getRequest() { return mRequest; }

    PipelineResponse *getResponse() { return mResponse; }

    void setResponse(PipelineResponse *response) { mResponse = response; }

    bool done() { return mResponse; };

private:
    PipelineRequest *mRequest = nullptr;
    PipelineResponse *mResponse = nullptr;
    cniai::id::IdType mId;
};

template <class PipelineRequest, class PipelineResponse>
class PipelineProcessor {

public:
    explicit PipelineProcessor() {
        mTaskQueue = std::make_shared<cniai::blocking_queue::BlockingQueue<
            std::shared_ptr<PipelineTask<PipelineRequest, PipelineResponse>>>>();
        mThread = std::make_unique<std::thread>(&PipelineProcessor::run, this);
    };

    ~PipelineProcessor() = default;

    cniai::id::IdType enqueue(PipelineRequest *request) {
        auto id = mIdGenerator.getNextId();
        auto task = std::make_shared<PipelineTask<PipelineRequest, PipelineResponse>>(
            id, request);
        {
            std::lock_guard<std::mutex> lock(mTaskMutex);
            mTaskQueue->enqueue(task);
        }
        mTaskCondVar.notify_one();
        return id;
    }

    PipelineResponse *waitResponse(cniai::id::IdType const &requestId) {
        std::unique_lock<std::mutex> lock(mResponseMapMutex);
        mResponseMapCondVar.wait(
            lock, [&] { return mResponseMap.find(requestId) != mResponseMap.end(); });

        auto response = mResponseMap[requestId];
        mResponseMap.erase(requestId);
        return response;
    }

    void enqueueResponse(const cniai::id::IdType &requestId, PipelineResponse *response) {
        {
            std::lock_guard<std::mutex> lock(mResponseMapMutex);
            mResponseMap[requestId] = response;
        }
        mResponseMapCondVar.notify_one();
    }

    virtual void run() = 0;

protected:
    std::atomic<bool> mIsRunning{true};
    cniai::id::IdGenerator mIdGenerator;
    std::mutex mTaskMutex;
    std::condition_variable mTaskCondVar;
    std::unordered_map<cniai::id::IdType, PipelineResponse *> mResponseMap;
    std::mutex mResponseMapMutex;
    std::condition_variable mResponseMapCondVar;
    std::unique_ptr<std::thread> mThread;

    std::shared_ptr<cniai::blocking_queue::BlockingQueue<
        std::shared_ptr<PipelineTask<PipelineRequest, PipelineResponse>>>>
        mTaskQueue;
    std::unique_ptr<PipelineProcessor<PipelineRequest, PipelineResponse>>
        mPipelineProcessor;
};

} // namespace cniai::pipeline
