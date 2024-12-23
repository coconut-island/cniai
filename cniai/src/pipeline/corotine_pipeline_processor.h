#pragma once

#include "pipeline_processor.h"
#include "tensorrt/executor.h"
#include "tensorrt/executor_stat.h"

#include <coroutine>
#include <utility>

namespace cniai::pipeline {

class AwaitManager {
public:
    virtual void start() = 0;

    virtual bool waiting() = 0;
};

class CoroutinePipelineGenerator {
public:
    struct promise_type {
        std::shared_ptr<AwaitManager> mAwaitManager;
        std::exception_ptr mException;

        std::suspend_never initial_suspend() { return {}; }

        std::suspend_always final_suspend() noexcept { return {}; }

        void unhandled_exception() { mException = std::current_exception(); }

        CoroutinePipelineGenerator get_return_object() {
            return CoroutinePipelineGenerator(
                std::coroutine_handle<promise_type>::from_promise(*this));
        }

        void return_void() {}

        template <typename U>
        U &&await_transform(U &&awaitable) noexcept {
            return static_cast<U &&>(awaitable);
        }
    };

    explicit CoroutinePipelineGenerator(std::coroutine_handle<promise_type> handle)
        : mHandle(handle) {}

    ~CoroutinePipelineGenerator() {}

    [[nodiscard]] bool canResume() const {
        return !mHandle.promise().mAwaitManager->waiting();
    };

    cniai::id::IdType taskId;
    std::coroutine_handle<promise_type> mHandle;
};

class ExecutorAwaitManager : public AwaitManager {
public:
    std::shared_ptr<tensorrt::Executor> mExecutor;
    std::shared_ptr<std::vector<cniai::tensorrt::ExecutorRequest *>> mRequests;
    std::vector<cniai::id::IdType> mIds;

    explicit ExecutorAwaitManager(std::shared_ptr<tensorrt::Executor> &executor,
                                  cniai::tensorrt::ExecutorRequest *request)
        : mExecutor(executor) {
        mRequests = std::make_shared<std::vector<cniai::tensorrt::ExecutorRequest *>>();
        mRequests->emplace_back(request);
    }

    explicit ExecutorAwaitManager(
        std::shared_ptr<tensorrt::Executor> &executor,
        std::vector<cniai::tensorrt::ExecutorRequest *> &requests)
        : mExecutor(executor) {
        mRequests = std::make_shared<std::vector<cniai::tensorrt::ExecutorRequest *>>();
        mRequests->assign(requests.begin(), requests.end());
    }

    ~ExecutorAwaitManager(){};

    void start() override {
        auto ids = mExecutor->enqueueRequests(*mRequests);
        mIds.assign(ids.begin(), ids.end());
    }

    bool waiting() override { return !mExecutor->hasResponses(mIds); }
};

template <class ExecutorRequest, class ExecutorResponse>
class CoroutineExecutorAwaitable {
public:
    std::shared_ptr<AwaitManager> mExecutorAwaitManager;
    std::coroutine_handle<CoroutinePipelineGenerator::promise_type> mHandle;

    explicit CoroutineExecutorAwaitable(std::shared_ptr<tensorrt::Executor> &executor,
                                        ExecutorRequest *request) {
        mExecutorAwaitManager = std::make_shared<ExecutorAwaitManager>(
            executor, reinterpret_cast<cniai::tensorrt::ExecutorRequest *>(request));
    }

    explicit CoroutineExecutorAwaitable(std::shared_ptr<tensorrt::Executor> &executor,
                                        std::vector<ExecutorRequest *> &requests) {
        std::vector<cniai::tensorrt::ExecutorRequest *> executorRequestRequests;
        for (const auto &request : requests) {
            executorRequestRequests.emplace_back(
                reinterpret_cast<cniai::tensorrt::ExecutorRequest *>(request));
        }
        mExecutorAwaitManager =
            std::make_shared<ExecutorAwaitManager>(executor, executorRequestRequests);
    }

    bool await_ready() { return false; }

    void await_suspend(
        std::coroutine_handle<CoroutinePipelineGenerator::promise_type> handle) {
        mHandle = handle;
        handle.promise().mAwaitManager = mExecutorAwaitManager;
        handle.promise().mAwaitManager->start();
    };

    std::vector<ExecutorResponse *> await_resume() {
        std::shared_ptr<ExecutorAwaitManager> executorAwaitManager =
            std::dynamic_pointer_cast<ExecutorAwaitManager>(mExecutorAwaitManager);
        auto responses =
            executorAwaitManager->mExecutor->awaitResponses(executorAwaitManager->mIds);
        std::vector<ExecutorResponse *> results;
        for (const auto &response : responses) {
            results.emplace_back(reinterpret_cast<ExecutorResponse *>(response));
        }
        return results;
    }
};

template <class PipelineRequest, class PipelineResponse>
class CoroutinePipelineProcessor
    : public PipelineProcessor<PipelineRequest, PipelineResponse> {
public:
    void run() override {
        std::list<CoroutinePipelineGenerator> gens;
        std::unordered_map<
            int, std::shared_ptr<PipelineTask<PipelineRequest, PipelineResponse>>>
            taskMap;
        while (this->mIsRunning) {
            auto task =
                gens.empty()
                    ? this->mTaskQueue->dequeue()
                    : this->mTaskQueue->dequeueWithTimeout(std::chrono::milliseconds(1));

            if (task) {
                auto gen = process(task.value());
                gen.taskId = task.value()->getId();
                taskMap[task.value()->getId()] = task.value();
                gens.emplace_back(gen);
            }

            for (auto genIt = gens.begin(); genIt != gens.end();) {
                auto &handle = genIt->mHandle;
                if (!handle.done() && genIt->canResume()) {
                    handle.resume();
                }

                if (handle.done()) {
                    this->enqueueResponse(genIt->taskId,
                                          taskMap[genIt->taskId]->getResponse());
                    genIt->mHandle.destroy();
                    taskMap.erase(genIt->taskId);
                    genIt = gens.erase(genIt);
                } else {
                    ++genIt;
                }
            }
        }
    }

    virtual CoroutinePipelineGenerator
    process(std::shared_ptr<PipelineTask<PipelineRequest, PipelineResponse>> task) = 0;
};

} // namespace cniai::pipeline
