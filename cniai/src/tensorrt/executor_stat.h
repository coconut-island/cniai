#pragma once

#include "tensorrt/cuda_stream.h"
#include "tensorrt/itensor.h"
#include "tensorrt/runtime.h"

namespace cniai::tensorrt {

struct ExecutorRequest;
struct ExecutorResponse;

struct ExecutorStatePreprocessParam {
    std::vector<cniai::tensorrt::ExecutorRequest *> *requests;
    std::shared_ptr<Runtime> runtime;
    int contextIndex;
    int profileIndex;
    std::unordered_map<std::string, std::shared_ptr<ITensor>> *inputTensorMap;
    std::shared_ptr<CudaStream> cudaStream;
};

struct ExecutorStatePostprocessParam {
    std::vector<ExecutorRequest *> *requests;
    std::vector<ExecutorResponse *> *responses;
    std::shared_ptr<Runtime> runtime;
    int contextIndex;
    int profileIndex;
    std::unordered_map<std::string, std::shared_ptr<ITensor>> *outputTensorMap;
    std::unordered_map<std::string, std::shared_ptr<ITensor>> *outputHostTensorMap;
    std::shared_ptr<CudaStream> cudaStream;
};

class ExecutorState {
public:
    explicit ExecutorState();
    virtual ~ExecutorState();

public:
    virtual void
    preprocess(ExecutorStatePreprocessParam *executorStatePreprocessParam) = 0;

    virtual void
    postprocess(ExecutorStatePostprocessParam *ExecutorStatePostprocessParam) = 0;

    virtual void initialize() = 0;
};

} // namespace cniai::tensorrt
