#include "tensorrt/runtime.h"
#include "tensorrt/logger.h"

#include "common/assert.h"
#include "common/logging.h"

// #include <nvtx3/nvtx3.hpp>

namespace cniai::tensorrt {

namespace {
cniai::tensorrt::Logger defaultLogger{};
} // namespace

Runtime::Runtime(void const *engineData, std::size_t engineSize,
                 nvinfer1::ILogger &logger)
    : mRuntime{nvinfer1::createInferRuntime(logger)},
      mEngine{mRuntime->deserializeCudaEngine(engineData, engineSize)},
      mEngineInspector{mEngine->createEngineInspector()} {
    assert(mEngine != nullptr && "Failed to deserialize cuda engine");
    auto const devMemorySize = mEngine->getDeviceMemorySize();
    mEngineBuffer = BufferManager::gpu(devMemorySize);

    // Print context memory size for CI/CD to track.
    LOG_INFO("Allocated {} MiB for execution context memory.",
             static_cast<double>(devMemorySize) / 1048576.0);
}

Runtime::Runtime(void const *engineData, std::size_t engineSize)
    : Runtime{engineData, engineSize, defaultLogger} {}

int32_t Runtime::addContext(int32_t profileIndex, const cudaStream_t &cudaStream) {
    assert(0 <= profileIndex && profileIndex < mEngine->getNbOptimizationProfiles());
    auto contextIndex = static_cast<int32_t>(mContexts.size());
    mContexts.emplace_back(mEngine->createExecutionContextWithoutDeviceMemory());
    auto &context = *mContexts.back();
    context.setDeviceMemory(mEngineBuffer->data());
    context.setOptimizationProfileAsync(profileIndex, cudaStream);
    // If nvtx verbosity is DETAILED, change it to LAYER_NAMES_ONLY for inference
    // performance
    if (context.getNvtxVerbosity() == nvinfer1::ProfilingVerbosity::kDETAILED) {
        context.setNvtxVerbosity(nvinfer1::ProfilingVerbosity::kLAYER_NAMES_ONLY);
    }
    return contextIndex;
}

void Runtime::clearContexts() {
    for (auto &context : mContexts) {
        context.reset();
    }
    mContexts.clear();
}

bool Runtime::executeContext(int32_t contextIndex, const cudaStream_t &cudaStream) const {
    // NVTX3_FUNC_RANGE();
    auto &context = getContext(contextIndex);
    return context.enqueueV3(cudaStream);
}

int Runtime::getProfileMaxBatchSize(int profileIndex) {
    int maxBatchSize = INT_MAX;
    int nbIOTensors = mEngine->getNbIOTensors();

    for (int ioTensorIndex = 0; ioTensorIndex < nbIOTensors; ++ioTensorIndex) {
        auto tensorName = mEngine->getIOTensorName(ioTensorIndex);
        if (mEngine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT) {
            if (!mEngine->isShapeInferenceIO(tensorName)) {
                auto maxShape = mEngine->getProfileShape(
                    tensorName, profileIndex, nvinfer1::OptProfileSelector::kMAX);
                if (maxBatchSize > maxShape.d[0]) {
                    maxBatchSize = maxShape.d[0];
                }
            } else {
                const int32_t *max_shapes = mEngine->getProfileTensorValues(
                    tensorName, profileIndex, nvinfer1::OptProfileSelector::kMAX);
                if (maxBatchSize > *max_shapes) {
                    maxBatchSize = *max_shapes;
                }
            }
        }
    }

    return maxBatchSize;
}

void Runtime::inferShapes(int32_t contextIndex) const {
    char const *missing;
    auto const nbMissing = getContext(contextIndex).inferShapes(1, &missing);
    if (nbMissing > 0) {
        CNIAI_THROW("Input shape not specified: {}", missing);
    }

    if (nbMissing < 0) {
        CNIAI_THROW("Invalid input shape");
    }
}

} // namespace cniai::tensorrt
