#pragma once

#include "buffer_manager.h"
#include "tensorrt/buffer_manager.h"
#include "tensorrt/cuda_event.h"
#include "tensorrt/cuda_stream.h"
#include "tensorrt/itensor.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <NvInfer.h>

namespace cniai::tensorrt {

class Runtime {
public:
    explicit Runtime(void const *engineData, std::size_t engineSize,
                     nvinfer1::ILogger &logger);

    explicit Runtime(nvinfer1::IHostMemory const &engineBuffer, nvinfer1::ILogger &logger)
        : Runtime{engineBuffer.data(), engineBuffer.size(), logger} {}

    explicit Runtime(void const *engineData, std::size_t engineSize);

    explicit Runtime(nvinfer1::IHostMemory const &engineBuffer)
        : Runtime{engineBuffer.data(), engineBuffer.size()} {}

    [[nodiscard]] int32_t getNbContexts() const {
        return static_cast<int32_t>(mContexts.size());
    }

    [[nodiscard]] nvinfer1::IExecutionContext &getContext(int32_t contextIndex) const {
        return *mContexts.at(contextIndex);
    }

    [[nodiscard]] int32_t getNbProfiles() const {
        return static_cast<int32_t>(mEngine->getNbOptimizationProfiles());
    }

    int32_t addContext(int32_t profileIndex, const cudaStream_t &cudaStream);

    void clearContexts();

    [[nodiscard]] bool executeContext(int32_t contextIndex,
                                      const cudaStream_t &cudaStream) const;

    nvinfer1::ICudaEngine &getEngine() { return *mEngine; }

    nvinfer1::IEngineInspector &getEngineInspector() { return *mEngineInspector; }

    int getProfileMaxBatchSize(int profileIndex);

    void inferShapes(int contextIndex) const;

private:
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
    BufferManager::IBufferPtr mEngineBuffer;
    std::vector<std::unique_ptr<nvinfer1::IExecutionContext>> mContexts;
    std::unique_ptr<ITensor> mDummyTensor;
    std::unique_ptr<nvinfer1::IEngineInspector> mEngineInspector;
};

} // namespace cniai::tensorrt