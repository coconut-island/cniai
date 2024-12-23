#include "yolov5_detect_pipeline_processor.h"

#include "common/logging.h"

namespace cniai::pipeline::yolov5_detect {

void Yolov5DetectPipelineProcessor::initialize() {
    mExecutorState = std::make_shared<Yolov5DetectExecutorState>();
    ExecutorConfig executorConfig;
    executorConfig.mDeviceId = 0;
    executorConfig.mEnginePath = "./models/yolov5s_dynamic.trt";
    ExecutorWorkerConfig executorWorkerConfig;
    executorWorkerConfig.mProfileIndex = 0;
    executorConfig.mExecutorWorkerConfigs.emplace_back(executorWorkerConfig);
    mExecutor = std::make_shared<Executor>(executorConfig, mExecutorState);

    mExecutorState->initialize();
}

} // namespace cniai::pipeline::yolov5_detect