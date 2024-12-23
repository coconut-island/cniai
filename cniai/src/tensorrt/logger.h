#pragma once

#include <NvInfer.h>

#include "common/logging.h"

namespace cniai::tensorrt {

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char *msg) noexcept override;
};

} // namespace cniai::tensorrt