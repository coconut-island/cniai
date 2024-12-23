#include "tensorrt/logger.h"

namespace cniai::tensorrt {

void Logger::log(Severity severity, const char *msg) noexcept {
    if (severity == Severity::kINFO) {
        LOG_INFO("kINFO: {}", msg);
    } else if (severity == Severity::kWARNING) {
        LOG_WARN("kWARNING: {}", msg);
    } else if (severity == Severity::kERROR) {
        LOG_ERROR("kERROR: {}", msg);
    } else if (severity == Severity::kINTERNAL_ERROR) {
        LOG_ERROR("kINTERNAL_ERROR: {}", msg);
    } else if (severity == Severity::kVERBOSE) {
        LOG_DEBUG("kVERBOSE: {}", msg);
    }
}

} // namespace cniai::tensorrt
