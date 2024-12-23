#pragma once

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include <spdlog/spdlog.h>

#include <cassert>
#include <iostream>

#define LOG_TRACE(...) SPDLOG_TRACE(__VA_ARGS__)
#define LOG_DEBUG(...) SPDLOG_DEBUG(__VA_ARGS__)
#define LOG_INFO(...) SPDLOG_INFO(__VA_ARGS__)
#define LOG_WARN(...) SPDLOG_WARN(__VA_ARGS__)
#define LOG_ERROR(...) SPDLOG_ERROR(__VA_ARGS__)
#define LOG_CRITICAL(...) SPDLOG_CRITICAL(__VA_ARGS__)

#define LOG_LEVEL_TRACE "trace"
#define LOG_LEVEL_DEBUG "debug"
#define LOG_LEVEL_INFO "info"
#define LOG_LEVEL_WARN "warn"
#define LOG_LEVEL_ERR "err"
#define LOG_LEVEL_ERROR "error"
#define LOG_LEVEL_CRITICAL "critical"
#define LOG_LEVEL_OFF "off"

#define SET_LOG_LEVEL(level)                                                   \
    spdlog::set_level(cniai::logging::getLogLevelEnumFromName(level))
#define SET_LOG_PATTERN(pattern) spdlog::set_pattern(pattern)

namespace cniai::logging {

inline spdlog::level::level_enum
getLogLevelEnumFromName(const std::string &levelName) {
    if (LOG_LEVEL_TRACE == levelName) {
        return spdlog::level::level_enum::trace;
    }

    if (LOG_LEVEL_DEBUG == levelName) {
        return spdlog::level::level_enum::debug;
    }

    if (LOG_LEVEL_INFO == levelName) {
        return spdlog::level::level_enum::info;
    }

    if (LOG_LEVEL_WARN == levelName) {
        return spdlog::level::level_enum::warn;
    }

    if (LOG_LEVEL_ERR == levelName || LOG_LEVEL_ERROR == levelName) {
        return spdlog::level::level_enum::err;
    }

    if (LOG_LEVEL_CRITICAL == levelName) {
        return spdlog::level::level_enum::critical;
    }

    if (LOG_LEVEL_OFF == levelName) {
        return spdlog::level::level_enum::off;
    }

    assert(0 && "not match any log levels, level name = ");
}

} // namespace cniai::logging
