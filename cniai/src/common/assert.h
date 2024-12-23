#pragma once

#include <cassert>

#include "common/exception.h"
#include "common/str_util.h"

namespace cniai::assert {

#if defined(_WIN32)
#define CNIAI_LIKELY(x) (__assume((x) == 1), (x))
#define CNIAI_UNLIKELY(x) (__assume((x) == 0), (x))
#else
#define CNIAI_LIKELY(x) __builtin_expect((x), 1)
#define CNIAI_UNLIKELY(x) __builtin_expect((x), 0)
#endif

#define CNIAI_THROW(info, ...)                                                           \
    do {                                                                                 \
        throw cniai::exception::CniaiException(                                          \
            __FILE__, __LINE__, cniai::str_util::fmtstr(info, ##__VA_ARGS__));           \
    } while (0)

#define CNIAI_CHECK(val)                                                                 \
    do {                                                                                 \
        CNIAI_LIKELY(static_cast<bool>(val))                                             \
        ? ((void)0) : assert(0);                                                         \
    } while (0)

#define CNIAI_CHECK_WITH_INFO(val, info, ...)                                            \
    do {                                                                                 \
        CNIAI_LIKELY(static_cast<bool>(val))                                             \
        ? ((void)0)                                                                      \
        : throw cniai::exception::CniaiException(                                        \
              __FILE__, __LINE__,                                                        \
              cniai::str_util::fmtstr("[Cniai][ERROR] Assertion failed: {}",             \
                                      cniai::str_util::fmtstr(info, ##__VA_ARGS__)));    \
    } while (0)

} // namespace cniai::assert