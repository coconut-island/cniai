#pragma once

#include <cassert>
#include <nvjpeg.h>

#ifndef NVJPEG_CHECK
#define NVJPEG_CHECK(call)                                                     \
    do {                                                                       \
        nvjpegStatus_t _e = call;                                              \
        if (_e != NVJPEG_STATUS_SUCCESS) {                                     \
            std::cerr << "NVJPEG error " << _e << " at " << __FILE__ << ":"    \
                      << __LINE__ << std::endl;                                \
            ;                                                                  \
            assert(0);                                                         \
        }                                                                      \
    } while (0)
#endif // NVJPEG_CHECK
