#pragma once

#include "backend.h"

namespace cniai_cuda_kernel::cuosd {

#ifdef ENABLE_TEXT_BACKEND_STB
std::shared_ptr<TextBackend> create_stb_backend();
#endif // ENABLE_TEXT_BACKEND_STB

} // namespace cniai_cuda_kernel::cuosd