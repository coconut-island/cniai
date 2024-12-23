#pragma once

#include "backend.h"

namespace cniai_cuda_kernel::cuosd {

#ifdef ENABLE_TEXT_BACKEND_PANGO
std::shared_ptr<TextBackend> create_pango_cairo_backend();
#endif // ENABLE_TEXT_BACKEND_PANGO

} // namespace cniai_cuda_kernel::cuosd