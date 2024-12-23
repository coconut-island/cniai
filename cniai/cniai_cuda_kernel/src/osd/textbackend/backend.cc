#include "backend.h"

#ifdef ENABLE_TEXT_BACKEND_PANGO
#include "pango-cairo.hpp"
#endif

#ifdef ENABLE_TEXT_BACKEND_STB
#include "stb.h"
#endif

#include <cstdio>
#include <sstream>

namespace cniai_cuda_kernel::cuosd {

const char *text_backend_type_name(TextBackendType backend) {

    switch (backend) {
    case TextBackendType::PangoCairo:
        return "PangoCairo";
    case TextBackendType::StbTrueType:
        return "StbTrueType";
    default:
        return "Unknow";
    }
}

std::shared_ptr<TextBackend> create_text_backend(TextBackendType backend) {

    switch (backend) {

#ifdef ENABLE_TEXT_BACKEND_PANGO
    case TextBackendType::PangoCairo:
        return create_pango_cairo_backend();
#endif

#ifdef ENABLE_TEXT_BACKEND_STB
    case TextBackendType::StbTrueType:
        return create_stb_backend();
#endif

    default:
        printf("Unsupport text backend: %s\n", text_backend_type_name(backend));
        return nullptr;
    }
}

std::string concat_font_name_size(const char *name, int size) {
    std::stringstream ss;
    ss << name;
    ss << " ";
    ss << size;
    return ss.str();
}

} // namespace cniai_cuda_kernel::cuosd