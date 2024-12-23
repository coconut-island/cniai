#include "exception.h"

#include "common/str_util.h"

#include <cstdlib>
#if !defined(_MSC_VER)
#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>
#endif
#include <sstream>

namespace cniai::exception {

namespace {
int constexpr VOID_PTR_SZ = 2 + sizeof(void *) * 2;
}

#if !defined(_MSC_VER)
CniaiException::CniaiException(char const *file, std::size_t line, std::string const &msg)
    : std::runtime_error{""} {
    mNbFrames = backtrace(mCallstack.data(), MAX_FRAMES);
    auto const trace = getTrace();
    std::runtime_error::operator=(std::runtime_error{cniai::str_util::fmtstr(
        "{} ({}:{})\n{}", msg.c_str(), file, line, trace.c_str())});
}
#else
CniaiException::CniaiException(char const *file, std::size_t line, std::string const &msg)
    : mNbFrames{},
      std::runtime_error{cniai::str_util::fmtstr("{} ({}:{})", msg.c_str(), file, line)} {
}
#endif

CniaiException::~CniaiException() noexcept = default;

std::string CniaiException::getTrace() const {
#if defined(_MSC_VER)
    return "";
#else
    auto const trace = backtrace_symbols(mCallstack.data(), mNbFrames);
    std::ostringstream buf;
    for (auto i = 1; i < mNbFrames; ++i) {
        Dl_info info;
        if (dladdr(mCallstack[i], &info) && info.dli_sname) {
            auto const clearName = demangle(info.dli_sname);
            buf << cniai::str_util::fmtstr(
                "{} {} {} + {}", i, VOID_PTR_SZ, mCallstack[i], clearName.c_str(),
                static_cast<char *>(mCallstack[i]) - static_cast<char *>(info.dli_saddr));
        } else {
            buf << cniai::str_util::fmtstr("{} {} {}", i, VOID_PTR_SZ, mCallstack[i],
                                           trace[i]);
        }
        if (i < mNbFrames - 1)
            buf << std::endl;
    }

    if (mNbFrames == MAX_FRAMES)
        buf << std::endl << "[truncated]";

    std::free(trace);
    return buf.str();
#endif
}

std::string CniaiException::demangle(char const *name) {
#if defined(_MSC_VER)
    return name;
#else
    std::string clearName{name};
    auto status = -1;
    auto const demangled = abi::__cxa_demangle(name, nullptr, nullptr, &status);
    if (status == 0) {
        clearName = demangled;
        std::free(demangled);
    }
    return clearName;
#endif
}

} // namespace cniai::exception
