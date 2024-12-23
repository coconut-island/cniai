#include "common/str_util.h"

namespace cniai::str_util {

bool endsWith(const std::string &str, const std::string &suffix) {
    if (str.length() < suffix.length()) {
        return false;
    }
    return str.substr(str.length() - suffix.length()) == suffix;
}

bool startsWith(const char *str, const char *prefix) {
    size_t strLen = strlen(str);
    size_t prefixLen = strlen(prefix);

    if (strLen < prefixLen) {
        return false;
    }

    return strncmp(str, prefix, prefixLen) == 0;
}

std::string vformat(char const *fmt, va_list args) {
    va_list args0;
    va_copy(args0, args);
    auto const size = vsnprintf(nullptr, 0, fmt, args0);
    if (size <= 0)
        return "";

    std::string stringBuf(size, char{});
    auto const size2 = std::vsnprintf(&stringBuf[0], size + 1, fmt, args);

    assert(size2 == size && std::string(std::strerror(errno)).c_str());

    return stringBuf;
}

std::string replace_placeholders(const std::string &format,
                                 const std::vector<std::string> &args) {
    std::string result;
    size_t lastPos = 0, pos;
    int argIndex = 0;

    while ((pos = format.find("{}", lastPos)) != std::string::npos) {
        result += format.substr(lastPos, pos - lastPos);
        if (argIndex < args.size()) {
            result += args[argIndex++];
        }
        lastPos = pos + 2; // 2 是 "{}" 的长度
    }

    result += format.substr(lastPos);
    return result;
}

} // namespace cniai::str_util