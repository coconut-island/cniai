#include "common/logging.h"
#include "gflags/gflags.h"

DEFINE_string(
    LOG_LEVEL, "info",
    "Log level, includes [trace, debug, info, warn, err, critical, off]");
DEFINE_string(LOG_PATTERN, "[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%t] [%@] %v",
              "Log pattern");

int main(int argc, char *argv[]) {
    gflags::SetUsageMessage("cniai usage");
    gflags::SetVersionString("0.0.1");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    SET_LOG_PATTERN(FLAGS_LOG_PATTERN);
    SET_LOG_LEVEL(FLAGS_LOG_LEVEL);

    LOG_INFO("Hello cniai!");

    gflags::ShutDownCommandLineFlags();
    return 0;
}