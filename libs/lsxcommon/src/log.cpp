#include "lsxcommon/log.h"

#include <spdlog/spdlog.h>

namespace lsxcommon {
namespace log {

void debug(const std::string &msg) { spdlog::debug("{}", msg); }
void info(const std::string &msg) { spdlog::info("{}", msg); }
void warn(const std::string &msg) { spdlog::warn("{}", msg); }
void error(const std::string &msg) { spdlog::error("{}", msg); }

} // namespace log
} // namespace lsxcommon
