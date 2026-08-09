#pragma once

#include <string>

namespace lsxcommon {
namespace log {

// Host-side spdlog bridge for CUDA translation units. NVCC cannot parse
// spdlog's bundled fmt headers ("character value is out of range" in
// format.h), so .cu files must not include <spdlog/spdlog.h> directly.
// All logging is routed through this facade; the implementation lives in a
// pure MSVC/GCC .cpp translation unit. Callers pass preformatted strings.

void debug(const std::string &msg);
void info(const std::string &msg);
void warn(const std::string &msg);
void error(const std::string &msg);

} // namespace log
} // namespace lsxcommon
