#pragma once

namespace lsxcommon {

// The version this binary was BUILT with, from the generated
// `lsxcommon/version.h` (configured from `project(logestix VERSION ...)`).
// Never spell a version in a source file: the two drifted once already, and a
// literal compiles just as well as the macro.
const char *version_string();

void version();
void log_gpu_info();

} // namespace logestix
