#ifndef LSXCOMMON_CORE_H
#define LSXCOMMON_CORE_H

#include <string>

namespace lsxcommon {

// Frees device memory, treating cudaErrorCudartUnloading as success: during
// program termination the CUDA runtime cannot be used, and the driver reclaims
// the memory with the context anyway. Intended for destructors of
// process/thread-lifetime device scratch (never throws; a null pointer is a
// no-op; other errors are logged but swallowed).
void cudaFreeIgnoreShutdown(void* ptr);

class Core {
public:
    static void Initialize(const std::string& log_file_path = "logestix.log");
    static void Shutdown();
};

} // namespace lsxcommon

#endif // LSXCOMMON_CORE_H
