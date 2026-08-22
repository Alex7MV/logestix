#include "lsxcommon/common.h"
#include "lsxcommon/log.h"

#include <absl/strings/str_format.h>

#ifdef LOGESTIX_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace lsxcommon {

void version()
{
    log::info("logestix v0.1.0");
}

void log_gpu_info()
{
#ifdef LOGESTIX_HAS_CUDA
    int count = 0;
    auto err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0)
    {
        log::warn("No CUDA-capable GPU found");
        return;
    }
    for (int i = 0; i < count; ++i)
    {
        cudaDeviceProp props{};
        if (cudaGetDeviceProperties(&props, i) == cudaSuccess)
        {
            log::info(absl::StrFormat(
                "GPU {}: {} (compute {}.{} | {} MiB)", i, props.name,
                props.major, props.minor,
                props.totalGlobalMem / (1024 * 1024)));
        }
    }
#else
    log::info("CUDA support not compiled in");
#endif
}

} // namespace lsxcommon
