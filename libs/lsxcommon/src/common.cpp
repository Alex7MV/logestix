#include <cstdio>
#include "lsxcommon/common.h"
#include <spdlog/spdlog.h>

#ifdef LOGESTIX_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace lsxcommon {

void version()
{
    std::fprintf(stdout, "logestix v0.1.0\n");
}

void log_gpu_info()
{
#ifdef LOGESTIX_HAS_CUDA
    int count = 0;
    auto err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0)
    {
        spdlog::warn("No CUDA-capable GPU found");
        return;
    }
    for (int i = 0; i < count; ++i)
    {
        cudaDeviceProp props{};
        if (cudaGetDeviceProperties(&props, i) == cudaSuccess)
        {
            spdlog::info("GPU {}: {} (compute {}.{} | {} MiB)",
                i, props.name, props.major, props.minor,
                props.totalGlobalMem / (1024 * 1024));
        }
    }
#else
    spdlog::info("CUDA support not compiled in");
#endif
}

} // namespace logestix
