#include "lsxcommon/gpu_perf.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cstring>

namespace lsxcommon {
namespace perf {

GpuPerfProfile ProbeGpuPerfProfile()
{
    GpuPerfProfile g;
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) return g;
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return g;
    g.name = p.name;
    g.sm_count = p.multiProcessorCount;
    g.bus_width_bits = p.memoryBusWidth;
    // CUDA 13 removed cudaDeviceProp::memoryClockRate; the attribute API
    // still exposes the peak memory clock (kHz).
    int mem_clock_khz = 0;
    cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, 0);
    g.memory_clock_khz = mem_clock_khz;
    g.l2_cache_bytes = p.l2CacheSize;
    g.vram_total_bytes = p.totalGlobalMem;
    g.cc_major = p.major;
    g.cc_minor = p.minor;
    return g;
}

double PeakBandwidthGbs(const GpuPerfProfile &g)
{
    if (g.bus_width_bits <= 0 || g.memory_clock_khz <= 0) return 0.0;
    return static_cast<double>(g.bus_width_bits) *
           static_cast<double>(g.memory_clock_khz) * 1000.0 * 2.0 / 8.0 /
           1e9;
}

std::string GpuProfileKey(const GpuPerfProfile &g)
{
    std::string key;
    key.reserve(g.name.size());
    for (char c : g.name)
        key.push_back(
            static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    for (const char *prefix : {"nvidia geforce ", "nvidia "}) {
        const size_t plen = std::strlen(prefix);
        if (key.compare(0, plen, prefix) == 0) {
            key.erase(0, plen);
            break;
        }
    }
    return key;
}

} // namespace perf
} // namespace lsxcommon
