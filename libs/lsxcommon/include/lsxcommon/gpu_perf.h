#pragma once

// ── GPU performance passport (model-neutral) ────────────────────────────────
// Hardware capability snapshot used by the auto-tuning policy: SM count,
// memory bus width and clock. Host-only; NEVER include from .cu TUs
// (host .cpp links fine — mirrors resource_governor.cpp).

#include <cstdint>
#include <string>

namespace lsxcommon {
namespace perf {

struct GpuPerfProfile {
    std::string name;              // cudaDeviceProp.name
    int sm_count = 0;              // multiProcessorCount
    int bus_width_bits = 0;        // memoryBusWidth
    int memory_clock_khz = 0;      // memoryClockRate
    int l2_cache_bytes = 0;        // l2CacheSize
    std::uint64_t vram_total_bytes = 0;
    int cc_major = 0, cc_minor = 0;
};

// Real probe of CUDA device 0. Never throws; failed probes leave a zeroed
// profile (heuristics must treat zeros as "unknown").
GpuPerfProfile ProbeGpuPerfProfile();

// Peak theoretical DRAM bandwidth, GB/s:
// bus_bits * clock_khz * 1000 * 2 (DDR) / 8 / 1e9.
double PeakBandwidthGbs(const GpuPerfProfile &g);

// Normalized base-table key: lowercase, leading "nvidia " and
// "nvidia geforce " prefixes stripped.
std::string GpuProfileKey(const GpuPerfProfile &g);

} // namespace perf
} // namespace lsxcommon
