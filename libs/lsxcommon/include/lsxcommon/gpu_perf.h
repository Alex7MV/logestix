#pragma once

// ── GPU performance passport (model-neutral) ────────────────────────────────
// Hardware capability snapshot used by the auto-tuning policy: SM count,
// memory bus width and clock. Host-only; NEVER include from .cu TUs
// (host .cpp links fine — mirrors resource_governor.cpp).

#include <cstdint>
#include <string>

#include "lsxcommon/fast_math_mode.h"

namespace lsxcommon {
namespace perf {

struct GpuPerfProfile {
    std::string name;              // cudaDeviceProp.name
    int sm_count = 0;              // multiProcessorCount
    int bus_width_bits = 0;        // memoryBusWidth
    int memory_clock_khz = 0;      // memoryClockRate (cudaDevAttrMemoryClockRate)
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

// ── Tuning knobs (per-GPU performance policy) ───────────────────────────────

struct LsxPerfKnobs {
    bool tc_fast_math_default = false;  // default-enable TC mxf4nvf4 path
    bool seq1_gemv_default = false;     // default-enable streaming GEMV
    int gemv_max_mib = 32;              // seq==1 GEMV weight-size limit
    int splitk_grid_min = 96;           // split-K grid_x threshold
    int expert_slots_pct = 100;         // % usable VRAM for expert cache
    bool dual_stream_staging = true;    // H2D/compute overlap staging
};

enum class PerfSource {
    kBase,         // committed base table (repo)
    kHeuristic,    // passport-derived fallback (unknown GPU)
    kCalibrated,   // local calibration cache (vault perf-cache/)
    kEnvOverride,  // env variables set (LSX_GEMV_MAX_MIB, LSX_TEST_FAST_MATH)
    kLegacy        // LSX_PERF_AUTO=0 escape hatch: pre-tuning behavior
};

struct LsxPerfKnobSet {
    LsxPerfKnobs knobs;
    PerfSource source = PerfSource::kBase;
};

// Reference bandwidth of the committed-base reference card (RTX 5090) used by
// the heuristic's GEMV limit scaling.
inline constexpr double kRtx5090PeakGbs = 1792.0;

// Resolution precedence: env > CLI(kOn/kOff) > calibrated > heuristic > base.
// Reads env at call time. Pure besides getenv.
LsxPerfKnobSet ResolvePerfKnobs(const GpuPerfProfile &gpu,
                                FastMathMode cli_mode = FastMathMode::kAuto);

// Process-wide resolved knob snapshot, applied at engine init. Consumers
// (expert-cache sizing, staging choice) read this instead of re-resolving.
LsxPerfKnobs CurrentKnobs();
PerfSource CurrentKnobsSource();
void SetCurrentKnobs(const LsxPerfKnobs &knobs, PerfSource source);

// Applies knobs to the process-global routing atomics + the snapshot above.
void ApplyKnobs(const LsxPerfKnobs &knobs);

} // namespace perf
} // namespace lsxcommon
