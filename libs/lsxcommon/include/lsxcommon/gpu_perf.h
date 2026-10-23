#pragma once

// ── GPU performance passport (model-neutral) ────────────────────────────────
// Hardware capability snapshot used by the auto-tuning policy: SM count,
// memory bus width and clock. Host-only; NEVER include from .cu TUs
// (host .cpp links fine — mirrors resource_governor.cpp).

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

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

// Real achievable device-to-device copy bandwidth (read + write), GB/s,
// measured on a fixed 512 MiB buffer (deliberately larger than the
// 96 MB L2, so it is DRAM-bound; one warmup + best of three). This is the
// perf baseline's ENVIRONMENT oracle: it depends only on the card's current
// thermal/power state, so it can separate "the room is warm" from "the code
// regressed" without the model certifying itself. Returns 0 when the probe
// cannot run (no CUDA / allocation failure); callers must treat 0 as "no
// oracle", never as a pass.
double MeasureMemoryBandwidthGbs(int mib = 512);

// ── Tuning knobs (per-GPU performance policy) ───────────────────────────────

struct LsxPerfKnobs {
    bool tc_fast_math_default = false;  // default-enable TC mxf4nvf4 path
    bool seq1_gemv_default = false;     // default-enable streaming GEMV
    bool gemv_large_default = false;    // large-N warp-per-row GEMV (decode)
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

// Resolution precedence: env > CLI(kOn/kOff, tests only) > calibrated >
// heuristic > base.
// Reads env at call time. Pure besides getenv.
LsxPerfKnobSet ResolvePerfKnobs(const GpuPerfProfile &gpu,
                                FastMathMode cli_mode = FastMathMode::kAuto);

// Process-wide resolved knob snapshot, applied at engine init. Consumers
// (expert-cache sizing, staging choice) read this instead of re-resolving.
LsxPerfKnobs CurrentKnobs();
PerfSource CurrentKnobsSource();
void SetCurrentKnobs(const LsxPerfKnobs &knobs, PerfSource source);

// Applies knobs to the process-global routing atomics + the snapshot above
// (published under the given source; never clobbers it).
void ApplyKnobs(const LsxPerfKnobs &knobs, PerfSource source);

// Call-once, idempotent policy application: resolves the per-GPU knob set
// (vault cache > base/heuristic via ResolvePerfKnobs, env honored) and
// publishes + applies it. Invoked from InferenceEngine::InitializeEngine and
// LsxModelFactory::Create so every consumer runs on the pinned policy by
// default; explicit SetCurrentKnobs overrides stay on top. No CUDA device ->
// marks applied and keeps legacy defaults.
void EnsurePerfPolicyApplied();

// ── Calibration cache (vault JSON, perf-cache/<gpu-key>.json) ───────────────

inline constexpr int kPerfCacheFormatVersion = 1;

struct PerfCacheEntry {
    GpuPerfProfile gpu;
    LsxPerfKnobs knobs;
};

// Hand-rolled JSON writer (stable field order); parser accepts the same
// schema and rejects foreign format versions (invalidation) and garbage.
std::string SerializePerfCache(const PerfCacheEntry &e);
std::optional<PerfCacheEntry> ParsePerfCache(std::string_view json);

// Vault path: lsx_model_metadata("perf-cache/<gpu-key>.json").
std::string PerfCachePath(const GpuPerfProfile &gpu);

// File-backed convenience wrappers. Never throw; failures return false.
bool LoadPerfCache(const GpuPerfProfile &gpu, LsxPerfKnobs &out);
bool SavePerfCache(const GpuPerfProfile &gpu, const LsxPerfKnobs &knobs);

} // namespace perf
} // namespace lsxcommon
