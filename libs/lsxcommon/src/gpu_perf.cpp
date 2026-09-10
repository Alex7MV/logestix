#include "lsxcommon/gpu_perf.h"

#include <cuda_runtime.h>

#include <absl/container/flat_hash_map.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <optional>

#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/lsx_tc_gemm.h"

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

// ── Knob resolution ─────────────────────────────────────────────────────────

namespace {

std::optional<LsxPerfKnobSet> BaseKnobs(const std::string &key)
{
    // Committed optimization base (repo). Author entries via
    // `lsxfabric --calibrate <model>` (values measured on real weights).
    static const absl::flat_hash_map<std::string, LsxPerfKnobs> kBase = [] {
        absl::flat_hash_map<std::string, LsxPerfKnobs> m;
        LsxPerfKnobs rtx5090;
        rtx5090.tc_fast_math_default = true;
        rtx5090.seq1_gemv_default = true;
        // gemv_max_mib 32 / splitk_grid_min 96 / slots 100 / dual true are
        // the profiled RTX 5090 defaults (see lsx_inference_nvfp4.cu).
        m.emplace("rtx 5090", rtx5090);
        return m;
    }();
    const auto it = kBase.find(key);
    if (it == kBase.end()) return std::nullopt;
    return LsxPerfKnobSet{it->second, PerfSource::kBase};
}

LsxPerfKnobSet HeuristicKnobs(const GpuPerfProfile &g)
{
    LsxPerfKnobs k;
    k.tc_fast_math_default = (g.cc_major == 12);
    k.seq1_gemv_default = true;
    const double bw = PeakBandwidthGbs(g);
    if (bw > 0) {
        const double scaled = 32.0 * bw / kRtx5090PeakGbs;
        k.gemv_max_mib = static_cast<int>(std::clamp(scaled, 8.0, 64.0));
    }
    k.splitk_grid_min = (g.sm_count > 0 && g.sm_count < 160) ? 64 : 96;
    k.expert_slots_pct = 100;
    k.dual_stream_staging = bw >= 768.0;
    return {k, PerfSource::kHeuristic};
}

bool LegacyAutoDisabled()
{
    const char *auto_env = std::getenv("LSX_PERF_AUTO");
    return auto_env != nullptr && std::strcmp(auto_env, "0") == 0;
}

} // namespace

LsxPerfKnobSet ResolvePerfKnobs(const GpuPerfProfile &gpu,
                                FastMathMode cli_mode)
{
    auto base = BaseKnobs(GpuProfileKey(gpu));
    LsxPerfKnobSet r = base ? *base : HeuristicKnobs(gpu);

    if (LegacyAutoDisabled())
        r = {LsxPerfKnobs{}, PerfSource::kLegacy};

    if (std::getenv("LSX_TEST_FAST_MATH")) {
        r.knobs.tc_fast_math_default = true;
        r.knobs.seq1_gemv_default = true;
        r.source = PerfSource::kEnvOverride;
    }
    if (const char *env = std::getenv("LSX_GEMV_MAX_MIB")) {
        const int v = std::atoi(env);
        if (v > 0) {
            r.knobs.gemv_max_mib = v;
            r.source = PerfSource::kEnvOverride;
        }
    }

    switch (cli_mode) {
        case FastMathMode::kOn:
            r.knobs.tc_fast_math_default = true;
            r.knobs.seq1_gemv_default = true;
            break;
        case FastMathMode::kOff:
            r.knobs.tc_fast_math_default = false;
            r.knobs.seq1_gemv_default = false;
            break;
        case FastMathMode::kAuto:
            break;
    }
    return r;
}

LsxPerfKnobs CurrentKnobs()
{
    static const LsxPerfKnobs kDefaults;  // legacy behavior snapshot
    return kDefaults;
}

PerfSource CurrentKnobsSource() { return PerfSource::kLegacy; }

void SetCurrentKnobs(const LsxPerfKnobs &, PerfSource)
{
    // Replaced by the atomic snapshot in Task 8 (engine wiring); kept as a
    // no-op so Task 2 compiles standalone.
}

void ApplyKnobs(const LsxPerfKnobs &knobs)
{
    SetCurrentKnobs(knobs, PerfSource::kBase);
    lsx_nvfp4::SetLsxTcFastMath(knobs.tc_fast_math_default);
    lsx_nvfp4::SetLsxSeq1Gemv(knobs.seq1_gemv_default);
    lsx_nvfp4::SetLsxGemmMaxMib(knobs.gemv_max_mib);
    lsx_nvfp4::SetLsxDualStreamStaging(knobs.dual_stream_staging);
    lsx_tc::SetLsxSplitKGridMin(knobs.splitk_grid_min);
}

} // namespace perf
} // namespace lsxcommon
