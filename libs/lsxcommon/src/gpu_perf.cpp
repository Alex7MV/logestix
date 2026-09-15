#include "lsxcommon/gpu_perf.h"

#include <cuda_runtime.h>

#include <absl/container/flat_hash_map.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>

#include <algorithm>
#include <mutex>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>

#include <simdjson.h>

#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/lsx_tc_gemm.h"
#include "lsxcommon/path_registry.h"
#include "lsxcommon/log.h"


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
    // Committed optimization base (repo). Reference values measured on real
    // weights; author new entries from the vault auto-calibration snippets
    // (perf-cache/<gpu-key>.json).
    static const absl::flat_hash_map<std::string, LsxPerfKnobs> kBase = [] {
        absl::flat_hash_map<std::string, LsxPerfKnobs> m;
        LsxPerfKnobs rtx5090;
        rtx5090.tc_fast_math_default = true;
        rtx5090.seq1_gemv_default = true;
        // 27B decode profile (2026-09-13): warp-per-row GEMV on large-N
        // dense shapes beats the TC M=64 tile path (27.5 -> 30.5 tok/s).
        rtx5090.gemv_large_default = true;
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
    // R3 (numeric contract §1.1.4): the numeric path must not depend on device
    // properties. Every knob that changes the arithmetic (GEMV path/limit,
    // split-K threshold, TC on/off) is pinned to the SAME canonical values as
    // the RTX 5090 base profile, so all GPUs in the supported set run the
    // identical numeric path. Only non-numeric knobs (expert-cache residency,
    // H2D/compute overlap) may be device-derived.
    LsxPerfKnobs k;
    k.tc_fast_math_default = (g.cc_major == 12);
    k.seq1_gemv_default = true;
    k.gemv_large_default = true;   // canonical (was base-only)
    k.gemv_max_mib = 32;           // canonical (was bandwidth-scaled)
    k.splitk_grid_min = 96;        // canonical (was SM-count-derived)
    k.expert_slots_pct = 100;
    const double bw = PeakBandwidthGbs(g);
    k.dual_stream_staging = bw >= 768.0;  // overlap only, not numerics
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

namespace {

// Process-wide resolved knob snapshot guarded by a mutex (the POD exceeds
// lock-free atomic width on some toolchains). Readers copy under the lock;
// writers (engine init, tests) publish once. Default = legacy behavior.
LsxPerfKnobs g_current_knobs{};
PerfSource g_current_knobs_source{PerfSource::kLegacy};
std::mutex g_current_knobs_mu;

} // namespace

LsxPerfKnobs CurrentKnobs()
{
    std::lock_guard<std::mutex> lk(g_current_knobs_mu);
    return g_current_knobs;
}

PerfSource CurrentKnobsSource()
{
    std::lock_guard<std::mutex> lk(g_current_knobs_mu);
    return g_current_knobs_source;
}

void SetCurrentKnobs(const LsxPerfKnobs &knobs, PerfSource source)
{
    std::lock_guard<std::mutex> lk(g_current_knobs_mu);
    g_current_knobs = knobs;
    g_current_knobs_source = source;
}

void ApplyKnobs(const LsxPerfKnobs &knobs, PerfSource source)
{
    SetCurrentKnobs(knobs, source);
    lsx_nvfp4::SetLsxTcFastMath(knobs.tc_fast_math_default);
    lsx_nvfp4::SetLsxSeq1Gemv(knobs.seq1_gemv_default);
    lsx_nvfp4::SetLsxGemmLarge(knobs.gemv_large_default);
    lsx_nvfp4::SetLsxGemmMaxMib(knobs.gemv_max_mib);
    lsx_nvfp4::SetLsxDualStreamStaging(knobs.dual_stream_staging);
    lsx_tc::SetLsxSplitKGridMin(knobs.splitk_grid_min);
}

void EnsurePerfPolicyApplied()
{
    static std::once_flag flag;
    std::call_once(flag, [] {
        const GpuPerfProfile gpu = ProbeGpuPerfProfile();
        if (gpu.name.empty()) {
            log::debug("[perf] no CUDA device: policy left at legacy defaults");
            return;
        }
        LsxPerfKnobs knobs;
        PerfSource source = PerfSource::kHeuristic;
        if (LoadPerfCache(gpu, knobs)) {
            source = PerfSource::kCalibrated;
        } else {
            const LsxPerfKnobSet resolved = ResolvePerfKnobs(gpu);
            knobs = resolved.knobs;
            source = resolved.source;
        }
        SetCurrentKnobs(knobs, source);
        ApplyKnobs(knobs, source);
        log::info(absl::StrFormat(
            "[perf] policy applied: gpu='%s' sm=%d bus=%dbit "
            "tc=%d gemv_mib=%d splitk_min=%d slots_pct=%d dual=%d",
            gpu.name.c_str(), gpu.sm_count, gpu.bus_width_bits,
            static_cast<int>(knobs.tc_fast_math_default), knobs.gemv_max_mib,
            knobs.splitk_grid_min, knobs.expert_slots_pct,
            static_cast<int>(knobs.dual_stream_staging)));
    });
}

// ── Calibration cache ───────────────────────────────────────────────────────

std::string SerializePerfCache(const PerfCacheEntry &e)
{
    std::string s;
    absl::StrAppend(&s, "{\n  \"format_version\": ", kPerfCacheFormatVersion,
                    ",\n  \"gpu\": {\"name\": \"", e.gpu.name,
                    "\", \"sm_count\": ", e.gpu.sm_count,
                    ", \"bus_width_bits\": ", e.gpu.bus_width_bits,
                    ", \"memory_clock_khz\": ", e.gpu.memory_clock_khz,
                    ", \"cc_major\": ", e.gpu.cc_major,
                    ", \"cc_minor\": ", e.gpu.cc_minor, "},\n  \"knobs\": {");
    absl::StrAppend(&s, "\"tc_fast_math_default\": ",
                    e.knobs.tc_fast_math_default ? "true" : "false",
                    ", \"seq1_gemv_default\": ",
                    e.knobs.seq1_gemv_default ? "true" : "false");
    absl::StrAppend(&s, ", \"gemv_max_mib\": ", e.knobs.gemv_max_mib,
                    ", \"splitk_grid_min\": ", e.knobs.splitk_grid_min,
                    ", \"expert_slots_pct\": ", e.knobs.expert_slots_pct,
                    ", \"dual_stream_staging\": ",
                    e.knobs.dual_stream_staging ? "true" : "false");
    absl::StrAppend(&s, "},\n  \"ts_unix\": ", std::time(nullptr), "\n}\n");
    return s;
}

std::optional<PerfCacheEntry> ParsePerfCache(std::string_view json_view)
{
    // simdjson's DOM parser borrows the input buffer: copy into stable storage
    // so temporaries (e.g. ParsePerfCache(SerializePerfCache(e))) stay valid.
    const std::string json(json_view);
    simdjson::dom::parser parser;
    simdjson::dom::element root;
    if (parser.parse(json).get(root) != simdjson::SUCCESS)
        return std::nullopt;
    int64_t ver = 0;
    if (root["format_version"].get(ver) != simdjson::SUCCESS ||
        ver != kPerfCacheFormatVersion)
        return std::nullopt;
    simdjson::dom::element gpu, knobs;
    if (root["gpu"].get(gpu) != simdjson::SUCCESS ||
        root["knobs"].get(knobs) != simdjson::SUCCESS)
        return std::nullopt;
    PerfCacheEntry e;
    std::string_view name;
    if (gpu["name"].get(name) != simdjson::SUCCESS) return std::nullopt;
    e.gpu.name = name;
    int64_t v = 0;
    if (gpu["sm_count"].get(v) == simdjson::SUCCESS)
        e.gpu.sm_count = static_cast<int>(v);
    if (gpu["bus_width_bits"].get(v) == simdjson::SUCCESS)
        e.gpu.bus_width_bits = static_cast<int>(v);
    if (gpu["memory_clock_khz"].get(v) == simdjson::SUCCESS)
        e.gpu.memory_clock_khz = static_cast<int>(v);
    if (gpu["cc_major"].get(v) == simdjson::SUCCESS)
        e.gpu.cc_major = static_cast<int>(v);
    if (gpu["cc_minor"].get(v) == simdjson::SUCCESS)
        e.gpu.cc_minor = static_cast<int>(v);
    bool b = false;
    if (knobs["tc_fast_math_default"].get_bool().get(b) == simdjson::SUCCESS)
        e.knobs.tc_fast_math_default = b;
    if (knobs["seq1_gemv_default"].get_bool().get(b) == simdjson::SUCCESS)
        e.knobs.seq1_gemv_default = b;
    if (knobs["gemv_max_mib"].get(v) == simdjson::SUCCESS)
        e.knobs.gemv_max_mib = static_cast<int>(v);
    if (knobs["splitk_grid_min"].get(v) == simdjson::SUCCESS)
        e.knobs.splitk_grid_min = static_cast<int>(v);
    if (knobs["expert_slots_pct"].get(v) == simdjson::SUCCESS)
        e.knobs.expert_slots_pct = static_cast<int>(v);
    if (knobs["dual_stream_staging"].get_bool().get(b) == simdjson::SUCCESS)
        e.knobs.dual_stream_staging = b;
    return e;
}

std::string PerfCachePath(const GpuPerfProfile &gpu)
{
    return lsx_model_metadata("perf-cache/" + GpuProfileKey(gpu) + ".json");
}

bool LoadPerfCache(const GpuPerfProfile &gpu, LsxPerfKnobs &out)
{
    std::ifstream f(PerfCachePath(gpu), std::ios::binary);
    if (!f.is_open()) return false;
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    auto entry = ParsePerfCache(json);
    if (!entry.has_value()) return false;
    out = entry->knobs;
    return true;
}

bool SavePerfCache(const GpuPerfProfile &gpu, const LsxPerfKnobs &knobs)
{
    const std::string path = PerfCachePath(gpu);
    std::error_code ec;
    std::filesystem::create_directories(
        std::filesystem::path(path).parent_path(), ec);
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f.is_open()) return false;
    f << SerializePerfCache({gpu, knobs});
    return f.good();
}

bool EnsureCalibrated(const GpuPerfProfile &gpu,
                      const std::function<bool(int &out_mib)> &run_bench)
{
    if (gpu.name.empty()) return false;
    if (LegacyAutoDisabled()) return false;
    if (BaseKnobs(GpuProfileKey(gpu)).has_value())
        return false;                        // known GPU: committed base wins
    LsxPerfKnobs cached{};
    if (LoadPerfCache(gpu, cached)) return false;  // already calibrated

    // No base entry, no cache: calibration pending (first Load on this GPU).
    if (!run_bench) return true;             // query-only mode (tests)
    int mib = 0;
    if (!run_bench(mib)) {
        log::warn("[perf] calibration benchmark failed; keeping heuristic");
        return false;
    }
    LsxPerfKnobs knobs = HeuristicKnobs(gpu).knobs;
    // R3 numeric contract §1.1.4: gemv_max_mib changes the GEMV numeric path,
    // so a per-GPU calibrated value would make the arithmetic device-dependent
    // and break cross-model bit-identity (5090 vs 5060 Ti). Keep the canonical
    // value; per-GPU numeric tuning is a C-fast-only facility (explicit opt-in).
    (void)mib;
    if (!SavePerfCache(gpu, knobs)) return false;
    SetCurrentKnobs(knobs, PerfSource::kCalibrated);
    ApplyKnobs(knobs, PerfSource::kCalibrated);
    log::info("[perf] canonical knobs applied (gpu '" + gpu.name +
              "'; per-GPU numeric calibration disabled for R3)");
    return true;
}

} // namespace perf
} // namespace lsxcommon
