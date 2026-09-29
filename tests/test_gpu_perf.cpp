#include <catch2/catch_test_macros.hpp>
#include "lsxcommon/gpu_perf.h"
#include "lsxcommon/fast_math_mode.h"
#include "lsxcommon/resource_governor.h"
#include "lsxcommon/perf_calibration.h"
#include "lsxcommon/path_registry.h"
#include <filesystem>
#include <system_error>
#include "lsxcommon/lsx_inference_nvfp4.h"

#include <absl/strings/str_cat.h>
#include <absl/strings/str_replace.h>

#include <cstdlib>

using lsxcommon::perf::GpuPerfProfile;
using lsxcommon::perf::GpuProfileKey;
using lsxcommon::perf::PeakBandwidthGbs;
using lsxcommon::FastMathMode;

TEST_CASE("PeakBandwidthGbs: bus_width * mem_clock * 2 DDR factor", "[gpu_perf]")
{
    GpuPerfProfile g;
    g.bus_width_bits = 512;
    g.memory_clock_khz = 1750000;  // 1.75 GHz * 2 (DDR) * 512 bit / 8 = 224 GB/s
    REQUIRE(PeakBandwidthGbs(g) == 224.0);

    GpuPerfProfile ddr;
    ddr.bus_width_bits = 256;
    ddr.memory_clock_khz = 1000000;  // 1 GHz * 2 * 256/8 = 64 GB/s
    REQUIRE(PeakBandwidthGbs(ddr) == 64.0);
}

TEST_CASE("GpuProfileKey: lowercase, vendor prefix stripped", "[gpu_perf]")
{
    GpuPerfProfile g;
    g.name = "NVIDIA GeForce RTX 5090";
    REQUIRE(GpuProfileKey(g) == "rtx 5090");
    g.name = "NVIDIA RTX 6000 Ada";
    REQUIRE(GpuProfileKey(g) == "rtx 6000 ada");
    g.name = "";
    REQUIRE(GpuProfileKey(g).empty());
}

// ── Task 2: knobs / base table / heuristic / resolution precedence ──────

static GpuPerfProfile Rtx5090()
{
    GpuPerfProfile g;
    g.name = "NVIDIA GeForce RTX 5090";
    g.sm_count = 170;
    g.bus_width_bits = 512;
    g.memory_clock_khz = 1750000;
    g.cc_major = 12;
    g.cc_minor = 0;
    g.vram_total_bytes = std::uint64_t{32} << 30;
    return g;
}

TEST_CASE("ResolvePerfKnobs: committed base entry (RTX 5090)", "[gpu_perf]")
{
    auto r = lsxcommon::perf::ResolvePerfKnobs(Rtx5090());
    REQUIRE(r.source == lsxcommon::perf::PerfSource::kBase);
    REQUIRE(r.knobs.tc_fast_math_default == true);
    REQUIRE(r.knobs.seq1_gemv_default == true);
    REQUIRE(r.knobs.gemv_max_mib == 32);
    REQUIRE(r.knobs.splitk_grid_min == 96);
    REQUIRE(r.knobs.expert_slots_pct == 100);
    REQUIRE(r.knobs.dual_stream_staging == true);
}

TEST_CASE("ResolvePerfKnobs: heuristic keeps numeric knobs device-independent",
          "[gpu_perf]")
{
    // R3 (numeric contract §1.1.4): numeric knobs must not vary with device
    // properties. The heuristic fallback (unknown GPU) therefore uses the SAME
    // canonical numeric values as the base profile; only non-numeric knobs
    // (overlap/residency) may be device-derived.
    GpuPerfProfile small = Rtx5090();
    small.name = "Unknown GPU X";
    small.sm_count = 64;
    small.bus_width_bits = 128;
    small.memory_clock_khz = 1000000;   // low bandwidth
    small.cc_major = 11;                // non-Blackwell
    auto r = lsxcommon::perf::ResolvePerfKnobs(small);
    REQUIRE(r.source == lsxcommon::perf::PerfSource::kHeuristic);
    REQUIRE(r.knobs.gemv_max_mib == 32);             // canonical, not scaled
    REQUIRE(r.knobs.splitk_grid_min == 96);          // canonical, not SM-based
    REQUIRE(r.knobs.gemv_large_default == true);     // canonical
    REQUIRE(r.knobs.dual_stream_staging == false);   // overlap: bw-derived OK
    REQUIRE(r.knobs.tc_fast_math_default == false);  // only CC 12
    REQUIRE(r.knobs.expert_slots_pct == 100);

    GpuPerfProfile blackwell = small;
    blackwell.cc_major = 12;
    auto rb = lsxcommon::perf::ResolvePerfKnobs(blackwell);
    REQUIRE(rb.knobs.tc_fast_math_default == true);
    // Same numeric knobs regardless of SM count / bandwidth.
    REQUIRE(rb.knobs.gemv_max_mib == r.knobs.gemv_max_mib);
    REQUIRE(rb.knobs.splitk_grid_min == r.knobs.splitk_grid_min);
}

TEST_CASE("ResolvePerfKnobs: precedence env > CLI > base/heuristic",
          "[gpu_perf]")
{
    // CLI kOff beats base TC default.
    auto r = lsxcommon::perf::ResolvePerfKnobs(Rtx5090(),
                                               FastMathMode::kOff);
    REQUIRE(r.source == lsxcommon::perf::PerfSource::kBase);
    REQUIRE(r.knobs.tc_fast_math_default == false);
    REQUIRE(r.knobs.seq1_gemv_default == false);

    // CLI kOn on a heuristic GPU still forces TC.
    GpuPerfProfile unknown = Rtx5090();
    unknown.name = "Mystery Card";
    auto rh = lsxcommon::perf::ResolvePerfKnobs(unknown, FastMathMode::kOn);
    REQUIRE(rh.knobs.tc_fast_math_default == true);

    // Env wins over everything.
    ::setenv("LSX_GEMV_MAX_MIB", "16", 1);
    auto re = lsxcommon::perf::ResolvePerfKnobs(Rtx5090());
    REQUIRE(re.source == lsxcommon::perf::PerfSource::kEnvOverride);
    REQUIRE(re.knobs.gemv_max_mib == 16);
    ::unsetenv("LSX_GEMV_MAX_MIB");
}

TEST_CASE("ResolvePerfKnobs: LSX_PERF_AUTO=0 legacy escape hatch", "[gpu_perf]")
{
    ::setenv("LSX_PERF_AUTO", "0", 1);
    auto r = lsxcommon::perf::ResolvePerfKnobs(Rtx5090());
    REQUIRE(r.source == lsxcommon::perf::PerfSource::kLegacy);
    REQUIRE(r.knobs.tc_fast_math_default == false);
    REQUIRE(r.knobs.seq1_gemv_default == false);
    REQUIRE(r.knobs.gemv_max_mib == 32);
    REQUIRE(r.knobs.splitk_grid_min == 96);
    REQUIRE(r.knobs.dual_stream_staging == true);
    ::unsetenv("LSX_PERF_AUTO");
}

// ── Task 3: calibration cache (JSON round-trip, invalidation, vault path) ──

using lsxcommon::perf::kPerfCacheFormatVersion;
using lsxcommon::perf::LsxPerfKnobs;
using lsxcommon::perf::ParsePerfCache;
using lsxcommon::perf::PerfCacheEntry;
using lsxcommon::perf::PerfCachePath;
using lsxcommon::perf::SerializePerfCache;

TEST_CASE("Perf cache: JSON round-trip", "[gpu_perf]")
{
    PerfCacheEntry e;
    e.gpu = Rtx5090();
    e.knobs.tc_fast_math_default = true;
    e.knobs.seq1_gemv_default = true;
    e.knobs.gemv_max_mib = 24;
    e.knobs.splitk_grid_min = 64;
    e.knobs.dual_stream_staging = false;
    const std::string j = SerializePerfCache(e);
    const auto parsed = ParsePerfCache(j);
    if (!parsed.has_value()) FAIL("NOPARSE json=" + j);
    REQUIRE(parsed.has_value());
    REQUIRE(parsed->gpu.name == e.gpu.name);
    REQUIRE(parsed->gpu.sm_count == 170);
    REQUIRE(parsed->knobs.gemv_max_mib == 24);
    REQUIRE(parsed->knobs.splitk_grid_min == 64);
    REQUIRE(parsed->knobs.dual_stream_staging == false);
    REQUIRE(parsed->knobs.tc_fast_math_default == true);
}

TEST_CASE("Perf cache: serializer escapes the gpu name", "[gpu_perf]")
{
    PerfCacheEntry e;
    e.gpu = Rtx5090();
    e.gpu.name = "GPU \"X\"\\Y";
    const std::string j = SerializePerfCache(e);
    const auto parsed = ParsePerfCache(j);
    REQUIRE(parsed.has_value());
    REQUIRE(parsed->gpu.name == e.gpu.name);
}

TEST_CASE("Perf cache: format-version invalidation", "[gpu_perf]")
{
    auto poisoned = absl::StrReplaceAll(
        SerializePerfCache({Rtx5090(), LsxPerfKnobs{}}),
        {{absl::StrCat("\"format_version\": ", kPerfCacheFormatVersion),
          "\"format_version\": 999"}});
    REQUIRE_FALSE(ParsePerfCache(poisoned).has_value());
    REQUIRE_FALSE(ParsePerfCache("not json at all").has_value());
    REQUIRE_FALSE(ParsePerfCache("").has_value());
}

TEST_CASE("Perf cache path lands in vault perf-cache/", "[gpu_perf]")
{
    const std::string p = PerfCachePath(Rtx5090());
    REQUIRE(p.find("perf-cache") != std::string::npos);
    REQUIRE(p.find("rtx 5090.json") != std::string::npos);
}

#include <cstddef>

using lsxcommon::lifecycle::ExpertCacheSlots;

TEST_CASE("ExpertCacheSlots honors the percentage knob", "[gpu_perf]")
{
    const size_t per = 100u << 20;  // 100 MiB per expert
    // 10 GiB free - 8 GiB reserve = 2 GiB usable -> 20 experts full.
    REQUIRE(ExpertCacheSlots(std::size_t{10} << 30, per, 288) == 20);
    REQUIRE(ExpertCacheSlots(std::size_t{10} << 30, per, 288, 50) == 10);
    REQUIRE(ExpertCacheSlots(std::size_t{10} << 30, per, 288, 0) == 0);
    // pct > 100 clamps to full budget; total-expert cap still applies.
    REQUIRE(ExpertCacheSlots(std::size_t{10} << 30, per, 288, 200) == 20);
    REQUIRE(ExpertCacheSlots(std::size_t{10} << 30, per, 4, 100) == 4);
    // Degenerate inputs unchanged.
    REQUIRE(ExpertCacheSlots(0, per, 288) == 0);
    REQUIRE(ExpertCacheSlots(std::size_t{10} << 30, 0, 288) == 288);
}

using lsxcommon::lsx_nvfp4::LsxFastMathScope;
using lsxcommon::lsx_nvfp4::LsxSeq1Gemv;
using lsxcommon::lsx_nvfp4::LsxTcFastMath;
using lsxcommon::lsx_nvfp4::SetLsxSeq1Gemv;
using lsxcommon::lsx_nvfp4::SetLsxTcFastMath;

TEST_CASE("LsxFastMathScope tri-state semantics", "[gpu_perf]")
{
    SetLsxTcFastMath(true);
    SetLsxSeq1Gemv(true);
    {
        const LsxFastMathScope s(FastMathMode::kAuto);
        REQUIRE(LsxTcFastMath() == true);   // untouched
        REQUIRE(LsxSeq1Gemv() == true);
    }
    {
        const LsxFastMathScope s(FastMathMode::kOff);
        REQUIRE(LsxTcFastMath() == false);
        REQUIRE(LsxSeq1Gemv() == false);
    }
    REQUIRE(LsxTcFastMath() == true);       // restored
    {
        const LsxFastMathScope s(FastMathMode::kOn);
        REQUIRE(LsxTcFastMath() == true);
        REQUIRE(LsxSeq1Gemv() == true);
    }
    SetLsxTcFastMath(false);
    SetLsxSeq1Gemv(false);
}

using lsxcommon::perf::EnsureCalibrated;

TEST_CASE("EnsureCalibrated: base GPUs skip, unknown GPUs run once", "[gpu_perf]")
{
    // RTX 5090 is in the committed base: no calibration, bench not invoked.
    int calls = 0;
    REQUIRE_FALSE(EnsureCalibrated(Rtx5090(), [&calls](int &) {
        ++calls;
        return true;
    }));
    REQUIRE(calls == 0);

    // Unknown GPU with no cache: bench runs once; result persists.
    GpuPerfProfile mystery = Rtx5090();
    mystery.name = "Test Mystery Card Unit";
    if (!std::filesystem::is_directory(
            lsxcommon::lsx_model_metadata_root()))
        SKIP();  // read-only CI vault: calibration persistence unavailable
    // Drop a stale cache entry to prove the bench path actually runs.
    std::error_code ec;
    std::filesystem::remove(lsxcommon::perf::PerfCachePath(mystery), ec);
    REQUIRE(EnsureCalibrated(mystery, [&calls](int &m) {
        ++calls;
        m = 24;
        return true;
    }));
    REQUIRE(calls == 1);
    // Second call: cache hit, bench not re-invoked.
    REQUIRE_FALSE(EnsureCalibrated(mystery, [&calls](int &) {
        ++calls;
        return true;
    }));
    REQUIRE(calls == 1);
    REQUIRE(std::filesystem::remove(lsxcommon::perf::PerfCachePath(mystery),
                                    ec));
}

// ── Fixed-speed policy guarantee: call-once apply, honest source ────────────

#include "lsxcommon/model_factory.h"

using lsxcommon::perf::CurrentKnobs;
using lsxcommon::perf::CurrentKnobsSource;
using lsxcommon::perf::EnsurePerfPolicyApplied;
using lsxcommon::perf::PerfSource;
using lsxcommon::perf::SetCurrentKnobs;

// Single case on purpose: Catch2 shuffles case order, and the process-global
// once-flag + knob snapshot must only be touched here (no other case in this
// binary calls EnsurePerfPolicyApplied), keeping the case order-independent.
TEST_CASE("EnsurePerfPolicyApplied: applies policy, call-once, "
          "overrides preserved",
          "[gpu_perf]")
{
    ::unsetenv("LSX_PERF_AUTO");
    ::unsetenv("LSX_GEMV_MAX_MIB");
    const GpuPerfProfile gpu = lsxcommon::perf::ProbeGpuPerfProfile();
    if (gpu.name.empty()) SKIP();
    EnsurePerfPolicyApplied();

    // Snapshot matches the documented precedence: vault cache > base/heuristic.
    LsxPerfKnobs expected{};
    PerfSource expected_source = PerfSource::kHeuristic;
    if (lsxcommon::perf::LoadPerfCache(gpu, expected)) {
        expected_source = PerfSource::kCalibrated;
    } else {
        const lsxcommon::perf::LsxPerfKnobSet resolved =
            lsxcommon::perf::ResolvePerfKnobs(gpu);
        expected = resolved.knobs;
        expected_source = resolved.source;
    }
    const LsxPerfKnobs got = CurrentKnobs();
    REQUIRE(got.tc_fast_math_default == expected.tc_fast_math_default);
    REQUIRE(got.seq1_gemv_default == expected.seq1_gemv_default);
    REQUIRE(got.gemv_max_mib == expected.gemv_max_mib);
    // Routing atomics are wired from the same snapshot.
    REQUIRE(lsxcommon::lsx_nvfp4::LsxTcFastMath() == got.tc_fast_math_default);
    REQUIRE(CurrentKnobsSource() == expected_source);

    // Call-once: a second call must not re-resolve over explicit overrides.
    LsxPerfKnobs sentinel{};
    sentinel.gemv_max_mib = 7;
    SetCurrentKnobs(sentinel, PerfSource::kEnvOverride);
    lsxcommon::lsx_nvfp4::SetLsxGemmMaxMib(7);
    EnsurePerfPolicyApplied();
    REQUIRE(CurrentKnobs().gemv_max_mib == 7);
    REQUIRE(CurrentKnobsSource() == PerfSource::kEnvOverride);
}
