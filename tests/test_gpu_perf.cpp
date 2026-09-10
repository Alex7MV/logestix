#include <catch2/catch_test_macros.hpp>
#include "lsxcommon/gpu_perf.h"
#include "lsxcommon/fast_math_mode.h"

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

TEST_CASE("ResolvePerfKnobs: heuristic scales with bandwidth and SM count",
          "[gpu_perf]")
{
    GpuPerfProfile small = Rtx5090();
    small.name = "Unknown GPU X";
    small.sm_count = 64;
    small.bus_width_bits = 128;
    small.memory_clock_khz = 1000000;   // 32 GB/s
    small.cc_major = 11;                // non-Blackwell
    auto r = lsxcommon::perf::ResolvePerfKnobs(small);
    REQUIRE(r.source == lsxcommon::perf::PerfSource::kHeuristic);
    REQUIRE(r.knobs.gemv_max_mib == 8);            // 32*32/1792 clamped low
    REQUIRE(r.knobs.splitk_grid_min == 64);        // sm < 160
    REQUIRE(r.knobs.dual_stream_staging == false); // bw < 768
    REQUIRE(r.knobs.tc_fast_math_default == false); // only base-12 TC
    REQUIRE(r.knobs.expert_slots_pct == 100);

    GpuPerfProfile blackwell = small;
    blackwell.cc_major = 12;
    auto rb = lsxcommon::perf::ResolvePerfKnobs(blackwell);
    REQUIRE(rb.knobs.tc_fast_math_default == true);
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
