# GPU Performance Auto-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **NOTE (this repo):** AGENTS.md forbids subagents. Execute inline with
> superpowers:executing-plans.

**Goal:** Inference always starts at maximum speed for the given GPU: a
committed per-GPU knob base (TC fast-math default, GEMV size limit, split-K
thresholds, expert-cache slot percentage, dual-stream staging) resolved from a
GPU passport (SM count + bus width), with heuristic + one-time local
calibration for unknown GPUs and a developer `--calibrate` run that produces
the values to fix in code.

**Architecture:** New model-neutral layer `gpu_perf` (host-only, no family
stem) owns the GPU passport, the committed base table, the heuristic, the
calibration cache and `ResolvePerfKnobs`. Knobs are applied to the existing
process-global atomics (`SetLsxTcFastMath` family) at engine init; `LsxFastMathScope`
gains a tri-state overload so models stop forcing `cfg_.fast_math` into the
globals. `ModelInitConfig`/`GenerationCliArgs` gain `FastMathMode`
(kAuto/kOn/kOff; legacy `bool fast_math` maps true→kOn, false→kAuto).

**Tech Stack:** C++20, CUDA 13.3 (Blackwell CC 12.0/12.1 only), Catch2 v3,
spdlog, absl::StrCat/StrFormat (printf-style conversions ONLY — see
`tests/test_absl_format_guard.cpp`), path vault via `lsx_model_metadata`.

## Global Constraints

- C++20; all logging via spdlog; no `std::cout`/`printf`.
- absl format guard: `absl::StrFormat` uses printf conversions (`%s`, `%d`,
  `%u`, `%f`) — NEVER `{}`.
- No `#include <absl/container/*>` in any header reachable from `.cu` TUs.
- New lib files are model-neutral: NO family prefix (`gpu_perf.*`,
  `perf_calibration.*`) — naming guard exempt (no family stem).
- CUDA arch 120-real;121-real only. Blackwell CC 12.x (`cc_major == 12`).
- `resource_governor.cpp` pattern: host-only `.cpp` may include
  `<cuda_runtime.h>` — follow it for `gpu_perf.cpp`.
- Env overrides win over everything (existing functional tests rely on this).
- Tests that pin specific functionality (bit-exactness, env-driven) must keep
  full override capability (`LSX_PERF_AUTO=0` escape hatch restores today's
  behavior).
- Windows-ONLY model tests: never add real-model dependencies to tests that
  run on Windows (`ctest -R "qwen|gemma"` filter constraint).
- Build/test (this Linux box):
  `cmake --build build/linux-release-znver4 -j 160` (configured dir) and
  `ctest --test-dir build/linux-release-znver4 -R <test>`.
- Branch: `feat/gpu-perf-autotuning` (already created).

**Design refinement vs spec section 3 (recorded decision):** the runtime local
calibration micro-benchmark synthesizes NVFP4 buffers with the fixed
representative shape classes (attn/mlp/expert/LM-head) instead of reading live
model weights — per-model weight registration would touch every upload call
site (8 models) for no measurable accuracy gain of the knob estimate. Real
weights ARE used by the developer `--calibrate` flow (Task 10), which is what
feeds the committed base. The calibration still runs at first Load and caches
into the vault exactly as specced.

---

### Task 1: `gpu_perf` — GPU passport + pure helpers

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/gpu_perf.h`
- Create: `libs/lsxcommon/src/gpu_perf.cpp`
- Modify: `libs/lsxcommon/CMakeLists.txt` (add `src/gpu_perf.cpp` to the
  lsxcommon sources list near `src/resource_governor.cpp`, line ~62)
- Test: `tests/test_gpu_perf.cpp` (new)
- Modify: `tests/CMakeLists.txt` (append at end):

```cmake
add_executable(test_gpu_perf test_gpu_perf.cpp)
target_link_libraries(test_gpu_perf PRIVATE lsxcommon Catch2::Catch2WithMain)
add_test(NAME test_gpu_perf COMMAND test_gpu_perf)
```

**Interfaces:**
- Produces (used by ALL later tasks):

```cpp
namespace lsxcommon { namespace perf {

struct GpuPerfProfile {
    std::string name;
    int sm_count = 0;
    int bus_width_bits = 0;
    int memory_clock_khz = 0;
    int l2_cache_bytes = 0;
    std::uint64_t vram_total_bytes = 0;
    int cc_major = 0, cc_minor = 0;
};

GpuPerfProfile ProbeGpuPerfProfile();          // device 0; zeroed on failure
double PeakBandwidthGbs(const GpuPerfProfile &g); // bus*kHz*2*8 → GB/s
std::string GpuProfileKey(const GpuPerfProfile &g);

}} // namespace lsxcommon::perf
```

- [ ] **Step 1: Write the failing tests** (`tests/test_gpu_perf.cpp`):

```cpp
#include <catch2/catch_test_macros.hpp>
#include "lsxcommon/gpu_perf.h"

using lsxcommon::perf::GpuPerfProfile;
using lsxcommon::perf::GpuProfileKey;
using lsxcommon::perf::PeakBandwidthGbs;

TEST_CASE("PeakBandwidthGbs: bus_width * mem_clock * 2 DDR factor", "[gpu_perf]")
{
    GpuPerfProfile g;
    g.bus_width_bits = 512;
    g.memory_clock_khz = 1750000;  // 1.75 GHz * 2 (DDR) * 512 bit / 8 = 224000 MB/s = 224 GB/s
    REQUIRE(PeakBandwidthGbs(g) == 224.0);
    GpuPerfProfile rtx5090;
    rtx5090.bus_width_bits = 512;
    rtx5090.memory_clock_khz = 1750000 * 8;  // 14 GHz GDDR7 effective (khz here = raw rate * 8 PAM3), see NOTE
    REQUIRE(PeakBandwidthGbs(rtx5090) == 1792.0);
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
```

> NOTE on bandwidth math: `cudaDeviceProp::memoryClockRate` is the raw memory
> clock in kHz. DDR/PAM multiplication factors differ per memory generation;
> for the heuristic we use the documented formula
> `bus_bits * clock_khz * 1000 * 2 / 8 / 1e9` GB/s and accept that GDDR7's
> effective rate exceeds the raw clock. `kRtx5090PeakGbs = 1792.0` (spec
> constant) is introduced in Task 2 as a fixed reference, NOT derived from the
> probe. The first test case above pins the raw formula on a GDDR6-class
> device; drop the `rtx5090` second block — replace with a plain DDR case
> (256-bit, 1000 MHz → 128 GB/s) so the test never encodes the PAM caveat:

```cpp
    GpuPerfProfile ddr;
    ddr.bus_width_bits = 256;
    ddr.memory_clock_khz = 1000000;  // 1 GHz * 2 * 256 / 8 = 128 GB/s
    REQUIRE(PeakBandwidthGbs(ddr) == 128.0);
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build/linux-release-znver4 -j 160 --target test_gpu_perf && ctest --test-dir build/linux-release-znver4 -R test_gpu_perf --output-on-failure`
Expected: build FAILS (`gpu_perf.h: No such file`).

- [ ] **Step 3: Write minimal implementation**

`libs/lsxcommon/include/lsxcommon/gpu_perf.h`:

```cpp
#pragma once

// ── GPU performance passport (model-neutral) ────────────────────────────────
// Hardware capability snapshot used by the auto-tuning policy: SM count,
// memory bus width and clock. Host-only; NEVER include from .cu TUs
// (hosts .cpp links fine — mirrors resource_governor.cpp).

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
```

`libs/lsxcommon/src/gpu_perf.cpp`:

```cpp
#include "lsxcommon/gpu_perf.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>

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
    g.memory_clock_khz = p.memoryClockRate;
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
```

Add `#include <cstring>` to the cpp (for `std::strlen`). Add the source file
to `libs/lsxcommon/CMakeLists.txt` next to `src/resource_governor.cpp`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build/linux-release-znver4 -j 160 --target test_gpu_perf && ctest --test-dir build/linux-release-znver4 -R test_gpu_perf --output-on-failure`
Expected: PASS (2 test cases).

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/gpu_perf.h libs/lsxcommon/src/gpu_perf.cpp libs/lsxcommon/CMakeLists.txt tests/test_gpu_perf.cpp tests/CMakeLists.txt
git commit -m "feat: gpu_perf passport layer (bus width, SM count, peak bandwidth)"
```

---

### Task 2: Knobs, tri-state `FastMathMode`, base table, heuristic, `ResolvePerfKnobs`

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/gpu_perf.h`
- Modify: `libs/lsxcommon/src/gpu_perf.cpp`
- Modify: `libs/lsxcommon/include/lsxcommon/model.h` (FastMathMode enum +
  `fast_math_mode` field on `ModelInitConfig`)
- Test: `tests/test_gpu_perf.cpp` (extend)

**Interfaces:**
- Produces:

```cpp
namespace lsxcommon {
// model.h
enum class FastMathMode { kAuto, kOn, kOff };
struct ModelInitConfig {
    // ... existing fields ...
    bool fast_math = false;                                  // legacy: true -> kOn
    FastMathMode fast_math_mode = FastMathMode::kAuto;       // used when fast_math == false
};
FastMathMode EffectiveFastMathMode(const ModelInitConfig &cfg);

namespace perf {
struct LsxPerfKnobs {
    bool tc_fast_math_default = false;
    bool seq1_gemv_default = false;
    int gemv_max_mib = 32;
    int splitk_grid_min = 96;
    int expert_slots_pct = 100;
    bool dual_stream_staging = true;
};
enum class PerfSource { kBase, kHeuristic, kCalibrated, kEnvOverride, kLegacy };
struct LsxPerfKnobSet { LsxPerfKnobs knobs; PerfSource source = PerfSource::kBase; };

LsxPerfKnobSet ResolvePerfKnobs(const GpuPerfProfile &gpu,
                                FastMathMode cli_mode = FastMathMode::kAuto);

// Applies knobs to the process-global atomics (lsx_nvfp4 + lsx_tc setters).
void ApplyKnobs(const LsxPerfKnobs &knobs);
}}}
```

- Consumes: Task 1 types.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_gpu_perf.cpp`):

```cpp
#include <cstdlib>
#include "lsxcommon/gpu_perf.h"
#include "lsxcommon/model.h"

using namespace lsxcommon;
using lsxcommon::perf::GpuPerfProfile;
using lsxcommon::perf::LsxPerfKnobs;
using lsxcommon::perf::LsxPerfKnobSet;
using lsxcommon::perf::PerfSource;
using lsxcommon::perf::ResolvePerfKnobs;

static GpuPerfProfile Rtx5090()
{
    GpuPerfProfile g;
    g.name = "NVIDIA GeForce RTX 5090";
    g.sm_count = 170;
    g.bus_width_bits = 512;
    g.memory_clock_khz = 1750000;
    g.cc_major = 12; g.cc_minor = 0;
    g.vram_total_bytes = std::uint64_t{32} << 30;
    return g;
}

TEST_CASE("ResolvePerfKnobs: committed base entry (RTX 5090)", "[gpu_perf]")
{
    auto r = ResolvePerfKnobs(Rtx5090());
    REQUIRE(r.source == PerfSource::kBase);
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
    small.memory_clock_khz = 1000000;   // 64 GB/s
    auto r = ResolvePerfKnobs(small);
    REQUIRE(r.source == PerfSource::kHeuristic);
    REQUIRE(r.knobs.gemv_max_mib == 8);          // 32*64/1792 clamped low
    REQUIRE(r.knobs.splitk_grid_min == 64);      // sm < 160
    REQUIRE(r.knobs.dual_stream_staging == false); // bw < 768
    REQUIRE(r.knobs.tc_fast_math_default == false); // heuristic: only base-12 TC
    REQUIRE(r.knobs.expert_slots_pct == 100);

    GpuPerfProfile blackwell = small;
    blackwell.cc_major = 12;
    blackwell.bus_width_bits = 384;
    blackwell.memory_clock_khz = 2000000;  // 384*2*2e9/8 = 192 GB/s? -> still < 768
    auto rb = ResolvePerfKnobs(blackwell);
    REQUIRE(rb.knobs.tc_fast_math_default == true);
}

TEST_CASE("ResolvePerfKnobs: precedence env > CLI > calibrated > heuristic > base",
          "[gpu_perf]")
{
    // CLI kOff beats base TC default.
    auto r = ResolvePerfKnobs(Rtx5090(), FastMathMode::kOff);
    REQUIRE(r.source == PerfSource::kBase);
    REQUIRE(r.knobs.tc_fast_math_default == false);
    REQUIRE(r.knobs.seq1_gemv_default == false);

    // CLI kOn on a heuristic GPU still forces TC.
    GpuPerfProfile unknown = Rtx5090();
    unknown.name = "Mystery Card";
    auto rh = ResolvePerfKnobs(unknown, FastMathMode::kOn);
    REQUIRE(rh.knobs.tc_fast_math_default == true);

    // Env wins over everything.
    ::setenv("LSX_GEMV_MAX_MIB", "16", 1);
    auto re = ResolvePerfKnobs(Rtx5090());
    REQUIRE(re.source == PerfSource::kEnvOverride);
    REQUIRE(re.knobs.gemv_max_mib == 16);
    ::unsetenv("LSX_GEMV_MAX_MIB");
}

TEST_CASE("ResolvePerfKnobs: LSX_PERF_AUTO=0 legacy escape hatch", "[gpu_perf]")
{
    ::setenv("LSX_PERF_AUTO", "0", 1);
    auto r = ResolvePerfKnobs(Rtx5090());
    REQUIRE(r.source == PerfSource::kLegacy);
    REQUIRE(r.knobs.tc_fast_math_default == false);
    REQUIRE(r.knobs.seq1_gemv_default == false);
    REQUIRE(r.knobs.gemv_max_mib == 32);
    REQUIRE(r.knobs.splitk_grid_min == 96);
    REQUIRE(r.knobs.dual_stream_staging == true);
    ::unsetenv("LSX_PERF_AUTO");
}

TEST_CASE("EffectiveFastMathMode maps legacy bool + mode", "[gpu_perf]")
{
    ModelInitConfig cfg;
    REQUIRE(EffectiveFastMathMode(cfg) == FastMathMode::kAuto);
    cfg.fast_math = true;
    REQUIRE(EffectiveFastMathMode(cfg) == FastMathMode::kOn);
    cfg.fast_math = false;
    cfg.fast_math_mode = FastMathMode::kOff;
    REQUIRE(EffectiveFastMathMode(cfg) == FastMathMode::kOff);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build/linux-release-znver4 -j 160 --target test_gpu_perf && ctest --test-dir build/linux-release-znver4 -R test_gpu_perf --output-on-failure`
Expected: build FAILS (`LsxPerfKnobs` / `ResolvePerfKnobs` undefined).

- [ ] **Step 3: Implement**

In `gpu_perf.h` add (inside `namespace perf`, after `GpuProfileKey`):

```cpp
// Tuning knobs resolved for one GPU (see docs/superpowers/specs/
// 2026-09-10-gpu-perf-autotuning-design.md).
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

// Resolution precedence: env > CLI(kOn/kOff) > calibrated > heuristic > base.
// Reads env at call time. Pure besides getenv.
LsxPerfKnobSet ResolvePerfKnobs(const GpuPerfProfile &gpu,
                                FastMathMode cli_mode = FastMathMode::kAuto);

// Applies knobs to the process-global routing atomics.
void ApplyKnobs(const LsxPerfKnobs &knobs);

// Reference bandwidth of the committed-base reference card (RTX 5090) used
// by the heuristic's gemv limit scaling.
inline constexpr double kRtx5090PeakGbs = 1792.0;
```

(`FastMathMode` comes from `"lsxcommon/model.h"` — add that include to
`gpu_perf.h`. `model.h` does not include `gpu_perf.h`, so no cycle.)

In `gpu_perf.cpp` add:

```cpp
namespace {
const LsxPerfKnobSet &BaseKnobs(const std::string &key)
{
    static const absl::flat_hash_map<std::string, LsxPerfKnobs> kBase = [] {
        absl::flat_hash_map<std::string, LsxPerfKnobs> m;
        LsxPerfKnobs rtx5090;
        rtx5090.tc_fast_math_default = true;
        rtx5090.seq1_gemv_default = true;
        rtx5090.gemv_max_mib = 32;
        rtx5090.splitk_grid_min = 96;
        rtx5090.expert_slots_pct = 100;
        rtx5090.dual_stream_staging = true;
        m.emplace("rtx 5090", rtx5090);
        return m;
    }();
    static const LsxPerfKnobSet kMiss{};
    const auto it = kBase.find(key);
    return it != kBase.end()
               ? *(new LsxPerfKnobSet{it->second, PerfSource::kBase})
               : kMiss;
}
```

> DO NOT literally keep that leaky pointer hack — return `LsxPerfKnobSet` by
> value instead:

```cpp
LsxPerfKnobSet BaseKnobs(const std::string &key)
{
    static const absl::flat_hash_map<std::string, LsxPerfKnobs> kBase = [] {
        absl::flat_hash_map<std::string, LsxPerfKnobs> m;
        LsxPerfKnobs rtx5090;
        rtx5090.tc_fast_math_default = true;
        rtx5090.seq1_gemv_default = true;
        m.emplace("rtx 5090", rtx5090);   // rest defaults = 32/96/100/true
        return m;
    }();
    const auto it = kBase.find(key);
    if (it == kBase.end()) return {};
    return {it->second, PerfSource::kBase};
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
} // namespace

LsxPerfKnobSet ResolvePerfKnobs(const GpuPerfProfile &gpu,
                                FastMathMode cli_mode)
{
    LsxPerfKnobSet r = BaseKnobs(GpuProfileKey(gpu));
    if (r.source != PerfSource::kBase) r = HeuristicKnobs(gpu);

    if (const char *auto_env = std::getenv("LSX_PERF_AUTO");
        auto_env && std::strcmp(auto_env, "0") == 0) {
        r = {LsxPerfKnobs{}, PerfSource::kLegacy};
    }

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
            if (r.source == PerfSource::kBase ||
                r.source == PerfSource::kHeuristic)
                r.source = PerfSource::kBase;  // CLI does not escalate source
            break;
        case FastMathMode::kOff:
            r.knobs.tc_fast_math_default = false;
            r.knobs.seq1_gemv_default = false;
            break;
        case FastMathMode::kAuto: break;
    }
    return r;
}

void ApplyKnobs(const LsxPerfKnobs &knobs)
{
    lsx_nvfp4::SetLsxTcFastMath(knobs.tc_fast_math_default);
    lsx_nvfp4::SetLsxSeq1Gemv(knobs.seq1_gemv_default);
    lsx_nvfp4::SetLsxGemmMaxMib(knobs.gemv_max_mib);   // Task 4
    lsx_nvfp4::SetLsxDualStreamStaging(knobs.dual_stream_staging);  // Task 6
    lsx_tc::SetLsxSplitKGridMin(knobs.splitk_grid_min);             // Task 5
}
```

`ApplyKnobs` references Task 4/5/6 setters — to keep Task 2 compilable, declare
the setters NOW in their headers (Tasks 4–6 fill the bodies):

- `lsx_inference_nvfp4.h`: `void SetLsxGemmMaxMib(int mib); void SetLsxDualStreamStaging(bool on);`
- `lsx_tc_gemm.h`: `void SetLsxSplitKGridMin(int v);`

with trivial default-value atomics in the corresponding `.cu` files
(`g_lsx_gemv_max_bytes{int64_t{32}<<20}`, `g_lsx_dual_stream{true}`,
`g_lsx_splitk_grid_min{96}`), bodies only store/load — full wiring into the
dispatcher happens in Tasks 4–6.

`model.h`: add enum + field + helper:

```cpp
// Blackwell fast-path routing mode (see gpu_perf.h auto-tuning):
// kAuto = GPU performance policy decides (committed base / heuristic /
// calibration); kOn = force TC mxf4nvf4; kOff = force bit-exact scalar.
enum class FastMathMode { kAuto, kOn, kOff };

inline FastMathMode EffectiveFastMathMode(const ModelInitConfig &cfg)
{
    return cfg.fast_math ? FastMathMode::kOn : cfg.fast_math_mode;
}
```

`ModelInitConfig` gets `FastMathMode fast_math_mode = FastMathMode::kAuto;`
under the existing `bool fast_math = false;`. Include "lsxcommon/gpu_perf.h"
NOT needed in model.h (enum is local).

`gpu_perf.cpp` includes: `<absl/container/flat_hash_map.h>`,
`"lsxcommon/lsx_inference_nvfp4.h"`, `"lsxcommon/lsx_tc_gemm.h"`,
`<cstdlib>`, `<cstring>`, `<algorithm>`. gpu_perf.cpp is a host TU in
`lsxcommon` — absl containers allowed here (NOT in a .cu TU).

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build/linux-release-znver4 -j 160 --target test_gpu_perf && ctest --test-dir build/linux-release-znver4 -R test_gpu_perf --output-on-failure`
Expected: PASS (7 test cases).

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/gpu_perf.h libs/lsxcommon/src/gpu_perf.cpp libs/lsxcommon/include/lsxcommon/model.h libs/lsxcommon/include/lsxcommon/lsx_inference_nvfp4.h libs/lsxcommon/src/lsx_inference_nvfp4.cu libs/lsxcommon/include/lsxcommon/lsx_tc_gemm.h libs/lsxcommon/src/lsx_tc_gemm.cu tests/test_gpu_perf.cpp
git commit -m "feat: perf knob resolution (base table, heuristic, tri-state fast math)"
```

---

### Task 3: Calibration cache — serialize/parse/validate (pure)

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/gpu_perf.h`
- Modify: `libs/lsxcommon/src/gpu_perf.cpp`
- Test: `tests/test_gpu_perf.cpp` (extend)

**Interfaces:**
- Produces:

```cpp
namespace lsxcommon { namespace perf {
inline constexpr int kPerfCacheFormatVersion = 1;

struct PerfCacheEntry { GpuPerfProfile gpu; LsxPerfKnobs knobs; };
std::string SerializePerfCache(const PerfCacheEntry &e);
std::optional<PerfCacheEntry> ParsePerfCache(std::string_view json);
// Vault path: lsx_model_metadata("perf-cache/<key>.json")
std::string PerfCachePath(const GpuPerfProfile &gpu);
bool LoadPerfCache(const GpuPerfProfile &gpu, LsxPerfKnobs &out);
bool SavePerfCache(const GpuPerfProfile &gpu, const LsxPerfKnobs &knobs);
}}
```

- [ ] **Step 1: Write the failing tests** (append):

```cpp
using lsxcommon::perf::PerfCacheEntry;
using lsxcommon::perf::SerializePerfCache;
using lsxcommon::perf::ParsePerfCache;
using lsxcommon::perf::PerfCachePath;
using lsxcommon::perf::kPerfCacheFormatVersion;

TEST_CASE("Perf cache: JSON round-trip", "[gpu_perf]")
{
    PerfCacheEntry e;
    e.gpu = Rtx5090();
    e.knobs.gemv_max_mib = 24;
    e.knobs.splitk_grid_min = 64;
    e.knobs.dual_stream_staging = false;
    const auto parsed = ParsePerfCache(SerializePerfCache(e));
    REQUIRE(parsed.has_value());
    REQUIRE(parsed->gpu.name == e.gpu.name);
    REQUIRE(parsed->gpu.sm_count == 170);
    REQUIRE(parsed->knobs.gemv_max_mib == 24);
    REQUIRE(parsed->knobs.splitk_grid_min == 64);
    REQUIRE(parsed->knobs.dual_stream_staging == false);
    REQUIRE(parsed->knobs.tc_fast_math_default == true);
}

TEST_CASE("Perf cache: format-version invalidation", "[gpu_perf]")
{
    auto json = SerializePerfCache({Rtx5090(), LsxPerfKnobs{}});
    const auto poisoned = absl::StrReplaceAll(
        absl::StrCat("\"format_version\": ", kPerfCacheFormatVersion),
        {absl::StrCat("\"format_version\": ", kPerfCacheFormatVersion),
         "\"format_version\": 999"});
    const auto parsed = ParsePerfCache(poisoned);
    REQUIRE_FALSE(parsed.has_value());
    REQUIRE_FALSE(ParsePerfCache("not json at all").has_value());
    REQUIRE_FALSE(ParsePerfCache("").has_value());
}

TEST_CASE("Perf cache path lands in vault perf-cache/", "[gpu_perf]")
{
    const std::string p = PerfCachePath(Rtx5090());
    REQUIRE(p.find("perf-cache") != std::string::npos);
    REQUIRE(p.find("rtx 5090.json") != std::string::npos);
}
```

(Include `<absl/strings/str_cat.h>`, `<absl/strings/str_replace.h>`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build/linux-release-znver4 -j 160 --target test_gpu_perf && ctest --test-dir build/linux-release-znver4 -R test_gpu_perf --output-on-failure`
Expected: build FAILS (`SerializePerfCache` undefined).

- [ ] **Step 3: Implement**

JSON format (hand-rolled writer, simdjson DOM parser — both already used in
repo). Writer:

```cpp
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
```

Parser with simdjson DOM (`#include <simdjson.h>`; gpu_perf.cpp is host-only —
allowed):

```cpp
std::optional<PerfCacheEntry> ParsePerfCache(std::string_view json)
{
    simdjson::dom::parser parser;
    simdjson::dom::element root;
    if (parser.parse(json).get(root) != simdjson::SUCCESS) return std::nullopt;
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
    (void)gpu["sm_count"].get(e.gpu.sm_count);
    (void)gpu["bus_width_bits"].get(e.gpu.bus_width_bits);
    (void)gpu["memory_clock_khz"].get(e.gpu.memory_clock_khz);
    (void)gpu["cc_major"].get(e.gpu.cc_major);
    (void)gpu["cc_minor"].get(e.gpu.cc_minor);
    bool b = false;
    if (knobs["tc_fast_math_default"].get(b) == simdjson::SUCCESS)
        e.knobs.tc_fast_math_default = b;
    if (knobs["seq1_gemv_default"].get(b) == simdjson::SUCCESS)
        e.knobs.seq1_gemv_default = b;
    (void)knobs["gemv_max_mib"].get(e.knobs.gemv_max_mib);
    (void)knobs["splitk_grid_min"].get(e.knobs.splitk_grid_min);
    (void)knobs["expert_slots_pct"].get(e.knobs.expert_slots_pct);
    if (knobs["dual_stream_staging"].get(b) == simdjson::SUCCESS)
        e.knobs.dual_stream_staging = b;
    return e;
}
```

`PerfCachePath`: `return lsx_model_metadata("perf-cache/" + GpuProfileKey(gpu) + ".json");`
(`#include "lsxcommon/path_registry.h"`).

`LoadPerfCache`: read file (`std::ifstream`, `std::ios::binary`), parse, on
success `out = e.knobs; return true`. Any failure → false. `SavePerfCache`:
create parent dirs (`std::filesystem::create_directories`), write, return
success. Add `<fstream>`, `<optional>`, `<filesystem>`, `<ctime>` includes.

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build/linux-release-znver4 -j 160 --target test_gpu_perf && ctest --test-dir build/linux-release-znver4 -R test_gpu_perf --output-on-failure`
Expected: PASS (10 test cases).

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/gpu_perf.h libs/lsxcommon/src/gpu_perf.cpp tests/test_gpu_perf.cpp
git commit -m "feat: perf calibration cache (vault JSON, format-version invalidation)"
```

---

### Task 4: GEMV size-limit knob wired into the NVFP4 dispatcher

**Files:**
- Modify: `libs/lsxcommon/src/lsx_inference_nvfp4.cu:364-377`
- Modify: `libs/lsxcommon/include/lsxcommon/lsx_inference_nvfp4.h:71-72`
- Test: `tests/test_perf_knobs_apply.cu` (new)
- Modify: `tests/CMakeLists.txt` (find the OBJECT library pattern used by
  CUDA tests — mirror `test_qwen_cu` at line ~159; append:

```cmake
add_executable(test_perf_knobs_apply test_perf_knobs_apply.cu)
target_link_libraries(test_perf_knobs_apply PRIVATE lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_perf_knobs_apply PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../libs/lsxcommon/src)
add_test(NAME test_perf_knobs_apply COMMAND test_perf_knobs_apply)
```

If the CUDA test target needs a POST_BUILD dll copy on WIN32, mirror the
`test_qwen_moe_kernels` block — on Linux none needed.)

**Interfaces:**
- Consumes: `SetLsxGemmMaxMib(int)` declared in Task 2.
- Produces: runtime-mutable GEMV eligibility bound; `LsxGemmMaxMib()` getter
  for tests.

- [ ] **Step 1: Write the failing test** (`tests/test_perf_knobs_apply.cu`):

```cpp
#include <catch2/catch_test_macros.hpp>
#include "lsxcommon/lsx_inference_nvfp4.h"

using namespace lsxcommon::lsx_nvfp4;

TEST_CASE("GEMV eligibility follows SetLsxGemmMaxMib", "[perf_knobs]")
{
    SetLsxTcFastMath(true);
    SetLsxSeq1Gemv(true);
    // Bytes for [N,K] NVFP4 weight: packed = K/2*N, scales = ceil(K/16)*N.
    // K=4096, N=3584 -> packed 7 MiB + scales 0.875 MiB ~ 7.875 MiB.
    // K=8192, N=11008 -> packed 44 MiB + scales 5.5 MiB ~ 49.5 MiB.
    auto bytes = [](int K, int N) -> int64_t {
        return int64_t(K / 2) * N + int64_t((K + 15) / 16) * N;
    };
    const int64_t small = bytes(4096, 3584);
    const int64_t large = bytes(8192, 11008);

    SetLsxGemmMaxMib(32);
    REQUIRE(small <= int64_t{32} << 20);
    REQUIRE(large > int64_t{32} << 20);

    SetLsxGemmMaxMib(64);
    REQUIRE(large <= int64_t{64} << 20);   // now eligible

    SetLsxGemmMaxMib(4);
    REQUIRE(small > int64_t{4} << 20);     // now ineligible

    SetLsxGemmMaxMib(32);  // restore default
    SetLsxTcFastMath(false);
    SetLsxSeq1Gemv(false);
}
```

(The test pins the knob→bound arithmetic, not a launch: dispatch decisions
read the same atomics; the launch-level behavior is covered by existing GEMV
tests.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build/linux-release-znver4 -j 160 --target test_perf_knobs_apply && ctest --test-dir build/linux-release-znver4 -R test_perf_knobs_apply --output-on-failure`
Expected: build FAILS (`LsxGemmMaxMib` undefined) — add the getter declaration
together with the setter now.

- [ ] **Step 3: Implement**

`lsx_inference_nvfp4.h` (next to `SetLsxSeq1Gemv`):

```cpp
// Weight-size bound (MiB) for the seq==1 streaming GEMV eligibility.
// Default 32; set from the resolved perf policy (SetLsxGemmMaxMib). The
// LSX_GEMV_MAX_MIB env read moved to perf::ResolvePerfKnobs (gpu_perf.cpp).
void SetLsxGemmMaxMib(int mib);
int LsxGemmMaxMib();

// Dual-stream expert-staging toggle (H2D/compute overlap). Default true.
void SetLsxDualStreamStaging(bool on);
bool LsxDualStreamStaging();
```

`lsx_inference_nvfp4.cu` — replace the `static const int64_t kGemvMaxBytes`
lambda (lines 364-370) with an atomic knob:

```cpp
static std::atomic<int64_t> g_lsx_gemv_max_bytes{int64_t{32} << 20};
static std::atomic<bool> g_lsx_dual_stream{true};

void SetLsxGemmMaxMib(int mib)
{
    if (mib > 0)
        g_lsx_gemv_max_bytes.store(int64_t{mib} << 20,
                                   std::memory_order_relaxed);
}

int LsxGemmMaxMib()
{
    return static_cast<int>(g_lsx_gemv_max_bytes.load(
               std::memory_order_relaxed) >> 20);
}

void SetLsxDualStreamStaging(bool on)
{
    g_lsx_dual_stream.store(on, std::memory_order_relaxed);
}

bool LsxDualStreamStaging()
{
    return g_lsx_dual_stream.load(std::memory_order_relaxed);
}
```

and in `RunLsxNvfp4Gemm`:

```cpp
    const bool gemv_small =
        g_lsx_tc_fast_math.load(std::memory_order_relaxed) &&
        LsxSeq1Gemv() &&
        (int64_t)(K / 2) * N + (int64_t)((K + 15) / 16) * N <=
            g_lsx_gemv_max_bytes.load(std::memory_order_relaxed);
```

Update the stale comment above it: env tunability now lives in
`perf::ResolvePerfKnobs` (`LSX_GEMV_MAX_MIB`), knob applied at engine init.

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build/linux-release-znver4 -j 160 --target test_perf_knobs_apply && ctest --test-dir build/linux-release-znver4 -R test_perf_knobs_apply --output-on-failure`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/lsx_inference_nvfp4.h libs/lsxcommon/src/lsx_inference_nvfp4.cu tests/test_perf_knobs_apply.cu tests/CMakeLists.txt
git commit -m "feat: GEMV size-limit knob in NVFP4 dispatcher (env read moved to policy)"
```

---

### Task 5: split-K threshold knob (`lsx_tc_gemm`)

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/lsx_tc_gemm.h` (declarations)
- Modify: `libs/lsxcommon/src/lsx_tc_gemm.cu:945-951` and `:1025-1036`
- Test: `tests/test_perf_knobs_apply.cu` (extend)

**Interfaces:**
- Consumes: knob declared in Task 2 (`SetLsxSplitKGridMin`).
- Produces: `int LsxSplitKGridMin();` getter; `LsxTcBatchSplitK` and the
  `TcGemmDispatch` split-K selection both read the knob.

- [ ] **Step 1: Write the failing test** (append to
`tests/test_perf_knobs_apply.cu`):

```cpp
#include "lsxcommon/lsx_tc_gemm.h"

using lsxcommon::lsx_tc::LsxTcBatchSplitK;
using lsxcommon::lsx_tc::SetLsxSplitKGridMin;

TEST_CASE("LsxTcBatchSplitK follows the splitk_grid_min knob", "[perf_knobs]")
{
    // K=8192, N=2048 -> grid_x = 2048/64 = 32. With min 96: 32*8=256 >= 96
    // and (8192/64)%8==0 -> split 8.
    SetLsxSplitKGridMin(96);
    REQUIRE(LsxTcBatchSplitK(8192, 2048) == 8);
    // Raise the threshold so even 8-way misses it -> no split.
    SetLsxSplitKGridMin(300);
    REQUIRE(LsxTcBatchSplitK(8192, 2048) == 1);
    // Lower threshold: 4-way qualifies (32*4=128 >= 100).
    SetLsxSplitKGridMin(100);
    REQUIRE(LsxTcBatchSplitK(8192, 2048) == 4);
    SetLsxSplitKGridMin(96);  // restore default
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build/linux-release-znver4 -j 160 --target test_perf_knobs_apply && ctest --test-dir build/linux-release-znver4 -R test_perf_knobs_apply --output-on-failure`
Expected: FAIL — `SetLsxSplitKGridMin` currently a no-op stub (Task 2), so the
8/4 expectations fail.

- [ ] **Step 3: Implement**

`lsx_tc_gemm.cu` — replace the magic constants in BOTH selection sites with
the knob:

```cpp
namespace {
std::atomic<int> g_lsx_splitk_grid_min{96};
} // namespace

void SetLsxSplitKGridMin(int v)
{
    if (v > 0) g_lsx_splitk_grid_min.store(v, std::memory_order_relaxed);
}

int LsxSplitKGridMin()
{
    return g_lsx_splitk_grid_min.load(std::memory_order_relaxed);
}

int LsxTcBatchSplitK(int K, int N)
{
    const int grid_min = g_lsx_splitk_grid_min.load(std::memory_order_relaxed);
    if (K < 2048 || (K % 512) != 0) return 1;
    const int grid_x = (N + kTileN - 1) / kTileN;
    if (grid_x >= 2 * grid_min) return 1;
    for (int cand : {8, 4, 2}) {
        if ((K / 64) % cand == 0 && grid_x * cand >= grid_min) return cand;
    }
    return 1;
}
```

`TcGemmDispatch` (line ~945): same substitution —

```cpp
        const int grid_min = g_lsx_splitk_grid_min.load(std::memory_order_relaxed);
        if (seq == 1 && !acc && K >= 2048 && grid.x < 2 * grid_min &&
            (K % 512) == 0) {
            for (int cand : {8, 4, 2}) {
                if ((K / 64) % cand == 0 && grid.x * cand >= grid_min) {
```

(The old `grid_x >= 192` cap generalizes to `>= 2 * grid_min`, keeping
default behavior identical at 96.)

Header additions to `lsx_tc_gemm.h` (near `LsxTcBatchSplitK`, line ~91):

```cpp
// split-K grid_x threshold knob (perf auto-tuning). Default 96. Both the
// single-slot dispatch and LsxTcBatchSplitK read the same value so split
// decisions stay consistent (run-to-run deterministic).
void SetLsxSplitKGridMin(int v);
int LsxSplitKGridMin();
```

(The atomic definition lives once in `lsx_tc_gemm.cu`; Task 2's stub body is
replaced by this one — remove the Task-2 placeholder from wherever it landed.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build/linux-release-znver4 -j 160 --target test_perf_knobs_apply && ctest --test-dir build/linux-release-znver4 -R test_perf_knobs_apply --output-on-failure`
Expected: PASS (2 test cases).

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/lsx_tc_gemm.h libs/lsxcommon/src/lsx_tc_gemm.cu tests/test_perf_knobs_apply.cu
git commit -m "feat: split-K grid threshold knob (single knob for both dispatch sites)"
```

---

### Task 6: Dual-stream staging knob at the consumers

**Files:**
- Modify: `libs/lsxcommon/src/glm_moe_flash_inference.cu:1530-1547`
- Modify: `libs/lsxcommon/src/qwen_moe_inference.cu:1543-1546`
- Test: covered by compile + existing generation tests (behavioral toggle is
  a pure stream-creation guard; no new unit test — a null copy_stream path
  already exists and is exercised by the non-staging tests).

**Interfaces:**
- Consumes: `lsx_nvfp4::LsxDualStreamStaging()` (Task 4).

- [ ] **Step 1: GLM-Flash generation driver** — guard the stream creation:

```cpp
    // Dual-stream expert staging: a dedicated H2D stream overlaps miss
    // uploads with the compute stream; per-expert copy events fence each
    // GEMM chain, per-slot events fence evictions (see RunFlashRoutedFFN).
    // Perf knob: narrow-bus GPUs keep single-stream staging (the overlap
    // cannot pay for the extra sync on low-bandwidth links).
    cudaStream_t h2d_stream = nullptr;
    cudaEvent_t copy_ev[16] = {};
    std::vector<cudaEvent_t> slot_free_ev;
    const bool dual = lsx_nvfp4::LsxDualStreamStaging() &&
                      cudaStreamCreateWithFlags(&h2d_stream,
                                                cudaStreamNonBlocking) ==
                          cudaSuccess;
    if (!dual) h2d_stream = nullptr;
```

(downstream code already handles `h2d_stream == nullptr` as the
single-stream path — verify with `rg "h2d_stream" libs/lsxcommon/src/glm_moe_flash_inference.cu`
that every use is null-guarded; the `RunFlashRoutedFFN` signature takes
`cudaStream_t n = nullptr` for exactly this.)

- [ ] **Step 2: Qwen MoE generation driver** (`qwen_moe_inference.cu:1546`):

```cpp
        cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking) ==
            cudaSuccess
```

becomes

```cpp
        lsxcommon::lsx_nvfp4::LsxDualStreamStaging() &&
        cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking) ==
            cudaSuccess
```

(verify the surrounding `if` handles the false case — it falls back to
`copy_stream = nullptr`, which the staging path already treats as
single-stream.)

- [ ] **Step 3: Build + run the guard tests**

Run: `cmake --build build/linux-release-znver4 -j 160 && ctest --test-dir build/linux-release-znver4 -R "glm_moe_flash_kernels|qwen_moe_kernels" --output-on-failure`
Expected: PASS (kernel-level tests exercise both null and non-null
copy-stream paths).

- [ ] **Step 4: Commit**

```bash
git add libs/lsxcommon/src/glm_moe_flash_inference.cu libs/lsxcommon/src/qwen_moe_inference.cu
git commit -m "feat: dual-stream staging knob respected by GLM-Flash/Qwen drivers"
```

---

### Task 7: Expert-cache slot percentage knob

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/resource_governor.h:183-197`
- Modify: call sites: `libs/lsxcommon/src/qwen_moe_model.cpp:166`,
  `libs/lsxcommon/src/qwen_moe_exp_model.cpp:162`,
  `libs/lsxcommon/src/glm_moe_flash_model.cpp:48-71` (`ResolveExpertSlots`),
  `libs/lsxcommon/src/gemma_base_model.cpp:102`,
  `libs/lsxcommon/src/glm_moe_model.cpp` (tuner floor path — leave; GLM-5.2
  sizes via RuntimeTuner),
  `libs/lsxcommon/src/kimi_moe_inference.cu:1590-1602`,
  `libs/lsxcommon/src/glm_moe_inference.cu` (same ExpertCacheSlots use).
- Test: `tests/test_gpu_perf.cpp` (extend — `ExpertCacheSlots` is pure and
  host-only).

**Interfaces:**
- Produces: `ExpertCacheSlots(free_bytes, per_expert_bytes, total_experts, int slots_pct = 100)`.
- [ ] **Step 1: Write the failing test** (append to `tests/test_gpu_perf.cpp`):

```cpp
#include "lsxcommon/resource_governor.h"

using lsxcommon::lifecycle::ExpertCacheSlots;

TEST_CASE("ExpertCacheSlots honors the percentage knob", "[gpu_perf]")
{
    const size_t per = 100u << 20;  // 100 MiB per expert
    // 10 GiB free - 4 GiB reserve = 6 GiB usable -> 61 experts full.
    REQUIRE(ExpertCacheSlots(std::size_t{10} << 30, per, 288) == 61);
    REQUIRE(ExpertCacheSlots(std::size_t{10} << 30, per, 288, 50) == 30);
    REQUIRE(ExpertCacheSlots(std::size_t{10} << 30, per, 288, 0) == 0);
    // pct > 100 clamps to full budget; total-expert cap still applies.
    REQUIRE(ExpertCacheSlots(std::size_t{10} << 30, per, 288, 200) == 61);
    REQUIRE(ExpertCacheSlots(std::size_t{10} << 30, per, 4, 100) == 4);
    // Degenerate inputs unchanged.
    REQUIRE(ExpertCacheSlots(0, per, 288) == 0);
    REQUIRE(ExpertCacheSlots(std::size_t{10} << 30, 0, 288) == 288);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build/linux-release-znver4 -j 160 --target test_gpu_perf && ctest --test-dir build/linux-release-znver4 -R test_gpu_perf --output-on-failure`
Expected: build FAILS (4-arg overload missing).

- [ ] **Step 3: Implement**

`resource_governor.h` — replace `ExpertCacheSlots`:

```cpp
// Pure, host-only slot estimator: usable = (free - reserve) * slots_pct / 100,
// clamped to the model's total expert count. per_expert_bytes == 0 (unknown
// geometry) degenerates to "everything resident"; free below reserve yields 0
// and the caller applies its graceful LRU floor. slots_pct comes from the
// resolved perf policy (default 100 = today's behavior).
inline int ExpertCacheSlots(std::size_t free_bytes,
                            std::size_t per_expert_bytes,
                            int total_experts,
                            int slots_pct = 100)
{
    if (total_experts <= 0) return 0;
    if (per_expert_bytes == 0) return total_experts;
    const std::size_t usable_raw =
        free_bytes > kExpertCacheVramReserveBytes
            ? free_bytes - kExpertCacheVramReserveBytes
            : 0;
    const std::size_t usable =
        usable_raw / 100u * static_cast<unsigned>(std::clamp(slots_pct, 0, 200));
    const std::size_t affordable = usable / per_expert_bytes;
    return static_cast<int>(affordable < static_cast<std::size_t>(total_experts)
                                ? affordable
                                : static_cast<std::size_t>(total_experts));
}
```

(Add `#include <algorithm>` to the header if absent.)

Call sites: pass the resolved pct. Each site resolves knobs once:

```cpp
    const int slots_pct =
        lsxcommon::perf::ResolvePerfKnobs(
            lsxcommon::perf::ProbeGpuPerfProfile())
            .knobs.expert_slots_pct;
```

— but repeated probing per call site is wasteful and ProbeGpuPerfProfile
needs CUDA. Instead add to `gpu_perf.h/.cpp`:

```cpp
// Process-wide resolved knob snapshot, applied at engine init. Models read
// this instead of re-resolving. Thread-safe via mutex-free const snapshot
// set once during InitializeEngine.
const LsxPerfKnobs &CurrentKnobs();
void SetCurrentKnobs(const LsxPerfKnobs &knobs);  // engine init / tests only
```

(implementation: `static LsxPerfKnobs g_cur{...defaults...};` with
`std::atomic<LsxPerfKnobs>`? — struct is trivially copyable < 64B; use a
`std::mutex`-guarded static like ResourceGovernor, or simply an atomic
snapshot via seq-cst load into a local copy:
`LsxPerfKnobs CurrentKnobs() { return g_cur.load(); }` with
`std::atomic<LsxPerfKnobs>` — allowed for trivially-copyable, lock-free not
guaranteed but correct. Return by VALUE.) Then call sites:

```cpp
    const int slots_pct = perf::CurrentKnobs().expert_slots_pct;
```

and `ApplyKnobs` (Task 2) also calls `SetCurrentKnobs(knobs)`. Default
`g_cur` = legacy defaults so tests that never init the engine see today's
behavior. In `glm_moe_flash_model.cpp` `ResolveExpertSlots`, multiply the
VRAM budget: `const int by_vram = static_cast<int>(budget * pct / 100 / slot_bytes);`
where `pct = perf::CurrentKnobs().expert_slots_pct` — keep the `slots_pct=100`
equivalence exact.

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build/linux-release-znver4 -j 160 --target test_gpu_perf && ctest --test-dir build/linux-release-znver4 -R test_gpu_perf --output-on-failure`
Expected: PASS (11 test cases).

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/resource_governor.h libs/lsxcommon/include/lsxcommon/gpu_perf.h libs/lsxcommon/src/gpu_perf.cpp libs/lsxcommon/src/qwen_moe_model.cpp libs/lsxcommon/src/qwen_moe_exp_model.cpp libs/lsxcommon/src/glm_moe_flash_model.cpp libs/lsxcommon/src/gemma_base_model.cpp libs/lsxcommon/src/kimi_moe_inference.cu libs/lsxcommon/src/glm_moe_inference.cu tests/test_gpu_perf.cpp
git commit -m "feat: expert-cache slot percentage from perf policy (CurrentKnobs snapshot)"
```

---

### Task 8: Engine wiring — probe/resolve/apply at init + tri-state `LsxFastMathScope` + model call sites

**Files:**
- Modify: `libs/lsxcommon/src/model_engine.cpp:15-21` (init wiring)
- Modify: `libs/lsxcommon/include/lsxcommon/lsx_inference_nvfp4.h:78-97`
  (tri-state scope)
- Modify: all model constructors + Infer scopes that pass `cfg_.fast_math`:
  `qwen_base_model.cpp:39-40,84`, `qwen_moe_model.cpp:34,298,319`,
  `qwen_moe_36_model.cpp:24`, `qwen_moe_exp_model.cpp:36,232`,
  `glm_moe_flash_model.cpp:80-81,151`, `gemma_base_model.cpp:149`,
  `kimi_moe_model.cpp:95-96,147-148`, `deepseek_moe_model.cpp:93,119`
  (DeepSeek uses its own `DeepseekFastMathScope` — add a tri-state overload
  there too), `model_runtime.cpp:182-187` (`CreateCliModel` maps
  `GenerationCliArgs`).
- Modify: `libs/lsxcommon/include/lsxcommon/model_runtime.h:79-86`
  (`GenerationCliArgs` gains `FastMathMode fast_math_mode = kAuto`).
- Test: `tests/test_gpu_perf.cpp` (scope semantics, host-only — the scope
  touches only atomics, no CUDA needed).

**Interfaces:**
- Consumes: `EffectiveFastMathMode`, `perf::ResolvePerfKnobs`, `perf::ApplyKnobs`,
  `perf::ProbeGpuPerfProfile`, `perf::LoadPerfCache`.
- Produces:

```cpp
// lsx_inference_nvfp4.h
class LsxFastMathScope {
public:
    // Legacy bool pair (unchanged semantics).
    LsxFastMathScope(bool tc_fast_math, bool seq1_gemv);
    // Tri-state: kAuto leaves globals untouched (engine-init policy stands);
    // kOn/kOff force the pair for the scope and restore after.
    explicit LsxFastMathScope(FastMathMode mode);
};
```

- [ ] **Step 1: Write the failing test** (append to `tests/test_gpu_perf.cpp`):

```cpp
#include "lsxcommon/lsx_inference_nvfp4.h"

using lsx_nvfp4::LsxFastMathScope;
using lsx_nvfp4::LsxTcFastMath;
using lsx_nvfp4::LsxSeq1Gemv;
using lsx_nvfp4::SetLsxTcFastMath;
using lsx_nvfp4::SetLsxSeq1Gemv;

TEST_CASE("LsxFastMathScope tri-state semantics", "[gpu_perf]")
{
    SetLsxTcFastMath(true);
    SetLsxSeq1Gemv(true);
    {
        const LsxFastMathScope s(FastMathMode::kAuto);
        REQUIRE(LsxTcFastMath() == true);   // untouched
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
    }
    SetLsxTcFastMath(false);
    SetLsxSeq1Gemv(false);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build/linux-release-znver4 -j 160 --target test_gpu_perf && ctest --test-dir build/linux-release-znver4 -R test_gpu_perf --output-on-failure`
Expected: build FAILS (tri-state ctor missing).

- [ ] **Step 3: Implement**

`lsx_inference_nvfp4.h` — add the overload (include `"lsxcommon/model.h"`
for `FastMathMode`; lsx_inference_nvfp4.h is reachable from .cu TUs — model.h
is `.cu`-safe: it includes inference_gpu/safetensors/tokenizer/benchmark
headers only, no absl containers; VERIFY with
`rg 'absl/container' libs/lsxcommon/include/lsxcommon/model.h libs/lsxcommon/include/lsxcommon/inference_gpu.h` — if unsafe, move
`enum class FastMathMode` into a new tiny header
`include/lsxcommon/fast_math_mode.h` and have both model.h and
lsx_inference_nvfp4.h include it. PREFER the tiny header unconditionally —
zero risk):

```cpp
    explicit LsxFastMathScope(FastMathMode mode)
        : prev_tc_(LsxTcFastMath()), prev_gemv_(LsxSeq1Gemv())
    {
        if (mode == FastMathMode::kOn) {
            SetLsxTcFastMath(true);
            SetLsxSeq1Gemv(true);
        } else if (mode == FastMathMode::kOff) {
            SetLsxTcFastMath(false);
            SetLsxSeq1Gemv(false);
        } else {
            prev_tc_ = prev_gemv_ = false;  // nothing to restore
        }
    }
```

(The restore path writes the saved values unconditionally; saving the current
values under kAuto keeps the destructor harmless.)

`model_engine.cpp` `InitializeEngine`:

```cpp
bool InferenceEngine::InitializeEngine()
{
    ResourceGovernor::Instance().Initialize(GovernorOptions{});
    const perf::GpuPerfProfile gpu = perf::ProbeGpuPerfProfile();
    perf::LsxPerfKnobs knobs;
    if (!gpu.name.empty()) {
        if (!perf::LoadPerfCache(gpu, knobs)) {
            knobs = perf::ResolvePerfKnobs(gpu).knobs;
        }
        perf::ApplyKnobs(knobs);
        log::info(absl::StrFormat(
            "[model_engine] perf policy source resolved: gpu='%s' sm=%d "
            "tc=%d gemv_mib=%d splitk_min=%d slots_pct=%d dual=%d",
            gpu.name.c_str(), gpu.sm_count,
            static_cast<int>(knobs.tc_fast_math_default), knobs.gemv_max_mib,
            knobs.splitk_grid_min, knobs.expert_slots_pct,
            static_cast<int>(knobs.dual_stream_staging)));
    } else {
        log::info("[model_engine] no CUDA device: perf policy left at legacy "
                  "defaults");
    }
    initialized_.store(true, std::memory_order_release);
    log::debug("[model_engine] initialized");
    return true;
}
```

(absl printf-style conversions ONLY.)

Model call-site migration (mechanical, every site):
- Constructor `SetLsxTcFastMath(cfg_.fast_math); SetLsxSeq1Gemv(cfg_.fast_math);`
  → wrap in a mode switch: only set when
  `EffectiveFastMathMode(cfg_) != FastMathMode::kAuto`:

```cpp
    const FastMathMode fm = EffectiveFastMathMode(cfg_);
    if (fm != FastMathMode::kAuto) {
        lsxcommon::lsx_nvfp4::SetLsxTcFastMath(fm == FastMathMode::kOn);
        lsxcommon::lsx_nvfp4::SetLsxSeq1Gemv(fm == FastMathMode::kOn);
    }
```

- Infer/SelfCheck `LsxFastMathScope fm_scope(cfg_.fast_math, cfg_.fast_math);`
  → `LsxFastMathScope fm_scope(EffectiveFastMathMode(cfg_));`
  (gemma's `(cfg_.fast_math, true)` — seq1 pinned ON: tri-state ctor handles
  the pair; the pinned-true seq1 means gemma keeps
  `SetLsxSeq1Gemv(true)` at construction and the scope restores it; use the
  tri-state ctor for TC and leave the explicit seq1 construction-set in
  place, noting gemma's bit-exact seq1 path does not depend on the knob.)
- DeepSeek `DeepseekFastMathScope(cfg_.fast_math)` (deepseek_moe_inference_moe.cu
  defines it over `g_deepseek_fast_math`): add
  `explicit DeepseekFastMathScope(FastMathMode mode)` with identical kAuto
  semantics; call sites pass `EffectiveFastMathMode(cfg_)`.
- `model_runtime.cpp` `CreateCliModel(model_path, fast_math)`:
  `cfg.fast_math = fast_math;` stays (legacy mapping kOn/kAuto), and
  `GenerationCliArgs` gains:

```cpp
    // Tri-state override: kOff forces bit-exact (CLI --no-fast-math),
    // kAuto follows the GPU perf policy. Only consulted when fast_math==false.
    FastMathMode fast_math_mode = FastMathMode::kAuto;
```

  and `RunCliGeneration`/`RunCliSelfCheck` copy
  `args.fast_math_mode` into the config before `CreateCliModel` (change
  `CreateCliModel` signature to
  `CliModel CreateCliModel(const std::string &model_path, FastMathMode mode)`
  — internal function; `bool` overloads removed; callers at :218/:245/:276
  pass `EffectiveFastMathMode`-compatible values: :276 keeps `kAuto`).

- [ ] **Step 4: Build everything and run the fast host test suite**

Run: `cmake --build build/linux-release-znver4 -j 160 && ctest --test-dir build/linux-release-znver4 -R "gpu_perf|perf_knobs|model_independence|model_factory|geometry|code_naming|absl" --output-on-failure`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A libs/lsxcommon
git commit -m "feat: tri-state fast-math scope + engine-init perf policy wiring"
```

---

### Task 9: Local calibration at first Load (unknown GPU)

**Files:**
- Create: `libs/lsxcommon/src/perf_calibration.cu`
- Create: `libs/lsxcommon/include/lsxcommon/perf_calibration.h`
- Modify: `libs/lsxcommon/CMakeLists.txt` (add `src/perf_calibration.cu` to
  the `lsxcommon_cu` OBJECT library source list, line ~113)
- Modify: `libs/lsxcommon/include/lsxcommon/gpu_perf.h` (+ orchestration decl)
- Modify: `libs/lsxcommon/src/gpu_perf.cpp` (+ orchestration impl)
- Modify: `libs/lsxcommon/src/model_base.cpp` (`FinishLoad` hook)
- Test: `tests/test_gpu_perf.cpp` (orchestration with a FAKE bench fn — the
  CUDA micro-benchmark itself is exercised by `test_perf_knobs_apply` on the
  dev GPU; no real-model dependency)

**Interfaces:**
- Produces:

```cpp
// perf_calibration.h
namespace lsxcommon { namespace perf {

struct CalibrationShape { int K; int N; };

// The fixed representative shape classes (attn qkv, mlp gate/up, expert w13,
// LM head) used when a caller has no live weights. Values chosen to cover the
// GEMV 32 MiB decision boundary from both sides (see lsx_inference_nvfp4.cu
// comment: 18.9 MB KDA, 28.1 MB gate/up, LM head 64 MiB).
const std::vector<CalibrationShape> &RepresentativeShapes();

// Micro-benchmark: for each shape, allocate synthetic NVFP4 buffers, time
// GEMV vs TC vs scalar (20 warmup + 50 timed reps, cudaEvent timing), return
// the largest per-shape GEMV-winning byte size expressed in MiB (clamped
// 8..64). Returns false on CUDA failure. Real-weight A/B refinement happens
// via the developer --calibrate flow (Task 10).
bool CalibrateGemvLimitMib(int &out_mib);

// Orchestration (host-only, unit-testable): returns true when calibration
// should run (no base entry, no cache, perf auto not disabled) and, when
// `run_bench` is provided and returns true, persists the result.
bool EnsureCalibrated(const GpuPerfProfile &gpu,
                      bool (*run_bench)(int &out_mib) = nullptr);
}}
```

- [ ] **Step 1: Write the failing test** (append to `tests/test_gpu_perf.cpp`):

```cpp
using lsxcommon::perf::EnsureCalibrated;
using lsxcommon::perf::GpuProfileKey;

TEST_CASE("EnsureCalibrated: base GPUs skip, unknown GPUs run once", "[gpu_perf]")
{
    // RTX 5090 is in the committed base: no calibration.
    int calls = 0;
    auto bench = [&](int &) { ++calls; return true; };
    REQUIRE_FALSE(EnsureCalibrated(Rtx5090(),
                   [](int &m) { m = 0; return true; }));
    REQUIRE(calls == 0);

    // Unknown GPU: bench invoked and knobs derived. Use a temp vault by
    // pointing the cache read at a nonexistent file (cache path derived from
    // the GPU key; the test GPU is unique enough — use a distinct name).
    GpuPerfProfile mystery = Rtx5090();
    mystery.name = "Test Mystery Card Unit";
    REQUIRE(EnsureCalibrated(mystery,
            [](int &m) { m = 24; return true; }));
}
```

> The second REQUIRE runs the real save path into the vault
> (`model-metadata/perf-cache/test mystery card unit.json`). Acceptable on the
> dev box; if the vault is read-only in CI, guard with a skip:
> `if (!std::filesystem::is_directory(lsx_model_metadata_root())) SKIP();`

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build/linux-release-znver4 -j 160 --target test_gpu_perf && ctest --test-dir build/linux-release-znver4 -R test_gpu_perf --output-on-failure`
Expected: build FAILS (`EnsureCalibrated` undefined).

- [ ] **Step 3: Implement**

`perf_calibration.cu` (goes into `lsxcommon_cu` — it launches nothing itself;
it calls `RunLsxNvfp4Gemm` and cudaEvent times it):

```cpp
#include "lsxcommon/perf_calibration.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <vector>

#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/log.h"

namespace lsxcommon {
namespace perf {

const std::vector<CalibrationShape> &RepresentativeShapes()
{
    static const std::vector<CalibrationShape> kShapes = {
        {4096, 4096},    // attn qkv/o class
        {2048, 5632},    // mlp gate/up class (~28 MiB class)
        {2048, 2048},    // expert w2 class
        {4096, 151936},  // LM-head class (GEMV ineligible: size)
    };
    return kShapes;
}

namespace {
double TimeGemmPath(int reps, bool use_gemv, bool use_tc,
                    const __half *x, const uint8_t *packed,
                    const uint8_t *scales, float gs, __half *y,
                    int K, int N, cudaStream_t stream)
{
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    // warmup
    for (int i = 0; i < 5; ++i) {
        SetLsxTcFastMath(use_tc);
        SetLsxSeq1Gemv(use_gemv);
        RunLsxNvfp4Gemm(stream, x, packed, scales, gs, y, 1, K, N);
    }
    cudaStreamSynchronize(stream);
    cudaEventRecord(a, stream);
    for (int i = 0; i < reps; ++i)
        RunLsxNvfp4Gemm(stream, x, packed, scales, gs, y, 1, K, N);
    cudaEventRecord(b, stream);
    cudaStreamSynchronize(stream);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    SetLsxTcFastMath(false);
    SetLsxSeq1Gemv(false);
    return ms / reps;
}
} // namespace

bool CalibrateGemvLimitMib(int &out_mib)
{
    // Save/restore the process routing state around the measurement.
    const bool prev_tc = lsx_nvfp4::LsxTcFastMath();
    const bool prev_gemv = lsx_nvfp4::LsxSeq1Gemv();

    cudaStream_t stream = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) !=
        cudaSuccess)
        return false;

    int best_mib = 8;
    const auto &shapes = RepresentativeShapes();
    for (const auto &sh : shapes) {
        const int64_t packed_bytes = int64_t{sh.K / 2} * sh.N;
        const int64_t scale_bytes = int64_t{(sh.K + 15) / 16} * sh.N;
        uint8_t *packed = nullptr, *scales = nullptr;
        __half *x = nullptr, *y = nullptr;
        const bool ok =
            cudaMalloc(&packed, packed_bytes) == cudaSuccess &&
            cudaMalloc(&scales, scale_bytes) == cudaSuccess &&
            cudaMalloc(&x, sizeof(__half) * sh.K) == cudaSuccess &&
            cudaMalloc(&y, sizeof(__half) * sh.N) == cudaSuccess;
        if (!ok) {
            cudaFree(packed); cudaFree(scales);
            cudaFree(x); cudaFree(y);
            cudaStreamDestroy(stream);
            return false;
        }
        cudaMemsetAsync(packed, 0x88, packed_bytes, stream);  // mid-level E2M1
        cudaMemsetAsync(scales, 0x38, scale_bytes, stream);   // E4M3 ~1.0
        cudaMemsetAsync(x, 0, sizeof(__half) * sh.K, stream);

        const double gemv_ms = TimeGemmPath(50, true, true, x, packed, scales,
                                            1.0f, y, sh.K, sh.N, stream);
        const double tc_ms = TimeGemmPath(50, false, true, x, packed, scales,
                                          1.0f, y, sh.K, sh.N, stream);
        const int64_t total_bytes = packed_bytes + scale_bytes;
        if (gemv_ms < tc_ms && total_bytes <= int64_t{64} << 20)
            best_mib = std::max(best_mib,
                static_cast<int>(total_bytes >> 20));
        cudaFree(packed); cudaFree(scales);
        cudaFree(x); cudaFree(y);
    }
    cudaStreamDestroy(stream);
    lsx_nvfp4::SetLsxTcFastMath(prev_tc);
    lsx_nvfp4::SetLsxSeq1Gemv(prev_gemv);
    out_mib = std::clamp(best_mib, 8, 64);
    return true;
}

} // namespace perf
} // namespace lsxcommon
```

`perf_calibration.h`:

```cpp
#pragma once

// Host+device calibration entry points (see gpu_perf.h). Host-only header —
// NEVER include from .cu TUs (pulls <vector> + lsx_tc declarations only, both
// safe; still keep it out of device code).

#include <vector>

#include "lsxcommon/gpu_perf.h"

namespace lsxcommon {
namespace perf {

struct CalibrationShape { int K; int N; };
const std::vector<CalibrationShape> &RepresentativeShapes();
bool CalibrateGemvLimitMib(int &out_mib);

bool EnsureCalibrated(const GpuPerfProfile &gpu,
                      bool (*run_bench)(int &out_mib) = nullptr);

} // namespace perf
} // namespace lsxcommon
```

`gpu_perf.cpp` — `EnsureCalibrated` orchestration:

```cpp
bool EnsureCalibrated(const GpuPerfProfile &gpu,
                      bool (*run_bench)(int &out_mib))
{
    if (gpu.name.empty()) return false;
    if (::getenv("LSX_PERF_AUTO") &&
        std::strcmp(::getenv("LSX_PERF_AUTO"), "0") == 0)
        return false;
    if (BaseKnobs(GpuProfileKey(gpu)).source == PerfSource::kBase)
        return false;                       // known GPU: committed base wins
    LsxPerfKnobs cached{};
    if (LoadPerfCache(gpu, cached)) return false;  // already calibrated

    // No base entry, no cache: calibrate now (first Load on this GPU).
    if (!run_bench) return true;            // query-only mode (tests)
    int mib = 0;
    if (!run_bench(mib)) {
        log::warn("[perf] calibration benchmark failed; keeping heuristic");
        return false;
    }
    LsxPerfKnobs knobs = HeuristicKnobs(gpu).knobs;
    knobs.gemv_max_mib = mib;
    if (!SavePerfCache(gpu, knobs)) return false;
    ApplyKnobs(knobs);
    log::info("[perf] calibrated gemv_max_mib=" + std::to_string(mib) +
              " (gpu '" + gpu.name + "')");
    return true;
}
```

`model_base.cpp` `FinishLoad` (find it with
`rg "FinishLoad" libs/lsxcommon/src/model_base.cpp`) — append before the
ready-flag set:

```cpp
    // Perf policy: one-time local calibration on unknown GPUs (no committed
    // base entry, no cache). Hundreds of ms, once per GPU per vault.
    perf::EnsureCalibrated(perf::ProbeGpuPerfProfile(),
                           perf::CalibrateGemvLimitMib);
```

(guard: only when `perf::CurrentKnobs().source != kCalibrated` — CurrentKnobs
snapshot gains a `PerfSource source` field so FinishLoad can skip cheaply.)

- [ ] **Step 4: Run test to verify it passes + full build**

Run: `cmake --build build/linux-release-znver4 -j 160 && ctest --test-dir build/linux-release-znver4 -R "gpu_perf|perf_knobs" --output-on-failure`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/perf_calibration.cu libs/lsxcommon/include/lsxcommon/perf_calibration.h libs/lsxcommon/CMakeLists.txt libs/lsxcommon/include/lsxcommon/gpu_perf.h libs/lsxcommon/src/gpu_perf.cpp libs/lsxcommon/src/model_base.cpp tests/test_gpu_perf.cpp
git commit -m "feat: one-time local GEMV calibration on unknown GPUs (vault-cached)"
```

---

### Task 10: lsxfabric `--calibrate` (developer base-authoring run) + `--no-fast-math`

**Files:**
- Modify: `apps/lsxfabric/src/main.cpp` (flags; see current `--fast-math`
  handling around lines 23-79)
- Modify: `libs/lsxcommon/include/lsxcommon/model_runtime.h` +
  `libs/lsxcommon/src/model_runtime.cpp` (new `RunCliCalibrate`)

**Interfaces:**
- Produces: `int RunCliCalibrate(const std::string &model_path);` — runs
  SelfCheck + a short benchmark under each candidate knob combination, prints
  the winning knobs (printf-style absl), saves them to the vault cache AND
  prints the C++ snippet to paste into `BaseKnobs()`.

- [ ] **Step 1: Implement `RunCliCalibrate`** in `model_runtime.cpp`:

```cpp
int RunCliCalibrate(const std::string &model_path)
{
    struct Candidate { const char *name; lsx_nvfp4::LsxFastMathScope (*make)(); };
    // Measure t/s at: (a) TC+GEMV with the current gemv bound, (b) TC only,
    // (c) bit-exact scalar. Reuses gpu::RunBenchmark with LsxFastMathScope.
    auto run = [&](bool tc, bool gemv) -> double {
        lsx_nvfp4::SetLsxTcFastMath(tc);
        lsx_nvfp4::SetLsxSeq1Gemv(gemv);
        const gpu::BenchmarkResult r = gpu::RunBenchmark(model_path, 512, tc);
        lsx_nvfp4::SetLsxTcFastMath(false);
        lsx_nvfp4::SetLsxSeq1Gemv(false);
        return r.tokens_per_sec;
    };
    const double t_gemv = run(true, true);
    const double t_tc = run(true, false);
    const double t_scalar = run(false, false);
    perf::GpuPerfProfile gpu = perf::ProbeGpuPerfProfile();
    perf::LsxPerfKnobs knobs;
    if (!perf::LoadPerfCache(gpu, knobs))
        knobs = perf::ResolvePerfKnobs(gpu).knobs;
    knobs.tc_fast_math_default = t_tc >= t_scalar;
    knobs.seq1_gemv_default = t_gemv >= t_tc && t_gemv >= t_scalar;
    spdlog::info(absl::StrFormat(
        "[calibrate] t/s: gemv=%.2f tc=%.2f scalar=%.2f -> tc=%d gemv=%d",
        t_gemv, t_tc, t_scalar, static_cast<int>(knobs.tc_fast_math_default),
        static_cast<int>(knobs.seq1_gemv_default)));
    if (!gpu.name.empty()) perf::SavePerfCache(gpu, knobs);
    spdlog::info(absl::StrFormat(
        "[calibrate] paste into BaseKnobs(): {\"%s\", {%d, %d, %d, %d, %d, %d}}",
        perf::GpuProfileKey(gpu).c_str(),
        static_cast<int>(knobs.tc_fast_math_default),
        static_cast<int>(knobs.seq1_gemv_default), knobs.gemv_max_mib,
        knobs.splitk_grid_min, knobs.expert_slots_pct,
        static_cast<int>(knobs.dual_stream_staging)));
    return 0;
}
```

(Exact `RunBenchmark` signature: `gpu::RunBenchmark(model_path, tokens, fast_math)`
per `benchmark_gpu.h:22` and `lsxfabric main.cpp:79` — verify before use; the
gemv/scalar A/B relies on the process toggles it observes. If `RunBenchmark`
needs a model already created by the factory, wrap with
`CreateCliModel(model_path, FastMathMode::kOn)` like `RunCliSelfCheck` does —
check `model_runtime.cpp:245`.)

Declare in `model_runtime.h` next to `RunCliSelfCheck`:

```cpp
// Developer base-authoring run: A/B benchmarks the GEMM paths on the real
// model, prints the winning knobs and the C++ snippet for the committed
// base table. Also persists to the vault perf cache.
int RunCliCalibrate(const std::string &model_path);
```

- [ ] **Step 2: Wire CLI flags** in `apps/lsxfabric/src/main.cpp`:

```cpp
        } else if (arg == "--calibrate" && i + 1 < argc) {
            calibrate_model = argv[++i];
        } else if (arg == "--no-fast-math") {
            args_fast_math_mode = FastMathMode::kOff;
        }
```

(existing `--fast-math` maps to `kOn`; the parsed mode feeds the
`GenerationCliArgs.fast_math_mode`; `--calibrate` calls
`Core::Initialize` → `runtime::RunCliCalibrate(path)` → `Core::Shutdown` and
returns its code.)

- [ ] **Step 3: Build + smoke**

Run: `cmake --build build/linux-release-znver4 -j 160 --target lsxfabric && ./build/linux-release-znver4/bin/lsxfabric --help`
Expected: builds; help lists `--calibrate` and `--no-fast-math`. (Do NOT run a
full calibration on real weights in this task — that is an inference-class
run; schedule it explicitly.)

- [ ] **Step 4: Commit**

```bash
git add apps/lsxfabric/src/main.cpp libs/lsxcommon/include/lsxcommon/model_runtime.h libs/lsxcommon/src/model_runtime.cpp
git commit -m "feat: lsxfabric --calibrate base-authoring run + --no-fast-math"
```

---

### Task 11: Guards, full suite, AGENTS.md

**Files:**
- Modify: `tests/test_model_independence_guard.cpp` (consumer list for the
  tri-state scope; assert `gpu_perf`/`perf_calibration` have no family stem
  via the existing scan logic)
- Modify: `AGENTS.md` (document the perf policy layer in Resources/Conventions)
- Test: full suite.

- [ ] **Step 1: Extend the independence guard** — find the consumer list
  (`rg "LsxFastMathScope" tests/test_model_independence_guard.cpp`) and add
  every model migrated in Task 8 to the tri-state consumers (same list), plus:

```cpp
TEST_CASE("perf layer is model-neutral (no family stem)", "[guard][perf]")
{
    // gpu_perf / perf_calibration are model-neutral infrastructure: no
    // family prefix, no family includes.
    REQUIRE_FALSE(std::filesystem::exists(
        "libs/lsxcommon/src/gpu_perf_family.cpp"));
    for (const auto &f : {"gpu_perf.h", "perf_calibration.h"}) {
        std::ifstream in(std::string("libs/lsxcommon/include/lsxcommon/") + f);
        REQUIRE(in.is_open());
        const std::string body((std::istreambuf_iterator<char>(in)),
                               std::istreambuf_iterator<char>());
        for (const char *stem : {"glm", "qwen", "gemma", "kimi", "deepseek"})
            REQUIRE(body.find(stem) == std::string::npos);
    }
}
```

(mirror the file-scanning style already used by that test — check how it
locates the repo root and copy it.)

- [ ] **Step 2: Update AGENTS.md** — add a short section under "## Conventions":

```markdown
## GPU Perf Auto-Tuning

- `gpu_perf` (model-neutral) resolves per-GPU knobs: TC fast-math default,
  GEMV size limit, split-K threshold, expert-slot pct, dual-stream staging.
- Priority: env > CLI(kOn/kOff) > vault calibration cache > heuristic >
  committed base table (`gpu_perf.cpp` `BaseKnobs`). `LSX_PERF_AUTO=0`
  restores legacy defaults. `LSX_TEST_FAST_MATH` forces TC (existing tests).
- New GPU: first Load runs a one-time GEMV calibration (perf-cache/ vault
  JSON). Author committed-base entries with `lsxfabric --calibrate <model>`.
- Tests pinning numeric classes must set `FastMathMode::kOff` explicitly.
```

- [ ] **Step 3: Full build + FULL ctest on this Linux box**

Run: `cmake --build build/linux-release-znver4 -j 160 && ctest --test-dir build/linux-release-znver4 --output-on-failure`
Expected: all PASS. Real-model selfchecks run with `kAuto` → TC path; their
3-cycle bit-exact determinism is run-to-run (TC kernels deterministic), so
they must stay green. If a test fails because it relied on bit-exact scalar
defaults, pin it with an explicit
`cfg.fast_math_mode = FastMathMode::kOff` (that is the spec section 9
exception rule) — do NOT loosen assertions.

- [ ] **Step 4: Commit**

```bash
git add tests/test_model_independence_guard.cpp AGENTS.md
git commit -m "test: perf-layer neutrality guard + AGENTS.md perf policy docs"
```

---

## Self-Review (completed during planning)

- **Spec coverage:** passport (T1), knobs+base+heuristic+precedence (T2),
  calibration cache (T3), GEMM routing + GEMV limit (T4), split-K (T5),
  dual-stream (T6), expert slots (T7), tri-state + engine wiring (T8),
  first-Load calibration (T9), developer --calibrate (T10), exception rule +
  guards (T11). Spec section 3's "real weights" nuance is deliberately
  refined to synthetic shapes at runtime + real-weight A/B in --calibrate
  (recorded in the header note).
- **Placeholders:** none — every step has concrete code/commands.
- **Type consistency:** `FastMathMode` defined once in model.h (or the tiny
  header if model.h proves .cu-unsafe — Task 8 step 3 prefers the tiny header
  `fast_math_mode.h` unconditionally); `SetLsxSplitKGridMin` declared in
  Task 2, defined in Task 5 exactly once; `CurrentKnobs()` returns by value
  (atomic snapshot); `ResolvePerfKnobs` signature consistent across T2/T8.
