# Fixed-Speed Inference Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. (Subagents are FORBIDDEN in this repo per AGENTS.md — execute inline.)

**Goal:** Per-GPU perf policy (committed base / vault cache / heuristic) applies by default in every consumer via a call-once guarantee in `LsxModelFactory::Create`.

**Architecture:** New idempotent `perf::EnsurePerfPolicyApplied()` centralizes resolve+apply (env escapes preserved); `InitializeEngine` and the factory choke point both call it; `ApplyKnobs` stops clobbering `PerfSource`; lsxhome opens `InferenceEngine::Session` for lifecycle symmetry.

**Tech Stack:** C++20, Catch2 v3 (label `fast`), absl::StrFormat (printf-style — repo guard), spdlog via `lsxcommon::log`.

**Spec:** `docs/superpowers/specs/2026-09-11-fixed-speed-policy-design.md`

## Global Constraints

- Host-only code: `absl::StrFormat` uses printf conversions (`%s/%d`), never `{}` (guard `test_absl_format_guard`).
- All logging through `lsxcommon::log` (spdlog); no `std::cout`/`printf`.
- No pre-Blackwell code paths touched; this work is host-only bookkeeping.
- Build/test via presets: `cmake --build --preset linux-debug --target <t>`; `ctest --preset linux-debug -R "<regex>"`. Full gate: `ctest --preset linux-debug` (41/41).
- Tests are process-global-sensitive (`std::call_once`): new cases appended at end of `tests/test_gpu_perf.cpp` in the order given.

---

### Task 1: `perf::EnsurePerfPolicyApplied()` + honest `ApplyKnobs` source

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/gpu_perf.h` (declare after `ApplyKnobs`, ~line 81)
- Modify: `libs/lsxcommon/src/gpu_perf.cpp` (`ApplyKnobs` ~line 192, `EnsureCalibrated` ~line 325, new impl at end of `perf` namespace)
- Test: `tests/test_gpu_perf.cpp` (append at end)

**Interfaces:**
- Consumes: existing `ProbeGpuPerfProfile`, `LoadPerfCache`, `ResolvePerfKnobs`, `SetCurrentKnobs`, `CurrentKnobs`, `CurrentKnobsSource`.
- Produces: `void lsxcommon::perf::EnsurePerfPolicyApplied();` and `void ApplyKnobs(const LsxPerfKnobs&, PerfSource source);` (source now required — no default).

- [ ] **Step 1: Write the failing tests** — append to `tests/test_gpu_perf.cpp`:

```cpp
// ── Fixed-speed policy guarantee: call-once apply, honest source ────────────

#include "lsxcommon/model_factory.h"

using lsxcommon::perf::CurrentKnobs;
using lsxcommon::perf::CurrentKnobsSource;
using lsxcommon::perf::EnsurePerfPolicyApplied;
using lsxcommon::perf::PerfSource;
using lsxcommon::perf::SetCurrentKnobs;

TEST_CASE("EnsurePerfPolicyApplied: applies policy without engine session",
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
        expected = lsxcommon::perf::ResolvePerfKnobs(gpu).knobs;
        expected_source = lsxcommon::perf::ResolvePerfKnobs(gpu).source;
    }
    const LsxPerfKnobs got = CurrentKnobs();
    REQUIRE(got.tc_fast_math_default == expected.tc_fast_math_default);
    REQUIRE(got.seq1_gemv_default == expected.seq1_gemv_default);
    REQUIRE(got.gemv_max_mib == expected.gemv_max_mib);
    // Routing atomics are wired from the same snapshot.
    REQUIRE(lsxcommon::lsx_nvfp4::LsxTcFastMath() == got.tc_fast_math_default);
    REQUIRE(CurrentKnobsSource() == expected_source);
}

TEST_CASE("EnsurePerfPolicyApplied: call-once, explicit overrides preserved",
          "[gpu_perf]")
{
    const GpuPerfProfile gpu = lsxcommon::perf::ProbeGpuPerfProfile();
    if (gpu.name.empty()) SKIP();
    EnsurePerfPolicyApplied();
    LsxPerfKnobs sentinel{};
    sentinel.gemv_max_mib = 7;
    SetCurrentKnobs(sentinel, PerfSource::kEnvOverride);
    lsxcommon::lsx_nvfp4::SetLsxGemmMaxMib(7);
    EnsurePerfPolicyApplied();  // must be a no-op (once-flag)
    REQUIRE(CurrentKnobs().gemv_max_mib == 7);
    REQUIRE(CurrentKnobsSource() == PerfSource::kEnvOverride);
}
```

- [ ] **Step 2: Verify RED** — `cmake --build --preset linux-debug --target test_gpu_perf`
  Expected: build FAILS — `'EnsurePerfPolicyApplied' is not a member of 'lsxcommon::perf'`.

- [ ] **Step 3: Implement** — in `gpu_perf.h` replace the `ApplyKnobs` block:

```cpp
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
```

In `gpu_perf.cpp`:

```cpp
void ApplyKnobs(const LsxPerfKnobs &knobs, PerfSource source)
{
    SetCurrentKnobs(knobs, source);
    lsx_nvfp4::SetLsxTcFastMath(knobs.tc_fast_math_default);
    lsx_nvfp4::SetLsxSeq1Gemv(knobs.seq1_gemv_default);
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
```

Add include `<absl/strings/str_format.h>` next to the existing `str_cat.h` include. In `EnsureCalibrated` (~line 325) change `ApplyKnobs(knobs);` → `ApplyKnobs(knobs, PerfSource::kCalibrated);`.

- [ ] **Step 4: Verify GREEN** — `cmake --build --preset linux-debug --target test_gpu_perf && ctest --preset linux-debug -R gpu_perf --output-on-failure`
  Expected: all `[gpu_perf]` cases PASS (the two new ones included).

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/gpu_perf.h libs/lsxcommon/src/gpu_perf.cpp tests/test_gpu_perf.cpp
git commit -m "feat: perf::EnsurePerfPolicyApplied call-once guarantee with honest PerfSource"
```

---

### Task 2: Factory choke point

**Files:**
- Modify: `libs/lsxcommon/src/model_factory.cpp` (top of `LsxModelFactory::Create`, ~line 57; includes at top)
- Test: `tests/test_model_factory.cpp` (append at end)

**Interfaces:**
- Consumes: `perf::EnsurePerfPolicyApplied()` (Task 1).
- Produces: every `LsxModelFactory::Create` caller gets the pinned policy.

- [ ] **Step 1: Write the failing test** — append to `tests/test_model_factory.cpp`:

```cpp
TEST_CASE("Create applies the per-GPU perf policy without engine session",
          "[factory]")
{
    if (lsxcommon::perf::ProbeGpuPerfProfile().name.empty()) SKIP();
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = R"(D:\nonexistent\model\path)";
    auto m = lsxcommon::LsxModelFactory::Create(cfg);  // returns nullptr; policy applied on entry
    REQUIRE(m == nullptr);
    REQUIRE(lsxcommon::perf::CurrentKnobsSource() !=
            lsxcommon::perf::PerfSource::kLegacy);
}
```

Add includes at top of the file: `#include "lsxcommon/gpu_perf.h"`.

- [ ] **Step 2: Verify RED** — `cmake --build --preset linux-debug --target test_model_factory && ctest --preset linux-debug -R model_factory --output-on-failure`
  Expected: the new case FAILS (`CurrentKnobsSource() == kLegacy (4) != kLegacy` — nothing called ensure in this binary yet).

- [ ] **Step 3: Implement** — in `model_factory.cpp` add `#include "lsxcommon/gpu_perf.h"` and make `Create` begin with:

```cpp
std::unique_ptr<IModel> LsxModelFactory::Create(const ModelInitConfig &cfg)
{
    // Fixed-speed default: every model consumer gets the pinned per-GPU
    // perf policy (call-once; explicit SetCurrentKnobs overrides win).
    perf::EnsurePerfPolicyApplied();
    EnsureModelRegistry();
    // ... rest unchanged
```

- [ ] **Step 4: Verify GREEN** — same command as Step 2. Expected: all `[factory]` cases PASS.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/model_factory.cpp tests/test_model_factory.cpp
git commit -m "feat: LsxModelFactory::Create applies pinned perf policy by default"
```

---

### Task 3: `InitializeEngine` delegates to the guarantee

**Files:**
- Modify: `libs/lsxcommon/src/model_engine.cpp` (`InitializeEngine`, lines 18–53; includes at top)

**Interfaces:**
- Consumes: `perf::EnsurePerfPolicyApplied()` (Task 1).
- Produces: single resolve path; behavior identical to today for fabric/benchmark.

- [ ] **Step 1: Refactor** — replace the body of `InitializeEngine` (keep governor init first, perf block removed):

```cpp
bool InferenceEngine::InitializeEngine()
{
    ResourceGovernor::Instance().Initialize(GovernorOptions{});
    // Perf auto-tuning: call-once resolve+apply of the per-GPU knob policy —
    // the same guarantee direct-factory consumers get via LsxModelFactory.
    perf::EnsurePerfPolicyApplied();
    initialized_.store(true, std::memory_order_release);
    log::debug("[model_engine] initialized");
    return true;
}
```

Remove the now-unused `#include <absl/strings/str_format.h>`; keep `gpu_perf.h`, `log.h`, `model_factory.h`, `resource_governor.h`.

- [ ] **Step 2: Verify** — `cmake --build --preset linux-debug --target test_model_engine test_gpu_perf && ctest --preset linux-debug -R "model_engine|gpu_perf" --output-on-failure`
  Expected: PASS (policy still applied through the engine path; log now emitted by `[perf]`).

- [ ] **Step 3: Commit**

```bash
git add libs/lsxcommon/src/model_engine.cpp
git commit -m "refactor: InitializeEngine delegates to perf::EnsurePerfPolicyApplied"
```

---

### Task 4: lsxhome session + docs + full fast gate

**Files:**
- Modify: `lsxhome/src/main_win32.cpp` (`ComputeProducer`, ~line 84; include block at top)
- Modify: `AGENTS.md` (section `## GPU Perf Auto-Tuning`)

**Interfaces:**
- Consumes: `lsxcommon::InferenceEngine::Session` (existing RAII).

- [ ] **Step 1: lsxhome edit** — add `#include "lsxcommon/model_engine.h"` to the lsxcommon include block; inside `ComputeProducer`, at the top of the `if (!model_path.empty())` block:

```cpp
    if (!model_path.empty()) {
        // Fixed-speed policy: the engine session applies the per-GPU knob
        // policy (same as lsxfabric/lsxbenchmark) on entry and tears the
        // engine down on scope exit.
        lsxcommon::InferenceEngine::Session engine_session;
        lsxcommon::ModelInitConfig cfg;
```

Note: file is Windows-only (`if(NOT WIN32) return()` in its CMakeLists) — not compiled by the Linux build; keep the edit to exactly the lines above.

- [ ] **Step 2: AGENTS.md** — in `## GPU Perf Auto-Tuning`, immediately after the priority-resolution paragraph, add:

```markdown
- The policy is applied by default in every model consumer: the call-once
  guarantee `perf::EnsurePerfPolicyApplied()` (resolve + apply, an honest
  `PerfSource` — `ApplyKnobs` does not clobber it) is invoked from
  `LsxModelFactory::Create` and `InferenceEngine::InitializeEngine`; lsxhome
  opens `InferenceEngine::Session` on the compute thread. Explicit test
  `SetCurrentKnobs` calls work on top of the guarantee (the once flag does not re-resolve).
```

- [ ] **Step 3: Full fast gate** — `cmake --build --preset linux-debug && ctest --preset linux-debug`
  Expected: 41/41 PASS (including `test_absl_format_guard`, `test_code_naming_guard`, `test_ctest_labels_guard`).

- [ ] **Step 4: Commit**

```bash
git add lsxhome/src/main_win32.cpp AGENTS.md
git commit -m "feat: lsxhome inference at pinned policy; docs for default perf guarantee"
```
