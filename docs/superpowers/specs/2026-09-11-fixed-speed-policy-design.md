# Fixed-Speed Inference Policy by Default — Design

Date: 2026-09-11
Status: approved (user confirmed interpretation and approach B)

## Problem

The per-GPU perf policy (committed base / vault calibration cache / heuristic)
is resolved and applied only inside `InferenceEngine::InitializeEngine()`.
That entry point is opened only by lsxfabric and lsxbenchmark
(`InferenceEngine::Session`). Every other consumer — any direct
`LsxModelFactory::Create` user — silently runs inference on legacy defaults
(TC fast-math off, scalar GEMV routing): not the pinned speed.

## Goal

The auto perf policy is applied by default in every application that runs
inference, without each consumer having to remember engine initialization.

## Approach (B: factory-level call-once guarantee)

1. **`perf::EnsurePerfPolicyApplied()`** (`gpu_perf.h/.cpp`)
   - `std::call_once` guard; idempotent.
   - `ProbeGpuPerfProfile()`; empty name (no CUDA device) → mark applied,
     keep legacy defaults, debug log, return.
   - `LoadPerfCache(gpu)` → `kCalibrated`, else `ResolvePerfKnobs(gpu)`
     (env > cache > heuristic > committed base; `LSX_PERF_AUTO=0` → legacy).
   - `SetCurrentKnobs(knobs, source)` + `ApplyKnobs(knobs, source)`.
   - Logs the same `[perf] policy ...` line the engine used to log.
2. **`ApplyKnobs(const LsxPerfKnobs&, PerfSource source)`** — new source
   parameter; fixes the bookkeeping bug where `ApplyKnobs` clobbered the
   published `PerfSource` to `kBase` (misreports policy origin). Callers:
   `EnsurePerfPolicyApplied` (resolved source), `EnsureCalibrated`
   (`kCalibrated`).
3. **`model_engine.cpp`** — `InitializeEngine()` delegates to
   `EnsurePerfPolicyApplied()`; duplicated resolve block removed.
4. **`model_factory.cpp`** — `LsxModelFactory::Create()` calls
   `EnsurePerfPolicyApplied()` on entry. Single choke point every model
   consumer passes through.
5. **AGENTS.md** — GPU Perf Auto-Tuning section: policy applies by default in
   every consumer via the factory call-once guarantee.

## Guarantees / preserved escapes

- `LSX_PERF_AUTO=0`, `LSX_TEST_FAST_MATH`, `LSX_GEMV_MAX_MIB` behave as
  before — resolution stays centralized in `ResolvePerfKnobs`.
- Explicit test `SetCurrentKnobs` overrides keep working: the guarantee fires
  at most once; test overrides happen on top.
- First-Load GEMV calibration (`EnsureCalibrated`) untouched.
- Model independence invariant intact: models still open their own
  `LsxFastMathScope(EffectiveFastMathMode(cfg_))`; `kAuto` now observes the
  applied policy in every consumer instead of legacy zeros.

## Testing (TDD, `tests/test_gpu_perf.cpp`)

1. Idempotence: second `EnsurePerfPolicyApplied()` changes neither knobs nor
   source.
2. Source preserved: known GPU (committed base) → `CurrentKnobsSource() ==
   kBase` after ensure.
3. Factory guarantee: `LsxModelFactory::Create()` without engine session
   applies the policy (`CurrentKnobsSource() != kLegacy`).

Fast suite (41/41) must stay green; no CUDA in CI env → ensure degrades to
no-op, tests that assert knob values are guarded by device presence the same
way existing `test_gpu_perf` cases are.
