# GPU-Performance Auto-Tuning Design

Date: 2026-09-10
Status: approved (design dialogue 2026-09-10)

## Problem

Different GPUs have different VRAM sizes (already handled), but also different
compute power (SM count) and memory bus width/bandwidth. Inference must always
start at maximum speed for the given GPU. All model optimizations must also
feed the optimization base so inference always runs at max speed — except for
tests that test specific functionality (they keep explicit overrides).

Today:
- `fast_math` (TC mxf4nvf4 path) defaults OFF everywhere in production;
  enabled only via `--fast-math` in lsxfabric.
- The seq==1 weight-streaming GEMV limit is a fixed 32 MiB
  (env `LSX_GEMV_MAX_MIB`), tuned empirically on RTX 5090.
- split-K thresholds (`LsxTcBatchSplitK`, `TcGemmDispatch`) are fixed constants.
- `ResourceGovernor` probes only GPU name and SM count; bus width and memory
  clock are never queried.
- Expert-cache slot count is already VRAM-adaptive.

## Decisions (from design dialogue)

1. Auto-enable TC fast-math by GPU profile (numbers become tolerance-class,
   as with today's `--fast-math`); explicit CLI fallback `--no-fast-math`.
2. The optimization base is committed to the repository (known-GPU table).
3. Unknown GPU: heuristic from hardware passport + one-time local calibration
   with a persistent local cache.
4. Tuning scope: GEMM routing + GEMV size limit, split-K thresholds,
   expert-cache slots percentage, dual-stream staging on/off.
5. Calibration runs on real loaded weights at first `Load` (hundreds of ms,
   once per GPU, cached).
6. Developer calibration run: a dedicated calibration mode that runs the
   model, collects the metrics, and prints the knob values to be fixed in
   code (committed to the repo base) — see section 6.

## 1. GPU passport — `gpu_perf`

New model-neutral files (no family stem, naming-guard exempt):
`libs/lsxcommon/include/lsxcommon/gpu_perf.h` + `libs/lsxcommon/src/gpu_perf.cpp`
(host-only `.cpp` in `lsxcommon`, includes `cuda_runtime.h` like
`resource_governor.cpp`).

```cpp
struct GpuPerfProfile {
    std::string name;          // cudaDeviceProp.name
    int sm_count = 0;          // multiProcessorCount
    int bus_width_bits = 0;    // memoryBusWidth
    int memory_clock_khz = 0;  // memoryClockRate
    int l2_cache_bytes = 0;    // l2CacheSize
    std::uint64_t vram_total_bytes = 0;
    int cc_major = 0, cc_minor = 0;
};
GpuPerfProfile ProbeGpuPerfProfile();   // device 0; never throws; zeroed on failure

// Peak theoretical bus bandwidth in GB/s (bus_width_bits * mem_clock * 2 / 8).
double PeakBandwidthGbs(const GpuPerfProfile &gpu);

// Normalized key: lowercase name, "NVIDIA GeForce " prefix stripped.
std::string GpuProfileKey(const GpuPerfProfile &gpu);
```

## 2. Knobs and the committed base

```cpp
struct LsxPerfKnobs {
    bool tc_fast_math_default;   // default-enable TC mxf4nvf4 path
    bool seq1_gemv_default;      // default-enable weight-streaming GEMV
    int  gemv_max_mib;           // GEMV weight-size limit (today: 32)
    int  splitk_grid_min;        // grid_x threshold for split-K (today: 96)
    int  expert_slots_pct;       // % of usable VRAM for expert cache (today: 100)
    bool dual_stream_staging;    // dual-stream H2D overlap
};

enum class PerfSource { kBase, kHeuristic, kCalibrated, kEnvOverride };

struct LsxPerfKnobSet { LsxPerfKnobs knobs; PerfSource source; };

LsxPerfKnobSet ResolvePerfKnobs(const GpuPerfProfile &gpu,
                                FastMathMode cli_mode /* kAuto|kOn|kOff */);
```

The base is a static table in `gpu_perf.cpp`, committed to the repo. Initial
entry: RTX 5090 (dev machine) with today's behavior — `tc_fast_math_default =
true` (per decision 1: production inference defaults to the TC path),
`gemv_max_mib = 32`, `splitk_grid_min = 96`, `expert_slots_pct = 100`,
`dual_stream_staging = true`.

Precedence: **env > CLI (kOn/kOff) > calibrated > heuristic > base**.
Set env variables win over everything (existing functional tests rely on
them). CLI `kOn/kOff` overrides the base/calibration; env still wins.

## 3. Heuristic + local calibration (unknown GPU)

Heuristic (pure, used when no base entry exists):

| Knob | Rule |
|---|---|
| `tc_fast_math_default` | `cc_major == 12` → true |
| `seq1_gemv_default` | true |
| `gemv_max_mib` | `clamp(32 * bw / bw_rtx5090, 8, 64)` |
| `splitk_grid_min` | sm >= 160 → 96; sm < 160 → 64 |
| `expert_slots_pct` | 100 |
| `dual_stream_staging` | bw >= 768 GB/s → true, else false |

Local calibration (first `Load` on an unknown GPU):
- Micro-benchmark on the first really-loaded NVFP4 weight of each shape class:
  GEMV vs TC around the 32 MiB threshold → refined `gemv_max_mib` and path
  choices for the model's typical shapes.
- Hundreds of ms, once per GPU. Result serialized as JSON into the vault via
  `lsx_model_metadata("perf-cache/<gpu-key>.json")` (`path_registry`).
- Next starts read the cache (`PerfSource::kCalibrated`), no measurement.
- Invalidation: GPU name + knob-format version in the JSON.

## 4. Wiring (knobs → kernels)

Global current-knob layer in `lsx_inference_nvfp4.{h,cu}` (atomics, same
pattern as `SetLsxTcFastMath`):

```cpp
void SetLsxGemmMaxMib(int mib);        // default 32; replaces the in-dispatcher env read
void SetLsxSplitKGridMin(int v);       // default 96; read by LsxTcBatchSplitK / TcGemmDispatch
void SetLsxDualStreamStaging(bool);    // default true
bool LsxDualStreamStaging();
```

- Engine init (after `ResourceGovernor::Initialize`): probe → resolve → apply
  knobs globally; unknown GPU → mark calibration pending.
- `LsxFastMathScope` extended to take the tri-state mode: `kAuto` leaves the
  global TC flag untouched (policy already applied at init); `kOn`/`kOff` force
  it for the Infer/SelfCheck scope and restore afterwards.
- Tri-state: `FastMathMode { kAuto, kOn, kOff }` in `ModelInitConfig` /
  `GenerationCliArgs`. The existing `bool fast_math` stays for compatibility
  (`true` → `kOn`, `false` → `kAuto`); CLI `--no-fast-math` → `kOff`.
- Expert-cache: `lifecycle::ExpertCacheSlots(..., int slots_pct = 100)`;
  models pass the resolved percentage at their existing sizing call sites
  (GLM-Flash `ResolveExpertSlots`, Qwen MoE `Load()`, gemma, kimi).
- Dual-stream consumers (GLM-Flash staging, Qwen ring) consult
  `LsxDualStreamStaging()` when picking the staging scheme.
- Env `LSX_GEMV_MAX_MIB` handling moves from `lsx_inference_nvfp4.cu` into
  `gpu_perf.cpp` (`ResolvePerfKnobs`); the dispatcher reads the knob value.

## 5. Testing (TDD)

Red → green per step.

1. `tests/test_gpu_perf.cpp` (host-only, no CUDA device needed):
   - profile → knobs: base entry (RTX 5090), heuristic on synthetic profiles
     (sm=128/bw=1000, sm=64/bw=300), precedence env > CLI > calibrated >
     heuristic > base;
   - `PeakBandwidthGbs`, `GpuProfileKey` normalization;
   - calibration-cache JSON round-trip + format-version invalidation;
   - tri-state parsing (`bool fast_math` → mode, CLI flags).
2. `tests/test_perf_knobs_apply.cu` (light, needs GPU): `SetLsxGemmMaxMib`
   changes GEMV eligibility across the threshold boundary; split-K thresholds
   reflected by `LsxTcBatchSplitK`.
3. `tests/test_model_independence_guard.cpp` extended: model-neutral layer has
   no family stem; consumer list for the tri-state scope.
4. Integration: real-model selfchecks on RTX 5090 run with `kAuto` → TC path;
   run-to-run Paris determinism preserved (tolerance-class vs bit-exact is
   unchanged semantics).

## 6. Developer calibration run (base-authoring tool)

A developer-facing mode that produces the values to fix in the committed base:
- New flag in lsxfabric: `--calibrate <model_path>` (and/or a standalone
  Calibrate entry in `model_runtime`): loads the model with all knob paths
  force-enabled one at a time, runs the micro-benchmarks from section 3 over
  the model's real weight shapes, plus a short generation throughput pass per
  promising knob combination.
- Output: the resolved `LsxPerfKnobs` (winning combination) + GPU passport,
  printed and written to `perf-cache/<gpu-key>.json` for review; the developer
  copies the winning knobs into the committed base table in `gpu_perf.cpp`.
- This is the "calibration run of the model to obtain the data fixed in code"
  requirement: every model optimization is expected to go through this flow so
  the committed base stays the single source of truth for known GPUs.

## 7. Files

- New: `include/lsxcommon/gpu_perf.h`, `src/gpu_perf.cpp`,
  `tests/test_gpu_perf.cpp`, `tests/test_perf_knobs_apply.cu`
- Modified: `lsx_inference_nvfp4.{h,cu}` (knob setters + dispatcher reads),
  `lsx_tc_gemm.cu` (split-K thresholds), `model.h` + `model_runtime.h`
  (tri-state), `resource_governor.h` (`ExpertCacheSlots` pct param),
  dual-stream consumers, `apps/lsxfabric/src/main.cpp` (`--calibrate`,
  `--no-fast-math`), `tests/test_model_independence_guard.cpp`, lsxcommon
  `CMakeLists.txt`.

## 8. Non-goals (YAGNI)

- No per-kernel grid/block autotuning.
- No per-model calibration runs at user Load time (calibration is shape-class
  micro-benchmarks + the developer base-authoring flow).
- No GPU-local cache outside the vault.

## 9. Exception rule

Tests that test specific functionality keep full override capability: env
variables win over everything; CLI `kOn/kOff` forces the numeric class; CI
bit-exactness tests pin `kOff` explicitly. Auto-tuning must never make a
pinning test non-deterministic.
