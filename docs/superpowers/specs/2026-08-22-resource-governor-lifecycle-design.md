# Resource Governor — Model Lifecycle Strategy Design

Date: 2026-08-22
Scope: `libs/lsxcommon` (+ integration into `GlmV5Model`, `ModelBase`,
`InferenceEngine`, `LsxOrchestrator`). Reference implementation: GLM-5.2.
Approved approach: B — central ResourceGovernor owning all VRAM/RAM/CPU budgets.

## Problem

Today there is no unified owner of hardware resources:

1. `GlmV5Model::Infer()` and `Benchmark()` reload all pinned weights on every
   call and free them afterwards (`glm_v5_model.cpp:88`, `cleanup()` at
   `:70-81`) — the most expensive resource, VRAM residency, is discarded
   between requests.
2. Hardware capability probing is ad-hoc: static GPU props are logged
   (`common.cpp`), free VRAM is queried once for expert-cache sizing
   (`glm_inference.cu:1224-1273`). CPU cores and system RAM are never queried.
3. Residency decisions (what lives in VRAM vs host-pinned vs mmap+staged,
   expert LRU capacity, staging/stream counts) are scattered per-model and
   cannot adapt to the actual machine.
4. Concurrency groundwork does not exist: models are single-consumer by
   convention only, with no budget accounting to arbitrate future parallel
   inference.

The strategy must exploit the specific features of each model (dense vs MoE
layer grid, NVFP4 expert stacks, heterogeneous indexer/MTP layers recorded in
the conversion manifest) to reach the maximum achievable inference speed on
the current hardware.

## Requirements

- **Hybrid decisioning**: a deterministic static plan built at `Load()` from
  hardware probing + manifest data, plus narrow runtime adaptation of a few
  parameters inside plan bounds. No persistence of learned parameters across
  process runs.
- **Full control scope**: memory/residency plus compute pipeline levers
  (copy streams, staging buffers, expert prefetch batching, host upload
  threads). Real multi-threaded/concurrent inference is out of scope for this
  iteration, but its groundwork (budgets, residency locks) is part of the
  contract.
- **Layered strategies**: general base → per-archetype (dense/MoE, derived
  from manifest topology) → per-model overrides; reference implementation
  `Glm52Strategy`.
- **GLM-5.2 first**: weights load once in `Load()` and are reused by every
  `Infer()`/`Benchmark()`/`SelfCheck()`. DeepSeek/Qwen adaptation follows in
  later iterations but inherits the contracts now.
- Tests run on machines without the real GLM weights: everything except
  real-model selfcheck is verified against synthetic fixtures.
- Memory-shortage refusal (`Required` lease failure aborting `Load()`) applies
  in production unconditionally; in tests it is enabled only in tests that
  explicitly verify it (see Testing).

## Architecture

New host-only C++ unit in `libs/lsxcommon`:
`include/lsxcommon/resource_governor.h`, `src/resource_governor.cpp`,
`include/lsxcommon/lifecycle_strategy.h`, `src/lifecycle_strategy.cpp`,
`include/lsxcommon/runtime_tuner.h`, `src/runtime_tuner.cpp`.
No abseil containers in headers reachable from `.cu` TUs (NVCC policy);
host-side maps use `absl::flat_hash_map`.

`ResourceGovernor` is an engine-level singleton following the existing
`InferenceEngine` pattern (`model_engine.h`): initialized inside
`InitializeEngine()`, shut down symmetrically in `ShutdownEngine()`.

### Components

| Component | Role |
|---|---|
| `HardwareProfile` | One-time probe at governor init: CPU core count + affinity (`std::thread::hardware_concurrency`, Win32 `GetLogicalProcessorInformationEx` / POSIX `sched_getaffinity`), RAM total/available (`GlobalMemoryStatusEx` / `sysinfo`), GPU SM count, clocks, total/free VRAM (`cudaDeviceGetAttributes`, `cudaMemGetInfo`) |
| `ILifecycleStrategy` + registry | Builds a `LifecyclePlan` from `(Topology, VramProfile, HardwareProfile)`. Resolution: per-model override → archetype (manifest has non-empty `moe_layers[]` → MoE, else dense) → `GeneralStrategy`. Static compile-time registry next to `model_registry.cpp` |
| `LifecyclePlan` | Deterministic residency plan (see below); validated before any allocation |
| `LeaseRegistry` | Accounted leases over VRAM/RAM budgets; refcounted RAII handles; residency locks for future concurrency |
| `RuntimeTuner` | Adaptive controller moving parameters only within `TuningBounds` of the plan; telemetry-driven |

Models never query hardware directly: `GlmV5Model::Load()` asks the governor
for the plan of its type and executes it. The existing direct
`cudaMemGetInfo` heuristic in `glm_inference.cu` migrates into the
governor/tuner path.

## Leases and Budgeting

Lease kinds (`enum class LeaseKind`):

| Kind | Owner | Lifetime |
|---|---|---|
| `kPermanentWeights` | Model | `Load()` → `Unload()` |
| `kExpertCache` | Expert LRU | `Load()` → `Unload()`; size tuned at runtime |
| `kInferArena` | One `Infer()` call | acquire → release on exit |
| `kHostPinned` / `kHostRam` | Any | as above, host memory |

Governor API (shape):

```cpp
class ResourceGovernor {
  const HardwareProfile &hardware() const;
  const LifecyclePlan &PlanFor(std::string_view model_type) const;
  Lease Acquire(LeaseKind kind, std::string_view model_type,
                uint64_t bytes, LeasePolicy policy);
};
```

`LeasePolicy`:

- **`Required`** — shortage is a deterministic `Load()` error raised *before*
  any allocation ("model X needs Y bytes, Z available"). Sum of all required
  leases of registered models is checked against free VRAM up front.
- **`Adaptive(min, max)`** — expert cache and staging: governor grants the
  plan's start value; the tuner moves within bounds by acquiring/returning
  bytes through the governor. A refused adaptive growth is soft: current size
  is kept, running inference is never torn down.

Bookkeeping: governor tracks bytes per category and periodically reconciles
against `cudaMemGetInfo` at `Infer()` boundaries (never inside the hot loop);
drift beyond a threshold logs a spdlog warning (catches external allocators
like cuBLAS workspace). Leases are RAII, so leaks structurally cannot persist
past model destruction.

Concurrency groundwork: each model owns a residency `std::shared_mutex`;
`Infer()` takes a shared lock today (single consumer, no measurable cost),
unique lock is reserved for future unload-during-inference and parallel
dispatch. The orchestrator's dispatch logic itself is unchanged this
iteration.

## LifecyclePlan

```cpp
struct LifecyclePlan {
  // Permanent VRAM residency
  bool pin_globals_in_vram;        // embed / lm_head / final_norm
  bool pin_dense_layers;           // dense layers fully resident
  uint64_t permanent_weight_bytes; // from VramProfile.min_bytes

  // Expert subsystem (MoE)
  size_t expert_cache_slots;       // start value, bounds [min, max]
  uint64_t expert_cache_min_bytes, expert_cache_max_bytes;
  size_t expert_prefetch_batch;    // experts per H2D batch

  // Compute pipeline
  int copy_streams;                // dual-stream ping-pong already exists
  size_t staging_ring_bytes;       // default 8 MiB, tunable
  size_t host_upload_threads;

  TuningBounds tuner_bounds;
};
```

Strategy logic:

- **GeneralStrategy**: `permanent = min(VramProfile.optimal_bytes,
  free_vram - keep_out)`; whatever does not fit stays host-pinned /
  mmap+staged. Expert cache = remaining VRAM minus reserve for infer arenas
  (using `VramProfile.max_bytes - optimal` as margin guidance). Conservative
  defaults for streams/staging.
- **MoEStrategy** adds: expert cache slot count from mean expert bytes (read
  from `model.index.json` index data already parsed by loaders — no extra
  disk IO), prefetch batch sized from bus/hardware profile, priority rule:
  dense/global weights are always resident before any expert-cache capacity.
- **Glm52Strategy** uses the heterogeneous layer grid from `Topology`
  (dense 0–2, MoE 3–77, indexer/MTP layer 78): dense + globals are
  required-resident. The MTP epilogue (layer 78) also stays permanently
  resident: the frozen single-pass generation loop reads its router/norms on
  every forward, so deferring it would create a mid-loop H2D stall
  (amendment 2026-08-22 during planning review). The current adaptive
  expert-cache algorithm becomes the lower bound of the tuner range.

Plan validation happens in the strategy before any allocation; an infeasible
plan fails `Load()` deterministically with the need/have diagnostic.

## RuntimeTuner and Telemetry

First iteration tunes exactly one parameter: expert LRU cache slots.

Telemetry sources (all already exist or are trivial additions), delivered to
the governor via a narrow `TelemetrySink` interface (fixed POD slots,
atomics on write from the generation loop, zero-alloc, read between requests):

| Metric | Source |
|---|---|
| Expert cache hit/miss | counters in `GlmExpertDeviceCache` (`glm_expert_cache.h`) |
| H2D / GEMM times | existing event pairs `ev_h2d[2]` / `ev_gemm[2]` (`glm_inference.cu`) |
| Free VRAM | `cudaMemGetInfo` at `Infer()` boundaries |
| Load wall-clock | timing log from `LoadGlmPinnedWeights` |

Controller: sliding-window miss-rate; sustained misses with free adaptive
budget → grow request through governor; sustained hits with surplus capacity
→ shrink and return the lease. Hysteresis + minimum interval between
reconfigurations prevents oscillation. Growth failures are soft.

Extension path: new tunables (staging ring size, prefetch batch) are new
fields in `TuningBounds` + controller branches; contracts unchanged.
No cross-run persistence.

## Integration

- **`GlmV5Model`**: `Load()` acquires plan + leases, calls
  `LoadGlmPinnedWeights` once, keeps `GlmPinnedWeights pw_` as a member,
  opens the expert index (`ExpertStore`). `Infer()/Benchmark()/SelfCheck()`
  reuse `pw_`; acquire/release `kInferArena`; per-call weight teardown
  (`cleanup()`) disappears — only transient state is freed. `Unload()`
  releases leases, frees weights, closes the store; idempotent after failed
  Load.
- **`ModelBase`**: optional `AttachGovernorLeases(const LifecyclePlan&)` hook
  and lease-derived `resident_bytes_` accounting; DeepSeek/Qwen adopt later
  without loader changes.
- **`InferenceEngine`**: init/shutdown wiring for the governor; `Session`
  verifies governor registration.
- **`LsxOrchestrator`**: dispatch logic unchanged; `DispatchOne` takes the
  model's shared residency lock around `Infer()`.
- Untouched: Arrow formats and manifests, CUDA inference kernels, the
  generation pipeline internals, DeepSeek/Qwen loaders, lsxnetwork, lsxforge.

Known risk: real-model verification is impossible on the development machine
(GLM weights absent; GLM tests SKIP). Bit-exactness and weight-reuse
correctness are covered by the synthetic fixture suite; final arbitration is
a real-model run of `test_glm_selfcheck` (bit-identical `final_logits`)
performed by the operator on the GLM-capable machine.

## Error Handling

- Plan infeasible (`required > free`): deterministic `Load()` error before
  any allocation; nothing leaked.
- `Acquire(Adaptive)` failure: soft; current sizing kept; inference unaffected.
- CUDA failure during planned load: release that model's leases, `Load()`
  returns the error (inherits strict glm-load-acceleration semantics).
- Bookkeeping/reality drift: spdlog warning, non-fatal.

## Testing (TDD)

Catch2 v3, existing style. New files under `tests/`. Test-only relaxation of
memory refusal: the governor accepts a test override disabling `Required`
shortage aborts (injectable constructor flag / `LSX_TEST_RELAX_VRAM_REFUSAL`);
only the dedicated refusal tests enable strict mode, so the suite passes on
small-VRAM machines.

1. **`test_hardware_profile.cpp`** — parsing/aggregation logic with injected
   raw query functions; edge cases; real-machine smoke: profile valid,
   VRAM total > 0.
2. **`test_lifecycle_plan.cpp`** — General/MoE/Glm52 strategies on injected
   `(Topology, VramProfile, HardwareProfile)`: plan fits free VRAM; strict-mode
   refusal on deliberate shortage; GLM grid yields dense-resident + fully
   resident MTP (amendment above);
   determinism (identical inputs → identical plan).
3. **`test_resource_governor.cpp`** — RAII lease release returns budget;
   Required refusal (strict mode) fires before allocations; Adaptive bounds
   respected; residency shared/unique locking; reconciliation detects drift.
4. **`test_runtime_tuner.cpp`** — synthetic hit/miss series: grows on misses
   with free budget, shrinks otherwise, hysteresis blocks oscillation, soft
   failure when budget is occupied.
5. **`test_glm_model_lifecycle.cu`** — synthetic GLM model via `glmtest`
   fixtures: single `Load()` allocates once; two consecutive `Infer()` calls
   reuse weights (device pointers unchanged, results bit-identical);
   `Unload()` cleans everything; Load-after-Unload works.
6. **Real model**: `test_glm_selfcheck` (SKIP without weights) remains the
   arbiter of bit-exactness and `final_logits` determinism on the operator's
   machine.

TDD order: failing test first per component; the governor core is pure
arithmetic over injected profiles (no GPU required); only the lifecycle `.cu`
test touches CUDA.

## Out of Scope

- Concurrent inference execution (batching or multi-threaded dispatch) —
  groundwork only.
- Cross-run persistence of tuned parameters.
- DeepSeek/Qwen strategy implementations (contracts ready).
- On-disk format changes; changes to CUDA kernels or the generation pipeline.
