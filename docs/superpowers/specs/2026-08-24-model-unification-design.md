# Model Implementation Unification — Design

Date: 2026-08-24
Scope: host model layer + weight loaders. CUDA kernels and converters are explicitly out of scope (next cycle).

## Goal

Maximize unification of shared algorithms across the 7 registered models
(GLM-5.2, DeepSeek-V4-Flash, Qwen3.5-35B-A3B MoE, Qwen3.5-9B, Qwen3.8-27B,
Gemma4-26B-A4B, Kimi-K27-Code) while keeping each model's specifics in the
model itself. Eliminate duplication found by the audit; close MODEL INGEST
INVARIANT protocol gaps; fix behavioral anti-patterns by migrating to the
reference patterns.

## Audit Findings Driving the Design

### Host model layer (~420 consolidatable LOC)

| Duplicate cluster | Sites | Reference locations |
|---|---|---|
| `GenerationResult`→`InferenceResult` mapping + decode | 5 | `glm_v5_model.cpp:216`, `kimi_v5_model.cpp:134`, `qwen_3_5_moe_model.cpp:173`, `qwen_base_model.cpp:115`, `gemma_base_model.cpp:157` |
| `EnsureExecutionContext` / `TeardownWeights` | 3 idioms | GLM≈Kimi byte-identical (`glm_v5_model.cpp:48`, `kimi_v5_model.cpp:25`) |
| Load skeleton + error ladder | 6 models | repeated `TeardownWeights(); tok_db_.reset(); return false;` |
| Incremental session policy | 2 verbatim + 1 vestigial | `qwen_base_model.cpp:82` ↔ `qwen_3_5_moe_model.cpp:149` |
| `SyntheticGeometryAllowed` | 3 copies | `qwen_base_model.cpp:19`, `gemma_base_model.cpp:42`, `qwen_3_5_moe_model.cpp:18` |
| Benchmark guard+run+map | 4 variants | per-model files |
| Registry blocks | 7 identical | `model_registry.cpp:16-43` |

Protocol gaps:
- DeepSeek has no frozen-geometry gate at all (runtime config read,
  `deepseek_v4_flash_model.cpp:36`) — violates protocol step 3.
- GLM/Kimi have constant tables but no `static_assert`
  (`glm_inference_load.h`, `kimi_inference_load.h:22-48`).
- Qwen35Moe duplicates frozen constants twice: class scope
  (`qwen_3_5_moe_model.h:26-36`) mirrors `src/qwen_moe_constants.h`.
- Only GLM/Kimi hold `shared_lock(residency_mutex_)` in Infer although the
  mutex exists in `ModelBase`.

Anti-patterns (behavior):
- GemmaBaseModel reloads the entire expert index from disk on every Infer
  (`gemma_base_model.cpp:119-144`).
- QwenBaseModel creates/destroys stream+cuBLAS on every Infer call
  (`qwen_base_model.cpp:73-77,140-141`) and again in Benchmark.
- Gemma custom SelfCheck reimplements `runtime::RunParisSelfCheck` structure
  with a window predicate (`gemma_base_model.cpp:191-234`).

### Loaders (~1.6–2.1 kLOC consolidatable)

| Duplicate cluster | Copies | Notes |
|---|---|---|
| Arena upload-job engine (validate→arena→ring upload) | 6 | DeepSeek/GLM/Qwen/QwenMoe/Gemma/Kimi; ≥60% identical |
| `kArenaAlign`/`AlignUp`/`StreamDrain` | 6–7 | trivial helpers |
| Suffix tensor lookup | 6 | identical |
| `UploadNvfp4Weight` | 2 verbatim | `qwen_inference_load.cpp:225`, `qwen_moe_inference_load.cpp:166` |
| Device LRU expert cache | 3 | GLM↔Qwen token-identical (~95 LOC), "port of port of DeepSeek" |
| Legacy `Loader` | 1 prod-dead | `loader.cpp` ~224 LOC, only `test_deepseek_loader.cpp` uses it |

Proven consolidation precedents in this repo: `PinnedStagingRing`,
`ExpertStore<T>`, `nvfp_codec.h` header-inline ODR-safe pattern.

## Design

### 1. Loaders — shared fastload infrastructure

**1.1 Arena job engine → `src/fast_load_internal.h`** (beside
`PinnedStagingRing`):

```
struct UploadJob { const TensorRecord* rec; void* dst; CvKind kind; };
cudaError_t ValidateJobs(span<const UploadJob>, std::string* error);
cudaError_t RunUploadJobs(          // validate → single arena cudaMalloc →
    span<UploadJob>,                // 256B-aligned views → ring upload
    fastload::PinnedStagingRing&,
    cudaStream_t, std::string* error);
```

Error channel convention (operator decision): error code return +
optional `std::string* error` out-param for details; caller decides to log or
propagate. Strictness semantics are preserved per model: GLM tolerant of
partial expert grids stays tolerant; Qwen strict stays strict.

Also hoisted here: `kArenaAlign`, `AlignUp`, `StreamDrain`,
suffix-based `FindTensor`, single `UploadNvfp4Weight`.

Each model keeps only its job list construction — that is the model-specific
part. Six private copies (~700–800 LOC) collapse onto one engine.

**1.2 Unified device LRU cache → `include/lsxcommon/expert_device_cache.h`:**

One template `ExpertDeviceCacheT` replacing
`ExpertDeviceCache` (DeepSeek) / `GlmExpertDeviceCache` /
`QwenMoeExpertDeviceCache`. Touch/Lookup/Insert become methods. Header-only
inline pattern (like `nvfp4_codec.h`) — safe against NVCC transitive include.
Host-side alloc/free/residency helpers become shared functions over the
template; GLM's inline cache management inside `glm_inference_load.cu` moves
to them.

**1.3 Dead code removal:**

- Port the 2 tests in `test_deepseek_loader.cpp` from legacy `Loader` to
  `ExpertSchema::UnpackBatch`, then delete `loader.cpp`/`loader.h` (~224 LOC).
- Delete empty stub `deepseek_inference_norm.cu` (empty kernel body, no
  launch site).

### 2. ModelBase — unified lifecycle and generation

**2.1 Persistent CUDA context:** `ModelBase` gains
`EnsureContext()` / `TeardownContext()` owning `cudaStream_t` +
optional `cublasHandle_t` (with `cublasSetStream`). Replaces all three
existing idioms:

- GLM/Kimi persistent contexts migrate onto the base.
- QwenBase temp-per-call context (anti-pattern) is fixed by migration.
- Gemma temp-per-call fixed likewise; DeepSeek temp-per-load likewise.

**2.2 Generation preamble:** `ModelBase::RunGeneration(...)` = guard +
`shared_lock(residency_mutex_)` + family generation call + post-processing.
The residency lock becomes a base-class invariant for all models instead of a
GLM/Kimi-only choice. Model specifics reduce to their `RunXxxGeneration` and
token bridge.

**2.3 Result mapping:** shared
`runtime::FillInferenceResult(InferenceResult&, const GenResultView&)` in
`model_runtime.cpp`. `GenResultView` is a lightweight view over the 5
same-shaped per-family result structs (tokens, final_logits, ttft_ms,
total_ms, error_msg, eos_terminated). Erases ~110 LOC across 5 sites;
Benchmark guard+run+map uses the same helper.

**2.4 Behavioral fixes delivered by migration:**

- Gemma: expert index loaded once in `Load`, freed in `Unload` (load-once
  lifecycle like all other models); session continuation policy follows
  QwenBase semantics.
- QwenBase: no more per-call stream/cuBLAS churn in Infer/Benchmark.
- All models hold the residency lock during generation.

### 3. GeometryGuard + protocol gap closure

**3.1 `GeometryGuard`:** one validator driven by a frozen name→value table
(`std::initializer_list<std::pair<std::string_view, size_t>>`), uniform error
text. Replaces 4 hand-written `ValidateGeometry` bodies and collapses 3
copies of `SyntheticGeometryAllowed` into one base implementation.

**3.2 Compile-time locking (protocol step 3):**

- **DeepSeek**: introduce a frozen-geometry passport — current topology
  becomes a `constexpr` table + `static_assert`; runtime config reading is
  replaced by verification against the passport (hard load failure on
  mismatch).
- **GLM/Kimi**: add the missing `static_assert`s over existing constant
  tables.
- **Qwen35Moe**: delete duplicated class-level constants; single source of
  truth remains `qwen_moe_constants.h`.

**3.3 Registry:** template `LsxModelFactory::Register<T>(kType)`;
7 boilerplate blocks collapse to one-liners.

### 4. SelfCheck — single harness

`runtime::RunParisSelfCheck` gains an options struct: chat-template framing
on/off, token search window size, pass predicate. `GemmaBaseModel::SelfCheck`
(~40 LOC custom reimplementation with a 12-token window) is deleted; its
semantics expressible via options. The "The capital of France is" → "Paris",
`r.selfcheck_passed == true` contract holds for all 7 models. Bit-exact
logits determinism stress-tests are untouched.

## TDD Strategy

Red→green on every step:

1. Characterization first: before each move, pin current behavior with a test
   where coverage is missing — arena-engine job validation, LRU cache
   semantics, `FillInferenceResult` mapping, `GeometryGuard` failures,
   `Register<T>`. Host-only code tested without CUDA where possible
   (precedent: unit-testable LRU bookkeeping in `test_qwen_moe_loader`).
2. Refactors are mechanical moves under protection of existing suites
   (`ctest -R glm`, `-R deepseek`, `-R qwen_moe`, then full `ctest`).
3. Anti-pattern fixes: write the target-behavior test first (Gemma load-once
   index; QwenBase persistent context — observable handle/no-churn), then
   migrate.
4. DeepSeek geometry gate: test that loading fails hard when topology
   mismatches the passport.
5. Cycle verification: `cmake --preset linux-debug && cmake --build --preset
   linux-debug && ctest --preset linux-debug`; real-weight selfchecks run
   separately where models are available.

## Out of Scope (next cycles)

- CUDA kernel dedup (~850 recoverable LOC): RMSNorm ×8, SwiGLU ×8, router
  top-k ×5, NVFP4 dequant ×9 + stray FP8 decoders, act_quant ×2, argmax ×4.
  GLM glibc-parity kernels stay frozen regardless.
- Converter pipeline unification (byte-deterministic `model.index.json`
  regression surface).
- Open question for operator: `lsxcommon_qwen_moe_cu` compiles with
  `-fmad=false` but its expert GEMMs live in `lsxcommon_qwen_cu` compiled
  with default FMA contraction — intent needs confirmation before any
  Qwen/Qwen-MoE consolidation.

## Success Criteria

- No duplicate arena engines, LRU caches, result-mapping blocks, session
  policy blocks, geometry validators, or registry boilerplate remain.
- All models share one lifecycle path (persistent context, residency lock)
  and one SelfCheck harness.
- DeepSeek/GLM/Kimi satisfy protocol step 3 compile-time locking.
- Full ctest suite green; new characterization tests pin consolidated units.
- Net reduction ≈ 2–2.5 kLOC of duplication without behavior change except
  the four listed intentional fixes.
