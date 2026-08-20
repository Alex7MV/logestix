# MoE Expert Store Unification — Design Spec (§3.2 of kernel unification)

**Date:** 2026-08-20
**Status:** Approved (pending implementation plan)
**Scope:** `libs/lsxcommon` — MoE mmap-expert LRU loaders only (host-side storage/lazy-parse/eviction). NOT the dequant kernels.
**Prerequisite:** NVFP4 codec unification (`nvfp4_codec.h`, `glm_expert_kernels.cuh`) already merged to `main`.

## 1. Problem Statement

Two independent, near-identical mmap-expert LRU loaders exist:

- DeepSeek `ExpertDb` / `load_expert_index` / `ensure_expert_resident` / `free_expert_index`
  (`deepseek_inference_load.cuh:40,80,92`, `deepseek_inference_load.cu:267-459`)
- GLM `GlmExpertDb` / `load_glm_expert_index` / `ensure_glm_expert_resident` / `free_glm_expert_index`
  (`glm_inference_load.h:234`, `glm_inference_load.cpp:568-663`)

Both:
- share the **identical 9-column `ExpertSchema`** (`schema.h:84-105`) with the same 6 binary column indices (`kIdxW1Weight`…`kIdxW3Scale`) + 3 float global-scale columns (`kIdxW1GlobalScale`…`kIdxW3GlobalScale`).
- run structurally identical Arrow extraction lambdas (`get_bin`, `get_scale_bin`, `get_gs`) in anonymous namespace.
- implement byte-identical LRU touch/evict, warm-up to `max_open`, the pre-eviction `cudaStreamSynchronize` barrier, and `free_*_index`.

The only real differences:
- record type (`ExpertEntry` vs `GlmExpertEntry`) — the host pointers are layout-identical.
- namespace (`gpu` vs `glm`).
- GLM additionally accepts a fallback `experts/` directory (in addition to `logestix/experts/`).

If left untouched, the upcoming MoE routing/expert-loading optimization lands the same
algorithm twice in separate files (the exact problem the kernel-unification spec §3.2 exists to
prevent). The spec deferred §3.2 with an explicit note that it still gates optimization.

## 2. Goals (non-goals)

**Goals**

- Collapse the two mmap-expert LRU loaders into ONE header-only `ExpertStore<TEntry>` template
  keyed on the expert-layout descriptor, preserving existing call-site names via aliases.
- Preserve exact runtime behavior — erasure-level parity with today's loaders, including the
  eviction stream barrier and the warm-up order.
- Keep extensibility: a future MoE model instantiates `ExpertStore<NewExpertEntry>` + a
  `Parse()` and reuses the store wholesale.

**Non-goals**

- NO change to dequant kernels. `GlmExpertDeviceBlock` (GLM) and DeepSeek's expert-dequant path
  stay model-specific and are optimized separately (kernel boundary is deliberately NOT unified).
- No behavior change in inference; bit-exact GLM semantics preserved exactly.
- No change to the pinned-weight loaders (`load_pinned_weights` / `LoadGlmPinnedWeights`).

## 3. Approach: header-only `ExpertStore<TEntry>` template

New header `libs/lsxcommon/src/expert_store.h`, namespace `lsxcommon::gpu::expertstore`.

### 3.1 Template shape

```cpp
// TEntry: the model record type (e.g. ExpertEntry / GlmExpertEntry). Must expose the
// 6 host_* pointer fields + 3 global-scale floats + 6 byte-count fields, layout-identical
// to today's ExpertEntry/GlmExpertEntry.
template <typename TEntry>
struct ExpertStore {
    std::vector<TEntry> entries;
    std::vector<std::shared_ptr<arrow::io::MemoryMappedFile>> mmaps;
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
    std::vector<std::string> paths;
    std::vector<size_t> lru;
    size_t max_open = 7000;               // Windows CRT fd cap (8192)
    int num_layers = 0;
    int experts_per_layer = 0;
    std::string error_msg;

    // @p parse: static model function `void Parse(TEntry &out,
    //   const std::shared_ptr<arrow::RecordBatch> &batch)` — the shared 6-bin+3-f32
    //   extraction, identical for both models, differing only in TEntry.
    // @p fallback_experts: when true, fall back to <model>/experts when
    //   <model>/logestix/experts is absent (GLM behavior). DeepSeek passes false.
    bool Load(const std::string &model_path, bool fallback_experts,
              void (*parse)(TEntry &out,
                            const std::shared_ptr<arrow::RecordBatch> &));

    // Touch resident entry (move to LRU front) or mmap+parse it, evicting the
    // least-recently-used entries under max_open. Stream is synchronized ONCE,
    // only when an eviction is about to happen (pre-munmap barrier — the
    // cudaMemcpyAsync H2D read must complete before the source host mmap is
    // unmapped). Mirrors today's byte-identical contract in both loaders.
    void EnsureResident(int layer, int expert, cudaStream_t stream);

    void Free();
};

// Parse helper used by both models (TEntry-agnostic: it hands the model the raw
// pointer/byte/scale records; the record type's field names already match).
```

### 3.2 Aliases preserving call sites

Keep existing names compile-compatible so `glm_inference.cu`, `deepseek_inference_pipeline.cu`,
and the tests keep their calls unchanged:

```cpp
// DeepSeek
namespace lsxcommon { namespace gpu {
using ExpertDb = expertstore::ExpertStore<lsxcommon::ExpertEntry>;
inline ExpertDb load_expert_index(const std::string &mp, int nl, int epl) {
    ExpertDb db;
    db.Load(mp, /*fallback=*/false, ParseDeepSeekEntry);
    db.num_layers = nl; db.experts_per_layer = epl;
    return db;
}
}}
```

(`ParseDeepSeekEntry` is the extracted DeepSeek `parse_expert_file` lambda body.)

```cpp
// GLM — same shape, fallback=true, ParseGlmEntry.
```

Rule: aliases live in the model's own header (`.cuh` for DeepSeek, `glm_inference_load.h` for
GLM), so `expert_store.h` stays free of model namespaces — it is the shared, architecture-agnostic
core requested by §5 of the parent spec.

### 3.3 Naming / ownership

- `ExpertStore<TEntry>` owns exactly the vectors today's `ExpertDb`/`GlmExpertDb` own.
- `TEntry` stays a model-owned struct (default-construction in `entries`); `ExpertStore` never
  defines the per-record layout — the template parameter supplies it.

## 4. Extensibility contract (model-roadmap guarantee)

1. New MoE model: define `TEntry` + `Parse`, alias `ExpertStore<TEntry>`, call `Load`/`EnsureResident`/`Free` — zero deep-divide.
2. Dequant kernels remain per-model kernels. Future models never read a merged store that binds their dequant.
3. Weight upload (`StageTensors`-style H2D) stays a model concern in its own `.cu`, reading the shared store only for source pointers.

## 5. Error handling

- `Load` reproduces the exact "mmap fail <path>: <status>" debug lines and the `error_msg` wiring
  of today's loaders (existing tests lock them).
- No new error paths. Parse failures degrade to a skipped/inactive slot exactly as today (null
  `host_*` pointers mark the entry inactive).

## 6. Verification

- Preserve bit-exact GLM behavior: `test_glm_experts`, `test_glm_moe_compute`, `test_glm_model`,
  `test_glm_selfcheck` (SKIP without checkpoint) — unchanged.
- DeepSeek unchanged: `test_deepseek_inference_kernels.cu` (4470 assertions), `test_deepseek_loader`.
- No duplicate loader bodies remain (grep `load_expert_index|load_glm_expert_index` → alias-only).
- Deterministic Qwen/DeepSeek SelfChecks unchanged.
- Full model subset `ctest -R glm -R qwen -R deepseek -R model`.

## 7. Out of scope / deferred

- Dequant-kernel unification (per-model boundary, per parent spec §4).
- Optimization of either MoE routing/of-LRU — post-unification, separate plans.
- No CUDA-architecture tuning (Blackwell CC 120/121 preserved).