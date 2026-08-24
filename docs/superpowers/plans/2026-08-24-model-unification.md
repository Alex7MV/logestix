# Model Implementation Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate duplication across the host model layer and weight loaders (spec: `docs/superpowers/specs/2026-08-24-model-unification-design.md`) while keeping model specifics in the models.

**Architecture:** Extend the proven `fastload` infrastructure (`PinnedStagingRing`) with a shared arena upload-job engine and one device LRU expert cache; enrich `ModelBase` with persistent CUDA context, residency-lock preamble, result mapping and geometry validation; close MODEL INGEST INVARIANT protocol gaps; unify SelfCheck via options.

**Tech Stack:** C++20, CUDA 13.x runtime/cuBLAS (host-side API only), Arrow (schema views), Catch2 v3, CMake presets, spdlog, absl.

## Global Constraints

- Work on branch `refactor/model-unification`. English commit messages with prefixes (`feat:`, `refactor:`, `test:`, `fix:`, `docs:`).
- All logging via spdlog (`lsxcommon::log` wrappers); never `std::cout`/`printf`.
- absl priority in host-only code: `absl::flat_hash_map`, `absl::StrCat` etc.; `std::string_view`/`std::span` stay `std::`.
- New headers reachable from `.cu` TUs must be NVCC-safe: no `absl/container/*`, no exceptions. Follow the header-inline pattern of `nvfp4_codec.h` / `glm_expert_cache.h`.
- Do NOT touch anything under CUDA kernel numeric contracts: `-fmad=false` object libs (`lsxcommon_glm_cu`, `lsxcommon_qwen_moe_cu`, `lsxcommon_kimi_cu`), `glm_device_math.h`, `sampling.cu`, `qwen_nvfp4_gemm_tile_kernel`. This plan is host-side only.
- Single-stream generation invariant: generation loops (`RunXxxGeneration*`) are not modified except their stream argument source.
- Strictness semantics preserved: GLM tolerant of partial expert grids; Qwen MoE expert store strict.
- Build/test commands (Linux): configure once with `cmake --preset linux-debug`; build target `cmake --build --preset linux-debug --target <target>`; run tests `ctest --test-dir build/linux-debug -R <regex>` — **sequential only for GPU/inference suites** (parallel runs fail spuriously on one RTX 5090). Host-only unit tests may run with `-j`.
- Baseline verified 2026-08-24: full suite green sequentially (100/100; `test_qwen_kv_cache` intentionally Skipped).

---

### Task 1: Shared fastload helpers (AlignUp, StreamDrain, suffix find)

**Files:**
- Modify: `libs/lsxcommon/src/fast_load_internal.h`
- Create: `tests/test_fast_load.cpp`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Produces (in `namespace lsxcommon::fastload`, header `src/fast_load_internal.h`):
  - `inline constexpr size_t kArenaAlign = 256;`
  - `inline size_t AlignUp(size_t v, size_t a);`
  - `struct StreamDrain { cudaStream_t s; ~StreamDrain(); };`
  - `const TensorView *FindTensorSuffix(const TensorBatchViews &, std::string_view);`
  - `const DenseNvfp4View *FindNvfp4Suffix(const DenseNvfp4BatchViews &, std::string_view);`

- [x] **Step 1: Write the failing test**

Create `tests/test_fast_load.cpp`:

```cpp
// TDD — fastload shared helpers (host-only: AlignUp + suffix lookup; no CUDA
// allocations, no model weights).
#include <catch2/catch_test_macros.hpp>

#include "fast_load_internal.h"
#include "lsxcommon/schema.h"

TEST_CASE("AlignUp rounds up to alignment", "[fastload]")
{
    using lsxcommon::fastload::AlignUp;
    STATIC_REQUIRE(lsxcommon::fastload::kArenaAlign == 256);
    CHECK(AlignUp(0, 256) == 0);
    CHECK(AlignUp(1, 256) == 256);
    CHECK(AlignUp(255, 256) == 256);
    CHECK(AlignUp(256, 256) == 256);
    CHECK(AlignUp(257, 256) == 512);
    CHECK(AlignUp(4096, 256) == 4096);
}

TEST_CASE("FindTensorSuffix matches by name suffix", "[fastload]")
{
    using lsxcommon::fastload::FindTensorSuffix;
    lsxcommon::TensorBatchViews batch{};
    lsxcommon::TensorView a{};
    a.name = "model.layers.3.input_layernorm.weight";
    lsxcommon::TensorView b{};
    b.name = "model.layers.3.mlp.gate.weight";
    batch.tensors = {a, b};

    const auto *hit = FindTensorSuffix(batch, "input_layernorm.weight");
    REQUIRE(hit != nullptr);
    CHECK(hit->name == "model.layers.3.input_layernorm.weight");

    CHECK(FindTensorSuffix(batch, "missing.weight") == nullptr);
    // A "suffix" longer than the name must not match.
    CHECK(FindTensorSuffix(batch, "x.model.layers.3.mlp.gate.weight") == nullptr);
}
```

Check field names against `libs/lsxcommon/include/lsxcommon/schema.h` (`TensorBatchViews::tensors`, `TensorView::name` are confirmed by usage in `qwen_moe_inference_load.cpp:195-205`). If `tensors` is not `std::vector<TensorView>` value-typed, adapt the fixture to how `LoadTensorBatchViews` fills it (see `src/schema.cpp:839`).

- [x] **Step 2: Register the test target**

In `tests/CMakeLists.txt`, copy the minimal pattern (mirroring lines 2-10):

```cmake
add_executable(test_fast_load test_fast_load.cpp)
target_link_libraries(test_fast_load PRIVATE lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_fast_load PRIVATE ${CMAKE_SOURCE_DIR}/libs/lsxcommon/src)
add_test(NAME test_fast_load COMMAND test_fast_load)
```

(`fast_load_internal.h` lives in `src/`, hence the private include dir.)

- [x] **Step 3: Run test to verify it fails**

Run: `cmake --build --preset linux-debug --target test_fast_load && ctest --test-dir build/linux-debug -R "^test_fast_load$"`
Expected: compile FAIL — `AlignUp`/`kArenaAlign`/`FindTensorSuffix` not declared in `lsxcommon::fastload`.

- [x] **Step 4: Implement**

Append to `libs/lsxcommon/src/fast_load_internal.h` (inside `namespace lsxcommon::fastload`, after `PinnedStagingRing`):

```cpp
// ── Shared loader helpers (deduplicated 2026-08: were copied per loader) ────

// Device-arena view alignment for coalesced access.
inline constexpr size_t kArenaAlign = 256;

inline size_t AlignUp(size_t v, size_t a) { return (v + a - 1) / a * a; }

// Drains the stream on scope exit so pinned staging never dies while a DMA
// transfer is still in flight (error paths of pinned-weight loaders).
struct StreamDrain {
    cudaStream_t s;
    ~StreamDrain() { if (s) cudaStreamSynchronize(s); }
};

// Returns the tensor whose name ends with `suffix`, or nullptr.
inline const TensorView *FindTensorSuffix(const TensorBatchViews &tensors,
                                          std::string_view suffix)
{
    for (const auto &t : tensors.tensors) {
        if (t.name.size() >= suffix.size() &&
            t.name.compare(t.name.size() - suffix.size(), suffix.size(),
                           suffix) == 0)
            return &t;
    }
    return nullptr;
}

// Same suffix lookup over NVFP4 record batches.
inline const DenseNvfp4View *FindNvfp4Suffix(const DenseNvfp4BatchViews &recs,
                                             std::string_view suffix)
{
    for (const auto &r : recs.records) {
        if (r.name.size() >= suffix.size() &&
            r.name.compare(r.name.size() - suffix.size(), suffix.size(),
                           suffix) == 0)
            return &r;
    }
    return nullptr;
}
```

Add includes at top of the header if missing: `#include <string_view>` and `#include "lsxcommon/schema.h"` (schema.h is NVCC-safe per AGENTS.md).

- [x] **Step 5: Run test to verify it passes**

Run: `cmake --build --preset linux-debug --target test_fast_load && ctest --test-dir build/linux-debug -R "^test_fast_load$"`
Expected: PASS (3 test cases).

- [x] **Step 6: Commit**

```bash
git add libs/lsxcommon/src/fast_load_internal.h tests/test_fast_load.cpp tests/CMakeLists.txt
git commit -m "refactor: hoist shared fastload helpers (AlignUp, StreamDrain, suffix find)"
```

---

### Task 2: Shared arena upload-job engine + migrate Qwen-MoE loader

The six per-loader engines differ only in job struct shape and error channel. The unified engine supersedes them: generalized `CvKind`, optional 2-D shape capture (GLM needs it), error out-param.

**Files:**
- Modify: `libs/lsxcommon/src/fast_load_internal.h`
- Modify: `libs/lsxcommon/src/qwen_moe_inference_load.cpp:21-217` (delete local engine, call shared)
- Test: existing `test_qwen_moe_loader` (characterization), new cases in `tests/test_fast_load.cpp`

**Interfaces:**
- Consumes: Task 1 helpers.
- Produces (in `lsxcommon::fastload`):
  - `enum class CvKind { kBf16ToHalf, kF32Raw, kF32ToHalf, kBf16ToF32 };`
  - `size_t OutElemSize(CvKind);`
  - `struct UploadJob { const TensorView *t; void **dst; CvKind kind; int64_t *rows; int64_t *cols; };` (aggregate-init with designated initializers at call sites; null t = skipped)
  - `cudaError_t RunUploadJobs(std::vector<UploadJob> jobs, void **arena_out, size_t *arena_bytes_out, cudaStream_t stream, PinnedStagingRing &ring, std::uint64_t &bytes_acc, std::string *error);`
  - `template <typename WeightT> cudaError_t UploadNvfp4Weight(const DenseNvfp4View *rec, WeightT &w, cudaStream_t stream, PinnedStagingRing &ring, std::uint64_t &bytes_acc, std::string *error);`

- [x] **Step 1: Write failing characterization tests**

Append to `tests/test_fast_load.cpp`:

```cpp
#include <catch2/catch_template_test_macros.hpp>
#include <cuda_runtime.h>

namespace {

// Minimal weight receiver satisfying the UploadNvfp4Weight concept contract.
struct FakeNvfp4Weight {
    int rows = 0;
    int cols = 0;
    float global_scale = 0.f;
    uint64_t packed_bytes = 0;
    uint64_t scale_bytes = 0;
    const uint8_t *d_packed = nullptr;
    const uint8_t *d_scales = nullptr;
    void Free()
    {
        if (d_packed) const_cast<uint8_t *>(d_packed) = nullptr;
        if (d_scales) const_cast<uint8_t *>(d_scales) = nullptr;
    }
};

} // namespace

TEST_CASE("RunUploadJobs allocates aligned arena and uploads", "[fastload][gpu]")
{
    using namespace lsxcommon::fastload;
    PinnedStagingRing ring;
    REQUIRE(ring.ok());

    // One tiny BF16 tensor: 2x3 -> 6 elements.
    std::vector<uint8_t> bf16(12, 0);
    lsxcommon::TensorView tv{};
    tv.name = "w";
    tv.rank = 2;
    tv.shape[0] = 2;
    tv.shape[1] = 3;
    tv.bytes = bf16.size();
    tv.data = bf16.data();

    void *arena = nullptr;
    size_t arena_bytes = 0;
    void *dst = nullptr;
    std::uint64_t acc = 0;
    std::string err;
    cudaError_t e = RunUploadJobs({UploadJob{&tv, &dst, CvKind::kBf16ToHalf}},
                                  &arena, &arena_bytes, nullptr /*legacy*/,
                                  ring, acc, &err);
    // Stream arg is filled below; see Step 4 signature (stream last-or-explicit).
    (void)e;
}
```

Note: write this test against the FINAL signature decided in Step 4 — the draft above pins behavior; adjust the literal call to match the implemented signature exactly (keep one canonical form: `RunUploadJobs(jobs, &arena, &arena_bytes, stream, ring, acc, &err)`).

Add a second case asserting alignment: two jobs, second `dst` minus first `dst` is a multiple of 256, and `arena_bytes >= sum of payload rounded up per job`.

- [x] **Step 2: Run to verify failure**

Run: `cmake --build --preset linux-debug --target test_fast_load`
Expected: FAIL — `RunUploadJobs`/`UploadNvfp4Weight` not declared.

- [x] **Step 3: Implement the engine**

Append to `fast_load_internal.h`:

```cpp
enum class CvKind {
    kBf16ToHalf,  // raw BF16 bits -> FP16 on device (in-place kernel)
    kF32Raw,      // raw F32 passthrough
    kF32ToHalf,   // host scalar F32 -> FP16 (tiny tensors only)
    kBf16ToF32,   // host scalar BF16 bits -> F32 (tiny tensors only)
};

inline size_t OutElemSize(CvKind k)
{
    return (k == CvKind::kF32Raw || k == CvKind::kBf16ToF32) ? 4 : 2;
}

struct UploadJob {
    const TensorView *t = nullptr;  // null = absent tensor, skipped
    void **dst = nullptr;
    CvKind kind = CvKind::kBf16ToHalf;
    int64_t *rows = nullptr;  // optional 2-D shape capture (GLM parity)
    int64_t *cols = nullptr;
};

inline cudaError_t ValidateUploadJob(const UploadJob &j, int64_t &numel,
                                     std::string &error)
{
    if (!j.t || !j.dst) {
        error = "null tensor or dst in upload job";
        return cudaErrorInvalidValue;
    }
    if (j.t->rank <= 0) {
        error = "malformed tensor '" + j.t->name + "' rank=0";
        return cudaErrorInvalidValue;
    }
    numel = 1;
    for (int64_t d = 0; d < j.t->rank; ++d) {
        if (j.t->shape[d] <= 0) {
            error = "malformed tensor '" + j.t->name + "'";
            return cudaErrorInvalidValue;
        }
        numel *= j.t->shape[d];
    }
    const size_t in_bytes =
        (j.kind == CvKind::kF32Raw || j.kind == CvKind::kF32ToHalf) ? 4 : 2;
    if (j.t->bytes < static_cast<size_t>(numel) * in_bytes) {
        error = "malformed tensor '" + j.t->name + "': data " +
                std::to_string(j.t->bytes) + " < needed " +
                std::to_string(static_cast<uint64_t>(numel) * in_bytes);
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}
```

Then `RunUploadJobs` — port the body of `qwen_moe_inference_load.cpp:79-162` verbatim with three changes: take `std::string *error` (may be null; write via a local `std::string err_sink` when null), fill `*j.rows`/`*j.cols` when non-null and rank==2 (GLM parity), and set `error` messages instead of returning bool. Signature:

```cpp
cudaError_t RunUploadJobs(std::vector<UploadJob> jobs,
                          void **arena_out, size_t *arena_bytes_out,
                          cudaStream_t stream, PinnedStagingRing &ring,
                          std::uint64_t &bytes_acc, std::string *error);
```

Then the NVFP4 uploader as a template (WeightT must expose `rows, cols, global_scale, packed_bytes, scale_bytes, d_packed, d_scales, Free()` — satisfied by `qwen::QwenNvfp4Weight`):

```cpp
template <typename WeightT>
cudaError_t UploadNvfp4Weight(const DenseNvfp4View *r, WeightT &w,
                              cudaStream_t stream, PinnedStagingRing &ring,
                              std::uint64_t &bytes_acc, std::string *error)
{
    if (!r || r->packed_bytes == 0) return cudaSuccess;
    w.rows = r->rank >= 2 ? static_cast<int>(r->shape[0]) : 0;
    w.cols = r->rank >= 2 ? static_cast<int>(r->shape[1]) : 0;
    w.global_scale = r->global_scale;
    w.packed_bytes = r->packed_bytes;
    w.scale_bytes = r->scale_bytes;
    void *d1 = nullptr;
    cudaError_t e = cudaMalloc(&d1, static_cast<size_t>(r->packed_bytes));
    if (e != cudaSuccess) { w.Free(); return e; }
    e = ring.UploadRaw(r->packed, d1, static_cast<size_t>(r->packed_bytes),
                       stream, bytes_acc);
    if (e != cudaSuccess) { cudaFree(d1); w.Free(); return e; }
    void *d2 = nullptr;
    e = cudaMalloc(&d2, static_cast<size_t>(r->scale_bytes));
    if (e != cudaSuccess) { cudaFree(d1); w.Free(); return e; }
    e = ring.UploadRaw(r->scales, d2, static_cast<size_t>(r->scale_bytes),
                       stream, bytes_acc);
    if (e != cudaSuccess) { cudaFree(d1); cudaFree(d2); w.Free(); return e; }
    w.d_packed = static_cast<const uint8_t *>(d1);
    w.d_scales = static_cast<const uint8_t *>(d2);
    return cudaSuccess;
}
```

- [x] **Step 4: Migrate qwen_moe_inference_load.cpp**

Delete lines 21-217 of the anonymous namespace (`kArenaAlign` … `FindNvfp4`) and replace uses:
- `RunUploadJobs(...)` → `fastload::RunUploadJobs(jobs, &arena, &bytes, stream_, ring, acc, &err)` — call sites currently pass positional args; update each (there are ~4 call sites building `std::vector<UploadJob>`; they keep their job lists verbatim, adding `, nullptr, nullptr` shape fields or designated initializers).
- `UploadNvfp4Weight(r, w, ...)` → `fastload::UploadNvfp4Weight(r, w, ...)` (template deduces `qwen::QwenNvfp4Weight`).
- `FindRaw(...)` → `fastload::FindTensorSuffix(...)`; `FindNvfp4(...)` → `fastload::FindNvfp4Suffix(...)`.
- Keep `NormKind()` (model-specific policy) local.

- [x] **Step 5: Verify**

Run: `cmake --build --preset linux-debug --target test_qwen_moe_loader test_fast_load && ctest --test-dir build/linux-debug -R "test_qwen_moe_loader|test_fast_load"`
Expected: PASS (loader roundtrip ~49 assertions + new unit tests).

- [x] **Step 6: Commit**

```bash
git add libs/lsxcommon/src/fast_load_internal.h libs/lsxcommon/src/qwen_moe_inference_load.cpp tests/test_fast_load.cpp
git commit -m "refactor: unify arena upload-job engine in fastload; migrate qwen-moe loader"
```

---

### Task 3: Migrate Qwen dense loader onto the shared engine

**Files:**
- Modify: `libs/lsxcommon/src/qwen_inference_load.cpp:79-251` (delete local `kArenaAlign`/`AlignUp`/`StreamDrain`/`CvKind`/`OutElemSize`/`UploadJob`/`ValidateJob`/`RunUploadJobs`/`UploadNvfp4Weight`/`FindRaw`)
- Test: existing `test_qwen_loader`

**Interfaces:**
- Consumes: Task 2 shared engine signatures.

- [x] **Step 1: Migrate**

Identical mechanical move to Task 2 Step 4: delete the duplicated block (it is the near-verbatim source of the shared engine), redirect calls to `lsxcommon::fastload::*`, keep `NormKind` and all job-list builders local. The dense loader logs internally (`log::error`) — pass `nullptr` for the error out-param where the old path logged, or capture into a local string and log after (preserve existing log text prefix `[qwen_load]`).

- [x] **Step 2: Verify**

Run: `cmake --build --preset linux-debug --target test_qwen_loader && ctest --test-dir build/linux-debug -R "^test_qwen_loader$"`
Expected: PASS.

- [x] **Step 3: Commit**

```bash
git add libs/lsxcommon/src/qwen_inference_load.cpp
git commit -m "refactor: migrate qwen dense loader onto shared fastload engine"
```

---

### Task 4: Migrate GLM + Kimi loaders (shape-capturing Fp16Job → UploadJob)

**Files:**
- Modify: `libs/lsxcommon/src/glm_inference_load.cpp:180-274` (delete local engine; rewrite `Fp16Job` list builders to `fastload::UploadJob`)
- Modify: `libs/lsxcommon/src/kimi_inference_load.cpp` (same pattern; local engine around lines 30-260)
- Test: existing `test_glm_loader`

**Interfaces:**
- Consumes: Task 2 engine. The GLM `Fp16Job{t, __half **dst, rows, cols}` maps to `UploadJob{t, reinterpret_cast<void **>(&dst_field), CvKind::kBf16ToHalf, rows, cols}` — `__half*` storage address casts to `void**` exactly as the qwen loaders already do.

- [x] **Step 1: GLM migration**

In `glm_inference_load.cpp`: delete `kArenaAlign`, `AlignUp`, `StreamDrain`, `Fp16Job`, `JobShape`, `ValidateJob`, `UploadJobs` (lines 180-274). In `LoadSingleLayer` (line 276+) convert the job vector:

```cpp
std::vector<fastload::UploadJob> jobs = {
    {in_norm,   reinterpret_cast<void **>(&lw.d_input_norm),
                fastload::CvKind::kBf16ToHalf},
    {post_norm, reinterpret_cast<void **>(&lw.d_post_attn_norm),
                fastload::CvKind::kBf16ToHalf},
    {FindTensorSuffix(tensors, "self_attn.q_a_proj.weight"),
                reinterpret_cast<void **>(&lw.d_q_a_proj),
                fastload::CvKind::kBf16ToHalf, &lw.q_a_rows, &lw.q_a_cols},
    // ... remaining entries follow the same mapping from the deleted
    // Fp16Job initializer list (every entry keeps its tensor/dst/rows/cols).
};
void *arena = nullptr; size_t arena_bytes = 0;
cudaError_t e = fastload::RunUploadJobs(std::move(jobs), &arena, &arena_bytes,
                                        stream, ring, bytes_acc, nullptr);
lw.d_arena = arena;
lw.arena_bytes = arena_bytes;
```

Keep the heterogeneous-grid required-tensor checks (`in_norm`/`post_norm`/dense-vs-moe gate) — that is model-specific policy. Replace local `FindTensor` with `fastload::FindTensorSuffix`.

- [x] **Step 2: Kimi migration**

Same transformation in `kimi_inference_load.cpp` (its `Fp16Job`/`JobShape` mirror GLM's). Delete the local trio, convert job lists, call `fastload::RunUploadJobs`.

- [x] **Step 3: Verify**

Run: `cmake --build --preset linux-debug --target test_glm_loader && ctest --test-dir build/linux-debug -R "glm_loader"`
Expected: PASS.

- [x] **Step 4: Commit**

```bash
git add libs/lsxcommon/src/glm_inference_load.cpp libs/lsxcommon/src/kimi_inference_load.cpp
git commit -m "refactor: migrate GLM and Kimi loaders onto shared fastload engine"
```

---

### Task 5: Migrate Gemma4 + DeepSeek loaders; retire remaining copies

**Files:**
- Modify: `libs/lsxcommon/src/gemma4_inference_load.cpp:53-84` (local `kArenaAlign`/`AlignUp`/`Fp16Job`/`ValidateJob`/`UploadJobs`)
- Modify: `libs/lsxcommon/src/deepseek_inference_load.cu:31-181` (local `find_tensor_suffix`/`ArenaJob`/`ValidateArenaJob`/`UploadLayerJobs`)
- Test: existing `test_gemma4_loader`

**Interfaces:**
- Consumes: Task 2 engine. DeepSeek's typed-constructor `ArenaJob` becomes construction-site code emitting `fastload::UploadJob`s (same tensor/dst pairs; kinds map: FP16 payloads → `kBf16ToHalf` where source is BF16, `kF32Raw` for F32 scales/buffers — verify each site's actual dtype before choosing; the audit confirms both variants exist).

- [x] **Step 1: Gemma4** — same deletion + conversion as Task 4.
- [x] **Step 2: DeepSeek** — delete `find_tensor_suffix`, `ArenaJob`, `ValidateArenaJob`, `UploadLayerJobs`; `UploadLayerJobs` call sites become `fastload::RunUploadJobs(std::move(jobs), ...)` with jobs built where `ArenaJob`s were built. Keep DeepSeek-specific log prefix `[load_pinned]` by capturing the error string and logging locally.
- [x] **Step 3: Verify**

Run: `cmake --build --preset linux-debug --target test_gemma4_loader test_deepseek_loader && ctest --test-dir build/linux-debug -R "gemma4_loader|deepseek_loader"`
Expected: PASS.

- [x] **Step 4: Commit**

```bash
git add libs/lsxcommon/src/gemma4_inference_load.cpp libs/lsxcommon/src/deepseek_inference_load.cu
git commit -m "refactor: migrate gemma4 and deepseek loaders onto shared fastload engine"
```

---

### Task 6: Unified device LRU expert cache

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/expert_device_cache.h`
- Delete: `libs/lsxcommon/include/lsxcommon/glm_expert_cache.h`, `libs/lsxcommon/include/lsxcommon/qwen_moe_expert_cache.h`
- Modify: `libs/lsxcommon/src/deepseek_inference_load.cuh:107-143` (delete `ExpertDeviceCache` struct, alias to shared)
- Modify: every include/user of the three deleted names:
  - `libs/lsxcommon/src/glm_v5_model.*` transitively via `glm_inference_load.h` (check `rg 'glm_expert_cache.h|GlmExpertDeviceCache'`)
  - `libs/lsxcommon/src/qwen_3_5_moe_model.h:16`, `qwen_moe_inference_load.{h,cpp}` (`AllocateQwenMoeExpertCache`, `FreeQwenMoeExpertCache`, `EnsureQwenMoeExpertResident`)
  - `libs/lsxcommon/src/glm_inference_load.cu` inline cache management (alloc/reset/free sites at lines ≈741, 909-911, 1004-1007, 1307)
  - `libs/lsxcommon/src/deepseek_inference_pipeline.cu` (bookkeeping users)
  - `tests/test_glm_expert_cache.cpp`, `tests/test_qwen_moe_kernels.cu`
- Test: existing `test_glm_expert_cache` (rewrite to shared type), `test_qwen_moe_kernels`

**Interfaces:**
- Produces (`lsxcommon::gpu`, NVCC-safe plain-int header):
  - `struct ExpertDeviceCache { uint8_t *d_base, *d_pack, *d_scale; int slots; int64_t pack_stride, scale_stride; int total_experts; int32_t *h_slot_of, *h_slot_owner, *h_lru; int64_t hits, misses; std::size_t DeviceBytes() const; };`
  - methods: `void Touch(int slot); int Lookup(int flat); int Insert(int flat);` (semantics identical to current GLM/Qwen free functions, bodies moved unchanged)
  - host helpers (free functions, shared): `cudaError_t AllocateExpertDeviceCache(const ExpertStore-like &store, int slots, gpu::ExpertDeviceCache &out, std::string *error);` `void FreeExpertDeviceCache(gpu::ExpertDeviceCache &c);` — port bodies from `AllocateQwenMoeExpertCache`/`FreeQwenMoeExpertCache` (`qwen_moe_inference_load.cpp:781-849`), parameterized on `(total_experts, pack_stride, scale_stride)` primitives rather than the store type so GLM's inline management and Qwen both call it. If GLM's allocation reads strides from different fields, extract those two ints at the call site.

- [x] **Step 1: Rewrite `tests/test_glm_expert_cache.cpp` to the new type first (red)**

Change `using GlmExpertDeviceCache = lsxcommon::glm::GlmExpertDeviceCache;` to `using Cache = lsxcommon::gpu::ExpertDeviceCache;` and method calls (`CacheLookup(c, i)` → `c.Lookup(i)` etc.). Assertion values stay identical (they pin LRU semantics). Also flip `tests/test_qwen_moe_kernels.cu:41-61` bookkeeping cases to `c.Lookup/Insert` form.

Run: `cmake --build --preset linux-debug --target test_glm_expert_cache` → Expected FAIL (header missing).

- [x] **Step 2: Create the shared header**

Port `glm_expert_cache.h` wholesale: same comment contract, rename struct → `ExpertDeviceCache` in `lsxcommon::gpu`, free helpers become inline methods `Touch/Lookup/Insert` (bodies unchanged), keep `DeviceBytes()` with the size_t-safe cast from the Qwen variant (superset). No absl, no CUDA API — pure int bookkeeping.

- [x] **Step 3: Rewire consumers**

- `deepseek_inference_load.cuh`: delete struct, add `using ExpertDeviceCache = lsxcommon::gpu::ExpertDeviceCache;` inside `namespace lsxcommon::gpu` — pipeline `.cu` files keep compiling; migrate their helper calls to methods opportunistically (mechanical).
- `qwen_moe_expert_cache.h`: delete; `qwen_moe_inference_load.h` includes `expert_device_cache.h`; `QwenMoeExpertDeviceCache` aliases disappear from public surface — update `qwen_3_5_moe_model.h:63` member type to `lsxcommon::gpu::ExpertDeviceCache`.
- `Allocate/Free/Ensure...Resident`: rename to shared `AllocateExpertDeviceCache(store, slots, cache, &err)` / `FreeExpertDeviceCache(cache)` / keep `EnsureQwenMoeExpertResident` name (it encodes Qwen staging layout — model-specific) but have it use the shared cache type. GLM inline sites switch to the shared Allocate/Free.
- Update `rg 'glm_expert_cache.h|qwen_moe_expert_cache.h'` until zero hits; delete both headers.

- [x] **Step 4: Verify**

Run: `cmake --build --preset linux-debug && ctest --test-dir build/linux-debug -R "expert_cache|moe_kernels|moe_loader"` (sequential)
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add -A libs/lsxcommon tests
git commit -m "refactor: unify three device LRU expert caches into gpu::ExpertDeviceCache"
```

---

### Task 7: Delete legacy Loader + empty RMSNorm stub

**Files:**
- Delete: `libs/lsxcommon/src/loader.cpp`, corresponding header (`libs/lsxcommon/include/lsxcommon/loader.h` — verify exact path via `rg 'class Loader'`)
- Modify: `libs/lsxcommon/CMakeLists.txt` (remove `loader.cpp` from sources), `tests/CMakeLists.txt` (rework `test_deepseek_loader` deps)
- Modify: `tests/test_deepseek_loader.cpp` (port 2 Loader-using test cases to `ExpertSchema::UnpackBatch`)
- Delete: `libs/lsxcommon/src/deepseek_inference_norm_kernels.cu` entry — the EMPTY STUB file is `libs/lsxcommon/src/deepseek_inference_norm.cu` (audit #6: body `{}`, zero launch sites). Remove the file and its CMake reference.

- [x] **Step 1: Port tests (red→green is the port itself)**

Read the 2 test cases referencing `Loader` (`test_deepseek_loader.cpp:63,257`). Each exercises expert-arrow decoding; re-express against `ExpertSchema::UnpackBatch` (`schema.cpp:159`) which produces `PackedExpert` from the same Arrow batch. Keep the same assertions (round-trip + E4M3 encoding). Example skeleton:

```cpp
TEST_CASE("ExpertSchema::UnpackBatch decodes legacy fixture", "[deepseek][loader]")
{
    // Arrange: build the same 9-column batch the old Loader test built
    // (reuse the fixture builder already present in the file).
    // Act: lsxcommon::ExpertSchema schema; schema.UnpackBatch(*batch, out);
    // Assert: identical packed-bytes/scale/global-scale expectations.
}
```

Preserve the existing fixture-builder functions; only the decode call changes.

- [x] **Step 2: Delete dead code**

Remove loader.cpp/.h from CMake sources; delete `deepseek_inference_norm.cu` and drop it from `lsxcommon_cu` sources in `libs/lsxcommon/CMakeLists.txt`. Grep `rg '\bLoader\b|loader.h' apps libs tests` to confirm nothing else references the class (production already doesn't).

- [x] **Step 3: Verify**

Run: `cmake --build --preset linux-debug && ctest --test-dir build/linux-debug -R "deepseek_loader"` then full sequential suite at Phase end.
Expected: PASS.

- [x] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: remove prod-dead legacy Loader and empty norm stub; port tests to ExpertSchema"
```

---

### Task 8: runtime::FillInferenceResult / FillBenchmarkResult

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/model_runtime.h`, `libs/lsxcommon/src/model_runtime.cpp`
- Modify (migration): `src/glm_v5_model.cpp:216-240`, `src/kimi_v5_model.cpp:134-154`, `src/qwen_3_5_moe_model.cpp:173-192`, `src/qwen_base_model.cpp:115-138`, `src/gemma_base_model.cpp:157-174`, benchmark blocks (`glm_v5_model.cpp:272-279`, `qwen_base_model.cpp:188-194`, `gemma_base_model.cpp:246-254`, `deepseek_v4_flash_model.cpp:91-98`)
- Test: extend `tests/test_model_runtime.cpp`

**Interfaces:**
- Produces:

```cpp
namespace lsxcommon::runtime {
// Non-owning view over the five same-shaped XxxGenerationResult structs.
struct GenResultView {
    const std::vector<int32_t> *tokens;
    const std::vector<float> *final_logits;   // may be null
    double ttft_ms;
    double total_ms;
    const std::string *error_msg;
    bool eos_terminated;
    int kv_slots_used;          // -1 => caller computes fallback itself
    float first_maxv;           // NaN/0 when family lacks it
    std::uint64_t resident_bytes;
};

// Maps a family generation result into InferenceResult (ok/num_tokens/
// gen_tokens/final_logits/ttft/tokens_per_sec/eos/kv/resident). Does NOT
// decode output_text and does NOT touch session state.
void FillInferenceResult(gpu::InferenceResult &r, const GenResultView &g,
                         int num_layers);

void FillBenchmarkResult(gpu::BenchmarkResult &r, double total_ms,
                         int num_tokens, int num_layers,
                         std::uint64_t resident_bytes);
} // namespace lsxcommon::runtime
```

- [x] **Step 1: Write failing unit tests** in `tests/test_model_runtime.cpp` (host-only):

```cpp
TEST_CASE("FillInferenceResult maps tokens, timings, eos", "[model_runtime]")
{
    using namespace lsxcommon;
    gpu::InferenceResult r;
    std::vector<int32_t> toks{7, 8, 9};
    std::vector<float> logits{0.1f, 0.2f};
    std::string err;
    runtime::GenResultView g{&toks, &logits, 12.5, 100.0, &err, true,
                             /*kv_slots_used=*/17, 3.25f, 12345};
    runtime::FillInferenceResult(r, g, /*num_layers=*/79);
    CHECK(r.ok);
    CHECK(r.num_tokens == 3);
    CHECK(r.num_layers == 79);
    CHECK(r.first_gen_tok_id == 7);
    CHECK(r.gen_tokens == toks);
    CHECK(r.final_logits == logits);
    CHECK(r.ttft_ms == 12.5);
    CHECK(r.total_time_ms == 100.0);
    CHECK(r.tokens_per_sec == Approx(3.0 / 0.1));   // 3 tok / 0.1 s
    CHECK(r.eos_terminated);
    CHECK(r.kv_slots_used == 17);
    CHECK(r.weight_resident_bytes == 12345);
    CHECK(r.output_text.empty());  // Fill must NOT decode text (caller does)
}

TEST_CASE("FillInferenceResult handles empty token stream", "[model_runtime]")
{
    using namespace lsxcommon;
    gpu::InferenceResult r;
    std::vector<int32_t> toks;
    runtime::GenResultView g{&toks, nullptr, 0, 0, nullptr, false, -1, 0.f, 0};
    runtime::FillInferenceResult(r, g, 40);
    CHECK(r.ok);
    CHECK(r.num_tokens == 0);
    CHECK(r.first_gen_tok_id == -1);
    CHECK(r.kv_slots_used == -1);           // caller fallback preserved
    CHECK(r.final_logits.empty());
}
```

- [x] **Step 2: Run red**: `ctest --test-dir build/linux-debug -R test_model_runtime` → FAIL (undefined).
- [x] **Step 3: Implement** in `model_runtime.cpp` — straight-line mapping extracted verbatim from `glm_v5_model.cpp:216-232` (tokens_per_sec guard included: `(total_ms > 0.0) ? (n / (ms/1000)) : 0.0`; note GLM divides by `gen_tokens.size()` post-move — normalize to `n = tokens->size()` computed BEFORE any move semantics; view holds const refs so no moves happen inside Fill; models keep moving their own vectors after Fill).
- [x] **Step 4: Migrate the 5 Infer sites + 4 Benchmark sites** — replace hand-mapped blocks with `FillInferenceResult` + family extras (`first_gen_maxv`, session_tail assignment, output_text decode stay at call sites). For Benchmark: `FillBenchmarkResult(r, gr.total_ms, n, layers, gr.resident_bytes)`.
- [x] **Step 5: Verify**: `cmake --build --preset linux-debug && ctest --test-dir build/linux-debug -R "model_runtime|glm_model|qwen_model|gemma4_model|moe_model"`.
- [x] **Step 6: Commit** `refactor: unify GenerationResult->InferenceResult/BenchmarkResult mapping in runtime`

---

### Task 9: Shared incremental-session policy (Qwen KV families)

Both `qwen_base_model.cpp:82-91,108-109` and `qwen_3_5_moe_model.cpp:148-158,168-169` duplicate the continuation logic over `qwen::QwenKvCache`.

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/qwen_inference.h` (declare) + a small impl near `qwen_inference_load.cpp` or header-inline
- Modify: `src/qwen_base_model.cpp`, `src/qwen_3_5_moe_model.cpp`
- Test: extend `tests/test_fast_load.cpp`? No — create cases in `tests/test_model_runtime.cpp` (pure host logic over a fake-allocated KvCache is NOT possible without CUDA; instead factor the decision part to be testable without a live KV: operate on `(kv_allocated, last_tok, tail)` primitives).

**Interfaces:**
- Produces (`lsxcommon::qwen`):

```cpp
struct SessionDecision {
    bool is_continuation = false;
    std::vector<int32_t> gen_input;   // {last_tok} on continuation, else input
};
SessionDecision DecideSession(bool kv_allocated, int32_t kv_last_tok,
                              const std::vector<int32_t> &session_tail,
                              const std::vector<int32_t> &req_input_ids);
```

Call sites keep ownership effects (`kv_.reset(); kv_ = make_unique…; session_tail_.clear()` when `!decision.is_continuation`; `session_tail_ = r.gen_tokens` after success; reset on failure).

- [x] **Step 1: Failing tests** in `test_model_runtime.cpp`:

```cpp
TEST_CASE("DecideSession continuation vs fresh", "[session]")
{
    using namespace lsxcommon::qwen;
    const std::vector<int32_t> tail{5, 6};
    const std::vector<int32_t> req{5, 6};
    auto d = DecideSession(/*kv_allocated=*/true, /*last_tok=*/6, tail, req);
    CHECK(d.is_continuation);
    CHECK((d.gen_input == std::vector<int32_t>{6}));

    d = DecideSession(true, 6, tail, {1, 2});
    CHECK_FALSE(d.is_continuation);
    CHECK((d.gen_input == std::vector<int32_t>{1, 2}));

    d = DecideSession(false, 6, tail, req);       // kv dead -> fresh
    CHECK_FALSE(d.is_continuation);
    d = DecideSession(true, 6, {}, req);          // no tail -> fresh
    CHECK_FALSE(d.is_continuation);
}
```

- [x] **Step 2: Red run**, implement (trivial predicate mirroring `qwen_base_model.cpp:82-91`), green run.
- [x] **Step 3: Migrate both call sites.**
- [x] **Step 4: Commit** `refactor: extract shared qwen incremental-session decision`

---

### Task 10: ModelBase owns the CUDA execution context

**Files:**
- Modify: `include/lsxcommon/model_base.h`, `src/model_base.cpp`
- Migrate: `src/glm_v5_model.{h,cpp}` (delete local EnsureExecutionContext/TeardownWeights context half), `src/kimi_v5_model.cpp` (same)
- Test: extend `tests/test_model_runtime.cpp` (context lifecycle needs CUDA; assert observable state: `ContextReady()`, idempotency, teardown resets)

**Interfaces:**
- Produces (protected in `ModelBase`):

```cpp
bool EnsureContext(bool with_cublas);   // lazy-create stream (+handle), sets cublasSetStream
void TeardownContext();                 // destroy handle+stream (weights freed separately)
cudaStream_t stream() const { return ctx_stream_; }
cublasHandle_t cublas() const { return ctx_cublas_; }
bool ContextReady() const { return ctx_stream_ != nullptr; }
```

Members `ctx_stream_`, `ctx_cublas_` (null default). `TeardownContext` also called from a new `ModelBase::UnloadCommon()`? No — subclasses keep their Unload order; they just call `TeardownContext()` where they destroyed stream_/cublas_.

- [x] **Step 1: Failing test** (needs CUDA runtime — mark test `[gpu]`, it links lsxcommon already):

```cpp
#include "lsxcommon/model_base.h"
namespace {
class ContextProbe final : public lsxcommon::ModelBase {
public:
    using ModelBase::ModelBase;
    using ModelBase::EnsureContext;
    using ModelBase::TeardownContext;
    bool Load() override { return false; }
    gpu::InferenceResult Infer(const ModelRequest &) override { return {}; }
    void Unload() override {}
    gpu::InferenceResult SelfCheck(const ModelRequest &) override { return {}; }
    const std::string &Type() const override { static std::string t{"probe"}; return t; }
};
}

TEST_CASE("ModelBase context is lazy, idempotent and teardown-safe", "[gpu][modelbase]")
{
    ContextProbe m{lsxcommon::ModelInitConfig{}};
    CHECK(m.stream() == nullptr);
    CHECK(m.EnsureContext(/*with_cublas=*/true));
    cudaStream_t s1 = m.stream();
    REQUIRE(s1 != nullptr);
    CHECK(m.EnsureContext(true));            // idempotent
    CHECK(m.stream() == s1);
    CHECK(m.cublas() != nullptr);
    m.TeardownContext();
    CHECK(m.stream() == nullptr);
    CHECK(m.cublas() == nullptr);
}
```

- [x] **Step 2: Red run → implement in `model_base.cpp`** (body = GLM's `EnsureExecutionContext`/destruction half, parameterized on `with_cublas`).
- [x] **Step 3: Migrate GLM + Kimi**: replace `stream_`/`cublas_` members with base accessors; `EnsureExecutionContext()` bodies deleted; `TeardownWeights` keeps model-weight freeing then calls `TeardownContext()`. GLM passes `with_cublas=true`, Kimi `true`, later Qwen35Moe `false`, DeepSeek `false`, QwenBase/Gemma `false` (their generation APIs take bare streams).
- [x] **Step 4: Verify**: `ctest -R "glm_model|glm_loader"` + `test_model_runtime` green.
- [x] **Step 5: Commit** `refactor: ModelBase owns persistent CUDA stream/cuBLAS context`

---

### Task 11: Migrate QwenMoE + DeepSeek + QwenBase + Gemma onto the base context

This lands the anti-pattern fixes: no temp-per-call contexts anywhere.

**Files:**
- Modify: `src/qwen_3_5_moe_model.{h,cpp}` (drop `stream_` member + local EnsureExecutionContext; `EnsureContext(false)` in Load; `TeardownContext()` in TeardownWeights; Infer guards `ContextReady()`)
- Modify: `src/deepseek_v4_flash_model.cpp` (create context in Load via `EnsureContext(false)`; pass `stream()` to `load_pinned_weights`; Unload calls `TeardownContext()`)
- Modify: `src/qwen_base_model.cpp` (Load: `EnsureContext(false)` once, pass `stream()` into `LoadQwenPinnedWeights`; Infer: DELETE cublas/stream create-destroy pair, use `stream()`; Benchmark likewise)
- Modify: `src/gemma_base_model.cpp` (same pattern; also drops temp stream in Load)
- Test: behavioral pin — new test cases in `tests/test_qwen_model.cpp` style are heavy; instead rely on existing selfcheck/model suites (they exercise Infer repeatedly; a regression to per-call churn shows as load-time growth, and the persistent handle makes `stream()` stable). Add one assertion-level test where cheap: Gemma/Qwen model tests already SKIP without real weights — acceptable; the guarantee is enforced by review + Task 13's uniform preamble.

- [x] **Step 1: Migrate four models** (each: Load creates context once; Infer/Benchmark consume `stream()`; Unload tears down).
- [x] **Step 2: Verify**: `ctest -R "moe_selfcheck|deepseek_selfcheck|qwen_selfcheck|gemma4_selfcheck"` sequential — all must pass (real weights available on this box).
- [x] **Step 3: Commit** `refactor: single persistent execution context for all models (fixes per-call churn)`

---

### Task 12: Residency-lock preamble helper

**Files:**
- Modify: `include/lsxcommon/model_base.h`
- Modify: `src/glm_v5_model.cpp:196,264`, `src/kimi_v5_model.cpp:123`, `src/qwen_3_5_moe_model.cpp` (wrap generation call), `src/qwen_base_model.cpp`, `src/gemma_base_model.cpp`, `src/deepseek_v4_flash_model.cpp` (wrap `RunInferenceModel`)
- Test: `tests/test_model_runtime.cpp` (host-only, fake lock probe)

**Interfaces:**
- Produces (protected, header-inline):

```cpp
// Runs `fn` under the shared residency lock: unload-during-inference and
// parallel dispatch serialize behind unique_lock; generation itself is
// shared-safe. Every Infer/Benchmark implementation MUST route its
// generation call through this helper (uniform preamble invariant).
template <typename F>
auto RunUnderResidencyLock(F &&fn) -> decltype(fn())
{
    std::shared_lock<std::shared_mutex> lock(residency_mutex_);
    return fn();
}
```

- [x] **Step 1: Failing test**:

```cpp
TEST_CASE("RunUnderResidencyLock executes and releases", "[modelbase]")
{
    ContextProbe m{lsxcommon::ModelInitConfig{}};
    int v = m.RunUnderResidencyLock([] { return 42; });
    CHECK(v == 42);
    // Lock must be releasable afterwards (unique_lock acquisition succeeds).
    std::unique_lock<std::shared_mutex> ul(m.ResidencyMutex());
}
```

Expose `RunUnderResidencyLock` in ContextProbe via `using`.
- [x] **Step 2: Red → implement → green.**
- [x] **Step 3: Wrap all seven models' generation invocations** (GLM/Kimi lose their manual `std::shared_lock` blocks; the other five gain locking — closes the residency-lock gap).
- [x] **Step 4: Verify**: sequential run of all selfcheck suites.
- [x] **Step 5: Commit** `refactor: uniform residency-lock preamble for every model generation`

---

### Task 13: SyntheticGeometryAllowed + field-table geometry validation

**Files:**
- Modify: `include/lsxcommon/model_base.h`, `src/model_base.cpp`
- Modify: `src/qwen_base_model.cpp:19-24` (delete local fn), `src/gemma_base_model.cpp:42-47`, `src/qwen_3_5_moe_model.cpp:18-22`, and validators: `src/qwen_v5_model.cpp:16-36`, `src/qwen_3_8_27b_model.cpp:16-36`, `src/qwen_3_5_moe_model.cpp:51-86`, `src/gemma_4_26b_a4b_model.cpp:29-44`
- Test: new file `tests/test_geometry_guard.cpp`

**Interfaces:**
- Produces (public static + protected in ModelBase):

```cpp
static bool SyntheticGeometryAllowed();   // LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY

struct GeoFieldCheck {
    const char *name;
    long long expected;
    long long actual;
};
// Logs one uniform error line per mismatch ("[<tag>] geometry mismatch:
// <name>: expected X, got Y") and returns false on the first mismatch.
// Returns true when every field matches. Synthetic bypass is the CALLER's
// decision (some validators gate layer_types loops separately).
bool CheckGeometryFields(const char *tag,
                         const GeoFieldCheck *fields, size_t n) const;
```

- [x] **Step 1: Failing tests** (`tests/test_geometry_guard.cpp`, register target like Task 1; link lsxcommon; reuse the ContextProbe pattern subclass to reach protected method):

```cpp
TEST_CASE("CheckGeometryFields reports first mismatch uniformly", "[geometry]")
{
    Probe p{lsxcommon::ModelInitConfig{}};   // subclass exposing CheckGeometryFields
    lsxcommon::GeoFieldCheck f[] = {
        {"hidden_size", 2048, 2048},
        {"num_layers",  40,   39},
    };
    CHECK_FALSE(p.CheckGeometryFields("probe-model", f, 2));
    f[1].actual = 40;
    CHECK(p.CheckGeometryFields("probe-model", f, 2));
}
```

- [x] **Step 2: Red → implement → green.**
- [x] **Step 3: Migrate validators** — each `check(top.X == K, "X")` ladder becomes a `GeoFieldCheck` array `{...}` + one call, preceded by `if (SyntheticGeometryAllowed()) return true;` (now `ModelBase::`). Qwen35Moe keeps its `layer_types` loop after the table call (model-specific hybrid pattern). Delete the 3 local `SyntheticGeometryAllowed` copies.
- [x] **Step 4: Verify**: `ctest -R "qwen_model|3_8_27b|moe_model|gemma4_model|geometry_guard"`.
- [x] **Step 5: Commit** `refactor: shared geometry-field validator and synthetic-geometry gate`

---

### Task 14: Compile-time geometry locks (protocol step 3)

**Files:**
- Modify: `libs/lsxcommon/src/deepseek_inference_load.cuh:22-26` (add `static_assert`s) + `src/deepseek_v4_flash_model.cpp` (topology cross-check)
- Modify: `include/lsxcommon/glm_inference_load.h:20-21`, `include/lsxcommon/kimi_inference_load.h:22-48` (add `static_assert`s)
- Modify: `src/qwen_3_5_moe_model.h:26-42` (delete duplicated constants; keep `using` of `qwen_moe::frozen`)
- Test: `tests/test_deepseek_selfcheck.cpp` gains a mismatch-rejection case; compile-fail coverage via static_assert is inherent.

- [x] **Step 1: static_asserts.** Add relational asserts mirroring the qwen pattern:

```cpp
// deepseek_inference_load.cuh (inside namespace gpu)
static_assert(kNumLayers > 0 && kNumLayers % 1 == 0, "bad layer count");
static_assert(kTopK > 0 && kTopK <= kNumRouted, "top-k exceeds routed experts");
static_assert(kHiddenDim % 256 == 0, "hidden must tile cleanly on Blackwell");
static_assert(kIntermediateDim > 0, "bad MLP geometry");

// glm_inference_load.h
static_assert(kGlmHidden == 6144 && kGlmNumLayers == 79,
              "GLM-5.2 frozen passport drifted");
static_assert(kGlmNumRouted == 256, "GLM-5.2 routed expert count locked");
// kimi_inference_load.h
static_assert(kKimiHidden == 7168 && kKimiNumLayers == 61,
              "Kimi-K2.7 frozen passport drifted");
```

(Verify exact constant names with `rg 'constexpr' include/lsxcommon/glm_inference_load.h include/lsxcommon/kimi_inference_load.h` before writing; adjust names/values to actual declarations.)
- [x] **Step 2: Qwen35Moe constant dedup.** In `qwen_3_5_moe_model.h` delete `kHidden…kSharedIntermediate` (lines 26-36) and the asserts referencing them (38-42); add `using namespace qwen_moe::frozen;` is already available in the .cpp via `qwen_moe_constants.h` — instead qualify explicitly (`qwen_moe::frozen::kHidden`) in the remaining `ValidateGeometry` and keep the `layer_types` loop. Move the three `static_assert`s to `qwen_moe_constants.h` bottom (single source of truth):

```cpp
static_assert(frozen::kTopK <= frozen::kNumExperts, "top-k exceeds experts");
static_assert(frozen::kLayers % frozen::kFullAttnInterval == 0,
              "hybrid interval pattern incomplete");
static_assert(frozen::kMoeIntermediate > 0 && frozen::kSharedIntermediate > 0,
              "bad MoE geometry");
```

- [x] **Step 3: DeepSeek runtime topology gate.** In `DeepSeekV4FlashModel::Load` after `load_topology`:

```cpp
if (topo.num_layers != gpu::kNumLayers ||
    topo.experts_per_layer != gpu::kNumRouted) {
    lsxcommon::log::error(absl::StrCat(
        "[deepseek-model] topology mismatches frozen passport: layers ",
        topo.num_layers, "!=", kNumLayers, " or experts ",
        topo.experts_per_layer, "!=", kNumRouted));
    return false;
}
```

- [x] **Step 4: Rejection test.** In `tests/test_deepseek_selfcheck.cpp` (or `test_deepseek_loader.cpp`) add a synthetic-config case: point a ModelInitConfig at a temp dir whose `logestix/model.index.json` declares `num_layers = kNumLayers + 1` (write a minimal index json in-test), expect `Load() == false` with the mismatch message. Mirror how existing synthetic fixtures fabricate model dirs (grep `model.index.json` in tests for a builder).
- [x] **Step 5: Verify**: full rebuild (static_asserts compile-gate) + `ctest -R "deepseek|moe_model"`.
- [x] **Step 6: Commit** `feat: enforce frozen-geometry passports (DeepSeek gate, GLM/Kimi/QwenMoE static_asserts)`

---

### Task 15: Template factory registration

**Files:**
- Modify: `include/lsxcommon/model_factory.h`, `src/model_registry.cpp`
- Test: existing `tests/test_model_factory.cpp` (must stay green; add one case registering a probe type via the template)

**Interfaces:**
- Produces:

```cpp
class LsxModelFactory {
public:
    static void Register(const std::string &type, CreateModelFn fn);
    // Typed registration: registers `T::kType` producing std::make_unique<T>.
    template <typename T>
    static void Register()
    {
        Register(T::kType, [](const ModelInitConfig &cfg) {
            return std::unique_ptr<IModel>(new T(cfg));
        });
    }
    ...
};
```

- [x] **Step 1: Failing test** in `test_model_factory.cpp`:

```cpp
SECTION("typed Register<T>() resolves kType and constructs")
{
    struct FakeModel final : lsxcommon::IModel { /* minimal overrides as existing fakes */ 
        static const std::string kType; };
    const std::string FakeModel::kType = "fake-probe";  // (file-scope definition)
    lsxcommon::LsxModelFactory::Register<FakeModel>();
    auto m = lsxcommon::LsxModelFactory::Create({/*path*/"", /*type*/"fake-probe"});
    REQUIRE(m != nullptr);
    CHECK(m->Type() == "fake-probe");
}
```

(Adapt to the fake pattern already used in that file; unregistering between sections follows whatever cleanup the file already does — if none, use a unique type string.)
- [x] **Step 2: Red → implement → green.**
- [x] **Step 3: Collapse registry**: each block becomes `LsxModelFactory::Register<DeepSeekV4FlashModel>();` × 7.
- [x] **Step 4: Verify**: `ctest -R model_factory` + `test_generation_cli`.
- [x] **Step 5: Commit** `refactor: templated LsxModelFactory::Register<T>; collapse registry boilerplate`

---

### Task 16: SelfCheck options; absorb Gemma custom SelfCheck

**Files:**
- Modify: `include/lsxcommon/model_runtime.h`, `src/model_runtime.cpp` (`RunParisSelfCheck` overload)
- Modify: `src/gemma_base_model.cpp:191-234` (replace body with delegation)
- Test: `tests/test_model_runtime.cpp` (fake IModel driving deterministic token streams)

**Interfaces:**
- Produces:

```cpp
struct ParisSelfCheckOptions {
    bool frame_prompt = false;      // true: rebuild input_ids from the Paris
                                    // prompt via model.BuildPromptInputIds
    int min_tokens = 2;             // forced lower bound on max_gen_tokens
    int answer_window = 1;          // decode first N generated tokens
    bool substring_match = false;   // true: pass iff decoded window contains
                                    // "Paris"; false: stripped equality
};
gpu::InferenceResult RunParisSelfCheck(IModel &model, const ModelRequest &req,
                                       const ParisSelfCheckOptions &opts = {});
```

Default options reproduce today's behavior byte-for-byte (existing callers unchanged). Gemma delegates with `{frame_prompt=true, min_tokens=8, answer_window=12, substring_match=true}` — preserving its framing, window and `find("Paris")` semantics.

- [x] **Step 1: Failing tests** with the existing fake-model harness in `test_model_runtime.cpp` (grep for an existing fake; if absent, minimal `FakeSelfCheckModel : IModel` returning canned `gen_tokens` decoded by a stub tokenizer — reuse whatever `test_model_factory` fakes do for Tokenizer()):

Cases: (a) default options force `max_gen_tokens=max(2,x)` and equality-match first token; (b) `substring_match` passes when "Paris" appears at position 5 of a 12-token window; (c) `frame_prompt` triggers `BuildPromptInputIds` and forces min_tokens=8; (d) not-ok Infer short-circuits with `selfcheck_passed=false`.
- [x] **Step 2: Red → implement** (generalize the loop at `model_runtime.cpp:50-88`: iterate ntok in `[1..answer_window]` decoding prefixes; match mode per option).
- [x] **Step 3: Gemma delegation** (body shrinks to one call). Existing `test_gemma4_selfcheck` (real weights) is the acceptance gate.
- [x] **Step 4: Verify**: `ctest -R "model_runtime|gemma4_selfcheck"` then FULL sequential suite (Phase C close-out).
- [x] **Step 5: Commit** `refactor: option-driven Paris selfcheck; delete Gemma custom copy`

---

### Task 17: Full-suite verification + spec success criteria sweep

- [x] **Step 1:** `rg 'SyntheticGeometryAllowed\(\)'` → exactly one hit (model_base). `rg 'RunUploadJobs'` → only fastload + call sites. `rg 'GlmExpertDeviceCache|QwenMoeExpertDeviceCache'` → zero hits outside history. `rg 'class Loader'` → zero.
- [x] **Step 2:** Full sequential suite: `ctest --test-dir build/linux-debug` (~30 min; run unattended, no `-j`). Expected 100% pass, `test_qwen_kv_cache` skipped as before.
- [x] **Step 3:** LOC delta sanity: `git diff --stat main...HEAD | tail -1` — expect net negative several hundred lines excluding tests.
- [x] **Step 4:** Commit any residual fixes `fix: post-unification sweep`; push branch if remote configured.

---

## Self-Review Notes (already applied)

- Spec coverage: §1.1 → Tasks 1-5; §1.2 → Task 6; §1.3 → Task 7; §2.1 → Tasks 10-11; §2.2 → Task 12; §2.3 → Task 8; §2.4 → Tasks 9/11/12; §3.1 → Task 13; §3.2 → Task 14; §3.3 → Task 15; §4 → Task 16; TDD strategy → per-task red steps + Task 17.
- Type consistency: `ExpertDeviceCache` (gpu) vs `GeoFieldCheck`/`GenResultView`/`ParisSelfCheckOptions`/`SessionDecision` names used consistently across consuming tasks.
- Known risk flagged in plan text: DeepSeek `ArenaJob` dtype mapping (Task 5) requires reading each site before choosing `CvKind` — the audit confirms both BF16-source and F32 buffers exist.
