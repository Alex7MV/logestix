# SelfCheck Speedup (Qwen3.5-35B-A3B + gemma-4-26B-A4B-it) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut `test_qwen_moe_selfcheck` (~177 s) and `test_gemma4_selfcheck` (~87 s) to seconds by hoisting per-call repeated work (expert-index parsing, dequantized tables, workspace allocs, H2D expert uploads) into persistent model-owned caches, without changing any numerics.

**Architecture:** Production-path caching (spec approach A). Gemma gets a persistent expert index + a device slot LRU cache for routed experts + a reusable cublas/workspace context. Qwen MoE drops a dead 1 GB head table, persists the dequantized FP16 embed table and its workspace arena, and widens the expert LRU. All caches hold byte-identical data computed once, so bit-exact determinism is preserved by construction; the existing selfcheck tests are the green TDD fixture.

**Tech Stack:** C++20 / CUDA 13.3 (Blackwell sm_120/121), CMake presets, Catch2 v3, spdlog, Arrow IPC mmap.

**Spec:** `docs/superpowers/specs/2026-08-25-selfcheck-speedup-design.md`

**Build/test commands (Windows, VS 2026 toolchain — cmake/ninja NOT in PATH):**

```powershell
$cmake = "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$ctest = "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe"
& $cmake --build build --config Debug --target <target>
& $ctest --preset windows-debug -R "qwen|gemma" -C Debug
```

**Windows test restriction (AGENTS.md):** only run tests matching `-R "qwen|gemma"`.

---

## Planning-time spec amendments

During planning, code inspection revised two spec items (applied in Task 0):

1. **Q4 (batch 40 router readbacks) rejected:** layer `l+1`'s router input depends on layer `l`'s MoE output — strict sequential dependency makes cross-layer batching impossible without device-side routing (approach C). Replacement: drop the **unused** `d_head_h` FP16 table (`V*H` ≈ 1 GB allocated *and* dequanted every call at `qwen_moe_inference.cu:461-465`; the head GEMM at line 638 reads packed NVFP4 directly via `RunQwenNvfp4Gemm`).
2. **G4 folded into G2:** duplicate per-(token,expert) uploads disappear via device-cache hits; the remaining syncs are the required greedy-sampling argmax handshakes.

---

### Task 0: Baseline timing + spec amendment

**Files:**
- Modify: `docs/superpowers/specs/2026-08-25-selfcheck-speedup-design.md`

- [ ] **Step 1: Record baseline wall-times**

```powershell
& $ctest --preset windows-debug -R "qwen_moe_selfcheck|gemma4_selfcheck" -C Debug --output-on-failure
```

Record both durations (expected ≈177 s / ≈87 s). These numbers go into the final commit message of Task 10.

- [ ] **Step 2: Amend the spec**

In the Qwen changes table, replace the Q4 row:

```markdown
- **Q4** ~~Batch the 40 per-layer D2H router readbacks~~ **Rejected during planning**: layer l+1's
  router input depends on layer l's MoE output (strict sequential dependency). Replacement:
  **drop the unused `d_head_h` FP16 table** (V×H ≈ 1 GB allocated + dequanted every call;
  the head GEMM reads packed NVFP4 directly via `RunQwenNvfp4Gemm`, `qwen_moe_inference.cu:638`).
```

In the Gemma changes list, replace G4:

```markdown
- **G4** Folded into G2: cache hits eliminate duplicate per-(token,expert) uploads; remaining
  syncs (per-step argmax handshake) are required for greedy sampling.
```

Update the VRAM budget table row for Qwen: remove "embed/head 2.0", use "embed table 1.0".

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-08-25-selfcheck-speedup-design.md
git commit -m "docs: amend selfcheck speedup spec after planning code audit"
```

---

### Task 1: ExpertStore parse counters (Gemma G1a)

The Gemma expert index (`ExpertStore<GemmaExpertEntry>`, `libs/lsxcommon/include/lsxcommon/expert_store.h`) re-parses all 3840 arrow files on every `Infer`. First we add observable counters so the reuse can be tested.

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/expert_store.h`
- Test: `tests/test_gemma4_loader.cpp`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gemma4_loader.cpp`:

```cpp
TEST_CASE("gemma expert store parse counters", "[gemma][loader]") {
    auto out = MakeConvertedSynthetic();

    auto db = lsxcommon::gemma::load_gemma_expert_index(
        out.string(), /*num_layers=*/4, /*experts_per_layer=*/8);
    REQUIRE(db.error_msg.empty());
    // Warm-up parses every file exactly once.
    CHECK(db.parse_calls == 32);
    CHECK(db.parse_ok == 32);

    // A resident entry must not trigger a re-parse.
    const int64_t after_load = db.parse_calls;
    lsxcommon::gemma::ensure_gemma_expert_resident(db, 2, 5, nullptr);
    CHECK(db.parse_calls == after_load);

    lsxcommon::gemma::free_gemma_expert_index(db);
    std::filesystem::remove_all(out.parent_path());
}
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
& $cmake --build build --config Debug --target test_gemma4_loader
& $ctest --preset windows-debug -R "gemma4_loader" -C Debug
```

Expected: compile error — `parse_calls` is not a member of `ExpertStore`.

- [ ] **Step 3: Add the counters**

In `libs/lsxcommon/include/lsxcommon/expert_store.h`, add below `std::string error_msg;` (line ~42):

```cpp
    // Diagnostics: ParseFile invocation / success counters (tests, loaders).
    int64_t parse_calls = 0;
    int64_t parse_ok = 0;
```

At the top of `ParseFile` (line ~127), add:

```cpp
        ++parse_calls;
```

Immediately before `mmaps[idx] = std::move(mmap);` (line ~188), add:

```cpp
        ++parse_ok;
```

- [ ] **Step 4: Run test to verify it passes**

Same commands as Step 2. Expected: all gemma loader tests PASS.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/expert_store.h tests/test_gemma4_loader.cpp
git commit -m "test: add ExpertStore parse invocation counters"
```

---

### Task 2: Hoist Gemma expert index into Load() (G1b)

`GemmaBaseModel::Infer` (`libs/lsxcommon/src/gemma_base_model.cpp:110-136`) creates and frees the expert index on every call — 3840 arrow opens per Infer. Move it to a model member filled once by `Load()`.

**Files:**
- Modify: `libs/lsxcommon/src/gemma_base_model.h`
- Modify: `libs/lsxcommon/src/gemma_base_model.cpp`
- Test: `tests/test_gemma4_model.cpp`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gemma4_model.cpp` (add include `"gemma_base_model.h"` next to the other includes — `libs/lsxcommon/src` is already on the test target's include path):

```cpp
TEST_CASE("gemma expert index parsed once across Infers", "[gemma][model]") {
    auto out = MakeConvertedSynthetic();

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = out.string();
    cfg.model_type = "gemma-4-26b-a4b-it";
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());

    lsxcommon::ModelRequest req;
    req.input_ids = {0, 3, 5, 7};
    req.max_gen_tokens = 2;

    auto r1 = model->Infer(req);
    if (!r1.ok) FAIL("first Infer failed: " << r1.error_msg);
    auto r2 = model->Infer(req);
    if (!r2.ok) FAIL("second Infer failed: " << r2.error_msg);

    auto *gm = dynamic_cast<lsxcommon::GemmaBaseModel *>(model.get());
    REQUIRE(gm != nullptr);
    const auto &idx = gm->ExpertIndexForTest();
    CHECK(idx.parse_calls == idx.num_layers * idx.experts_per_layer);
    CHECK(idx.parse_ok == idx.num_layers * idx.experts_per_layer);

    model->Unload();
    fs::remove_all(out.parent_path());
}
```

Note: `fs` alias already exists in this file (`namespace fs = std::filesystem;`).

- [ ] **Step 2: Run test to verify it fails**

Build `test_gemma4_model`, run it. Expected: compile error — no member `ExpertIndexForTest`.

- [ ] **Step 3: Add the member + accessor**

In `libs/lsxcommon/src/gemma_base_model.h`, add to the public section (after `SetType`):

```cpp
    // Test/diagnostics access to the persistent routed-expert index.
    const gemma::GemmaExpertDb &ExpertIndexForTest() const { return expert_db_; }
```

Add to the protected members (after `session_tail_`):

```cpp
    // Routed-expert index: parsed once at Load(), reused by every Infer.
    gemma::GemmaExpertDb expert_db_;
    bool expert_db_ready_ = false;
```

- [ ] **Step 4: Fill it in Load(), consume it in Infer()**

In `gemma_base_model.cpp` `Load()`, insert before `return true;` (after `ValidateGeometry`):

```cpp
    if (!expert_db_ready_) {
        expert_db_ = gemma::load_gemma_expert_index(
            cfg_.model_path, static_cast<int>(pw_.layers.size()),
            pw_.top.error_msg.empty() ? pw_.top.num_experts : 1);
        expert_db_ready_ = true;
        spdlog::info("[gemma-model] expert index ready: {}/{} files parsed",
                     expert_db_.parse_ok,
                     expert_db_.num_layers * expert_db_.experts_per_layer);
    }
```

In `Infer()`:
- Delete the local construction block (lines 110-113):

```cpp
    auto expert_db = gemma::load_gemma_expert_index(
        cfg_.model_path,
        static_cast<int>(pw_.layers.size()),
        pw_.top.error_msg.empty() ? pw_.top.num_experts : 1);
```

and replace with:

```cpp
    GemmaExpertDb &expert_db = expert_db_;
```

- In the kv-alloc failure path (line ~121) delete `free_gemma_expert_index(expert_db);`.
- After generation (line ~136) delete `free_gemma_expert_index(expert_db);`.

In `Unload()` (before `TeardownContext();`):

```cpp
    free_gemma_expert_index(expert_db_);
    expert_db_ready_ = false;
```

- [ ] **Step 5: Run gemma tests**

```powershell
& $cmake --build build --config Debug --target test_gemma4_model test_gemma4_loader test_gemma4_selfcheck
& $ctest --preset windows-debug -R "gemma4_model|gemma4_loader|gemma4_selfcheck" -C Debug --output-on-failure
```

Expected: PASS (selfcheck runs only if real weights present). This alone should already cut a large fraction of the 87 s.

- [ ] **Step 6: Commit**

```bash
git add libs/lsxcommon/src/gemma_base_model.h libs/lsxcommon/src/gemma_base_model.cpp tests/test_gemma4_model.cpp
git commit -m "perf(gemma): parse routed-expert index once at Load instead of per Infer"
```

---

### Task 3: Persistent GemmaGenContext — cublas handle + workspace (G3)

Every `RunGemmaGeneration` call creates/destroys a cuBLAS handle and ~30 workspace buffers (`gemma_inference.cu:1039-1103,1234`). Introduce an opaque per-model context owning the handle plus a capacity-grown workspace.

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/gemma4_inference.h`
- Modify: `libs/lsxcommon/src/gemma_inference.cu`
- Modify: `libs/lsxcommon/src/gemma_base_model.h`, `libs/lsxcommon/src/gemma_base_model.cpp`
- Test: existing `test_gemma4_model` (bit-exact across two Infers) is the green regression.

- [ ] **Step 1: Declare the context in the public header**

In `libs/lsxcommon/include/lsxcommon/gemma4_inference.h`, above the `RunGemmaGeneration` declaration (line ~93):

```cpp
// Opaque persistent generation context: cuBLAS handle + capacity-grown
// workspace + routed-expert device cache. Owned by the model; created lazily,
// single-threaded (one consumer).
struct GemmaGenContext;
GemmaGenContext *CreateGemmaGenContext();
void DestroyGemmaGenContext(GemmaGenContext *ctx);
```

Change the signature (add trailing param):

```cpp
cudaError_t RunGemmaGeneration(cudaStream_t stream, const GemmaPinnedWeights &pw,
                               GemmaExpertDb &db,
                               const std::vector<int32_t> &input_ids,
                               int max_gen_tokens, int eos_id, float softcap,
                               GemmaGenerationResult &out, GemmaKvCache &kv,
                               GemmaGenContext *ctx = nullptr);
```

- [ ] **Step 2: Implement the context in gemma_inference.cu**

After `struct GenWorkspace {...};` (line ~687), add:

```cpp
struct GemmaGenContext {
    ~GemmaGenContext()
    {
        if (cublas) cublasDestroy(cublas);
        if (ec.d_base) cudaFree(ec.d_base);
        delete[] ec.h_slot_of;
        delete[] ec.h_slot_owner;
        delete[] ec.h_lru;
        ec = {};
    }
    cublasHandle_t cublas = nullptr;
    GenWorkspace ws;
    bool ws_ready = false;
    size_t seq0_cap = 0, q_rows_cap = 0, kv_rows_cap = 0, rot_half_cap = 0;
    int K_cap = 0;
};

GemmaGenContext *CreateGemmaGenContext() { return new GemmaGenContext(); }
void DestroyGemmaGenContext(GemmaGenContext *ctx) { delete ctx; }

// Capacity-grown workspace reuse: reallocates only when a request exceeds
// every recorded capacity. Same allocation layout as the previous per-call
// path; contents are scratch and never carry values between calls.
cudaError_t EnsureWorkspace(GemmaGenContext &ctx, int64_t H, int64_t V,
                            int64_t MI, int64_t I_max, size_t S,
                            size_t q_rows_max, size_t kv_rows_max,
                            size_t rot_half_max, int K)
{
    GenWorkspace &ws = ctx.ws;
    if (ctx.ws_ready && ctx.seq0_cap >= S && ctx.q_rows_cap >= q_rows_max &&
        ctx.kv_rows_cap >= kv_rows_max && ctx.rot_half_cap >= rot_half_max &&
        ctx.K_cap >= K)
        return cudaSuccess;
    ws.Free();
    cudaError_t e = cudaSuccess;
    auto A = [&](void **p, size_t bytes) { return cudaMalloc(p, bytes); };
    if ((e = A((void **)&ws.d_x, S * H * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_normed, S * H * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_tmp, S * q_rows_max * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_qr, S * q_rows_max * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_kr, S * kv_rows_max * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_vr, S * kv_rows_max * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_tmp2, S * kv_rows_max * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_cos, S * rot_half_max * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_sin, S * rot_half_max * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_pos, S * 4)) != cudaSuccess ||
        (e = A((void **)&ws.d_logits, V * sizeof(float))) != cudaSuccess ||
        (e = A((void **)&ws.d_argmax, 4)) != cudaSuccess ||
        (e = A((void **)&ws.d_sel, S * K * 4)) != cudaSuccess ||
        (e = A((void **)&ws.d_selw, S * K * sizeof(float))) != cudaSuccess ||
        (e = A((void **)&ws.d_p1, MI * H)) != cudaSuccess ||
        (e = A((void **)&ws.d_s1, MI * H / 16 + 16)) != cudaSuccess ||
        (e = A((void **)&ws.d_p2, MI * H)) != cudaSuccess ||
        (e = A((void **)&ws.d_s2, MI * H / 16 + 16)) != cudaSuccess ||
        (e = A((void **)&ws.d_p3, MI * H)) != cudaSuccess ||
        (e = A((void **)&ws.d_s3, MI * H / 16 + 16)) != cudaSuccess ||
        (e = A((void **)&ws.d_w1, MI * H * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_w3, MI * H * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_w2, MI * H * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_eout, H * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_acc, S * H * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_gg, S * I_max * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_gu, S * I_max * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_eg, MI * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_eu, MI * 2)) != cudaSuccess ||
        cudaMallocHost(&ws.h_sel, S * K * 4) != cudaSuccess ||
        cudaMallocHost(&ws.h_selw, S * K * sizeof(float)) != cudaSuccess ||
        cudaMallocHost(&ws.h_arg, 4) != cudaSuccess) {
        return e != cudaSuccess ? e : cudaErrorMemoryAllocation;
    }
    ctx.seq0_cap = S; ctx.q_rows_cap = q_rows_max;
    ctx.kv_rows_cap = kv_rows_max; ctx.rot_half_cap = rot_half_max;
    ctx.K_cap = K;
    ctx.ws_ready = true;
    return cudaSuccess;
}
```

(The allocation chain above is copied verbatim from the current `RunGemmaGeneration` body, lines 1069-1100.)

Call site replacing lines 1060-1103:

```cpp
    e = EnsureWorkspace(c, H, V, MI, I_max, S, q_rows_max, kv_rows_max,
                        rot_half_max, K);
    if (e != cudaSuccess) {
        out.error_msg = "workspace alloc failed";
        return e;
    }
    GenWorkspace &ws = c.ws;
```

(`H`, `V`, `MI`, `I_max`, `K`, `S`, `q_rows_max`, `kv_rows_max`, `rot_half_max` are already in scope — lines 1029-1058.) Delete the old `GenWorkspace ws;` stack object and the field-by-field assignments (`ws.seq0 = S;` etc.). All downstream code keeps using `ws.` unchanged.

- [ ] **Step 3: Rewire RunGemmaGeneration**

In `RunGemmaGeneration` (line 1014):
- Add parameter `GemmaGenContext *ctx` (done in header; mirror here).
- Replace the `cublasCreate` block (lines 1039-1043) with:

```cpp
    GemmaGenContext local_ctx;
    GemmaGenContext &c = ctx ? *ctx : local_ctx;
    if (!c.cublas &&
        cublasCreate(&c.cublas) != CUBLAS_STATUS_SUCCESS) {
        out.error_msg = "cublas create failed";
        return cudaErrorInvalidValue;
    }
    cublasHandle_t cublas = c.cublas;
```

- Replace the whole `ws.*` allocation block (lines 1060-1103) with the `EnsureWorkspace` call site shown above.
- Delete every error-path `cublasDestroy(cublas)` (lines 1110, 1116, 1123, 1138 region) and the final cleanup pair `cublasDestroy(cublas)` / implicit ws free (lines 1233-1234). Keep `g_expert_db = nullptr;`.
- Keep all `fail`-style early returns otherwise untouched. The `local_ctx` destructor handles cleanup when `ctx == nullptr`.

- [ ] **Step 4: Own the context in GemmaBaseModel**

`gemma_base_model.h` protected members:

```cpp
    // Persistent generation context (cublas + workspace [+ expert cache]).
    gemma::GemmaGenContext *gen_ctx_ = nullptr;
```

`gemma_base_model.cpp` `Infer()` — inside `RunUnderResidencyLock`, pass the context:

```cpp
    if (!gen_ctx_) gen_ctx_ = gemma::CreateGemmaGenContext();
    ...
    return gemma::RunGemmaGeneration(
        st, pw_, expert_db, gen_input, req.max_gen_tokens,
        tok_db_ ? tok_db_->eos_id : -1,
        pw_.top.error_msg.empty() ? pw_.top.final_logit_softcapping : 0.f,
        gen, *kv_, gen_ctx_);
```

`Unload()` (before `TeardownContext();`):

```cpp
    gemma::DestroyGemmaGenContext(gen_ctx_);
    gen_ctx_ = nullptr;
```

- [ ] **Step 5: Run gemma tests (green regression)**

```powershell
& $cmake --build build --config Debug --target test_gemma4_model test_gemma4_selfcheck
& $ctest --preset windows-debug -R "gemma4_model|gemma4_selfcheck" -C Debug --output-on-failure
```

Expected: PASS — two consecutive Infers remain bit-exact (workspace contents are scratch; dequant/GEMM inputs unchanged).

- [ ] **Step 6: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/gemma4_inference.h libs/lsxcommon/src/gemma_inference.cu libs/lsxcommon/src/gemma_base_model.h libs/lsxcommon/src/gemma_base_model.cpp
git commit -m "perf(gemma): persist cublas handle and generation workspace across Infers"
```

---

### Task 4: Gemma routed-expert device slot cache (G2)

Replace per-(token,expert) H2D upload (~20 GB per cycle) with an LRU device slot cache holding each expert's NVFP4 pack+scales. Reuses `gpu::ExpertDeviceCache` bookkeeping (`libs/lsxcommon/include/lsxcommon/expert_device_cache.h`). Numerics unchanged: `DequantNvfp4` reads the same bytes from a stable device address instead of freshly uploaded ones.

**Files:**
- Modify: `libs/lsxcommon/src/gemma_inference.cu`
- Test: existing `test_gemma4_model` (two Infers: first cold misses, second warm hits — bit-exactness covers both paths).

- [ ] **Step 1: Extend GemmaGenContext**

Add the include at the top of `gemma_inference.cu` (next to the other lsxcommon includes):

```cpp
#include "lsxcommon/expert_device_cache.h"
```

Inside `struct GemmaGenContext` add:

```cpp
    // Routed-expert device slot cache: [slots] × (pack_stride + scale_stride),
    // one slot per flat expert id. Bookkeeping per gpu::ExpertDeviceCache.
    // Cleanup already lives in ~GemmaGenContext (Task 3).
    gpu::ExpertDeviceCache ec{};
    int64_t pack_stride = 0;
    int64_t scale_stride = 0;
```

And an initializer called from `RunGemmaGeneration` right after topology validation (needs `pw.top` + `db`):

```cpp
cudaError_t EnsureGemmaExpertCache(GemmaGenContext &ctx,
                                   const GemmaPinnedWeights &pw,
                                   const GemmaExpertDb &db)
{
    if (ctx.ec.slots > 0) return cudaSuccess;
    const int64_t MIH = static_cast<int64_t>(pw.top.moe_intermediate_size) *
                        pw.top.hidden_size;
    const int64_t pack = MIH / 2 * 3;       // w1|w2|w3 nibble-packed bytes
    const int64_t scale = MIH / 16 + 16;    // E4M3 per-block scales (+slack,
                                            // matches ws.d_s1 sizing)
    int slots = db.num_layers * db.experts_per_layer;
    if (const char *env = std::getenv("LSX_GEMMA_EXPERT_CACHE_SLOTS")) {
        const int v = std::atoi(env);
        if (v > 0) slots = std::min(slots, v);
    }
    slots = std::min(slots, 1024);
    if (slots <= 0) return cudaErrorInvalidValue;

    ctx.pack_stride = pack;
    ctx.scale_stride = scale;
    ctx.ec.slots = slots;
    ctx.ec.total_experts = db.num_layers * db.experts_per_layer;
    void *base = nullptr;
    cudaError_t e = cudaMalloc(&base, ctx.ec.DeviceBytes());
    if (e != cudaSuccess) return e;
    ctx.ec.d_base = static_cast<uint8_t *>(base);
    ctx.ec.d_pack = ctx.ec.d_base;
    ctx.ec.d_scale = ctx.ec.d_base +
                     static_cast<int64_t>(slots) * ctx.pack_stride;
    ctx.ec.h_slot_of = new int32_t[static_cast<size_t>(ctx.ec.total_experts)];
    ctx.ec.h_slot_owner = new int32_t[static_cast<size_t>(slots)];
    ctx.ec.h_lru = new int32_t[static_cast<size_t>(slots)];
    for (int i = 0; i < ctx.ec.total_experts; ++i) ctx.ec.h_slot_of[i] = -1;
    for (int i = 0; i < slots; ++i) {
        ctx.ec.h_slot_owner[i] = -1;
        ctx.ec.h_lru[i] = i;
    }
    return cudaSuccess;
}
```

Call site in `RunGemmaGeneration`, after the `top.error_msg` guard (line ~1028):

```cpp
    e = EnsureGemmaExpertCache(c, pw, db);
    if (e != cudaSuccess) {
        out.error_msg = "expert cache alloc failed";
        return e;
    }
    g_gen_ctx = &c;
```

Add file-scope global next to `g_expert_db` (line ~633):

```cpp
GemmaGenContext *g_gen_ctx = nullptr;
```

Set `g_gen_ctx = nullptr;` everywhere `g_expert_db = nullptr;` is assigned.

- [ ] **Step 2: Rewrite the MoE upload loop in ForwardLayer**

Replace lines 909-940 (from `ensure_gemma_expert_resident(...)` through the sixth `cudaMemcpyAsync`) with:

```cpp
            const auto &ent =
                g_expert_db->entries[static_cast<size_t>(layer_idx) *
                                         static_cast<size_t>(E) +
                                     static_cast<size_t>(expert)];
            if (!ent.host_w1 || !ent.host_w2) continue;

            gpu::ExpertDeviceCache &ec = g_gen_ctx->ec;
            const int flat = layer_idx * E + expert;
            int slot = ec.Lookup(flat);
            const bool miss = slot < 0;
            if (miss) slot = ec.Insert(flat);
            uint8_t *slot_pack = ec.d_pack +
                static_cast<int64_t>(slot) * g_gen_ctx->pack_stride;
            uint8_t *slot_scale = ec.d_scale +
                static_cast<int64_t>(slot) * g_gen_ctx->scale_stride;
            if (miss) {
                // Stream-ordered writes into the slot: safe against prior
                // same-stream readers of an evicted victim slot. Host mmaps
                // never evict (max_open=7000 > total experts post-G1), so no
                // munmap barrier is needed here.
                uint8_t *pk = slot_pack;
                uint8_t *sc = slot_scale;
                e = cudaMemcpyAsync(pk, ent.host_w1,
                                    static_cast<size_t>(ent.w1_bytes),
                                    cudaMemcpyHostToDevice, stream);
                if (e != cudaSuccess) return e;
                pk += ent.w1_bytes;
                e = cudaMemcpyAsync(pk, ent.host_w2,
                                    static_cast<size_t>(ent.w2_bytes),
                                    cudaMemcpyHostToDevice, stream);
                if (e != cudaSuccess) return e;
                pk += ent.w2_bytes;
                e = cudaMemcpyAsync(pk, ent.host_w3,
                                    static_cast<size_t>(ent.w3_bytes),
                                    cudaMemcpyHostToDevice, stream);
                if (e != cudaSuccess) return e;
                e = cudaMemcpyAsync(sc, ent.host_w1_scales,
                                    static_cast<size_t>(ent.w1_scales_bytes),
                                    cudaMemcpyHostToDevice, stream);
                if (e != cudaSuccess) return e;
                sc += ent.w1_scales_bytes;
                e = cudaMemcpyAsync(sc, ent.host_w2_scales,
                                    static_cast<size_t>(ent.w2_scales_bytes),
                                    cudaMemcpyHostToDevice, stream);
                if (e != cudaSuccess) return e;
                sc += ent.w2_scales_bytes;
                e = cudaMemcpyAsync(sc, ent.host_w3_scales,
                                    static_cast<size_t>(ent.w3_scales_bytes),
                                    cudaMemcpyHostToDevice, stream);
                if (e != cudaSuccess) return e;
            }
```

Then update the three `DequantNvfp4` calls to read from the slot (offsets mirror the upload order):

```cpp
            e = DequantNvfp4(stream, slot_pack, ent.w1_bytes, slot_scale,
                             ent.w1_scales_bytes, ent.w1_global_scale,
                             ws.d_w1, MI * H);
            ...
            e = DequantNvfp4(stream, slot_pack + ent.w1_bytes, ent.w2_bytes,
                             slot_scale + ent.w1_scales_bytes,
                             ent.w2_scales_bytes, ent.w2_global_scale,
                             ws.d_w2, H * MI);
            ...
            e = DequantNvfp4(stream, slot_pack + ent.w1_bytes + ent.w2_bytes,
                             ent.w3_bytes,
                             slot_scale + ent.w1_scales_bytes +
                                 ent.w2_scales_bytes,
                             ent.w3_scales_bytes, ent.w3_global_scale,
                             ws.d_w3, MI * H);
```

Delete the now-unused `ensure_gemma_expert_resident(...)` call. Keep `ws.d_p1..d_s3` allocations (harmless scratch; removal is optional cleanup — leave them to minimize diff).

- [ ] **Step 3: Run gemma tests**

```powershell
& $cmake --build build --config Debug --target test_gemma4_model test_gemma4_moe test_gemma4_selfcheck
& $ctest --preset windows-debug -R "gemma4" -C Debug --output-on-failure
```

Expected: PASS. The synthetic two-Infer test exercises cold-miss then warm-hit paths and requires bit-exact equality.

- [ ] **Step 4: Commit**

```bash
git add libs/lsxcommon/src/gemma_inference.cu
git commit -m "perf(gemma): LRU device slot cache for routed experts replaces per-token H2D"
```

---

### Task 5: Drop the dead d_head_h table (Qwen)

`d_head_h` (V×H halves ≈ 1 GB) is allocated and dequanted on every `RunQwenMoeGeneration` call but never consumed — the head GEMM uses packed weights directly (`qwen_moe_inference.cu:638`). Remove it.

**Files:**
- Modify: `libs/lsxcommon/src/qwen_moe_inference.cu`
- Test: existing suite (behavior-preserving deletion; `final_logits` must stay bit-identical).

- [ ] **Step 1: Remove the buffer**

In `qwen_moe_inference.cu`:
1. Delete the half_sz entry `(size_t)V * H,                   // d_head_half` (line 387).
2. Delete `__half *d_head_h = adv(ph, half_sz[19]);` (line 440).
3. Renumber subsequent positional indices down by one: `half_sz[20]`→`[19]`, `[21]`→`[20]`, … `[27]`→`[26]` (lines 441-448).
4. Delete the head-dequant block (lines 461-465):

```cpp
    e = lsxcommon::gpu::dequant_shared_nvfp4(
        stream, pw.head.d_packed, pw.head.packed_bytes, pw.head.d_scales,
        pw.head.scale_bytes, pw.head.global_scale, d_head_h,
        (int64_t)V * H);
    if (e != cudaSuccess) { result.error_msg = "dequant head"; cudaFree(base); return e; }
```

- [ ] **Step 2: Build and run Qwen tests**

```powershell
& $cmake --build build --config Debug --target lsxcommon_qwen_moe_cu test_qwen_moe_kernels test_qwen_moe_selfcheck
& $ctest --preset windows-debug -R "qwen_moe" -C Debug --output-on-failure
```

Expected: PASS with identical logits (kernels-test determinism case asserts `memcmp == 0` across runs).

- [ ] **Step 3: Commit**

```bash
git add libs/lsxcommon/src/qwen_moe_inference.cu
git commit -m "perf(qwen-moe): remove unused per-call lm_head dequant table"
```

---

### Task 6: Persistent Qwen generation scratch — embed table + arena (Q1+Q2)

Persist the dequantized FP16 embed table (computed once) and a capacity-grown workspace arena across `Infer` calls.

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/qwen_moe_inference.h`
- Modify: `libs/lsxcommon/src/qwen_moe_inference.cu`
- Modify: `libs/lsxcommon/src/qwen_3_5_moe_model.h`, `libs/lsxcommon/src/qwen_3_5_moe_model.cpp`
- Modify: `tests/test_qwen_moe_kernels.cu:478-485` (call-site signature update)
- Test: `tests/test_qwen_moe_selfcheck.cpp` (new TEST_CASE, real-model gated)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_qwen_moe_selfcheck.cpp` (add `#include "qwen_3_5_moe_model.h"` with the other includes):

```cpp
TEST_CASE("Qwen3.5-35B-A3B generation scratch persists across SelfCheck cycles",
          "[qwen-moe][model]")
{
    const std::filesystem::path model_dir = QwenMoeConvertedPath();
    const bool have_pinned =
        std::filesystem::exists(model_dir / "logestix" / "pinned") ||
        std::filesystem::exists(model_dir / "logestix-qwen35moe" / "pinned");
    if (!have_pinned)
        SKIP("Qwen3.5-35B-A3B converted model directory not present");

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model_dir.string();
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Load());

    const std::vector<int32_t> ids = EncodeSelfCheckPrompt(m->Tokenizer());
    REQUIRE(!ids.empty());

    lsxcommon::ModelRequest req;
    req.input_ids = ids;
    req.max_gen_tokens = 2;

    auto sc0 = m->SelfCheck(req);
    REQUIRE(sc0.ok);

    auto *qm = dynamic_cast<lsxcommon::Qwen35MoeModel *>(m.get());
    REQUIRE(qm != nullptr);
    const auto &scratch = qm->GenScratchForTest();
    CHECK(scratch.dequant_runs == 1);
    CHECK(scratch.arena_allocs <= 1);

    auto sc1 = m->SelfCheck(req);
    REQUIRE(sc1.ok);
    CHECK(qm->GenScratchForTest().dequant_runs == 1);
    CHECK(qm->GenScratchForTest().arena_allocs <= 1);

    REQUIRE(sc1.final_logits.size() == sc0.final_logits.size());
    CHECK(std::memcmp(sc1.final_logits.data(), sc0.final_logits.data(),
                      sc0.final_logits.size() * sizeof(float)) == 0);

    m->Unload();
}
```

- [ ] **Step 2: Verify it fails**

Build `test_qwen_moe_selfcheck`. Expected: compile error — `GenScratchForTest` does not exist.

- [ ] **Step 3: Declare scratch API in qwen_moe_inference.h**

Above `RunQwenMoeGeneration` (line ~83):

```cpp
// Persistent generation scratch owned by the model: the dequantized FP16
// embedding table (computed once, byte-identical to per-call dequant) and a
// capacity-grown workspace arena reused across Infer calls.
struct QwenMoeGenScratch {
    __half *d_embed = nullptr;      // [V*H]
    int64_t table_elems = 0;
    void *arena = nullptr;
    size_t arena_capacity = 0;
    int64_t dequant_runs = 0;       // diagnostics/tests
    int64_t arena_allocs = 0;       // diagnostics/tests
};

cudaError_t EnsureQwenMoeScratch(cudaStream_t stream,
                                 const QwenMoePinnedWeights &pw,
                                 QwenMoeGenScratch &scratch);
cudaError_t ReserveQwenMoeArena(QwenMoeGenScratch &scratch, size_t bytes);
void FreeQwenMoeGenScratch(QwenMoeGenScratch &scratch);
```

Extend the driver signature:

```cpp
cudaError_t RunQwenMoeGeneration(
    cudaStream_t stream, const QwenMoePinnedWeights &pw,
    QwenMoeExpertStore &store, QwenMoeExpertDeviceCache &cache,
    QwenMoeGenScratch &scratch,
    const std::vector<int32_t> &prompt, int max_new, int eos_id,
    lsxcommon::qwen::QwenGenerationResult &result,
    lsxcommon::sample::ISampler &sampler, lsxcommon::qwen::QwenKvCache *kv);
```

- [ ] **Step 4: Implement in qwen_moe_inference.cu**

```cpp
cudaError_t EnsureQwenMoeScratch(cudaStream_t stream,
                                 const QwenMoePinnedWeights &pw,
                                 QwenMoeGenScratch &scratch)
{
    const int64_t elems = static_cast<int64_t>(pw.top.vocab_size) *
                          pw.top.hidden_size;
    if (scratch.d_embed && scratch.table_elems == elems)
        return cudaSuccess;
    FreeQwenMoeGenScratch(scratch);
    cudaError_t e = cudaMalloc(reinterpret_cast<void **>(&scratch.d_embed),
                               static_cast<size_t>(elems) * sizeof(__half));
    if (e != cudaSuccess) return e;
    e = lsxcommon::gpu::dequant_shared_nvfp4(
        stream, pw.embed.d_packed, pw.embed.packed_bytes, pw.embed.d_scales,
        pw.embed.scale_bytes, pw.embed.global_scale, scratch.d_embed, elems);
    if (e != cudaSuccess) return e;
    scratch.table_elems = elems;
    ++scratch.dequant_runs;
    return cudaSuccess;
}

cudaError_t ReserveQwenMoeArena(QwenMoeGenScratch &scratch, size_t bytes)
{
    if (scratch.arena && scratch.arena_capacity >= bytes)
        return cudaSuccess;
    if (scratch.arena) { cudaFree(scratch.arena); scratch.arena = nullptr; }
    cudaError_t e = cudaMalloc(&scratch.arena, bytes);
    if (e != cudaSuccess) { scratch.arena_capacity = 0; return e; }
    scratch.arena_capacity = bytes;
    ++scratch.arena_allocs;
    return cudaSuccess;
}

void FreeQwenMoeGenScratch(QwenMoeGenScratch &scratch)
{
    if (scratch.d_embed) cudaFree(scratch.d_embed);
    if (scratch.arena) cudaFree(scratch.arena);
    scratch = {};
}
```

In `RunQwenMoeGeneration`:
1. At the top (before computing sizes): 

```cpp
    cudaError_t pre = EnsureQwenMoeScratch(stream, pw, scratch);
    if (pre != cudaSuccess) { result.error_msg = "scratch ensure failed"; return pre; }
```

2. Delete the `(size_t)V * H, // d_embed_half` entry from `half_sz` (line 386) and decrement every subsequent literal index by one — after Task 5 they run `half_sz[19]`…`half_sz[26]` and become `[18]`…`[25]`.
3. Replace `__half *d_embed_h = adv(ph, half_sz[18]);` usage: delete the line; substitute `scratch.d_embed` at the `RunQwenEmbed` call site (line 555).
4. Delete the embed-dequant block (lines 456-460).
5. Arena (lines 411-413): replace

```cpp
    void *base = nullptr;
    cudaError_t e = cudaMalloc(&base, half_bytes + float_bytes);
    if (e != cudaSuccess) { result.error_msg = "arena alloc failed"; return e; }
```

with

```cpp
    cudaError_t e = ReserveQwenMoeArena(scratch, half_bytes + float_bytes);
    if (e != cudaSuccess) { result.error_msg = "arena reserve failed"; return e; }
    void *base = scratch.arena;
```

6. In the `fail` lambda: delete `cudaFree(base);`. At function end: delete `cudaFree(base);` (line 698). Everything else stays.

- [ ] **Step 5: Update the kernels-test call site**

`tests/test_qwen_moe_kernels.cu` line 478-485 becomes:

```cpp
    lsxcommon::qwen_moe::QwenMoeGenScratch gen_scratch;
    REQUIRE(lsxcommon::qwen_moe::EnsureQwenMoeScratch(stream, pw, gen_scratch) ==
            cudaSuccess);

    auto run_once = [&](lsxcommon::qwen::QwenGenerationResult &r) {
        lsxcommon::qwen::QwenKvCache kv;
        lsxcommon::sample::GreedySampler sampler;
        std::vector<int32_t> prompt = {1, 2, 3};
        return lsxcommon::qwen_moe::RunQwenMoeGeneration(
            stream, pw, store, cache, gen_scratch, prompt, /*max_new=*/4,
            /*eos_id=*/pw.moe_top.eos_token_id, r, sampler, &kv);
    };
```

(Release `FreeQwenMoeGenScratch(gen_scratch);` beside the existing teardown at lines 450-455.)

- [ ] **Step 6: Model wiring**

`qwen_3_5_moe_model.h` private members (next to `cache_{}`):

```cpp
    qwen_moe::QwenMoeGenScratch scratch_{};
```

Public accessor (after `Type()`):

```cpp
    // Test/diagnostics access to persistent generation scratch.
    const qwen_moe::QwenMoeGenScratch &GenScratchForTest() const
    {
        return scratch_;
    }
```

`qwen_3_5_moe_model.cpp`: in `TeardownWeights()` add `qwen_moe::FreeQwenMoeGenScratch(scratch_);` (first line); in `Infer()` pass `scratch_` into `RunQwenMoeGeneration` (after `cache_`).

- [ ] **Step 7: Run Qwen tests**

```powershell
& $cmake --build build --config Debug --target test_qwen_moe_kernels test_qwen_moe_selfcheck test_qwen_moe_loader test_qwen_moe_convert test_qwen_moe_model
& $ctest --preset windows-debug -R "qwen_moe" -C Debug --output-on-failure
```

Expected: PASS including the new scratch-persistence case (real-model parts SKIP without weights).

- [ ] **Step 8: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/qwen_moe_inference.h libs/lsxcommon/src/qwen_moe_inference.cu libs/lsxcommon/src/qwen_3_5_moe_model.h libs/lsxcommon/src/qwen_3_5_moe_model.cpp tests/test_qwen_moe_kernels.cu tests/test_qwen_moe_selfcheck.cpp
git commit -m "perf(qwen-moe): persist dequantized embed table and workspace arena across Infers"
```

---

### Task 7: Widen Qwen expert LRU default (Q3)

**Files:**
- Modify: `libs/lsxcommon/src/qwen_3_5_moe_model.cpp:99-106`

- [ ] **Step 1: Change the default bound**

Line 100:

```cpp
    slots = std::min(slots, 512);  // VRAM-bounded default: 512 × ~1.73 MB ≈ 0.9 GB
```

Keep the env override (`LSX_QWEN_MOE_EXPERT_CACHE_SLOTS`) unchanged.

- [ ] **Step 2: Verify**

```powershell
& $cmake --build build --config Debug --target test_qwen_moe_selfcheck
& $ctest --preset windows-debug -R "qwen_moe_selfcheck" -C Debug --output-on-failure
```

Expected: PASS, bit-exact (cache residency never affects selected weights' values). Note the timing delta in the log.

- [ ] **Step 3: Commit**

```bash
git add libs/lsxcommon/src/qwen_3_5_moe_model.cpp
git commit -m "perf(qwen-moe): raise default expert LRU slots 64 -> 512"
```

---

### Task 8: Single-pass Qwen expert store load (Q5)

`LoadQwenMoeExpertStore` reads and Arrow-parses every expert file twice (`qwen_moe_inference_load.cpp:531-589`): pass 1 for sizes, pass 2 for payloads. Stage payloads during pass 1.

**Files:**
- Modify: `libs/lsxcommon/src/qwen_moe_inference_load.cpp`
- Test: existing `test_qwen_moe_loader` roundtrip (behavior-preserving refactor).

First verify `PackedExpert` field types in `libs/lsxcommon/include/lsxcommon/schema.h` (`w1_packed` etc. are expected to be `std::vector<uint8_t>`; adjust the staging struct if they are views).

- [ ] **Step 1: Stage payloads in the first loop**

Replace the section building records + slab (lines 527-589) with:

```cpp
        // Build all records, staging payloads once (single disk pass), then
        // one pinned slab per layer.
        struct StagedExpert {
            std::vector<uint8_t> w1, w2, w3, s1, s2, s3;
        };
        std::vector<StagedExpert> staged(static_cast<size_t>(E));
        auto &recs = store.records[static_cast<size_t>(l)];
        recs.assign(static_cast<size_t>(E), {});
        size_t total = 0;
        for (int e = 0; e < E; ++e) {
            const std::string path =
                (layer_dir / absl::StrCat("exp_", e, ".arrow")).string();
            auto batch_res = ReadRecordBatchFromFile(path);
            if (!batch_res.ok()) {
                error = absl::StrCat("read failed: ", path);
                return false;
            }
            auto pe_res = ExpertSchema::UnpackBatch(*batch_res);
            if (!pe_res.ok()) {
                error = absl::StrCat("unpack failed: ", path);
                return false;
            }
            auto &pe = *pe_res;
            auto &st = staged[static_cast<size_t>(e)];
            st.w1 = std::move(pe.w1_packed);
            st.w2 = std::move(pe.w2_packed);
            st.w3 = std::move(pe.w3_packed);
            st.s1 = std::move(pe.w1_scales);
            st.s2 = std::move(pe.w2_scales);
            st.s3 = std::move(pe.w3_scales);
            auto &r = recs[static_cast<size_t>(e)];
            r.w1_off = total; r.w1_bytes = static_cast<int64_t>(st.w1.size());
            total += r.w1_bytes;
            r.w2_off = total; r.w2_bytes = static_cast<int64_t>(st.w2.size());
            total += r.w2_bytes;
            r.w3_off = total; r.w3_bytes = static_cast<int64_t>(st.w3.size());
            total += r.w3_bytes;
            r.gs1 = pe.w1_global_scale;
            r.gs2 = pe.w2_global_scale;
            r.gs3 = pe.w3_global_scale;
            total = AlignUp(total, 16);
            r.s1_off = total; r.s1_bytes = static_cast<int64_t>(st.s1.size()); total += r.s1_bytes;
            r.s2_off = total; r.s2_bytes = static_cast<int64_t>(st.s2.size()); total += r.s2_bytes;
            r.s3_off = total; r.s3_bytes = static_cast<int64_t>(st.s3.size()); total += r.s3_bytes;
        }
        store.slab_bytes[static_cast<size_t>(l)] = total;

        uint8_t *slab = nullptr;
        if (cudaMallocHost(reinterpret_cast<void **>(&slab), total) !=
            cudaSuccess) {
            error = absl::StrCat("pinned slab alloc failed (layer ", l, ")");
            return false;
        }
        store.h_slabs[static_cast<size_t>(l)] = slab;

        for (int e = 0; e < E; ++e) {
            const auto &st = staged[static_cast<size_t>(e)];
            const auto &r = recs[static_cast<size_t>(e)];
            std::memcpy(slab + r.w1_off, st.w1.data(), r.w1_bytes);
            std::memcpy(slab + r.w2_off, st.w2.data(), r.w2_bytes);
            std::memcpy(slab + r.w3_off, st.w3.data(), r.w3_bytes);
            std::memcpy(slab + r.s1_off, st.s1.data(), r.s1_bytes);
            std::memcpy(slab + r.s2_off, st.s2.data(), r.s2_bytes);
            std::memcpy(slab + r.s3_off, st.s3.data(), r.s3_bytes);
        }
        staged.clear();
        staged.shrink_to_fit();
```

Peak extra RAM per layer equals that layer's slab (~0.43 GB avg) — within host resource policy.

- [ ] **Step 2: Run loader roundtrip + selfcheck**

```powershell
& $cmake --build build --config Debug --target test_qwen_moe_loader test_qwen_moe_selfcheck
& $ctest --preset windows-debug -R "qwen_moe" -C Debug --output-on-failure
```

Expected: PASS; Load phase roughly halves its expert-store I/O.

- [ ] **Step 3: Commit**

```bash
git add libs/lsxcommon/src/qwen_moe_inference_load.cpp
git commit -m "perf(qwen-moe): single-pass expert store load stages payloads once"
```

---

### Task 9: Phase timers (observability)

**Files:**
- Modify: `libs/lsxcommon/src/qwen_3_5_moe_model.cpp` (Load)
- Modify: `libs/lsxcommon/src/gemma_base_model.cpp` (Load)

- [ ] **Step 1: Time the Qwen load phases**

Add a local helper near the top of `qwen_3_5_moe_model.cpp` (below the includes, plus `#include <chrono>`):

```cpp
namespace {
double MsSince(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0)
        .count();
}
} // namespace
```

Wrap each phase in `Load()` and log:

```cpp
    const auto t0 = std::chrono::steady_clock::now();
    ... dense load ...
    log::info(absl::StrCat("[qwen-moe-model] dense load ", MsSince(t0), " ms"));
    const auto t1 = std::chrono::steady_clock::now();
    ... expert store load ...
    log::info(absl::StrCat("[qwen-moe-model] expert store ",
                           MsSince(t1), " ms"));
    const auto t2 = std::chrono::steady_clock::now();
    ... cache alloc ...
    log::info(absl::StrCat("[qwen-moe-model] cache alloc ", MsSince(t2), " ms"));
```

- [ ] **Step 2: Time the Gemma load phases**

In `GemmaBaseModel::Load()` (`gemma_base_model.cpp`), accumulate two chrono intervals — around `LoadGemmaPinnedWeights` (`pinned_ms`) and around the new index load block from Task 2 (`index_ms`) — then log once before `return true;`:

```cpp
    spdlog::info("[gemma-model] pinned load {:.1f} ms, expert index {:.1f} ms",
                 pinned_ms, index_ms);
```

- [ ] **Step 3: Verify logs appear**

Rebuild + run either selfcheck test; confirm phase timings in output.

- [ ] **Step 4: Commit**

```bash
git add libs/lsxcommon/src/qwen_3_5_moe_model.cpp libs/lsxcommon/src/gemma_base_model.cpp
git commit -m "feat: log load-phase timings for qwen-moe and gemma models"
```

---

### Task 10: Final verification + wall-time comparison

- [ ] **Step 1: Full clean-ish rebuild of affected targets**

```powershell
& $cmake --build build --config Debug --target lsxcommon lsxcommon_qwen_moe_cu test_qwen_moe_kernels test_qwen_moe_convert test_qwen_moe_loader test_qwen_moe_model test_qwen_moe_selfcheck test_gemma4_convert test_gemma4_loader test_gemma4_model test_gemma4_attn test_gemma4_norm test_gemma4_rotary test_gemma4_moe test_gemma4_selfcheck
```

- [ ] **Step 2: Full Windows-allowed test sweep**

```powershell
& $ctest --preset windows-debug -R "qwen|gemma" -C Debug --output-on-failure
```

Expected: zero failures; selfcheck durations recorded vs Task 0 baseline (expect order-of-magnitude drop; report actual numbers).

- [ ] **Step 3: Commit any residual changes and summarize**

```bash
git status
git add -A
git commit -m "perf: selfcheck speedup results qwen=<X>s (<baseline>) gemma=<Y>s (<baseline>)"
```

If any regression appears, stop, bisect the task commits (`git bisect` or manual revert), and fix before proceeding.
