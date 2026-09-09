# Model Independence Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. (Subagent dispatch is FORBIDDEN by AGENTS.md — main session only.) Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce model independence and per-call fast-math policy across all 10 registered models; remove qwen-shim coupling in glm-5.3-flash and the KNOWN-BROKEN fused-decode dead code.

**Architecture:** Incremental TDD on branch `audit/model-independence`: every task first extends the guard suite (`tests/test_model_independence_guard.cpp`), sees it red, then makes the minimal source edit, sees it green, commits. Numeric GEMM behavior is unchanged (qwen shims are inline 1:1 forwarders over `lsx_nvfp4`).

**Tech Stack:** C++20, CUDA 13.3 (Blackwell sm_120/121), Catch2 v3, CMake presets (`linux-release-znver4-aocc` build dir already configured).

## Global Constraints

- NO subagents (AGENTS.md) — main session only.
- Branch: `audit/model-independence` (already created; spec committed as 31b26b0).
- Commit messages: English, prefixes `feat:`/`fix:`/`refactor:`/`test:`/`docs:`; each task = one commit (bisectable).
- Build/verify commands (from repo root):
  - Reconfigure after CMake changes: `cmake -S . -B build/linux-release-znver4-aocc -DCMAKE_POLICY_VERSION_MINIMUM=3.5` (existing cache supplies the rest; if the cache is stale, full flags from AGENTS.md AOCC section).
  - Build: `cmake --build build/linux-release-znver4-aocc -j 160`
  - Guard tests: `ctest --test-dir build/linux-release-znver4-aocc -R "independence_guard|absl_format|code_naming" --output-on-failure`
  - Full host suite: `ctest --test-dir build/linux-release-znver4-aocc --output-on-failure -E "selfcheck"`
  - Selfchecks: `ctest --test-dir build/linux-release-znver4-aocc -R "selfcheck" --output-on-failure` (glm-5.2 SKIPs — no weights in vault; expected).
- Do not change numeric paths: `qwen::RunQwenNvfp4Gemm` shims are inline forwarders to `lsxcommon::lsx_nvfp4::RunLsxNvfp4Gemm` (`libs/lsxcommon/include/lsxcommon/qwen_inference.h:103-118`) — replacing call sites is behavior-identical.
- Shared types stay shared: `qwen::QwenGenerationResult` (qwen_inference.h:276) and `qwen_moe::QwenMoeRoutedSlot` are legitimately consumed by glm-5.3-flash — do NOT rename or move them in this plan.
- glm-5.2/gemma keep family-own numeric paths (cuBLAS / gemma nvfp4gemv); the guard tracks them via a greppable policy marker, no numeric change.

---

### Task 1: Guard test — consumer list + no-qwen-shims rule (RED)

**Files:**
- Modify: `tests/test_model_independence_guard.cpp:213-236` (consumers TEST_CASE) and append new TEST_CASE at end of file.

**Interfaces:**
- Consumes: existing TEST_CASE "every consumer model asserts its own fast-math policy per Infer" with `consumers` vector of `const char *`.
- Produces: guard failure messages used by Tasks 2-4 to verify red state.

- [ ] **Step 1: Edit the consumers list** in TEST_CASE "every consumer model asserts its own fast-math policy per Infer":

```cpp
    const std::vector<const char *> consumers = {
        "qwen_base_model.cpp",      // qwen dense (3.5-9B / 3.8-27B)
        "qwen_moe_model.cpp",       // qwen3.5-35b-a3b
        "qwen_moe_36_model.cpp",    // qwen3.6-35b-a3b
        "qwen_moe_exp_model.cpp",   // qwen3.8-flash-next
        "glm_moe_flash_model.cpp",  // glm-5.3-flash
        "kimi_moe_model.cpp",       // kimi-k2.7-code (TC batch, 100 t/s round)
        "deepseek_moe_model.cpp",   // deepseek-v4-flash (family-scoped toggle)
        "glm_moe_model.cpp",        // glm-5.2 (family-own numeric path)
        "gemma_base_model.cpp",     // gemma-4-26b-a4b-it (family-own path)
    };
```

- [ ] **Step 2: Add two new TEST_CASEs at the end of the file** (after the consumers TEST_CASE, before final newline):

```cpp
TEST_CASE("glm-5.3-flash calls the neutral NVFP4 layer directly",
          "[independence]")
{
    // The qwen-named GEMM shims (RunQwenNvfp4Gemm*) are qwen-family call-site
    // aliases over lsx_nvfp4. A glm model must not depend on them: flash
    // kernels call lsxcommon::lsx_nvfp4::RunLsxNvfp4Gemm directly. The
    // shared slot/result TYPES (QwenMoeRoutedSlot, QwenGenerationResult)
    // remain legitimate cross-model data carriers.
    for (const char *f : {"glm_moe_flash_inference.cu",
                          "glm_moe_flash_model.cpp"}) {
        const std::string text = ReadFile(kSrcDir / f);
        INFO(f << " must not use qwen::RunQwenNvfp4Gemm shims");
        REQUIRE(text.find("qwen::RunQwenNvfp4Gemm") == std::string::npos);
    }
}

TEST_CASE("family-own models document their fast-math policy",
          "[independence]")
{
    // glm-5.2 and gemma do not route through the neutral NVFP4 layer (no
    // process-global toggle exists for them); each model file must carry the
    // greppable policy marker so the audit trail stays structural.
    for (const char *f : {"glm_moe_model.cpp", "gemma_base_model.cpp"}) {
        const std::string text = ReadFile(kSrcDir / f);
        INFO(f << " must declare: fast-math policy: family-own");
        REQUIRE(text.find(
                    "fast-math policy: family-own (no process-global toggles)") !=
                std::string::npos);
    }
}
```

NOTE: the loop body must be exactly:

```cpp
    for (const char *f : {"glm_moe_flash_inference.cu",
                          "glm_moe_flash_model.cpp"}) {
        const std::string text = ReadFile(kSrcDir / f);
        INFO(f << " must not use qwen::RunQwenNvfp4Gemm shims");
        REQUIRE(text.find("qwen::RunQwenNvfp4Gemm") == std::string::npos);
    }
```

- [ ] **Step 3: Build and run the guard test — verify RED**

Run: `cmake --build build/linux-release-znver4-aocc -j 160 --target test_model_independence_guard && ctest --test-dir build/linux-release-znver4-aocc -R independence_guard --output-on-failure`
Expected: FAIL — 3 failures: deepseek (no FastMathScope string), glm_moe_model/gemma_base (no marker), flash files (qwen::RunQwenNvfp4Gemm present).

- [ ] **Step 4: Commit (red state, test-only)**

```bash
git add tests/test_model_independence_guard.cpp
git commit -m "test: guard consumers + no-qwen-shims + family-own policy rules (red)"
```

---

### Task 2: glm-5.3-flash — qwen shims → lsx_nvfp4 (GREEN part 1)

**Files:**
- Modify: `libs/lsxcommon/src/glm_moe_flash_inference.cu` (17 call sites: lines 545, 549, 553, 673, 910, 917, 922, 945, 1289, 1292, 1302, 1320, 1324, 1335, 1760, 1767, 1781)
- Modify: `libs/lsxcommon/src/glm_moe_flash_model.cpp:79-80` (ctor toggles)

**Interfaces:**
- Consumes: `lsxcommon::lsx_nvfp4::RunLsxNvfp4Gemm` — signature identical to the shim (stream, x, packed, scales, global_scale, y, seq, K, N, acc=false, seg_wgs_dev=nullptr, seg_bounds_dev=nullptr); `SetLsxTcFastMath(bool)`, `SetLsxSeq1Gemv(bool)`. Header `lsxcommon/lsx_inference_nvfp4.h` is transitively reachable from the .cu TU (already included by qwen_inference.h chain; add explicit include).
- Produces: flash files free of `qwen::RunQwenNvfp4Gemm` strings.

- [ ] **Step 1: Add explicit include** after line 12 include block in `glm_moe_flash_inference.cu`:

```cpp
#include "lsxcommon/lsx_inference_nvfp4.h"
```

- [ ] **Step 2: Replace all shim call sites.** Mechanical substitution in `glm_moe_flash_inference.cu`:

```
qwen::RunQwenNvfp4Gemm(  →  lsxcommon::lsx_nvfp4::RunLsxNvfp4Gemm(
```

(All 17 occurrences; arguments unchanged — the shim forwarded them 1:1. The file is inside `namespace glm_moe_flash`, so fully qualify `lsxcommon::lsx_nvfp4::` exactly as kimi does at `kimi_moe_inference.cu:615`.)

- [ ] **Step 3: Replace ctor toggles** in `glm_moe_flash_model.cpp:79-80`:

```cpp
    // Both toggles are process-wide (see Qwen35MoeModel): without them every
    // NVFP4 GEMM runs the shared-memory tile kernel even at seq==1 decode.
    lsxcommon::lsx_nvfp4::SetLsxTcFastMath(cfg_.fast_math);
    lsxcommon::lsx_nvfp4::SetLsxSeq1Gemv(cfg_.fast_math);
```

(`lsx_inference_nvfp4.h` reaches this TU via `glm_moe_flash_model.h` → keep `qwen_inference.h` include — `qwen::QwenGenerationResult` is still used at line 152.)

- [ ] **Step 4: Build + run guard tests — verify shims test GREEN, policy tests still RED**

Run: `cmake --build build/linux-release-znver4-aocc -j 160 --target test_model_independence_guard lsxcommon && ctest --test-dir build/linux-release-znver4-aocc -R independence_guard --output-on-failure`
Expected: flash-shim TEST_CASE passes; deepseek/family-own TEST_CASEs still fail.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/glm_moe_flash_inference.cu libs/lsxcommon/src/glm_moe_flash_model.cpp
git commit -m "refactor: glm-5.3-flash calls lsx_nvfp4 directly (drops qwen GEMM shims)"
```

---

### Task 3: deepseek-v4-flash — per-call fast-math scope (GREEN part 2)

**Files:**
- Modify: `libs/lsxcommon/src/deepseek_moe_load.cuh:20-33` (add RAII scope next to flag accessors)
- Modify: `libs/lsxcommon/src/deepseek_moe_model.cpp:30-34` (remove ctor set), `:89-96` (Infer scope), `:105-108` (SelfCheck scope)

**Interfaces:**
- Consumes: `SetDeepseekFastMath(bool)` / `DeepseekFastMath()` (existing, deepseek_moe_inference_moe.cu:25-33).
- Produces: `DeepseekFastMathScope` — copy-deleted RAII; ctor takes `bool fast_math`, saves prev, stores new; dtor restores prev. Guard string "FastMathScope" + "cfg_.fast_math" present in deepseek_moe_model.cpp.

- [ ] **Step 1: Add the scope in `deepseek_moe_load.cuh`** directly after `bool DeepseekFastMath();` (replace lines 20-33 comment block accordingly):

```cpp
// Blackwell TC W4A4 toggle (F-12). Process-wide; every Infer/SelfCheck call
// asserts its OWN policy via DeepseekFastMathScope — the last-constructed
// model's config must never leak into another model's kernel routing
// (LsxFastMathScope parity, lsx_inference_nvfp4.h).
void SetDeepseekFastMath(bool on);
bool DeepseekFastMath();

class DeepseekFastMathScope {
public:
    explicit DeepseekFastMathScope(bool on)
        : prev_(DeepseekFastMath()) { SetDeepseekFastMath(on); }
    ~DeepseekFastMathScope() { SetDeepseekFastMath(prev_); }
    DeepseekFastMathScope(const DeepseekFastMathScope &) = delete;
    DeepseekFastMathScope &operator=(const DeepseekFastMathScope &) = delete;

private:
    bool prev_;
};
```

(Delete the old comment "set from ModelInitConfig::fast_math at model construction; default OFF preserves bit-exact legacy routing." — replaced by the per-call contract above. Keep the flag default OFF.)

- [ ] **Step 2: Remove the constructor set** in `deepseek_moe_model.cpp:30-34`. The ctor becomes:

```cpp
DeepSeekV4FlashModel::DeepSeekV4FlashModel(ModelInitConfig cfg)
    : ModelBase(std::move(cfg))
{
    type_ = kType;
}
```

(Inspect the current ctor body first: delete ONLY the `gpu::SetDeepseekFastMath(cfg_.fast_math);` line and any comment pinned to it; keep everything else.)

- [ ] **Step 3: Open the scope in `Infer`** (`deepseek_moe_model.cpp:89-96`):

```cpp
gpu::InferenceResult DeepSeekV4FlashModel::Infer(const ModelRequest &req)
{
    // Model independence: assert THIS model's fast-math policy for the
    // duration of the call (constructor-time flags would leak the
    // last-constructed model's config across the registry).
    const gpu::DeepseekFastMathScope fm_scope(cfg_.fast_math);
    gpu::InferenceResult r;
    if (!ValidateInferRequest(req, r)) return r;
    if (!loaded_) { r.error_msg = "model not loaded"; return r; }
    return RunUnderResidencyLock([&] {
        return gpu::RunInferenceModel(cfg_.model_path, req.input_ids,
                                      req.max_gen_tokens, *tok_db_, pw_,
                                      lsxcommon::sample::DefaultSampler(),
                                      stream(), cublas());
    });
}
```

- [ ] **Step 4: Open the scope in `Benchmark`** the same way (the TC toggle routes the moe path during benchmarks too — mirror QwenMoeModelBase::Benchmark). Insert as the first statement after `RequireLoaded`:

```cpp
    const gpu::DeepseekFastMathScope fm_scope(cfg_.fast_math);
```

(SelfCheck delegates to `runtime::RunParisSelfCheck(*this, req)` → re-enters `Infer` → scope applies; no extra line needed there.)

- [ ] **Step 5: Build + guard tests — verify deepseek GREEN, family-own still RED**

Run: `cmake --build build/linux-release-znver4-aocc -j 160 --target test_model_independence_guard lsxcommon && ctest --test-dir build/linux-release-znver4-aocc -R independence_guard --output-on-failure`
Expected: only the family-own marker TEST_CASE fails.

- [ ] **Step 6: Commit**

```bash
git add libs/lsxcommon/src/deepseek_moe_load.cuh libs/lsxcommon/src/deepseek_moe_model.cpp
git commit -m "fix: deepseek fast-math policy is per-call (DeepseekFastMathScope in Infer)"
```

---

### Task 4: glm-5.2 / gemma policy markers (GREEN part 3, completes Task 1)

**Files:**
- Modify: `libs/lsxcommon/src/glm_moe_model.cpp` (marker comment near top, inside namespace or at file scope)
- Modify: `libs/lsxcommon/src/gemma_base_model.cpp` (marker comment near top)

- [ ] **Step 1: Add marker to `glm_moe_model.cpp`** directly above `const std::string GlmV5Model::kType = "glm-5.2";`:

```cpp
// fast-math policy: family-own (no process-global toggles). GLM-5.2 kernels
// run the bit-exact cuBLAS path unconditionally; no TC/GEMV routing state.
```

- [ ] **Step 2: Add marker to `gemma_base_model.cpp`** directly above the first function definition (before `bool GemmaBaseModel::ValidateGeometry`):

```cpp
// fast-math policy: family-own (no process-global toggles). gemma runs its
// own nvfp4gemv path (gemma_moe_inference.cu) unconditionally.
```

- [ ] **Step 3: Build + run ALL guard tests — verify GREEN**

Run: `cmake --build build/linux-release-znver4-aocc -j 160 --target test_model_independence_guard lsxcommon && ctest --test-dir build/linux-release-znver4-aocc -R independence_guard --output-on-failure`
Expected: PASS (all TEST_CASEs).

- [ ] **Step 4: Run sibling structural guards (naming, absl)**

Run: `ctest --test-dir build/linux-release-znver4-aocc -R "code_naming|absl_format" --output-on-failure`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/glm_moe_model.cpp libs/lsxcommon/src/gemma_base_model.cpp
git commit -m "test: glm-5.2/gemma family-own fast-math policy markers (guard green)"
```

---

### Task 5: Dead code removal — fused decode attention (RED → GREEN)

**Files:**
- Modify: `tests/test_model_independence_guard.cpp` (absence test — RED first)
- Modify: `libs/lsxcommon/include/lsxcommon/qwen_inference.h:201-243` (delete declarations)
- Modify: `libs/lsxcommon/src/qwen_inference.cu:796-933, 1484-1650` (delete kernels + wrappers)
- Delete: `tests/test_qwen_dense_fused_decode_attn.cu`
- Modify: `tests/CMakeLists.txt:168, 278` (remove target references)

**Interfaces:**
- Consumes: guard ReadFile helper.
- Produces: `RunQwenFusedDecodeAttnKernel`/`RunQwenFusedQkv`/`RunQwenFusedAttn`/`RunQwenFusedOProj` gone from header and TU; nothing else referenced them (verified: only test_qwen_dense_fused_decode_attn.cu + internal call sites).

- [ ] **Step 1: Add absence TEST_CASE** to `tests/test_model_independence_guard.cpp` (end of file):

```cpp
TEST_CASE("broken fused-decode attention path is removed", "[independence]")
{
    // KNOWN-BROKEN (diverged on the real Qwen3.5-9B checkpoint; unwired from
    // generation in 19e8d4f) and test-only since — deleted by the 2026-09
    // model-independence audit. Reintroduction requires a full-layer parity
    // re-validation on the real checkpoint (see AGENTS.md selfcheck rules).
    const std::string header = ReadFile(kIncDir / "qwen_inference.h");
    REQUIRE(header.find("RunQwenFusedDecodeAttnKernel") == std::string::npos);
    const std::string tu = ReadFile(kSrcDir / "qwen_inference.cu");
    REQUIRE(tu.find("RunQwenFusedDecodeAttnKernel") == std::string::npos);
    REQUIRE(tu.find("RunQwenFusedQkv") == std::string::npos);
    REQUIRE(tu.find("RunQwenFusedOProj") == std::string::npos);
}
```

- [ ] **Step 2: Verify RED**

Run: `ctest --test-dir build/linux-release-znver4-aocc -R independence_guard --output-on-failure`
Expected: FAIL ("broken fused-decode attention path is removed").

- [ ] **Step 3: Commit the red test**

```bash
git add tests/test_model_independence_guard.cpp
git commit -m "test: guard asserts removal of broken fused-decode attention path (red)"
```

- [ ] **Step 4: Delete declarations from `qwen_inference.h`** — remove the four blocks (lines 201-243): the fused-decoder comment + `RunQwenFusedDecodeAttnKernel`, `RunQwenFusedQkv`, `RunQwenFusedAttn`, `RunQwenFusedOProj` declarations (keep the "// Dense SwiGLU MLP" block that follows).

- [ ] **Step 5: Delete implementations from `qwen_inference.cu`:**
  - kernels `fused_qkv_kernel` (~line 796), `fused_attn_kernel` (~885), `fused_o_proj_kernel` (~933) with their doc comments;
  - wrappers `RunQwenFusedQkv` (~1490), `RunQwenFusedAttn` (~1540), `RunQwenFusedOProj` (~1554), `RunQwenFusedDecodeAttnKernel` (~1587-~1650).
  Verify nothing else in the TU references them: `rg -n "RunQwenFused|fused_qkv_kernel|fused_attn_kernel|fused_o_proj_kernel" libs/lsxcommon/src/qwen_inference.cu` → no matches (if `RunQwenFusedAttn` has callers besides the decode wrapper, STOP and re-scope).

- [ ] **Step 6: Delete test + CMake references:**
  - `git rm tests/test_qwen_dense_fused_decode_attn.cu`
  - `tests/CMakeLists.txt:168` — delete the `test_qwen_dense_fused_decode_attn.cu` source line;
  - `tests/CMakeLists.txt:278` — remove `test_qwen_dense_fused_decode_attn` from the foreach list.

- [ ] **Step 7: Reconfigure + full build**

Run: `cmake -S . -B build/linux-release-znver4-aocc -DCMAKE_POLICY_VERSION_MINIMUM=3.5 && cmake --build build/linux-release-znver4-aocc -j 160`
Expected: green, no undefined references.

- [ ] **Step 8: Guard tests GREEN + host suite green**

Run: `ctest --test-dir build/linux-release-znver4-aocc -R "independence_guard|code_naming|absl_format" --output-on-failure && ctest --test-dir build/linux-release-znver4-aocc --output-on-failure -E "selfcheck"`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "refactor: remove KNOWN-BROKEN fused decode attention path (qwen dense)"
```

---

### Task 6: Full verification — all selfchecks

**Files:** none (verification only).

- [ ] **Step 1: Run the complete selfcheck suite** (weights vault `/mnt/llm_models/models` — deepseek, glm-5.3-flash, gemma, kimi, qwen 3.5/3.6/3.8 present; glm-5.2 SKIPs by design):

Run: `ctest --test-dir build/linux-release-znver4-aocc -R "selfcheck" --output-on-failure`
Expected: PASS or SKIP (glm-5.2); no FAIL. Resource rule: selfchecks are inference — run sequentially, nothing else heavy in parallel; each test frees the model on exit.

- [ ] **Step 2: Run the full host suite once more**

Run: `ctest --test-dir build/linux-release-znver4-aocc --output-on-failure`
Expected: 0 FAIL.

- [ ] **Step 3: Final audit summary** — post to the operator: list of commits, guard-test inventory (models × policy), remaining out-of-scope items (RunExpQsaAttention reuse in flash, gemma nvfp4gemv neutralization, glm-5.2 weights absent → selfcheck unverified).
