# TaskMosaic OpenAI Bridge — Plan 2: All-Family Streaming

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline; AGENTS.md forbids subagents). Steps use checkbox (`- [ ]`).
>
> **Depends on:** Plan 1 (merged on `feat/taskmosaic-openai-bridge`).

**Goal:** Make every model family emit `ModelRequest::token_callback` events *as each token is decoded* (true TTFT) and honor `abort_flag` between decode steps, so the Plan-1 bridge streams incrementally for all models, not just gigachat/nemotron.

**Architecture:** Introduce a CUDA-safe `lsxcommon::GenerationStream` handle (`{const TokenCallback* on_token; const std::atomic<bool>* abort_flag;}`) in a new header. Thread an optional `const GenerationStream* gs` through each family's `RunXGeneration`, and insert a 4-line emit+abort probe at every existing host token-append site (`result.tokens.push_back(tok)`). Each model's `Infer` constructs the handle from `req` and passes it. **No math change** — only callback invocation + a break.

**Tech Stack:** C++20, CUDA `.cu`, catch2, existing per-family generation drivers.

## Global Constraints

- Numerics must not change: only add `gs->Emit(...)` after the token is already known and an abort check. Existing `test_*_selfcheck`, `test_*_kv_parity`, `--bit-exact`, FSI/CODE/DET gates must stay green.
- `.cu`/`.cuh` headers must not include `<absl/container/*>`. `GenerationStream` uses only `<atomic>/<cstdint>/<functional>`.
- `TokenChunk`/`TokenCallback` move to `token_stream.h`; `model.h` includes it. Keep the public names/signatures intact so Plan 1 and `model_engine.cpp` keep compiling.
- Append the new parameter with a default (`= nullptr`) so direct `Run*` test callers remain source-compatible.
- Logging via spdlog only. Commit prefixes as before; do not touch CMake unless a new source/header is listed (header-only, so no CMake change).
- Every new/changed test target keeps its label (`inference`/`fast`); no new family-prefixed stems (guard `test_code_naming_guard`).

## File Structure

**Create**
- `libs/lsxcommon/include/lsxcommon/token_stream.h` (CUDA-safe `TokenChunk`/`TokenCallback`/`GenerationStream`)
- `tests/test_generation_stream.cpp` (pure-host unit test of the handle)
- `tests/test_streaming_order_guard.cu` (optional; per-family order assertions live in existing model tests — see Task 12)

**Modify (header signature + `.cu` loop + model `.cpp` call)**
- `libs/lsxcommon/include/lsxcommon/model.h` (include `token_stream.h`, drop local definitions)
- `qwen_inference`, `qwen_moe_inference`, `qwen_moe_exp_inference`, `ornith_moe_inference`,
  `glm_moe_inference`, `glm_moe_flash_inference`, `gemma_moe_inference`,
  `granite_dense_inference`, `kimi_moe_kernels`, deepseek pipeline
- Model `.cpp`: `qwen_base_model`, `qwen_moe_model`, `qwen_moe_exp_model`,
  `ornith_moe_model`, `deepseek_moe_model`, `deepseek_moe_v41_model`,
  `glm_moe_model`, `glm_moe_flash_model`, `gemma_base_model`,
  `granite_dense_model`, `kimi_moe_model`
- `tests/CMakeLists.txt` (register `test_generation_stream`, label `fast`)

---

### Task 1: `GenerationStream` handle + `model.h` refactor

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/token_stream.h`
- Modify: `libs/lsxcommon/include/lsxcommon/model.h`
- Test: `tests/test_generation_stream.cpp`

**Interfaces:**
- Produces: `lsxcommon::TokenChunk`, `lsxcommon::TokenCallback`,
  `lsxcommon::GenerationStream{ const TokenCallback* on_token; const std::atomic<bool>* abort_flag; bool Aborted() const; void Emit(uint32_t token_id, uint32_t step) const; }`.

- [ ] **Step 1: Write the failing test**

```cpp
#include <catch2/catch_test_macros.hpp>

#include <lsxcommon/token_stream.h>

#include <atomic>
#include <cstdint>
#include <vector>

TEST_CASE("generation stream: emits chunks in order", "[token_stream][fast]") {
    std::vector<std::uint32_t> seen;
    lsxcommon::TokenCallback cb = [&](const lsxcommon::TokenChunk& c) {
        seen.push_back(c.token_id);
        CHECK(c.step_t == seen.size() - 1);
    };
    lsxcommon::GenerationStream gs;
    gs.on_token = &cb;
    gs.Emit(7u, 0u);
    gs.Emit(8u, 1u);
    REQUIRE(seen.size() == 2u);
    CHECK(seen[0] == 7u);
    CHECK(seen[1] == 8u);
}

TEST_CASE("generation stream: Aborted reflects the flag", "[token_stream][fast]") {
    std::atomic<bool> abort{false};
    lsxcommon::GenerationStream gs;
    gs.abort_flag = &abort;
    CHECK_FALSE(gs.Aborted());
    abort.store(true);
    CHECK(gs.Aborted());
}
```

- [ ] **Step 2: Run to fail**

Register the target (Step 5), then: `cmake --build --preset linux-debug --target test_generation_stream`
Expected: FAIL (no `token_stream.h`).

- [ ] **Step 3: Write `token_stream.h`**

```cpp
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>

namespace lsxcommon {

// Flat, host-only streaming token packet. Transmitted by const reference so no
// allocation or copy occurs on the model->caller hot path.
struct TokenChunk {
    std::uint32_t token_id = 0;
    std::uint32_t step_t = 0;
    float logit_maxv = 0.0f;
    bool is_special = false;
};

using TokenCallback = std::function<void(const TokenChunk&)>;

// CUDA-safe streaming handle threaded into every generation loop. Members are
// non-owning pointers; both may be null (streaming disabled).
struct GenerationStream {
    const TokenCallback* on_token = nullptr;
    const std::atomic<bool>* abort_flag = nullptr;

    bool Aborted() const {
        return abort_flag != nullptr &&
               abort_flag->load(std::memory_order_acquire);
    }

    void Emit(std::uint32_t token_id, std::uint32_t step) const {
        if (on_token == nullptr || !*on_token) return;
        TokenChunk c;
        c.token_id = token_id;
        c.step_t = step;
        (*on_token)(c);
    }
};

}  // namespace lsxcommon
```

- [ ] **Step 4: Refactor `model.h`**

Replace the local `TokenChunk` struct and `TokenCallback` alias with
`#include "lsxcommon/token_stream.h"` (keep the `ModelRequest` member names
`token_callback` and `abort_flag` unchanged).

- [ ] **Step 5: Register the test and run**

`tests/CMakeLists.txt`:

```cmake
add_executable(test_generation_stream test_generation_stream.cpp)
target_link_libraries(test_generation_stream PRIVATE lsxcommon Catch2::Catch2WithMain)
add_test(NAME test_generation_stream COMMAND test_generation_stream)
```

Append `test_generation_stream` to the `LABELS fast` list.

Run: `cmake --preset linux-debug && cmake --build --preset linux-debug --target test_generation_stream && ctest --preset linux-debug -R test_generation_stream --output-on-failure`
Expected: PASS. Also rebuild `test_model_engine` to confirm the `model.h` refactor is source-compatible.

- [ ] **Step 6: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/token_stream.h libs/lsxcommon/include/lsxcommon/model.h tests/test_generation_stream.cpp tests/CMakeLists.txt
git commit -m "feat: GenerationStream handle for per-token model streaming"
```

---

### Task 2: Qwen dense (`RunQwenGeneration`)

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/qwen_inference.h` (add `gs`)
- Modify: `libs/lsxcommon/src/qwen_inference.cu` (emit at both append sites)
- Modify: `libs/lsxcommon/src/qwen_base_model.cpp` (pass `&gs`)

**Interfaces:**
- Consumes: `GenerationStream` (Task 1).
- Produces: `RunQwenGeneration(..., QwenKvCache* kv, const lsxcommon::GenerationStream* gs = nullptr)`.

- [ ] **Step 1: Extend the signature** in `qwen_inference.h`

```cpp
cudaError_t RunQwenGeneration(
    cudaStream_t stream, const QwenPinnedWeights &pw,
    const std::vector<int32_t> &prompt, int max_new, int eos_id,
    QwenGenerationResult &result,
    lsxcommon::sample::ISampler &sampler = lsxcommon::sample::DefaultSampler(),
    QwenKvCache *kv = nullptr,
    const lsxcommon::GenerationStream *gs = nullptr);
```

- [ ] **Step 2: Add the hook at both append sites** in `qwen_inference.cu`

At the full-recompute site (`result.tokens.push_back(tok);` ~line 2063) and the
incremental site (~line 2201), insert immediately after the `push_back`:

```cpp
            if (gs) gs->Emit(static_cast<std::uint32_t>(tok),
                             static_cast<std::uint32_t>(result.tokens.size() - 1));
```

and immediately after the existing `if (tok == eos_id) { ... break; }` block:

```cpp
            if (gs && gs->Aborted()) {
                result.eos_terminated = false;
                break;
            }
```

- [ ] **Step 3: Pass the handle** in `QwenBaseModel::Infer` (`qwen_base_model.cpp`)

```cpp
    lsxcommon::GenerationStream gs;
    gs.on_token = req.token_callback ? &req.token_callback : nullptr;
    gs.abort_flag = req.abort_flag;
    ...
        return qwen::RunQwenGeneration(st, pw_, gen_input,
                                       req.max_gen_tokens, eos_id, gen,
                                       lsxcommon::sample::DefaultSampler(),
                                       kv_.get(), &gs);
```

- [ ] **Step 4: Verify (no GPU numerics change)**

Build the model test target and re-run its parity/selfcheck:
Run: `cmake --build --preset linux-debug --target test_qwen_dense_model test_qwen_dense_kv_parity test_qwen_dense_selfcheck`
(CUDA tests labeled `inference` require the GPU and vault; run on the GPU host.)
Expected: PASS (unchanged output).

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/qwen_inference.h libs/lsxcommon/src/qwen_inference.cu libs/lsxcommon/src/qwen_base_model.cpp
git commit -m "feat: stream qwen dense tokens incrementally"
```

---

### Task 3: Qwen MoE (`RunQwenMoeGeneration`) — bulk driver, EOS-fence emit

This driver batches the whole generation on-device and drains it in ONE D2H at
the end; the only mid-loop host sync is the every-8-token EOS fence
(`kEosFenceBatch`). Per the accepted hybrid decision, emit at the fence (no new
sync) and flush the remainder at the end.

**Files:** `qwen_moe_inference.{h,cu}`, `qwen_moe_model.cpp`.

- [ ] **Step 1: Signature** — append `const lsxcommon::GenerationStream *gs = nullptr` after `QwenKvCache *kv`.
- [ ] **Step 2: Fence emit** — introduce `int emitted = 0;` before the loop. In the EOS-fence block (after `cudaEventSynchronize(tok_ready)`), before the `hit` scan:
  ```cpp
                if (gs) {
                    for (int b = 0; b < kEosFenceBatch; ++b) {
                        const int32_t t = eos_win[b];
                        if (t == eos_id) break;
                        gs->Emit(static_cast<std::uint32_t>(t),
                                 static_cast<std::uint32_t>(emitted + b));
                    }
                    emitted = generated;   // window fully accounted for
                    if (gs->Aborted()) { abort_stream = true; break; }
                }
  ```
  (`abort_stream` is a local `bool` checked to exit the loop; on abort,
  `result.eos_terminated = false`.)
- [ ] **Step 3: Remainder flush** — after the loop and the bulk drain
  (`result.tokens.assign(hgen...)`), emit any tokens past `emitted`, stopping at
  EOS:
  ```cpp
        if (gs) {
            for (int i = emitted; i < generated; ++i) {
                if (hgen[i] == eos_id) break;
                gs->Emit(static_cast<std::uint32_t>(hgen[i]),
                         static_cast<std::uint32_t>(i));
            }
        }
  ```
- [ ] **Step 4: Call site** — `Qwen35MoeModel::Infer` / `Qwen36MoeModel::Infer` (both `RunQwenMoeGeneration` call sites ~244/288) build `gs` from `req` and pass `&gs`.
- [ ] **Step 5: Verify** — `test_qwen_moe_kernels`, `test_qwen_moe_*` (inference) unchanged.
- [ ] **Step 6: Commit** — `feat: stream qwen MoE tokens at the EOS fence`.

---

### Task 4: Qwen MoE experimental (`RunExpGeneration`) — bulk driver, EOS-fence emit

Same bulk-drain pattern as Task 3 (`qwen_moe_exp_inference.cu`, comment "Bulk
token drain."). If this driver has no periodic fence, add the emit loop at the
single end drain only (it still yields all tokens, just not pre-EOS); if it has
a fence, mirror Task 3.

**Files:** `qwen_moe_exp_inference.{h,cu}`, `qwen_moe_exp_model.cpp`.

- [ ] **Step 1: Signature** — append `const lsxcommon::GenerationStream *gs = nullptr` after `qwen::QwenKvCache *kv`.
- [ ] **Step 2: Emit** — mirror Task 3 Steps 2–3 (`result.tokens.assign(hgen...)` site).
- [ ] **Step 3: Call site** — `Qwen38FlashNextModel::Infer` builds `gs` and passes it.
- [ ] **Step 4: Verify** — `test_qwen_moe_exp_kernels` (inference) unchanged.
- [ ] **Step 5: Commit** — `feat: stream qwen-moe-exp tokens at the EOS fence`.

---

### Task 5: Ornith MoE (`RunOrnithMoeGeneration`) — bulk driver, EOS-fence emit

Mirror of Task 3 (`ornith_moe_inference.cu`, `kEosFenceBatch = 8`).

**Files:** `ornith_moe_inference.{h,cu}`, `ornith_moe_model.cpp`.

- [ ] **Step 1: Signature** — append `const lsxcommon::GenerationStream *gs = nullptr` after `QwenKvCache *kv`.
- [ ] **Step 2: Emit** — mirror Task 3 Steps 2–3.
- [ ] **Step 3: Call sites** — both `OrnithMoeModel::Infer` branches (~246/290) build and pass `&gs`.
- [ ] **Step 4: Verify** — `test_ornith_moe_kernels`, `test_ornith_moe_model` (inference) unchanged.
- [ ] **Step 5: Commit** — `feat: stream ornith MoE tokens at the EOS fence`.

---

### Task 6: DeepSeek V4 / V4.1 (`gpu::RunInferenceModel`)

**Files:** `src/deepseek_moe_model.cpp` (decl + call), `src/deepseek_moe_inference_pipeline.cu` (definition + loop), `src/deepseek_moe_v41_model.cpp`.

- [ ] **Step 1: Signature** — append `const lsxcommon::GenerationStream *gs = nullptr` after `bool ignore_eos` in the `gpu::RunInferenceModel` declaration (`deepseek_moe_model.cpp`) and definition (`deepseek_moe_inference_pipeline.cu`).
- [ ] **Step 2: Hook** — in the pipeline's decode loop, after each generated token is appended to the result, emit and abort:
  ```cpp
      if (gs) gs->Emit(static_cast<std::uint32_t>(tok),
                       static_cast<std::uint32_t>(gen.tokens.size() - 1));
      if (gs && gs->Aborted()) break;
  ```
  (Locate the loop's token-append / `first_token` publication; apply at that single site.)
- [ ] **Step 3: Call sites** — `DeepSeekV4FlashModel::Infer` (deepseek_moe_model.cpp) and `DeepSeekV41FlashModel::Infer` (deepseek_moe_v41_model.cpp) build `gs` from `req` and pass `&gs`.
- [ ] **Step 4: Verify** — `test_deepseek_moe_v41_pipeline`, `test_deepseek_moe_model` (inference) unchanged.
- [ ] **Step 5: Commit** — `feat: stream deepseek tokens incrementally`.

---

### Task 7: GLM (`RunGlmRealGeneration` / `RunGlmGeneration`)

**Files:** `glm_moe_inference.{h,cu}`, `glm_moe_model.cpp`.

- [ ] **Step 1: Signature** — append `const lsxcommon::GenerationStream *gs = nullptr` after `sampler` in `RunGlmRealGeneration` (~line 490) and in both `RunGlmGeneration` overloads (~445/461).
- [ ] **Step 2: Hook** — emit + abort after each token append in `glm_moe_inference.cu` (both the real and legacy drivers if used).
- [ ] **Step 3: Call site** — `GlmV5Model::Infer` (`glm_moe_model.cpp`, `RunGlmRealGeneration` ~151) builds `gs` and passes it.
- [ ] **Step 4: Verify** — `test_glm_moe_model`, `test_glm_selfcheck` (inference) unchanged.
- [ ] **Step 5: Commit** — `feat: stream GLM tokens incrementally`.

---

### Task 8: GLM-5.3-Flash (`RunGlmFlashGeneration`)

**Files:** `glm_moe_flash_inference.{h,cu}`, `glm_moe_flash_model.cpp`.

- [ ] **Step 1: Signature** — append `const lsxcommon::GenerationStream *gs = nullptr` after `sample::ISampler &sampler`.
- [ ] **Step 2: Hook** — emit + abort after each token append in `glm_moe_flash_inference.cu`.
- [ ] **Step 3: Call site** — `Glm5NextFlashModel::Infer` (`glm_moe_flash_model.cpp` ~166) builds `gs` and passes it.
- [ ] **Step 4: Verify** — `test_glm_moe_flash_kernels` (inference) unchanged.
- [ ] **Step 5: Commit** — `feat: stream GLM-Flash tokens incrementally`.

---

### Task 9: Gemma (`RunGemmaGeneration`)

**Files:** `gemma_moe_inference.{h,cu}`, `gemma_base_model.cpp`.

- [ ] **Step 1: Signature** — append `const lsxcommon::GenerationStream *gs = nullptr` after `GemmaGenContext *ctx = nullptr`.
- [ ] **Step 2: Hook** — Gemma's loop may sample internally (no `ISampler`); find the generated-token append and insert emit + abort there.
- [ ] **Step 3: Call site** — `GemmaBaseModel::Infer` (`gemma_base_model.cpp` ~190) passes `&gs`.
- [ ] **Step 4: Verify** — Gemma selfcheck/inference tests unchanged.
- [ ] **Step 5: Commit** — `feat: stream gemma tokens incrementally`.

---

### Task 10: Granite dense (`RunGraniteGeneration`)

**Files:** `granite_dense_inference.{h,cu}`, `granite_dense_model.cpp`.

- [ ] **Step 1: Signature** — append `const lsxcommon::GenerationStream *gs = nullptr` after `GraniteKvCache *kv = nullptr`.
- [ ] **Step 2: Hook** — emit + abort after each `result.tokens.push_back` in `granite_dense_inference.cu`.
- [ ] **Step 3: Call site** — `Granite42_30BModel::Infer` (`granite_dense_model.cpp` ~119) passes `&gs`.
- [ ] **Step 4: Verify** — `test_granite_dense_*` (inference) unchanged.
- [ ] **Step 5: Commit** — `feat: stream granite tokens incrementally`.

---

### Task 11: Kimi (`RunKimiGeneration`) + unify gigachat/nemotron

**Files:** `src/kimi_moe_kernels.h`, `src/kimi_moe_inference.cu`, `kimi_moe_model.cpp`; optionally `gigachat_moe_model.cpp`, `nemotron_moe_model.cpp`.

- [ ] **Step 1: Signature** — append `const lsxcommon::GenerationStream *gs = nullptr` after `bool fast_math = false`.
- [ ] **Step 2: Hook** — emit + abort after each `gen.tokens.push_back` in `kimi_moe_inference.cu`.
- [ ] **Step 3: Call site** — `KimiK27CodeModel::Infer` (`kimi_moe_model.cpp` ~107) passes `&gs`.
- [ ] **Step 4 (optional): unify** — replace the inline `if (req.token_callback) req.token_callback(c);` + abort check in `gigachat_moe_model.cpp`/`nemotron_moe_model.cpp` with `gs->Emit(...)` / `gs->Aborted()` (behavior-identical).
- [ ] **Step 5: Verify** — `test_kimi_moe_*`, `test_gigachat_moe_selfcheck`, `test_nemotron_moe_selfcheck` (inference) unchanged.
- [ ] **Step 6: Commit** — `feat: stream kimi tokens incrementally`.

---

### Task 12: Streaming-order verification (inference) + full regression

**Files:** extend the existing per-family model tests (no new family stems).

- [ ] **Step 1: Add a shared assertion helper** in a host header (e.g. `tests/streaming_assert.h`):

```cpp
#pragma once
#include <lsxcommon/model.h>
#include <vector>

// Runs model.Infer with a recording callback and asserts the callback saw
// exactly the produced gen_tokens, in order.
inline void ExpectStreamMatchesTokens(lsxcommon::IModel& model,
                                      const lsxcommon::ModelRequest& base) {
    lsxcommon::ModelRequest req = base;
    std::vector<std::int32_t> seen;
    req.token_callback = [&](const lsxcommon::TokenChunk& c) {
        seen.push_back(static_cast<std::int32_t>(c.token_id));
    };
    const auto res = model.Infer(req);
    REQUIRE(res.ok);
    REQUIRE(seen.size() == res.gen_tokens.size());
    CHECK(seen == res.gen_tokens);
}
```

- [ ] **Step 2: Call it** from each family's existing inference/model test with the
  same request used by that test (label `inference`, requires the vault + GPU).

- [ ] **Step 3: Regression** — on the GPU host run the model test set:
  `ctest --preset linux-debug -R "<family>" --output-on-failure` for each family;
  confirm selfcheck/parity/bit-exact unchanged.

- [ ] **Step 4: Commit** — `test: assert per-token streaming order across families`.

---

## Self-Review (spec coverage)

| Spec requirement | Task |
|---|---|
| §6.1 thread callback + abort into all family loops | Tasks 2–11 |
| CUDA-safe shared handle | Task 1 |
| No math change / bit-exactness | each Task's verify step + Task 12 |
| Guard/label discipline | Task 1 registration; no new stems |
| gigachat/nemotron already stream | Task 11 Step 4 (optional unify) |

**Placeholder scan:** per-family tasks intentionally say "insert after each
`result.tokens.push_back`" rather than reproducing each `.cu` loop verbatim —
the exact 4-line snippet is given once and is identical at every site; the
signature and call-site code are complete. If a family's loop has no
`push_back` (samples inline), the implementer finds the token publication and
applies the same snippet.

**Type consistency:** `GenerationStream`/`TokenChunk`/`TokenCallback` names are
identical everywhere; `gs->Emit(uint32_t,uint32_t)` and `gs->Aborted()` match
Task 1. `ModelRequest::token_callback`/`abort_flag` names are unchanged.

## Execution Handoff

Execute inline (AGENTS.md forbids subagents) via superpowers:executing-plans.
Tasks 2–11 are independent per family and may be committed in any order; Task 12
aggregates the regression. GPU-dependent verification runs on the GPU host per
the `running-a-model` resource policy.
