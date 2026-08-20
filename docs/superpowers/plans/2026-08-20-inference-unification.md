# Inference Unification & Test Actualization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Zero-behaviour-change refactor unifying duplicated `IModel` plumbing into `ModelBase` (Load preamble, Inspect, ResidentWeightBytes, num_layers) plus actualized tests (fix stale GLM test, add Qwen/DeepSeek deterministic SelfCheck tests).

**Architecture:** Add protected helpers `BeginLoad`/`FinishLoad` and a `resident_bytes_` member to `ModelBase`, move identical `Inspect()`/`ResidentWeightBytes()` into the base, set `num_layers` in Qwen `Infer`. Three model subclasses migrate to the helpers. Two new real-model SelfCheck tests added, one stale GLM test fixed.

**Tech Stack:** C++20, spdlog, CUDA (nvcc 13.3), Catch2 v3.7.0, CMake Presets (windows-debug/linux-debug). Real-model tests SKIP via CTest `SKIP_RETURN_CODE 4`.

---

## File map

- Modify: `libs/lsxcommon/include/lsxcommon/model_base.h` — add helpers + default Inspect/ResidentWeightBytes.
- Modify: `libs/lsxcommon/src/model_base.cpp` — implement BeginLoad/FinishLoad.
- Modify: `libs/lsxcommon/src/glm_v5_model.h` / `glm_v5_model.cpp` — drop Inspect/ResidentWeightBytes overrides; Load via helpers; assign resident_bytes_.
- Modify: `libs/lsxcommon/src/deepseek_v4_flash_model.h` / `.cpp` — drop Inspect override; Load via helpers.
- Modify: `libs/lsxcommon/src/qwen_base_model.cpp` + header `qwen_base_model.h` — Load via helpers; assign resident_bytes_; set num_layers in Infer; drop Inspect override + ResidentWeightBytes override.
- Modify: `tests/test_glm_model.cpp` — replace stale "not implemented" block with guard-semantics assertions.
- Create: `tests/test_qwen_selfcheck.cpp`, `tests/test_deepseek_selfcheck.cpp`.
- Modify: `tests/CMakeLists.txt` — wire two new test executables.

---

## Task 1: Add `BeginLoad`/`FinishLoad` helpers to `ModelBase`

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/model_base.h`
- Modify: `libs/lsxcommon/src/model_base.cpp`

- [ ] **Step 1: Add the helper declarations and members to the header**

Edit `libs/lsxcommon/include/lsxcommon/model_base.h`. After `bool LoadTokenizerDb(...)`, add:
```cpp
    protected:
        bool BeginLoad(const std::string &model_path);
        void FinishLoad() { loaded_ = true; }
        std::uint64_t resident_bytes_ = 0;
```
The `Inspect()` and `ResidentWeightBytes()` declarations already live on `IModel`. They become base implementations below.

- [ ] **Step 2: Implement `BeginLoad` in the source**

Edit `libs/lsxcommon/src/model_base.cpp`, inside `namespace lsxcommon {}`, before `ValidateInferRequest`:
```cpp
bool ModelBase::BeginLoad(const std::string &model_path)
{
    if (loaded_) return true;
    if (model_path.empty()) {
        spdlog::error("[model] empty model_path");
        return false;
    }
    return LoadTokenizerDb(model_path);
}
```

- [ ] **Step 3: Add base `Inspect()` and `ResidentWeightBytes()` implementations**

In `model_base.cpp`, inside the same namespace block, add:
```cpp
SafetensorsModel ModelBase::Inspect()
{
    return lsxcommon::scan_directory(cfg_.model_path);
}

std::uint64_t ModelBase::ResidentWeightBytes() const
{
    return resident_bytes_;
}
```
`scan_directory` is declared in `lsxcommon/safetensors.h` (already included via `model.h`). No new include required.

- [ ] **Step 4: Build to confirm the base compiles**

Run: `"C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build build --config Debug --target test_model_runtime`
Expected: test_model_runtime target compiles and links.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/model_base.h libs/lsxcommon/src/model_base.cpp
git commit -m "refactor: add BeginLoad/FinishLoad + base Inspect/ResidentWeightBytes to ModelBase"
```

---

## Task 2: Migrate DeepSeek `Load`/`Inspect` to base helpers

**Files:**
- Modify: `libs/lsxcommon/src/deepseek_v4_flash_model.h`
- Modify: `libs/lsxcommon/src/deepseek_v4_flash_model.cpp`

- [ ] **Step 1: Remove the `Inspect` override from the header**

Edit `libs/lsxcommon/src/deepseek_v4_flash_model.h`, delete line 18:
```cpp
    SafetensorsModel Inspect() override;
```

- [ ] **Step 2: Rewrite `Load` to use the helper**

Edit `libs/lsxcommon/src/deepseek_v4_flash_model.cpp`. Replace the `Load()` body (lines 29-39):
```cpp
bool DeepSeekV4FlashModel::Load()
{
    if (!BeginLoad(cfg_.model_path)) return false;
    return true;
}
```
Note: `BeginLoad` already calls `LoadTokenizerDb`; the `loaded_` early-return and `empty model_path` guard are inside `BeginLoad`. **Important behavior change:** DeepSeek previously logged `[model] empty model_path`; `BeginLoad` logs the same tag, so CLI/app output is unchanged.

- [ ] **Step 3: Remove `Inspect` from the source**

Edit `libs/lsxcommon/src/deepseek_v4_flash_model.cpp`, delete the entire `Inspect()` body (lines 56-59):
```cpp
    SafetensorsModel DeepSeekV4FlashModel::Inspect()
    {
        return lsxcommon::scan_directory(cfg_.model_path);
    }
```

- [ ] **Step 4: Verify the two existing DeepSeek tests still pass**

Run: `ctest --preset windows-debug -R deepseek_prompt_output -R deepseek_loader`
Expected: both pass (prompt_output SKIPs without a real model; loader is synthetic and passes).

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/deepseek_v4_flash_model.h libs/lsxcommon/src/deepseek_v4_flash_model.cpp
git commit -m "refactor: DeepSeek Load/Inspect via ModelBase helpers"
```

---

## Task 3: Migrate GLM `Load`/`Inspect`/`ResidentWeightBytes`

**Files:**
- Modify: `libs/lsxcommon/src/glm_v5_model.h`
- Modify: `libs/lsxcommon/src/glm_v5_model.cpp`
- Modify: `libs/lsxcommon/src/glm_inference.cu` (no change — do NOT touch)

**Design constraint (spec 3.3):** GLM is MoE. `ResidentWeightBytes()` today returns pinned-persistent bytes (`pw_.resident_bytes`), NOT the per-infer peak. Preserve this exactly by assigning `resident_bytes_` at pin time.

- [ ] **Step 1: Header — remove overrides, keep the pinned-weights member**

Edit `libs/lsxcommon/src/glm_v5_model.h`. Delete line 19 (`SafetensorsModel Inspect() override;`). Change line 24 from:
```cpp
    std::uint64_t ResidentWeightBytes() const override { return pw_.resident_bytes; }
```
to (remove the override — it now lives in base):
```cpp
```
Delete the line entirely. Keep `pw_` private member and `Type()` inline override unchanged.

- [ ] **Step 2: Rewrite `Load` to use the helper**

Edit `libs/lsxcommon/src/glm_v5_model.cpp`, replace `Load()` body (lines 21-31):
```cpp
bool GlmV5Model::Load()
{
    if (!BeginLoad(cfg_.model_path)) return false;
    FinishLoad();
    return true;
}
```

- [ ] **Step 3: Sync `resident_bytes_` at pin time and remove GLM `Inspect`**

GLM pins weights inside `Infer` (via `glm::LoadGlmPinnedWeights`), then frees them in `cleanup()`. Because `ResidentWeightBytes()` must keep the pinned-persistent semantics (not the transient per-run peak), set `resident_bytes_` from `pw_.resident_bytes` each time the pinned set is loaded, before generation:
```cpp
    cudaError_t e = glm::LoadGlmPinnedWeights(cfg_.model_path, pw_, st);
    if (e != cudaSuccess)
        return fail(...);
    resident_bytes_ = pw_.resident_bytes;   // persist pinned set — NOT the arena/peak
```
Also remove the `Inspect()` override body (`glm_v5_model.cpp:147-150`):
```cpp
SafetensorsModel GlmV5Model::Inspect() { return lsxcommon::scan_directory(cfg_.model_path); }
```

- [ ] **Step 4: Verify the GLM selfcheck test still passes (or SKIPs)**

Run: `ctest --preset windows-debug -R glm_selfcheck`
Expected: SKIP without real checkpoint; PASS with real weights — no failure introduced.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/glm_v5_model.h libs/lsxcommon/src/glm_v5_model.cpp
git commit -m "refactor: GLM Load/Inspect/ResidentWeightBytes via ModelBase"
```

---

## Task 4: Migrate Qwen `Load`/`Inspect`/`ResidentWeightBytes` + set `num_layers`

**Files:**
- Modify: `libs/lsxcommon/src/qwen_base_model.cpp`
- Modify: `libs/lsxcommon/include/lsxcommon/qwen_base_model.h`
- Modify: `libs/lsxcommon/src/qwen_v5_model.cpp`, `libs/lsxcommon/src/qwen_3_8_27b_model.cpp` (no change — both inherit)

- [ ] **Step 1: Update `qwen_base_model.h`**

Delete line 23 (`SafetensorsModel Inspect() override;`) and line 29 (`std::uint64_t ResidentWeightBytes() const override { return pw_.resident_bytes; }`).

- [ ] **Step 2: Rewrite `Load` with helpers + set `resident_bytes_`**

Edit `libs/lsxcommon/src/qwen_base_model.cpp`. Replace the current `Load()` (lines 21-52) body with the same guard/load logic routed through `BeginLoad`, preserving the `tokenizer_optional_` order and weight pin. New body:
```cpp
bool QwenBaseModel::Load()
{
    tokenizer_optional_ = true;
    if (!BeginLoad(cfg_.model_path)) return false;
    if (!weights_loaded_) {
        cudaStream_t st = nullptr;
        if (cudaStreamCreate(&st) == cudaSuccess) {
            bool ok = qwen::LoadQwenPinnedWeights(cfg_.model_path, pw_, st);
            cudaStreamSynchronize(st);
            cudaStreamDestroy(st);
            weights_loaded_ = ok;
            if (!ok) {
                spdlog::error("[qwen-model] LoadQwenPinnedWeights failed");
                return false;
            }
            resident_bytes_ = pw_.resident_bytes;   // dense: full model resident after Load
            spdlog::info("[qwen-model] pinned weights loaded: {} layers "
                         "({} bytes resident)",
                         pw_.layers.size(), pw_.resident_bytes);
        } else {
            spdlog::error("[qwen-model] stream create failed");
            return false;
        }
    }
    FinishLoad();
    ValidateGeometry(pw_.top);
    return true;
}
```
The `tokenizer_optional_ = true;` must precede `BeginLoad(...)` — it affects `LoadTokenizerDb` behaviour inside `BeginLoad`.

- [ ] **Step 3: Sync `resident_bytes_` on `Unload` and drop `Inspect`**

Back in `QwenBaseModel::Unload()`, after `pw_.Free();`, add:
```cpp
    resident_bytes_ = 0;
```
Delete the `Inspect()` method (lines 140-143):
```cpp
SafetensorsModel QwenBaseModel::Inspect() { return lsxcommon::scan_directory(cfg_.model_path); }
```

- [ ] **Step 4: Set `num_layers` in `QwenBaseModel::Infer`**

In `Infer()`, after `r.ok = true; r.num_tokens = ...;`, add:
```cpp
    r.num_layers = pw_.top.num_layers;
```
(`pw_.top.num_layers` is the layer count, `qwen_convert.h:51`, verified `ValidateGeometry` uses `pw_.top`.)

- [ ] **Step 5: Verify Qwen model tests pass**

Run: `ctest --preset windows-debug -R qwen_model -R qwen_3_8_27b_model`
Expected: both pass (synthetic paths).

- [ ] **Step 6: Commit**

```bash
git add libs/lsxcommon/src/qwen_base_model.cpp libs/lsxcommon/include/lsxcommon/qwen_base_model.h
git commit -m "refactor: Qwen Load/Inspect/ResidentWeightBytes via Base, set num_layers in Infer"
```

---

## Task 5: Fix stale GLM guard test

**Files:**
- Modify: `tests/test_glm_model.cpp`

- [ ] **Step 1: Replace the "not implemented" block with guard-semantics assertions**

Edit `tests/test_glm_model.cpp`, replace lines 48-58:
```cpp
    // Inference pipeline is not yet implemented for GLM.
    lsxcommon::ModelRequest req;
    req.input_ids = {154826};
    req.max_gen_tokens = 0;
    auto r = m->Infer(req);
    CHECK(!r.ok);
    CHECK(!r.error_msg.empty());

    auto sc = m->SelfCheck(req);
    CHECK(!sc.ok);
    CHECK(!sc.error_msg.empty());
```
with:
```cpp
    // Guard semantics (ModelBase::ValidateInferRequest): a loaded model with a
    // synthetic dir and zero max_gen_tokens must reject the request; no real
    // 79-layer weights are needed.
    lsxcommon::ModelRequest zero;
    zero.input_ids = {154826};
    zero.max_gen_tokens = 0;
    auto r0 = m->Infer(zero);
    CHECK(!r0.ok);
    CHECK(r0.error_msg == "max_gen_tokens must be positive");

    auto sc0 = m->SelfCheck(zero);
    CHECK(!sc0.ok);
    CHECK(sc0.error_msg == "max_gen_tokens must be positive");

    lsxcommon::ModelRequest empty;
    empty.input_ids = {};
    empty.max_gen_tokens = 8;
    auto rem = m->Infer(empty);
    CHECK(!rem.ok);
    CHECK(rem.error_msg == "empty input_ids");
```
Note: `Infer` with a valid request on a synthetic dir (no real GLM weights) will FAIL at the pinned-weight stage — that is expected and NOT asserted here, to keep the test robust to absent checkpoints.

- [ ] **Step 2: Run the test**

Run: `ctest --test-dir build/tests --test test-glm-model -c`
Expected: PASS. Exactly 1 test case; no SKIP.

- [ ] **Step 3: Commit**

```bash
git add tests/test_glm_model.cpp
git commit -m "test: GLM guard test asserts ValidateInferRequest semantics, not pipeline absence"
```

---

## Task 6: New `test_qwen_selfcheck.cpp`

**Files:**
- Create: `tests/test_qwen_selfcheck.cpp`

Copy the structure of `tests/test_glm_selfcheck.cpp` and adapt to Qwen. Qwen uses the `<|user|>`/`<|assistant|>` chat framing similar to GLM, but **must** be verified against the actual Qwen tokenizer; the generic fallback is the raw encode (like DeepSeek). Use `LSX_TEST_QWEN_MODEL_PATH` env (default `lsxcommon::lsx_model_models_dir() / "Qwen3.5-9B"`).

```cpp
// TDD — Qwen3.5-9B SelfCheck harness (Model Ingestion Protocol step 8).
// Real-model test: SKIPs when no Qwen checkpoint is present (CTest
// SKIP_RETURN_CODE 4).
#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <lsxcommon/model.h>
#include <lsxcommon/model_factory.h>
#include <lsxcommon/model_runtime.h>
#include <lsxcommon/path_registry.h>
#include <lsxcommon/safetensors.h>
#include <lsxcommon/tokenizer.h>

namespace {

constexpr int kStressCycles = 3;

int SelfCheckCycles()
{
    if (const char *p = std::getenv("LSX_TEST_QWEN_SELFCHECK_CYCLES")) {
        const int n = std::atoi(p);
        if (n >= 1) return n;
    }
    return kStressCycles;
}

std::filesystem::path QwenModelPath()
{
    const char *env = std::getenv("LSX_TEST_QWEN_MODEL_PATH");
    return env ? std::filesystem::path(env)
               : std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
                     "Qwen3.5-9B";
}

// Encode the shared "capital of France" prompt with the raw tokenizer encode.
// TODO(impl): if the Qwen tokenizer requires chat-template framing, mirror the
// GLM EncodeSelfCheckPrompt pattern here; start with the raw encode and confirm
// against the real model that first_gen_text == "Paris".
std::vector<int32_t> EncodeSelfCheckPrompt(lsxcommon::ArrowTokenizerDb &tok)
{
    const char *prompt = "The capital of France is";
    std::vector<int32_t> ids(512, 0);
    size_t cap = ids.size();
    const int32_t n = lsxcommon::arrow_tokenizer_encode(
        tok, prompt, std::strlen(prompt), ids.data(), &cap);
    if (n <= 0) return {};
    ids.resize(static_cast<size_t>(n));
    return ids;
}

} // namespace

TEST_CASE("Qwen3.5-9B SelfCheck: 'The capital of France is' -> 'Paris'",
          "[qwen][model][selfcheck][real-model]")
{
    const std::filesystem::path model = QwenModelPath();
    if (!std::filesystem::exists(model / "logestix" / "pinned"))
        SKIP("Qwen real model directory not present");

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model.string();
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Type() == "qwen3.5-9b");
    REQUIRE(m->Load());

    const std::vector<int32_t> ids = EncodeSelfCheckPrompt(m->Tokenizer());
    REQUIRE(ids.size() > 0);

    lsxcommon::ModelRequest req;
    req.input_ids = ids;
    req.max_gen_tokens = 2;

    auto sc0 = m->SelfCheck(req);
    REQUIRE(sc0.ok);
    REQUIRE(sc0.selfcheck_passed);
    REQUIRE(sc0.first_gen_text == "Paris");
    REQUIRE(!sc0.final_logits.empty());

    const std::vector<float> &ref_logits = sc0.final_logits;
    const int32_t ref_first = sc0.first_gen_tok_id;
    const int cycles = SelfCheckCycles();

    for (int i = 1; i < cycles; ++i) {
        auto sc = m->SelfCheck(req);
        REQUIRE(sc.ok);
        REQUIRE(sc.selfcheck_passed);
        CHECK(sc.first_gen_text == "Paris");
        CHECK(sc.first_gen_tok_id == ref_first);
        CHECK(sc.final_logits.size() == ref_logits.size());
        CHECK(std::memcmp(sc.final_logits.data(), ref_logits.data(),
                          ref_logits.size() * sizeof(float)) == 0);
    }

    m->Unload();
}
```

**Open during implementation:** the exact `QwenTopology` / `pw_.top` field is confirmed `num_layers` (Task 4). For `EncodeSelfCheckPrompt`, if the raw encode does not yield `"Paris"` with the real Qwen checkpoint, replicate the GLM jinja framing (`<|user|>...<|assistant|>`) — the contract is `first_gen_text == "Paris"`.

- [ ] **Step 1: Write the file exactly as above**

- [ ] **Step 2: Wire into CMake (Task 8 does the CMake)**

- [ ] **Step 3: Run to confirm it compiles/SKIPs**

Run: `ctest --test-dir windows/tests -R test_qwen_selfcheck`
Expected: 1 of 1 test case SKIPPED when no Qwen model present; compiles clean.

- [ ] **Step 4: Commit** (combined with Task 8 wiring below)

---

## Task 7: New `tests/test_deepseek_selfcheck.cpp`

**Files:**
- Create: `tests/test_deepseek_selfcheck.cpp`

Mirror the GLM selfcheck test, using the DeepSeek **raw** BPE encode pattern from `tests/test_deepseek_prompt_output.cpp` and `LSX_TEST_DEEPSEEK_MODEL_PATH`.

[Write `tests/test_deepseek_selfcheck.cpp` with this exact content:]

```cpp
// TDD — DeepSeek-V4-Flash SelfCheck harness (Model Ingestion Protocol step 8).
// Real-model test: SKIPs when no DeepSeek checkpoint is present (CTest
// SKIP_RETURN_CODE 4).
#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <lsxcommon/model.h>
#include <lsxcommon/model_factory.h>
#include <lsxcommon/path_registry.h>
#include <lsxcommon/safetensors.h>
#include <lsxcommon/tokenizer.h>

namespace {

constexpr int kStressCycles = 3;

int SelfCheckCycles()
{
    if (const char *p = std::getenv("LSX_TEST_DEEPSEEK_SELFCHECK_CYCLES")) {
        const int n = std::atoi(p);
        if (n >= 1) return n;
    }
    return kStressCycles;
}

std::filesystem::path DeepSeekModelPath()
{
    const char *env = std::getenv("LSX_TEST_DEEPSEEK_MODEL_PATH");
    return env ? std::filesystem::path(env)
               : std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
                     "DeepSeek-V4-Flash";
}

// DeepSeek consumes the raw prompt (no chat-template framing), same as
// test_deepseek_prompt_output.cpp.
std::vector<int32_t> EncodeSelfCheckPrompt(lsxcommon::ArrowTokenizerDb &tok)
{
    const char *prompt = "The capital of France is";
    std::vector<int32_t> ids(512, 0);
    size_t cap = ids.size();
    const int32_t n = lsxcommon::arrow_tokenizer_encode(
        tok, prompt, std::strlen(prompt), ids.data(), &cap);
    if (n <= 0) return {};
    ids.resize(static_cast<size_t>(n));
    return ids;
}

} // namespace

TEST_CASE("DeepSeek-V4-Flash SelfCheck: 'The capital of France is' -> 'Paris', "
          "3 stress cycles with bit-exact deterministic logits",
          "[deepseek][model][selfcheck][real-model]")
{
    const std::filesystem::path model = DeepSeekModelPath();
    if (!std::filesystem::exists(model / "logestix" / "pinned"))
        SKIP("DeepSeek real model directory not present");

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model.string();
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Type() == "deepseek-v4-flash");
    REQUIRE(m->Load());

    const std::vector<int32_t> ids = EncodeSelfCheckPrompt(m->Tokenizer());
    REQUIRE(ids.size() > 0);

    lsxcommon::ModelRequest req;
    req.input_ids = ids;
    req.max_gen_tokens = 2;

    auto sc0 = m->SelfCheck(req);
    REQUIRE(sc0.ok);
    REQUIRE(sc0.selfcheck_passed);
    REQUIRE(sc0.first_gen_text == "Paris");
    REQUIRE(!sc0.final_logits.empty());

    const std::vector<float> &ref_logits = sc0.final_logits;
    const int32_t ref_first = sc0.first_gen_tok_id;
    const int cycles = SelfCheckCycles();

    for (int i = 1; i < cycles; ++i) {
        auto sc = m->SelfCheck(req);
        REQUIRE(sc.ok);
        REQUIRE(sc.selfcheck_passed);
        CHECK(sc.first_gen_text == "Paris");
        CHECK(sc.first_gen_tok_id == ref_first);
        CHECK(sc.final_logits.size() == ref_logits.size());
        CHECK(std::memcmp(sc.final_logits.data(), ref_logits.data(),
                          ref_logits.size() * sizeof(float)) == 0);
    }

    m->Unload();
}
```

**NOTE:** the DeepSeek `SelfCheck` already routes through `runtime::RunParisSelfCheck`
(`model_runtime.cpp:47`), so `selfcheck_passed` is populated without any refactor;
the test only asserts it.

- [ ] **Step 1: Write the file with this exact content**

- [ ] **Step 2: Run to confirm it compiles + SKIPs**

Run: `ctest --preset windows-debug -R deepseek_selfcheck`
Expected: 1 of 1 test case SKIPPED when no DeepSeek model present; compiles clean.

- [ ] **Step 3: Wire into CMakeLists (Task 8, Step 2)**

---

## Task 8: Wire the two new tests into `tests/CMakeLists.txt`

**Files:**
- Modify: `tests/CMakeLists.txt`

Add two blocks, mirroring `test_glm_selfcheck` (lines 355-375) and `test_deepseek_prompt_output` (lines 428-446) with their `set_tests_properties(... SKIP_RETURN_CODE 4)`.

- [ ] **Step 1: Append the Qwen selfcheck block** (after line ~446, before the `test_model_factory` block):

```cmake
# ── Тест Qwen3.5-9B SelfCheck (шаг 8 протокола): инкрементальный 1->2-токенный
#    селфчек на реальной модели, минимум 3 цикла, детерминированный final_logits.
#    SKIP без чекпоинта.
add_executable(test_qwen_selfcheck test_qwen_selfcheck.cpp)
target_compile_definitions(test_qwen_selfcheck PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
    )
target_link_libraries(test_qwen_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
if(WIN32)
    add_custom_command(TARGET test_qwen_selfcheck POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_qwen_selfcheck>")
    add_custom_command(TARGET test_qwen_selfcheck POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_qwen_selfcheck>")
endif()
add_test(NAME test_qwen_selfcheck COMMAND test_qwen_selfcheck)
set_tests_properties(test_qwen_selfcheck PROPERTIES SKIP_RETURN_CODE 4)
```

- [ ] **Step 2: Append the DeepSeek self-test block** (mirroring `test_deepseek_prompt_output`, no `target_compile_definitions`):

```cmake
add_executable(test_deepseek_selfcheck test_deepseek_selfcheck.cpp)
target_link_libraries(test_deepseek_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
if(WIN32) ... copy_if_different arrow/arrow_cuda ... endif()
add_test(NAME test_deepseek_selfcheck COMMAND test_deepseek_selfcheck)
set_tests_properties(test_deepseek_selfcheck PROPERTIES SKIP_RETURN_CODE 4)
```

- [ ] **Step 3: Reconfigure CMake to pick up the new targets**

Run: `cmake --preset windows-debug`
(or the `cmake --build build --config Debug` path in AGENTS.md).

- [ ] **Step 4: Build and run the new tests**

```bash
git add tests/CMakeLists.txt tests/test_qwen_selfcheck.cpp tests/test_deepseek_selfcheck.cpp
git commit -m "test: add deterministic Qwen + DeepSeek SelfCheck harnesses"
```

---

## Task 9: Full-suite verification

- [ ] **Step 1: Build the full project**

Run: `cmake -E time -pdebug && cmake -E time -build -pdebug` (or the AGENTS.md-scripted preset command). Expected: clean build.

- [ ] **Step 2: Run the targeted model tests**

Run: `ctest --preset windows-debug -R glm -R qwen -R deepseek -R model_runtime -R model_engine -R model_factory`
Expected: all pass; real-model tests SKIP when weights absent.

- [ ] **Step 3: Confirm the refactor is zero-behaviour-change**

Confirm `test_model_runtime` (ValidateInferRequest guard messages) still passes unchanged, and no test that asserted "Infer not implemented" remains. Grep: `rg "not yet implemented"` → no matches for GLM.

- [ ] **Step 4: Commit**

```bash
git add .
git commit -m "test: add qwen + deepseek selfcheck; fix GLM guard test; unconditional pass"
```
(Only if Step 3 surfaces changes not yet committed.)

---

## Self-Review

Crosscheck against the spec (docs/2026-08-20-inference-unification-design.md):

- **3.3 dense vs MoE residency:** Task 3 (GLM pinned-persistent `resident_bytes_` at pin time) and Task 4 (Qwen full-model `resident_bytes_` at Load, reset at Unload) preserve today's semantics; DeepSeek keeps default `0` (Task 2, no override). ✓
- **num_layers consistency:** Task 4 Step 4 sets `r.num_layers = pw_.top.num_layers` ✓.
- **Stale GLM test (3 numbering):** Task 5 rewrites to guard semantics ✓.
- **Missing SelfCheck tests for Qwen/DeepSeek:** Tasks 6-7 add them; CMake wired in Task 8 ✓.
- **Adaptive test_model_factory:** not touched; still only tests present models ✓.

**Type consistency:** `beginLoad` unused; `resident_bytes_` member name used consistently across Tasks 1-4; `RunParisSelfCheck` referenced correctly (already in `model_runtime.h`). `EncodeSelfCheckPrompt` name kept consistent with GLM test.