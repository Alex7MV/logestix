# GigaChat3.5 Ingestion — Plan 1: Converter, NVFP4, Registration and Loader

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **No subagents** (AGENTS.md autonomy rule): execute inline in the main session.

**Goal:** Convert the FP8 GigaChat3.5-432B checkpoint into the `logestix-gigachat-nvfp4` layout, register the family, and make the model loadable/inspectable with a tokenizer bridge — without CUDA inference kernels (Plan 2).

**Architecture:** A new family (`gigachat`) mirrors the Nemotron converter/loader pattern: a GigaChat-owned tensor classifier + topology parser, an FP8-block-128→BF16 dequant GPU step feeding the neutral `gpu::quantize_tensor_gpu` NVFP4 packer, a frozen compile-time passport, a strict index/geometry loader, and a `ModelBase`-derived `GigaChat35Model`.

**Tech Stack:** C++20, Apache Arrow 25.0.1, CUDA 13.4, abseil, simdjson, Catch2 v3.16.0, CMake presets.

## Global Constraints

- Family token `gigachat`; `kType = "gigachat3.5-432b-a28b-reasoning"`; namespace `lsxcommon::gigachat`; passport namespace `lsxcommon::gigachat::frozen`.
- Format tag `logestix-gigachat-nvfp4`.
- No cross-family includes: `gigachat_moe_*` must not include `glm_*`/`qwen_*`/`deepseek_*`/`gemma_*`/`kimi_*`/`nemotron_*` headers.
- Geometry frozen as compile-time constants + `static_assert`; runtime config reads forbidden.
- Norms, biases, `A_log`, `dt_bias`, `conv1d`, `in_proj_ba`, `e_score_correction_bias`, router `gate.weight`, embeddings, `lm_head`, and all gate projections are never quantized.
- Converter tests: label `convert`, bounded `TIMEOUT 120`, synthetic checkpoint only (`test_synth_checkpoint.h`), `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1` as a runtime ENV (never a compile define).
- Every new test gets its ctest label in the same commit; MTP is cut/stored + counted, never applied.
- Logging only via spdlog; no `std::cout`/`printf`. English commit messages with `feat:`/`test:`/`fix:` prefixes.
- `gigachat_moe_model.cpp` must contain `LsxFastMathScope` and `EffectiveFastMathMode(cfg_)`.

---

## File Structure

- `libs/lsxcommon/include/lsxcommon/gigachat_moe_convert.h` — public converter + classifier + topology API.
- `libs/lsxcommon/src/gigachat_moe_constants.h` — frozen passport (leaf).
- `libs/lsxcommon/src/gigachat_moe_convert.cpp` — converter implementation.
- `libs/lsxcommon/src/gigachat_moe_dequant.cu` — FP8-block-128 → BF16 GPU dequant.
- `libs/lsxcommon/src/gigachat_moe_load.cpp` — index parse + geometry gate + weight upload.
- `libs/lsxcommon/include/lsxcommon/gigachat_moe_model.h` + `src/gigachat_moe_model.cpp` — `GigaChat35Model`.
- `tests/test_gigachat_moe_fixture.h` — family synthetic checkpoint fixture.
- `tests/test_gigachat_moe_math.cpp` — host reference math (passport / formulas).
- `tests/test_gigachat_moe_convert.cpp` — synthetic E2E.
- `tests/test_gigachat_moe_loader.cpp` — loader.
- `tests/test_gigachat_moe_model.cpp` — model/registry/inspect.

---

### Task 1: Family token + frozen passport + independence guard row

**Files:**
- Modify: `tests/test_code_naming_guard.cpp` (`SchemeRegex` `fam`, `FamilyStems()`)
- Modify: `tests/test_model_independence_guard.cpp` (`kPassports`, `owner`, `consumers`)
- Modify: `tests/test_model_suite_guard.cpp` (`FamilyStems()`)
- Create: `libs/lsxcommon/src/gigachat_moe_constants.h`
- Test: the four guard executables

**Interfaces:**
- Produces: `libs/lsxcommon/src/gigachat_moe_constants.h` with `namespace lsxcommon::gigachat::frozen` and all constants from the spec's frozen-geometry tables (`kType`, `kHidden`, `kLayers`, `kVocab`, `kFullAttnLayers`, `kFirstKDense`, `kRmsEps`, `kNormGatingWeight`, `kGateRank`, `kMtpLayers`, MLA block, GDN block, MoE block, token ids) with `static_assert`s.

- [ ] **Step 1: Add the family token and passport row (RED)**

In `tests/test_code_naming_guard.cpp` change `fam` to
`"(?:glm|qwen|gemma|kimi|deepseek|nemotron|gigachat)"` and add `"gigachat"` to
`FamilyStems()`. In `tests/test_model_suite_guard.cpp` add `"gigachat"` to
`FamilyStems()`. In `tests/test_model_independence_guard.cpp` append to
`kPassports`:
```cpp
    {"gigachat3.5-432b-a28b-reasoning", "gigachat_moe_constants.h",
     "lsxcommon::gigachat::frozen"},
```
add to the `owner` map `{"gigachat_moe_constants.h", "namespace frozen {"}`.
(The `consumers` row is added in Task 9, when `gigachat_moe_model.cpp` exists —
the consumer test reads the file, so the row cannot precede it.)

- [ ] **Step 2: Run the guard to verify it fails**

Run: `cmake --build --preset linux-debug --target test_model_independence_guard && ./build/linux-debug/tests/test_model_independence_guard`
Expected: FAIL — `gigachat_moe_constants.h` does not exist.

- [ ] **Step 3: Create the frozen passport**

Create `libs/lsxcommon/src/gigachat_moe_constants.h`:
```cpp
#pragma once

// Frozen GigaChat3.5-432B-A28B-Reasoning geometry (compile-time passport).
// Source of truth: /mnt/llm_models/models/GigaChat3.5-432B-A28B-Reasoning/
// config.json + safetensors inventory audit (2026-09-20), and the reference
// modeling_gigachat3_5.py (not vendored). 40 layers + 3 MTP; MLA full-attention
// at {3,7,...,39}; GatedDeltaNet elsewhere; MoE 3..39; ZeroCenteredGatedNorm.

#include <cstdint>

namespace lsxcommon::gigachat::frozen {

inline constexpr const char *kType = "gigachat3.5-432b-a28b-reasoning";

inline constexpr int kHidden = 7168;
inline constexpr int kLayers = 40;
inline constexpr int kVocab = 128256;
inline constexpr int kFirstKDense = 3;
inline constexpr float kRmsEps = 1e-6f;
inline constexpr float kNormGatingWeight = 2.0f;
inline constexpr int kGateRank = 16;

inline constexpr int kFullAttnLayers[10] = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39};
inline constexpr int kNumFullAttn = 10;
inline constexpr int kMtpLayers[3] = {40, 41, 42};

// MLA.
inline constexpr int kHeads = 64;
inline constexpr int kKVHeads = 64;
inline constexpr int kQkNopeHeadDim = 128;
inline constexpr int kQkRopeHeadDim = 64;
inline constexpr int kQkHeadDim = kQkNopeHeadDim + kQkRopeHeadDim;  // 192
inline constexpr int kVHeadDim = 128;
inline constexpr int kQLoraRank = 1536;
inline constexpr int kKvLoraRank = 512;
inline constexpr float kRopeTheta = 100000.0f;
inline constexpr double kYarnFactor = 8.0;
inline constexpr double kYarnOrig = 32768.0;
inline constexpr double kYarnBetaFast = 32.0;
inline constexpr double kYarnBetaSlow = 1.0;
inline constexpr double kYarnMscale = 1.0;
inline constexpr double kYarnMscaleAllDim = 1.0;

// GatedDeltaNet.
inline constexpr int kLinearNumKeyHeads = 32;
inline constexpr int kLinearNumValueHeads = 64;
inline constexpr int kLinearKvRepeat = kLinearNumValueHeads / kLinearNumKeyHeads;  // 2
inline constexpr int kLinearKeyHeadDim = 128;
inline constexpr int kLinearValueHeadDim = 128;
inline constexpr int kLinearConvKernel = 4;
inline constexpr int kLinearKeyDim = kLinearNumKeyHeads * kLinearKeyHeadDim;      // 4096
inline constexpr int kLinearValueDim = kLinearNumValueHeads * kLinearValueHeadDim; // 8192
inline constexpr int kLinearConvDim = 2 * kLinearKeyDim + kLinearValueDim;         // 16384
inline constexpr int kLinearQkvzDim = 2 * kLinearKeyDim + 2 * kLinearValueDim;     // 24576
inline constexpr int kLinearBaDim = 2 * kLinearNumValueHeads;                      // 128
inline constexpr float kLinearSigmoidGateScale = 2.0f;
inline constexpr float kLinearONormEps = 1e-6f;

// MoE / MLP.
inline constexpr int kNumRouted = 256;
inline constexpr int kTopK = 8;
inline constexpr int kMoeIntermediate = 2048;
inline constexpr int kSharedExperts = 1;
inline constexpr int kSharedIntermediate = kMoeIntermediate * kSharedExperts;
inline constexpr int kDenseIntermediate = 18432;
inline constexpr float kRoutedScale = 2.5f;
inline constexpr float kSwigluLimit = 10.0f;

// Tokens.
inline constexpr int kBosId = 1;
inline constexpr int kEosId = 2;
inline constexpr int kPadId = 2;
inline constexpr int kRoleSepId = 128000;
inline constexpr int kMessageSepId = 128001;
inline constexpr int kThinkId = 128014;
inline constexpr int kThinkEndId = 128015;

// Frozen-geometry invariants (compile-time).
static_assert(kQkHeadDim == 192, "MLA qk head dim");
static_assert(kLinearKvRepeat * kLinearNumKeyHeads == kLinearNumValueHeads,
              "GDN head repeat");
static_assert(kLinearQkvzDim == 2 * kLinearKeyDim + 2 * kLinearValueDim,
              "fused qkvz width");
static_assert(kSharedIntermediate == 2048, "shared expert width");
static_assert(kNumFullAttn == 10, "full-attention layer count");
static_assert(kTopK <= kNumRouted, "top-k exceeds expert count");
static_assert(kHidden % 128 == 0, "hidden must tile by 128");

} // namespace lsxcommon::gigachat::frozen
```

- [ ] **Step 4: Run the guards to verify they pass**

Run the four guard executables. Expected: all PASS (the independence guard now
finds the passport; the naming/suite guards accept the token).

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/gigachat_moe_constants.h \
  tests/test_code_naming_guard.cpp tests/test_model_independence_guard.cpp \
  tests/test_model_suite_guard.cpp
git commit -m "feat: gigachat family token + frozen geometry passport"
```

---

### Task 2: Host reference math for the passport

**Files:**
- Create: `tests/test_gigachat_moe_math.cpp`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: the passport constants.
- Produces: `test_gigachat_moe_math` executable (label `fast`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_gigachat_moe_math.cpp`:
```cpp
// Host reference math for GigaChat3.5 (no CUDA): pins the derived geometry the
// kernels will implement. Model Ingestion Protocol step 3.
#include <catch2/catch_test_macros.hpp>

#include "gigachat_moe_constants.h"
#include <cmath>

namespace frozen = lsxcommon::gigachat::frozen;

TEST_CASE("gigachat derived geometry matches the reference", "[gigachat][fast]")
{
    CHECK(frozen::kQkHeadDim == 192);
    CHECK(frozen::kLinearKeyDim == 4096);
    CHECK(frozen::kLinearValueDim == 8192);
    CHECK(frozen::kLinearConvDim == 16384);
    CHECK(frozen::kLinearQkvzDim == 24576);
    CHECK(frozen::kLinearBaDim == 128);
    CHECK(frozen::kSharedIntermediate == 2048);

    const double mscale = 0.1 * 1.0 * std::log(8.0) + 1.0;  // yarn_get_mscale(8,1)
    CHECK(mscale == Approx(1.20794).margin(1e-5));
    const double scale = std::pow(192.0, -0.5) * mscale * mscale;
    CHECK(scale == Approx(0.105299).margin(1e-5));

    const double alpha_q = std::sqrt(7168.0 / 1536.0);
    const double alpha_kv = std::sqrt(7168.0 / 512.0);
    CHECK(alpha_q == Approx(2.160246).margin(1e-5));
    CHECK(alpha_kv == Approx(3.741657).margin(1e-5));
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cmake --build --preset linux-debug --target test_gigachat_moe_math`
Expected: FAIL — target/build error: `test_gigachat_moe_math` not yet in
`tests/CMakeLists.txt`.

- [ ] **Step 3: Register the target**

In `tests/CMakeLists.txt` add (mirror the `test_nemotron_moe_math` block):
```cmake
# ── GigaChat3.5-432B-A28B-Reasoning tests (hybrid MLA+GDN+MoE) ─────────────
add_executable(test_gigachat_moe_math test_gigachat_moe_math.cpp)
target_link_libraries(test_gigachat_moe_math PRIVATE Catch2::Catch2WithMain)
target_include_directories(test_gigachat_moe_math PRIVATE
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/src")
add_test(NAME test_gigachat_moe_math COMMAND test_gigachat_moe_math)
```
Add `test_gigachat_moe_math` to the `LABELS fast` list at the bottom of the file.

- [ ] **Step 4: Run it to verify it passes**

Run: `cmake --build --preset linux-debug --target test_gigachat_moe_math && ./build/linux-debug/tests/test_gigachat_moe_math`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_gigachat_moe_math.cpp tests/CMakeLists.txt
git commit -m "test: gigachat host reference geometry math"
```

---

### Task 3: Tensor classifier

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/gigachat_moe_convert.h`
- Create: `libs/lsxcommon/src/gigachat_moe_convert.cpp` (classifier only)
- Create: `tests/test_gigachat_moe_convert.cpp` (unit part)
- Modify: `tests/CMakeLists.txt`
- Modify: `libs/lsxcommon/CMakeLists.txt` (add source)

**Interfaces:**
- Produces:
```cpp
namespace lsxcommon::gigachat {
enum class GigaChatTensorKind {
    Unknown, Embed, LmHead, FinalNormWeight, FinalNormGateUp, FinalNormGateDown,
    LayerNormWeight, LayerNormGateUp, LayerNormGateDown,
    QaProj, QbProj, KvaProj, KvbProj, OProj, AttnGateProj,
    MlpGate, MlpUp, MlpDown,
    GdnInProjQkvz, GdnOutProj, GdnConv1d, GdnALog, GdnDtBias, GdnInProjBa,
    GdnNormWeight,
    RouterWeight, RouterBias, SharedGate, SharedUp, SharedDown,
    ExpertGate, ExpertUp, ExpertDown,
    Mtp,
};
struct ParsedGigaChatTensor { GigaChatTensorKind kind = GigaChatTensorKind::Unknown;
                              int layer = -1; int expert = -1; };
ParsedGigaChatTensor classify_gigachat_tensor(const std::string &name);
} // namespace lsxcommon::gigachat
```

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gigachat_moe_convert.cpp`:
```cpp
#include <catch2/catch_test_macros.hpp>
#include "lsxcommon/gigachat_moe_convert.h"

using namespace lsxcommon::gigachat;

TEST_CASE("gigachat tensor classifier", "[gigachat][convert]")
{
    auto k = [](const char *n) { return classify_gigachat_tensor(n).kind; };
    CHECK(k("model.embed_tokens.weight") == GigaChatTensorKind::Embed);
    CHECK(k("lm_head.weight") == GigaChatTensorKind::LmHead);
    CHECK(k("model.norm.weight") == GigaChatTensorKind::FinalNormWeight);
    CHECK(k("model.norm.gate_up_projection.weight") == GigaChatTensorKind::FinalNormGateUp);
    CHECK(k("model.layers.3.input_layernorm.weight") == GigaChatTensorKind::LayerNormWeight);
    CHECK(k("model.layers.7.post_feedforward_layernorm.gate_down_projection.weight")
          == GigaChatTensorKind::LayerNormGateDown);
    CHECK(k("model.layers.3.self_attn.q_a_proj.weight") == GigaChatTensorKind::QaProj);
    CHECK(k("model.layers.3.self_attn.kv_a_proj_with_mqa.weight") == GigaChatTensorKind::KvaProj);
    CHECK(k("model.layers.3.self_attn.gate_proj.weight") == GigaChatTensorKind::AttnGateProj);
    CHECK(k("model.layers.4.self_attn.in_proj_qkvz.weight") == GigaChatTensorKind::GdnInProjQkvz);
    CHECK(k("model.layers.4.self_attn.in_proj_ba.weight") == GigaChatTensorKind::GdnInProjBa);
    CHECK(k("model.layers.4.self_attn.conv1d.weight") == GigaChatTensorKind::GdnConv1d);
    CHECK(k("model.layers.4.self_attn.A_log") == GigaChatTensorKind::GdnALog);
    CHECK(k("model.layers.4.self_attn.norm.weight") == GigaChatTensorKind::GdnNormWeight);
    CHECK(k("model.layers.5.mlp.gate.weight") == GigaChatTensorKind::RouterWeight);
    CHECK(k("model.layers.5.mlp.gate.e_score_correction_bias") == GigaChatTensorKind::RouterBias);
    CHECK(k("model.layers.5.mlp.shared_experts.gate_proj.weight") == GigaChatTensorKind::SharedGate);
    CHECK(k("model.layers.5.mlp.experts.200.up_proj.weight") == GigaChatTensorKind::ExpertUp);
    CHECK(k("model.layers.0.mlp.down_proj.weight") == GigaChatTensorKind::MlpDown);
    CHECK(k("model.layers.40.eh_proj.weight") == GigaChatTensorKind::Mtp);
    CHECK(k("model.layers.41.shared_head.head.weight") == GigaChatTensorKind::Mtp);
    CHECK(k("garbage") == GigaChatTensorKind::Unknown);

    auto e = classify_gigachat_tensor("model.layers.5.mlp.experts.200.up_proj.weight");
    CHECK(e.layer == 5);
    CHECK(e.expert == 200);
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cmake --build --preset linux-debug --target test_gigachat_moe_convert`
Expected: FAIL — missing header/target (register the target in Step 3 first).

- [ ] **Step 3: Register the header, source stub and target**

Create `libs/lsxcommon/include/lsxcommon/gigachat_moe_convert.h` with the enum,
struct and function declarations above plus:
```cpp
struct GigaChatTopology {
    int hidden = 0, layers = 0, vocab = 0, heads = 0, kv_heads = 0;
    int num_experts = 0, top_k = 0, moe_intermediate = 0;
    int first_k_dense = 0;
    double rms_eps = 0.0, routed_scale = 0.0, swiglu_limit = 0.0;
    int bos_id = -1, eos_id = -1, pad_id = -1;
    std::string error_msg;  // non-empty when config is invalid
};
GigaChatTopology ParseGigaChatConfig(const std::string &config_json_text);
arrow::Status ConvertGigaChatModel(const std::string &input_path,
                                   const std::string &output_path);
```
Create `libs/lsxcommon/src/gigachat_moe_convert.cpp` with the file-scan shard
helper (copy the `read_tensor_data` pattern from
`nemotron_moe_convert.cpp:109-125`) and `classify_gigachat_tensor` only.
Add `src/gigachat_moe_convert.cpp` to the `lsxcommon` source list in
`libs/lsxcommon/CMakeLists.txt`.
In `tests/CMakeLists.txt` add `test_gigachat_moe_convert` with
`target_compile_definitions(... LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1)`,
`target_link_libraries(... lsxcommon Catch2::Catch2WithMain)`,
`add_test(...)`, `set_tests_properties(... TIMEOUT 120 ENVIRONMENT
"LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1")`, and label `convert`.

- [ ] **Step 4: Implement the classifier, run to green**

Implement `classify_gigachat_tensor` by exact suffix matching (longest-first)
with the regex `model\.layers\.(\d+)\.`. For layers in `{40,41,42}` return
`Mtp` (check first). Run:
`cmake --build --preset linux-debug --target test_gigachat_moe_convert && ./build/linux-debug/tests/test_gigachat_moe_convert`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/gigachat_moe_convert.h \
  libs/lsxcommon/src/gigachat_moe_convert.cpp \
  tests/test_gigachat_moe_convert.cpp tests/CMakeLists.txt libs/lsxcommon/CMakeLists.txt
git commit -m "feat: gigachat tensor classifier"
```

---

### Task 4: Topology config parser

**Files:**
- Modify: `libs/lsxcommon/src/gigachat_moe_convert.cpp`
- Modify: `tests/test_gigachat_moe_convert.cpp`

**Interfaces:**
- Consumes: `GigaChatTopology`.
- Produces: `ParseGigaChatConfig(text)` with `error_msg` empty on success, and a
  frozen-geometry cross-check helper used by the loader in Task 6.

- [ ] **Step 1: Write the failing test**

Append:
```cpp
TEST_CASE("gigachat config parser reads the authoritative fields",
          "[gigachat][convert]")
{
    const char *cfg = R"({
      "hidden_size": 7168, "num_hidden_layers": 40, "vocab_size": 128256,
      "num_attention_heads": 64, "num_key_value_heads": 64,
      "n_routed_experts": 256, "num_experts_per_tok": 8,
      "moe_intermediate_size": 2048, "first_k_dense_replace": 3,
      "rms_norm_eps": 1e-06, "routed_scaling_factor": 2.5, "swiglu_limit": 10,
      "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 2
    })";
    auto t = ParseGigaChatConfig(cfg);
    CHECK(t.error_msg.empty());
    CHECK(t.hidden == 7168);
    CHECK(t.layers == 40);
    CHECK(t.vocab == 128256);
    CHECK(t.num_experts == 256);
    CHECK(t.top_k == 8);
    CHECK(t.bos_id == 1);
    CHECK(t.pad_id == 2);

    auto bad = ParseGigaChatConfig("{ not json ");
    CHECK_FALSE(bad.error_msg.empty());
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cmake --build --preset linux-debug --target test_gigachat_moe_convert && ./build/linux-debug/tests/test_gigachat_moe_convert`
Expected: FAIL — `ParseGigaChatConfig` not implemented (link error).

- [ ] **Step 3: Implement with simdjson**

Implement `ParseGigaChatConfig` using `simdjson::ondemand` (mirror
`nemotron_moe_convert.cpp` config parsing). Require `hidden_size`,
`num_hidden_layers`, `vocab_size`; fill the rest with defaults from the spec.
On any parse error set `error_msg` and return.

- [ ] **Step 4: Run to green**

Run the test. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/gigachat_moe_convert.cpp tests/test_gigachat_moe_convert.cpp
git commit -m "feat: gigachat topology config parser"
```

---

### Task 5: FP8 block-128 → BF16 GPU dequant (partial tiles)

**Files:**
- Create: `libs/lsxcommon/src/gigachat_moe_dequant.cu`
- Create: `tests/test_gigachat_moe_kernels.cu` (dequant part)
- Modify: `libs/lsxcommon/CMakeLists.txt` (new OBJECT lib `lsxcommon_gigachat_cu`)
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Produces: `arrow::Status lsxcommon::gigachat::DequantFp8Block128ToBf16Gpu(const std::vector<uint8_t>& fp8, const std::vector<int64_t>& shape, const std::vector<uint8_t>& scale_f32, const std::vector<int64_t>& scale_shape, std::vector<uint8_t>& bf16);`
  declared in `gigachat_moe_convert.h`. `shape` is `[R,C]`, `scale_shape` is
  `[ceil(R/128), ceil(C/128)]`; the partial last tile is clamped.

- [ ] **Step 1: Write the failing parity test**

Create `tests/test_gigachat_moe_kernels.cu`:
```cpp
// Kernel parity tests for GigaChat3.5 (label inference).
#include <catch2/catch_test_macros.hpp>
#include "lsxcommon/gigachat_moe_convert.h"
#include <cmath>
#include <cstring>
#include <vector>

using namespace lsxcommon::gigachat;

static float Fp8ToFloat(uint8_t b)
{
    // E4M3 decode (exponent bias 7), sign-preserving.
    const int s = (b >> 7) & 1, e = (b >> 3) & 0xF, m = b & 0x7;
    if (e == 0) return (s ? -1.f : 1.f) * std::ldexp(m / 8.0f, -6);
    if (e == 15 && m == 7) return 0.0f;  // NaN sentinel -> 0 in test fixture
    return (s ? -1.f : 1.f) * std::ldexp(1.0f + m / 8.0f, e - 7);
}

TEST_CASE("gigachat fp8 block-128 dequant parity, partial tile",
          "[gigachat][inference]")
{
    // 130 rows x 130 cols -> scale grid ceil(130/128)=2 x 2.
    const int R = 130, C = 130;
    std::vector<uint8_t> fp8(R * C);
    for (int i = 0; i < R * C; ++i) fp8[i] = static_cast<uint8_t>((i % 15) | 0x08);
    std::vector<uint8_t> scale(2 * 2 * 4);
    const float s0 = 0.5f, s1 = 2.0f, s2 = 1.0f, s3 = 0.25f;
    std::memcpy(scale.data() + 0, &s0, 4);
    std::memcpy(scale.data() + 4, &s1, 4);
    std::memcpy(scale.data() + 8, &s2, 4);
    std::memcpy(scale.data() + 12, &s3, 4);

    std::vector<uint8_t> bf16;
    auto st = DequantFp8Block128ToBf16Gpu(fp8, {R, C}, scale, {2, 2}, bf16);
    REQUIRE(st.ok());
    REQUIRE(bf16.size() == static_cast<size_t>(R) * C * 2);

    auto bf16_at = [&](int r, int c) {
        uint16_t b = static_cast<uint16_t>(bf16[(r * C + c) * 2]) |
                     (static_cast<uint16_t>(bf16[(r * C + c) * 2 + 1]) << 8);
        const int s = (b >> 15) & 1, e = (b >> 7) & 0xFF, m = b & 0x7F;
        float v;
        if (e == 0) v = (m / 128.0f) * std::ldexp(1.0f, -126);
        else v = (1.0f + m / 128.0f) * std::ldexp(1.0f, e - 127);
        return s ? -v : v;
    };
    // Tile scales apply per 128x128 block; the last row/col use tile [1][1].
    const float expect00 = Fp8ToFloat(fp8[0]) * s0;
    CHECK(bf16_at(0, 0) == Approx(expect00).margin(1e-3));
    const float expect_last = Fp8ToFloat(fp8[129 * C + 129]) * s3;
    CHECK(bf16_at(129, 129) == Approx(expect_last).margin(1e-3));
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cmake --build --preset linux-debug --target test_gigachat_moe_kernels`
Expected: FAIL — no target / undefined `DequantFp8Block128ToBf16Gpu`.

- [ ] **Step 3: Implement the dequant**

Create `libs/lsxcommon/src/gigachat_moe_dequant.cu` with a kernel that, for each
output element `(r,c)`, computes `tile = (r/128) * sc + (c/128)` and writes
`bf16(fp8[r*C+c] * scale[tile])`, clamping `r<sr*128` and `c<sc*128` naturally
because the grid is exactly `R*C`. Mirror the `DevGuard`/launch structure of
`glm_moe_flash_inference.cu:2124-2185` (do NOT include the glm header). Declare
the function in `gigachat_moe_convert.h`. In `libs/lsxcommon/CMakeLists.txt`
add:
```cmake
add_library(lsxcommon_gigachat_cu OBJECT
    src/gigachat_moe_dequant.cu)
set_target_properties(lsxcommon_gigachat_cu PROPERTIES
    CUDA_ARCHITECTURES "120-real;121-real" CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
target_compile_options(lsxcommon_gigachat_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;--expt-relaxed-constexpr;-fmad=false>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>)
target_include_directories(lsxcommon_gigachat_cu PRIVATE ${LSXCOMMON_CU_INCLUDES})
target_compile_definitions(lsxcommon_gigachat_cu PRIVATE ${LSXCOMMON_CU_DEFS})
target_link_libraries(lsxcommon_gigachat_cu
    "${CUDA_CUDART_LIBRARY}" absl::strings)
```
and append `lsxcommon_gigachat_cu` to the `lsxcommon` link list. In
`tests/CMakeLists.txt` add `test_gigachat_moe_kernels` (link `lsxcommon`
`Catch2::Catch2WithMain`), label `inference`.

- [ ] **Step 4: Run to green**

Run the test. Expected: PASS (partial tile handled).

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/gigachat_moe_dequant.cu \
  libs/lsxcommon/include/lsxcommon/gigachat_moe_convert.h \
  libs/lsxcommon/CMakeLists.txt tests/test_gigachat_moe_kernels.cu tests/CMakeLists.txt
git commit -m "feat: gigachat fp8 block-128 dequant (partial tiles)"
```

---

### Task 6: Converter E2E on a synthetic checkpoint

**Files:**
- Create: `tests/test_gigachat_moe_fixture.h`
- Modify: `libs/lsxcommon/src/gigachat_moe_convert.cpp`
- Modify: `tests/test_gigachat_moe_convert.cpp`
- Modify: `apps/lsxforge/src/main.cpp`

**Interfaces:**
- Consumes: classifier, parser, dequant, `gpu::quantize_tensor_gpu`,
  `DenseNvfp4Schema`, `ExpertSchema`, `PinnedSchema`, `WriteModelIndex`.
- Produces: `ConvertGigaChatModel(input_path, output_path)` writing
  `<output>/logestix/{model.index.json,pinned/*,experts/layer_N/*,mtp/*}`.

- [ ] **Step 1: Write the fixture and the failing E2E test**

Create `tests/test_gigachat_moe_fixture.h` re-exporting the neutral primitives
(`namespace gigachattest { using synthtest::FakeTensor; ... }`), plus a
`WriteGigaChatCheckpoint(dir)` that writes a `config.json` and one
`model.safetensors` shard with: `model.embed_tokens.weight` BF16, `lm_head.weight`
BF16, `model.norm.{weight,gate_up_projection.weight,gate_down_projection.weight}`
BF16, one GDN layer (`in_proj_qkvz` FP8 + F32 scale, `out_proj` FP8 + scale,
`conv1d` BF16, `A_log`, `dt_bias`, `in_proj_ba`, `norm.weight`, four layer
norms with gate projections), one MLA layer (`q_a_proj`, `q_b_proj`,
`kv_a_proj_with_mqa` with a partial tile [576,7168] scale [5,56], `kv_b_proj`,
`o_proj`, `gate_proj`, `q_a_layernorm`/`kv_a_layernorm`), one dense layer MLP,
one MoE layer with 256 experts (gate/up/down FP8 + scales), and one MTP layer
(`model.layers.40.eh_proj.weight`). Use `synthtest::WriteSafetensors`.

Append:
```cpp
TEST_CASE("gigachat converter emits the logestix-gigachat-nvfp4 layout",
          "[gigachat][convert]")
{
    const auto src = gigachattest::MakeTempDir("gc_src");
    const auto dst = gigachattest::MakeTempDir("gc_dst");
    gigachattest::WriteGigaChatCheckpoint(src);
    auto st = ConvertGigaChatModel(src.string(), dst.string());
    REQUIRE(st.ok());
    std::ifstream in(dst / "logestix" / "model.index.json");
    REQUIRE(in.good());
    std::stringstream ss; ss << in.rdbuf(); const std::string json = ss.str();
    CHECK(json.find("\"format\": \"logestix-gigachat-nvfp4\"") != std::string::npos);
    CHECK(json.find("\"full_attention_layers\": [3,7,11,15,19,23,27,31,35,39]")
          != std::string::npos);
    CHECK(json.find("\"mtp\": \"stored\"") != std::string::npos);
    CHECK(fs::exists(dst / "logestix" / "pinned" / "global.arrow"));
    CHECK(fs::exists(dst / "logestix" / "mtp" / "layer_040.arrow"));
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1 ./build/linux-debug/tests/test_gigachat_moe_convert`
Expected: FAIL — `ConvertGigaChatModel` unimplemented.

- [ ] **Step 3: Implement the converter**

Implement `ConvertGigaChatModel` mirroring `nemotron_moe_convert.cpp`:
`scan_directory` → per-shard loop → `classify_gigachat_tensor` → for FP8
linears: read `weight` + `weight_scale_inv`, call
`DequantFp8Block128ToBf16Gpu`, then `gpu::quantize_tensor_gpu(bf16, "BF16", ...)`
into `DenseNvfp4Record` (dense) or `ExpertData`→`PackedExpert` (experts); for
BF16/F32 tensors pass through into `PinnedSchema`; MTP kinds go to
`mtp/layer_NNN.arrow` with a counted record. Write `pinned/global.arrow`,
`pinned/layer_NNN.arrow`, `experts/layer_NNN/exp_EEE.arrow`,
`pinned/shared_experts.arrow`, the tokenizer arrows
(`BuildTokenizerArrowFiles(tokenizer.json, pinned)`), the VRAM block
(`EstimateVram` + `FormatVramJson`) and `model.index.json` with the format tag,
`layer_types`, `full_attention_layers`, geometry, `files`, and
`deferred: {mtp: "stored", mtp_layers: [40,41,42]}`. Any unknown tensor is a
hard error. Add `--gigachat` to `apps/lsxforge/src/main.cpp` (mirror the
`--nemotron` flag and dispatch block).

- [ ] **Step 4: Run to green**

Run: `cmake --build --preset linux-debug --target test_gigachat_moe_convert lsxforge && LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1 ./build/linux-debug/tests/test_gigachat_moe_convert`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/gigachat_moe_convert.cpp \
  tests/test_gigachat_moe_fixture.h tests/test_gigachat_moe_convert.cpp \
  apps/lsxforge/src/main.cpp
git commit -m "feat: gigachat converter (logestix-gigachat-nvfp4)"
```

---

### Task 7: Format-tag recognition in model integrity

**Files:**
- Modify: `libs/lsxcommon/src/model_integrity.cpp:261-264`
- Test: `tests/test_model_integrity.cpp`

- [ ] **Step 1: Write the failing test**

Add a case asserting that an index with
`"format": "logestix-gigachat-nvfp4"` passes the known-format gate (and an
unknown tag still fails), using the existing test helpers.

- [ ] **Step 2: Run to verify it fails**

Run: `cmake --build --preset linux-debug --target test_model_integrity && ./build/linux-debug/tests/test_model_integrity`
Expected: FAIL — unknown format tag.

- [ ] **Step 3: Add the tag**

Add `"logestix-gigachat-nvfp4"` to `kKnownFmts` in
`libs/lsxcommon/src/model_integrity.cpp`.

- [ ] **Step 4: Run to green**

Run the test. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/model_integrity.cpp tests/test_model_integrity.cpp
git commit -m "feat: recognize logestix-gigachat-nvfp4 format tag"
```

---

### Task 8: Loader (index parse, geometry gate, weight upload)

**Files:**
- Create: `libs/lsxcommon/src/gigachat_moe_load.cpp`
- Create: `tests/test_gigachat_moe_loader.cpp`
- Modify: `libs/lsxcommon/include/lsxcommon/gigachat_moe_model.h`
- Modify: `libs/lsxcommon/CMakeLists.txt`, `tests/CMakeLists.txt`

**Interfaces:**
- Produces: `GigaChat35Model::Load` internals — `ParseIndex` (format tag,
  geometry, `layer_types`, `files`), `CheckGeometry` (presets
  `GeoFieldCheck` rows from `gigachat_moe_constants.h`; synthetic geometry
  bypass via `SyntheticGeometryAllowed()`), and loading of pinned tensors.
  (Expert streaming / host mmap cache is Plan 2; Plan 1 loads and validates the
  index and pinned tensors only.)

- [ ] **Step 1: Write the failing loader test**

Create `tests/test_gigachat_moe_loader.cpp` with a synthetic converted fixture
(run `ConvertGigaChatModel` into a temp dir, then point the model at
`<dir>/logestix`). Cases: (a) a valid fixture loads and `Inspect()` reports the
tensor count; (b) a corrupted `model.index.json` (bad `format`) fails Load;
(c) a geometry mismatch fails unless `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1`.

- [ ] **Step 2: Run to verify it fails**

Run: `./build/linux-debug/tests/test_gigachat_moe_loader`
Expected: FAIL — model/loader unimplemented.

- [ ] **Step 3: Implement `Load` and `Inspect`**

Mirror `nemotron_moe_load.cpp` and `GLM`'s `ParseIndex`/`CheckGeometryFields`
usage in `ModelBase`. Read the index via `simdjson`, reject a non-gigachat
format tag, validate against the frozen passport, then load the pinned tensors.
Expert streaming/device cache is Plan 2. `Inspect()` returns the source
`SafetensorsModel` inventory.

- [ ] **Step 4: Run to green**

Run the loader test. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/gigachat_moe_load.cpp \
  libs/lsxcommon/src/gigachat_moe_model.h tests/test_gigachat_moe_loader.cpp \
  libs/lsxcommon/CMakeLists.txt tests/CMakeLists.txt
git commit -m "feat: gigachat loader + frozen-geometry gate"
```

---

### Task 9: Model class + factory registration + autograd-free `Infer` stub

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/gigachat_moe_model.h`
- Create: `libs/lsxcommon/src/gigachat_moe_model.cpp`
- Modify: `libs/lsxcommon/src/model_registry.cpp`
- Create: `tests/test_gigachat_moe_model.cpp`
- Modify: `libs/lsxcommon/CMakeLists.txt`, `tests/CMakeLists.txt`

**Interfaces:**
- Produces: `class GigaChat35Model final : public ModelBase` with `kType`,
  `Load`, `Infer` (opens `LsxFastMathScope(EffectiveFastMathMode(cfg_))`; returns
  a not-yet-implemented error until Plan 2), `SelfCheck`, `Tokenizer`,
  `BuildPromptInputIds`. Registered via `LsxModelFactory::Register`.

- [ ] **Step 1: Write the failing model test**

Create `tests/test_gigachat_moe_model.cpp`: (a) the factory resolves
`"gigachat3.5-432b-a28b-reasoning"`; (b) `Load` on a synthetic converted fixture
succeeds and `Inspect()`/`ResidentWeightBytes()` are sane; (c) `Infer` returns
`ok == false` with the Plan-2 sentinel (documents the boundary).

- [ ] **Step 2: Run to verify it fails**

Run: `./build/linux-debug/tests/test_gigachat_moe_model`
Expected: FAIL — no `GigaChat35Model` / not registered.

- [ ] **Step 3: Implement and register**

Implement the class on `ModelBase` (see `nemotron_moe_model.cpp`), add
`GigaChat35Model::kType = "gigachat3.5-432b-a28b-reasoning"`, and add
`LsxModelFactory::Register<GigaChat35Model>();` to `model_registry.cpp`.
`Infer` must contain `LsxFastMathScope` and `EffectiveFastMathMode(cfg_)`
(guard requirement) even though it returns not-implemented in this plan.
Add `"gigachat_moe_model.cpp"` to the `consumers` list in
`tests/test_model_independence_guard.cpp` (the file now exists).

- [ ] **Step 4: Run to green**

Run the model test and the independence guard. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/gigachat_moe_model.h \
  libs/lsxcommon/src/gigachat_moe_model.cpp \
  libs/lsxcommon/src/model_registry.cpp tests/test_gigachat_moe_model.cpp \
  libs/lsxcommon/CMakeLists.txt tests/CMakeLists.txt
git commit -m "feat: GigaChat35Model + factory registration"
```

---

### Task 10: Tokenizer bridge + prompt framing

**Files:**
- Modify: `libs/lsxcommon/src/gigachat_moe_model.cpp` (`BuildPromptInputIds`)
- Create/modify: tokenizer test in `tests/test_gigachat_moe_model.cpp`
- Modify: `libs/lsxcommon/src/gigachat_moe_convert.cpp` (tokenizer arrows in E2E)

**Interfaces:**
- Produces: `bool GigaChat35Model::BuildPromptInputIds(const ArrowTokenizerDb &tok, std::string_view raw, std::vector<int32_t> &out)` — chatml-v5 single-turn with the default Russian system turn and a prefilled empty `<think></think>`; special tokens resolved via `arrow_tokenizer_lookup`, system text via `arrow_tokenizer_encode`.

- [ ] **Step 1: Write the failing tokenizer test**

Add a case that builds the arrow vocab/merges from a synthetic `tokenizer.json`
containing the special tokens `<s>`, `</s>`, `<|role_sep|>\n`,
`<|message_sep|>\n\n`, `<think>`, `</think>`, then asserts: `tok.bos_id == 1`,
`tok.eos_id == 2`, every special token resolves via `arrow_tokenizer_lookup`, and
`BuildPromptInputIds(tok, "The capital of France is", ids)` starts with BOS and
ends with the `<think>`/`</think>` sentinels.

- [ ] **Step 2: Run to verify it fails**

Run: `./build/linux-debug/tests/test_gigachat_moe_model`
Expected: FAIL — framing not implemented.

- [ ] **Step 3: Implement the framing**

Implement `BuildPromptInputIds` (and wire `BuildTokenizerArrowFiles` into the
converter E2E, asserting the tokenizer arrows exist). Prefer a dedicated
`gigachat::BuildPromptInputIds` function in `gigachat_moe_convert.{h,cpp}` owned
by the family (do not edit `glm`/`gemma` framing namespaces).

- [ ] **Step 4: Run to green**

Run the tokenizer test and the full `fast` + `convert` suites.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/gigachat_moe_model.cpp \
  libs/lsxcommon/src/gigachat_moe_convert.cpp \
  libs/lsxcommon/include/lsxcommon/gigachat_moe_convert.h \
  tests/test_gigachat_moe_model.cpp
git commit -m "feat: gigachat tokenizer bridge + chatml-v5 framing"
```

---

### Task 11: Suite registry + labels + AGENTS structure

**Files:**
- Modify: `tests/test_model_suite_guard.cpp` (`Suites()` row)
- Modify: `AGENTS.md` (Structure section)
- Verify: `tests/test_ctest_labels_guard.cpp`

**Interfaces:**
- Produces: the `gigachat3.5-432b-a28b-reasoning` suite row
  (`convert` → `test_gigachat_moe_convert`; `load` → `test_gigachat_moe_loader`;
  `model` → `test_gigachat_moe_model`; `kernels` → `test_gigachat_moe_kernels`;
  `selfcheck` → `test_gigachat_moe_selfcheck` added in Plan 3), plus all labels.

- [ ] **Step 1: Add the row and labels; run the guards**

Add the suite row with the stages that exist now (`convert`, `load`, `model`,
`kernels`); Plan 3 appends `selfcheck` when that target exists. Also extend the
`model_stage_re` regex in `test_model_suite_guard.cpp` from
`(glm|qwen|deepseek|gemma|kimi|nemotron)` to include `gigachat`, so the vault-path
discipline applies to `test_gigachat_moe_model.cpp`. Ensure every new test
carries a label. Run:
`./build/linux-debug/tests/test_model_suite_guard && ./build/linux-debug/tests/test_ctest_labels_guard`
Expected: PASS.

- [ ] **Step 2: Update `AGENTS.md`**

Add the `gigachat` family files to the Structure section.

- [ ] **Step 3: Commit**

```bash
git add tests/test_model_suite_guard.cpp AGENTS.md
git commit -m "test/docs: gigachat suite registry + AGENTS structure"
```

---

## Plan 1 verification

```bash
cmake --preset linux-debug && cmake --build --preset linux-debug
ctest --preset linux-debug                     # fast
ctest --preset linux-debug-convert             # converter sweep
cmake -DLSX_PRESET=linux-debug -DLSX_LIST=ON -P tools/test_router.cmake
```
The four guard tests plus `test_gigachat_moe_{math,convert,loader,model,kernels}`
must be green. Real-model conversion is NOT run here (operator step, Plan 3).

## Out of scope (Plan 2 / Plan 3)

- Plan 2: `gigachat_moe_{norm_kernels,kernels,inference_attn,inference_moe,inference_pipeline}.cu`, `gigachat_moe_device_math.h`, `gigachat_moe_expert_cache.{h,cpp}`, forward parity tests.
- Plan 3: `test_gigachat_moe_selfcheck.cpp`, `test_perf_baseline_gigachat` + `perf_baselines.json`, passport/`LsxTokenTranslator`.
