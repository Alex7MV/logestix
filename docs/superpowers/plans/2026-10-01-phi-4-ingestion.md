# phi-4 Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **This repo forbids Task subagents (AGENTS.md "Autonomous Mode"). Execute INLINE** with superpowers:executing-plans — one task at a time, run the verification, commit, move on.

**Goal:** Ingest `microsoft/phi-4` (`Phi3ForCausalLM`) as a new dense `phi` family into logestix, covering the full 8-stage Model Ingestion Protocol plus the tool bridge, the FSI/CODE/DET parity gates and a perf baseline.

**Architecture:** phi-4 is architecturally identical to `granite-4.2-30b` (dense decoder-only, GQA + full-dim RoPE + SwiGLU), and its attention/MLP projections are already fused (`qkv_proj`, `gate_up_proj`) — exactly the decode layout the granite loader expects. The family is a mechanical specialization of `granite_dense_*` with new geometry, a computed `1/sqrt(head_dim)` attention scale, phi's ChatML framing, and the phi special tokens.

**Tech Stack:** C++20/C23, CUDA 13.4 (sm_120/121), CMake presets, Arrow, NVFP4, Catch2 v3.16.0, spdlog, abseil (host-only), simdjson.

## Global Constraints

- Target arch: Blackwell only — `CUDA_ARCHITECTURES "120-real;121-real"`. No sm_90/sm_80.
- No backward compatibility.
- Family token `phi`, type `dense`; file names `phi_dense_<role>.{h,cpp,cu}`. `kType = "phi-4"`.
- Format tag `logestix-phi`. Index model name `Phi-4`.
- Frozen geometry: 40 layers, hidden 5120, 40 Q heads, 10 KV heads, head_dim 128, intermediate 17920, vocab 100352, eps 1e-5, rope_theta 250000, attention scale `1/sqrt(128)`.
- Special tokens: bos 100257, eos 100265, pad 100349.
- NVFP4 only via the model-neutral `lsx_nvfp4` layer; norms raw FP16; one stream; no `cudaDeviceSynchronize` per token; no `atomicAdd` feeding `final_logits`; shared-mem sandbox.
- No abseil containers/streams in headers reachable from `.cu`/`.cuh`.
- Every new test gets its ctest label in the same commit; selfcheck/fsi/code/det targets get `SKIP_RETURN_CODE 4`.
- Commit messages: English, `feat:`/`test:`/`build:`/`docs:` prefixes. Commit only when a task's steps say so.
- Build/verify: `cmake --preset linux-debug && cmake --build --preset linux-debug`; `ctest --preset linux-debug` (fast); `ctest --preset linux-debug-convert`; `ctest --preset linux-debug-inference`.

**Global rename table** used by every "copy + specialize" step below. Apply literally:

| from | to |
|---|---|
| `granite` (lowercase; paths, namespaces, log tags, `logestix-granite`) | `phi` |
| `Granite` (types) | `Phi` |
| `Granite42_30BModel` | `Phi4Model` |
| `grantest` | `phitest` |
| `"granite-4.2-30b"` | `"phi-4"` |
| `"Granite-4.2-30B"` | `"Phi-4"` |
| `"GraniteForCausalLM"` | `"Phi3ForCausalLM"` |
| `attention_multiplier` (topology field/JSON key) | `attention_scale` |
| `kAttentionMultiplier` | `kAttentionScale` |

---

### Task 1: Family passport + naming/independence guards

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/phi_dense_constants.h`
- Modify: `tests/test_code_naming_guard.cpp`
- Modify: `tests/test_model_independence_guard.cpp`

**Interfaces:**
- Produces: `lsxcommon::phi::PhiTopology`, `lsxcommon::phi::frozen::{kHiddenSize=5120, kNumLayers=40, kNumHeads=40, kNumKVHeads=10, kHeadDim=128, kIntermediateSize=17920, kVocabSize=100352, kRmsNormEps=1e-5, kRopeTheta=250000.0, kAttentionScale=0.08838834764831845f, kBosId=100257, kEosId=100265, kPadId=100349}`.

- [ ] **Step 1: Write the failing guard checks**

In `tests/test_code_naming_guard.cpp`, add `phi` to the `fam` regex in `SchemeRegex()` (line 27) and to `FamilyStems()`, and add a new `TEST_CASE`:

```cpp
TEST_CASE("naming guard accepts phi_dense_* stems", "[naming][guard]")
{
    auto m = [](const std::string &s) {
        return std::regex_match(s, SchemeRegex());
    };
    CHECK(m("phi_dense_constants"));
    CHECK(m("phi_dense_convert"));
    CHECK(m("phi_dense_load"));
    CHECK(m("phi_dense_model"));
    CHECK(m("phi_dense_inference"));
}
```

In `tests/test_model_independence_guard.cpp`, add to `kPassports`:

```cpp
    {"phi-4", "../include/lsxcommon/phi_dense_constants.h",
     "lsxcommon::phi::frozen"},
```

and to the `owner` map in "each passport defines its geometry in its own namespace or class":

```cpp
        {"../include/lsxcommon/phi_dense_constants.h", "namespace frozen {"},
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cmake --build --preset linux-debug --target test_code_naming_guard test_model_independence_guard && ctest --preset linux-debug -R "test_code_naming_guard|test_model_independence_guard"`
Expected: FAIL (missing `phi_dense_constants.h`; `phi` family not accepted before the regex edit).

- [ ] **Step 3: Create the passport**

Create `libs/lsxcommon/include/lsxcommon/phi_dense_constants.h`:

```cpp
#pragma once

// phi-4 (Microsoft) frozen geometry passport (Model Ingestion Protocol
// step 3). Phi3ForCausalLM is a dense decoder-only transformer: RMSNorm, GQA
// attention with full-dim RoPE, SwiGLU MLP. No MoE, no QK-norm, no biases,
// no config-level attention/residual multipliers. This header is a LEAF: no
// other model's passport may be included (model-independence invariant). It
// is device-safe (no abseil containers/streams).

#include <cstddef>
#include <cstdint>
#include <string>

#include "lsxcommon/vram_estimate.h"

namespace lsxcommon {
namespace phi {

// Loaded/converted topology of a phi checkpoint (geometry + special tokens +
// the mandatory conversion-time VRAM profile). Device-safe.
struct PhiTopology {
    int num_layers = 0;
    int hidden_size = 0;
    int intermediate_size = 0;
    int vocab_size = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int head_dim = 0;
    double rms_norm_eps = 1e-5;
    double rope_theta = 250000.0;
    // phi applies the textbook 1/sqrt(head_dim) attention score scale (no
    // config multiplier). Pinned at conversion time.
    double attention_scale = 0.08838834764831845;
    int bos_id = -1;
    int eos_id = -1;
    int pad_id = -1;
    int total_pinned_tensors = 0;
    VramProfile vram;        // conversion-time VRAM requirements (mandatory)
    std::string error_msg;   // non-empty when index.json is invalid
};

namespace frozen {

// ── Frozen compile-time geometry (config.json, 2026-10-01) ─────────────────
inline constexpr int kHiddenSize = 5120;
inline constexpr int kNumLayers = 40;
inline constexpr int kNumHeads = 40;
inline constexpr int kNumKVHeads = 10;
inline constexpr int kHeadDim = 128;
inline constexpr int kIntermediateSize = 17920;
inline constexpr int kVocabSize = 100352;

inline constexpr double kRmsNormEps = 1e-5;
inline constexpr double kRopeTheta = 250000.0;
// 1 / sqrt(kHeadDim).
inline constexpr float kAttentionScale = 0.08838834764831845f;

// Special tokens (config.json): bos <|endoftext|> / eos <|im_end|> /
// pad <|dummy_85|>.
inline constexpr int32_t kBosId = 100257;
inline constexpr int32_t kEosId = 100265;
inline constexpr int32_t kPadId = 100349;

// Compile-time geometry invariants.
static_assert(kHiddenSize == kNumHeads * kHeadDim,
              "hidden_size must equal num_attention_heads * head_dim");
static_assert(kNumKVHeads > 0 && kNumKVHeads <= kNumHeads,
              "num_key_value_heads must be in (0, num_attention_heads]");
static_assert(kHiddenSize % kNumKVHeads == 0,
              "hidden_size must be divisible by num_key_value_heads");
static_assert(kNumLayers > 0 && kIntermediateSize > 0 && kVocabSize > 0,
              "positive phi geometry required");

} // namespace frozen
} // namespace phi
} // namespace lsxcommon
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cmake --build --preset linux-debug --target test_code_naming_guard test_model_independence_guard && ctest --preset linux-debug -R "test_code_naming_guard|test_model_independence_guard"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/phi_dense_constants.h \
        tests/test_code_naming_guard.cpp tests/test_model_independence_guard.cpp
git commit -m "feat: add phi-4 frozen passport and guard registrations"
```

---

### Task 2: Converter + synthetic fixture + convert test

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/phi_dense_convert.h`
- Create: `libs/lsxcommon/src/phi_dense_convert.cpp`
- Create: `tests/test_phi_dense_fixture.h`
- Create: `tests/test_phi_dense_convert.cpp`
- Modify: `libs/lsxcommon/CMakeLists.txt` (add `src/phi_dense_convert.cpp`)
- Modify: `tests/CMakeLists.txt` (convert target block after the granite block)

**Interfaces:**
- Produces: `lsxcommon::phi::classify_phi_tensor`, `phi_tensor_is_nvfp4`, `load_phi_topology`, `WritePhiModelIndex`, `ConvertPhiModel(input, output)`, `RecomputePhiVram`.
- Consumes: `phi::PhiTopology` (Task 1).

- [ ] **Step 1: Write the failing convert test**

Create `tests/test_phi_dense_fixture.h` (family fixture; re-exports the neutral primitives):

```cpp
#pragma once

// Shared fixture builder for phi-4 converter / loader / kernel / model tests.
// Tensor-writing primitives live in the family-neutral test_synth_checkpoint.h
// (suite-autonomy invariant) and are re-exported under phitest.

#include "test_synth_checkpoint.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace phitest {

using synthtest::FakeTensor;
using synthtest::MakeBf16;
using synthtest::MakeBf16Values;
using synthtest::MakeF32Values;
using synthtest::TensorNamesIn;
using synthtest::WriteSafetensors;

// Synthetic geometry — tiny so CPU/CUDA references run fast. Mirror the real
// model proportionally (GQA 4:1, head_dim = hidden / heads) and store the
// projections PRE-FUSED exactly like the real phi-4 checkpoint.
struct PhiTestConfig {
    int hidden_size = 64;
    int intermediate_size = 128;
    int vocab_size = 64;
    int num_layers = 2;
    int num_attention_heads = 4;
    int num_key_value_heads = 2;
    int head_dim = 16;  // hidden / num_attention_heads

    double rms_norm_eps = 1e-5;
    double rope_theta = 10000.0;
    double attention_scale = 0.25;  // 1 / sqrt(head_dim)

    int bos_id = 0;
    int eos_id = 1;
    int pad_id = 2;
};

inline const PhiTestConfig &TestCfg()
{
    static PhiTestConfig cfg;
    return cfg;
}

inline int QProjOut() { return TestCfg().num_attention_heads * TestCfg().head_dim; }
inline int KVProjOut() { return TestCfg().num_key_value_heads * TestCfg().head_dim; }

inline std::vector<FakeTensor> BuildSyntheticPhiModel()
{
    const auto &c = TestCfg();
    const int H = c.hidden_size;
    const int I = c.intermediate_size;
    const int V = c.vocab_size;
    const int QKV = QProjOut() + 2 * KVProjOut();

    std::vector<FakeTensor> t;

    t.push_back({"model.embed_tokens.weight", "BF16", {V, H}, MakeBf16(V * H)});
    t.push_back({"lm_head.weight", "BF16", {V, H}, MakeBf16(V * H)});
    t.push_back({"model.norm.weight", "BF16", {H},
                 MakeBf16Values(H, [](int64_t i) { return (float)i * 0.01f; })});

    for (int L = 0; L < c.num_layers; ++L) {
        std::string lp = "model.layers." + std::to_string(L) + ".";

        t.push_back({lp + "input_layernorm.weight", "BF16", {H}, MakeBf16(H)});
        t.push_back(
            {lp + "post_attention_layernorm.weight", "BF16", {H}, MakeBf16(H)});

        auto sa = lp + "self_attn.";
        // Fused qkv (q | k | v), like the real checkpoint.
        t.push_back({sa + "qkv_proj.weight", "BF16", {QKV, H},
                     MakeBf16(QKV * H)});
        t.push_back({sa + "o_proj.weight", "BF16", {H, QProjOut()},
                     MakeBf16(QProjOut() * H)});

        // Fused gate_up (gate | up), like the real checkpoint.
        t.push_back({lp + "mlp.gate_up_proj.weight", "BF16", {2 * I, H},
                     MakeBf16(2 * I * H)});
        t.push_back({lp + "mlp.down_proj.weight", "BF16", {H, I},
                     MakeBf16(I * H)});
    }

    return t;
}

inline void WriteConfig(const std::filesystem::path &dir)
{
    const auto &c = TestCfg();
    std::string json =
        "{\"architectures\":[\"Phi3ForCausalLM\"],\"model_type\":\"phi3\","
        "\"hidden_size\":" + std::to_string(c.hidden_size) +
        ",\"intermediate_size\":" + std::to_string(c.intermediate_size) +
        ",\"num_hidden_layers\":" + std::to_string(c.num_layers) +
        ",\"vocab_size\":" + std::to_string(c.vocab_size) +
        ",\"num_attention_heads\":" + std::to_string(c.num_attention_heads) +
        ",\"num_key_value_heads\":" + std::to_string(c.num_key_value_heads) +
        ",\"head_dim\":" + std::to_string(c.head_dim) +
        ",\"rms_norm_eps\":" + std::to_string(c.rms_norm_eps) +
        ",\"rope_theta\":" + std::to_string(c.rope_theta) +
        ",\"tie_word_embeddings\":false,\"hidden_act\":\"silu\","
        "\"attention_bias\":false,\"attention_dropout\":0.0,"
        "\"bos_token_id\":" + std::to_string(c.bos_id) +
        ",\"eos_token_id\":" + std::to_string(c.eos_id) +
        ",\"pad_token_id\":" + std::to_string(c.pad_id) +
        ",\"max_position_embeddings\":16384}";
    std::ofstream f(dir / "config.json");
    f << json;
}

} // namespace phitest
```

Create `tests/test_phi_dense_convert.cpp`:

```cpp
#include <catch2/catch_all.hpp>

#include <lsxcommon/phi_dense_convert.h>
#include <lsxcommon/schema.h>

#include "test_phi_dense_fixture.h"

#include <arrow/io/file.h>
#include <cstdint>
#include <filesystem>

using namespace lsxcommon::phi;

namespace {

std::vector<std::string> DenseNvfp4NamesIn(const std::filesystem::path &p)
{
    auto batch = lsxcommon::ReadRecordBatchFromFile(p);
    if (!batch.ok()) return {};
    auto recs = lsxcommon::DenseNvfp4Schema::UnpackBatch(*batch);
    if (!recs.ok()) return {};
    std::vector<std::string> names;
    for (auto &r : *recs) names.push_back(r.name);
    return names;
}

bool Contains(const std::vector<std::string> &v, const std::string &s)
{
    return std::find(v.begin(), v.end(), s) != v.end();
}

} // namespace

TEST_CASE("phi tensor name classifier", "[phi][convert]")
{
    auto check = [](const std::string &name, TensorKind expected,
                    int layer = -1) {
        auto p = classify_phi_tensor(name);
        CHECK(p.kind == expected);
        if (layer >= 0) CHECK(p.layer == layer);
    };
    check("model.embed_tokens.weight", TensorKind::Embed);
    check("lm_head.weight", TensorKind::LmHead);
    check("model.norm.weight", TensorKind::FinalNorm);
    check("model.layers.0.input_layernorm.weight", TensorKind::InputNorm, 0);
    check("model.layers.7.post_attention_layernorm.weight",
          TensorKind::PostNorm, 7);
    check("model.layers.3.self_attn.qkv_proj.weight", TensorKind::AttnQKV, 3);
    check("model.layers.3.self_attn.o_proj.weight", TensorKind::AttnO, 3);
    check("model.layers.1.mlp.gate_up_proj.weight", TensorKind::MlpGateUp, 1);
    check("model.layers.1.mlp.down_proj.weight", TensorKind::MlpDown, 1);
    check("model.layers.0.self_attn.q_norm.weight", TensorKind::Unknown);
    check("unknown_tensor", TensorKind::Unknown);

    CHECK(phi_tensor_is_nvfp4(TensorKind::AttnQKV));
    CHECK(phi_tensor_is_nvfp4(TensorKind::MlpDown));
    CHECK(phi_tensor_is_nvfp4(TensorKind::Embed));
    CHECK_FALSE(phi_tensor_is_nvfp4(TensorKind::InputNorm));
    CHECK_FALSE(phi_tensor_is_nvfp4(TensorKind::FinalNorm));
}

TEST_CASE("phi converter builds logestix-phi index and arrow layout",
          "[phi][convert]")
{
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "phi_conv_test";
    remove_all(tmp);
    create_directories(tmp);

    auto tensors = phitest::BuildSyntheticPhiModel();
    phitest::WriteSafetensors(tmp, "model.safetensors", tensors);
    phitest::WriteConfig(tmp);
    copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
              tmp / "tokenizer.json");

    auto st = ConvertPhiModel(tmp.string(), tmp.string());
    REQUIRE(st.ok());

    auto top = load_phi_topology(
        (tmp / "logestix" / "model.index.json").string());
    CHECK(top.error_msg.empty());
    CHECK(top.num_layers == 2);
    CHECK(top.hidden_size == 64);
    CHECK(top.intermediate_size == 128);
    CHECK(top.vocab_size == 64);
    CHECK(top.num_attention_heads == 4);
    CHECK(top.num_key_value_heads == 2);
    CHECK(top.head_dim == 16);
    CHECK(top.bos_id == 0);
    CHECK(top.eos_id == 1);
    CHECK(top.attention_scale > 0.0);

    CHECK(top.vram.min_bytes > 0);
    CHECK(top.vram.optimal_bytes >= top.vram.min_bytes);
    CHECK(top.vram.max_bytes >= top.vram.optimal_bytes);
    CHECK(top.vram.context_tokens > 0);

    CHECK(exists(tmp / "logestix" / "pinned" / "global.arrow"));
    CHECK(exists(tmp / "logestix" / "pinned" / "global_nvfp4.arrow"));
    CHECK(exists(tmp / "logestix" / "pinned" / "layer_0.arrow"));
    CHECK(exists(tmp / "logestix" / "pinned" / "layer_0_nvfp4.arrow"));
    CHECK(exists(tmp / "logestix" / "pinned" / "layer_1_nvfp4.arrow"));

    auto names = DenseNvfp4NamesIn(tmp / "logestix" / "pinned" /
                                   "layer_0_nvfp4.arrow");
    CHECK(Contains(names, "model.layers.0.self_attn.qkv_proj.weight"));
    CHECK(Contains(names, "model.layers.0.self_attn.o_proj.weight"));
    CHECK(Contains(names, "model.layers.0.mlp.gate_up_proj.weight"));
    CHECK(Contains(names, "model.layers.0.mlp.down_proj.weight"));

    auto gnames = DenseNvfp4NamesIn(tmp / "logestix" / "pinned" /
                                    "global_nvfp4.arrow");
    CHECK(Contains(gnames, "model.embed_tokens.weight"));
    CHECK(Contains(gnames, "lm_head.weight"));

    auto rnames = synthtest::TensorNamesIn(tmp / "logestix" / "pinned" /
                                           "global.arrow");
    CHECK(Contains(rnames, "model.norm.weight"));

    remove_all(tmp);
}
```

- [ ] **Step 2: Register the test target**

In `tests/CMakeLists.txt`, after the granite convert block (`set_tests_properties(test_granite_dense_convert PROPERTIES LABELS convert)`), add:

```cmake
# ── phi-4 converter test (classifier + index + arrow layout) ──
add_executable(test_phi_dense_convert test_phi_dense_convert.cpp)
target_compile_definitions(test_phi_dense_convert PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_phi_dense_convert PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_phi_dense_convert POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_phi_dense_convert>"
    )
    add_custom_command(TARGET test_phi_dense_convert POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_phi_dense_convert>"
    )
endif()

add_test(NAME test_phi_dense_convert COMMAND test_phi_dense_convert)
set_tests_properties(test_phi_dense_convert PROPERTIES
    TIMEOUT 120)
set_tests_properties(test_phi_dense_convert PROPERTIES LABELS convert)
```

- [ ] **Step 3: Run to verify it fails**

Run: `cmake --preset linux-debug && cmake --build --preset linux-debug --target test_phi_dense_convert`
Expected: FAIL to compile (`phi_dense_convert.h` missing).

- [ ] **Step 4: Create the converter header**

Create `libs/lsxcommon/include/lsxcommon/phi_dense_convert.h`. Copy `granite_dense_convert.h`, apply the Global rename table. Keep `#include "lsxcommon/phi_dense_constants.h"`. Replace the `TensorKind` enum with the phi taxonomy (the source is pre-fused):

```cpp
enum class TensorKind {
    Unknown,
    Embed,       // model.embed_tokens.weight
    LmHead,      // lm_head.weight
    FinalNorm,   // model.norm.weight
    InputNorm,   // model.layers.N.input_layernorm.weight
    PostNorm,    // model.layers.N.post_attention_layernorm.weight
    AttnQKV,     // model.layers.N.self_attn.qkv_proj.weight (fused q|k|v)
    AttnO,       // model.layers.N.self_attn.o_proj.weight
    MlpGateUp,   // model.layers.N.mlp.gate_up_proj.weight (fused gate|up)
    MlpDown,     // model.layers.N.mlp.down_proj.weight
};
```

Declarations: `classify_phi_tensor`, `phi_tensor_is_nvfp4`, `load_phi_topology`, `WritePhiModelIndex`, `ConvertPhiModel`, `RecomputePhiVram`.

- [ ] **Step 5: Create the converter implementation**

Create `libs/lsxcommon/src/phi_dense_convert.cpp`. Copy `granite_dense_convert.cpp` and apply the Global rename table, then make these semantic edits:

1. `read_phi_config`: read the flat phi3 keys `hidden_size`, `intermediate_size`, `vocab_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `rms_norm_eps`, `rope_theta`, `bos_token_id`, `eos_token_id`, `pad_token_id`. Remove the `rope_parameters` and `attention_multiplier` reads. After parsing, compute:
   ```cpp
   if (t.head_dim == 0 && t.num_attention_heads > 0)
       t.head_dim = t.hidden_size / t.num_attention_heads;
   t.attention_scale = 1.0 / std::sqrt(static_cast<double>(t.head_dim));
   ```
   (`#include <cmath>`.)
2. `classify_phi_tensor`: emit `AttnQKV` for `self_attn.qkv_proj.weight`, `AttnO` for `o_proj`, `MlpGateUp` for `mlp.gate_up_proj.weight`, `MlpDown` for `mlp.down_proj.weight`, plus norms/embed/head. Skip unknown names.
3. `phi_tensor_is_nvfp4`: true for Embed/LmHead/AttnQKV/AttnO/MlpGateUp/MlpDown; false for norms.
4. `generate_phi_expected_names`: expected list is embed, lm_head, norm, then per layer `input_layernorm.weight`, `post_attention_layernorm.weight`, `self_attn.qkv_proj.weight`, `self_attn.o_proj.weight`, `mlp.gate_up_proj.weight`, `mlp.down_proj.weight`.
5. `phi_derive_actual_shapes`: for `AttnQKV`, set `hidden = shape[1]`; for `MlpDown`, set `intermediate = shape[1]`; for embed/head vocab/hidden as in granite.
6. Conversion loop: since qkv and gate_up are already fused, there is **no** `pending_qkv`/`pending_gate` fusion step. Every NVFP4 tensor is quantized directly:
   ```cpp
   DenseNvfp4Record r;
   r.name = t.name;
   r.shape = t.shape;
   r.dtype = t.dtype;
   if (!gpu::quantize_tensor_gpu(data.data(), t.numel(), t.dtype, nullptr,
                                 t.shape, {}, r.packed, r.scales,
                                 r.global_scale))
       r.packed.clear();
   nvfp4_by_layer[layer].push_back(std::move(r));
   ```
   Delete `PendingGate`, `PendingQKV`, `fused_gate_up_name` and the q/k/v interception. Keep the embed/lm_head global path and the final-norm raw path.
7. `load_phi_topology`: read the JSON keys `num_layers`, `hidden_size`, `intermediate_size`, `vocab_size`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `rms_norm_eps`, `rope_theta`, `attention_scale`, `bos_id`, `eos_id`, `pad_id`, `total_pinned_tensors`, then `ParseVramBlock`.
8. `WritePhiModelIndex`: emit `"format": "logestix-phi"`, `"model": "Phi-4"`, and `"attention_scale": <value>` (no multiplier/embedding blocks).
9. `RecomputePhiVram`: same as granite with the phi classifier.

In `libs/lsxcommon/CMakeLists.txt`, after `src/granite_dense_convert.cpp` add:

```cmake
    src/phi_dense_convert.cpp
```

- [ ] **Step 6: Run to verify it passes**

Run: `cmake --preset linux-debug && cmake --build --preset linux-debug --target test_phi_dense_convert && ctest --preset linux-debug-convert -R test_phi_dense_convert`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/phi_dense_convert.h \
        libs/lsxcommon/src/phi_dense_convert.cpp \
        libs/lsxcommon/CMakeLists.txt \
        tests/test_phi_dense_fixture.h tests/test_phi_dense_convert.cpp \
        tests/CMakeLists.txt
git commit -m "feat: phi-4 lsxforge converter + synthetic convert test"
```

---

### Task 3: Loader + loader test

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/phi_dense_load.h`
- Create: `libs/lsxcommon/src/phi_dense_load.cpp`
- Create: `tests/test_phi_dense_loader.cpp`
- Modify: `libs/lsxcommon/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Produces: `lsxcommon::phi::{PhiLayerWeights, PhiPinnedWeights, LoadPhiPinnedWeights}` with the same field names as granite (`qkv`, `q`, `k`, `v`, `o`, `mlp_gate`, `mlp_up`, `mlp_down`, `mlp_gate_up`, `d_input_norm`, `d_post_norm`).
- Consumes: `PhiTopology` (Task 1); `ConvertPhiModel` (Task 2) for the test fixture.

- [ ] **Step 1: Write the failing loader test**

Create `tests/test_phi_dense_loader.cpp` by copying `tests/test_granite_dense_loader.cpp` and applying the Global rename table (`granite`→`phi`, `Granite`→`Phi`, `grantest`→`phitest`). Then update the fixture calls to `phitest::BuildSyntheticPhiModel()` / `phitest::WriteConfig()` and the `#include "test_phi_dense_fixture.h"`, and change `LoadGranitePinnedWeights`→`LoadPhiPinnedWeights`. Run it (Step 2) before creating the loader.

- [ ] **Step 2: Register the target + run to verify it fails**

Add the loader target to `tests/CMakeLists.txt` (copy the granite loader block, rename, `LABELS inference`). Then:

Run: `cmake --preset linux-debug && cmake --build --preset linux-debug --target test_phi_dense_loader`
Expected: FAIL to compile (`phi_dense_load.h` missing).

- [ ] **Step 3: Create the loader header**

Copy `granite_dense_load.h` → `libs/lsxcommon/include/lsxcommon/phi_dense_load.h`, apply the rename table (types become `PhiLayerWeights`, `PhiPinnedWeights`; function `LoadPhiPinnedWeights`). Keep all fields (`qkv`, `q`, `k`, `v`, `o`, `mlp_gate`, `mlp_up`, `mlp_down`, `mlp_gate_up`, `d_input_norm`, `d_post_norm`, arena).

- [ ] **Step 4: Create the loader implementation**

Copy `granite_dense_load.cpp` → `libs/lsxcommon/src/phi_dense_load.cpp`, apply the rename table, then in `LoadLayer`: delete the separate-q/k/v fallback branch (lines ~102-111) and the `lw.q.d_packed && lw.k.d_packed && lw.v.d_packed` alternative (phi is always fused). Keep the fused `qkv` split and the fused `gate_up` split verbatim. Add `src/phi_dense_load.cpp` to `libs/lsxcommon/CMakeLists.txt` after `src/granite_dense_load.cpp`.

- [ ] **Step 5: Run to verify it passes**

Run: `cmake --preset linux-debug && cmake --build --preset linux-debug --target test_phi_dense_loader && ctest --preset linux-debug-inference -R test_phi_dense_loader`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/phi_dense_load.h \
        libs/lsxcommon/src/phi_dense_load.cpp libs/lsxcommon/CMakeLists.txt \
        tests/test_phi_dense_loader.cpp tests/CMakeLists.txt
git commit -m "feat: phi-4 pinned NVFP4 loader + loader test"
```

---

### Task 4: Model class + registration + model test

**Files:**
- Create: `libs/lsxcommon/src/phi_dense_model.h`
- Create: `libs/lsxcommon/src/phi_dense_model.cpp`
- Create: `tests/test_phi_dense_model.cpp`
- Modify: `libs/lsxcommon/src/model_registry.cpp`
- Modify: `libs/lsxcommon/CMakeLists.txt`
- Modify: `tests/test_model_independence_guard.cpp` (consumer list)
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Produces: `lsxcommon::Phi4Model final : ModelBase`, `kType = "phi-4"`.
- Consumes: `PhiPinnedWeights`, `LoadPhiPinnedWeights` (Task 3).

- [ ] **Step 1: Write the failing model test**

Copy `tests/test_granite_dense_model.cpp` → `tests/test_phi_dense_model.cpp`, apply the rename table, set the expected `Type() == "phi-4"` and the synthetic fixture calls to `phitest::`. In `tests/test_model_independence_guard.cpp`, add `"phi_dense_model.cpp",  // phi-4` to the `consumers` vector.

- [ ] **Step 2: Register + run to verify it fails**

Add the model target to `tests/CMakeLists.txt` (copy granite model block, rename, `ENVIRONMENT "LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1"`, `LABELS inference`).

Run: `cmake --preset linux-debug && cmake --build --preset linux-debug --target test_phi_dense_model`
Expected: FAIL to compile (`phi_dense_model.h` missing).

- [ ] **Step 3: Create the model class**

Copy `granite_dense_model.h` → `libs/lsxcommon/src/phi_dense_model.h`, apply the rename table. Rename the class `Granite42_30BModel` → `Phi4Model`; clean up the header comment to describe phi-4 (40L / 5120H / 40Q / 10KV / 17920I, SwiGLU, full RoPE, `1/sqrt(head_dim)` scale). Keep the static constexpr members mirroring `phi::frozen`.

- [ ] **Step 4: Create the model implementation**

Copy `granite_dense_model.cpp` → `libs/lsxcommon/src/phi_dense_model.cpp`, apply the rename table; set `kType = "phi-4"`. Keep `Load`, `ValidateGeometry` (fields vs `phi::frozen`), `Infer` with `const lsx_nvfp4::LsxFastMathScope fm_scope(EffectiveFastMathMode(cfg_));`, and `SelfCheck` → `runtime::RunParisSelfCheck`.

In `libs/lsxcommon/src/model_registry.cpp`:
- add `#include "phi_dense_model.h"` (alphabetical with the other quoted family headers),
- add `LsxModelFactory::Register<Phi4Model>();` after `Register<Granite42_30BModel>();`.

In `libs/lsxcommon/CMakeLists.txt`, add `src/phi_dense_model.cpp` after `src/granite_dense_model.cpp`.

- [ ] **Step 5: Run to verify it passes**

Run: `cmake --preset linux-debug && cmake --build --preset linux-debug --target test_phi_dense_model test_model_independence_guard && ctest --preset linux-debug -R "test_phi_dense_model|test_model_independence_guard"`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add libs/lsxcommon/src/phi_dense_model.h libs/lsxcommon/src/phi_dense_model.cpp \
        libs/lsxcommon/src/model_registry.cpp libs/lsxcommon/CMakeLists.txt \
        tests/test_phi_dense_model.cpp tests/test_model_independence_guard.cpp \
        tests/CMakeLists.txt
git commit -m "feat: phi-4 IModel + factory registration"
```

---

### Task 5: CUDA inference + kernel parity test

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/phi_dense_inference.h`
- Create: `libs/lsxcommon/src/phi_dense_inference.cu`
- Create: `tests/test_phi_dense_kernels.cu`
- Modify: `libs/lsxcommon/CMakeLists.txt` (OBJECT lib)
- Modify: `tests/CMakeLists.txt` (OBJECT lib + target)

**Interfaces:**
- Produces: `RunPhiRopeKernel`, `RunPhiAttentionKernel`, `RunPhiAttentionSplitKernel`, `RunPhiRmsNormKernel`, `PhiKvCache`, `PhiGenerationResult`, `RunPhiGeneration`.
- Consumes: `PhiPinnedWeights` (Task 3), `PhiTopology` (Task 1).

- [ ] **Step 1: Write the failing kernel parity test**

Copy `tests/test_granite_dense_kernels.cu` → `tests/test_phi_dense_kernels.cu`, apply the rename table, update the fixture calls to `phitest::`, and update any hardcoded attention-scale assertions so the reference computes `1/sqrt(head_dim)` (the synthetic `attention_scale` field is `0.25` for head_dim 16). Keep the RoPE, RMSNorm, causal-GQA and forced-split parity cases.

- [ ] **Step 2: Register + run to verify it fails**

Add to `libs/lsxcommon/CMakeLists.txt` after the `lsxcommon_granite_cu` block:

```cmake
add_library(lsxcommon_phi_cu OBJECT
    src/phi_dense_inference.cu
)
set_target_properties(lsxcommon_phi_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES "120-real;121-real"
)
target_compile_options(lsxcommon_phi_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(lsxcommon_phi_cu PRIVATE ${LSXCOMMON_CU_INCLUDES})
target_compile_definitions(lsxcommon_phi_cu PRIVATE ${LSXCOMMON_CU_DEFS})
target_link_libraries(lsxcommon_phi_cu
    PRIVATE ...
)
```

Mirror the granite OBJECT-lib link list exactly (read the `lsxcommon_granite_cu` block for the precise `target_link_libraries` contents) and add `lsxcommon_phi_cu` to the parent `lsxcommon` link list right after `lsxcommon_granite_cu` (line ~367).

Add the test OBJECT lib + executable to `tests/CMakeLists.txt` (copy the `test_granite_cu`/`test_granite_dense_kernels` blocks, rename, `CUDA_ARCHITECTURES "120-real;121-real"`, `LABELS inference`).

Run: `cmake --preset linux-debug && cmake --build --preset linux-debug --target test_phi_dense_kernels`
Expected: FAIL to compile (`phi_dense_inference.h` missing).

- [ ] **Step 3: Create the inference header**

Copy `granite_dense_inference.h` → `libs/lsxcommon/include/lsxcommon/phi_dense_inference.h`, apply the rename table. Keep the exposed kernel entry points and structs.

- [ ] **Step 4: Create the inference implementation**

Create `libs/lsxcommon/src/phi_dense_inference.cu` by copying `granite_dense_inference.cu` and applying the rename table. Then:

1. Change the single attention-scale use (in `RunLayer`, the `GraniteAttention(...)` call) from `static_cast<float>(top.attention_multiplier)` to `static_cast<float>(top.attention_scale)`.
2. Confirm the header comment describes phi-4 (40 layers, `1/sqrt(128)`).
3. Leave everything else (fused qkv GEMV + `GraniteSplitQKV`, fused gate_up, attention split/combine, RMSNorm, RoPE, generation loop) byte-for-byte in behavior.

- [ ] **Step 5: Run to verify it passes**

Run: `cmake --preset linux-debug && cmake --build --preset linux-debug --target test_phi_dense_kernels && ctest --preset linux-debug-inference -R test_phi_dense_kernels`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/phi_dense_inference.h \
        libs/lsxcommon/src/phi_dense_inference.cu libs/lsxcommon/CMakeLists.txt \
        tests/test_phi_dense_kernels.cu tests/CMakeLists.txt
git commit -m "feat: phi-4 CUDA inference kernels + parity test"
```

---

### Task 6: Token bridge (phi ChatML) + fast framing test

**Files:**
- Create: `libs/lsxcommon/src/phi_tool_format.h` (declarations only; no family type required)
- Modify: `libs/lsxcommon/src/phi_dense_model.h` / `.cpp` (override the framing virtuals)
- Create: `tests/test_phi_tool_framing.cpp`
- Modify: `tests/test_model_suite_guard.cpp` or `tests/test_code_naming_guard.cpp` only if a family-stem scanner needs the new stem (tests are not scanned, so no change needed).
- Modify: `tests/CMakeLists.txt` (`fast` label)

**Interfaces:**
- Consumes: `ArrowTokenizerDb`, `ChatConversation`, `ToolCall`, `tools::*` (model-neutral).
- Produces: `Phi4Model::BuildPromptInputIds`, `Phi4Model::BuildConversationInputIds`; internal `phi::RenderPhiChat(role, content)`.

- [ ] **Step 1: Write the failing framing test**

Create `tests/test_phi_tool_framing.cpp` modeled on `tests/test_qwen_tool_framing.cpp` (read it first). It must:
- include `lsxcommon/tokenizer.h` and `test_tool_roundtrip.h`,
- open the synthetic tokenizer `TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json"`,
- assert the rendered phi frame for a system+user+assistant conversation contains the sentinel ids for `<|im_start|>`, `<|im_sep|>`, `<|im_end|>` in the correct order,
- assert `ParseToolCalls` recovers a bare `{"name":"get_weather","arguments":{"city":"Paris"}}` object.

Before writing the assertions, add the phi sentinels to `tests/data/synthetic_tok.json` `added_tokens` (`<|im_start|>`, `<|im_sep|>`, `<|im_end|>`, `<|endoftext|>`) with fresh ids not colliding with the existing sentinels, and reference those ids in the test.

- [ ] **Step 2: Register + run to verify it fails**

Add the target to `tests/CMakeLists.txt` (copy an existing fast framing target, e.g. `test_qwen_tool_framing`; `LABELS fast`).

Run: `cmake --preset linux-debug && cmake --build --preset linux-debug --target test_phi_tool_framing && ctest --preset linux-debug -R test_phi_tool_framing`
Expected: FAIL (framing not implemented).

- [ ] **Step 3: Implement the framing**

In `phi_dense_model.h`, add the two overrides:

```cpp
    bool BuildPromptInputIds(const ArrowTokenizerDb &tok,
                             std::string_view raw_prompt,
                             std::vector<int32_t> &out_ids) const override;
    bool BuildConversationInputIds(const ArrowTokenizerDb &tok,
                                   const ChatConversation &conv,
                                   std::vector<int32_t> &out_ids) const override;
```

In `phi_dense_model.cpp`, implement:

```cpp
namespace {

// Appends one phi ChatML turn. Control markers are special added_tokens and
// are injected as ids; only `content` is BPE-encoded.
void AppendPhiTurn(const ArrowTokenizerDb &tok, const char *role,
                   std::string_view content, std::vector<int32_t> &out)
{
    auto marker = [&](const char *s) {
        int32_t id = -1;
        if (arrow_tokenizer_lookup(tok, s, static_cast<int32_t>(std::strlen(s)),
                                   &id) == 0)
            out.push_back(id);
    };
    marker("<|im_start|>");
    std::vector<int32_t> rid(16, 0);
    size_t rcap = rid.size();
    if (arrow_tokenizer_encode(tok, role, std::strlen(role), rid.data(),
                               &rcap) > 0)
        out.insert(out.end(), rid.begin(), rid.begin() + rcap);
    marker("<|im_sep|>");
    if (!content.empty()) {
        std::vector<int32_t> cid(content.size() + 16, 0);
        size_t ccap = cid.size();
        if (arrow_tokenizer_encode(tok, content.data(), content.size(),
                                   cid.data(), &ccap) > 0)
            out.insert(out.end(), cid.begin(), cid.begin() + ccap);
    }
    marker("<|im_end|>");
}

} // namespace

bool Phi4Model::BuildPromptInputIds(const ArrowTokenizerDb &tok,
                                    std::string_view raw_prompt,
                                    std::vector<int32_t> &out_ids) const
{
    out_ids.clear();
    AppendPhiTurn(tok, "user", raw_prompt, out_ids);
    std::vector<int32_t> tail;
    AppendPhiTurn(tok, "assistant", {}, tail);
    out_ids.insert(out_ids.end(), tail.begin(), tail.end());
    return true;
}

bool Phi4Model::BuildConversationInputIds(const ArrowTokenizerDb &tok,
                                          const ChatConversation &conv,
                                          std::vector<int32_t> &out_ids) const
{
    out_ids.clear();
    if (!conv.tools.empty())
        AppendPhiTurn(tok, "system", tools::RenderToolSchemas(conv.tools),
                      out_ids);
    for (const auto &m : conv.messages)
        AppendPhiTurn(tok, m.role.c_str(), m.content, out_ids);
    std::vector<int32_t> tail;
    AppendPhiTurn(tok, "assistant", {}, tail);
    out_ids.insert(out_ids.end(), tail.begin(), tail.end());
    return true;
}
```

Use the actual accessor/field names and the actual tool-schema renderer name — verify against `libs/lsxcommon/include/lsxcommon/chat_conversation.h`, `libs/lsxcommon/include/lsxcommon/prompt_bridge.h` and `tool_format.h` (`RenderConversation`, `RenderToolSchemas`, message field names). `ParseToolCalls` stays the inherited generic default.

- [ ] **Step 4: Run to verify it passes**

Run: `ctest --preset linux-debug -R "test_phi_tool_framing|test_phi_dense_model"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/phi_dense_model.h libs/lsxcommon/src/phi_dense_model.cpp \
        tests/test_phi_tool_framing.cpp tests/data/synthetic_tok.json \
        tests/CMakeLists.txt
git commit -m "feat: phi-4 ChatML token bridge + framing test"
```

---

### Task 7: SelfCheck test

**Files:**
- Create: `tests/test_phi_dense_selfcheck.cpp`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `Phi4Model` (Task 4), `RunToolRoundTrip` (existing helper).

- [ ] **Step 1: Write the test**

Copy `tests/test_granite_dense_selfcheck.cpp` → `tests/test_phi_dense_selfcheck.cpp`, apply the rename table (`granite`→`phi`, `Granite`→`Phi`, model dir `phi-4`, env `LSX_TEST_PHI_MODEL_PATH`, cycles env `LSX_TEST_PHI_SELFCHECK_CYCLES`, tag `[phi]`, type `"phi-4"`). The tool round-trip test uses `RunToolRoundTrip(*m)` and checks `get_weather` / `Paris`.

- [ ] **Step 2: Register + run**

Add the target to `tests/CMakeLists.txt` (copy the granite selfcheck block, rename, `LABELS inference`, `SKIP_RETURN_CODE 4`).

Run: `cmake --preset linux-debug && cmake --build --preset linux-debug --target test_phi_dense_selfcheck && ctest --preset linux-debug-inference -R test_phi_dense_selfcheck`
Expected: SKIP (converted phi-4 not present yet) — exit code 4, not a failure.

- [ ] **Step 3: Commit**

```bash
git add tests/test_phi_dense_selfcheck.cpp tests/CMakeLists.txt
git commit -m "test: phi-4 SelfCheck + tool round-trip harness"
```

---

### Task 8: FSI / CODE / DET parity-gate tests + suite guard

**Files:**
- Create: `tests/test_phi_dense_fsi_selfcheck.cpp`
- Create: `tests/test_phi_dense_code_selfcheck.cpp`
- Create: `tests/test_phi_dense_det_selfcheck.cpp`
- Modify: `tests/test_model_suite_guard.cpp`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `fsi::RunFsiSelfCheck`, `code::RunCodeSelfCheck`, `det::RunDetSelfCheck`, golden dirs `LSX_{FSI,CODE,DET}_GOLDEN_DIR`.

- [ ] **Step 1: Create the three tests**

Copy the corresponding granite tests (`tests/test_granite_dense_{fsi,code,det}_selfcheck.cpp`) and apply the rename table: model dir `phi-4`, env `LSX_TEST_PHI_MODEL_PATH`, golden file `<model_type>` = `phi-4`, tag `[phi]`, `cfg.fast_math_mode = kOff`. Keep the `SKIP` when the model or golden is absent.

- [ ] **Step 2: Register + suite guard, run**

Add the three targets to `tests/CMakeLists.txt` (copy the granite `fsi`/`code`/`det` blocks; `SKIP_RETURN_CODE 4`, label `inference`, `LSX_FSI_GOLDEN_DIR`/`LSX_CODE_GOLDEN_DIR`/`LSX_DET_GOLDEN_DIR` compile definitions as in the granite blocks).

In `tests/test_model_suite_guard.cpp`:
- add `"phi"` to `FamilyStems()` (line ~48),
- add a `SuiteSpec` row after `mimo-v2.6-flash-rl`:

```cpp
        {"phi-4",
         {{"convert", {"test_phi_dense_convert"}},
          {"load", {"test_phi_dense_loader"}},
          {"model", {"test_phi_dense_model"}},
          {"kernels", {"test_phi_dense_kernels"}},
          {"selfcheck", {"test_phi_dense_selfcheck"}},
          {"fsi", {"test_phi_dense_fsi_selfcheck"}},
          {"code", {"test_phi_dense_code_selfcheck"}},
          {"det", {"test_phi_dense_det_selfcheck"}}}},
```

- update the regex at line ~381 to include `phi`.

Run: `cmake --preset linux-debug && cmake --build --preset linux-debug --target test_phi_dense_fsi_selfcheck test_phi_dense_code_selfcheck test_phi_dense_det_selfcheck test_model_suite_guard && ctest --preset linux-debug -R "test_model_suite_guard|test_ctest_labels_guard"`
Expected: PASS (guard green); the three real-model targets SKIP.

- [ ] **Step 3: Commit**

```bash
git add tests/test_phi_dense_fsi_selfcheck.cpp tests/test_phi_dense_code_selfcheck.cpp \
        tests/test_phi_dense_det_selfcheck.cpp tests/test_model_suite_guard.cpp \
        tests/CMakeLists.txt
git commit -m "test: phi-4 FSI/CODE/DET gates + suite-guard registration"
```

---

### Task 9: Perf baseline target + entry

**Files:**
- Modify: `tests/perf/perf_baselines.json`
- Modify: `tests/CMakeLists.txt`
- Modify: `tests/test_perf_baseline.cpp` (vault list, if it enumerates models)

**Interfaces:**
- Consumes: the perf-baseline harness (`test_perf_baseline` with `LSX_PERF_VAULT_SUBDIR`).

- [ ] **Step 1: Add the CMake target**

Copy the granite perf-baseline block (`add_test(NAME test_perf_baseline_granite_4_2_30b ...)` around line 1771) and rename:

```cmake
add_test(NAME test_perf_baseline_phi_4
         COMMAND test_perf_baseline "perf baseline: phi-4")
set_tests_properties(test_perf_baseline_phi_4 PROPERTIES
    ...
    ENVIRONMENT "LSX_PERF_VAULT_SUBDIR=phi-4")
set_tests_properties(test_perf_baseline_phi_4 PROPERTIES LABELS perf)
```

Read the exact granite block and mirror its properties (`TIMEOUT`, `LABELS perf`, any `RESOURCE_LOCK`). If `tests/test_perf_baseline.cpp` enumerates vault models in a hardcoded list (check around lines 47-70), add the `phi-4` entry (`model_type` `"phi-4"`, `vault_subdir` `"phi-4"`).

- [ ] **Step 2: Leave the JSON entry for Task 11**

Do **not** invent a `tokens_per_sec`. The `perf_baselines.json` entry is added in Task 11 with the measured value (a JSON entry without a target is unguarded; a target without a measured entry would fail). This step only wires the target.

- [ ] **Step 3: Build check + commit**

Run: `cmake --preset linux-debug && cmake --build --preset linux-debug --target test_perf_baseline`
Expected: builds.

```bash
git add tests/CMakeLists.txt tests/test_perf_baseline.cpp
git commit -m "test: phi-4 perf-baseline target"
```

---

### Task 10: lsxforge `--phi` flag + AGENTS.md structure

**Files:**
- Modify: `apps/lsxforge/src/main.cpp`
- Modify: `AGENTS.md`

- [ ] **Step 1: Add the converter CLI flag**

In `apps/lsxforge/src/main.cpp`:
- add `#include "lsxcommon/phi_dense_convert.h"` next to the granite include (line ~23),
- add the flag next to `ABSL_FLAG(bool, granite, ...)`:

```cpp
ABSL_FLAG(bool, phi, false, "phi-4 (dense) conversion pipeline");
```

- read it next to `bool granite = ...`:

```cpp
    bool phi = absl::GetFlag(FLAGS_phi);
```

- add the dispatch branch next to the granite branch:

```cpp
    if (phi) {
        lsxcommon::log::info("phi-4 pipeline (logestix-phi layout)");
        auto st = lsxcommon::phi::ConvertPhiModel(input_path, output_path);
        if (!st.ok()) {
            lsxcommon::log::error(
                absl::StrFormat("phi conversion failed: %s", st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("phi conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }
```

- add `phi` to the `--vram-recompute` format sniff (index contains `logestix-phi`):

```cpp
        else if (json.find("logestix-phi") != std::string::npos)
            st = lsxcommon::phi::RecomputePhiVram(input_path, output_path);
```

- [ ] **Step 2: Build**

Run: `cmake --preset linux-debug && cmake --build --preset linux-debug --target lsxforge`
Expected: builds.

- [ ] **Step 3: Update AGENTS.md Structure**

In the `libs/lsxcommon/` bullet, after the Granite-4.2-30B description, add a phi-4 paragraph naming `phi_dense_constants.h`, `phi_dense_convert.{h,cpp}`, `phi_dense_load.{h,cpp}`, `phi_dense_inference.{h,cu}`, `phi_dense_model.{h,cpp}`, format tag `logestix-phi`, and `lsxforge --phi`.

- [ ] **Step 4: Commit**

```bash
git add apps/lsxforge/src/main.cpp AGENTS.md
git commit -m "feat: lsxforge --phi conversion pipeline; document phi-4"
```

---

### Task 11: Real conversion, golden capture, selfcheck + perf

**Files:**
- Create: `tests/fixtures/fsi/phi-4.json`
- Create: `tests/fixtures/code/phi-4.json`
- Create: `tests/fixtures/det/phi-4.json`
- Modify: `tests/perf/perf_baselines.json`
- Modify: `docs/FSI_SELFCHECK_FINDINGS.md`, `docs/CODE_SELFCHECK_FINDINGS.md`, `docs/DET_SELFCHECK_FINDINGS.md` (capture recipes/outcomes)

This task is operational (RTX 5090, one GPU load at a time).

- [ ] **Step 1: Build release path and convert**

```bash
cmake --preset linux-debug && cmake --build --preset linux-debug
lsxforge --help   # confirm the flag set (binary under build/linux-debug/bin/)
lsxforge --phi --input /mnt/llm_models/models/phi-4 --output /mnt/llm_models/models/phi-4
```

Expected: `logestix/model.index.json` + `logestix/pinned/*` created; log reports the VRAM profile and tensor count.

- [ ] **Step 2: Capture goldens**

Read the `OVERRIDES` tables first. phi-4 needs no special loader, but confirm the tokenizer defines a chat template (it does, embedded in `tokenizer_config.json`) and that capture uses it.

```bash
python tools/fsi_capture.py  --model /mnt/llm_models/models/phi-4 --type phi-4 --out tests/fixtures/fsi/phi-4.json
python tools/code_capture.py --model /mnt/llm_models/models/phi-4 --type phi-4 --out tests/fixtures/code/phi-4.json --max-new-tokens 512
python tools/det_capture.py  --model /mnt/llm_models/models/phi-4 --type phi-4 --out tests/fixtures/det/phi-4.json
```

Run each capture's `--selftest` first if the model path is wrong. Inspect each golden's `source`/`expected`; abort and report if extraction is malformed.

- [ ] **Step 3: Run the parity gates on the converted model**

```bash
ctest --preset linux-debug-inference -R "test_phi_dense_(selfcheck|fsi_selfcheck|code_selfcheck|det_selfcheck)" --output-on-failure
```

Expected: PASS. A failure is a real finding (kernel/quantization divergence) to investigate, not to loosen.

- [ ] **Step 4: Measure and record perf**

```bash
ctest --preset linux-debug -R test_perf_baseline_phi_4 --output-on-failure
```

Read the measured t/s from the output, then add a `perf_baselines.json` entry (copy the granite entry shape: `model_type` `"phi-4"`, `gpu_key` `"rtx 5090"`, `os` `"linux"`, `mode` `"perf"`, `tokens_per_sec`, `max_gen` 2048, prompt `"The capital of France is"`, `date`, `commit` `"-"`, `gen_tokens`, `ttft_ms`, power fields). Re-run the target to confirm the entry is guarded and green.

- [ ] **Step 5: Record findings and commit**

Append the phi-4 capture recipe/outcome to the three findings docs.

```bash
git add tests/fixtures/fsi/phi-4.json tests/fixtures/code/phi-4.json \
        tests/fixtures/det/phi-4.json tests/perf/perf_baselines.json \
        docs/FSI_SELFCHECK_FINDINGS.md docs/CODE_SELFCHECK_FINDINGS.md \
        docs/DET_SELFCHECK_FINDINGS.md
git commit -m "test: phi-4 real-weight goldens, parity gates and perf baseline"
```

Note: the converted weights under `/mnt/llm_models/models/phi-4/logestix/` are vault data, not repo files — do not add them.

---

### Task 12: Full guard sweep + Definition of Done

- [ ] **Step 1: Build everything and run the fast suite**

```bash
cmake --preset linux-debug && cmake --build --preset linux-debug
ctest --preset linux-debug
```

Expected: all `fast` green, including the four guards.

- [ ] **Step 2: Run the convert and inference sweeps**

```bash
ctest --preset linux-debug-convert
ctest --preset linux-debug-inference -R "phi" --output-on-failure
```

Expected: convert green; phi inference green (selfchecks execute on the converted model).

- [ ] **Step 3: Router dry-run**

```bash
cmake -DLSX_PRESET=linux-debug -DLSX_LIST=ON -P tools/test_router.cmake
```

Expected: lists the phi targets with the expected labels.

- [ ] **Step 4: Verify the Definition of Done checklist**

- `kType` registered; `Load` rejects a frozen-geometry mismatch.
- Converter test (label `convert`, bounded TIMEOUT, synthetic checkpoint).
- Loader/model/kernels tests present and labelled.
- Tool bridge implemented; `test_phi_tool_framing` green.
- `test_phi_dense_selfcheck` >=3 cycles, bit-exact, `SKIP_RETURN_CODE 4`.
- FSI/CODE/DET goldens committed and gates green on the converted model.
- Perf entry + target both present.
- All four guard registries updated and green.
- AGENTS.md Structure updated.

- [ ] **Step 5: Final commit (if any stray changes)**

```bash
git status --short
git add -A ':!*/logestix/*'
git commit -m "test: phi-4 ingestion guard sweep"   # only if changes remain
```

---

## Self-Review

- **Spec coverage**: Task 1 = passport + guards; Task 2 = stages 1/2 (converter+NVFP4); Task 3 = loader (stage 3 host); Task 4 = stage 3 (registration/model); Task 5 = stages 5/6 + stage 7 (device-safe passport shared); Task 6 = stage 4/4b; Task 7 = stage 8 SelfCheck; Task 8 = stage 8 FSI/CODE/DET; Task 9-11 = perf + real-weight verification; Task 12 = DoD. The stage-7 token translator (`BuildPassport`/`IoWorker`) is inherited from the existing `lsxnetwork` runtime and needs no phi-specific edit (the passport is built from the converted tokenizer arrow files, which Task 2 writes).
- **Placeholder scan**: the "copy + rename" steps give the exact source file, the exact rename table and the exact semantic edits; no "TBD"/"handle edge cases" remain.
- **Type consistency**: `PhiTopology.attention_scale` (Task 1) is the field consumed by the inference `RunLayer` (Task 5) and written by the converter (Task 2). `TensorKind::{AttnQKV, MlpGateUp}` (Task 2) are the only new enum values, referenced in the loader split logic (Task 3).
- **Known implementer check**: exact accessor names for `ChatConversation`/`tool_format` (Task 6) and the exact `lsxcommon_granite_cu` link list (Task 5) must be read from the referenced files before writing; the plan names the files to read.
