# Granite-Switch 4.1 30B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **AGENTS.md overrides the default execution mode:** this repository forbids
> Task/parallel subagents. Execute this plan **inline** with
> `superpowers:executing-plans`; the main session writes every file.

**Goal:** Ingest `granite-switch-4.1-30b` (Granite 4.1 30B dense + two embedded
aLoRA adapters + control-token switch) as a logestix family variant with full
model lifecycle, parity gates and a measured perf row.

**Architecture:** New sibling files `granite_dense_switch_*` (variant `switch`
of the `granite` family) reuse the neutral NVFP4 layer and shared kernel
primitives. A new frozen passport, converter, loader, CUDA inference loop and
model class are added; the Granite 4.2 files are untouched. Adapter routing is
the proven prefix scan over control tokens; aLoRA deltas are FP16 shrink/expand
passes over the NVFP4 base GEMV.

**Tech Stack:** C++20/C23, CUDA 13.4 (sm_120a), CMake presets, Arrow 26,
Catch2 v3.16.1, spdlog; Python oracle `transformers==5.17.0` +
`granite-switch==0.1.1` (CPU BF16).

## Global Constraints

- Target Blackwell only (CC 12.0 / 12.1). No sm_90/sm_80 paths or guards.
- One CUDA stream; no per-token `cudaDeviceSynchronize`; no `atomicAdd` feeding
  `final_logits`.
- NVFP4 only through `lsx_nvfp4` (`RunLsxNvfp4Gemm`), never a family dispatcher.
- Norms, biases, LoRA and the switch codebook are never NVFP4-quantized.
- Frozen geometry as compile-time constants with `static_assert`; `Load` must
  reject a mismatch.
- No abseil containers/streams in headers reachable from `.cu`/`.cuh`.
- Logging only via spdlog (`lsxcommon::log`), never `std::cout`/`printf`.
- Every model opens `LsxFastMathScope(EffectiveFastMathMode(cfg_))` inside
  `Infer` and implements the shared lifecycle (residency lock, `GenerationStream`
  + abort, `TeardownContext`, governor lease, result fill, session state under
  the lock).
- Every new ctest target gets a label and (selfcheck/fsi/code/det/perf)
  `SKIP_RETURN_CODE 4`; converter tests `TIMEOUT 120`.
- Commit prefixes: `build:`/`feat:`/`fix:`/`refactor:`/`test:`/`docs:`.
- TDD: the failing test exists before production code; each stage lands with
  its guard-registry entries.
- No ASR/audio: the 470M speech encoder is not in the checkpoint.
- Exact model facts: 64 decoder layers (config says 66 = 64 + 2 switch cache
  slots), vocab 100355, tied embeddings, multipliers 12 / 0.175 / 16,
  control ids 100352/100353 -> 27/14711.

---

### Task 1: Frozen passport, naming tokens and converter classifier

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/granite_dense_switch_constants.h`
- Create: `libs/lsxcommon/include/lsxcommon/granite_dense_switch_convert.h`
- Create: `libs/lsxcommon/src/granite_dense_switch_convert.cpp`
- Create: `tests/test_granite_switch_fixture.h`
- Create: `tests/test_granite_switch_convert.cpp`
- Modify: `tests/test_code_naming_guard.cpp`
- Modify: `tests/test_model_independence_guard.cpp` (`kPassports` row only)
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Produces:
  - `lsxcommon::granite_switch::GraniteSwitchTopology` with the frozen fields
    plus `std::vector<int> adapter_ranks/adapter_token_ids/`
    `adapter_substitute_token_ids`, `int switch_cache_layers`,
    `bool tie_word_embeddings`, multipliers, `VramProfile vram`,
    `std::string error_msg`.
  - `namespace lsxcommon::granite_switch::frozen` constants:
    `kNumLayers=64, kHiddenSize=4096, kNumHeads=32, kNumKVHeads=8,
    kHeadDim=128, kIntermediateSize=32768, kVocabSize=100355,
    kSwitchCacheLayers=2, kEmbeddingMultiplier=12.0f,
    kResidualMultiplier=0.175f, kLogitsScaling=16.0f,
    kControlTokenIds={100352,100353}, kSubstituteTokenIds={27,14711},
    kAdapterRanks={16,64}, kBosId=100257, kEosId=100257, kPadId=100256`.
  - `enum class GraniteSwitchTensorKind` and
    `ParsedGraniteSwitchTensor classify_granite_switch_tensor(const std::string&)`
    where `ParsedGraniteSwitchTensor { kind; int layer; int slice; }`.
  - `GraniteSwitchTopology load_granite_switch_topology(const std::string&)`.
  - `arrow::Status ConvertGraniteSwitchModel(const std::string&, const std::string&)`.

- [ ] **Step 1: Write the failing classifier/fixture test**

`tests/test_granite_switch_fixture.h` (complete synthetic model builder):

```cpp
#pragma once
#include "test_synth_checkpoint.h"
#include <cstdint>
#include <filesystem>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace gswtest {
using synthtest::FakeTensor;
using synthtest::MakeBf16;
using synthtest::MakeBf16Values;
using synthtest::MakeF32Values;
using synthtest::TensorNamesIn;
using synthtest::WriteSafetensors;

struct SwitchTestConfig {
    int hidden_size = 64;
    int intermediate_size = 128;
    int vocab_size = 72;
    int num_layers = 2;
    int num_attention_heads = 4;
    int num_key_value_heads = 2;
    int head_dim = 16;
    int max_lora_rank = 8;
    int switch_cache_layers = 2;
    std::vector<int> adapter_ranks = {4, 8};
    std::vector<int> adapter_token_ids = {64, 65};
    std::vector<int> adapter_substitute_token_ids = {6, 7};
    double rms_norm_eps = 1e-5;
    double rope_theta = 10000.0;
    double attention_multiplier = 0.0625;
    double embedding_multiplier = 12.0;
    double residual_multiplier = 0.175;
    double logits_scaling = 16.0;
    int bos_id = 0, eos_id = 1, pad_id = 2;
};
inline const SwitchTestConfig& Cfg() { static SwitchTestConfig c; return c; }
inline int QOut() { return Cfg().num_attention_heads * Cfg().head_dim; }
inline int KVOut() { return Cfg().num_key_value_heads * Cfg().head_dim; }

inline std::vector<FakeTensor> BuildModel()
{
    const auto& c = Cfg();
    const int H = c.hidden_size, I = c.intermediate_size, V = c.vocab_size;
    const int A = static_cast<int>(c.adapter_ranks.size());
    const int R = c.max_lora_rank;
    std::vector<FakeTensor> t;
    t.push_back({"model.embed_tokens.weight", "BF16", {V, H}, MakeBf16(V * H)});
    t.push_back({"model.norm.weight", "BF16", {H}, MakeBf16(H)});
    t.push_back({"model.switch.codebook", "F32", {2048, 64},
                 MakeF32Values(2048 * 64, [](int64_t i) { return (float)(i % 7) * 0.01f; })});
    std::vector<int64_t> lut(V, -1);
    for (int a = 0; a < A; ++a) lut[c.adapter_token_ids[a]] = c.adapter_substitute_token_ids[a];
    std::vector<uint8_t> lut_bytes(lut.size() * sizeof(int64_t));
    std::memcpy(lut_bytes.data(), lut.data(), lut_bytes.size());
    t.push_back({"model.switch.control_to_substitute_lut", "I64", {V}, std::move(lut_bytes)});
    for (int L = 0; L < c.num_layers; ++L) {
        const std::string lp = "model.layers." + std::to_string(L) + ".";
        t.push_back({lp + "input_layernorm.weight", "BF16", {H}, MakeBf16(H)});
        t.push_back({lp + "post_attention_layernorm.weight", "BF16", {H}, MakeBf16(H)});
        const std::string sa = lp + "self_attn.";
        t.push_back({sa + "qkv_proj.base_layer.weight", "BF16",
                     {QOut() + 2 * KVOut(), H}, MakeBf16((QOut() + 2 * KVOut()) * H)});
        t.push_back({sa + "o_proj.base_layer.weight", "BF16", {H, QOut()}, MakeBf16(H * QOut())});
        for (int s = 0; s < 3; ++s) {
            const int out = s == 0 ? QOut() : KVOut();
            const std::string slice = std::to_string(s);
            t.push_back({sa + "qkv_proj.lora_A_slices." + slice, "BF16",
                         {A, 1, R, H}, MakeBf16(A * R * H)});
            t.push_back({sa + "qkv_proj.lora_B_slices." + slice, "BF16",
                         {A, 1, out, R}, MakeBf16(A * out * R)});
        }
        t.push_back({sa + "o_proj.lora_A", "BF16", {A, 1, R, QOut()},
                     MakeBf16(A * R * QOut())});
        t.push_back({sa + "o_proj.lora_B", "BF16", {A, 1, H, R}, MakeBf16(A * H * R)});
        const std::string ml = lp + "shared_mlp.";
        t.push_back({ml + "input_linear.base_layer.weight", "BF16", {2 * I, H}, MakeBf16(2 * I * H)});
        t.push_back({ml + "output_linear.base_layer.weight", "BF16", {H, I}, MakeBf16(H * I)});
        for (int s = 0; s < 2; ++s) {
            const std::string slice = std::to_string(s);
            t.push_back({ml + "input_linear.lora_A_slices." + slice, "BF16",
                         {A, 1, R, H}, MakeBf16(A * R * H)});
            t.push_back({ml + "input_linear.lora_B_slices." + slice, "BF16",
                         {A, 1, I, R}, MakeBf16(A * I * R)});
        }
        t.push_back({ml + "output_linear.lora_A", "BF16", {A, 1, R, I}, MakeBf16(A * R * I)});
        t.push_back({ml + "output_linear.lora_B", "BF16", {A, 1, H, R}, MakeBf16(A * H * R)});
    }
    // Adapter 0 targets attention only: zero its MLP deltas like the real
    // guardian-core. The adapter axis is outermost, so adapter 0 is the first
    // half of each stacked LoRA tensor.
    for (auto& ft : t) {
        if (ft.name.find("shared_mlp.") != std::string::npos &&
            (ft.name.find("lora_A") != std::string::npos ||
             ft.name.find("lora_B") != std::string::npos)) {
            std::fill(ft.data.begin(), ft.data.begin() + ft.data.size() / 2, 0);
        }
    }
    return t;
}

inline void WriteConfig(const std::filesystem::path& dir)
{
    const auto& c = Cfg();
    std::string j = "{\"architectures\":[\"GraniteSwitchForCausalLM\"],"
        "\"model_type\":\"granite_switch\",\"hidden_size\":" + std::to_string(c.hidden_size) +
        ",\"intermediate_size\":" + std::to_string(c.intermediate_size) +
        ",\"shared_intermediate_size\":" + std::to_string(c.intermediate_size) +
        ",\"num_hidden_layers\":" + std::to_string(c.num_layers + c.switch_cache_layers) +
        ",\"vocab_size\":" + std::to_string(c.vocab_size) +
        ",\"num_attention_heads\":" + std::to_string(c.num_attention_heads) +
        ",\"num_key_value_heads\":" + std::to_string(c.num_key_value_heads) +
        ",\"projection_head_dim\":" + std::to_string(c.head_dim) +
        ",\"num_adapters\":2,\"max_lora_rank\":" + std::to_string(c.max_lora_rank) +
        ",\"adapter_ranks\":[4,8],\"adapter_token_ids\":[64,65],"
        "\"adapter_substitute_token_ids\":[6,7],\"tie_word_embeddings\":true,"
        "\"embedding_multiplier\":12.0,\"residual_multiplier\":0.175,"
        "\"logits_scaling\":16.0,\"attention_multiplier\":0.0625,"
        "\"rms_norm_eps\":1e-5,\"rope_theta\":10000.0,\"hidden_act\":\"silu\","
        "\"bos_token_id\":0,\"eos_token_id\":1,\"pad_token_id\":2,"
        "\"max_position_embeddings\":4096}";
    std::ofstream(dir / "config.json") << j;
}
} // namespace gswtest
```

`tests/test_granite_switch_convert.cpp` first test:

```cpp
#include <catch2/catch_all.hpp>
#include <lsxcommon/granite_dense_switch_convert.h>
#include "test_granite_switch_fixture.h"

using namespace lsxcommon::granite_switch;

TEST_CASE("granite-switch tensor classifier", "[granite_switch][convert]")
{
    auto check = [](const std::string& n, GraniteSwitchTensorKind k,
                    int layer = -1, int slice = -1) {
        auto p = classify_granite_switch_tensor(n);
        CHECK(p.kind == k);
        if (layer >= 0) CHECK(p.layer == layer);
        if (slice >= 0) CHECK(p.slice == slice);
    };
    check("model.embed_tokens.weight", GraniteSwitchTensorKind::Embed);
    check("model.norm.weight", GraniteSwitchTensorKind::FinalNorm);
    check("model.switch.codebook", GraniteSwitchTensorKind::SwitchCodebook);
    check("model.switch.control_to_substitute_lut", GraniteSwitchTensorKind::SwitchLut);
    check("model.layers.3.input_layernorm.weight", GraniteSwitchTensorKind::InputNorm, 3);
    check("model.layers.3.self_attn.qkv_proj.base_layer.weight", GraniteSwitchTensorKind::AttnQkvBase, 3);
    check("model.layers.3.self_attn.qkv_proj.lora_A_slices.2", GraniteSwitchTensorKind::QkvLoraA, 3, 2);
    check("model.layers.3.self_attn.qkv_proj.lora_B_slices.1", GraniteSwitchTensorKind::QkvLoraB, 3, 1);
    check("model.layers.3.self_attn.o_proj.base_layer.weight", GraniteSwitchTensorKind::AttnOBase, 3);
    check("model.layers.3.self_attn.o_proj.lora_A", GraniteSwitchTensorKind::AttnOLoraA, 3);
    check("model.layers.3.shared_mlp.input_linear.base_layer.weight", GraniteSwitchTensorKind::MlpUpBase, 3);
    check("model.layers.3.shared_mlp.input_linear.lora_A_slices.1", GraniteSwitchTensorKind::MlpUpLoraA, 3, 1);
    check("model.layers.3.shared_mlp.output_linear.base_layer.weight", GraniteSwitchTensorKind::MlpDownBase, 3);
    check("model.layers.3.shared_mlp.output_linear.lora_B", GraniteSwitchTensorKind::MlpDownLoraB, 3);
    CHECK_FALSE(gswtest::Cfg().adapter_ranks.empty());
}
```

- [ ] **Step 2: Run it and verify it fails**

```bash
cmake --build --preset linux-debug --target test_granite_switch_convert
./build/linux-debug/tests/test_granite_switch_convert
```
Expected: compile/link failure (header/classifier missing).

- [ ] **Step 3: Implement the passport and classifier**

Write the passport with `namespace frozen {` and `static_assert`s (hidden ==
num_heads*head_dim, vocab > control ids). The classifier is a regex/suffix table
mapping the names above; unknown -> `Unknown`. `load_granite_switch_topology`
parses `model.index.json` (written later) with the fields listed in the spec.
`ConvertGraniteSwitchModel` initially returns `Invalid("not implemented")` so
the classifier test links; Task 2 replaces it.

Add the naming token `switch` to `var` and the check case:

```cpp
CHECK(m("granite_dense_switch_constants"));
CHECK(m("granite_dense_switch_convert"));
CHECK(m("granite_switch_tool_format")); // added to AllowedCommon
```

Add the `kPassports` row:

```cpp
{"granite-switch-4.1-30b", "granite_dense_switch_constants.h",
 "lsxcommon::granite_switch::frozen"},
```

Add the CMake target with `LABELS convert`, `TIMEOUT 120`,
`ENVIRONMENT "LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1"`.

- [ ] **Step 4: Run the classifier test**

```bash
cmake --build --preset linux-debug --target test_granite_switch_convert
./build/linux-debug/tests/test_granite_switch_convert
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/granite_dense_switch_constants.h \
        libs/lsxcommon/include/lsxcommon/granite_dense_switch_convert.h \
        libs/lsxcommon/src/granite_dense_switch_convert.cpp \
        tests/test_granite_switch_fixture.h tests/test_granite_switch_convert.cpp \
        tests/test_code_naming_guard.cpp tests/test_model_independence_guard.cpp \
        tests/CMakeLists.txt
git commit -m "feat(granite-switch): frozen passport and tensor classifier"
```

---

### Task 2: Converter pipeline (index, NVFP4 base, rank-trimmed adapters, LUT)

**Files:**
- Modify: `libs/lsxcommon/src/granite_dense_switch_convert.cpp`
- Modify: `tests/test_granite_switch_convert.cpp`
- Modify: `libs/lsxcommon/CMakeLists.txt` (add the `.cpp` to `lsxcommon`)

**Interfaces:**
- Consumes: classifier + passport from Task 1.
- Produces: `<out>/logestix/model.index.json` and pinned Arrow files described in
  the spec; `load_granite_switch_topology` round-trips them.

- [ ] **Step 1: Add the failing E2E test**

Append to `tests/test_granite_switch_convert.cpp`:

```cpp
TEST_CASE("granite-switch converter builds the pinned layout",
          "[granite_switch][convert]")
{
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "granite_switch_conv_test";
    remove_all(tmp); create_directories(tmp);
    gswtest::WriteSafetensors(tmp, "model.safetensors", gswtest::BuildModel());
    gswtest::WriteConfig(tmp);
    copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json", tmp / "tokenizer.json");

    REQUIRE(ConvertGraniteSwitchModel(tmp.string(), tmp.string()).ok());

    auto top = load_granite_switch_topology(
        (tmp / "logestix" / "model.index.json").string());
    CHECK(top.error_msg.empty());
    CHECK(top.num_layers == 2);
    CHECK(top.switch_cache_layers == 2);
    CHECK(top.vocab_size == 72);
    CHECK(top.adapter_ranks == std::vector<int>({4, 8}));
    CHECK(top.tie_word_embeddings);
    CHECK(top.embedding_multiplier == Catch::Approx(12.0));
    CHECK(top.residual_multiplier == Catch::Approx(0.175));
    CHECK(top.logits_scaling == Catch::Approx(16.0));

    const auto pinned = tmp / "logestix" / "pinned";
    CHECK(exists(pinned / "global.arrow"));
    CHECK(exists(pinned / "global_nvfp4.arrow"));
    CHECK(exists(pinned / "layer_0_nvfp4.arrow"));
    CHECK(exists(pinned / "layer_1_nvfp4.arrow"));
    CHECK(exists(pinned / "layer_0_adapters.arrow"));

    auto gn = gswtest::TensorNamesIn(pinned / "global_nvfp4.arrow");
    CHECK(std::find(gn.begin(), gn.end(), "model.embed_tokens.weight") != gn.end());
    CHECK(std::find(gn.begin(), gn.end(), "lm_head.weight") == gn.end()); // tied

    auto an = gswtest::TensorNamesIn(pinned / "layer_0_adapters.arrow");
    CHECK(std::find(an.begin(), an.end(),
          "model.layers.0.self_attn.qkv_proj.lora_A_slices.0.adapter0") != an.end());
    CHECK(std::find(an.begin(), an.end(),
          "model.layers.0.shared_mlp.input_linear.lora_A_slices.0.adapter0") == an.end());

    remove_all(tmp);
}
```

- [ ] **Step 2: Run it (fails: `not implemented`)**

```bash
cmake --build --preset linux-debug --target test_granite_switch_convert
./build/linux-debug/tests/test_granite_switch_convert
```

- [ ] **Step 3: Implement the converter**

Follow `granite_dense_convert.cpp` structure, with these deltas:

1. Read `config.json`: geometry + `projection_head_dim` + multipliers + tied
   flag + adapter arrays. Validate
   `cfg.num_hidden_layers - cfg.switch_cache_layers == tensor_layers`.
2. Verify coverage of every expected base/LoRA/switch tensor; no `lm_head`.
3. NVFP4: `embed` once; `qkv_proj.base_layer.weight`, `o_proj.base_layer.weight`,
   `input_linear.base_layer.weight` (already `[2I,H]`), `output_linear.base_layer.weight`.
4. Adapters: for adapter `a` and each site, trim
   `A[0:r, :]` and `B[:, 0:r]` (r = `adapter_ranks[a]`), write BF16 raw
   records named `<base_name>.adapter<a>`. Skip sites whose whole tensor is zero
   (guardian MLP). Assert `A[r:]` is all zero before trimming.
5. Read the I64 LUT; verify `lut[control]=substitute` and `-1` elsewhere; write
   the mapping to the index and drop the tensor. Store the codebook raw F32.
6. Write the index fields from the spec; `head` is not written.
7. `Vram` block from the source inventory (adapter tensors counted FP16).
8. Tokenizer: `BuildAndVerifyTokenizer` with `bos/eos/pad` required.

- [ ] **Step 4: Run the converter test**

Expected: PASS, and `layer_0_adapters.arrow` contains adapter 1 MLP records but
not adapter 0.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/granite_dense_switch_convert.cpp \
        libs/lsxcommon/CMakeLists.txt tests/test_granite_switch_convert.cpp
git commit -m "feat(granite-switch): converter emits base NVFP4 and adapter planes"
```

---

### Task 3: Loader (base + adapters, tied head alias, free)

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/granite_dense_switch_load.h`
- Create: `libs/lsxcommon/src/granite_dense_switch_load.cpp`
- Create: `tests/test_granite_switch_loader.cpp`
- Modify: `libs/lsxcommon/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Produces:
  - `struct GraniteSwitchLora { __half *d_A; __half *d_B; int rank, in, out; }`
  - `struct GraniteSwitchProjection { LsxNvfp4Weight base; std::vector<GraniteSwitchLora> lora; }`
  - `struct GraniteSwitchLayerWeights { __half *d_input_norm,*d_post_norm; GraniteSwitchProjection qkv,o,gate_up,down; void Free(); }`
  - `struct GraniteSwitchPinnedWeights { GraniteSwitchTopology top; LsxNvfp4Weight embed, head; __half *d_final_norm; std::vector<GraniteSwitchLayerWeights> layers; uint64_t resident_bytes; void Free(); }`
  - `bool LoadGraniteSwitchPinnedWeights(const std::string&, GraniteSwitchPinnedWeights&, cudaStream_t)`

- [ ] **Step 1: Write the failing loader test (synthetic tree only)**

```cpp
TEST_CASE("granite-switch loader uploads base and adapters",
          "[granite_switch][inference]")
{
    // Build the synthetic tree with Task 1/2 helpers, convert it, then:
    GraniteSwitchPinnedWeights pw;
    REQUIRE(LoadGraniteSwitchPinnedWeights(tmp.string(), pw, 0));
    CHECK(pw.layers.size() == 2);
    CHECK(pw.embed.d_packed != nullptr);
    CHECK(pw.head.d_packed == pw.embed.d_packed);          // tied alias
    CHECK(pw.head.global_scale == Catch::Approx(pw.embed.global_scale / 16.0f));
    CHECK(pw.layers[0].qkv.lora[0].rank == 4);
    CHECK(pw.layers[0].qkv.lora[1].rank == 8);
    CHECK(pw.layers[0].down.lora[0].d_A == nullptr);       // guardian has no MLP
    CHECK(pw.layers[0].down.lora[1].d_A != nullptr);
    pw.Free();
    CHECK(pw.embed.d_packed == nullptr);
    CHECK(pw.layers.empty());
}
```

- [ ] **Step 2: Run it (fails: header missing).**

- [ ] **Step 3: Implement the loader**

Use `fastload::UploadNvfp4Weight` / `RunUploadJobs` exactly as
`granite_dense_load.cpp`. Adapter records are raw BF16 -> FP16 device arrays
(one `cudaMalloc` arena per layer or per adapter; freed in `Free()`). Set
`head` as a copy of `embed` pointers with `global_scale / 16` and
`view_only = true`. Validate `top` against the passport unless
`SyntheticGeometryAllowed()`.

- [ ] **Step 4: Run the test; expected PASS.**
- [ ] **Step 5: Commit** `feat(granite-switch): loader for base and adapter planes`.

---

### Task 4: Host request preparation (token exchange, routing, continuation)

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/granite_dense_switch_inference.h`
- Create: `libs/lsxcommon/src/granite_dense_switch_prepare.cpp` (host-only)
- Create: `tests/test_granite_switch_prepare.cpp`
- Modify: `tests/CMakeLists.txt`, `libs/lsxcommon/CMakeLists.txt`

**Interfaces:**
- Produces:
  - `struct GraniteSwitchPrepared { std::vector<int32_t> embed_ids, adapter_ids; int last_adapter; }`
  - `GraniteSwitchPrepared PrepareGraniteSwitchTokens(const GraniteSwitchTopology&, const std::vector<int32_t>& ids, int carry_adapter);`
  - `int32_t GraniteSwitchSubstitute(const GraniteSwitchTopology&, int32_t id);`

- [ ] **Step 1: Write the failing test**

```cpp
TEST_CASE("switch prefix scan and token exchange", "[granite_switch][prepare]")
{
    GraniteSwitchTopology top;
    top.adapter_token_ids = {100352, 100353};
    top.adapter_substitute_token_ids = {27, 14711};
    std::vector<int32_t> ids = {5, 100352, 6, 100353, 7, 8};
    auto p = PrepareGraniteSwitchTokens(top, ids, 0);
    CHECK(p.adapter_ids == std::vector<int32_t>({0, 1, 1, 2, 2, 2}));
    CHECK(p.embed_ids == std::vector<int32_t>({5, 27, 6, 14711, 7, 8}));
    CHECK(p.last_adapter == 2);
    auto c = PrepareGraniteSwitchTokens(top, {100353}, 1);   // continuation
    CHECK(c.adapter_ids == std::vector<int32_t>({2}));
    CHECK(c.last_adapter == 2);
}
```

- [ ] **Step 2: Run (fails).**
- [ ] **Step 3: Implement the prefix scan + substitute mapping.**

`adapter_ids[i] = control id at i ? its adapter index+1 : carry`; carry updates
at control tokens. `embed_ids` uses substitutes. No CUDA.
- [ ] **Step 4: PASS.**
- [ ] **Step 5: Commit** `feat(granite-switch): host token exchange and adapter routing`.

---

### Task 5: CUDA kernels (multipliers + LoRA shrink/expand) with host references

**Files:**
- Create: `libs/lsxcommon/src/granite_dense_switch_inference.cu`
- Create: `tests/test_granite_switch_kernels.cu` (+ CMake OBJECT library like
  `test_granite_cu`)
- Modify: `libs/lsxcommon/CMakeLists.txt`, `tests/CMakeLists.txt`

**Interfaces:**
- Produces:
  - `cudaError_t RunGraniteSwitchRmsNorm(...)`, `RunGraniteSwitchRope(...)`,
    `RunGraniteSwitchAttention(...)`, `RunGraniteSwitchSwiGLU(...)`,
    `RunGraniteSwitchSplitQKV(...)` — copies of the plain Granite kernels
    (`granite_dense_inference.cu:44-490`), same signatures with the
    `GraniteSwitch` prefix.
  - `cudaError_t RunGraniteSwitchScale(cudaStream_t, __half *x, int64_t n, float s)`
  - `cudaError_t RunGraniteSwitchScaleAdd(cudaStream_t, __half *dst, const __half *src, int64_t n, float s)`
  - `cudaError_t RunGraniteSwitchLoraShrink(cudaStream_t, const __half *x, const __half *A, __half *y, int seq, int K, int rank)`
  - `cudaError_t RunGraniteSwitchLoraExpandAdd(cudaStream_t, const __half *y, const __half *B, __half *out, int seq, int rank, int N)`
  - `cudaError_t RunGraniteSwitchApplyLora(cudaStream_t, const __half *x, const GraniteSwitchProjection &proj, const int32_t *d_adapter, __half *out, int seq, int K, int N, __half *y_scratch, int slice_offset, int slice_count)`

- [ ] **Step 1: Write failing kernel tests**

Use a host reference in `double` for:
1. `Scale` and `ScaleAdd` (12.0, 0.175).
2. LoRA: random `x[seq,K]`, `A[rank,K]`, `B[N,rank]`, adapter ids
   `{0,1,2}`; compare `x + B(Ax)` per token, tolerance 2e-2 (FP16).
3. Skip when adapter 0.
4. `RunGraniteSwitchApplyLora` over a qkv projection with 3 slices and a
   gate/up projection with 2 slices.

All buffers `cudaMalloc` in the test, destroyed after each section.

- [ ] **Step 2: Run; expected FAIL (symbols missing).**

- [ ] **Step 3: Implement**

Copy the deterministic kernel block from `granite_dense_inference.cu` (RoPE,
attention, split-K, combine, split-qkv, SwiGLU, RMSNorm) into the switch TU.
Add:

```cpp
__global__ void switch_scale_kernel(__half *x, int64_t n, float s) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) x[i] = __float2half_rn(__half2float(x[i]) * s);
}
__global__ void switch_scale_add_kernel(__half *dst, const __half *src,
                                        int64_t n, float s) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half_rn(
        __half2float(dst[i]) + __half2float(src[i]) * s);
}
// shrink: one block per token; warp 0 has rank lanes
__global__ void switch_lora_shrink_kernel(const __half *x, const __half *A,
                                          __half *y, int seq, int K, int rank) {
    int t = blockIdx.x;
    if (t >= seq) return;
    int r = threadIdx.x;
    if (r >= rank) return;
    float acc = 0.f;
    const __half *xr = x + (int64_t)t * K;
    const __half *ar = A + (int64_t)r * K;
    for (int k = 0; k < K; ++k)
        acc = fmaf(__half2float(xr[k]), __half2float(ar[k]), acc);
    y[(int64_t)t * rank + r] = __float2half_rn(acc);
}
// expand-add: one block per (token, output-tile)
__global__ void switch_lora_expand_add_kernel(const __half *y, const __half *B,
                                              __half *out, int seq, int rank,
                                              int N) {
    int t = blockIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= seq || n >= N) return;
    float acc = 0.f;
    const __half *yr = y + (int64_t)t * rank;
    const __half *br = B + (int64_t)n * rank;
    for (int r = 0; r < rank; ++r)
        acc = fmaf(__half2float(yr[r]), __half2float(br[r]), acc);
    out[(int64_t)t * N + n] = __float2half_rn(
        __half2float(out[(int64_t)t * N + n]) + acc);
}
```

`ApplyLora` launches shrink then expand only for tokens with `adapter>0`.
Because the kernels have no per-token host sync, dispatch per adapter span:
build `adapter` list on device (host already has it) and launch a shrink kernel
that early-exits `adapter[t]==0`; the expand kernel early-exits on a device
`active` mask. No host loop over tokens.

- [ ] **Step 4: Run kernel tests; expected PASS.**
- [ ] **Step 5: Commit** `feat(granite-switch): multiplier and aLoRA kernels`.

---

### Task 6: Generation loop (multipliers + LoRA + KV) and model class

**Files:**
- Modify: `libs/lsxcommon/src/granite_dense_switch_inference.cu`
- Create: `libs/lsxcommon/src/granite_dense_switch_model.h`
- Create: `libs/lsxcommon/src/granite_dense_switch_model.cpp`
- Create: `tests/test_granite_switch_model.cpp`
- Modify: `libs/lsxcommon/src/model_registry.cpp`
- Modify: `tests/test_model_independence_guard.cpp` (`consumers`, `kLifecycle`)
- Modify: `tests/CMakeLists.txt`, `libs/lsxcommon/CMakeLists.txt`

**Interfaces:**
- Produces:
  - `struct GraniteSwitchKvCache` (same fields as `GraniteKvCache` plus
    `int last_adapter = 0`).
  - `cudaError_t RunGraniteSwitchGeneration(cudaStream_t, const GraniteSwitchPinnedWeights&, const std::vector<int32_t>& prompt, int max_new, int eos_id, GraniteSwitchGenerationResult&, sample::ISampler&, GraniteSwitchKvCache*, const GenerationStream*)`.
  - `class GraniteSwitch41_30BModel final : public ModelBase`, `kType =
    "granite-switch-4.1-30b"`.

- [ ] **Step 1: Write the failing model test**

Synthetic geometry + `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1`; assert:
- `Load()` succeeds and `Type() == "granite-switch-4.1-30b"`.
- `Infer` with a control token returns tokens and a non-zero adapter route
  (expose `last_adapter` in the result or inspect the model test hook).
- A second call with the generated tail continues the session and keeps the
  adapter.
- Streaming callback receives every token; pre-set abort stops after the first.
- `TeardownWeights` frees everything.

- [ ] **Step 2: Run (fails).**

- [ ] **Step 3: Implement the loop**

Copy `RunGraniteGeneration` (`granite_dense_inference.cu:824-1006`) and add:
1. `PrepareGraniteSwitchTokens` for prompt and decode token (`carry_adapter`).
2. After embed gather: `RunGraniteSwitchScale(x, seq*H, 12.0f)`.
3. Residual adds: `RunGraniteSwitchScaleAdd(x, proj, seq*H, 0.175f)`.
4. After each base GEMV: `ApplyLora` for qkv (3 slices), o (1), gate_up (2),
   down (1) using `d_adapter` (uploaded per step; for continuation carry
   `last_adapter`).
5. Final norm + tied head GEMV (head.global_scale already /16).
6. Persist `cache->last_adapter`.
7. Same streaming/abort/result fields as Granite.

Implement the model class by copying `granite_dense_model.cpp` and using the
new types; keep the lifecycle exactly (residency lock, `GenerationStream`,
`TeardownContext`, leases, `FillInferenceResult`).

- [ ] **Step 4: Run model test; expected PASS.**
- [ ] **Step 5: Commit** `feat(granite-switch): generation loop and model registration`.

---

### Task 7: SelfCheck, registry, guards and CMake full wiring

**Files:**
- Create: `tests/test_granite_switch_selfcheck.cpp`
- Modify: `tests/CMakeLists.txt` (all switch targets, labels, SKIP 4)
- Modify: `tests/test_model_suite_guard.cpp` (`Suites()` row, tool verdict)
- Modify: `tests/test_model_independence_guard.cpp` (`kLifecycle` row)
- Modify: `AGENTS.md` (Structure bullet)

**Interfaces:**
- Consumes: `GraniteSwitch41_30BModel`.
- Produces: a green `ctest -L fast` guard set.

- [ ] **Step 1: Write the selfcheck test** (3 cycles, bit-exact `final_logits`,
  `SKIP()` when the vault model is absent, `SKIP_RETURN_CODE 4`).

- [ ] **Step 2: Run with no vault → exit 4 (SKIP).**

- [ ] **Step 3: Add the full `Suites()` row**

```cpp
{"granite-switch-4.1-30b",
 {{"convert", {"test_granite_switch_convert"}},
  {"load", {"test_granite_switch_loader"}},
  {"model", {"test_granite_switch_model",
             "test_granite_switch_tool_framing"}},
  {"kernels", {"test_granite_switch_kernels"}},
  {"selfcheck", {"test_granite_switch_selfcheck"}},
  {"fsi", {"test_granite_switch_fsi_selfcheck"}},
  {"code", {"test_granite_switch_code_selfcheck"}},
  {"det", {"test_granite_switch_det_selfcheck"}},
  {"perf", {"test_perf_baseline_granite_switch_4_1_30b"}}}},
```

Create every referenced target (FSI/CODE/DET/perf targets may be stubs that
`SKIP()` until Task 9/10 fills their goldens; the suite guard requires them to
exist with the right labels/skip codes).

- [ ] **Step 4: Run the guard tests**

```bash
cmake --build --preset linux-debug --target test_code_naming_guard \
  test_model_independence_guard test_model_suite_guard test_ctest_labels_guard
ctest --preset linux-debug-fast -R "naming|independence|suite|labels"
```
Expected: PASS.

- [ ] **Step 5: Commit** `feat(granite-switch): selfcheck and guard registries`.

---

### Task 8: Native Granite-4.1 chat/tool codec + adapter_name plumbing

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/granite_switch_tool_format.h`
- Create: `libs/lsxcommon/src/granite_switch_tool_format.cpp`
- Modify: `libs/lsxcommon/include/lsxcommon/chat_conversation.h` (`std::string adapter_name;`)
- Modify: `libs/lsxnetwork/include/lsxnetwork/openai_endpoint.h` + parse/emit
- Modify: `apps/taskmosaic/src/chat_bridge.cpp`
- Modify: `tests/test_code_naming_guard.cpp` (`AllowedCommon`)
- Create: `tests/test_granite_switch_tool_framing.cpp`

**Interfaces:**
- Produces:
  - `bool RenderGraniteSwitchConversation(const ChatConversation&, std::string&)`
  - `std::vector<ToolCall> ParseGraniteSwitchToolCalls(std::string_view)`
  - `bool AppendGraniteSwitchIds(const ArrowTokenizerDb&, const std::string&, std::vector<int32_t>&)`
  - `ChatConversation.adapter_name` populated from
    `chat_template_kwargs.adapter_name`.

- [ ] **Step 1: Failing framing test** with expected literal strings for:
  base system/user/assistant turns; tools block; adapter insertion via
  `adapter_name="guardian-core"` (`<|guardian-core|>guardian>`), via
  `"decision"` (`<|decision|> DECISION`), and explicit control token; parse
  `<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>`.

- [ ] **Step 2: Run (fails).**
- [ ] **Step 3: Implement** the template exactly from
  `chat_template.jinja` (start/end role markers, `\n` placement, tools prefix
  and suffix). Tokenizer id builder injects `<|start_of_role|>`,
  `<|end_of_role|>`, `<|end_of_text|>`, `<|guardian-core|>`, `<|decision|>`,
  `<|audio|>` via lookup; BPE-encode only free text. Adapter activation
  precedence: `adapter_name` > explicit control token > invocation text.
- [ ] **Step 4: PASS.**
- [ ] **Step 5: Commit** `feat(granite-switch): native chat, adapter and tool codec`.

---

### Task 9: Oracle capture, FSI/CODE/DET goldens and adapter parity

**Files:**
- Create: `tools/granite_switch_oracle.py`
- Modify: `tools/fsi_capture.py`, `tools/code_capture.py`, `tools/det_capture.py`
  (`loader: "granite_switch"`, CPU BF16, `adapter_name` option)
- Create: `tests/fixtures/fsi/granite-switch-4.1-30b.json`
- Create: `tests/fixtures/code/granite-switch-4.1-30b.json`
- Create: `tests/fixtures/det/granite-switch-4.1-30b.json`
- Create: `tests/fixtures/adapters/granite-switch-4.1-30b.json`
- Create: `tests/test_granite_switch_{fsi,code,det}_selfcheck.cpp`
- Create: `tests/test_granite_switch_adapter_parity.cpp`
- Modify: `tests/CMakeLists.txt`, `tests/test_model_suite_guard.cpp`

**Interfaces:**
- `granite_switch_oracle.load(model_path, adapter_name=None)` returns a CPU BF16
  `GraniteSwitchForCausalLM`.
- Capture CLI: `python3 tools/fsi_capture.py --model <path> --model-type
  granite-switch-4.1-30b --out tests/fixtures/fsi/granite-switch-4.1-30b.json`.

- [ ] **Step 1: Add the failing gate tests** (mirror
  `test_granite_dense_fsi_selfcheck.cpp`; `SKIP()` on missing golden/vault,
  exit 4).
- [ ] **Step 2: Implement the oracle loader**

```python
def load(model_path, adapter_name=None):
    from transformers import AutoTokenizer
    from granite_switch.hf.modeling_granite_switch import GraniteSwitchForCausalLM
    tok = AutoTokenizer.from_pretrained(model_path)
    model = GraniteSwitchForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cpu",
        low_cpu_mem_usage=True)
    return tok, model
```

Install note in the module docstring:

```bash
python3 -m venv --system-site-packages .venv-gs
.venv-gs/bin/pip install --ignore-requires-python \
    'transformers==5.17.0' 'granite-switch==0.1.1'
```

- [ ] **Step 3: Capture base-mode FSI/CODE/DET goldens** with the native chat
  template; verify FSI stays 3/3 (already measured on the oracle).
- [ ] **Step 4: Capture adapter goldens**: 3 prompts each for `guardian-core`,
  `decision`, and base; record `input_ids`, first-token top-1 and the softmax
  top-5.
- [ ] **Step 5: Run the gates on real weights after Task 11; if red on a
  near-tie, follow the residual-plane procedure before touching goldens.**
- [ ] **Step 6: Commit** `test(granite-switch): oracle capture and parity goldens`.

---

### Task 10: Perf baseline

**Files:**
- Modify: `tests/perf/perf_baselines.json`
- Modify: `tests/test_perf_baseline.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Add the failing pairing test** `TEST_CASE("perf baseline:
  granite-switch-4.1-30b")` + `add_test(test_perf_baseline_granite_switch_4_1_30b
  ... `perf baseline: granite-switch-4.1-30b`)` with label `perf`,
  `SKIP_RETURN_CODE 4`, `LSX_PERF_VAULT_SUBDIR=granite-switch-4.1-30b`.
- [ ] **Step 2: Run it (fails: no baseline row).**
- [ ] **Step 3: Measure through the harness** (`lsxbenchmark --bench`, 2048
  tokens, exclusive GPU) and let it write the row; never hand-edit.
- [ ] **Step 4: Re-run the pairing logic test; PASS.**
- [ ] **Step 5: Commit** `perf(granite-switch): seed the 2048-token baseline`.

---

### Task 11: Real conversion and end-to-end verification

**Files:**
- No source changes unless a defect is found.

- [ ] **Step 1: Convert the real vault model in place**

```bash
build/linux-debug/apps/lsxforge/lsxforge --granite-switch \
  /mnt/fast_storage/huggingface/granite-switch-4.1-30b \
  /mnt/fast_storage/huggingface/granite-switch-4.1-30b
```

- [ ] **Step 2: Run SelfCheck x3** (`lsxbenchmark --selfcheck`), then FSI, CODE,
  DET and adapter parity.
- [ ] **Step 3: If a gate is red, record the source and converted margin at the
  fork, run the bug checklist (global scales, `Free`, codec), and only then try
  a residual plane.** Re-measure VRAM; do not exceed the card.
- [ ] **Step 4: Run the fast/convert/inference ctest presets.**
- [ ] **Step 5: Commit** fixes (if any) with `fix(granite-switch): ...`.

---

### Task 12: Documentation and the stage-9 report

**Files:**
- Modify: `AGENTS.md` (Structure bullet)
- Modify: `docs/OPEN_TASKS.md` (ASR out of scope, context ceiling, perf)
- Modify: `docs/SELFCHECK_MATRIX.md` (model x gate row)
- Create: `docs/model-reports/2026-10-31-granite-switch-4.1-30b.md`

- [ ] **Step 1: Write the report** with the measured evidence (gates, t/s,
  resident GiB, adapter routing, residual decision).
- [ ] **Step 2: Update the open-tasks debt** (ASR absent; 131k context not
  representable on 32 GiB).
- [ ] **Step 3: Run the guard tests one last time.**

```bash
cmake --build --preset linux-debug
ctest --preset linux-debug-fast
ctest --preset linux-debug-convert
```
- [ ] **Step 4: Commit** `docs(granite-switch): model report and open-task notes`.

---

## Self-review

- Spec coverage: passport/converter (1-2), loader (3), routing (4), kernels (5),
  inference/model (6), selfcheck/guards (7), chat/tools/adapter_name (8),
  oracle/gates (9), perf (10), real run (11), report (12).
- No placeholders: every task names exact files, signatures, commands and
  expected result.
- Type consistency: `GraniteSwitchTopology`, `GraniteSwitchPinnedWeights`,
  `GraniteSwitchPrepared`, `GraniteSwitchKvCache`, `RunGraniteSwitchGeneration`
  are defined once and reused by later tasks.
- Deliberate deviations: the adapter parity gate is tolerance-class (source is
  BF16, engine is FP16 + NVFP4 base); the run/convert/perf tasks are the only
  ones allowed to touch the real vault.
