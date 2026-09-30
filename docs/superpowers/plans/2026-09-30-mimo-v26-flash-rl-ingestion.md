# MiMo-V2.6-Flash-RL Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for
> tracking. (This repo forbids Task subagents — AGENTS.md "Autonomous Mode" —
> so inline execution is the only mode.)

**Goal:** Ingest `MiMo-V2.6-Flash-RL` (new `mimo` family, hybrid SWA/full sparse
MoE) into logestix end-to-end: converter, registration, tokenizer/tool bridge,
CUDA kernels, expert cache, selfcheck/FSI/CODE/DET gates and perf baseline.

**Architecture:** New autonomous family `mimo` with its own frozen passport
(`lsxcommon::mimo::frozen`), own FP8-block-128 dequant + attention/pipeline
kernels, reusing only model-neutral infra (`lsx_inference_nvfp4`, `lsx_gemv`,
`PackExpertNVFP4`, `ExpertSchema`, `expert_store`, `schema`, `path_registry`,
`gpu_perf`). Dense backbone resident on the GPU; 12032 MXFP4 experts stored as
NVFP4 in a host mmap store with a device slot-LRU.

**Tech Stack:** C++20 / C23, CUDA 13.4 (sm_120/121 Blackwell), CMake presets,
Catch2 v3.16.0, Apache Arrow, spdlog, abseil.

## Global Constraints

- Target Blackwell CC 12.0/12.1 only (`CUDA_ARCHITECTURES "120-real;121-real"`).
- Language C++20, logging only via spdlog (no `std::cout`/`printf`).
- abseil containers only in host TUs; NEVER in headers reachable from `.cu`/`.cuh`
  (containers/streams in device TUs stay `std::`). `absl::StrFormat` is
  printf-style (`%s/%d`), never `{}`.
- TDD: no production code without a failing test first; every new ctest target
  gets its label (`fast`/`convert`/`inference`/`perf`) in the SAME commit.
- Converter tests: label `convert`, bounded `TIMEOUT` (120 s), synthetic
  checkpoint only, `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1` as a runtime ENV (not a
  compile define). Never convert a real model in a test.
- selfcheck/fsi/code/det targets: `SKIP_RETURN_CODE 4`.
- Frozen geometry: compile-time constants + `static_assert`; runtime config
  reads are legacy.
- Norms, router gate, `e_score_correction_bias`, `attention_sink_bias`,
  embeddings, `lm_head` are never quantized.
- One CUDA stream, no per-token `cudaDeviceSynchronize`; NVFP4 only through the
  neutral `lsx_nvfp4`; shared-memory sandbox (thread-0 RMW + `__syncthreads()`).
- Every model opens `LsxFastMathScope(EffectiveFastMathMode(cfg_))` inside its
  own `Infer`.
- Model independence: `mimo_moe_*` includes no other family's header; test
  suites include only `test_mimo_moe_fixture.h` + `test_synth_checkpoint.h`.
- Commit prefixes: `build:`/`feat:`/`fix:`/`refactor:`/`test:`/`docs:`. Commit
  only on explicit request; here the operator has requested the full phased
  ingestion, so commit per task.
- Branch: `feat/model-MiMo-V2.6-Flash-RL` (already created).
- Source checkpoint (read-only): `/mnt/llm_models/models/MiMo-V2.6-Flash-RL`.
- Spec: `docs/superpowers/specs/2026-09-30-mimo-v26-flash-rl-ingestion-design.md`
  (authoritative for all formulas/geometry; do not restate, follow it).

## File Structure

Create:

- `libs/lsxcommon/src/mimo_moe_constants.h` — frozen passport (leaf).
- `libs/lsxcommon/src/mimo_moe_dequant.cu` — FP8-block-128 → BF16; MXFP4 → BF16.
- `libs/lsxcommon/src/mimo_moe_kernels.{h,cu}` — RMSNorm, SwiGLU, router.
- `libs/lsxcommon/src/mimo_moe_load.cpp` — loader.
- `libs/lsxcommon/src/mimo_moe_model.cpp` — `IModel` subclass + token bridge.
- `libs/lsxcommon/src/mimo_moe_expert_cache.cpp` — host store + LRU.
- `libs/lsxcommon/src/mimo_moe_inference_attn.cu` — attention.
- `libs/lsxcommon/src/mimo_moe_inference_pipeline.cu` — layer pipeline + decode.
- `libs/lsxcommon/src/mimo_moe_device_math.h` — host reference math.
- `libs/lsxcommon/src/mimo_tool_format.cpp` — ChatML tool codec.
- `libs/lsxcommon/include/lsxcommon/mimo_moe_{convert,load,model,expert_cache}.h`
- `libs/lsxcommon/include/lsxcommon/mimo_moe_inference_{attn,pipeline}.h`
- `libs/lsxcommon/include/lsxcommon/mimo_tool_format.h`
- `tests/test_mimo_moe_fixture.h`
- `tests/test_mimo_moe_{convert,loader,model,selfcheck,fsi_selfcheck,code_selfcheck,det_selfcheck}.cpp`
- `tests/test_mimo_moe_kernels.cu`
- `tests/test_mimo_moe_geometry.cpp`
- `tests/test_mimo_tool_framing.cpp`

Modify (additive only):

- `libs/lsxcommon/CMakeLists.txt` — sources + `lsxcommon_mimo_cu` OBJECT lib.
- `libs/lsxcommon/src/model_registry.cpp` — `Register<MiMoV26FlashModel>()`.
- `apps/lsxforge/src/main.cpp` — `--mimo` flag + dispatch.
- `tests/CMakeLists.txt` — targets/labels.
- `tests/test_perf_baseline.cpp`, `tests/perf/perf_baselines.json`.
- `tests/test_code_naming_guard.cpp`, `tests/test_model_independence_guard.cpp`,
  `tests/test_model_suite_guard.cpp`.
- `AGENTS.md`, `.opencode/skills/logestix-repo-map/SKILL.md`.

---

### Task 1: Frozen passport + guard registries

**Files:**
- Create: `libs/lsxcommon/src/mimo_moe_constants.h`
- Create: `tests/test_mimo_moe_geometry.cpp`
- Modify: `tests/test_code_naming_guard.cpp`, `tests/test_model_independence_guard.cpp`, `tests/test_model_suite_guard.cpp`, `tests/CMakeLists.txt`

**Interfaces:**
- Produces: namespace `lsxcommon::mimo::frozen` with `kType`, `kHidden`,
  `kLayers`, `kVocab`, `kFullAttnLayers` (`std::array<int,9>`), `kMoeFirstLayer`,
  `kMtpLayers`, `kHeads`, `kKvHeadsGa`, `kKvHeadsSwa`, `kQkHeadDim`,
  `kVHeadDim`, `kRopeDim`, `kRopeThetaGa`, `kRopeThetaSwa`, `kSlidingWindow`,
  `kValueScale`, `kQSize`, `kKSizeGa`, `kKSizeSwa`, `kVSizeGa`, `kVSizeSwa`,
  `kOSize`, `kAttnScale`, `kEosId`, `kPadId`, `kBosId`, `kImStartId`,
  `IsFullAttnLayer(int)`, `IsMoeLayer(int)`.

- [ ] **Step 1: Write the failing geometry test**

Create `tests/test_mimo_moe_geometry.cpp`:

```cpp
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include "mimo_moe_constants.h"

using namespace lsxcommon::mimo::frozen;

TEST_CASE("mimo frozen geometry is internally consistent", "[mimo][geometry]")
{
    STATIC_REQUIRE(kLayers == 48);
    STATIC_REQUIRE(kHidden == 4096);
    STATIC_REQUIRE(kVocab == 152576);
    STATIC_REQUIRE(kFullAttnLayers.size() == 9);
    STATIC_REQUIRE(kQSize == kHeads * kQkHeadDim);
    STATIC_REQUIRE(kOSize == kHeads * kVHeadDim);
    STATIC_REQUIRE(kRopeDim == 64);
    REQUIRE(std::fabs(kAttnScale - std::pow(192.0, -0.5)) < 1e-9);
    REQUIRE(IsFullAttnLayer(0));
    REQUIRE(IsFullAttnLayer(47));
    REQUIRE_FALSE(IsFullAttnLayer(1));
    REQUIRE(IsMoeLayer(1));
    REQUIRE_FALSE(IsMoeLayer(0));
}
```

- [ ] **Step 2: Run it, verify it fails to compile**

Run: `cmake --preset linux-debug && cmake --build --preset linux-debug --target test_mimo_moe_geometry`
Expected: FAIL — `mimo_moe_constants.h` not found.

- [ ] **Step 3: Write the passport**

Create `libs/lsxcommon/src/mimo_moe_constants.h` with the frozen block:

```cpp
#pragma once
#include <array>
#include <cstdint>

namespace lsxcommon {
namespace mimo {
namespace frozen {

inline constexpr const char* kType = "mimo-v2.6-flash-rl";
inline constexpr int   kHidden            = 4096;
inline constexpr int   kLayers            = 48;
inline constexpr int   kVocab             = 152576;
inline constexpr int   kDenseIntermediate = 16384;
inline constexpr int   kMoeIntermediate   = 2048;
inline constexpr int   kNumRouted         = 256;
inline constexpr int   kTopK              = 8;
inline constexpr float kRmsEps            = 1e-6f;
inline constexpr int   kMoeFirstLayer     = 1;   // layer 0 is dense
inline constexpr std::array<int, 3> kMtpLayers = {0, 1, 2};

inline constexpr int kHeads      = 64;
inline constexpr int kKvHeadsGa  = 4;
inline constexpr int kKvHeadsSwa = 8;
inline constexpr int kQkHeadDim  = 192;
inline constexpr int kVHeadDim   = 128;
inline constexpr int kRopeDim    = 64;
inline constexpr double kRopeThetaGa  = 10000000.0;
inline constexpr double kRopeThetaSwa = 10000.0;
inline constexpr int   kSlidingWindow = 128;
inline constexpr float kValueScale    = 0.707f;

inline constexpr int kQSize   = kHeads * kQkHeadDim;         // 12288
inline constexpr int kKSizeGa = kKvHeadsGa  * kQkHeadDim;    // 768
inline constexpr int kVSizeGa = kKvHeadsGa  * kVHeadDim;     // 512
inline constexpr int kKSizeSwa = kKvHeadsSwa * kQkHeadDim;   // 1536
inline constexpr int kVSizeSwa = kKvHeadsSwa * kVHeadDim;    // 1024
inline constexpr int kOSize    = kHeads * kVHeadDim;         // 8192
inline constexpr float kAttnScale = 0.07216878364870322f;    // 192^-0.5

inline constexpr std::array<int, 9> kFullAttnLayers = {0, 5, 11, 17, 23, 29, 35, 41, 47};

inline constexpr int32_t kEosId     = 151645;  // <|im_end|>
inline constexpr int32_t kPadId     = 151643;  // <|endoftext|>
inline constexpr int32_t kBosId     = -1;      // no BOS
inline constexpr int32_t kImStartId = 151644;  // <|im_start|>

inline constexpr int kSinkHeads = kHeads;      // SWA attention_sink_bias [64]

inline constexpr bool IsFullAttnLayer(int i)
{
    for (int l : kFullAttnLayers) if (l == i) return true;
    return false;
}
inline constexpr bool IsMoeLayer(int i) { return i >= kMoeFirstLayer && i < kLayers; }

static_assert(kQSize == 12288, "Q geometry");
static_assert(kKSizeGa + kVSizeGa == 13568, "GA qkv rows");
static_assert(kKSizeSwa + kVSizeSwa == 14848, "SWA qkv rows");
static_assert(kRopeDim % 2 == 0, "rope dim must be even");
static_assert(kFullAttnLayers.size() == 9, "9 GA layers");

} // namespace frozen
} // namespace mimo
} // namespace lsxcommon
```

- [ ] **Step 4: Register the family in the three guards**

  - `tests/test_code_naming_guard.cpp`: add `mimo` to the `fam` string and to
    `FamilyStems()`; add `"mimo_tool_format"` to `kAllowed` in `AllowedCommon()`.
  - `tests/test_model_independence_guard.cpp`: add
    `{"mimo-v2.6-flash-rl", "mimo_moe_constants.h", "lsxcommon::mimo::frozen"}`
    to `kPassports`; add `{"mimo_moe_constants.h", "namespace frozen {"}` to the
    `owner` map; add `"mimo_moe_model.cpp"` to `consumers`.
  - `tests/test_model_suite_guard.cpp`: add `"mimo"` to `FamilyStems()` and to
    the `model_stage_re` alternation; the `Suites()` row lands in Task 12.

- [ ] **Step 5: Wire the geometry test target**

In `tests/CMakeLists.txt` add (host-only pattern):

```cmake
add_executable(test_mimo_moe_geometry test_mimo_moe_geometry.cpp)
target_include_directories(test_mimo_moe_geometry PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
target_link_libraries(test_mimo_moe_geometry PRIVATE lsxcommon Catch2::Catch2WithMain)
add_test(NAME test_mimo_moe_geometry COMMAND test_mimo_moe_geometry)
set_tests_properties(test_mimo_moe_geometry PROPERTIES LABELS fast)
```

- [ ] **Step 6: Build and run**

Run: `cmake --build --preset linux-debug --target test_mimo_moe_geometry test_code_naming_guard test_model_independence_guard test_model_suite_guard && ctest --preset linux-debug -R "mimo_moe_geometry|code_naming_guard|model_independence_guard|model_suite_guard"`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add libs/lsxcommon/src/mimo_moe_constants.h tests/test_mimo_moe_geometry.cpp \
  tests/test_code_naming_guard.cpp tests/test_model_independence_guard.cpp \
  tests/test_model_suite_guard.cpp tests/CMakeLists.txt
git commit -m "feat: MiMo-V2.6-Flash-RL frozen passport and guard registries"
```

---

### Task 2: ChatML token bridge + tool format (stage 4b)

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/mimo_tool_format.h`
- Create: `libs/lsxcommon/src/mimo_tool_format.cpp`
- Create: `tests/test_mimo_tool_framing.cpp`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Produces:
  - `bool mimo::BuildChatInputIds(const ArrowTokenizerDb& tok, const ChatConversation& conv, bool add_generation_prompt, bool enable_thinking, std::vector<int32_t>& out_ids);`
  - `std::vector<ToolCall> mimo::ParseChatToolCalls(const ArrowTokenizerDb& tok, const std::vector<int32_t>& gen_tokens);`
- Consumes: `arrow_tokenizer_lookup`, `arrow_tokenizer_encode`,
  `arrow_tokenizer_decode`, `tools::ParseToolCallsText`,
  `tools::ParseBareToolCallJson`.

Verified facts (from the checkpoint's `chat_template.jinja` + `tokenizer.json`):
`<|im_start|>`=151644 / `<|im_end|>`=151645 are added tokens; `<think>`,
`<tool_call>`, `<function=`, `<parameter=`, `<tools>` are plain BPE text.

- [ ] **Step 1: Write the failing framing test**

Create `tests/test_mimo_tool_framing.cpp` (host-only, synthetic tokenizer in
`tests/data`, same fixture source as `test_qwen_tool_framing.cpp`; add the
sentinels `<|im_start|>`/`<|im_end|>` if missing):

```cpp
#include <catch2/catch_test_macros.hpp>
#include <string>
#include <vector>
#include "lsxcommon/mimo_tool_format.h"
#include "lsxcommon/tokenizer.h"

namespace {
lsxcommon::ArrowTokenizerDb LoadSynthetic() { /* same loader as test_qwen_tool_framing.cpp */ }
}

TEST_CASE("mimo chat template renders a system+user+assistant frame", "[mimo][tools]")
{
    auto tok = LoadSynthetic();
    lsxcommon::ChatConversation conv;
    conv.messages.push_back({"system", "You are helpful.", "", {}});
    conv.messages.push_back({"user", "hi", "", {}});
    std::vector<int32_t> ids;
    REQUIRE(lsxcommon::mimo::BuildChatInputIds(tok, conv, /*add_generation_prompt=*/true,
                                               /*enable_thinking=*/false, ids));
    // assistant generation prompt prefilled with empty think block
    const auto text = DecodeAll(tok, ids);
    REQUIRE(text.find("<|im_start|>system\nYou are helpful.<|im_end|>") != std::string::npos);
    REQUIRE(text.find("<|im_start|>user\nhi<|im_end|>") != std::string::npos);
    REQUIRE(text.ends_with("<|im_start|>assistant\n<think></think>"));
}

TEST_CASE("mimo native tool call parses from decoded text", "[mimo][tools]")
{
    auto tok = LoadSynthetic();
    const std::string body =
        "<|im_start|>assistant\n<think></think>"
        "<tool_call><function=get_weather><parameter=city>Paris</parameter>"
        "</function></tool_call><|im_end|>";
    std::vector<int32_t> ids;
    REQUIRE(lsxcommon::arrow_tokenizer_encode(tok, body.data(), body.size(), ids.data(),
                                              /*cap*/nullptr) > 0); // test util: encode helper
    auto calls = lsxcommon::mimo::ParseChatToolCalls(tok, ids);
    REQUIRE(calls.size() == 1);
    REQUIRE(calls[0].name == "get_weather");
    REQUIRE(calls[0].arguments.find("Paris") != std::string::npos);
}
```

- [ ] **Step 2: Run it, verify it fails**

Run: `cmake --build --preset linux-debug --target test_mimo_tool_framing`
Expected: FAIL — `mimo_tool_format.h` not found.

- [ ] **Step 3: Implement the codec**

`src/mimo_tool_format.cpp`:
- `RenderToolsBlock`: `You are provided with the following tools:\n\n<tools>` +
  one `\n`+`tojson(tool)` per tool + `\n</tools>`.
- `BuildChatInputIds`: emit `<|im_start|>` id, BPE-encode `role` + `\n` + body +
  `<|im_end|>` id; tools in the system turn (and in any turn carrying
  `message.tools`); assistant turns emit `<think>` + reasoning + `</think>` +
  content + the `<tool_call>` block; when `add_generation_prompt`, append
  `<|im_start|>assistant\n` and, if `!enable_thinking`, `<think></think>`.
  Neutralize control markers in untrusted content with
  `tools::NeutralizeControlMarkers`.
- `ParseChatToolCalls`: decode ids, extract the
  `<tool_call>…</tool_call>` region, split on `<function=` / `<parameter=`, build
  `ToolCall{name, arguments-as-JSON}`; fall back to
  `tools::ParseToolCallsText` then `tools::ParseBareToolCallJson`.

Header declares the two functions in `namespace lsxcommon::mimo`.

- [ ] **Step 4: Wire the target and run**

```cmake
add_executable(test_mimo_tool_framing test_mimo_tool_framing.cpp)
target_compile_definitions(test_mimo_tool_framing PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(test_mimo_tool_framing PRIVATE lsxcommon Catch2::Catch2WithMain)
add_test(NAME test_mimo_tool_framing COMMAND test_mimo_tool_framing)
set_tests_properties(test_mimo_tool_framing PROPERTIES LABELS fast)
```

Run: `ctest --preset linux-debug -R test_mimo_tool_framing`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/mimo_tool_format.h \
  libs/lsxcommon/src/mimo_tool_format.cpp tests/test_mimo_tool_framing.cpp \
  tests/CMakeLists.txt
git commit -m "feat: MiMo ChatML token bridge and tool-call codec (stage 4b)"
```

---

### Task 3: Converter (stage 1+2) — synthetic E2E

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/mimo_moe_convert.h`
- Create: `libs/lsxcommon/src/mimo_moe_convert.cpp`
- Create: `libs/lsxcommon/src/mimo_moe_dequant.cu`
- Create: `tests/test_mimo_moe_fixture.h`
- Create: `tests/test_mimo_moe_convert.cpp`
- Modify: `libs/lsxcommon/CMakeLists.txt`, `apps/lsxforge/src/main.cpp`, `tests/CMakeLists.txt`

**Interfaces:**
- Produces:
  - `arrow::Status lsxcommon::mimo::ConvertMiMoModel(const std::string& input_path, const std::string& output_root);`
  - `arrow::Status lsxcommon::mimo::DequantFp8Block128ToBf16Gpu(const std::vector<uint8_t>& fp8, const std::vector<int64_t>& shape, const std::vector<uint8_t>& scale, const std::vector<int64_t>& scale_shape, std::vector<uint8_t>& out_bf16);`
  - `arrow::Status lsxcommon::mimo::DequantMxfp4ToBf16Gpu(const std::vector<uint8_t>& packed, const std::vector<int64_t>& shape, const std::vector<uint8_t>& e8m0, const std::vector<int64_t>& scale_shape, std::vector<uint8_t>& out_bf16);`
- Consumes: `gpu::quantize_tensor_gpu`, `lsxcommon::PackExpertNVFP4`,
  `DenseNfp4Schema`, `ExpertSchema`, `WriteRecordBatchToFile`,
  `lsx_model_metadata`.

- [ ] **Step 1: Write the synthetic fixture + failing E2E test**

`tests/test_mimo_moe_fixture.h` re-exports `test_synth_checkpoint.h` primitives
(`synthtest::FakeTensor`, `WriteSafetensors`, `MakeBf16*`, `TensorNamesIn`).
Write `config.json` for a 2-layer synthetic model (one GA + one MoE) plus a
`*.safetensors` with: `model.embed_tokens.weight`, `lm_head.weight`,
`model.norm.weight`, layer 0 `input_layernorm`/`post_attention_layernorm`,
`self_attn.qkv_proj.weight` (F8_E4M3, rows a multiple of 128 — GA quirk test
uses a scale with **one spare row block**), `self_attn.o_proj.weight` (BF16),
layer 1 `self_attn.attention_sink_bias` BF16 [64], `mlp.gate.weight` BF16
[256,64], `mlp.gate.e_score_correction_bias` F32 [256], and 4 experts' MXFP4
`{gate,up,down}_proj.weight` (U8) + `.weight_scale` (U8 E8M0 per-16 elements in
the fixture). Add an unknown multimodal tensor to assert it is stripped.

`tests/test_mimo_moe_convert.cpp` (label `convert`, TIMEOUT 120,
`ENVIRONMENT LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1`):

```cpp
#include <catch2/catch_test_macros.hpp>
#include "test_mimo_moe_fixture.h"
#include "lsxcommon/mimo_moe_convert.h"

TEST_CASE("mimo converter emits a valid logestix-mimo-nvfp4 layout", "[mimo][convert]")
{
    auto tmp = mimotest::MakeSyntheticCheckpoint();          // fixture helper
    auto out = tmp / "converted";
    REQUIRE(lsxcommon::mimo::ConvertMiMoModel(tmp.string(), out.string()).ok());
    const auto idx = mimotest::ReadJson(out / "logestix" / "model.index.json");
    CHECK(idx["format"] == "logestix-mimo-nvfp4");
    CHECK(idx["num_layers"] == 2);
    CHECK(idx["deferred"]["mtp_layers"].size() == 3);        // counted, not applied
    CHECK(mimotest::ExistsExpertFile(out, 1, 0));            // NVFP4 expert written
    CHECK_FALSE(mimotest::HasUnknownTensorError());          // multimodal stripped
}
```

- [ ] **Step 2: Run it, verify it fails**

Run: `cmake --build --preset linux-debug --target test_mimo_moe_convert`
Expected: FAIL — converter not implemented.

- [ ] **Step 3: Implement the dequant kernels**

`src/mimo_moe_dequant.cu`:
- `mimo_fp8_block128_to_bf16_kernel`: one thread per element; `v =
  F8E4m3ToFloat(fp8[r*C+c]) * scale[(r/128)*sc_cols + (c/128)]`; index by
  `r/128` so a `scale_rows·128 > R` tail is never read; launch one stream,
  `cudaMemcpyAsync` back. Return `arrow::Status`.
- `mimo_mxfp4_to_bf16_kernel`: nibble `n = (packed[(r*C+c)/2] >> (4*((r*C+c)&1))) & 0xF`,
  decode E2M1 via the shared codec, multiply by `E8M0 e8m0[(r/32)*sc_cols +
  (c/32)]`, write BF16.

- [ ] **Step 4: Implement the converter**

`src/mimo_moe_convert.cpp`:
- Parse `config.json`; validate against `frozen` unless
  `SyntheticGeometryAllowed()`.
- Register tensors by name in a family classifier (`classify_mimo_tensor`).
- Reject/strip `visual.*`, `audio_encoder.*`, `speech_embeddings.*`; store +
  count `model.mtp.*`.
- Globals → BF16 pass-through (`PinnedSchema`).
- Dense FP8 → `DequantFp8Block128ToBf16Gpu` → `gpu::quantize_tensor_gpu` →
  `DenseNvfp4Schema` records.
- Experts (`mlp.experts.E.{gate,up,down}_proj` U8 + `.weight_scale` U8): build
  `ExpertData` (w1=gate, w3=up, w2=down) → `PackExpertNVFP4`; on failure of the
  exact repack (`ark::Status`/false), `DequantMxfp4ToBf16Gpu` →
  `quantize_tensor_gpu`; write `experts/layer_N/exp_E.arrow`.
- Norms/`o_proj`/gate/`e_score_correction_bias`/`attention_sink_bias` BF16 or
  F32 pass-through verbatim.
- Write `logestix/model.index.json` with format/geometry/layer_types/vram/files/
  deferred, under `lsx_model_metadata`-relative output.
- Build the tokenizer arrow files via `BuildTokenizerArrowFiles`.

- [ ] **Step 5: `--mimo` forge flag + CMake wiring**

In `apps/lsxforge/src/main.cpp` add `ABSL_FLAG(bool, mimo, false, "MiMo-V2.6-Flash-RL ...")`
and a dispatch branch calling `lsxcommon::mimo::ConvertMiMoModel(input_path, output_path)`.
Register `mimo_moe_convert.cpp` (host) + `mimo_moe_dequant.cu` in
`libs/lsxcommon/CMakeLists.txt` (new `lsxcommon_mimo_cu` OBJECT lib,
`CUDA_ARCHITECTURES "120-real;121-real"`, linked PUBLIC into `lsxcommon`).

- [ ] **Step 6: Run green**

Run: `ctest --preset linux-debug -R test_mimo_moe_convert`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/mimo_moe_convert.h \
  libs/lsxcommon/src/mimo_moe_convert.cpp libs/lsxcommon/src/mimo_moe_dequant.cu \
  libs/lsxcommon/CMakeLists.txt apps/lsxforge/src/main.cpp \
  tests/test_mimo_moe_fixture.h tests/test_mimo_moe_convert.cpp tests/CMakeLists.txt
git commit -m "feat: MiMo-V2.6-Flash-RL converter (FP8 dense, MXFP4 experts, stripped multimodal)"
```

---

### Task 4: Loader (stage 3) — format/geometry gates on synthetic fixture

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/mimo_moe_load.h`
- Create: `libs/lsxcommon/src/mimo_moe_load.cpp`
- Create: `tests/test_mimo_moe_loader.cpp`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Produces: `struct mimo::MiMoModelFiles` (paths to globals/layers/experts) and
  `arrow::Result<MiMoModelFiles> mimo::LoadMiMoLayout(const std::string& root);`
  plus `arrow::Status mimo::ValidateMimoGeometry(const MiMoTopology&);`
- Consumes: `DenseNvfp4Schema`, `ExpertSchema`, `expert_store::ExpertStore`.

- [ ] **Step 1: Write the failing loader test**

`tests/test_mimo_moe_loader.cpp` builds a synthetic converted fixture (reuse
`test_mimo_moe_fixture.h`) and asserts: valid layout loads; a wrong format tag
is rejected; a geometry mismatch vs `frozen` is rejected naming the field; a
missing per-layer file is rejected naming the tensor. Label `inference`.

- [ ] **Step 2: Run red**
Run: `cmake --build --preset linux-debug --target test_mimo_moe_loader`
Expected: FAIL.

- [ ] **Step 3: Implement loader**

Parse `model.index.json`; `REQUIRE format == "logestix-mimo-nvfp4"`; cross-check
geometry against `frozen` (unless synthetic); resolve per-layer pinned files and
expert directories; produce diagnostics naming any missing/mis-shaped tensor.

- [ ] **Step 4: Wire target, run green, commit**

```cmake
add_executable(test_mimo_moe_loader test_mimo_moe_loader.cpp)
target_link_libraries(test_mimo_moe_loader PRIVATE lsxcommon Catch2::Catch2WithMain)
add_test(NAME test_mimo_moe_loader COMMAND test_mimo_moe_loader)
set_tests_properties(test_mimo_moe_loader PROPERTIES LABELS inference)
```

```bash
git commit -am "feat: MiMo loader with format and frozen-geometry gates"
```

---

### Task 5: Expert cache (stage 6 support)

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/mimo_moe_expert_cache.h`
- Create: `libs/lsxcommon/src/mimo_moe_expert_cache.cpp`
- Create: `tests/test_mimo_moe_expert_cache.cpp`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Produces: `class mimo::MimoExpertCache` wrapping
  `gpu::expert_store::ExpertStore<MimoExpertEntry>`, with
  `int Slots() const`, `void Touch(int layer, int expert)`,
  `bool Resident(int layer, int expert)`, `uint64_t HostBytes() const`; knob
  `LSX_MIMO_EXPERT_CACHE_SLOTS`.

- [ ] **Step 1-3:** Write a failing LRU bookkeeping test (touch > slots evicts
  the least-recently-used; hit keeps residency), implement over `expert_store`,
  run green.
- [ ] **Step 4: Commit** `feat: MiMo expert host store and slot-LRU`.

---

### Task 6: Kernels A — norms, SwiGLU, router (stage 5)

**Files:**
- Create: `libs/lsxcommon/src/mimo_moe_device_math.h`
- Create: `libs/lsxcommon/src/mimo_moe_kernels.{h,cu}`
- Create: `tests/test_mimo_moe_kernels.cu`
- Modify: `libs/lsxcommon/CMakeLists.txt`, `tests/CMakeLists.txt`

**Interfaces:**
- Produces (declared in `mimo_moe_kernels.h`, defined in the `.cu`):
  - `arrow::Status RunMimoRmsNorm(const float* x, const float* w, int rows, int cols, float eps, float* out);`
  - `arrow::Status RunMimoSwiGLU(const float* x, const void* w1, const void* w3, const void* w2, ...)` (NVFP4 through `lsx_nvfp4`);
  - `arrow::Status RunMimoRouter(const float* x, const float* gate_w, const float* bias, int rows, int* idx, float* w, cudaStream_t);`

- [ ] **Step 1: Write failing parity tests** in `tests/test_mimo_moe_kernels.cu`
  (device vs the host reference in `mimo_moe_device_math.h`), covering RMSNorm
  (eps 1e-6), SwiGLU, and the sigmoid `noaux_tc` router (`topk8` over
  `sigmoid(logits)+bias`, weights from un-biased sigmoid, normalized by sum).
  Use `-fmad=false` (OBJECT lib) and bit-exact comparison.
- [ ] **Step 2:** Build red.
- [ ] **Step 3:** Implement kernels per the pinned formulas in the spec; the
  router kernel must not use `atomicAdd` in anything feeding `final_logits`.
- [ ] **Step 4:** Build green; label `inference`; commit
  `feat: MiMo RMSNorm/SwiGLU/router kernels`.

---

### Task 7: Kernels B — attention (stage 5)

**Files:**
- Create: `libs/lsxcommon/src/mimo_moe_inference_attn.cu`
- Create: `libs/lsxcommon/include/lsxcommon/mimo_moe_inference_attn.h`
- Create: `tests/test_mimo_moe_attn.cu`
- Modify: `libs/lsxcommon/CMakeLists.txt`, `tests/CMakeLists.txt`

**Interfaces:**
- Produces: `struct MimoAttnState` (KV ring for SWA, full KV for GA) and
  `arrow::Status RunMimoAttention(const float* x, int layer, bool is_swa, MimoAttnState&, cudaStream_t);`

- [ ] **Step 1: Write failing parity tests** for: fused-QKV split (GA and SWA
  shapes), partial rotate-half RoPE on the first 64 dims (GA θ=1e7 / SWA
  θ=1e4), GQA repeat (4 and 8 KV heads), causal mask, sliding-window mask (128),
  sink-biased softmax (SWA), `v_scale=0.707`, and a decode step reusing the
  cache.
- [ ] **Step 2:** Build red.
- [ ] **Step 3:** Implement per the pinned attention formula; single stream,
  shared-memory reduction sandbox for the softmax (thread-0 RMW behind
  `__syncthreads()`, no `atomicAdd`).
- [ ] **Step 4:** Build green; label `inference`; commit
  `feat: MiMo hybrid SWA/full attention kernels with sink bias`.

---

### Task 8: Dequant kernel parity (stage 5)

**Files:**
- Create: `tests/test_mimo_moe_dequant.cu`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1:** Failing parity test for `DequantFp8Block128ToBf16Gpu` on a
  matrix whose scale has spare row blocks (GA `[13568,4096]` / scale `[108,32]`)
  and for `DequantMxfp4ToBf16Gpu` (nibble + E8M0 per-32), compared to a scalar
  host reference.
- [ ] **Step 2-4:** Red → implement (already in Task 3) → green; label
  `inference`; commit `test: MiMo FP8/MXFP4 dequant parity`.

---

### Task 9: Pipeline + model + registry + forge flag (stage 3/5)

**Files:**
- Create: `libs/lsxcommon/src/mimo_moe_inference_pipeline.cu`
- Create: `libs/lsxcommon/include/lsxcommon/mimo_moe_inference_pipeline.h`
- Create: `libs/lsxcommon/src/mimo_moe_model.cpp`
- Create: `libs/lsxcommon/include/lsxcommon/mimo_moe_model.h`
- Create: `tests/test_mimo_moe_model.cpp`
- Modify: `libs/lsxcommon/CMakeLists.txt`, `libs/lsxcommon/src/model_registry.cpp`, `tests/CMakeLists.txt`

**Interfaces:**
- Produces: `class MiMoV26FlashModel final : public IModel` (in
  `lsxcommon::mimo`), overriding `Load/Infer/Unload/Inspect/SelfCheck/Type/
  Tokenizer/BuildPromptInputIds/BuildConversationInputIds/ParseToolCalls`.
  `Infer` opens `LsxFastMathScope(EffectiveFastMathMode(cfg_))`.

- [ ] **Step 1:** Failing model test on the synthetic converted fixture
  (`LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1`): factory resolves `kType`, `Load`
  succeeds, a 1→2-token `Infer` runs without error, frozen-geometry mismatch is
  rejected. Label `inference`.
- [ ] **Step 2:** Red.
- [ ] **Step 3:** Implement the layer loop (`embed → 48 × (attn + dense/MoE) →
  norm → lm_head`); route all NVFP4 through `lsx_nvfp4`/`RunLxGemmVec`; wire the
  expert cache; the token bridge overrides delegate to `mimo_tool_format`
  (Task 2). Register `MiMoV26FlashModel` in `model_registry.cpp`.
- [ ] **Step 4:** Green; commit
  `feat: MiMo inference pipeline, IModel, and factory registration`.

---

### Task 10: SelfCheck (stage 8)

**Files:**
- Create: `tests/test_mimo_moe_selfcheck.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1:** Clone the granite selfcheck harness: real model via
  `LSX_TEST_MIMO_MODEL_PATH` (default `lsx_model_models_dir()/
  "MiMo-V2.6-Flash-RL"`), `SKIP()` when `logestix/pinned` absent, `>=3` cycles
  (`LSX_TEST_MIMO_SELFCHECK_CYCLES`), `RunParisSelfCheck(..., {.frame_prompt=true})`
  over `"The capital of France is" → "Paris"`, bit-exact `final_logits` across
  cycles.
- [ ] **Step 2:** Wire target with `LABELS inference` + `SKIP_RETURN_CODE 4`.
- [ ] **Step 3:** Commit `feat: MiMo SelfCheck (Paris, bit-exact, skip-if-absent)`.

---

### Task 11: FSI / CODE / DET gates (stage 8)

**Files:**
- Create: `tests/test_mimo_moe_{fsi,code,det}_selfcheck.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1:** Per the `fsi-selfcheck`, `code-selfcheck`, `det-selfcheck`
  skills: capture `tests/fixtures/{fsi,code,det}/mimo.json` from the source
  safetensors (offline, not in tests), clone the per-family tests, `SKIP 4`.
- [ ] **Step 2:** Wire all three targets (`inference`, `SKIP_RETURN_CODE 4`),
  run, commit `feat: MiMo FSI/CODE/DET selfcheck gates`.

---

### Task 12: Perf baseline + suite-guard row + docs

**Files:**
- Modify: `tests/test_perf_baseline.cpp`, `tests/perf/perf_baselines.json`,
  `tests/test_model_suite_guard.cpp`, `AGENTS.md`,
  `.opencode/skills/logestix-repo-map/SKILL.md`

- [ ] **Step 1:** Add the `Suites()` row:
  `convert → load → model → kernels → selfcheck → fsi → code → det` with targets
  `test_mimo_moe_convert`, `test_mimo_moe_loader`, `test_mimo_moe_model`,
  `{test_mimo_moe_kernels, test_mimo_moe_dequant}`, `test_mimo_moe_selfcheck`,
  `test_mimo_moe_fsi_selfcheck`, `test_mimo_moe_code_selfcheck`,
  `test_mimo_moe_det_selfcheck`.
- [ ] **Step 2:** Add `kMiMo` + `TEST_CASE("perf baseline: mimo-v2.6-flash-rl",
  "[perf]")` to `tests/test_perf_baseline.cpp` and a `mimo-v2.6-flash-rl` entry
  to `tests/perf/perf_baselines.json` (measured after real conversion).
- [ ] **Step 3:** Update `AGENTS.md` Structure and the repo-map skill with the
  `mimo` files/knobs.
- [ ] **Step 4:** Run the four guards + `-DLSX_PRESET=linux-debug -DLSX_LIST=ON
  -P tools/test_router.cmake`; commit `docs: MiMo ingestion docs and suite guard`.
- [ ] **Step 5:** Run the real conversion offline (`lsxforge --mimo --input
  /mnt/llm_models/models/MiMo-V2.6-Flash-RL --output <vault>`), then the
  selfcheck/FSI/CODE/DET/perf suites; backfill baseline values.

---

## Self-Review

- **Spec coverage:** stages 1 (Task 3), 2 (Task 3), 3 (Tasks 4, 9), 4/4b
  (Task 2), 5 (Tasks 6-8), 6 (Tasks 5, 9), 7 (Task 9 passport wiring), 8
  (Tasks 10-12) + guards (Tasks 1, 12) + docs (Task 12). Autonomy invariant is
  enforced in Task 1 and re-checked in Task 12.
- **Placeholders:** none — every step names files, signatures and commands.
  Kernel bodies follow the pinned formulas in the committed spec (no
  "implement later").
- **Type consistency:** `ConvertMiMoModel`, `DequantFp8Block128ToBf16Gpu`,
  `DequantMxfp4ToBf16Gpu`, `RunMimoRmsNorm`, `RunMimoSwiGLU`, `RunMimoRouter`,
  `RunMimoAttention`, `MimoExpertCache`, `MiMoV26FlashModel`,
  `BuildChatInputIds`, `ParseChatToolCalls` are named once and reused.
