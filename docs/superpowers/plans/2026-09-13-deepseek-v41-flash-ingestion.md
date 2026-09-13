# DeepSeek-V4.1-Flash Ingestion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **NOTE (repo policy, AGENTS.md):** subagents are FORBIDDEN in this repository. Execution MUST use `superpowers:executing-plans` inline, in the main session, one task at a time.

**Goal:** Add DeepSeek-V4.1-Flash (`deepseek-v4.1-flash`) as an autonomous model contour in logestix, ending with a bit-exact "Paris" selfcheck on the text backbone.

**Architecture:** New independent `deepseek_moe_v41_*` files (approach B) — no dependency on DeepSeek-V4-Flash code, only model-neutral infrastructure (`model_base`, `model_runtime`, `expert_store`, `lsx_inference_nvfp4`, `tokenizer`, `schema`, `path_registry`, `gpu_perf`, `sampling`). 40-layer CED sparse-attention MoE with `SharedAttentionRuntime`, FP8 32×32 dense, FP4 per-32 experts repacked to NVFP4, hyper-connections, `sqrtsoftplus` router.

**Tech Stack:** C++20, CUDA 13.4 (Blackwell `sm_120/121`), Arrow IPC, simdjson, CMake presets, Catch2.

## Global Constraints

- Forbidden architectures: Hopper (90), Ada (89), Ampere (80); Blackwell only. Max 48 warps/SM.
- All logging via `spdlog`/`lsxcommon::log`; never `std::cout`/`printf`.
- `absl::` containers/strings in host-only C++ (not NVCC-reachable). `.cu`/`.cuh` use `std::`.
- No comments unless the surrounding code has them (repo style: docs comments ARE present in model code — follow neighboring files).
- `static_assert` frozen geometry at load (Model Ingestion Protocol step 3); escape hatch `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY`.
- Converter/loader tests run on SYNTHETIC fixtures only (suite guard); real weights only via env-opt-in + `SKIP_RETURN_CODE 4`.
- New test targets get a CTest label and, if selfcheck, `SKIP_RETURN_CODE 4` in the same commit (`test_ctest_labels_guard`).
- `deepseek_moe_v41_*` must never include any `deepseek_moe_*` V4 header. Cross-family test-header borrowing is forbidden.
- Numerics: `-fmad=false` on the V4.1 CUDA object library; `rms_eps = 1e-20`.
- Windows: only Qwen/Gemma tests run; DeepSeek suites are Linux-only.

---

## Roadmap (7 plans, each independently testable)

| Plan | Deliverable | Stage label | Depends on |
|---|---|---|---|
| **P1** | Frozen passport + topology/classification + fixture | `fast` | — |
| **P2** | Converter (`ConvertDeepSeekV41Model`, `--deepseek_v41`) + tokenizer arrows + synthetic convert e2e | `convert` | P1 |
| **P3** | Loader: pinned upload, FP8 dequant, expert store/LRU, strictness | `fast`/`inference` | P1, P2 |
| **P4** | Device math + HC/Sinkhorn + RMSNorm/RoPE + window sparse attention | `inference` | P3 |
| **P5** | Compressor + `SharedAttentionRuntime` (CED) + Indexer + candidate blocks | `inference` | P4 |
| **P6** | Router/MoE/shared + head + generation driver + `DeepSeekV41FlashModel` + registry + guards | `inference`/`fast` | P5 |
| **P7** | SelfCheck Paris + perf pass + engram decision | `inference` | P6 |

This document details **Plan 1**. Plans 2–7 are written after the prior plan is green.

---

# Plan 1 — Frozen passport, topology, synthetic fixture

**Produces working software:** a host-only topology contract for the
`logestix-v41-nvfp4` layout, locked to the frozen geometry, with tests and
guard registration. No model class is registered yet.

Files:
- Create: `libs/lsxcommon/src/deepseek_moe_v41_constants.h`
- Create: `libs/lsxcommon/src/deepseek_moe_v41_load.h`
- Create: `libs/lsxcommon/src/deepseek_moe_v41_load.cpp`
- Create: `tests/test_deepseek_moe_v41_fixture.h`
- Create: `tests/test_deepseek_moe_v41_model.cpp`
- Create: `tests/test_deepseek_moe_v41_loader.cpp`
- Modify: `tests/CMakeLists.txt`
- Modify: `tests/test_code_naming_guard.cpp`

---

### Task 1: Frozen passport constants

**Interfaces:**
- Consumes: nothing.
- Produces: `lsxcommon::deepseek_v41::{kType, kHidden, kLayers, kMoeIntermediate, kSharedIntermediate, kNumRouted, kTopK, kHeads, kHeadDim, kRopeHeadDim, kQLoraRank, kOLoraRank, kOGroups, kIndexNHeads, kIndexHeadDim, kIndexTopk, kWindow, kVocab, kHcMult, kHcMix, kHcDim, kSinkhornIters, kRmsEps, kHcEps, kRouteScale, kGateTemp, kSwigluLimit, kCompressRopeTheta, kRopeTheta, kRopeFactor, kOriginalSeqLen, kBetaFast, kBetaSlow, kCandidateBlockSize, kCandidateTopkBlocks, kMtpLayers, CompressRatio(int), IsKvSource(int), IsIndexSource(int), kCandidateSourceLayer, IsCandidateSource(int), struct LayerMeta, LayerMetaAt(int)}`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_deepseek_moe_v41_model.cpp`:

```cpp
// tests/test_deepseek_moe_v41_model.cpp — DeepSeek-V4.1-Flash model-stage
// (host-only): frozen passport geometry and the CED per-layer classification.
// The model class itself lands in Plan 6; this stage pins the constants the
// whole contour compiles against (Model Ingestion Protocol step 3).
#include <catch2/catch_test_macros.hpp>

#include "deepseek_moe_v41_constants.h"

using namespace lsxcommon::deepseek_v41;

TEST_CASE("deepseek-v41 frozen geometry is locked", "[deepseek-v41][model]")
{
    CHECK(kHidden == 5120);
    CHECK(kLayers == 40);
    CHECK(kMoeIntermediate == 2304);
    CHECK(kSharedIntermediate == 2304);
    CHECK(kNumRouted == 384);
    CHECK(kTopK == 6);
    CHECK(kHeads == 64);
    CHECK(kHeadDim == 512);
    CHECK(kRopeHeadDim == 64);
    CHECK(kQLoraRank == 1280);
    CHECK(kOLoraRank == 1024);
    CHECK(kOGroups == 8);
    CHECK(kIndexNHeads == 32);
    CHECK(kIndexHeadDim == 128);
    CHECK(kIndexTopk == 512);
    CHECK(kWindow == 128);
    CHECK(kVocab == 129280);
    CHECK(kHcMult == 4);
    CHECK(kHcMix == 24);
    CHECK(kHcDim == 20480);
    CHECK(kSinkhornIters == 20);
    CHECK(kCandidateBlockSize == 8);
    CHECK(kCandidateTopkBlocks == 2048);
    CHECK(kMtpLayers == 3);
}

TEST_CASE("deepseek-v41 CED layer classification matches config",
          "[deepseek-v41][model]")
{
    // compress_ratios: layers 0-1 -> 0, 2..19 -> 2, 20..39 -> 1.
    CHECK(CompressRatio(0) == 0);
    CHECK(CompressRatio(1) == 0);
    CHECK(CompressRatio(2) == 2);
    CHECK(CompressRatio(19) == 2);
    CHECK(CompressRatio(20) == 1);
    CHECK(CompressRatio(39) == 1);

    // kv_source_layers = {2,8,14,20}
    CHECK(IsKvSource(2));
    CHECK(IsKvSource(8));
    CHECK(IsKvSource(14));
    CHECK(IsKvSource(20));
    CHECK_FALSE(IsKvSource(21));

    // index_source_layers = {2,8,14,20,24,28,32,36}
    CHECK(IsIndexSource(2));
    CHECK(IsIndexSource(36));
    CHECK_FALSE(IsIndexSource(35));

    CHECK(kCandidateSourceLayer == 20);
    CHECK(IsCandidateSource(20));
    CHECK_FALSE(IsCandidateSource(21));

    // Every kv source is also an index source; every index source compresses.
    for (int l = 0; l < kLayers; ++l) {
        if (IsKvSource(l)) CHECK(IsIndexSource(l));
        if (IsIndexSource(l)) CHECK(CompressRatio(l) > 0);
    }
    // Candidate source publishes the level-one mask and owns both caches.
    CHECK(IsKvSource(kCandidateSourceLayer));
    CHECK(IsIndexSource(kCandidateSourceLayer));

    const LayerMeta m = LayerMetaAt(20);
    CHECK(m.compress_ratio == 1);
    CHECK(m.is_kv_source);
    CHECK(m.is_index_source);
    CHECK(m.is_candidate_source);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build --preset linux-debug --target test_deepseek_moe_v41_model`
Expected: FAIL — `deepseek_moe_v41_constants.h: No such file or directory`.

- [ ] **Step 3: Write minimal implementation**

Create `libs/lsxcommon/src/deepseek_moe_v41_constants.h`:

```cpp
#pragma once

#include <cstdint>

namespace lsxcommon {
namespace deepseek_v41 {

// Registry id (detect_model_type normalization of "DeepSeek-V4.1-Flash").
inline constexpr char kType[] = "deepseek-v4.1-flash";

// ── Frozen text-backbone geometry (source: config.json text_config +
//    inference/config.json). Locked compile-time per MODEL INGEST step 3.
inline constexpr int kHidden = 5120;
inline constexpr int kLayers = 40;
inline constexpr int kMoeIntermediate = 2304;
inline constexpr int kSharedIntermediate = 2304;
inline constexpr int kNumRouted = 384;
inline constexpr int kTopK = 6;
inline constexpr int kHeads = 64;
inline constexpr int kHeadDim = 512;
inline constexpr int kRopeHeadDim = 64;
inline constexpr int kQLoraRank = 1280;
inline constexpr int kOLoraRank = 1024;
inline constexpr int kOGroups = 8;
inline constexpr int kIndexNHeads = 32;
inline constexpr int kIndexHeadDim = 128;
inline constexpr int kIndexTopk = 512;
inline constexpr int kWindow = 128;              // sliding-window ring size
inline constexpr int kVocab = 129280;
inline constexpr int kHcMult = 4;
inline constexpr int kHcMix = (2 + kHcMult) * kHcMult;  // 24
inline constexpr int kHcDim = kHcMult * kHidden;        // 20480
inline constexpr int kSinkhornIters = 20;
inline constexpr float kRmsEps = 1e-20f;
inline constexpr float kHcEps = 1e-6f;
inline constexpr float kRouteScale = 1.5f;
inline constexpr float kGateTemp = 1.0f;
inline constexpr float kSwigluLimit = 10.0f;
inline constexpr float kCompressRopeTheta = 160000.0f;  // ratio > 0 layers
inline constexpr float kRopeTheta = 10000.0f;           // pure-SWA layers
inline constexpr int kRopeFactor = 16;
inline constexpr int kOriginalSeqLen = 65536;
inline constexpr int kBetaFast = 32;
inline constexpr int kBetaSlow = 1;
inline constexpr int kCandidateBlockSize = 8;
inline constexpr int kCandidateTopkBlocks = 2048;
inline constexpr int kMtpLayers = 3;   // stored-not-applied

// ── CED classification (inference/config.json compress_ratios + source sets).
// Layers 0-1: pure sliding window. 2..19: compressed, ratio 2 (softmax
// pooling). 20..39: compressed, ratio 1 (plain projection). The three MTP
// layers (indices 40..42) are ratio 0 but never run in the backbone.
constexpr int CompressRatio(int layer)
{
    return layer < 2 ? 0 : (layer < 20 ? 2 : 1);
}

constexpr bool IsKvSource(int layer)
{
    return layer == 2 || layer == 8 || layer == 14 || layer == 20;
}

constexpr bool IsIndexSource(int layer)
{
    return layer == 2 || layer == 8 || layer == 14 || layer == 20 ||
           layer == 24 || layer == 28 || layer == 32 || layer == 36;
}

inline constexpr int kCandidateSourceLayer = 20;

constexpr bool IsCandidateSource(int layer)
{
    return layer == kCandidateSourceLayer;
}

struct LayerMeta {
    int compress_ratio;
    bool is_kv_source;
    bool is_index_source;
    bool is_candidate_source;
};

constexpr LayerMeta LayerMetaAt(int layer)
{
    return LayerMeta{CompressRatio(layer), IsKvSource(layer),
                     IsIndexSource(layer), IsCandidateSource(layer)};
}

// ── Compile-time relations (MODEL INGEST INVARIANT step 3).
static_assert(kLayers == 40, "V4.1 backbone is 40 layers");
static_assert(kHidden == 5120, "V4.1 hidden is 5120");
static_assert(kHcDim == kHcMult * kHidden, "hc flattened width");
static_assert(kHcMix == (2 + kHcMult) * kHcMult, "hc mix rows");
static_assert(kHidden % 256 == 0, "hidden tiles cleanly on Blackwell");
static_assert(kMoeIntermediate % 256 == 0, "moe intermediate tiles cleanly");
static_assert(kTopK > 0 && kTopK <= kNumRouted, "bad top-k");
static_assert(kHeadDim > kRopeHeadDim, "rope head must be a sub-vector");
static_assert(kHeads * kHeadDim % kOGroups == 0, "grouped wo_a geometry");
static_assert(kIndexNHeads * kIndexHeadDim == 32 * 128,
              "indexer head geometry");
static_assert(CompressRatio(0) == 0 && CompressRatio(1) == 0,
              "layers 0-1 are pure SWA");
static_assert(CompressRatio(2) == 2 && CompressRatio(19) == 2,
              "layers 2-19 compress at ratio 2");
static_assert(CompressRatio(20) == 1 && CompressRatio(39) == 1,
              "layers 20-39 compress at ratio 1");
static_assert(IsKvSource(2) && IsKvSource(20) && !IsKvSource(21),
              "kv_source set");
static_assert(IsIndexSource(2) && IsIndexSource(36) && !IsIndexSource(35),
              "index_source set");

constexpr bool KvSourceIsIndexSource()
{
    for (int l = 0; l < kLayers; ++l)
        if (IsKvSource(l) && !IsIndexSource(l)) return false;
    return true;
}
static_assert(KvSourceIsIndexSource(), "kv sources must run an indexer");

} // namespace deepseek_v41
} // namespace lsxcommon
```

- [ ] **Step 4: Register the target, run the test, verify it passes**

Modify `tests/CMakeLists.txt` — insert immediately after the
`test_deepseek_moe_model` block (around line 109):

```cmake
# ── DeepSeek-V4.1-Flash model-stage host tests (P1: frozen passport) ─────
add_executable(test_deepseek_moe_v41_model
               test_deepseek_moe_v41_model.cpp)
target_link_libraries(test_deepseek_moe_v41_model PRIVATE lsxcommon
                      Catch2::Catch2WithMain)
target_include_directories(test_deepseek_moe_v41_model PRIVATE
    ${CMAKE_SOURCE_DIR}/libs/lsxcommon/src)
add_test(NAME test_deepseek_moe_v41_model
         COMMAND test_deepseek_moe_v41_model)
```

Then add `test_deepseek_moe_v41_model` to the `fast` label list (the
`set_tests_properties(... PROPERTIES LABELS fast)` block that already lists
`test_deepseek_moe_loader` / `test_deepseek_moe_model`):

```cmake
    test_deepseek_moe_v41_model
```

Run:

```bash
cmake --preset linux-debug
cmake --build --preset linux-debug --target test_deepseek_moe_v41_model
ctest --preset linux-debug -R test_deepseek_moe_v41_model --output-on-failure
```

Expected: PASS (2 test cases).

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/deepseek_moe_v41_constants.h tests/test_deepseek_moe_v41_model.cpp
git commit -m "feat: DeepSeek-V4.1-Flash frozen passport constants"
```

---

### Task 2: Topology contract (`deepseek_moe_v41_load.{h,cpp}`)

**Interfaces:**
- Consumes: Task 1 constants.
- Produces:
  - `lsxcommon::deepseek_v41::kIndexFormat` (`char[]` = `"logestix-v41-nvfp4"`)
  - `struct ModelTopology { int num_layers; int experts_per_layer; int64_t total_experts; lsxcommon::VramProfile vram; std::string format; std::string model; std::array<LayerMeta, 40> layers; std::string error_msg; }`
  - `ModelTopology LoadTopology(const std::string &model_path)`
  - `bool ValidateGeometry(const ModelTopology &topo, std::string *error)`
  - `bool IsSupportedFormat(const ModelTopology &topo)`

- [ ] **Step 1: Write the failing test**

Create `tests/test_deepseek_moe_v41_loader.cpp`:

```cpp
// tests/test_deepseek_moe_v41_loader.cpp — DeepSeek-V4.1-Flash load-stage
// (host-only): the logestix-v41-nvfp4 index contract. Synthetic fixtures only
// (suite guard); real weights live in the selfcheck stage.
#include <catch2/catch_test_macros.hpp>

#include <filesystem>

#include "deepseek_moe_v41_load.h"

#include "test_deepseek_moe_v41_fixture.h"

namespace fs = std::filesystem;

TEST_CASE("deepseek-v41 load_topology parses the synthetic index",
          "[deepseek-v41][load]")
{
    const auto dir = fs::temp_directory_path() / "lsx_v41_topo";
    fs::remove_all(dir);
    REQUIRE(v41fixtest::BuildSyntheticIndex(dir).ok());

    const auto topo = lsxcommon::deepseek_v41::LoadTopology(dir.string());
    CHECK(topo.error_msg.empty());
    CHECK(topo.format == lsxcommon::deepseek_v41::kIndexFormat);
    CHECK(topo.model == "DeepSeek-V4.1-Flash");
    CHECK(topo.num_layers == lsxcommon::deepseek_v41::kLayers);
    CHECK(topo.experts_per_layer == lsxcommon::deepseek_v41::kNumRouted);
    CHECK(topo.total_experts ==
          static_cast<int64_t>(lsxcommon::deepseek_v41::kLayers) *
              lsxcommon::deepseek_v41::kNumRouted);
    CHECK(topo.vram.min_bytes == 1024);
    CHECK(topo.vram.optimal_bytes == 2048);
    CHECK(topo.vram.max_bytes == 4096);

    std::string err;
    CHECK(lsxcommon::deepseek_v41::ValidateGeometry(topo, &err));
    CHECK(err.empty());
    CHECK(lsxcommon::deepseek_v41::IsSupportedFormat(topo));
    fs::remove_all(dir);
}

TEST_CASE("deepseek-v41 load_topology rejects broken JSON",
          "[deepseek-v41][load]")
{
    const auto dir = fs::temp_directory_path() / "lsx_v41_topo_broken";
    fs::remove_all(dir);
    REQUIRE(v41fixtest::BuildSyntheticIndexRaw(dir, "]]] not an object")
                .ok());
    const auto topo = lsxcommon::deepseek_v41::LoadTopology(dir.string());
    CHECK_FALSE(topo.error_msg.empty());
    CHECK(topo.num_layers == 0);
    fs::remove_all(dir);
}

TEST_CASE("deepseek-v41 topology rejects foreign geometry and format",
          "[deepseek-v41][load]")
{
    const auto dir = fs::temp_directory_path() / "lsx_v41_topo_foreign";
    fs::remove_all(dir);
    REQUIRE(v41fixtest::BuildSyntheticIndex(dir, /*num_layers=*/43,
                                            /*experts=*/256,
                                            /*format=*/"logestix-v1-nvfp4")
                .ok());
    const auto topo = lsxcommon::deepseek_v41::LoadTopology(dir.string());
    CHECK_FALSE(topo.error_msg.empty());  // format gate at parse time
    std::string err;
    CHECK_FALSE(lsxcommon::deepseek_v41::ValidateGeometry(topo, &err));
    CHECK_FALSE(err.empty());
    fs::remove_all(dir);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build --preset linux-debug --target test_deepseek_moe_v41_loader`
Expected: FAIL — `deepseek_moe_v41_load.h` / fixture missing.

- [ ] **Step 3: Write minimal implementation**

First create `tests/test_deepseek_moe_v41_fixture.h` (folded scaffolding for
this task's test — the synthetic index writer):

```cpp
#pragma once

// Synthetic DeepSeek-V4.1-Flash index fixture (host-only). P1 needs only the
// index JSON; P2/P3 extend this header with pinned/ and expert Arrow writers.
// Family-neutral helpers (FakeTensor/WriteSafetensors/MakeBf16*) come from
// test_synth_checkpoint.h when P2 lands — never from another family's fixture.
#include <filesystem>
#include <fstream>
#include <string>

#include <arrow/status.h>

namespace v41fixtest {

inline void WriteIndex(const std::filesystem::path &model_dir,
                       const std::string &json)
{
    std::filesystem::create_directories(model_dir / "logestix");
    std::ofstream f(model_dir / "logestix" / "model.index.json");
    f << json;
}

inline std::string DefaultIndexJson(int num_layers, int experts,
                                    const std::string &format)
{
    return "{\n"
           "  \"format\": \"" + format + "\",\n"
           "  \"model\": \"DeepSeek-V4.1-Flash\",\n"
           "  \"num_layers\": " + std::to_string(num_layers) + ",\n"
           "  \"experts_per_layer\": " + std::to_string(experts) + ",\n"
           "  \"total_experts\": " +
           std::to_string(static_cast<long long>(num_layers) * experts) +
           ",\n"
           "  \"vram\": {\n"
           "    \"min_bytes\": 1024,\n"
           "    \"optimal_bytes\": 2048,\n"
           "    \"max_bytes\": 4096,\n"
           "    \"context_tokens\": 1048576\n"
           "  }\n"
           "}\n";
}

inline arrow::Status BuildSyntheticIndex(
    const std::filesystem::path &model_dir, int num_layers = 40,
    int experts = 384, const std::string &format = "logestix-v41-nvfp4")
{
    WriteIndex(model_dir, DefaultIndexJson(num_layers, experts, format));
    return arrow::Status::OK();
}

inline arrow::Status BuildSyntheticIndexRaw(
    const std::filesystem::path &model_dir, const std::string &json)
{
    WriteIndex(model_dir, json);
    return arrow::Status::OK();
}

} // namespace v41fixtest
```

Then create `libs/lsxcommon/src/deepseek_moe_v41_load.h`:

```cpp
#pragma once

#include <array>
#include <cstdint>
#include <string>

#include "lsxcommon/vram_estimate.h"

#include "deepseek_moe_v41_constants.h"

namespace lsxcommon {
namespace deepseek_v41 {

// Sole supported index format for this contour (anti-cross-load gate).
inline constexpr char kIndexFormat[] = "logestix-v41-nvfp4";

struct ModelTopology {
    int num_layers = 0;
    int experts_per_layer = 0;
    int64_t total_experts = 0;
    lsxcommon::VramProfile vram;
    std::string format;
    std::string model;
    std::array<LayerMeta, kLayers> layers{};
    std::string error_msg;
};

// Reads <model_path>/logestix/model.index.json (vault manifest fallback,
// same policy as every other family). Fills error_msg on any failure and
// rejects an index whose `format` is not kIndexFormat.
ModelTopology LoadTopology(const std::string &model_path);

// Frozen-passport cross-check: layer/expert counts must equal the compile-time
// constants. @p error receives a human-readable reason on mismatch.
bool ValidateGeometry(const ModelTopology &topo, std::string *error);

inline bool IsSupportedFormat(const ModelTopology &topo)
{
    return topo.format == kIndexFormat;
}

} // namespace deepseek_v41
} // namespace lsxcommon
```

Create `libs/lsxcommon/src/deepseek_moe_v41_load.cpp`:

```cpp
#include "deepseek_moe_v41_load.h"

#include <filesystem>
#include <fstream>
#include <iterator>

#include <simdjson.h>

#include "lsxcommon/path_registry.h"

namespace lsxcommon {
namespace deepseek_v41 {

ModelTopology LoadTopology(const std::string &model_path)
{
    ModelTopology topo;
    auto ipath =
        std::filesystem::path(model_path) / "logestix" / "model.index.json";
    const auto vault = std::filesystem::path(
        lsxcommon::lsx_model_metadata("manifests/model.index.json"));
    if (!std::filesystem::exists(ipath) && std::filesystem::exists(vault))
        ipath = vault;
    std::ifstream f(ipath);
    if (!f) {
        topo.error_msg = "Cannot open " + ipath.string();
        return topo;
    }
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());

    simdjson::ondemand::parser parser;
    simdjson::padded_string pad(json.data(), json.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(pad).get(doc) != simdjson::SUCCESS) {
        topo.error_msg = "JSON parse error in " + ipath.string();
        return topo;
    }
    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS) {
        topo.error_msg = "Expected JSON object in " + ipath.string();
        return topo;
    }

    std::string verr;
    if (!lsxcommon::ParseVramBlock(json, topo.vram, verr)) {
        ModelTopology failed;
        failed.error_msg = verr + " in " + ipath.string();
        return failed;
    }

    std::string_view fmt;
    if (root["format"].get_string().get(fmt) == simdjson::SUCCESS)
        topo.format.assign(fmt);
    std::string_view mdl;
    if (root["model"].get_string().get(mdl) == simdjson::SUCCESS)
        topo.model.assign(mdl);

    int64_t nl = 0, epl = 0, te = 0;
    root["num_layers"].get_int64().get(nl);
    root["experts_per_layer"].get_int64().get(epl);
    root["total_experts"].get_int64().get(te);
    topo.num_layers = static_cast<int>(nl);
    topo.experts_per_layer = static_cast<int>(epl);
    topo.total_experts = te;

    // Per-layer CED metadata, when present, is the source of truth for the
    // runtime; otherwise the frozen passport fills it in.
    for (int l = 0; l < kLayers; ++l) topo.layers[l] = LayerMetaAt(l);
    simdjson::ondemand::array arr;
    if (root["layers"].get_array().get(arr) == simdjson::SUCCESS) {
        int l = 0;
        for (auto elem : arr) {
            if (l >= kLayers) break;
            simdjson::ondemand::object lo;
            if (elem.get_object().get(lo) != simdjson::SUCCESS) break;
            int64_t cr = 0;
            bool kv = false, ix = false, cand = false;
            lo["compress_ratio"].get_int64().get(cr);
            lo["is_kv_source"].get_bool().get(kv);
            lo["is_index_source"].get_bool().get(ix);
            lo["is_candidate_source"].get_bool().get(cand);
            topo.layers[l] =
                LayerMeta{static_cast<int>(cr), kv, ix, cand};
            ++l;
        }
    }

    if (topo.format != kIndexFormat) {
        ModelTopology failed;
        failed.error_msg = "unsupported index format '" +
                           (topo.format.empty() ? std::string("<missing>")
                                                : topo.format) +
                           "' in " + ipath.string() + " (expected " +
                           std::string(kIndexFormat) + ")";
        return failed;
    }
    return topo;
}

bool ValidateGeometry(const ModelTopology &topo, std::string *error)
{
    auto fail = [error](const std::string &m) {
        if (error) *error = m;
        return false;
    };
    if (topo.num_layers != kLayers)
        return fail("layers " + std::to_string(topo.num_layers) + " != " +
                    std::to_string(kLayers));
    if (topo.experts_per_layer != kNumRouted)
        return fail("experts " + std::to_string(topo.experts_per_layer) +
                    " != " + std::to_string(kNumRouted));
    if (topo.total_experts != static_cast<int64_t>(kLayers) * kNumRouted)
        return fail("total_experts " + std::to_string(topo.total_experts) +
                    " != " +
                    std::to_string(static_cast<int64_t>(kLayers) * kNumRouted));
    return true;
}

} // namespace deepseek_v41
} // namespace lsxcommon
```

- [ ] **Step 4: Register the source + test target, run the tests**

Register the new translation unit in `libs/lsxcommon/CMakeLists.txt` — add
immediately after `src/deepseek_moe_model.cpp` (line 20) in the
`add_library(lsxcommon STATIC ...)` source list:

```cmake
    src/deepseek_moe_v41_load.cpp
```

Then modify `tests/CMakeLists.txt` — insert immediately after the Task 1
`test_deepseek_moe_v41_model` block:

```cmake
add_executable(test_deepseek_moe_v41_loader
               test_deepseek_moe_v41_loader.cpp)
target_link_libraries(test_deepseek_moe_v41_loader PRIVATE lsxcommon
                      Catch2::Catch2WithMain)
target_include_directories(test_deepseek_moe_v41_loader PRIVATE
    ${CMAKE_SOURCE_DIR}/libs/lsxcommon/src)
add_test(NAME test_deepseek_moe_v41_loader
         COMMAND test_deepseek_moe_v41_loader)
```

Then add `test_deepseek_moe_v41_loader` to the `fast` label list:

```cmake
    test_deepseek_moe_v41_loader
```

Run:

```bash
cmake --preset linux-debug
cmake --build --preset linux-debug --target \
  test_deepseek_moe_v41_model test_deepseek_moe_v41_loader
ctest --preset linux-debug -R "test_deepseek_moe_v41_(model|loader)" --output-on-failure
```

Expected: PASS (model: 2 cases; loader: 3 cases).

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/CMakeLists.txt \
        libs/lsxcommon/src/deepseek_moe_v41_load.h \
        libs/lsxcommon/src/deepseek_moe_v41_load.cpp \
        tests/test_deepseek_moe_v41_fixture.h \
        tests/test_deepseek_moe_v41_loader.cpp \
        tests/CMakeLists.txt
git commit -m "feat: DeepSeek-V4.1-Flash topology contract + loader-stage tests"
```

---

### Task 3: Naming-guard variant `v41`

**Interfaces:**
- Consumes: nothing.
- Produces: `test_code_naming_guard` accepts `deepseek_moe_v41_*` stems.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_code_naming_guard.cpp` (after the existing
`qwen_moe_exp` test case):

```cpp
TEST_CASE("naming guard accepts deepseek_moe_v41_* stems", "[naming][guard]")
{
    auto m = [](const std::string &s) {
        return std::regex_match(s, SchemeRegex());
    };
    CHECK(m("deepseek_moe_v41_constants"));
    CHECK(m("deepseek_moe_v41_load"));
    CHECK(m("deepseek_moe_v41_convert"));
    CHECK(m("deepseek_moe_v41_model"));
    CHECK(m("deepseek_moe_v41_inference"));
    CHECK(m("deepseek_moe_v41_inference_attn"));
    CHECK(m("deepseek_moe_v41_device_math"));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build --preset linux-debug --target test_code_naming_guard && ctest --preset linux-debug -R test_code_naming_guard --output-on-failure`
Expected: FAIL — `deepseek_moe_v41_constants` does not match `SchemeRegex()`.

- [ ] **Step 3: Write minimal implementation**

In `tests/test_code_naming_guard.cpp`, change the variant line (currently
line 29):

```cpp
    const std::string var    = "(?:27b|flash|36|v41)";
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build --preset linux-debug --target test_code_naming_guard && ctest --preset linux-debug -R test_code_naming_guard --output-on-failure`
Expected: PASS. Also re-run the whole host sweep to prove no regression:

```bash
ctest --preset linux-debug --output-on-failure
```

Expected: all `fast` tests pass, including `test_model_suite_guard`,
`test_ctest_labels_guard`, `test_model_independence_guard`.

- [ ] **Step 5: Commit**

```bash
git add tests/test_code_naming_guard.cpp
git commit -m "test: naming guard accepts deepseek_moe_v41_* variant"
```

---

## Plan 1 self-review

**Spec coverage:** frozen geometry table → Task 1; `logestix-v41-nvfp4` format
tag + topology + geometry gate → Task 2 (which also creates the synthetic index
fixture and registers the TU in `lsxcommon`); naming/variant guard → Task 3.
Deferred to later plans (correctly out of P1): converter (P2), loader (P3),
kernels/CED (P4–P5), model/registry (P6), selfcheck (P7).

**Placeholder scan:** no TBD/TODO; all steps carry code or exact commands.

**Type consistency:** `ModelTopology`, `LayerMeta`, `LoadTopology`,
`ValidateGeometry`, `IsSupportedFormat`, `kIndexFormat`, `BuildSyntheticIndex`,
`BuildSyntheticIndexRaw` are used consistently across tasks. `LayerMetaAt` is
declared in Task 1 and used in Task 2. `kLayers`/`kNumRouted` referenced by the
fixture default arguments exist in Task 1. The Task 1 CMake target
`test_deepseek_moe_v41_model` and the Task 2 target
`test_deepseek_moe_v41_loader` are both added to the `fast` label list exactly
once.

**Fixture note:** the fixture is created inside Task 2 (Step 3) rather than as a
separate task, because it is scaffolding for Task 2's test; this removes the
earlier cross-task ordering hazard.

---

## Status (2026-09-13, milestone 1 text backbone)

Plans 2–7 landed as `deepseek_moe_v41_*` (autonomous contour, no V4-Flash
includes). All stages are green on the synthetic path:

| Plan | Deliverable | Evidence |
|---|---|---|
| P2 | converter `ConvertDeepSeekV41Model` + `lsxforge --deepseek_v41` + tokenizer bridge | `test_deepseek_moe_v41_convert` (label `convert`) |
| P3 | pinned loader + FP8 dequant + expert store | `test_deepseek_moe_v41_loader`, `..._load_fast`, `..._dequant` |
| P4 | RMSNorm/RoPE/HC/Window sparse attention | `test_deepseek_moe_v41_kernels` |
| P5 | compressor / CED runtime / indexer / candidate blocks | `test_deepseek_moe_v41_kernels` |
| P6 | router, routed MoE, shared expert, head, `DeepSeekV41FlashModel`, registry, guards | `test_deepseek_moe_v41_pipeline` (synthetic full geometry, SWA + CED) |
| P7 | Paris selfcheck harness | `test_deepseek_moe_v41_selfcheck` (SKIP without vault) |

**Runtime validation done:** `test_deepseek_moe_v41_pipeline` builds a synthetic
FULL-GEOMETRY converted model (hidden 5120, one ratio-0 layer and one ratio-1
CED kv/index source, 2 experts) and runs the production
`RunDeepSeekV41Inference` graph end-to-end; tokens are byte-identical across
repeated runs. This is the execution-graph proof (no real numerics).

**Remaining operator step (not an AI session):** convert the real
`DeepSeek-V4.1-Flash` checkpoint via
`lsxforge --deepseek_v41 --input <model_dir> --output <model_dir>` (~350 GB
read, heavy GPU/disk job), then run
`ctest -R test_deepseek_moe_v41_selfcheck` with `LSX_TEST_DEEPSEEK_V41_ROOT`
pointing at the converted tree for the `"Paris"` / bit-exact selfcheck.
Until then the selfcheck SKIPs (`SKIP_RETURN_CODE 4`), as designed.

**Deferred by design:** engram (not converted), DSpark MTP (stored), vision
(cut), performance pass (correctness first).
