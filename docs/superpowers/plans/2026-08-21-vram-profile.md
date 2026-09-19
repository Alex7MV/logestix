# VRAM Requirements Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every converted model gets three computed VRAM parameters (`min_bytes`, `optimal_bytes`, `max_bytes`) written into `model.index.json` during conversion and exposed at runtime via `IModel::VramRequirements()`.

**Architecture:** New host-only estimation module `lsxcommon::vram_estimate` computes a `VramProfile` from the scanned tensor inventory using each converter's own NVFP4-classification rules. All three index writers emit a mandatory `"vram"` JSON block; all three topology readers parse it strictly. `ModelBase` carries the profile; models populate it from their topology at `Load()`.

**Tech Stack:** C++20, CMake 4.3 (VS 2026 preset `windows-debug`), simdjson (ondemand), spdlog, Catch2 v3.7.0.

**Spec:** `docs/superpowers/specs/2026-08-21-vram-profile-design.md`

**Deviations from spec (deliberate):**
1. `absl::Span` → `const std::vector<VramTensorInfo>&` in `EstimateVram`: the header is reachable from `.cuh` TUs (via `model.h` / `deepseek_inference_load.cuh`), and AGENTS.md forbids absl container/span headers in NVCC-reachable headers.
2. Spec's single `classify_nvfp4` callback → `VramTensorPolicy` struct with two function pointers (`is_nvfp4`, `mem_class`): the min/max split requires knowing which tensors are *routed experts*, which is family-specific and distinct from NVFP4 classification (shared experts are NVFP4 but resident).

**Build/test commands (PowerShell, this machine):**

```powershell
$cmake = "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$ctest = "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe"
& $cmake --build build --config Debug --target <target>
& $ctest --preset windows-debug -R <test-name> --output-on-failure
```

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `libs/lsxcommon/include/lsxcommon/vram_estimate.h` | Create | Public types (`VramTensorInfo`, `VramTensorPolicy`, `VramGeometry`, `VramProfile`) + API. NVCC-safe (std only, no absl). |
| `libs/lsxcommon/src/vram_estimate.cpp` | Create | `EstimateVram`, `ReadContextLength`, `FormatVramJson`, `ParseVramBlock`, `FormatBytesHuman`, `kEmptyVramProfile`. |
| `libs/lsxcommon/CMakeLists.txt` | Modify | Add `src/vram_estimate.cpp` to `lsxcommon` sources. |
| `tests/test_vram_estimate.cpp` | Create | Unit tests for the module. |
| `tests/CMakeLists.txt` | Modify | Add `test_vram_estimate` executable + DLL copy + `add_test`. |
| `libs/lsxcommon/include/lsxcommon/converter.h` | Modify | `WriteModelIndex` gains `const VramProfile &vram` param. |
| `libs/lsxcommon/src/converter.cpp` | Modify | DeepSeek: policy fns, geometry derivation, profile computation, writer emits block. |
| `libs/lsxcommon/include/lsxcommon/glm_convert.h` | Modify | `Topology` gains `VramProfile vram`. |
| `libs/lsxcommon/src/glm_convert.cpp` | Modify | GLM: policy fns, computation, writer emits block, `load_topology` strict parse. |
| `libs/lsxcommon/include/lsxcommon/qwen_convert.h` | Modify | `QwenTopology` gains `VramProfile vram` + `std::string error_msg`. |
| `libs/lsxcommon/src/qwen_convert.cpp` | Modify | Qwen: computation, writer emits block, `load_qwen_topology` strict parse. |
| `libs/lsxcommon/src/qwen_inference_load.cpp` | Modify | Loader guard on `pw.top.error_msg`. |
| `libs/lsxcommon/src/deepseek_inference_load.cuh` | Modify | `ModelTopology` gains `VramProfile vram`. |
| `libs/lsxcommon/src/deepseek_inference_load.cu` | Modify | DeepSeek `load_topology` strict parse. |
| `libs/lsxcommon/include/lsxcommon/model.h` | Modify | `IModel::VramRequirements()` virtual + include. |
| `libs/lsxcommon/include/lsxcommon/model_base.h` | Modify | Override + protected `vram_`. |
| `libs/lsxcommon/include/lsxcommon/glm_inference_load.h` | Modify | `GlmPinnedWeights` gains `VramProfile vram`. |
| `libs/lsxcommon/src/glm_inference_load.cpp` | Modify | Populate `pw.vram` from topology. |
| `libs/lsxcommon/src/glm_v5_model.cpp` | Modify | Copy `pw_.vram` → `vram_` after load. |
| `libs/lsxcommon/src/qwen_base_model.cpp` | Modify | Copy `pw_.top.vram` → `vram_` after load. |
| `libs/lsxcommon/src/deepseek_v4_flash_model.cpp` | Modify | `Load()` reads topology, populates `vram_`. |
| `libs/lsxcommon/src/model_runtime.cpp` | Modify | `RunCliInspect` prints the profile. |
| `tests/test_glm_convert.cpp` | Modify | Fixture JSONs + new assertions + new test case. |
| `tests/test_qwen_convert.cpp` | Modify | New assertions on round-trip profile. |
| `tests/test_glm_loader.cpp` (if it synthesizes index.json) | Modify | Add `vram` block to fixtures. |

---

### Task 1: `vram_estimate` module — `EstimateVram`

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/vram_estimate.h`
- Create: `libs/lsxcommon/src/vram_estimate.cpp`
- Modify: `libs/lsxcommon/CMakeLists.txt:44-47` (source list)
- Create: `tests/test_vram_estimate.cpp`
- Modify: `tests/CMakeLists.txt` (append new test target)

- [ ] **Step 1: Write the failing test**

Create `tests/test_vram_estimate.cpp`:

```cpp
#include <catch2/catch_test_macros.hpp>
#include <lsxcommon/vram_estimate.h>

#include <string>
#include <vector>

using lsxcommon::VramGeometry;
using lsxcommon::VramProfile;
using lsxcommon::VramTensorClass;
using lsxcommon::VramTensorInfo;
using lsxcommon::VramTensorPolicy;

namespace {

// Capture-free lambdas convert implicitly to the plain function pointers
// stored in VramTensorPolicy.
constexpr auto kAllNvfp4 = +[](const std::string &) { return true; };
constexpr auto kNoneNvfp4 = +[](const std::string &) { return false; };
constexpr auto kAllRouted = +[](const std::string &) {
    return VramTensorClass::RoutedExpert;
};
constexpr auto kAllResident = +[](const std::string &) {
    return VramTensorClass::Resident;
};

VramGeometry BaseGeom()
{
    VramGeometry g;
    g.hidden_dim = 4096;
    g.kv_dim = 512;
    g.num_attn_layers = 1;
    g.num_moe_layers = 1;
    g.experts_per_layer = 1;
    g.context_length = 2048;
    return g;
}

} // namespace

TEST_CASE("NVFP4 packing math: numel/2 packed + numel/16 scales", "[vram]")
{
    // 16384 elements: 8192 bytes packed E2M1 + 1024 bytes E4M3 scales.
    std::vector<VramTensorInfo> tensors = {
        {"layers.0.ffn.experts.0.w1.weight", 16384, 32768},
    };
    auto p = lsxcommon::EstimateVram(tensors, {&kAllNvfp4, &kAllRouted}, BaseGeom());
    CHECK(p.min_bytes == 0);          // routed expert -> not resident
    CHECK(p.optimal_bytes == p.min_bytes + 37748736ull); // kv+ws @ ctx=2048, 1 attn layer
    CHECK(p.max_bytes == 9216ull + 37748736ull);         // 16384/2 + 16384/16
}

TEST_CASE("Non-quantized tensors keep raw dtype size", "[vram]")
{
    std::vector<VramTensorInfo> tensors = {
        {"model.embed_tokens.weight", 1000, 2000}, // BF16, resident
    };
    auto p = lsxcommon::EstimateVram(tensors, {&kNoneNvfp4, &kAllResident}, BaseGeom());
    CHECK(p.min_bytes == 2000);
    CHECK(p.max_bytes == p.optimal_bytes); // no routed experts
}

TEST_CASE("Hierarchy invariant min <= optimal <= max", "[vram]")
{
    std::vector<VramTensorInfo> tensors = {
        {"model.embed_tokens.weight", 1000, 2000},
        {"layers.0.ffn.experts.0.w1.weight", 16384, 32768},
    };
    auto p = lsxcommon::EstimateVram(tensors, {&kAllNvfp4, &kAllRouted}, BaseGeom());
    CHECK(p.min_bytes <= p.optimal_bytes);
    CHECK(p.optimal_bytes <= p.max_bytes);
}

TEST_CASE("KV cache and workspace scale with context length", "[vram]")
{
    std::vector<VramTensorInfo> tensors = {
        {"model.embed_tokens.weight", 1000, 2000},
    };
    auto g = BaseGeom();
    g.context_length = 2048;
    auto p1 = lsxcommon::EstimateVram(tensors, {&kNoneNvfp4, &kAllResident}, g);
    g.context_length = 4096;
    auto p2 = lsxcommon::EstimateVram(tensors, {&kNoneNvfp4, &kAllResident}, g);
    // kv = ctx*attn*kv_dim*2*2; ws = ctx*hidden*attn*2*2 -> both double.
    CHECK(p2.optimal_bytes - p2.min_bytes ==
          2 * (p1.optimal_bytes - p1.min_bytes));
    CHECK(p2.context_tokens == 4096);
}

TEST_CASE("Empty inventory yields zero profile", "[vram]")
{
    auto p = lsxcommon::EstimateVram({}, {&kAllNvfp4, &kAllRouted}, BaseGeom());
    CHECK(p.min_bytes == 0);
    CHECK(p.optimal_bytes == 0);
    CHECK(p.max_bytes == 0);
}

TEST_CASE("Zero context_length falls back to 2048", "[vram]")
{
    auto g = BaseGeom();
    g.context_length = 0;
    auto p = lsxcommon::EstimateVram({}, {&kNoneNvfp4, &kAllResident}, g);
    CHECK(p.context_tokens == 2048);
}

TEST_CASE("kEmptyVramProfile is all zeros", "[vram]")
{
    CHECK(lsxcommon::kEmptyVramProfile.min_bytes == 0);
    CHECK(lsxcommon::kEmptyVramProfile.context_tokens == 0);
}
```

- [ ] **Step 2: Add the test target to `tests/CMakeLists.txt`**

Append at the end of `tests/CMakeLists.txt` (follow the existing pattern, e.g. `test_path_registry`):

```cmake
# ── VRAM-profile estimator test (conversion-time) ───────────────────
add_executable(test_vram_estimate test_vram_estimate.cpp)
target_link_libraries(test_vram_estimate PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_vram_estimate POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_vram_estimate>"
    )
    add_custom_command(TARGET test_vram_estimate POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_vram_estimate>"
    )
endif()

add_test(NAME test_vram_estimate COMMAND test_vram_estimate)
```

- [ ] **Step 3: Run test to verify it fails to build**

```powershell
& $cmake --build build --config Debug --target test_vram_estimate
```

Expected: CMake configure error — `lsxcommon/vram_estimate.h` not found (target not yet compilable).

- [ ] **Step 4: Create the header**

Create `libs/lsxcommon/include/lsxcommon/vram_estimate.h`:

```cpp
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace lsxcommon {

// One scanned safetensors tensor (pre-conversion inventory entry).
struct VramTensorInfo {
    std::string name;           // safetensors tensor name
    std::int64_t numel = 0;     // element count
    std::int64_t raw_bytes = 0; // original size: dtype_size * numel
};

// Memory residency class of a tensor in the converted model.
enum class VramTensorClass { Resident, RoutedExpert };

// Family-specific classification rules (supplied by each converter).
// Function pointers (not std::function) to keep the header NVCC-safe.
struct VramTensorPolicy {
    bool (*is_nvfp4)(const std::string &name) = nullptr;
    VramTensorClass (*mem_class)(const std::string &name) = nullptr;
};

// Model geometry needed by the estimate (derived from config.json + shapes).
struct VramGeometry {
    int hidden_dim = 0;
    int kv_dim = 0;             // KV latent dim (attention value width)
    int num_attn_layers = 0;
    int num_moe_layers = 0;
    int experts_per_layer = 0;
    std::int64_t context_length = 0; // from source config.json
};

// Three VRAM parameters computed at conversion time.
struct VramProfile {
    std::uint64_t min_bytes = 0;     // device-resident weights only
    std::uint64_t optimal_bytes = 0; // min + KV cache + workspace
    std::uint64_t max_bytes = 0;     // optimal + full routed-expert pool
    std::int64_t context_tokens = 0; // seq_len used in the estimate
};

extern const VramProfile kEmptyVramProfile;

// Computes the conversion-time VRAM profile. NVFP4 tensor size:
// numel/2 (packed E2M1 nibbles) + numel/16 (E4M3 per-block scales);
// all other tensors keep raw_bytes (norms/biases precision invariant).
VramProfile EstimateVram(const std::vector<VramTensorInfo> &tensors,
                         const VramTensorPolicy &policy,
                         const VramGeometry &geom);

// Reads the model context length from config.json. Fallback chain:
// max_position_embeddings -> context_length -> seq_length -> 2048
// (root first, then nested "text_config"). Missing fields: warning + 2048.
std::int64_t ReadContextLength(const std::string &config_json_path);

// Renders the mandatory "vram" JSON block (no trailing comma, 2-space base
// indent) for insertion into model.index.json by any index writer.
std::string FormatVramJson(const VramProfile &p);

// Strictly parses the "vram" object out of a model.index.json document.
// Returns false and fills err when the block is missing/malformed/inconsistent.
bool ParseVramBlock(const std::string &index_json_text,
                    VramProfile &out, std::string &err);

// Human-readable byte size ("123 GiB" / "45 MiB" / ...).
std::string FormatBytesHuman(std::uint64_t bytes);

} // namespace lsxcommon
```

- [ ] **Step 5: Create the implementation**

Create `libs/lsxcommon/src/vram_estimate.cpp` (ReadContextLength/Format/Parse bodies are stubs for now — Tasks 2/3 fill them; include them so the header links):

```cpp
#include "lsxcommon/vram_estimate.h"

#include <spdlog/spdlog.h>
#include <simdjson.h>

#include <fstream>
#include <iterator>

namespace lsxcommon {

const VramProfile kEmptyVramProfile{};

namespace {

constexpr std::int64_t kDefaultContextTokens = 2048;

// NVFP4 converted size: packed E2M1 nibbles + E4M3 scales (block of 16).
std::int64_t Nvfp4ConvertedBytes(std::int64_t numel)
{
    return numel / 2 + numel / 16;
}

} // namespace

VramProfile EstimateVram(const std::vector<VramTensorInfo> &tensors,
                         const VramTensorPolicy &policy,
                         const VramGeometry &geom)
{
    VramProfile p;
    p.context_tokens = geom.context_length > 0
        ? geom.context_length : kDefaultContextTokens;

    std::uint64_t resident = 0;
    std::uint64_t expert_pool = 0;

    for (const auto &t : tensors) {
        const bool quantized =
            policy.is_nvfp4 && policy.is_nvfp4(t.name);
        const std::uint64_t bytes = quantized
            ? static_cast<std::uint64_t>(Nvfp4ConvertedBytes(t.numel))
            : static_cast<std::uint64_t>(t.raw_bytes);
        const bool routed = policy.mem_class &&
            policy.mem_class(t.name) == VramTensorClass::RoutedExpert;
        if (routed)
            expert_pool += bytes;
        else
            resident += bytes;
    }

    const std::int64_t ctx = p.context_tokens;
    const std::int64_t attn =
        static_cast<std::int64_t>(geom.num_attn_layers > 0 ? geom.num_attn_layers : 0);
    const std::int64_t kv_dim =
        geom.kv_dim > 0 ? geom.kv_dim : 512;
    const std::int64_t hidden =
        geom.hidden_dim > 0 ? geom.hidden_dim : 4096;

    // Same formulas as print_model() (converter.cpp) but with the
    // config-declared context length instead of hardcoded 2048.
    const std::int64_t kv_cache = ctx * attn * kv_dim * 2 * 2;      // K+V, FP16
    const std::int64_t workspace = ctx * hidden * attn * 2 * 2;     // FP16

    p.min_bytes = resident;
    p.optimal_bytes =
        resident + static_cast<std::uint64_t>(kv_cache + workspace);
    p.max_bytes = p.optimal_bytes + expert_pool;
    return p;
}

std::int64_t ReadContextLength(const std::string &config_json_path)
{
    // Implemented in Task 2.
    (void)config_json_path;
    return kDefaultContextTokens;
}

std::string FormatVramJson(const VramProfile &p)
{
    // Implemented in Task 3.
    (void)p;
    return {};
}

bool ParseVramBlock(const std::string &index_json_text,
                    VramProfile &out, std::string &err)
{
    // Implemented in Task 3.
    (void)index_json_text;
    out = {};
    err = "not implemented";
    return false;
}

std::string FormatBytesHuman(std::uint64_t bytes)
{
    if (bytes >= 1024ull * 1024 * 1024)
        return std::to_string(bytes / (1024ull * 1024 * 1024)) + " GiB";
    if (bytes >= 1024ull * 1024)
        return std::to_string(bytes / (1024ull * 1024)) + " MiB";
    if (bytes >= 1024ull)
        return std::to_string(bytes / 1024ull) + " KiB";
    return std::to_string(bytes) + " B";
}

} // namespace lsxcommon
```

- [ ] **Step 6: Register the source in `libs/lsxcommon/CMakeLists.txt`**

In the `add_library(lsxcommon STATIC ...)` list (lines 16–47), insert `src/vram_estimate.cpp` after `src/unicode.cpp` (keeps alphabetical order):

```cmake
    src/unicode.cpp
    src/vram_estimate.cpp
)
```

- [ ] **Step 7: Run tests to verify they pass**

```powershell
& $cmake --build build --config Debug --target test_vram_estimate && & $ctest --preset windows-debug -R test_vram_estimate --output-on-failure
```

Expected: `test_vram_estimate` — all 7 test cases PASS.

- [ ] **Step 8: Commit**

```powershell
git add libs/lsxcommon/include/lsxcommon/vram_estimate.h libs/lsxcommon/src/vram_estimate.cpp libs/lsxcommon/CMakeLists.txt tests/test_vram_estimate.cpp tests/CMakeLists.txt
git commit -m "feat: add vram_estimate module with conversion-time VRAM profile"
```

---

### Task 2: `ReadContextLength`

**Files:**
- Modify: `libs/lsxcommon/src/vram_estimate.cpp`
- Modify: `tests/test_vram_estimate.cpp`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_vram_estimate.cpp`:

```cpp
#include <filesystem>
#include <fstream>

namespace {

fs::path WriteTempConfig(const std::string &json)
{
    static int counter = 0;   // portable unique-enough temp name per process
    auto p = fs::temp_directory_path() /
             ("logestix_vram_cfg_" + std::to_string(counter++) + ".json");
    std::ofstream f(p, std::ios::trunc);
    f << json;
    return p;
}

} // namespace

TEST_CASE("ReadContextLength: max_position_embeddings at root", "[vram]")
{
    auto p = WriteTempConfig(R"({"hidden_size": 4096,
        "max_position_embeddings": 163840})");
    CHECK(lsxcommon::ReadContextLength(p.string()) == 163840);
    fs::remove(p);
}

TEST_CASE("ReadContextLength: nested under text_config (Qwen)", "[vram]")
{
    auto p = WriteTempConfig(R"({"text_config": {
        "max_position_embeddings": 262144}})");
    CHECK(lsxcommon::ReadContextLength(p.string()) == 262144);
    fs::remove(p);
}

TEST_CASE("ReadContextLength: context_length / seq_length fallbacks", "[vram]")
{
    auto p1 = WriteTempConfig(R"({"context_length": 4096})");
    CHECK(lsxcommon::ReadContextLength(p1.string()) == 4096);
    fs::remove(p1);

    auto p2 = WriteTempConfig(R"({"seq_length": 8192})");
    CHECK(lsxcommon::ReadContextLength(p2.string()) == 8192);
    fs::remove(p2);
}

TEST_CASE("ReadContextLength: missing field / missing file -> 2048", "[vram]")
{
    auto p = WriteTempConfig(R"({"hidden_size": 4096})");
    CHECK(lsxcommon::ReadContextLength(p.string()) == 2048);
    fs::remove(p);

    CHECK(lsxcommon::ReadContextLength("Z:/definitely/not/here.json") == 2048);
}
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
& $cmake --build build --config Debug --target test_vram_estimate && & $ctest --preset windows-debug -R test_vram_estimate --output-on-failure
```

Expected: FAIL — `ReadContextLength` always returns 2048.

- [ ] **Step 3: Implement**

Replace the stub body in `libs/lsxcommon/src/vram_estimate.cpp`:

```cpp
std::int64_t ReadContextLength(const std::string &config_json_path)
{
    namespace sj = simdjson;
    std::ifstream f(config_json_path);
    if (!f) {
        spdlog::warn("[vram] cannot open {}: defaulting context to {}",
                     config_json_path, kDefaultContextTokens);
        return kDefaultContextTokens;
    }
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());

    sj::ondemand::parser parser;
    sj::padded_string pad(json.data(), json.size());
    sj::ondemand::document doc;
    if (parser.iterate(pad).get(doc) != sj::SUCCESS) {
        spdlog::warn("[vram] {} parse failed: defaulting context to {}",
                     config_json_path, kDefaultContextTokens);
        return kDefaultContextTokens;
    }
    sj::ondemand::object root;
    if (doc.get_object().get(root) != sj::SUCCESS)
        return kDefaultContextTokens;

    auto find_ctx = [](sj::ondemand::object &o) -> std::int64_t {
        for (const char *k : {"max_position_embeddings", "context_length",
                              "seq_length"}) {
            int64_t v;
            if (o[k].get_int64().get(v) == sj::SUCCESS && v > 0) return v;
        }
        return 0;
    };

    if (int64_t v = find_ctx(root); v > 0) return v;
    // Qwen3.5 nests the language-model config under "text_config".
    if (auto tc = root["text_config"].get_object();
        tc.error() == sj::SUCCESS) {
        sj::ondemand::object src = tc.value();
        if (int64_t v = find_ctx(src); v > 0) return v;
    }
    spdlog::warn("[vram] no context field in {}: defaulting to {}",
                 config_json_path, kDefaultContextTokens);
    return kDefaultContextTokens;
}
```

Also add `#include <string>` is already transitively present via the header; no new includes needed (`simdjson.h`, `<fstream>`, `<iterator>` already included in Task 1).

- [ ] **Step 4: Run tests to verify they pass**

```powershell
& $cmake --build build --config Debug --target test_vram_estimate && & $ctest --preset windows-debug -R test_vram_estimate --output-on-failure
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```powershell
git add libs/lsxcommon/src/vram_estimate.cpp tests/test_vram_estimate.cpp
git commit -m "feat: add ReadContextLength config parser to vram_estimate"
```

---

### Task 3: `FormatVramJson` + `ParseVramBlock`

**Files:**
- Modify: `libs/lsxcommon/src/vram_estimate.cpp`
- Modify: `tests/test_vram_estimate.cpp`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_vram_estimate.cpp`:

```cpp
TEST_CASE("FormatVramJson / ParseVramBlock round-trip", "[vram]")
{
    VramProfile in;
    in.min_bytes = 8734629888ull;
    in.optimal_bytes = 9126804480ull;
    in.max_bytes = 412316860416ull;
    in.context_tokens = 163840;

    std::string json = "{" + lsxcommon::FormatVramJson(in) + "}";

    VramProfile out;
    std::string err;
    REQUIRE(lsxcommon::ParseVramBlock(json, out, err));
    CHECK(err.empty());
    CHECK(out.min_bytes == in.min_bytes);
    CHECK(out.optimal_bytes == in.optimal_bytes);
    CHECK(out.max_bytes == in.max_bytes);
    CHECK(out.context_tokens == in.context_tokens);
}

TEST_CASE("ParseVramBlock rejects missing vram block", "[vram]")
{
    VramProfile out;
    std::string err;
    REQUIRE_FALSE(lsxcommon::ParseVramBlock(
        R"({"format": "logestix-glm", "num_layers": 79})", out, err));
    CHECK(err.find("vram") != std::string::npos);
}

TEST_CASE("ParseVramBlock rejects malformed vram block", "[vram]")
{
    VramProfile out;
    std::string err;
    REQUIRE_FALSE(lsxcommon::ParseVramBlock(
        R"({"vram": {"min_bytes": 1}})", out, err));
    CHECK(err.find("malformed") != std::string::npos);
}

TEST_CASE("ParseVramBlock rejects inconsistent values", "[vram]")
{
    VramProfile out;
    std::string err;
    // max < optimal
    REQUIRE_FALSE(lsxcommon::ParseVramBlock(
        R"({"vram": {"min_bytes": 10, "optimal_bytes": 30,
            "max_bytes": 20, "context_tokens": 2048}})", out, err));
    CHECK(err.find("inconsistent") != std::string::npos);

    // context_tokens <= 0
    REQUIRE_FALSE(lsxcommon::ParseVramBlock(
        R"({"vram": {"min_bytes": 10, "optimal_bytes": 20,
            "max_bytes": 30, "context_tokens": 0}})", out, err));
}

TEST_CASE("ParseVramBlock rejects invalid JSON document", "[vram]")
{
    VramProfile out;
    std::string err;
    REQUIRE_FALSE(lsxcommon::ParseVramBlock("not json {{{", out, err));
    CHECK_FALSE(err.empty());
}
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
& $cmake --build build --config Debug --target test_vram_estimate && & $ctest --preset windows-debug -R test_vram_estimate --output-on-failure
```

Expected: new cases FAIL (`not implemented` / empty format output).

- [ ] **Step 3: Implement**

Replace the two stubs in `libs/lsxcommon/src/vram_estimate.cpp`:

```cpp
std::string FormatVramJson(const VramProfile &p)
{
    std::string s = "  \"vram\": {\n";
    s += "    \"min_bytes\": " + std::to_string(p.min_bytes) + ",\n";
    s += "    \"optimal_bytes\": " + std::to_string(p.optimal_bytes) + ",\n";
    s += "    \"max_bytes\": " + std::to_string(p.max_bytes) + ",\n";
    s += "    \"context_tokens\": " + std::to_string(p.context_tokens) + "\n";
    s += "  }";
    return s;
}

bool ParseVramBlock(const std::string &index_json_text,
                    VramProfile &out, std::string &err)
{
    namespace sj = simdjson;
    out = {};
    sj::ondemand::parser parser;
    sj::padded_string pad(index_json_text.data(), index_json_text.size());
    sj::ondemand::document doc;
    if (parser.iterate(pad).get(doc) != sj::SUCCESS) {
        err = "JSON parse error";
        return false;
    }
    sj::ondemand::object root;
    if (doc.get_object().get(root) != sj::SUCCESS) {
        err = "expected JSON object";
        return false;
    }
    sj::ondemand::object vram;
    if (root["vram"].get_object().get(vram) != sj::SUCCESS) {
        err = "missing 'vram' block";
        return false;
    }
    int64_t mn = 0, op = 0, mx = 0, ct = 0;
    if (vram["min_bytes"].get_int64().get(mn) != sj::SUCCESS ||
        vram["optimal_bytes"].get_int64().get(op) != sj::SUCCESS ||
        vram["max_bytes"].get_int64().get(mx) != sj::SUCCESS ||
        vram["context_tokens"].get_int64().get(ct) != sj::SUCCESS) {
        err = "malformed 'vram' block";
        return false;
    }
    if (mn < 0 || op < mn || mx < op || ct <= 0) {
        err = "inconsistent 'vram' values";
        return false;
    }
    out.min_bytes = static_cast<std::uint64_t>(mn);
    out.optimal_bytes = static_cast<std::uint64_t>(op);
    out.max_bytes = static_cast<std::uint64_t>(mx);
    out.context_tokens = ct;
    return true;
}
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
& $cmake --build build --config Debug --target test_vram_estimate && & $ctest --preset windows-debug -R test_vram_estimate --output-on-failure
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```powershell
git add libs/lsxcommon/src/vram_estimate.cpp tests/test_vram_estimate.cpp
git commit -m "feat: add vram index.json format/parse helpers"
```

---

### Task 4: DeepSeek converter — compute + write profile

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/converter.h` (WriteModelIndex declaration)
- Modify: `libs/lsxcommon/src/converter.cpp` (anonymous namespace ~line 25-104, WriteModelIndex ~line 962, call site ~line 1325)

- [ ] **Step 1: Extend `WriteModelIndex` signature**

In `libs/lsxcommon/include/lsxcommon/converter.h`, add `#include "lsxcommon/vram_estimate.h"` and change:

```cpp
arrow::Status WriteModelIndex(
    const std::string &model_name,
    int num_layers,
    int experts_per_layer,
    int64_t total_pinned,
    int64_t total_experts,
    const VramProfile &vram,
    const std::filesystem::path &output_root);
```

- [ ] **Step 2: Emit the block in the writer**

In `libs/lsxcommon/src/converter.cpp` `WriteModelIndex` (~line 962), change the JSON emission — replace:

```cpp
    f << "  \"total_experts\": " << total_experts << ",\n";
    f << "  \"files\": {\n";
```

with:

```cpp
    f << "  \"total_experts\": " << total_experts << ",\n";
    f << FormatVramJson(vram) << ",\n";
    f << "  \"files\": {\n";
```

- [ ] **Step 3: Add policy functions**

In the anonymous namespace of `converter.cpp` (after `parse_shared_expert_name`, ~line 102):

```cpp
// ── VRAM profile classification (spec 2026-08-21) ───────────────────────────

bool DsIsNvfp4Tensor(const std::string &name)
{
    return parse_expert_name(name).has_value()
        || parse_shared_expert_name(name).has_value();
}

VramTensorClass DsMemClass(const std::string &name)
{
    return parse_expert_name(name).has_value()
        ? VramTensorClass::RoutedExpert
        : VramTensorClass::Resident;
}
```

- [ ] **Step 4: Compute the profile at the call site**

In `Converter::Run()` (~line 1318), insert between `SaveSharedExpertsToArrow(...)` and `WriteModelIndex(...)`:

```cpp
    // Conversion-time VRAM profile (spec 2026-08-21-vram-profile-design.md).
    VramGeometry vg;
    vg.num_attn_layers = max_layer;
    vg.num_moe_layers = max_layer;
    vg.experts_per_layer = max_experts_per_layer;
    vg.context_length = ReadContextLength(
        (fs::path(impl_->input_path) / "config.json").string());
    for (const auto &shard : model.shards) {
        for (const auto &t : shard.tensors) {
            if (vg.kv_dim == 0 && t.shape.size() >= 2 &&
                t.name.find("wkv.weight") != std::string::npos) {
                vg.kv_dim = static_cast<int>(t.shape[0]);
                vg.hidden_dim = static_cast<int>(t.shape[1]);
            }
        }
    }

    std::vector<VramTensorInfo> vram_inventory;
    for (const auto &shard : model.shards)
        for (const auto &t : shard.tensors)
            vram_inventory.push_back({t.name, t.numel(), t.byte_size()});

    VramProfile vram_profile = EstimateVram(
        vram_inventory, {&DsIsNvfp4Tensor, &DsMemClass}, vg);
    spdlog::info("VRAM profile: min={} optimal={} max={} (ctx={})",
        FormatBytesHuman(vram_profile.min_bytes),
        FormatBytesHuman(vram_profile.optimal_bytes),
        FormatBytesHuman(vram_profile.max_bytes),
        vram_profile.context_tokens);
```

and change the call:

```cpp
    ARROW_RETURN_NOT_OK(WriteModelIndex(
        fs::path(impl_->input_path).filename().string(),
        max_layer, max_experts_per_layer,
        static_cast<int64_t>(pinned_tensors.size()),
        static_cast<int64_t>(experts_processed),
        vram_profile,
        impl_->output_path));
```

Note: `TensorInfo::numel()` exists (used by `print_nvfp4_forecast`, converter.cpp:308). `converter.cpp` already includes `<vector>` transitively; add `#include "lsxcommon/vram_estimate.h"` to its include block (line 1-21).

- [ ] **Step 5: Build**

```powershell
& $cmake --build build --config Debug --target lsxcommon
```

Expected: BUILD SUCCEEDED.

- [ ] **Step 6: Commit**

```powershell
git add libs/lsxcommon/include/lsxcommon/converter.h libs/lsxcommon/src/converter.cpp
git commit -m "feat: DeepSeek converter computes and writes vram profile to model.index.json"
```

---

### Task 5: GLM family — converter, writer, strict reader, tests

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/glm_convert.h` (Topology)
- Modify: `libs/lsxcommon/src/glm_convert.cpp` (policy fns, writer, converter, load_topology)
- Modify: `tests/test_glm_convert.cpp`

- [ ] **Step 1: Write the failing tests**

In `tests/test_glm_convert.cpp`:

1. In `TEST_CASE("GLM topology parses extended index (79 layers)")` — add `"vram"` to the fixture JSON (replace the last fixture lines):

```
       "total_pinned_tensors": 1024,
       "total_experts": 19456,
       "vram": {
         "min_bytes": 1000000,
         "optimal_bytes": 2000000,
         "max_bytes": 3000000,
         "context_tokens": 2048
       }
     })");
```

and after `CHECK(topo.total_experts == 19456);` add:

```cpp
    CHECK(topo.vram.min_bytes == 1000000);
    CHECK(topo.vram.optimal_bytes == 2000000);
    CHECK(topo.vram.max_bytes == 3000000);
    CHECK(topo.vram.context_tokens == 2048);
```

2. New test case after `"GLM topology rejects malformed index"`:

```cpp
TEST_CASE("GLM topology rejects index without vram block", "[glm][topology]")
{
    PutIndex(R"({
      "format": "logestix-glm",
      "model": "GLM-5.2",
      "num_layers": 79,
      "experts_per_layer": 256,
      "total_pinned_tensors": 1024,
      "total_experts": 19456
    })");
    auto topo = lsxcommon::glm::load_topology(TempIndexPath().string());
    CHECK(!topo.error_msg.empty());
    CHECK(topo.error_msg.find("vram") != std::string::npos);
}
```

3. In `TEST_CASE("GLM converter writes logestix layout from synthetic shards")`, after the topology assertions (`CHECK(topo.indexer_layers...)`), add:

```cpp
    // VRAM profile: mandatory, internally consistent.
    CHECK(topo.vram.min_bytes > 0);
    CHECK(topo.vram.optimal_bytes >= topo.vram.min_bytes);
    CHECK(topo.vram.max_bytes > topo.vram.optimal_bytes); // routed experts exist
    CHECK(topo.vram.context_tokens > 0);
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
& $cmake --build build --config Debug --target test_glm_convert && & $ctest --preset windows-debug -R test_glm_convert --output-on-failure
```

Expected: `Topology` has no member `vram` — compile error.

- [ ] **Step 3: Extend `Topology`**

In `libs/lsxcommon/include/lsxcommon/glm_convert.h`: add `#include "lsxcommon/vram_estimate.h"` after the existing includes, and add to `struct Topology` (after `indexer_layers`, line ~49):

```cpp
    VramProfile vram;   // conversion-time VRAM requirements (mandatory block)
```

- [ ] **Step 4: Emit the block in `WriteGlmModelIndex`**

In `libs/lsxcommon/src/glm_convert.cpp` (~line 399-401), replace:

```cpp
    f << "  \"total_pinned_tensors\": " << total_pinned << ",\n";
    f << "  \"total_experts\": " << total_experts << "\n";
    f << "}\n";
```

with:

```cpp
    f << "  \"total_pinned_tensors\": " << total_pinned << ",\n";
    f << "  \"total_experts\": " << total_experts << ",\n";
    f << FormatVramJson(topo.vram) << "\n";
    f << "}\n";
```

- [ ] **Step 5: Add policy functions + compute in `ConvertGlmModel`**

In the anonymous namespace starting at line 288 (after `GlmExpertKeyHash`):

```cpp
// ── VRAM profile classification (spec 2026-08-21) ───────────────────────────

bool GlmIsNvfp4Tensor(const std::string &name)
{
    const auto p = classify(name);
    return p.kind == TensorKind::RoutedExpert ||
           p.kind == TensorKind::SharedExpert;
}

VramTensorClass GlmMemClass(const std::string &name)
{
    return classify(name).kind == TensorKind::RoutedExpert
        ? VramTensorClass::RoutedExpert
        : VramTensorClass::Resident;
}
```

In `ConvertGlmModel`, insert between the `Topology topo; ...` fill (line ~661) and the `WriteGlmModelIndex(...)` call (line ~663):

```cpp
    // Conversion-time VRAM profile (spec 2026-08-21-vram-profile-design.md).
    {
        VramGeometry vg;
        vg.hidden_dim = cfg_hidden_size;
        vg.num_attn_layers = static_cast<int>(topo.num_layers);
        vg.num_moe_layers = static_cast<int>(topo.moe_layers.size());
        vg.experts_per_layer = static_cast<int>(topo.experts_per_layer);
        vg.context_length = ReadContextLength(cfg_path.string());
        for (const auto &shard : model.shards) {
            for (const auto &t : shard.tensors) {
                if (vg.kv_dim == 0 && t.shape.size() >= 2 &&
                    t.name.find("kv_a_proj_with_mqa") != std::string::npos) {
                    vg.kv_dim = static_cast<int>(t.shape[0]);
                    if (vg.hidden_dim <= 0)
                        vg.hidden_dim = static_cast<int>(t.shape[1]);
                }
            }
        }
        if (vg.hidden_dim <= 0) vg.hidden_dim = 4096;

        std::vector<VramTensorInfo> vram_inventory;
        for (const auto &shard : model.shards)
            for (const auto &t : shard.tensors)
                vram_inventory.push_back({t.name, t.numel(), t.byte_size()});
        topo.vram = EstimateVram(vram_inventory,
                                 {&GlmIsNvfp4Tensor, &GlmMemClass}, vg);
        spdlog::info("VRAM profile: min={} optimal={} max={} (ctx={})",
            FormatBytesHuman(topo.vram.min_bytes),
            FormatBytesHuman(topo.vram.optimal_bytes),
            FormatBytesHuman(topo.vram.max_bytes),
            topo.vram.context_tokens);
    }
```

(`cfg_path` and `cfg_hidden_size` are already in scope — lines 426-431.)

- [ ] **Step 6: Strict parse in `load_topology`**

In `load_topology` (line ~265-270), replace:

```cpp
    topo.indexer_layers = read_int_array(root, "indexer_layers", ok);
    if (!ok) {
        topo.error_msg = "invalid indexer_layers in " + index_json_path;
        return topo;
    }
    return topo;
```

with:

```cpp
    topo.indexer_layers = read_int_array(root, "indexer_layers", ok);
    if (!ok) {
        topo.error_msg = "invalid indexer_layers in " + index_json_path;
        return topo;
    }
    std::string verr;
    if (!ParseVramBlock(json, topo.vram, verr)) {
        Topology failed;   // plain reset: no partial layers + error
        failed.error_msg = verr + " in " + index_json_path;
        return failed;
    }
    return topo;
```

- [ ] **Step 7: Run tests to verify they pass**

```powershell
& $cmake --build build --config Debug --target test_glm_convert && & $ctest --preset windows-debug -R test_glm_convert --output-on-failure
```

Expected: all `test_glm_convert` cases PASS (pipeline test now exercises the real writer → strict reader round-trip).

- [ ] **Step 8: Update any other GLM fixtures**

```powershell
rg '"total_experts"' tests apps
```

Every synthetic `model.index.json` fixture (e.g. in `tests/test_glm_loader.cpp` if present) must gain a valid block or its reader will reject it. Canonical block to insert before the closing `}` (adjust the preceding line to end with a comma):

```
  "vram": {
    "min_bytes": 1000000,
    "optimal_bytes": 2000000,
    "max_bytes": 3000000,
    "context_tokens": 2048
  }
```

Re-run the touched tests.

- [ ] **Step 9: Commit**

```powershell
git add libs/lsxcommon/include/lsxcommon/glm_convert.h libs/lsxcommon/src/glm_convert.cpp tests/test_glm_convert.cpp
git commit -m "feat: GLM converter writes mandatory vram profile; strict topology parsing"
```

---

### Task 6: Qwen family — converter, writer, strict reader, loader guard, tests

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/qwen_convert.h`
- Modify: `libs/lsxcommon/src/qwen_convert.cpp`
- Modify: `libs/lsxcommon/src/qwen_inference_load.cpp:330-334`
- Modify: `tests/test_qwen_convert.cpp`

- [ ] **Step 1: Extend `QwenTopology`**

In `libs/lsxcommon/include/lsxcommon/qwen_convert.h`: add `#include "lsxcommon/vram_estimate.h"` and append to `struct QwenTopology` (after `total_pinned_tensors`, line ~67):

```cpp
    VramProfile vram;        // conversion-time VRAM requirements (mandatory)
    std::string error_msg;   // non-empty when index.json is invalid
```

- [ ] **Step 2: Write the failing test**

In `tests/test_qwen_convert.cpp`, find the test that reads back `logestix/model.index.json` via `load_qwen_topology` (grep `load_qwen_topology` — around line 78) and add after the existing topology assertions:

```cpp
    CHECK(top.vram.min_bytes > 0);
    CHECK(top.vram.optimal_bytes >= top.vram.min_bytes);
    CHECK(top.vram.max_bytes >= top.vram.optimal_bytes);
    CHECK(top.vram.context_tokens > 0);
    CHECK(top.error_msg.empty());
```

- [ ] **Step 3: Run test to verify it fails**

```powershell
& $cmake --build build --config Debug --target test_qwen_convert && & $ctest --preset windows-debug -R test_qwen_convert --output-on-failure
```

Expected: FAIL — `min_bytes == 0` (converter does not write the block yet).

- [ ] **Step 4: Compute + write in the converter**

In `libs/lsxcommon/src/qwen_convert.cpp` `ConvertQwenModel`, insert between `top.total_pinned_tensors = static_cast<int>(total_pinned);` (line ~587) and `WriteQwenModelIndex(output_root, top)` (line ~588):

```cpp
    // Conversion-time VRAM profile (spec 2026-08-21-vram-profile-design.md).
    {
        VramGeometry vg;
        vg.hidden_dim = top.hidden_size;
        vg.kv_dim = top.head_dim > 0
            ? top.head_dim * top.num_key_value_heads : 0;
        int full_attn_layers = 0;
        for (const auto &lt : top.layer_types)
            if (lt == "full_attention") ++full_attn_layers;
        vg.num_attn_layers = full_attn_layers;
        vg.context_length = ReadContextLength(cfg_path.string());

        std::vector<VramTensorInfo> vram_inventory;
        for (const auto &shard : model.shards)
            for (const auto &t : shard.tensors)
                vram_inventory.push_back({t.name, t.numel(), t.byte_size()});

        // Qwen is dense: no routed experts, mem_class stays nullptr.
        top.vram = EstimateVram(vram_inventory,
            {+[](const std::string &n) {
                 return qwen_tensor_is_nvfp4(classify_qwen_tensor(n).kind);
             },
             nullptr},
            vg);
        spdlog::info("VRAM profile: min={} optimal={} max={} (ctx={})",
            lsxcommon::FormatBytesHuman(top.vram.min_bytes),
            lsxcommon::FormatBytesHuman(top.vram.optimal_bytes),
            lsxcommon::FormatBytesHuman(top.vram.max_bytes),
            top.vram.context_tokens);
    }
```

(`cfg_path` is in scope at line ~456. `qwen_convert.h` needs `#include "lsxcommon/vram_estimate.h"` — done in Step 1.)

In `WriteQwenModelIndex` (line ~322), replace the tail of the JSON builder:

```cpp
      << ",\n  \"total_pinned_tensors\": " << top.total_pinned_tensors
      << ",\n" << FormatVramJson(top.vram)
      << "\n}";
```

- [ ] **Step 5: Strict parse in `load_qwen_topology`**

In `libs/lsxcommon/src/qwen_convert.cpp` `load_qwen_topology` (after the `layer_types` array read, line ~279), add before `return top;`:

```cpp
    std::string verr;
    if (!ParseVramBlock(json, top.vram, verr)) {
        QwenTopology failed;   // plain reset: defaults + error_msg
        failed.error_msg = verr;
        return failed;
    }
    return top;
```

- [ ] **Step 6: Loader guard**

In `libs/lsxcommon/src/qwen_inference_load.cpp` (lines 330-334), replace:

```cpp
    if (fs::exists(idx))
        pw.top = load_qwen_topology(idx.string());
```

with:

```cpp
    if (fs::exists(idx)) {
        pw.top = load_qwen_topology(idx.string());
        if (!pw.top.error_msg.empty()) {
            log::error("[qwen_load] topology error in " + idx.string()
                       + ": " + pw.top.error_msg);
            return false;
        }
    }
```

- [ ] **Step 7: Run tests to verify they pass**

```powershell
& $cmake --build build --config Debug --target test_qwen_convert test_qwen_loader && & $ctest --preset windows-debug -R "test_qwen" --output-on-failure
```

Expected: `test_qwen_convert` PASS. If `test_qwen_loader` synthesizes an index.json, add the canonical `vram` block (Task 5 Step 8 snippet) to its fixtures and re-run.

- [ ] **Step 8: Commit**

```powershell
git add libs/lsxcommon/include/lsxcommon/qwen_convert.h libs/lsxcommon/src/qwen_convert.cpp libs/lsxcommon/src/qwen_inference_load.cpp tests/test_qwen_convert.cpp
git commit -m "feat: Qwen converter writes mandatory vram profile; strict topology parsing"
```

---

### Task 7: DeepSeek reader — strict parse

**Files:**
- Modify: `libs/lsxcommon/src/deepseek_inference_load.cuh:59-64` (ModelTopology)
- Modify: `libs/lsxcommon/src/deepseek_inference_load.cu:199-236` (load_topology)
- Modify: `tests/test_deepseek_loader.cpp` (if it builds manifest fixtures)

- [ ] **Step 1: Extend `ModelTopology`**

In `libs/lsxcommon/src/deepseek_inference_load.cuh`: add `#include "lsxcommon/vram_estimate.h"` near the top (the header is std-only and NVCC-safe), and extend the struct:

```cpp
struct ModelTopology {
    int num_layers = 0;
    int experts_per_layer = 0;
    int64_t total_experts = 0;
    lsxcommon::VramProfile vram;   // conversion-time VRAM requirements
    std::string error_msg;
};
```

- [ ] **Step 2: Strict parse in `load_topology`**

In `libs/lsxcommon/src/deepseek_inference_load.cu` `load_topology` (line ~228-235), insert the vram parse BEFORE the num_layers assignment so failure yields `num_layers == 0` (the pipeline's existing guard at `deepseek_inference_pipeline.cu:159` then fails cleanly). Replace:

```cpp
    int64_t nl = 0, epl = 0, te = 0;
    root["num_layers"].get_int64().get(nl);
    root["experts_per_layer"].get_int64().get(epl);
    root["total_experts"].get_int64().get(te);
    topo.num_layers = static_cast<int>(nl);
    topo.experts_per_layer = static_cast<int>(epl);
    topo.total_experts = te;
    return topo;
```

with:

```cpp
    std::string verr;
    if (!lsxcommon::ParseVramBlock(json, topo.vram, verr)) {
        ModelTopology failed;   // num_layers == 0 -> pipeline guard fires
        failed.error_msg = verr + " in " + ipath.string();
        return failed;
    }
    int64_t nl = 0, epl = 0, te = 0;
    root["num_layers"].get_int64().get(nl);
    root["experts_per_layer"].get_int64().get(epl);
    root["total_experts"].get_int64().get(te);
    topo.num_layers = static_cast<int>(nl);
    topo.experts_per_layer = static_cast<int>(epl);
    topo.total_experts = te;
    return topo;
```

- [ ] **Step 3: Update DeepSeek manifest fixtures**

```powershell
rg '"total_experts"' tests
```

For every synthetic DeepSeek manifest (e.g. `tests/test_deepseek_loader.cpp`, `tests/test_expert_store.cpp` if they write `model.index.json`), insert the canonical block from Task 5 Step 8 before the closing `}` (adjust the preceding line to end with a comma). Rebuild and run:

```powershell
& $cmake --build build --config Debug --target test_deepseek_loader test_expert_store && & $ctest --preset windows-debug -R "deepseek|expert_store" --output-on-failure
```

Expected: PASS. If a test constructs `ModelTopology` directly (no JSON), no change is needed.

- [ ] **Step 4: Commit**

```powershell
git add libs/lsxcommon/src/deepseek_inference_load.cuh libs/lsxcommon/src/deepseek_inference_load.cu tests/
git commit -m "feat: DeepSeek topology reader strictly parses vram profile"
```

---

### Task 8: Runtime surface — `IModel::VramRequirements()`

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/model.h:62-64`
- Modify: `libs/lsxcommon/include/lsxcommon/model_base.h`
- Modify: `libs/lsxcommon/include/lsxcommon/glm_inference_load.h`
- Modify: `libs/lsxcommon/src/glm_inference_load.cpp:415-420`
- Modify: `libs/lsxcommon/src/glm_v5_model.cpp:72,180`
- Modify: `libs/lsxcommon/src/qwen_base_model.cpp:28-33`
- Modify: `libs/lsxcommon/src/deepseek_v4_flash_model.cpp:29-34`

- [ ] **Step 1: Interface + base**

In `libs/lsxcommon/include/lsxcommon/model.h`: add `#include "lsxcommon/vram_estimate.h"` to the include block, and after `ResidentWeightBytes()` (line 64) add:

```cpp
    // VRAM requirements computed at conversion time ("vram" block of
    // model.index.json). Default: empty profile keeps test fakes additive.
    virtual const VramProfile &VramRequirements() const
    {
        return kEmptyVramProfile;
    }
```

In `libs/lsxcommon/include/lsxcommon/model_base.h`: add the override next to `ResidentWeightBytes()` (line 37):

```cpp
    const VramProfile &VramRequirements() const override { return vram_; }
```

and in the `protected:` section (next to `resident_bytes_`, line 48):

```cpp
    // Conversion-time VRAM requirements; populated by subclass Load() from
    // the model's model.index.json "vram" block.
    VramProfile vram_;
```

- [ ] **Step 2: GLM population**

In `libs/lsxcommon/include/lsxcommon/glm_inference_load.h`: add `#include "lsxcommon/vram_estimate.h"` and add to `struct GlmPinnedWeights` next to `resident_bytes` (~line 103-106):

```cpp
    lsxcommon::VramProfile vram;   // from model.index.json "vram" block
```

In `libs/lsxcommon/src/glm_inference_load.cpp`, right after the topology check (line ~419, after the `if (!topo.error_msg.empty())` block):

```cpp
    pw.vram = topo.vram;
```

In `libs/lsxcommon/src/glm_v5_model.cpp`, at BOTH `LoadGlmPinnedWeights` call sites (lines 72 and 180), after the `e == cudaSuccess` check passes (in the success branch, before returning true):

```cpp
    vram_ = pw_.vram;
```

- [ ] **Step 3: Qwen population**

In `libs/lsxcommon/src/qwen_base_model.cpp` (line ~28-33), after `LoadQwenPinnedWeights` succeeds (inside the `if (ok)` success path, before `FinishLoad()`/`return true`):

```cpp
    vram_ = pw_.top.vram;
```

- [ ] **Step 4: DeepSeek population**

Replace `DeepSeekV4FlashModel::Load` (line 29-34):

```cpp
bool DeepSeekV4FlashModel::Load()
{
    if (!BeginLoad(cfg_.model_path)) return false;
    auto topo = gpu::load_topology(cfg_.model_path);
    if (!topo.error_msg.empty()) {
        spdlog::error("[deepseek-model] topology error: {}", topo.error_msg);
        return false;
    }
    vram_ = topo.vram;
    FinishLoad();
    return true;
}
```

- [ ] **Step 5: Build + run model tests**

```powershell
& $cmake --build build --config Debug --target test_model_runtime test_model_factory test_glm_loader && & $ctest --preset windows-debug -R "test_model_runtime|test_model_factory|test_glm_loader" --output-on-failure
```

Expected: PASS (FakeModel uses the additive default; real-model factory tests SKIP without models).

- [ ] **Step 6: Commit**

```powershell
git add libs/lsxcommon/include/lsxcommon/model.h libs/lsxcommon/include/lsxcommon/model_base.h libs/lsxcommon/include/lsxcommon/glm_inference_load.h libs/lsxcommon/src/glm_inference_load.cpp libs/lsxcommon/src/glm_v5_model.cpp libs/lsxcommon/src/qwen_base_model.cpp libs/lsxcommon/src/deepseek_v4_flash_model.cpp
git commit -m "feat: expose conversion-time VRAM profile via IModel::VramRequirements"
```

---

### Task 9: CLI inspect display

**Files:**
- Modify: `libs/lsxcommon/src/model_runtime.cpp:168-179`

- [ ] **Step 1: Print the profile in `RunCliInspect`**

In `libs/lsxcommon/src/model_runtime.cpp` `RunCliInspect` (line 168-179), after `lsxcommon::print_model(mi);` add:

```cpp
    const auto &vp = m.model->VramRequirements();
    if (vp.min_bytes > 0) {
        spdlog::info("=== VRAM Profile (conversion-time, ctx={}) ===",
                     vp.context_tokens);
        spdlog::info("min     (weights only)         : {}",
                     FormatBytesHuman(vp.min_bytes));
        spdlog::info("optimal (+KV cache+workspace)  : {}",
                     FormatBytesHuman(vp.optimal_bytes));
        spdlog::info("max     (full expert placement): {}",
                     FormatBytesHuman(vp.max_bytes));
    }
```

(`model_runtime.cpp` includes `model_runtime.h` → `model.h` → `vram_estimate.h`; `CreateCliModel` calls `Load()`, so the profile is populated.)

- [ ] **Step 2: Build + smoke**

```powershell
& $cmake --build build --config Debug --target lsxcli
```

Expected: BUILD SUCCEEDED. (Runtime display verified with a real converted model; not unit-tested.)

- [ ] **Step 3: Commit**

```powershell
git add libs/lsxcommon/src/model_runtime.cpp
git commit -m "feat: lsxcli --info/--infofull prints conversion-time VRAM profile"
```

---

### Task 10: Full verification

- [ ] **Step 1: Build everything**

```powershell
& $cmake --build build --config Debug
```

Expected: BUILD SUCCEEDED for all targets.

- [ ] **Step 2: Run the full test suite**

```powershell
& $ctest --preset windows-debug --output-on-failure
```

Expected: all tests PASS (real-model tests SKIP without models, per existing convention).

- [ ] **Step 3: Fix any fallout, then commit**

```powershell
git add -A
git commit -m "test: full suite green with mandatory vram profile blocks"
```

(If Step 2 is already green with no modified files, skip the commit.)

- [ ] **Step 4: Reconversion note**

Existing converted models now have invalid `model.index.json` (missing `vram`). Reconvert via lsxforge pipelines (`--glm`, DeepSeek/Qwen equivalents) before running any real-model selfcheck/benchmark. This is a data operation, not a code task — flag it to the operator.

---

## Self-Review Notes

- **Spec coverage:** estimation module (Tasks 1-3), three converters write mandatory block (Tasks 4-6), three readers strict (Tasks 5-7), runtime IModel surface (Task 8), CLI consumer (Task 9), reconversion contract (Task 10 Step 4). LSXHome display deferred — spec lists it as consumer but acceptance criteria do not require it; follow-up once the profile exists at runtime.
- **Type consistency:** `VramProfile`/`VramTensorPolicy`/`VramTensorClass` names identical across all tasks; `FormatVramJson(topo.vram)` in GLM writer vs `vram` param in DeepSeek writer matches each writer's local state; `ParseVramBlock(json, ...)` signature consistent in all three readers.
- **NVCC safety:** `vram_estimate.h` is std-only (no absl) and is included from `model.h` and `deepseek_inference_load.cuh`; `vram_estimate.cpp` (simdjson) is compiled by MSVC only into the `lsxcommon` STATIC lib, never into `.cu` object libraries.
