# LsxModelFactory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce `LsxModelFactory` + `IModel` so future models can be registered as pluggable classes, and migrate the current DeepSeek-V4-Flash inference into this structure.

**Architecture:** `IModel` is the product interface (`Load`/`Infer`/`Unload`/`Inspect`/`SelfCheck`/`Type`/`Tokenizer`). `LsxModelFactory` owns a static registry (`model_registry.cpp`) and resolves the concrete model by explicit `model_type` or autodetect from `logestix/model.index.json`. `DeepSeekV4FlashModel` wraps the existing CUDA pipeline; the pipeline entry points are renamed to internal `RunInferenceModel`/`RunSelfCheckModel` that accept pre-tokenized `input_ids` and an external `ArrowTokenizerDb`. Apps `lsxfabric` and `lsxcli` are migrated; `RunInference`/`RunSelfCheck` are removed from the public header.

**Tech Stack:** C++20, CUDA (NVCC 13.3), CMake, Arrow (mmap), simdjson, spdlog, Catch2.

**Spec:** `docs/superpowers/specs/2026-08-09-model-factory-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `libs/lsxcommon/include/lsxcommon/model.h` | Create | `IModel`, `ModelInitConfig`, `ModelRequest` |
| `libs/lsxcommon/include/lsxcommon/model_factory.h` | Create | `LsxModelFactory`, `CreateModelFn`, `RegisterModel`, `EnsureModelRegistry` |
| `libs/lsxcommon/src/deepseek_v4_flash_model.h` | Create | `DeepSeekV4FlashModel` class decl |
| `libs/lsxcommon/src/deepseek_v4_flash_model.cpp` | Create | `DeepSeekV4FlashModel` implementation |
| `libs/lsxcommon/src/model_factory.cpp` | Create | `Create`/`DetectType`/`Register` impl |
| `libs/lsxcommon/src/model_registry.cpp` | Create | static registration callback |
| `libs/lsxcommon/include/lsxcommon/inference_gpu.h` | Modify | remove `RunInference`/`RunSelfCheck` declarations |
| `libs/lsxcommon/src/inference_load.cu` | Modify | add `detect_model_type()` |
| `libs/lsxcommon/src/inference_load.cuh` | Modify | declare `detect_model_type()` |
| `libs/lsxcommon/src/inference_pipeline.cu` | Modify | rename + re-sign entry points; drop hardcoded prompt/tokenizer |
| `libs/lsxcommon/CMakeLists.txt` | Modify | add model_factory/registry/model cpp sources |
| `apps/lsxfabric/src/main.cpp` | Modify | migrate to factory |
| `apps/lsxcli/src/main.cpp` | Modify | migrate to factory |
| `tests/test_model_factory.cpp` | Create | factory tests |
| `tests/CMakeLists.txt` | Modify | add `test_model_factory` target |

---

### Task 1: Public API headers — `model.h` and `model_factory.h`

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/model.h`
- Create: `libs/lsxcommon/include/lsxcommon/model_factory.h`

- [ ] **Step 1: Create `model.h`**

```cpp
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "lsxcommon/inference_gpu.h"   // gpu::InferenceResult
#include "lsxcommon/safetensors.h"     // SafetensorsModel
#include "lsxcommon/tokenizer.h"       // ArrowTokenizerDb

namespace lsxcommon {

struct ModelInitConfig {
    std::string model_path;   // required
    std::string model_type;   // empty = autodetect from logestix/model.index.json
    int max_gen_tokens = 64;
};

struct ModelRequest {
    std::vector<int32_t> input_ids;   // pre-tokenized prompt
    int max_gen_tokens = 64;
};

class IModel {
public:
    virtual ~IModel() = default;

    virtual bool Load() = 0;
    virtual gpu::InferenceResult Infer(const ModelRequest &req) = 0;
    virtual void Unload() = 0;
    virtual SafetensorsModel Inspect() = 0;
    virtual gpu::InferenceResult SelfCheck(const ModelRequest &req) = 0;

    virtual const std::string &Type() const = 0;
    virtual ArrowTokenizerDb &Tokenizer() = 0;
};

} // namespace lsxcommon
```

- [ ] **Step 2: Create `model_factory.h`**

```cpp
#pragma once

#include <functional>
#include <memory>
#include <string>

#include "lsxcommon/model.h"

namespace lsxcommon {

using CreateModelFn = std::function<std::unique_ptr<IModel>(const ModelInitConfig &)>;

class LsxModelFactory {
public:
    static void Register(const std::string &type, CreateModelFn fn);
    static std::unique_ptr<IModel> Create(const ModelInitConfig &cfg);
    static std::string DetectType(const std::string &model_path);
};

// Guaranteed registry initialization before Create (needed for STATIC linking).
void EnsureModelRegistry();

// Registration helper used by model_registry.cpp.
bool RegisterModel(const char *type, CreateModelFn fn);

} // namespace lsxcommon
```

- [ ] **Step 3: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/model.h libs/lsxcommon/include/lsxcommon/model_factory.h
git commit -m "feat: add IModel interface and LsxModelFactory public headers"
```

---

### Task 2: `detect_model_type()` helper in inference_load

**Files:**
- Modify: `libs/lsxcommon/src/inference_load.cu` (after `load_topology`, ≈line 203)
- Modify: `libs/lsxcommon/src/inference_load.cuh` (near `load_topology` decl, line 63)

- [ ] **Step 1: Add declaration to `inference_load.cuh`**

Insert after the `load_topology` declaration (line 63):

```cpp
ModelTopology load_topology(const std::string &model_path);

// Read `model` field from <model_path>/logestix/model.index.json.
// Returns normalized type (lowercase, ' ' -> '-'), or "" on any error.
std::string detect_model_type(const std::string &model_path);
```

- [ ] **Step 2: Add implementation to `inference_load.cu`**

Insert after `load_topology` (after line 203):

```cpp
std::string detect_model_type(const std::string &model_path)
{
    std::string result;
    auto ipath = std::filesystem::path(model_path) / "logestix" / "model.index.json";
    std::ifstream f(ipath);
    if (!f) return result;
    std::string json((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());

    simdjson::ondemand::parser parser;
    simdjson::padded_string pad(json.data(), json.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(pad).get(doc) != simdjson::SUCCESS) return result;
    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS) return result;

    std::string_view name;
    if (root["model"].get_string().get(name) != simdjson::SUCCESS) return result;
    std::string raw(name);
    // Normalize: "DeepSeek-V4-Flash" -> "deepseek-v4-flash"
    for (char &c : raw) {
        if (c == ' ') c = '-';
        else if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    }
    return raw;
}
```

Note: this function is host code in a `.cu` file, exactly like the existing `load_topology`.

- [ ] **Step 3: Commit**

```bash
git add libs/lsxcommon/src/inference_load.cu libs/lsxcommon/src/inference_load.cuh
git commit -m "feat: add detect_model_type() helper for model autodetect"
```

---

### Task 3: Refactor pipeline entry points — `RunInferenceModel` / `RunSelfCheckModel`

**Files:**
- Modify: `libs/lsxcommon/src/inference_pipeline.cu`

The current `RunInference(const InferenceConfig&)` at line 253:
1. loads its own tokenizer (lines 258–275),
2. tokenizes a hardcoded prompt (lines 277–292),
3. uses `cfg.max_gen_tokens` in 3 places (lines 331, 401).
`RunSelfCheck` (line 1278) wraps `RunInference` with `max_gen_tokens=1`.

New signature: `RunInferenceModel` takes a pre-tokenized prompt + external `ArrowTokenizerDb`.

- [ ] **Step 1: Add new internal signature + implementation**

Replace the body-start of `RunInference` (lines 253–292) with:

```cpp
InferenceResult RunInferenceModel(
    const std::string &model_path,
    const std::vector<int32_t> &input_ids,
    int max_gen_tokens,
    ::lsxcommon::ArrowTokenizerDb &tok_db)
{
    InferenceResult result;
    auto mp = model_path;

    // 1. Log the incoming prompt tokens
    std::string tok_line;
    for (size_t i = 0; i < input_ids.size(); ++i)
        tok_line += std::to_string(input_ids[i]) + " ";
    log::info("[inference] prompt=" + std::to_string(input_ids.size())
        + " tokens: " + tok_line);
```

This deletes the old tokenizer-load block (lines 258–275) and the hardcoded
prompt encode (lines 277–287). `tok_db` and `input_ids` now come from the caller.

Keep the macro block, buffer allocation, and everything else in `RunInference`
untouched **except** the two `cfg.max_gen_tokens` uses:

- Line 331: `int max_gen = std::min(cfg.max_gen_tokens, 4096);` → `int max_gen = std::min(max_gen_tokens, 4096);`
- Line 401: `int max_compressed = (cfg.max_gen_tokens + static_cast<int>(input_ids.size())) / 4 + 1;` → `int max_compressed = (max_gen_tokens + static_cast<int>(input_ids.size())) / 4 + 1;`

The body continues unchanged through the end of the current `RunInference` (line 1273).

- [ ] **Step 2: Ensure the rest of the function still compiles**

- The variable `input_ids` is now the parameter (`std::vector<int32_t>`), not locally built — downstream code (lines 548, 608, etc.) iterates it unchanged.
- `tok_db` is now the reference parameter — downstream uses (line 1166 `tok_db.eos_id`, etc.) are unchanged.
- The line `int32_t n_ids = ::lsxcommon::arrow_tokenizer_encode(...)` no longer exists (removed in Step 1) and line 288–292 tok_line kept.

- [ ] **Step 3: Rewrite `RunSelfCheck` (lines 1278–1298)**

Final signature — takes the prompt ids explicitly (the old `RunSelfCheck`
tokenized a hardcoded prompt internally; now the caller owns that decision):

```cpp
InferenceResult RunSelfCheckModel(
    const std::string &model_path,
    const std::vector<int32_t> &input_ids,
    int max_gen_tokens,
    ::lsxcommon::ArrowTokenizerDb &tok_db)
{
    InferenceResult r = RunInferenceModel(model_path, input_ids, max_gen_tokens, tok_db);
    if (!r.ok) {
        r.selfcheck_passed = false;
        return r;
    }
    if (r.first_gen_tok_id >= 0) {
        std::string t = r.first_gen_text;
        auto b = t.begin(), e = t.end();
        while (b != e && std::isspace(static_cast<unsigned char>(*b))) ++b;
        while (e != b && std::isspace(static_cast<unsigned char>(*(e - 1)))) --e;
        r.selfcheck_passed = (std::string(b, e) == "Paris");
    }
    log::info("[selfcheck] top1_tok=" + std::to_string(r.first_gen_tok_id)
              + " text='" + r.first_gen_text + "' passed="
              + (r.selfcheck_passed ? "true" : "false"));
    return r;
}
```

Do NOT leave any `sc_prompt_ids` helper or old `RunSelfCheck` body in the file.

- [ ] **Step 4: Commit**

```bash
git add libs/lsxcommon/src/inference_pipeline.cu
git commit -m "refactor: rename inference entry points to internal RunInferenceModel/RunSelfCheckModel"
```

---

### Task 4: `DeepSeekV4FlashModel`

**Files:**
- Create: `libs/lsxcommon/src/deepseek_v4_flash_model.h`
- Create: `libs/lsxcommon/src/deepseek_v4_flash_model.cpp`

- [ ] **Step 1: Create `deepseek_v4_flash_model.h`**

```cpp
#pragma once

#include <memory>
#include <string>

#include "lsxcommon/model.h"

namespace lsxcommon {

class DeepSeekV4FlashModel final : public IModel {
public:
    explicit DeepSeekV4FlashModel(ModelInitConfig cfg);
    ~DeepSeekV4FlashModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    void Unload() override;
    SafetensorsModel Inspect() override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    const std::string &Type() const override { return kType; }
    ArrowTokenizerDb &Tokenizer() override;

    static constexpr const char *kType = "deepseek-v4-flash";

private:
    ModelInitConfig cfg_;
    std::unique_ptr<ArrowTokenizerDb> tok_db_;
    bool loaded_ = false;
};

} // namespace lsxcommon
```

- [ ] **Step 2: Create `deepseek_v4_flash_model.cpp`**

```cpp
#include "deepseek_v4_flash_model.h"

#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>
#include <filesystem>
#include <spdlog/spdlog.h>

#include "lsxcommon/safetensors.h"
#include "lsxcommon/tokenizer.h"

namespace lsxcommon {

namespace gpu {
// Internal engine entry points (inference_pipeline.cu).
InferenceResult RunInferenceModel(
    const std::string &model_path,
    const std::vector<int32_t> &input_ids,
    int max_gen_tokens,
    ::lsxcommon::ArrowTokenizerDb &tok_db);
InferenceResult RunSelfCheckModel(
    const std::string &model_path,
    const std::vector<int32_t> &input_ids,
    int max_gen_tokens,
    ::lsxcommon::ArrowTokenizerDb &tok_db);
} // namespace gpu

DeepSeekV4FlashModel::DeepSeekV4FlashModel(ModelInitConfig cfg)
    : cfg_(std::move(cfg)) {}

DeepSeekV4FlashModel::~DeepSeekV4FlashModel() { Unload(); }

bool DeepSeekV4FlashModel::Load()
{
    if (loaded_) return true;
    if (cfg_.model_path.empty()) {
        spdlog::error("[model] empty model_path");
        return false;
    }
    tok_db_ = std::make_unique<ArrowTokenizerDb>();
    auto pinned = std::filesystem::path(cfg_.model_path) / "logestix" / "pinned";
    auto v = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_vocab.arrow").string(), arrow::io::FileMode::READ);
    auto m = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_merges.arrow").string(), arrow::io::FileMode::READ);
    if (!v.ok() || !m.ok()) {
        spdlog::error("[model] cannot open tokenizer arrows in {}", pinned.string());
        return false;
    }
    auto st = load_tokenizer_metadata(*v, *m, *tok_db_);
    if (!st.ok()) {
        spdlog::error("[model] load_tokenizer_metadata: {}", st.ToString());
        return false;
    }
    loaded_ = true;
    return true;
}

gpu::InferenceResult DeepSeekV4FlashModel::Infer(const ModelRequest &req)
{
    gpu::InferenceResult r;
    if (!loaded_) {
        r.error_msg = "model not loaded";
        return r;
    }
    return gpu::RunInferenceModel(cfg_.model_path, req.input_ids,
                                  req.max_gen_tokens, *tok_db_);
}

void DeepSeekV4FlashModel::Unload()
{
    // RunInferenceModel loads pinned weights per-call and frees them on return.
    tok_db_.reset();
    loaded_ = false;
}

SafetensorsModel DeepSeekV4FlashModel::Inspect()
{
    return lsxcommon::scan_directory(cfg_.model_path);
}

gpu::InferenceResult DeepSeekV4FlashModel::SelfCheck(const ModelRequest &req)
{
    gpu::InferenceResult r;
    if (!loaded_) {
        r.error_msg = "model not loaded";
        return r;
    }
    return gpu::RunSelfCheckModel(cfg_.model_path, req.input_ids,
                                  req.max_gen_tokens, *tok_db_);
}

ArrowTokenizerDb &DeepSeekV4FlashModel::Tokenizer()
{
    return *tok_db_;
}

} // namespace lsxcommon
```

- [ ] **Step 3: Compile check (factory not wired yet — verify the two sources translate)**

Run (from repo root, after `cmake -S . -B build` once):

```powershell
git stash
# NOT required — lsxcommon_cu compiles inference_pipeline.cu already.
```

Actually verify by adding the model cpp file to CMake in Task 6 and building then.
Skip build here; proceed to Task 5.

- [ ] **Step 4: Commit later with CMake wiring (Task 6)**

---

### Task 5: `model_factory.cpp` + `model_registry.cpp`

**Files:**
- Create: `libs/lsxcommon/src/model_factory.cpp`

- [ ] **Step 1: Create `model_factory.cpp`**

```cpp
#include "lsxcommon/model_factory.h"

#include <unordered_map>

#include <spdlog/spdlog.h>

#include "lsxcommon/inference_gpu.h"
#include "lsxcommon/tokenizer.h"

namespace lsxcommon {

namespace {
std::unordered_map<std::string, CreateModelFn> &Registry()
{
    static std::unordered_map<std::string, CreateModelFn> reg;
    return reg;
}
} // namespace

void LsxModelFactory::Register(const std::string &type, CreateModelFn fn)
{
    Registry()[type] = std::move(fn);
}

bool RegisterModel(const char *type, CreateModelFn fn)
{
    LsxModelFactory::Register(type, std::move(fn));
    return true;
}

std::string LsxModelFactory::DetectType(const std::string &model_path)
{
    return detect_model_type(model_path);
}

std::unique_ptr<IModel> LsxModelFactory::Create(const ModelInitConfig &cfg)
{
    EnsureModelRegistry();

    std::string type = cfg.model_type;
    if (type.empty()) {
        type = DetectType(cfg.model_path);
        if (type.empty()) {
            spdlog::error("[factory] cannot autodetect model type from {}",
                          cfg.model_path);
            return nullptr;
        }
    }

    auto &reg = Registry();
    auto it = reg.find(type);
    if (it == reg.end()) {
        spdlog::error("[factory] no registered model for type '{}'", type);
        return nullptr;
    }
    return it->second(cfg);
}

} // namespace lsxcommon
```

Note: `detect_model_type` is declared in `inference_load.cuh` and implemented in
`inference_load.cu` (Task 2). Add that include:

```cpp
#include "inference_load.cuh"
```

(put it at top of `model_factory.cpp`)

- [ ] **Step 2: Create `model_registry.cpp`**

```cpp
#include "lsxcommon/model_factory.h"

#include "deepseek_v4_flash_model.h"

namespace lsxcommon {

namespace {
const bool kRegDeepSeekV4Flash = RegisterModel(
    DeepSeekV4FlashModel::kType,
    [](const ModelInitConfig &cfg) {
        return std::make_unique<DeepSeekV4FlashModel>(cfg);
    });
} // namespace

void EnsureModelRegistry()
{
    static bool init = [] {
        (void)kRegDeepSeekV4Flash;
        return true;
    }();
    (void)init;
}

} // namespace lsxcommon
```

- [ ] **Step 3: Commit**

```bash
git add libs/lsxcommon/src/model_factory.cpp libs/lsxcommon/src/model_registry.cpp
git commit -m "feat: add LsxModelFactory registry and static registration"
```

---

### Task 6: Public header cleanup + CMake wiring

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/inference_gpu.h`
- Modify: `libs/lsxcommon/CMakeLists.txt`

- [ ] **Step 1: Remove `RunInference`/`RunSelfCheck` declarations from `inference_gpu.h`**

Delete lines 32–34:

```cpp
InferenceResult RunInference(const InferenceConfig &cfg);

InferenceResult RunSelfCheck(const InferenceConfig &cfg);
```

- [ ] **Step 2: Add new sources to `lsxcommon` in CMakeLists.txt**

In `add_library(lsxcommon STATIC ...)` (lines 14–25) add:

```cmake
    src/model_factory.cpp
    src/model_registry.cpp
    src/deepseek_v4_flash_model.cpp
```

`model_factory.cpp` includes `inference_load.cuh` (a CUDA-compiled unit's header).
Since the `.cuh` only declares host functions and `inference_load.cu` is compiled by
NVCC into `lsxcommon_cu`, and `lsxcommon` links `lsxcommon_cu PUBLIC` (line 76),
the symbols resolve at link time. `model_factory.cpp` is plain C++ — the `.cuh`
includes `<cuda_runtime.h>` etc., which `lsxcommon` already exposes PUBLIC
(include dir, line 88). No extra config needed.

- [ ] **Step 3: Build**

```powershell
$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"
cmake --build build --config Debug 2>&1 | Select-String -Pattern "error|model_factory|model_registry|deepseek_v4_flash_model|inference_pipeline"
```

Fix any compile errors until clean.

- [ ] **Step 4: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/inference_gpu.h libs/lsxcommon/CMakeLists.txt
git add libs/lsxcommon/src/deepseek_v4_flash_model.h libs/lsxcommon/src/deepseek_v4_flash_model.cpp
git commit -m "feat: wire LsxModelFactory into lsxcommon; remove public RunInference API"
```

---

### Task 7: Tests — `test_model_factory.cpp`

**Files:**
- Create: `tests/test_model_factory.cpp`
- Modify: `tests/CMakeLists.txt`

Default model dir for tests: env `LSX_TEST_MODEL_PATH`, fallback
`D:\Programs\huggingface.co\DeepSeek-V4-Flash`.

- [ ] **Step 1: Write the test file**

```cpp
#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <string>

#include "lsxcommon/model.h"
#include "lsxcommon/model_factory.h"

namespace {

std::string TestModelPath()
{
    if (const char *p = std::getenv("LSX_TEST_MODEL_PATH"))
        return p;
    return R"(D:\Programs\huggingface.co\DeepSeek-V4-Flash)";
}

} // namespace

TEST_CASE("DetectType returns normalized type", "[factory]")
{
    std::string t = lsxcommon::LsxModelFactory::DetectType(TestModelPath());
    REQUIRE(t == "deepseek-v4-flash");
}

TEST_CASE("DetectType returns empty for invalid path", "[factory]")
{
    std::string t = lsxcommon::LsxModelFactory::DetectType(
        R"(D:\nonexistent\model\path)");
    REQUIRE(t.empty());
}

TEST_CASE("Create returns nullptr for unknown type", "[factory]")
{
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = TestModelPath();
    cfg.model_type = "no-such-model";
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m == nullptr);
}

TEST_CASE("Registry round-trip with mock model", "[factory]")
{
    bool created = false;
    lsxcommon::RegisterModel("mock-model",
        [&](const lsxcommon::ModelInitConfig &) {
            created = true;
            return std::unique_ptr<lsxcommon::IModel>(nullptr);
        });

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = TestModelPath();
    cfg.model_type = "mock-model";
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(created);
    REQUIRE(m == nullptr);  // mock returns nullptr by design
}

TEST_CASE("Create + Load + Inspect on real model dir", "[factory]")
{
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = TestModelPath();
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Type() == "deepseek-v4-flash");
    REQUIRE(m->Load());
    auto info = m->Inspect();
    REQUIRE(!info.shards.empty());
    m->Unload();
}
```

- [ ] **Step 2: Add CMake target in `tests/CMakeLists.txt`**

Insert after the `test_loader` block (after line 38):

```cmake
# ── Model factory test ────────────────────────────────────────────────
add_executable(test_model_factory test_model_factory.cpp)
target_link_libraries(test_model_factory PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_model_factory POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_model_factory>"
    )
    add_custom_command(TARGET test_model_factory POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_model_factory>"
    )
endif()

add_test(NAME test_model_factory COMMAND test_model_factory)
```

- [ ] **Step 3: Build and run the test**

```powershell
$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --config Debug
ctest --test-dir build --output-on-failure -C Debug -R test_model_factory
```

Expected: FAIL for real-model tests if path missing; otherwise PASS. If the model
dir isn't present on the CI machine, run with `LSX_TEST_MODEL_PATH` set.

- [ ] **Step 4: Commit**

```bash
git add tests/test_model_factory.cpp tests/CMakeLists.txt
git commit -m "test: add LsxModelFactory tests (detect, registry, create/load/inspect)"
```

---

### Task 8: Migrate `apps/lsxfabric`

**Files:**
- Modify: `apps/lsxfabric/src/main.cpp`

- [ ] **Step 1: Replace the `--testprompt` branch (lines 33–44)**

```cpp
    if (testprompt) {
        spdlog::info("Running prompt inference...");
        lsxcommon::ModelInitConfig cfg;
        cfg.model_path = model_path;
        cfg.max_gen_tokens = 64;
        auto model = lsxcommon::LsxModelFactory::Create(cfg);
        if (!model || !model->Load()) {
            spdlog::error("Failed to create/load model: {}", model_path);
            lsxcommon::Core::Shutdown();
            return 1;
        }
        auto &tok = model->Tokenizer();

        // Same prompt as the previous hardcoded pipeline prompt.
        const char *prompt = "The capital of France is";
        std::vector<int32_t> input_ids(512);
        size_t in_cap = input_ids.size();
        int32_t n_ids = lsxcommon::arrow_tokenizer_encode(
            tok, prompt, std::strlen(prompt), input_ids.data(), &in_cap);
        if (n_ids < 0) {
            spdlog::error("Tokenizer encode failed (code {})", n_ids);
            model->Unload();
            lsxcommon::Core::Shutdown();
            return 1;
        }
        input_ids.resize(static_cast<size_t>(n_ids));

        lsxcommon::ModelRequest req{std::move(input_ids), 64};
        auto result = model->Infer(req);
        if (!result.ok) {
            spdlog::error("Inference failed: {}", result.error_msg);
        }
        model->Unload();
        lsxcommon::Core::Shutdown();
        return result.ok ? 0 : 1;
    }
```

Add includes at top:

```cpp
#include "lsxcommon/model.h"
#include "lsxcommon/model_factory.h"
#include "lsxcommon/tokenizer.h"
```

- [ ] **Step 2: Replace the `--selfcheck` branch (lines 45–62)**

```cpp
    if (selfcheck) {
        spdlog::info("Running single-token self-check...");
        lsxcommon::ModelInitConfig cfg;
        cfg.model_path = model_path;
        auto model = lsxcommon::LsxModelFactory::Create(cfg);
        if (!model || !model->Load()) {
            spdlog::error("Failed to create/load model: {}", model_path);
            lsxcommon::Core::Shutdown();
            return 1;
        }
        auto &tok = model->Tokenizer();
        const char *prompt = "The capital of France is";
        std::vector<int32_t> input_ids(512);
        size_t in_cap = input_ids.size();
        int32_t n_ids = lsxcommon::arrow_tokenizer_encode(
            tok, prompt, std::strlen(prompt), input_ids.data(), &in_cap);
        if (n_ids < 0) {
            spdlog::error("Tokenizer encode failed (code {})", n_ids);
            model->Unload();
            lsxcommon::Core::Shutdown();
            return 1;
        }
        input_ids.resize(static_cast<size_t>(n_ids));

        lsxcommon::ModelRequest req{std::move(input_ids), 1};
        auto result = model->SelfCheck(req);
        if (result.ok) {
            spdlog::info("=== --selfcheck Results ===");
            spdlog::info("first token : {} '{}'",
                         result.first_gen_tok_id, result.first_gen_text);
            spdlog::info("maxv        : {:.4f}", result.first_gen_maxv);
            spdlog::info("passed      : {}",
                         result.selfcheck_passed ? "yes" : "no");
        } else {
            spdlog::error("SelfCheck failed: {}", result.error_msg);
        }
        model->Unload();
        lsxcommon::Core::Shutdown();
        return (result.ok && result.selfcheck_passed) ? 0 : 1;
    }
```

- [ ] **Step 3: Build**

```powershell
$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"
cmake --build build --config Debug 2>&1 | Select-String -Pattern "error"
```

- [ ] **Step 4: Commit**

```bash
git add apps/lsxfabric/src/main.cpp
git commit -m "refactor: route lsxfabric inference through LsxModelFactory"
```

---

### Task 9: Migrate `apps/lsxcli`

**Files:**
- Modify: `apps/lsxcli/src/main.cpp`

- [ ] **Step 1: Replace the `--info` branch (lines 27–34)**

```cpp
    if (info || infofull) {
        lsxcommon::ModelInitConfig cfg;
        cfg.model_path = model_path;
        auto model = lsxcommon::LsxModelFactory::Create(cfg);
        if (!model || !model->Load()) {
            spdlog::error("Failed to create/load model: {}", model_path);
            lsxcommon::Core::Shutdown();
            return 1;
        }
        auto mi = model->Inspect();
        if (infofull) {
            for (auto &shard : mi.shards)
                lsxcommon::print_shard(shard);
        }
        lsxcommon::print_model(mi);
        model->Unload();
    }
```

Add includes:

```cpp
#include "lsxcommon/model.h"
#include "lsxcommon/model_factory.h"
```

- [ ] **Step 2: Build**

```powershell
$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"
cmake --build build --config Debug 2>&1 | Select-String -Pattern "error"
```

- [ ] **Step 3: Commit**

```bash
git add apps/lsxcli/src/main.cpp
git commit -m "refactor: route lsxcli model inspection through LsxModelFactory"
```

---

### Task 10: Full verification

- [ ] **Step 1: Configure + build + full test suite**

```powershell
$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"
git submodule update --init --recursive
cmake -S . -B build -G "Visual Studio 18 2026" -DBUILD_TESTING=ON
cmake --build build --config Debug
ctest --test-dir build --output-on-failure -C Debug
```

All tests must pass (existing `test_core`, `test_loader`, `test_tokenizer`,
`test_inference_kernels`, `test_model_factory`, plus app tests).

- [ ] **Step 2: Smoke-run lsxcli --info**

```powershell
.\build\bin\Debug\lsxcli.exe --info
```

Expected: model info lines printed (no error).

- [ ] **Step 3: Verify RunInference/RunSelfCheck gone from public API**

```bash
rg -n "RunInference|RunSelfCheck" libs/lsxcommon/include apps/lsxfabric/src apps/lsxcli/src
```

Expected: only `ModelRequest`/docs mentions; no call sites. (Internal
`RunInferenceModel`/`RunSelfCheckModel` in `inference_pipeline.cu` allowed.)

- [ ] **Step 4: Commit any stragglers**

```bash
git status
git add -A
git commit -m "chore: finalize LsxModelFactory migration"
```

---

## Self-Review

**Spec coverage:**
- §3 IModel / ModelInitConfig / ModelRequest — Task 1 ✓
- §4 DeepSeekV4FlashModel — Tasks 4 + 6 ✓
- §5 engine refactor — Task 3 ✓
- §6 factory + registry + DetectType — Tasks 2 + 5 ✓
- §7 lsxfabric + lsxcli — Tasks 8 + 9 ✓
- §8 tests — Task 7 ✓
- §9 CMake — Tasks 6 + 7 ✓
- §10 error handling — `Load()` bool, `Infer()` result.ok, `Create()` nullptr ✓

**Placeholder scan:** No TBD/TODO left after Task 3 Step 4 cleanup. The
`sc_prompt_ids` placeholder is explicitly removed.

**Type consistency:** `ModelRequest.input_ids` everywhere; `RegisterModel(type, fn)`
matches model_registry.cpp usage. `DetectType` returns normalized string; test
asserts `"deepseek-v4-flash"`.

**Open item resolved:** test model dir via `LSX_TEST_MODEL_PATH` env with fallback.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-09-model-factory.md`.
Two execution options:

1. Subagent-Driven (recommended) — dispatch a fresh subagent per task, review between tasks, fast iteration
2. Inline Execution — execute tasks in this session, batch with checkpoints

Which approach?