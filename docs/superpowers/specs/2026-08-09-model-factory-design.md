# LsxModelFactory — Design Spec

## 1. Overview

New models will be added to the project. Right now the whole engine is a single
function `gpu::RunInference(cfg)` in `libs/lsxcommon/src/inference_pipeline.cu`, hard-wired
for DeepSeek-V4-Flash (geometry constants in `inference_load.cuh`:
`kHiddenDim=4096`, `kNumLayers=43`, `kNumRouted=256`, etc.).

This sprint creates a factory pattern for models:

- `IModel` — an abstract "product" interface (Load / Infer / Unload / Inspect / SelfCheck).
- `LsxModelFactory` — a factory with a static registry of implementations.
- `DeepSeekV4FlashModel` — the current single model, moved onto the interface
  as a wrapper over the existing engine (without parameterizing the constants — they stay
  in `inference_load.cuh`).
- All apps (`lsxfabric`, `lsxcli`) and tests switch to working through the factory;
  `RunInference` / `RunSelfCheck` are removed from the public API.

## 2. Constraints

- **No engine parameterization.** The geometry constants (`kHiddenDim`, `kNumLayers`,
  MLA/HC/CSA) stay in `inference_load.cuh` and shared with the engine. A new model
  of a different architecture is not added in this sprint; only the extension
  point is designed.
- **Full API replacement.** `gpu::RunInference` / `gpu::RunSelfCheck` are removed from
  the public API. Everything switches to `LsxModelFactory`.
- **Static registry.** New models are added by an entry in the central file
  `model_registry.cpp` (2-3 models — a static list is manageable).
- **Hybrid type selection:** an explicit `ModelInitConfig.model_type` or autodetect from
  the `model` field of the `logestix/model.index.json` file.
- **All model work goes through the factory.** Scanning safetensors (`--info`)
  is also available only via `IModel::Inspect()`.

## 3. Public API

New header `libs/lsxcommon/include/lsxcommon/model.h`:

```cpp
#pragma once
#include <memory>
#include <string>
#include <vector>
#include <cstdint>
#include "lsxcommon/inference_gpu.h"   // gpu::InferenceResult
#include "lsxcommon/tokenizer.h"       // ArrowTokenizerDb
#include "lsxcommon/safetensors.h"     // SafetensorsModel

namespace lsxcommon {

struct ModelInitConfig {
    std::string model_path;   // required
    std::string model_type;   // empty = autodetect from model.index.json
    int max_gen_tokens = 64;
};

struct ModelRequest {
    std::vector<int32_t> input_ids;   // the prompt, already tokenized
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

Notes:

- Tokenization is done **outside** the model: `ModelRequest.input_ids` arrives ready.
  `IModel::Tokenizer()` hands out the tokenizer so the caller (app/test) can
  tokenize the prompt with the same tokenizer the model uses.
- `Inspect()` returns `SafetensorsModel` — the existing type from `safetensors.h`
  (what `scan_directory` returns).
- `SelfCheck()` is mandatory (a model must be able to self-diagnose); for models without
  such support the implementation returns `gpu::InferenceResult{ok=false, error_msg}`.

## 4. DeepSeekV4FlashModel

New file `libs/lsxcommon/src/deepseek_v4_flash_model.cpp` + private header
`libs/lsxcommon/src/deepseek_v4_flash_model.h`:

```cpp
namespace lsxcommon {

class DeepSeekV4FlashModel final : public IModel {
public:
    explicit DeepSeekV4FlashModel(ModelInitConfig cfg);
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

Implementation:

- `Load()` — prepares `ArrowTokenizerDb` and the materials shared by Infer/SelfCheck
  (in the current `RunInference` the tokenizer is created on every call; `Load` does
  it once).
- `Infer(req)` — calls the internal engine:
  `gpu::InferenceResult RunInferenceModel(model_path, req.input_ids, max_gen_tokens, *tok_db_)`.
- `SelfCheck(req)` — likewise via `gpu::RunSelfCheckModel(...)`.
- `Inspect()` — calls `lsxcommon::scan_directory(cfg_.model_path)`. This is
  the only place the safetensors scanning utility is called.
- `Unload()` — releases resources (`cudaFree` pinned weights, etc.), resets `loaded_`.

## 5. Engine refactor (inference_pipeline.cu)

`RunInference(cfg)` / `RunSelfCheck(cfg)` are rewritten into internal functions with a
new signature (they stay in the same file, namespace `lsxcommon::gpu`):

```cpp
// The input prompt arrives tokenized; the tokenizer — from IModel.
gpu::InferenceResult RunInferenceModel(
    const std::string &model_path,
    const std::vector<int32_t> &input_ids,
    int max_gen_tokens,
    ::lsxcommon::ArrowTokenizerDb &tok_db);

gpu::InferenceResult RunSelfCheckModel(
    const std::string &model_path,
    ::lsxcommon::ArrowTokenizerDb &tok_db);
```

Changes inside the body of `RunInference`:

- Remove the hardcoded prompt and its tokenization (lines ~277):
  `input_ids` is now a parameter.
- Accept `ArrowTokenizerDb&` instead of loading it again.
- Everything else (buffer allocation, the layer loop, generation, dump handles,
  the `LSX_*` env flags) is not touched.

Changes inside the body of `RunSelfCheck`:

- Likewise: no re-loading of the tokenizer, it receives `tok_db` from outside.

`InferenceConfig` / `InferenceResult` stay in `inference_gpu.h` as internal
engine types (used by `DeepSeekV4FlashModel::Infer`).

## 6. LsxModelFactory + static registry

New header `libs/lsxcommon/include/lsxcommon/model_factory.h`:

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

// File model_registry.cpp: guaranteed registry initialization before Create.
void EnsureModelRegistry();

// Registration helper.
bool RegisterModel(const char *type, CreateModelFn fn);

} // namespace lsxcommon
```

Implementation of `libs/lsxcommon/src/model_factory.cpp`:

- The registry is a `std::unordered_map<std::string, CreateModelFn>` (a singleton via a
  static-local function).
- `DetectType(path)` — reads `logestix/model.index.json`, returns the `model` field
  (e.g. `"DeepSeek-V4-Flash"`), or an empty string on error. The parsing
  reuses the existing simdjson code; a separate helper function
  `detect_model_type(path)` is extracted in `inference_load.cu`, called from `DetectType`.
- `Create(cfg)`:
  1. `type = cfg.model_type`; if empty → `cfg.model_type = DetectType(cfg.model_path)`.
  2. `EnsureModelRegistry()` — guarantees the entries are registered (a callback
     at the front of `Create`; we do not rely on static initialization order under
     STATIC linking).
  3. Look up the type in the registry; no entry → `nullptr` + `spdlog::error`.
  4. Returns `fn(cfg)`.

The type → model-name map (DetectType):

- `model.index.json` `"model": "DeepSeek-V4-Flash"` → type `"deepseek-v4-flash"`
  (lowercase, hyphen instead of space). Normalization: lowercase + replace
  non-`[a-z0-9]` with `-`, trim possible suffixes if needed (refined at
  implementation — see the "Open items" section).

File `libs/lsxcommon/src/model_registry.cpp`:

```cpp
#include "lsxcommon/model_factory.h"
#include "deepseek_v4_flash_model.h"

namespace lsxcommon {
namespace {
const bool kRegDeepSeekV4Flash = RegisterModel(
    DeepSeekV4FlashModel::kType,
    [](const ModelInitConfig &cfg) { return std::make_unique<DeepSeekV4FlashModel>(cfg); });
} // namespace

void EnsureModelRegistry() {
    static bool init = [] {
        (void)kRegDeepSeekV4Flash;   // touch: keeps the entry under STATIC linking
        return true;
    }();
    (void)init;
}
} // namespace lsxcommon
```

## 7. App updates

### apps/lsxfabric/src/main.cpp

Replace direct calls to `gpu::RunInference(cfg)` / `gpu::RunSelfCheck(cfg)`:

```cpp
lsxcommon::ModelInitConfig cfg{model_path, /*model_type=*/"", /*max_gen_tokens=*/64};
auto model = lsxcommon::LsxModelFactory::Create(cfg);
if (!model || !model->Load()) {
    spdlog::error("Failed to create/load model: {}", model_path);
    return 1;
}
auto &tok = model->Tokenizer();
// tokenize the prompt (as in the current testprompt code, via tok),
// result → std::vector<int32_t> input_ids
lsxcommon::ModelRequest req{std::move(input_ids), 64};
auto result = model->Infer(req);
spdlog::info(... result.output_text, result.tokens_per_sec ...);
model->Unload();
```

- `--localmodel`, `--testprompt`, `--selfcheck` flags and their meaning are unchanged.
- For `--selfcheck`:
  ```cpp
  lsxcommon::ModelRequest req{std::move(prompt_ids), 1};
  auto result = model->SelfCheck(req);
  ```
  (`RunSelfCheck` in the current code wraps `RunInference` with `max_gen_tokens=1`
  — we reproduce that inside `SelfCheck`).

### apps/lsxcli/src/main.cpp

Fully switches to the factory (`--info` / `--infofull`):

```cpp
lsxcommon::ModelInitConfig cfg{model_path, /*model_type=*/"", /*max_gen_tokens=*/64};
auto model = lsxcommon::LsxModelFactory::Create(cfg);
if (!model || !model->Load()) {
    spdlog::error("Failed to create/load model: {}", model_path);
    return 1;
}
if (info || infofull) {
    auto mi = model->Inspect();
    if (infofull)
        for (auto &shard : mi.shards)
            lsxcommon::print_shard(shard);
    lsxcommon::print_model(mi);
}
model->Unload();
```

- `scan_directory` / `print_shard` / `print_model` remain public utilities
  in `safetensors.h`, but from now on are called only via `IModel::Inspect()`.

## 8. Tests

New file `tests/test_model_factory.cpp` (Catch2, added to CMake alongside
`test_core.cpp`):

1. `DetectType(valid_path)` → `deepseek-v4-flash` (for an existing model path).
2. `DetectType(invalid_path)` → empty string, no exceptions.
3. `Create({model_type="unknown-type"})` → `nullptr`.
4. Registry round-trip: `Register("mock", ...)` + `Create` returns an instance.
5. `Create(valid_path)` → a non-null `IModel`, `Type() == "deepseek-v4-flash"`,
   `Load()` ok. `Inspect()` returns a non-zero number of shards. (Does not require a GPU —
   file scanning.) To skip when the model is absent — a guard on the existence of
   `logestix/model.index.json`.

Existing tests (`tests/test_core.cpp` etc.) are unchanged, except the need to
recompile: they do not call `RunInference`/`RunSelfCheck`.

## 9. CMake

- `libs/lsxcommon/CMakeLists.txt`: new sources in `lsxcommon`:
  `src/model_factory.cpp`, `src/model_registry.cpp`,
  `src/deepseek_v4_flash_model.cpp`.
- `tests/CMakeLists.txt`: new test target `test_model_factory` on `tests/test_model_factory.cpp`, links `lsxcommon` (like the existing tests).
- Header-only API: `model.h`, `model_factory.h` — in `target_include_directories`
  PUBLIC include `libs/lsxcommon/include` (matches the existing structure).

## 10. Error handling

- No exceptions in the public `IModel` contract.
  - `Load()` → `bool`; on error it logs `spdlog::error` and returns `false`.
  - `Infer()` / `SelfCheck()` → `gpu::InferenceResult`; on error
    `result.ok == false` + `result.error_msg`.
  - `Create()` → `nullptr` on an unknown type / a missing `model.index.json`.
  - `DetectType()` → empty string on error (does not throw).
- The app cascade: check `Create()` for `nullptr`, `Load()` for `false`,
  `Infer()` for `result.ok` — at each stage.

## 11. Non-goals

- Parameterizing the engine geometry constants (a second model of this architecture with
  other sizes is not in scope).
- New models (testing the factory on a real second model).
- Sampling: `temperature`/`top_p`/sampling on top of argmax, batching, the server wrapper.
- Changing the `logestix/model.index.json` format.

## 12. Open items (resolved at implementation)

- The exact `model` → `type` normalization: for now `"DeepSeek-V4-Flash"` → `"deepseek-v4-flash"`
  (lowercase, ` `→`-`, `-`→`-`). If real models show other patterns —
  refine the transformation table.
- Prompt tokenization in `lsxfabric` testprompt: the current code tokenizes a hardcoded
  prompt; the prompt text can stay there, but via `tok` from `IModel::Tokenizer()`.
- The home model directory for tests (the `LSX_TEST_MODEL_PATH` environment variable,
  falling back to `D:\Programs\huggingface.co\DeepSeek-V4-Flash`).
