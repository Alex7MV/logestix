# Centralized Metadata Vault (model-metadata/) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a single compile-time-selected platform root (`D:\Programs\huggingface.co` / `/mnt/llm_models/models`) and route structural JSON manifests and the binary translation passport into the unified `model-metadata/` vault by default.

**Architecture:** A single path authority (`lsxcommon/path_registry`) resolves the platform root + auto-appends the invariant `model-metadata/` block to every metadata read/write query. Manifest reads use vault-first, per-checkpoint-fallback resolution (compositional, non-breaking). The passport builder defaults its output into the vault. Validated by platform-asset unit tests; real-model tests stay SKIP-guarded and unaffected.

**Tech Stack:** C++20 (path registry), CUDA `.cu` consumers via `lsxcommon_cu` include path, Catch2 v3.7.0 (test only), CMake 3.25+, spdlog.

**Guardrail (explicitly out of scope):** Model *weight* tensors (`pinned/*.arrow`, `experts/*.arrow`, `.safetensors`) and the tokenizer arrows co-located inside `logestix/pinned/` are NOT pure metadata — they stay at `{model_path}/logestix/...`. Moving them would silently fragment the runtime loaders. Only isolated metadata artifacts are re-routed.

---

### Task 1: Path registry authority (`lsxcommon`)

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/path_registry.h`
- Create: `libs/lsxcommon/src/path_registry.cpp`
- Modify: `libs/lsxcommon/CMakeLists.txt` (add source to `lsxcommon`)

- [ ] **Step 1: Add the header**

Create `libs/lsxcommon/include/lsxcommon/path_registry.h` with:

```cpp
#pragma once

#include <string>

namespace lsxcommon {

// Compile-time platform selector for the model root (CC 12.0 build matrix).
#if defined(_WIN32)
constexpr const char* kModelRootBase = "D:\\Programs\\huggingface.co";
#else
constexpr const char* kModelRootBase = "/mnt/llm_models/models";
#endif

// Invariant metadata block appended to every metadata read/write query.
constexpr const char* kModelMetadataDir = "model-metadata";

// "<base>/model-metadata" (platform separator) — the vault root.
std::string lsx_model_metadata_root();

// Appends the invariant "model-metadata/" block to a relative subpath:
//   lsx_model_metadata("passport/passport.bin")
// returns "<base>/model-metadata/passport/passport.bin" with platform
// separators. No other code may reconstruct the vault root.
std::string lsx_model_metadata(const std::string& rel);

// "<base>" — the models directory holding named checkpoints.
std::string lsx_model_models_dir();

} // namespace lsxcommon
```

- [ ] **Step 2: Add the implementation**

Create `libs/lsxcommon/src/path_registry.cpp` with:

```cpp
#include "lsxcommon/path_registry.h"

#include <filesystem>

namespace lsxcommon {

namespace {
std::filesystem::path base()
{
    return std::filesystem::path(kModelRootBase);
}
} // namespace

std::string lsx_model_metadata_root()
{
    return (base() / kModelMetadataDir).string();
}

std::string lsx_model_metadata(const std::string& rel)
{
    return (base() / kModelMetadataDir / rel).string();
}

std::string lsx_model_models_dir()
{
    return base().string();
}

} // namespace lsxcommon
```

- [ ] **Step 3: Register the source**

In `libs/lsxcommon/CMakeLists.txt`, add `src/path_registry.cpp` to the `lsxcommon` STATIC library source list (inside the `add_library(lsxcommon STATIC ...)` list, keeping alphabetical order — e.g. after `src/model_registry.cpp`).

- [ ] **Step 4: Configure and build**

Run: `cmake --preset windows-debug && cmake --build --preset windows-debug`
Expected: `lsxcommon` compiles with no new errors. (Windows is the primary host; `linux-debug` is equivalent on the Ubuntu box.)

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/path_registry.h libs/lsxcommon/src/path_registry.cpp libs/lsxcommon/CMakeLists.txt
git commit -m "feat: add centralized model path registry (path_registry)"
```

---

### Task 2: Path registry unit tests

**Files:**
- Create: `tests/test_path_registry.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write the failing test**

Create `tests/test_path_registry.cpp` with:

```cpp
#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/path_registry.h"

#include <filesystem>
#include <string>

TEST_CASE("path_registry platform root base", "[path_registry]")
{
    const std::string models = lsxcommon::lsx_model_models_dir();
    REQUIRE_FALSE(models.empty());
#if defined(_WIN32)
    CHECK(models == R"(D:\Programs\huggingface.co)");
#else
    CHECK(models == "/mnt/llm_models/models");
#endif
}

TEST_CASE("path_registry vault root carries model-metadata block", "[path_registry]")
{
    const std::string root = lsxcommon::lsx_model_metadata_root();
    const std::filesystem::path p(root);
    REQUIRE(p.filename() == "model-metadata");
    REQUIRE(p.parent_path().string() == lsxcommon::lsx_model_models_dir());
}

TEST_CASE("path_registry appends inherited subpath after block", "[path_registry]")
{
    const std::string pathed = lsxcommon::lsx_model_metadata("passport/passport.bin");
    const std::filesystem::path p(pathed);
    REQUIRE(p.filename() == "passport.bin");
    REQUIRE(p.parent_path().filename() == "passport");
    REQUIRE(p.parent_path().parent_path().filename() == "model-metadata");
}

TEST_CASE("path_registry round-trips into product directory", "[path_registry]")
{
    namespace fs = std::filesystem;
    auto mans = lsxcommon::lsx_model_metadata("manifests/model.index.json");
    auto dir = fs::path(mans).parent_path();
    REQUIRE(dir.string().rfind(lsxcommon::lsx_model_metadata_root(), 0) == 0);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `ctest --preset windows-debug -R test_path_registry --output-on-failure`
Expected: FAIL — target `test_path_registry` not found.

- [ ] **Step 3: Register the test**

In `tests/CMakeLists.txt`, append (mirroring `test_deepseek_loader`):

```cmake
add_executable(test_path_registry test_path_registry.cpp)
target_link_libraries(test_path_registry PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_path_registry POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_path_registry>"
    )
    add_custom_command(TARGET test_path_registry POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_path_registry>"
    )
endif()

add_test(NAME test_path_registry COMMAND test_path_registry)
```

- [ ] **Step 4: Rebuild and run to verify it passes**

Run: `cmake --preset windows-debug && cmake --build --preset windows-debug && ctest --preset windows-debug -R test_path_registry --output-on-failure`
Expected: PASS — 4 test cases green.

- [ ] **Step 5: Commit**

```bash
git add tests/test_path_registry.cpp tests/CMakeLists.txt
git commit -m "test: add path_registry unit tests"
```

---

### Task 3: Route structural JSON manifest reads to the vault

**Files:**
- Modify: `libs/lsxcommon/src/inference_load.cu` (`detect_model_type` ~L205, `load_topology` ~L170)

Vault-first, per-checkpoint-fallback resolution keeps synthetic tests (which pass explicit temp `model_path`) and existing real-model loads working; only when the checkpoint manifest is absent AND the vault manifest exists is the vault consulted.

- [ ] **Step 1: Add the registry include**

At the top of `libs/lsxcommon/src/inference_load.cu`, add the include (the file already sits in the `lsxcommon_cu` OBJECT lib whose include dirs include `lsxcommon/include`):

```cpp
#include "lsxcommon/path_registry.h"
```

- [ ] **Step 2: Modify `load_topology` (DeepSeek + GLM-generic path)**

Locate `lsxcommon::load_topology(const std::string &model_path)` starting at the line `ModelTopology load_topology(const std::string &model_path)`. Replace the single line

```cpp
    auto ipath = std::filesystem::path(model_path) / "logestix" / "model.index.json";
```

with

```cpp
    auto ipath = std::filesystem::path(model_path) / "logestix" / "model.index.json";
    const auto vault = std::filesystem::path(
        lsxcommon::lsx_model_metadata("manifests/model.index.json"));
    if (!std::filesystem::exists(ipath) && std::filesystem::exists(vault))
        ipath = vault;
```

- [ ] **Step 3: Modify `detect_model_type`**

Locate `std::string detect_model_type(const std::string &model_path)` at L205. Replace the single line

```cpp
    auto ipath = std::filesystem::path(model_path) / "logestix" / "model.index.json";
```

with the same two-line vault-first resolution as Step 2 (identical snippet).

- [ ] **Step 4: Build and run the DeepSeek + factory tests**

Run: `cmake --build --preset windows-debug && ctest --preset windows-debug -R "test_deepseek_loader|test_model_factory" --output-on-failure`
Expected: PASS (existing synthetic round-trip and factory registration unchanged; real-model cases SKIP as before).

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/inference_load.cu
git commit -m "feat: route model index JSON manifest through model-metadata vault"
```

---

### Task 4: Route the binary translation passport into the vault

**Files:**
- Modify: `tools/lsx_token_packer.cpp`
- Modify: `tools/CMakeLists.txt`
- Modify: `libs/lsxnetwork/include/lsxnetwork/lsx_token_translator.h` (doc note only)

- [ ] **Step 1: Default the passport output to the vault**

In `tools/lsx_token_packer.cpp`:

- Add includes at the top:

```cpp
#include "lsxcommon/path_registry.h"
#include <filesystem>
```

- Change the `out_path` default from empty to the vault default:

```cpp
    std::string glm_path, ds_path, out_path =
        lsxcommon::lsx_model_metadata("passport/passport.bin");
```

- Relax the required-args gate so only the two vocabularies are mandatory. Replace

```cpp
    if (!(have_glm && have_ds && have_out)) {
        PrintUsage();
        return 2;
    }
```

with

```cpp
    if (!(have_glm && have_ds)) {
        PrintUsage();
        return 2;
    }
    if (!have_out)
        SPDLOG_INFO("lsx_token_packer: --out defaulted to '{}'", out_path);
```

- [ ] **Step 2: Link lsxcommon into the packer**

In `tools/CMakeLists.txt`, change the link line to add `lsxcommon`, and stage the Arrow DLLs (the STATIC-linker discards unreferenced Arrow objects, but staging keeps the tool robust on Windows regardless):

```cmake
add_executable(lsx_token_packer lsx_token_packer.cpp)
target_link_libraries(lsx_token_packer PRIVATE lsxnetwork lsxcommon spdlog::spdlog)

if(WIN32)
    add_custom_command(TARGET lsx_token_packer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>" "$<TARGET_FILE_DIR:lsx_token_packer>"
    )
    add_custom_command(TARGET lsx_token_packer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>" "$<TARGET_FILE_DIR:lsx_token_packer>"
    )
endif()

install(TARGETS lsx_token_packer RUNTIME DESTINATION bin)
```

- [ ] **Step 3: Document the vault default in the translator header**

In `libs/lsxnetwork/include/lsxnetwork/lsx_token_translator.h`, directly above `int WritePassport(const char* path, const BuildPassportOptions& opts);`, add a doc comment:

```cpp
/// The passport is written/read from the unified vault by default; callers
/// resolve the location via lsxcommon::lsx_model_metadata("passport/passport.bin").
```

- [ ] **Step 4: Build and smoke-test the packer**

Run: `cmake --preset windows-debug && cmake --build --preset windows-debug`
Then: `build\bin\Debug\lsx_token_packer.exe --help`
Expected: prints usage; build has no link errors and no missing-DLL error on invocation.

- [ ] **Step 5: Commit**

```bash
git add tools/lsx_token_packer.cpp tools/CMakeLists.txt libs/lsxnetwork/include/lsxnetwork/lsx_token_translator.h
git commit -m "feat: route translation passport into model-metadata vault by default"
```

---

### Task 5: Unify hardcoded model-root defaults across apps

**Files:**
- Modify: `apps/lsxcli/src/main.cpp:11`
- Modify: `apps/lsxfabric/src/main.cpp:17`

- [ ] **Step 1: Add the registry include to both apps**

Add to both `apps/lsxcli/src/main.cpp` and `apps/lsxfabric/src/main.cpp` (they already include `lsxcommon/*` headers):

```cpp
#include "lsxcommon/path_registry.h"
```

- [ ] **Step 2: Replace the Windows-only default model path in `lsxcli`**

In `apps/lsxcli/src/main.cpp`, replace

```cpp
    std::string model_path = R"(D:\Programs\huggingface.co\DeepSeek-V4-Flash)";
```

with

```cpp
    auto model_path = (std::filesystem::path(lsxcommon::lsx_model_models_dir())
                       / "DeepSeek-V4-Flash").string();
```

and add `#include <filesystem>` to its includes block.

- [ ] **Step 3: Replace the Windows-only default in `lsxfabric`**

In `apps/lsxfabric/src/main.cpp`, replace

```cpp
    std::string model_path = R"(D:\Programs\huggingface.co\DeepSeek-V4-Flash)";
```

with the identical `std::filesystem::path(lsxcommon::lsx_model_models_dir()) / "DeepSeek-V4-Flash"` expression (Step 2), adding `#include <filesystem>`.

- [ ] **Step 4: Build both apps**

Run: `cmake --build --preset windows-debug`
Expected: `lsxcli.exe` and `lsxfabric.exe` build cleanly.

- [ ] **Step 5: Commit**

```bash
git add apps/lsxcli/src/main.cpp apps/lsxfabric/src/main.cpp
git commit -m "refactor: derive app model-path defaults from path_registry"
```

---

## Self-Review

**Spec coverage:**
- Requirement 1 (compile-time `_WIN32`/Linux root selector + forced `model-metadata/` append): Tasks 1–2. ✔
- Requirement 2a (model loading registries route structural JSON manifests into vault): Task 3 (`detect_model_type`/`load_topology` vault-first). ✔
- Requirement 2b (token packing config headers route binary passport into vault): Task 4. ✔
- CLI hardcoded-root routing: Task 5. ✔

**Out of scope (documented in Goal/Guardrail):** weight tensors + tokenizer arrows inside `logestix/pinned/` remain co-located with the checkpoint; their relocation is a tracked follow-up requiring loader migration. This keeps the plan a single self-contained spec unit and Phase 4 tests green.

**Placeholder scan:** every code step has exact final code and exact commands. No TBD/TODO/imitation-of-prior-task. The only verbatim-repeated snippets (Task 3 Steps 2/3 vault resolution, Task 5 Steps 2/3 app default) are repeated intentionally and fully spelled out as the writing-plans skill requires.

**Type consistency:** `lsx_model_metadata_root()` / `lsx_model_metadata(rel)` / `lsx_model_models_dir()` names are defined once in Task 1 and used identically in Tasks 2–5. `kModelRootBase` / `kModelMetadataDir` constants match. `test_path_registry` target name is consistent between Task 2's test file, CMake registration, and ctest run.