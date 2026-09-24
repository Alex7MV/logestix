# lsxfabric Audit Remediation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Do NOT dispatch subagents** (AGENTS.md forbids them).

**Goal:** Resolve the High/Medium lsxfabric findings by delegating validation to the inference engine and fixing the CLI parser.

**Architecture:** `--validate` becomes a thin wrapper over `InferenceEngine::CreateModel` → `IModel::Load` → `Unload`. All model-specific logic is deleted from `validator.cpp`, which keeps one function. `cli.cpp` gains strict value parsing that never swallows a known flag.

**Tech Stack:** C++20, CMake `linux-debug`, Catch2 v3.16.0, `lsxcommon` (engine/registry), CUDA 13.4, RTX 5090.

## Global Constraints

- Branch `audit/lsxfabric-functional-audit`; production code in `apps/lsxfabric/src/` may now change.
- `libs/lsxcommon/` is NOT modified.
- Every CTest target keeps its existing label; no new `add_test`.
- Commit prefixes `test:` / `refactor:` / `docs:`; stage only intended files.
- GPU-exclusivity: the integration task runs last, sequentially.
- Logging in tests follows the existing `Core::Initialize`/`Shutdown` pattern.
- Use `absl::StrFormat` (printf-style) for any new logging.

---

### Task 1: CLI parser fix (F4/F5)

**Files:**
- Modify: `apps/lsxfabric/tests/test_fabric_cli_edges.cpp`
- Modify: `apps/lsxfabric/src/cli.cpp`

**Interfaces:**
- Consumes: `lsxfabric::ParseFabricCliArgs(int, char**)` → `FabricCliArgs{model_path, prompt, max_gen, validate, testprompt}`.
- Produces: same signature; new semantics (no flag swallowing, strict `--max-gen`).

- [ ] **Step 1: Rewrite the CLI tests to the fixed behaviour**

Replace the entire contents of `apps/lsxfabric/tests/test_fabric_cli_edges.cpp` with:

```cpp
#include <catch2/catch_test_macros.hpp>
#include "cli.h"

#include <string>
#include <vector>

namespace {

std::vector<char *> MakeArgv(const std::vector<std::string> &storage) {
    std::vector<char *> argv;
    argv.reserve(storage.size());
    for (const std::string &s : storage)
        argv.push_back(const_cast<char *>(s.c_str()));
    return argv;
}

lsxfabric::FabricCliArgs Parse(const std::vector<std::string> &args) {
    std::vector<char *> argv = MakeArgv(args);
    return lsxfabric::ParseFabricCliArgs(static_cast<int>(argv.size()),
                                         argv.data());
}

} // namespace

TEST_CASE("fabric cli: flag without a value is ignored", "[lsxfabric][cli]") {
    REQUIRE(Parse({"lsxfabric", "--localmodel"}).model_path.find(
                "DeepSeek-V4-Flash") != std::string::npos);
    REQUIRE(Parse({"lsxfabric", "--max-gen"}).max_gen == 64);
    REQUIRE(Parse({"lsxfabric", "--prompt"}).prompt ==
            "The capital of France is");
}

TEST_CASE("fabric cli: a value flag does not swallow a known flag",
          "[lsxfabric][cli]") {
    const lsxfabric::FabricCliArgs p = Parse({"lsxfabric", "--prompt", "--validate"});
    REQUIRE(p.prompt == "The capital of France is");
    REQUIRE(p.validate);

    const lsxfabric::FabricCliArgs m =
        Parse({"lsxfabric", "--localmodel", "--validate", "/tmp/x"});
    REQUIRE(m.model_path.find("DeepSeek-V4-Flash") != std::string::npos);
    REQUIRE(m.validate);
}

TEST_CASE("fabric cli: non-numeric or non-positive --max-gen keeps the default",
          "[lsxfabric][cli]") {
    REQUIRE(Parse({"lsxfabric", "--max-gen", "abc"}).max_gen == 64);
    REQUIRE(Parse({"lsxfabric", "--max-gen", "0"}).max_gen == 64);
    REQUIRE(Parse({"lsxfabric", "--max-gen", "-5"}).max_gen == 64);
    REQUIRE(Parse({"lsxfabric", "--max-gen", "12x"}).max_gen == 64);
    REQUIRE(Parse({"lsxfabric", "--max-gen", "12"}).max_gen == 12);
}

TEST_CASE("fabric cli: a non-flag value is still consumed", "[lsxfabric][cli]") {
    REQUIRE(Parse({"lsxfabric", "--prompt", "hello world"}).prompt == "hello world");
    REQUIRE(Parse({"lsxfabric", "--localmodel", "/tmp/model_x"}).model_path ==
            "/tmp/model_x");
}

TEST_CASE("fabric cli: repeated flags, last one wins", "[lsxfabric][cli]") {
    const lsxfabric::FabricCliArgs a =
        Parse({"lsxfabric", "--max-gen", "5", "--max-gen", "9",
               "--localmodel", "/a", "--localmodel", "/b"});
    REQUIRE(a.max_gen == 9);
    REQUIRE(a.model_path == "/b");
}

TEST_CASE("fabric cli: help flags are unknown and ignored", "[lsxfabric][cli]") {
    const lsxfabric::FabricCliArgs a = Parse({"lsxfabric", "--help", "-h"});
    REQUIRE(a.model_path.find("DeepSeek-V4-Flash") != std::string::npos);
    REQUIRE_FALSE(a.validate);
    REQUIRE_FALSE(a.testprompt);
}
```

- [ ] **Step 2: Build and confirm the tests fail**

Run: `cmake --build --preset linux-debug --target test_fabric_cli && ./build/linux-debug/apps/lsxfabric/tests/test_fabric_cli`
Expected: failures in "does not swallow a known flag" and "--max-gen keeps the default".

- [ ] **Step 3: Implement the fixed parser**

Replace the entire contents of `apps/lsxfabric/src/cli.cpp` with:

```cpp
#include "cli.h"

#include "lsxcommon/path_registry.h"

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <filesystem>

namespace lsxfabric {

namespace {

bool IsKnownFlag(const char *s) {
    return std::strcmp(s, "--localmodel") == 0 ||
           std::strcmp(s, "--validate") == 0 ||
           std::strcmp(s, "--testprompt") == 0 ||
           std::strcmp(s, "--max-gen") == 0 ||
           std::strcmp(s, "--prompt") == 0;
}

bool ParsePositiveInt(const char *s, int &out) {
    errno = 0;
    char *end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v < 1 || v > INT_MAX)
        return false;
    out = static_cast<int>(v);
    return true;
}

} // namespace

FabricCliArgs ParseFabricCliArgs(int argc, char *argv[]) {
    FabricCliArgs a;
    a.model_path = (std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
                    "DeepSeek-V4-Flash")
                       .string();
    for (int i = 1; i < argc; ++i) {
        const bool has_value = (i + 1 < argc) && !IsKnownFlag(argv[i + 1]);
        if (std::strcmp(argv[i], "--localmodel") == 0) {
            if (has_value) a.model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--validate") == 0) {
            a.validate = true;
        } else if (std::strcmp(argv[i], "--testprompt") == 0) {
            a.testprompt = true;
        } else if (std::strcmp(argv[i], "--max-gen") == 0) {
            int v = 0;
            if (has_value && ParsePositiveInt(argv[i + 1], v)) {
                a.max_gen = v;
                ++i;
            }
        } else if (std::strcmp(argv[i], "--prompt") == 0) {
            if (has_value) a.prompt = argv[++i];
        }
    }
    return a;
}

} // namespace lsxfabric
```

- [ ] **Step 4: Build and confirm green**

Run: `cmake --build --preset linux-debug --target test_fabric_cli && ./build/linux-debug/apps/lsxfabric/tests/test_fabric_cli`
Expected: all pass (including the pre-existing `test_fabric_cli.cpp` cases).

- [ ] **Step 5: Commit**

```bash
git add apps/lsxfabric/src/cli.cpp apps/lsxfabric/tests/test_fabric_cli_edges.cpp
git commit -m "fix: stop lsxfabric CLI flags from swallowing values and validate --max-gen"
```

---

### Task 2: Delegate validation to the engine (F1/F6/F7/F8/F12/F14)

**Files:**
- Modify: `apps/lsxfabric/src/validator.h`
- Modify: `apps/lsxfabric/src/validator.cpp`
- Modify: `apps/lsxfabric/src/main.cpp`
- Modify: `apps/lsxfabric/tests/test_validator.cpp`
- Delete: `apps/lsxfabric/tests/test_validator_contract.cpp`
- Modify: `apps/lsxfabric/tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `lsxcommon::InferenceEngine::Instance()`, `CreateModel(const ModelInitConfig&)`, `IModel::Load()/Unload()`.
- Produces: `bool lsxfabric::RunModelValidation(const std::string &model_path)`.

- [ ] **Step 1: Rewrite the validator wrapper test (fails to compile until Step 3)**

Replace the entire contents of `apps/lsxfabric/tests/test_validator.cpp` with:

```cpp
#include <catch2/catch_test_macros.hpp>
#include "validator.h"
#include "lsxcommon/core.h"
#include "lsxcommon/model_engine.h"

#include <string>

TEST_CASE("RunModelValidation rejects an undetectable model path",
          "[lsxfabric][validate]") {
    const std::string log_path = "test_validator.log";
    lsxcommon::Core::Initialize(log_path);
    {
        lsxcommon::InferenceEngine::Session session;
        REQUIRE_FALSE(
            lsxfabric::RunModelValidation("/nonexistent/logestix-audit-model"));
    }
    lsxcommon::Core::Shutdown();
    std::remove(log_path.c_str());
}
```

- [ ] **Step 2: Delete the removed-API tests and unregister them**

```bash
git rm apps/lsxfabric/tests/test_validator_contract.cpp
```

In `apps/lsxfabric/tests/CMakeLists.txt`, delete the line:

```cmake
target_sources(test_lsxfabric PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test_validator_contract.cpp)
```

- [ ] **Step 3: Replace the validator implementation with the engine wrapper**

Replace the entire contents of `apps/lsxfabric/src/validator.h` with:

```cpp
#pragma once

#include <string>

namespace lsxfabric {

// Validates a converted model by loading it through the inference engine.
// Every model-specific detail (format tag, tensor/expert/shared/router layout,
// schemas) is owned by the engine's registry and per-family loaders; this
// wrapper only drives the engine lifecycle. Returns true iff the model is
// detected, created and loaded successfully.
bool RunModelValidation(const std::string &model_path);

} // namespace lsxfabric
```

Replace the entire contents of `apps/lsxfabric/src/validator.cpp` with:

```cpp
#include "validator.h"

#include "lsxcommon/log.h"
#include "lsxcommon/model_engine.h"

#include <absl/strings/str_format.h>

#include <memory>

namespace lsxfabric {

bool RunModelValidation(const std::string &model_path) {
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model_path;

    std::unique_ptr<lsxcommon::IModel> model =
        lsxcommon::InferenceEngine::Instance().CreateModel(cfg);
    if (!model) {
        lsxcommon::log::error(absl::StrFormat(
            "Validation failed: cannot create a model for %s", model_path));
        return false;
    }

    const bool ok = model->Load();
    model->Unload();
    if (!ok)
        lsxcommon::log::error(
            absl::StrFormat("Validation failed: load error for %s", model_path));
    return ok;
}

} // namespace lsxfabric
```

- [ ] **Step 4: Point `main` at the wrapper**

In `apps/lsxfabric/src/main.cpp`, replace the validation branch:

```cpp
        } else if (cli.validate) {
            lsxcommon::log::info("Running validation...");
            auto result = lsxfabric::RunValidator(cli.model_path);
            rc = result.ok ? 0 : 1;
        }
```

with:

```cpp
        } else if (cli.validate) {
            lsxcommon::log::info("Running validation...");
            rc = lsxfabric::RunModelValidation(cli.model_path) ? 0 : 1;
        }
```

- [ ] **Step 5: Build and run**

Run: `cmake --build --preset linux-debug --target test_lsxfabric && ./build/linux-debug/apps/lsxfabric/tests/test_lsxfabric`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add apps/lsxfabric/src/validator.h apps/lsxfabric/src/validator.cpp \
        apps/lsxfabric/src/main.cpp apps/lsxfabric/tests/test_validator.cpp \
        apps/lsxfabric/tests/CMakeLists.txt
git commit -m "refactor: validate lsxfabric models through the inference engine"
```

---

### Task 3: Integration evidence (GPU-exclusive, run LAST)

**Files:** none; capture to `/tmp/lsxfabric_audit2/`.

- [ ] **Step 1: Build the app**

Run: `cmake --build --preset linux-debug --target lsxfabric`
Expected: builds clean.

- [ ] **Step 2: Negative case**

Run: `./build/linux-debug/apps/lsxfabric/lsxfabric --validate --localmodel /nonexistent/model; echo rc=$?`
Expected: non-zero `rc` (validation must fail for an undetectable model).

- [ ] **Step 3: Positive sweep over three representatives (sequential)**

Run, one at a time (each is a real load; nothing else on the GPU):

```bash
mkdir -p /tmp/lsxfabric_audit2
for m in DeepSeek-V4-Flash granite-4.2-30b Ornith-1.5-35B-A3B-MLX; do
  ./build/linux-debug/apps/lsxfabric/lsxfabric \
      --localmodel "/mnt/llm_models/models/$m" --validate \
      > "/tmp/lsxfabric_audit2/validate_$m.log" 2>&1
  echo "$m rc=$?"
done
```

Expected: `rc=0` for all three (each loads through its family loader). Capture each log tail into the report.

- [ ] **Step 4: Generation smoke**

Run: `./build/linux-debug/apps/lsxfabric/lsxfabric --localmodel /mnt/llm_models/models/gemma-4-26B-A4B-it --testprompt --max-gen 32`
Expected: `rc=0`, token stream printed.

- [ ] **Step 5: Commit nothing** — evidence only.

---

### Task 4: Update the audit report

**Files:**
- Modify: `docs/LSXFABRIC_AUDIT.md`

- [ ] **Step 1: Add the Remediation section**

Insert, immediately after the "Findings" table, a section:

```markdown
## Remediation

`--validate` no longer reimplements model knowledge. It is now a thin wrapper
over the inference engine (`InferenceEngine::CreateModel` → `IModel::Load` →
`Unload`); the per-family loaders own every model specific. `validator.cpp`'s
duplicated model logic was deleted. The CLI parser was fixed.

| id | status | note |
|---|---|---|
| F1 F6 F7 F8 F12 F14 | Fixed | delegated to the engine load path (no format allow-list, no fabricated expert loop, no pinned accounting) |
| F4 F5 | Fixed | `cli.cpp`: value flags never swallow a known flag; `--max-gen` is parsed strictly |
| F2 F9 F10 F13 | Moot | the code that carried them was deleted |
| F11 | Deferred | silent no-op mode in `main` is out of scope |

Related observation (out of scope): `lsxcommon::integrity::VerifyConvertedModel`
(used by `lsxforge --verify` and converters) keeps its own stale format list and
`pinned/`-only count; it rejects `logestix-ornithmoe`,
`logestix-qwen38flashnext`, `logestix-glm5next` and `logestix-v41-nvfp4`.

### Integration evidence (remediated)

<integration2>
```

- [ ] **Step 2: Fill in `<integration2>` from Task 3** (negative case, three-model sweep results, generation smoke), then confirm no placeholder remains:

Run: `grep -n "<integration2>" docs/LSXFABRIC_AUDIT.md`
Expected: no output.

- [ ] **Step 3: Update the "how to run" note** — remove the hidden-spec commands, since `[.audit-spec]` tests no longer exist.

- [ ] **Step 4: Commit**

```bash
git add docs/LSXFABRIC_AUDIT.md
git commit -m "docs: record lsxfabric audit remediation"
```

---

### Task 5: Final verification

- [ ] **Step 1: Fast suite green**

Run: `cmake --build --preset linux-debug && ctest --preset linux-debug -R "fabric|lsxfabric|labels"`
Expected: `test_lsxfabric`, `test_fabric_cli`, `test_ctest_labels_guard` pass.

- [ ] **Step 2: No dead references to removed APIs**

Run: `grep -rn "RunValidator\|ValidateExpert\|ValidateNVFP4Data\|LoadModelIndex\|ValidatePinnedTensors" apps/lsxfabric`
Expected: no output.

- [ ] **Step 3: Working tree clean and review the branch**

Run: `git status --short && git log --oneline main..HEAD`
Expected: clean tree; the audit commits plus the remediation commits.

---

## Self-Review

- **Spec coverage:** §3 architecture → Task 2; §4 CLI → Task 1; §5 disposition + §8 report → Task 4; §6 tests → Tasks 1–2; §7 integration → Task 3; §9 risks → Task 5 Step 2 (no dead refs) and the integration subset; §10 out of scope → report note.
- **Placeholder scan:** the only marker is `<integration2>`, explicitly resolved in Task 4 Step 2 before the commit.
- **Type consistency:** `RunModelValidation(const std::string&) -> bool` is declared in Task 2 Step 3 and consumed unchanged in Task 2 Step 4 and the test in Step 1. `IsKnownFlag`/`ParsePositiveInt` are internal to `cli.cpp`.
