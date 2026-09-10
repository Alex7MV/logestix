# lsxbenchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for
> tracking. **AGENTS.md forbids subagents** (`ЗАПРЕТ на
> параллельных/субагентов`) — execute inline in the main session only.

**Goal:** Create `apps/lsxbenchmark` as the single owner of generation-speed
measurement (`--bench`) and bit-exactness verification (`--selfcheck
[--bit-exact]`); prune `lsxfabric` to max-speed validate+generate; delete the
consumer-less `IModel::Benchmark` API.

**Architecture:** Thin app (parser + dispatcher, lsxfabric pattern) over a new
`lsxcommon::runtime::RunCliBenchmark` helper next to `RunCliGeneration` /
`RunCliSelfCheck`. Speed = real generation through `IModel::Infer`
(`InferenceResult` timings, uniform for every model family). The old per-model
`IModel::Benchmark` / `RunBenchmark` shim / `BenchmarkResult` /
`FillBenchmarkResult` / `RequireLoaded(BenchmarkResult&)` are deleted.

**Tech Stack:** C++20, CMake presets, Catch2 v3 (FetchContent), spdlog,
abseil strings, lsxcommon model runtime.

**Spec:** `docs/superpowers/specs/2026-09-10-lsxbenchmark-design.md`

## Global Constraints

- C++20; all logging via spdlog (`lsxcommon::log`) — no `printf`/`std::cout`.
- abseil format discipline (`tests/test_absl_format_guard.cpp`): printf-style
  conversions in `absl::StrFormat` (`%s`, `%d`, `%.1f`), never `{}`;
  prefer `absl::StrCat` for plain concatenations.
- Every new test gets a CTest label in the same commit (`fast`) and
  `TIMEOUT 120` (guard: `test_ctest_labels_guard`).
- Commit messages in English with prefixes `feat:`/`test:`/`refactor:`/`docs:`.
- Linux build: `cmake --preset linux-debug` / `cmake --build --preset
  linux-debug`; tests via `ctest --preset linux-debug` (fast only).
- No Windows-only APIs in new code (`_putenv` etc. forbidden unguarded).
- `CMAKE_RUNTIME_OUTPUT_DIRECTORY` is `${CMAKE_BINARY_DIR}/bin` (set at root).

## File Map (final state)

```
apps/lsxbenchmark/CMakeLists.txt          [create] app target + install + tests subdir
apps/lsxbenchmark/src/cli.h               [create] BenchmarkCliArgs + parser decl
apps/lsxbenchmark/src/cli.cpp             [create] parser impl
apps/lsxbenchmark/src/main.cpp            [create] dispatcher: selfcheck | bench
apps/lsxbenchmark/tests/CMakeLists.txt    [create] test_lsxbenchmark_cli, label fast, TIMEOUT 120
apps/lsxbenchmark/tests/test_lsxbenchmark_cli.cpp [create] 5 parser test cases
apps/CMakeLists.txt                       [modify] + add_subdirectory(lsxbenchmark)
apps/lsxfabric/src/cli.h                  [modify] drop benchmark/selfcheck/bit_exact
apps/lsxfabric/src/cli.cpp                [modify] drop 3 parse branches
apps/lsxfabric/src/main.cpp               [modify] drop benchmark/selfcheck branches + fm var
apps/lsxfabric/tests/test_fabric_cli.cpp  [rewrite] new 3-case contract
libs/lsxcommon/include/lsxcommon/model_runtime.h  [modify] +RunCliBenchmark, -FillBenchmarkResult
libs/lsxcommon/src/model_runtime.cpp      [modify] +RunCliBenchmark, -FillBenchmarkResult
libs/lsxcommon/include/lsxcommon/model.h  [modify] -Benchmark virtual, -benchmark_gpu.h include
libs/lsxcommon/include/lsxcommon/model_base.h     [modify] -RequireLoaded(BenchmarkResult&)
libs/lsxcommon/src/model_base.cpp         [modify] -RequireLoaded impl
libs/lsxcommon/include/lsxcommon/benchmark_gpu.h  [delete]
libs/lsxcommon/src/benchmark_gpu.cpp      [delete]
libs/lsxcommon/src/benchmark_gpu.cu       [delete]
libs/lsxcommon/CMakeLists.txt             [modify] drop benchmark_gpu.cpp / .cu lines
libs/lsxcommon/src/glm_moe_model.{h,cpp}          [modify] -Benchmark override
libs/lsxcommon/src/qwen_moe_model.{h,cpp}         [modify] -Benchmark override (QwenMoeModelBase)
libs/lsxcommon/src/qwen_moe_exp_model.{h,cpp}     [modify] -Benchmark override
libs/lsxcommon/src/kimi_moe_model.{h,cpp}         [modify] -Benchmark override
libs/lsxcommon/src/deepseek_moe_model.{h,cpp}     [modify] -Benchmark override
libs/lsxcommon/src/glm_moe_flash_model.{h,cpp}    [modify] -Benchmark override
libs/lsxcommon/src/gemma_base_model.{h,cpp}       [modify] -Benchmark override
libs/lsxcommon/include/lsxcommon/qwen_base_model.h [modify] -Benchmark override decl
libs/lsxcommon/src/qwen_base_model.cpp    [modify] -Benchmark override impl
tests/test_model_runtime.cpp              [modify] -5 Benchmark test cases, -include, +RunCliBenchmark test
AGENTS.md                                 [modify] flag lists + lsxbenchmark section
```

---

### Task 1: lsxbenchmark CLI parser — RED (test + CMake wiring)

**Files:**
- Create: `apps/lsxbenchmark/CMakeLists.txt`
- Create: `apps/lsxbenchmark/tests/CMakeLists.txt`
- Create: `apps/lsxbenchmark/tests/test_lsxbenchmark_cli.cpp`
- Modify: `apps/CMakeLists.txt`

**Interfaces:**
- Consumes: nothing (new surface).
- Produces: `lsxbenchmark::BenchmarkCliArgs` and
  `lsxbenchmark::ParseBenchmarkCliArgs(int, char**)` — the contract the
  parser tests lock. Struct fields: `std::string model_path; bool bench;
  bool selfcheck; bool bit_exact; std::string prompt; int max_gen;`
  Defaults: `prompt = "The capital of France is"`, `max_gen = 2048`,
  `bench = true` (bench is the default mode), model_path = vault
  `DeepSeek-V4-Flash`.

- [ ] **Step 1: Create `apps/lsxbenchmark/CMakeLists.txt`** (minimal for now —
  only the tests subdirectory, so CMake configure succeeds while the app
  sources do not exist yet; Task 4 completes it with the executable block)

```cmake
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
```

- [ ] **Step 2: Create `apps/lsxbenchmark/tests/CMakeLists.txt`** (label
  `fast` + `TIMEOUT 120` — mandatory guard discipline)

```cmake
add_executable(test_lsxbenchmark_cli test_lsxbenchmark_cli.cpp ../src/cli.cpp)
target_include_directories(test_lsxbenchmark_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(test_lsxbenchmark_cli PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_lsxbenchmark_cli POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_lsxbenchmark_cli>"
    )
    add_custom_command(TARGET test_lsxbenchmark_cli POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_lsxbenchmark_cli>"
    )
endif()

add_test(NAME test_lsxbenchmark_cli COMMAND test_lsxbenchmark_cli)
set_tests_properties(test_lsxbenchmark_cli PROPERTIES LABELS fast)
set_tests_properties(test_lsxbenchmark_cli PROPERTIES TIMEOUT 120)
```

- [ ] **Step 3: Create `apps/lsxbenchmark/tests/test_lsxbenchmark_cli.cpp`**

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

lsxbenchmark::BenchmarkCliArgs Parse(const std::vector<std::string> &args) {
    std::vector<char *> argv = MakeArgv(args);
    return lsxbenchmark::ParseBenchmarkCliArgs(static_cast<int>(argv.size()),
                                               argv.data());
}

} // namespace

TEST_CASE("lsxbenchmark cli: defaults to bench mode on the vault model",
          "[lsxbenchmark][cli]") {
    const lsxbenchmark::BenchmarkCliArgs a = Parse({"lsxbenchmark"});
    REQUIRE(a.bench);
    REQUIRE_FALSE(a.selfcheck);
    REQUIRE_FALSE(a.bit_exact);
    REQUIRE(a.max_gen == 2048);
    REQUIRE(a.prompt == "The capital of France is");
    REQUIRE(a.model_path.find("DeepSeek-V4-Flash") != std::string::npos);
}

TEST_CASE("lsxbenchmark cli: --selfcheck selects verification mode",
          "[lsxbenchmark][cli]") {
    const lsxbenchmark::BenchmarkCliArgs a = Parse({"lsxbenchmark", "--selfcheck"});
    REQUIRE(a.selfcheck);
    REQUIRE_FALSE(a.bench);
    REQUIRE_FALSE(a.bit_exact);
}

TEST_CASE("lsxbenchmark cli: --bit-exact pins the deterministic scalar mode",
          "[lsxbenchmark][cli]") {
    const lsxbenchmark::BenchmarkCliArgs a =
        Parse({"lsxbenchmark", "--selfcheck", "--bit-exact"});
    REQUIRE(a.selfcheck);
    REQUIRE(a.bit_exact);

    const lsxbenchmark::BenchmarkCliArgs b =
        Parse({"lsxbenchmark", "--bench", "--bit-exact"});
    REQUIRE(b.bench);
    REQUIRE(b.bit_exact);
}

TEST_CASE("lsxbenchmark cli: bench options parse", "[lsxbenchmark][cli]") {
    const lsxbenchmark::BenchmarkCliArgs a = Parse(
        {"lsxbenchmark", "--model", "/tmp/model_x", "--bench", "--prompt", "hi",
         "--max-gen", "7"});
    REQUIRE(a.model_path == "/tmp/model_x");
    REQUIRE(a.bench);
    REQUIRE(a.prompt == "hi");
    REQUIRE(a.max_gen == 7);
}

TEST_CASE("lsxbenchmark cli: unknown flags are silently ignored",
          "[lsxbenchmark][cli]") {
    const lsxbenchmark::BenchmarkCliArgs a = Parse(
        {"lsxbenchmark", "--fast-math", "--calibrate", "--wat"});
    REQUIRE(a.bench);
    REQUIRE_FALSE(a.selfcheck);
    REQUIRE_FALSE(a.bit_exact);
    REQUIRE(a.max_gen == 2048);
}
```

- [ ] **Step 4: Register the app directory** — in `apps/CMakeLists.txt` add
  the subdirectory line (keep the existing order, lsxbenchmark after lsxcli):

```cmake
add_subdirectory(lsxfabric)
add_subdirectory(lsxforge)
add_subdirectory(taskmosaic)
add_subdirectory(lsxcli)
add_subdirectory(lsxbenchmark)
add_subdirectory(lsxhome)
```

- [ ] **Step 5: Reconfigure and verify RED** (the parser does not exist yet)

Run:
```bash
cmake --preset linux-debug
```
Expected: CMake **configure fails** on
`apps/lsxbenchmark/tests/CMakeLists.txt` (`Cannot find source file:
.../src/cli.cpp`). This is the RED state — the requested behavior does not
exist yet.

- [ ] **Step 6: Commit the RED test**

```bash
git add apps/lsxbenchmark apps/CMakeLists.txt
git commit -m "test: lsxbenchmark cli parser contract (RED)"
```

---

### Task 2: lsxbenchmark CLI parser — GREEN

**Files:**
- Create: `apps/lsxbenchmark/src/cli.h`
- Create: `apps/lsxbenchmark/src/cli.cpp`

**Interfaces:**
- Consumes: `lsxcommon::lsx_model_models_dir()` from
  `lsxcommon/path_registry.h` (same default-model convention as
  `apps/lsxfabric/src/cli.cpp`).
- Produces: `lsxbenchmark::BenchmarkCliArgs` +
  `lsxbenchmark::ParseBenchmarkCliArgs(int argc, char *argv[])` — consumed by
  Task 4's `main.cpp`. Mode rule: `selfcheck = true` iff `--selfcheck`;
  otherwise bench mode (`bench = true`). `--bench` is accepted explicitly and
  keeps bench mode.

- [ ] **Step 1: Create `apps/lsxbenchmark/src/cli.h`**

```cpp
#pragma once

#include <string>

namespace lsxbenchmark {

// lsxbenchmark launch parameters. This app owns generation-speed measurement
// (--bench) and bit-exactness verification (--selfcheck [--bit-exact]);
// every other app always runs the per-GPU maximum-speed auto perf policy.
struct BenchmarkCliArgs {
    std::string model_path;
    bool bench = true;      // default mode when no mode flag is given
    bool selfcheck = false; // --selfcheck: Paris verification harness
    bool bit_exact = false; // --bit-exact: FastMathMode::kOff for the active mode
    std::string prompt = "The capital of France is";
    int max_gen = 2048;
};

// Parses argv (argv[0] is skipped). Unknown tokens are silently ignored.
// Mode rule: --selfcheck selects verification mode; anything else (including
// no mode flag) is bench mode.
BenchmarkCliArgs ParseBenchmarkCliArgs(int argc, char *argv[]);

} // namespace lsxbenchmark
```

- [ ] **Step 2: Create `apps/lsxbenchmark/src/cli.cpp`**

```cpp
#include "cli.h"

#include "lsxcommon/path_registry.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>

namespace lsxbenchmark {

BenchmarkCliArgs ParseBenchmarkCliArgs(int argc, char *argv[]) {
    BenchmarkCliArgs a;
    a.model_path = (std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
                    "DeepSeek-V4-Flash")
                       .string();
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc)
            a.model_path = argv[++i];
        else if (std::strcmp(argv[i], "--bench") == 0)
            a.bench = true;
        else if (std::strcmp(argv[i], "--selfcheck") == 0) {
            a.selfcheck = true;
            a.bench = false;
        } else if (std::strcmp(argv[i], "--bit-exact") == 0)
            a.bit_exact = true;
        else if (std::strcmp(argv[i], "--max-gen") == 0 && i + 1 < argc)
            a.max_gen = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--prompt") == 0 && i + 1 < argc)
            a.prompt = argv[++i];
    }
    return a;
}

} // namespace lsxbenchmark
```

- [ ] **Step 3: Verify the parser test goes GREEN** — the app executable
  still lacks `main.cpp`, so build only the test target:

```bash
cmake --preset linux-debug && cmake --build --preset linux-debug --target test_lsxbenchmark_cli && ctest --preset linux-debug -R test_lsxbenchmark_cli --output-on-failure
```
Expected: `test_lsxbenchmark_cli` builds, **5 test cases PASS**.

- [ ] **Step 4: Commit**

```bash
git add apps/lsxbenchmark/src/cli.h apps/lsxbenchmark/src/cli.cpp
git commit -m "feat: lsxbenchmark cli parser (bench default, selfcheck + bit-exact modes)"
```

---

### Task 3: `runtime::RunCliBenchmark` in lsxcommon — TDD

**Files:**
- Modify: `tests/test_model_runtime.cpp` (add failing test)
- Modify: `libs/lsxcommon/include/lsxcommon/model_runtime.h`
- Modify: `libs/lsxcommon/src/model_runtime.cpp`

**Interfaces:**
- Consumes: existing `CreateCliModel` (anon-namespace helper in
  `model_runtime.cpp`), `IModel::BuildPromptInputIds`, `IModel::Infer`,
  `GenerationCliArgs` (field `fast_math_mode`).
- Produces: `int lsxcommon::runtime::RunCliBenchmark(const GenerationCliArgs
  &args)` — returns `0` on success, `1` on model-load / encode / infer
  failure. Consumed by Task 4's `main.cpp`.

- [ ] **Step 1: Add the failing host-only test** to
  `tests/test_model_runtime.cpp`, right after the
  `"model_runtime: RunBenchmark shim delegates to IModel::Benchmark"`
  TEST_CASE (which Task 6 deletes):

```cpp
TEST_CASE("model_runtime: RunCliBenchmark returns 1 on a missing model",
          "[model_runtime]")
{
    lsxcommon::runtime::GenerationCliArgs args;
    args.model_path = (fs::temp_directory_path() / "no_such_model_xyz").string();
    REQUIRE(lsxcommon::runtime::RunCliBenchmark(args) == 1);
}
```

- [ ] **Step 2: Verify RED** (declaration missing -> TU fails to compile)

Run: `cmake --build --preset linux-debug --target test_model_runtime`
Expected: **compile error** — `'RunCliBenchmark' is not a member of
'lsxcommon::runtime'`.

- [ ] **Step 3: Declare it** in `libs/lsxcommon/include/lsxcommon/
model_runtime.h`, immediately after the `RunCliSelfCheck` declaration
(currently `int RunCliSelfCheck(const GenerationCliArgs &args);`):

```cpp
int RunCliSelfCheck(const GenerationCliArgs &args);
// Speed-measurement lifecycle (lsxbenchmark --bench): Create -> Load ->
// frame/encode -> Infer -> print -> Unload. Runs at args.fast_math_mode
// (kAuto = maximum-speed auto policy; kOff = bit-exact measurement).
// Returns a process exit code (0 success, 1 failure).
int RunCliBenchmark(const GenerationCliArgs &args);
```

- [ ] **Step 4: Implement it** in `libs/lsxcommon/src/model_runtime.cpp`,
  immediately after `RunCliSelfCheck` (before `RunCliInspect`). Reuse the
  anon-namespace `CliModel`/`CreateCliModel` already in the file:

```cpp
int RunCliBenchmark(const GenerationCliArgs &args)
{
    lsxcommon::log::info(absl::StrFormat(
        "Running generation benchmark on prompt: %s", args.prompt));
    CliModel m = CreateCliModel(args.model_path, args.fast_math_mode);
    if (!m.ok) return 1;

    std::vector<int32_t> input_ids;
    if (!m.model->BuildPromptInputIds(m.model->Tokenizer(), args.prompt,
                                      input_ids)) {
        lsxcommon::log::error(
            absl::StrFormat("Tokenizer encode failed for prompt: %s",
                            args.prompt));
        m.model->Unload();
        return 1;
    }
    lsxcommon::ModelRequest req{std::move(input_ids), args.max_gen_tokens};
    gpu::InferenceResult result = m.model->Infer(req);
    m.model->Unload();
    if (!result.ok) {
        lsxcommon::log::error(
            absl::StrFormat("Benchmark inference failed: %s",
                            result.error_msg));
        return 1;
    }
    lsxcommon::log::info("=== lsxbenchmark: bench ===");
    // StrCat for concatenations (abseil format guard: printf-style specs
    // only inside StrFormat).
    lsxcommon::log::info(absl::StrCat("prompt   : ", args.prompt));
    lsxcommon::log::info(absl::StrCat(
        "tokens   : ", result.num_tokens, " at ",
        absl::StrFormat("%.1f", result.tokens_per_sec), " t/s (",
        absl::StrFormat("%.1f", result.total_time_ms), " ms)"));
    lsxcommon::log::info(
        absl::StrFormat("ttft     : %.2f ms", result.ttft_ms));
    lsxcommon::log::info(absl::StrFormat(
        "resident : %.0f MiB",
        static_cast<double>(result.weight_resident_bytes) /
            (1024.0 * 1024.0)));
    return 0;
}
```

- [ ] **Step 5: Verify GREEN**

Run:
```bash
cmake --build --preset linux-debug --target test_model_runtime && ctest --preset linux-debug -R test_model_runtime --output-on-failure
```
Expected: build OK, all `test_model_runtime` cases PASS (including the new
`RunCliBenchmark returns 1 on a missing model`).

- [ ] **Step 6: Commit**

```bash
git add tests/test_model_runtime.cpp libs/lsxcommon/include/lsxcommon/model_runtime.h libs/lsxcommon/src/model_runtime.cpp
git commit -m "feat: runtime::RunCliBenchmark unified speed-measurement lifecycle"
```

---

### Task 4: lsxbenchmark app main — GREEN app target

**Files:**
- Create: `apps/lsxbenchmark/src/main.cpp`

**Interfaces:**
- Consumes: `lsxbenchmark::ParseBenchmarkCliArgs` (Task 2),
  `lsxcommon::runtime::RunCliBenchmark` / `RunCliSelfCheck` (Task 3 /
  existing), `lsxcommon::Core::Initialize/Shutdown`,
  `lsxcommon::version`, `lsxcommon::log_gpu_info`,
  `lsxcommon::FastMathMode`, `lsxcommon::GenerationCliArgs`.
- Produces: the `lsxbenchmark` executable and the completed
  `apps/lsxbenchmark/CMakeLists.txt` (executable block mirrors
  `apps/lsxcli/CMakeLists.txt`). The app numeric-mode flag value:
  `--bit-exact` maps to `FastMathMode::kOff`, otherwise `kAuto`; `--selfcheck`
  delegates to `RunCliSelfCheck` (Paris harness, internally forces
  `max(2, max_gen_tokens)`), bench delegates to `RunCliBenchmark`.

- [ ] **Step 1: Complete `apps/lsxbenchmark/CMakeLists.txt`** — replace the
  Task 1 stub with the full app target (exe + link + WIN32 DLL copies +
  install + tests subdir):

```cmake
add_executable(lsxbenchmark src/main.cpp src/cli.cpp)
target_link_libraries(lsxbenchmark PRIVATE lsxcommon arrow_shared arrow_cuda_shared simdjson::simdjson ${CMAKE_DL_LIBS})

if(WIN32)
    add_custom_command(TARGET lsxbenchmark POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:lsxbenchmark>"
    )
    add_custom_command(TARGET lsxbenchmark POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:lsxbenchmark>"
    )
endif()

install(TARGETS lsxbenchmark RUNTIME DESTINATION bin)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
```

- [ ] **Step 2: Create `apps/lsxbenchmark/src/main.cpp`**

```cpp
#include "lsxcommon/core.h"
#include <absl/strings/str_format.h>
#include "lsxcommon/common.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/log.h"
#include "cli.h"

int main(int argc, char *argv[]) {
    const lsxbenchmark::BenchmarkCliArgs cli =
        lsxbenchmark::ParseBenchmarkCliArgs(argc, argv);

    lsxcommon::Core::Initialize("lsxbenchmark_runtime.log");
    lsxcommon::version();
    lsxcommon::log_gpu_info();
    lsxcommon::log::info(absl::StrFormat("Model path: %s", cli.model_path));

    lsxcommon::runtime::GenerationCliArgs args;
    args.model_path = cli.model_path;
    args.prompt = cli.prompt;
    // Paris selfcheck contract: 1-token request (the harness forces the
    // 1->2-token minimum internally, legacy lsxfabric --selfcheck parity);
    // bench honors --max-gen (default 2048).
    args.max_gen_tokens = cli.selfcheck ? 1 : cli.max_gen;
    // Numeric policy: kAuto = per-GPU maximum-speed perf policy;
    // kOff = bit-exact deterministic scalar path (--bit-exact).
    args.fast_math_mode = cli.bit_exact ? lsxcommon::FastMathMode::kOff
                                        : lsxcommon::FastMathMode::kAuto;

    const int rc = cli.selfcheck
                       ? lsxcommon::runtime::RunCliSelfCheck(args)
                       : lsxcommon::runtime::RunCliBenchmark(args);
    lsxcommon::Core::Shutdown();
    return rc;
}
```

- [ ] **Step 3: Reconfigure, build the app target and re-run the parser suite**

```bash
cmake --preset linux-debug && cmake --build --preset linux-debug --target lsxbenchmark test_lsxbenchmark_cli && ctest --preset linux-debug -R "test_lsxbenchmark_cli" --output-on-failure
```
Expected: `lsxbenchmark` links cleanly; 5 parser cases PASS.

- [ ] **Step 4: Commit**

```bash
git add apps/lsxbenchmark/CMakeLists.txt apps/lsxbenchmark/src/main.cpp
git commit -m "feat: lsxbenchmark app (bench + selfcheck dispatcher)"
```

---

### Task 5: Prune lsxfabric — measurement flags move out

**Files:**
- Rewrite: `apps/lsxfabric/tests/test_fabric_cli.cpp`
- Modify: `apps/lsxfabric/src/cli.h`
- Modify: `apps/lsxfabric/src/cli.cpp`
- Modify: `apps/lsxfabric/src/main.cpp`

**Interfaces:**
- Consumes: existing `lsxcommon::runtime::RunCliGeneration`,
  `lsxfabric::RunValidator`.
- Produces: `lsxfabric::FabricCliArgs` reduced to `model_path`, `prompt`,
  `max_gen`, `validate`, `testprompt`. `--benchmark` / `--selfcheck` /
  `--bit-exact` become unrecognized (silently ignored, same policy as the
  already-removed `--fast-math`/`--calibrate`). Note: the rewritten fabric
  test cannot observe removed fields (they no longer exist), so it is a
  regression guard that must stay green through this task; the RED driver of
  the feature was Task 1.

- [ ] **Step 1: Rewrite `apps/lsxfabric/tests/test_fabric_cli.cpp`**

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

TEST_CASE("fabric cli: defaults are maximum-speed generation", "[lsxfabric][cli]") {
    const lsxfabric::FabricCliArgs a = Parse({"lsxfabric"});
    REQUIRE_FALSE(a.validate);
    REQUIRE_FALSE(a.testprompt);
    REQUIRE(a.max_gen == 64);
    REQUIRE(a.prompt == "The capital of France is");
    REQUIRE(a.model_path.find("DeepSeek-V4-Flash") != std::string::npos);
}

TEST_CASE("fabric cli: measurement flags moved to lsxbenchmark are not recognized",
          "[lsxfabric][cli]") {
    // Silently ignored, same policy as the removed --fast-math/--calibrate.
    const lsxfabric::FabricCliArgs a = Parse(
        {"lsxfabric", "--benchmark", "--selfcheck", "--bit-exact",
         "--fast-math", "--no-fast-math", "--calibrate"});
    REQUIRE_FALSE(a.validate);
    REQUIRE_FALSE(a.testprompt);
    REQUIRE(a.max_gen == 64);
    REQUIRE(a.prompt == "The capital of France is");
}

TEST_CASE("fabric cli: kept flags parse", "[lsxfabric][cli]") {
    const lsxfabric::FabricCliArgs a = Parse(
        {"lsxfabric", "--localmodel", "/tmp/model_x", "--prompt", "hi",
         "--max-gen", "7", "--validate", "--testprompt"});
    REQUIRE(a.model_path == "/tmp/model_x");
    REQUIRE(a.prompt == "hi");
    REQUIRE(a.max_gen == 7);
    REQUIRE(a.validate);
    REQUIRE(a.testprompt);
}
```

- [ ] **Step 2: Rewrite `apps/lsxfabric/src/cli.h`**

```cpp
#pragma once

#include <string>

namespace lsxfabric {

// lsxfabric launch parameters. The engine always runs the per-GPU
// maximum-speed auto perf policy. Speed measurement and bit-exactness
// verification live in lsxbenchmark; this surface is validate + generate.
struct FabricCliArgs {
    std::string model_path;
    std::string prompt = "The capital of France is";
    int max_gen = 64;
    bool validate = false;
    bool testprompt = false;
};

// Parses argv (argv[0] is skipped). Unknown tokens — including the
// benchmark-era --benchmark/--selfcheck/--bit-exact and the removed
// --fast-math/--no-fast-math/--calibrate — are silently ignored.
FabricCliArgs ParseFabricCliArgs(int argc, char *argv[]);

} // namespace lsxfabric
```

- [ ] **Step 3: Rewrite `apps/lsxfabric/src/cli.cpp`**

```cpp
#include "cli.h"

#include "lsxcommon/path_registry.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>

namespace lsxfabric {

FabricCliArgs ParseFabricCliArgs(int argc, char *argv[]) {
    FabricCliArgs a;
    a.model_path = (std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
                    "DeepSeek-V4-Flash")
                       .string();
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--localmodel") == 0 && i + 1 < argc)
            a.model_path = argv[++i];
        else if (std::strcmp(argv[i], "--validate") == 0)
            a.validate = true;
        else if (std::strcmp(argv[i], "--testprompt") == 0)
            a.testprompt = true;
        else if (std::strcmp(argv[i], "--max-gen") == 0 && i + 1 < argc)
            a.max_gen = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--prompt") == 0 && i + 1 < argc)
            a.prompt = argv[++i];
    }
    return a;
}

} // namespace lsxfabric
```

- [ ] **Step 4: Rewrite `apps/lsxfabric/src/main.cpp`** (benchmark/selfcheck
  branches, `benchmark_gpu.h` include and the `fm` variable are gone;
  `--testprompt` always runs at `kAuto`)

```cpp
#include "lsxcommon/core.h"
#include <absl/strings/str_format.h>
#include "lsxcommon/common.h"
#include "lsxcommon/model_engine.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/log.h"
#include "cli.h"
#include "validator.h"

int main(int argc, char *argv[]) {
    const lsxfabric::FabricCliArgs cli =
        lsxfabric::ParseFabricCliArgs(argc, argv);

    lsxcommon::Core::Initialize("logestix_runtime.log");
    lsxcommon::version();
    lsxcommon::log_gpu_info();
    // Engine auto policy: resolve the per-GPU perf knobs (TC fast-math,
    // expert-cache sizing, dual-stream staging) for every launch mode.
    lsxcommon::InferenceEngine::Session engine_session;
    lsxcommon::log::info(absl::StrFormat("Model path: %s", cli.model_path));
    if (cli.testprompt) {
        lsxcommon::runtime::GenerationCliArgs args;
        args.model_path = cli.model_path;
        args.prompt = cli.prompt;
        args.max_gen_tokens = cli.max_gen;
        args.fast_math_mode = lsxcommon::FastMathMode::kAuto;
        int rc = lsxcommon::runtime::RunCliGeneration(args);
        lsxcommon::Core::Shutdown();
        return rc;
    }
    if (cli.validate) {
        lsxcommon::log::info("Running validation...");
        auto result = lsxfabric::RunValidator(cli.model_path);
        lsxcommon::Core::Shutdown();
        return result.ok ? 0 : 1;
    }
    lsxcommon::Core::Shutdown();
    return 0;
}
```

- [ ] **Step 5: Build and run the whole fabric test suite**

```bash
cmake --build --preset linux-debug --target lsxfabric test_fabric_cli test_lsxfabric && ctest --preset linux-debug -R "test_fabric_cli|test_lsxfabric" --output-on-failure
```
Expected: everything builds; all fabric test cases PASS (rewritten cli suite +
existing lifecycle/validator suite).

- [ ] **Step 6: Commit**

```bash
git add apps/lsxfabric/src/cli.h apps/lsxfabric/src/cli.cpp apps/lsxfabric/src/main.cpp apps/lsxfabric/tests/test_fabric_cli.cpp
git commit -m "refactor: lsxfabric drops measurement flags (move to lsxbenchmark)"
```

---

### Task 6: Delete the consumer-less IModel::Benchmark API

**Files:**
- Modify: `tests/test_model_runtime.cpp`
- Modify: `libs/lsxcommon/include/lsxcommon/model.h`
- Modify: `libs/lsxcommon/include/lsxcommon/model_base.h`
- Modify: `libs/lsxcommon/src/model_base.cpp`
- Modify: `libs/lsxcommon/include/lsxcommon/model_runtime.h`
- Modify: `libs/lsxcommon/src/model_runtime.cpp`
- Delete: `libs/lsxcommon/include/lsxcommon/benchmark_gpu.h`
- Delete: `libs/lsxcommon/src/benchmark_gpu.cpp`
- Delete: `libs/lsxcommon/src/benchmark_gpu.cu`
- Modify: `libs/lsxcommon/CMakeLists.txt`
- Modify: 8 model `.h`/`.cpp` pairs (exact edits below)

**Interfaces:**
- Consumes: nothing removed is referenced by remaining code after this task
  (verified by grep in Step 8).
- Produces: `IModel` without `Benchmark`; `gpu::BenchmarkResult`,
  `RunBenchmark`, `FillBenchmarkResult`, `RequireLoaded(BenchmarkResult&)`
  no longer exist anywhere. `RequireLoaded` callers are Benchmark-only
  (Infer uses `ValidateInferRequest`).

- [ ] **Step 1: Update `tests/test_model_runtime.cpp`**
  - Delete the `#include <lsxcommon/benchmark_gpu.h>` line (line 18).
  - Delete the now-unused `#include "kimi_moe_model.h"` and
    `#include "qwen_moe_model.h"` lines (lines 20-21) — their only consumers
    were the deleted test case from step below.
  - Delete these 5 TEST_CASE blocks entirely:
    1. `"model_runtime: IModel::Benchmark default returns ok=false"`
    2. `"model_runtime: RunBenchmark shim delegates to IModel::Benchmark"`
    3. `"FillBenchmarkResult computes tokens-per-second"`
    4. `"RequireLoaded: canonical Benchmark guard"` (`[model_base][lifecycle]`)
    5. `"Concrete unloaded models report 'model not loaded' from Benchmark"`
  - Keep the Task 3 `RunCliBenchmark` test case.

- [ ] **Step 2: `libs/lsxcommon/include/lsxcommon/model.h`** — delete the
  include line and the virtual:

```cpp
// DELETE this line:
#include "lsxcommon/benchmark_gpu.h"   // gpu::BenchmarkResult

// DELETE this member from class IModel (lines 78-80):
    // Runs the model's own engine loop on synthetic tokens. Additive default:
    // ok=false keeps test fakes compiling without an override.
    virtual gpu::BenchmarkResult Benchmark(int num_tokens = 2048) { return {}; }
```

- [ ] **Step 3: `libs/lsxcommon/include/lsxcommon/model_base.h`** — delete
  the declaration block (lines 77-80):

```cpp
// DELETE (comment + declaration):
    // Canonical Benchmark guard: false (with "model not loaded" + ok=false)
    // when the model is not loaded; true otherwise. Every Benchmark()
    // implementation MUST start with `if (!RequireLoaded(r)) return r;`.
    bool RequireLoaded(gpu::BenchmarkResult &r) const;
```

- [ ] **Step 4: `libs/lsxcommon/src/model_base.cpp`** — delete the
  implementation (lines 38-44):

```cpp
// DELETE:
bool ModelBase::RequireLoaded(gpu::BenchmarkResult &r) const
{
    if (loaded_) return true;
    r.ok = false;
    r.error_msg = "model not loaded";
    return false;
}
```

- [ ] **Step 5: `model_runtime.h` / `model_runtime.cpp`** — delete the
  declaration block in the header:

```cpp
// DELETE from include/lsxcommon/model_runtime.h:
// Shared Benchmark result tail: ok + tokens/timing/occupancy mapping.
void FillBenchmarkResult(gpu::BenchmarkResult &r, double total_ms,
                         int num_tokens, int num_layers,
                         std::uint64_t resident_bytes);
```

and the implementation in `src/model_runtime.cpp`:

```cpp
// DELETE:
void FillBenchmarkResult(gpu::BenchmarkResult &r, double total_ms,
                         int num_tokens, int num_layers,
                         std::uint64_t resident_bytes)
{
    r.ok = true;
    r.num_tokens = num_tokens;
    r.num_layers = num_layers;
    r.total_time_ms = total_ms;
    r.tokens_per_sec =
        total_ms > 0.0 ? (static_cast<double>(num_tokens) / total_ms) * 1000.0
                       : 0.0;
    r.resident_bytes = resident_bytes;
}
```

- [ ] **Step 6: Delete the three benchmark files and drop them from CMake**
  - `git rm libs/lsxcommon/include/lsxcommon/benchmark_gpu.h
    libs/lsxcommon/src/benchmark_gpu.cpp libs/lsxcommon/src/benchmark_gpu.cu`
  - In `libs/lsxcommon/CMakeLists.txt`: remove the `src/benchmark_gpu.cpp`
    line from the `add_library(lsxcommon STATIC ...)` list and the
    `src/benchmark_gpu.cu` line from `add_library(lsxcommon_cu OBJECT ...)`.

- [ ] **Step 7: Remove the 8 Benchmark overrides** (declaration from `.h`,
  whole function body from `.cpp`; bodies are exactly the
  `gpu::BenchmarkResult <Class>::Benchmark(int num_tokens) { ... }` blocks):
  - `src/glm_moe_model.h:26` + `src/glm_moe_model.cpp`
    (`GlmV5Model::Benchmark`)
  - `src/qwen_moe_model.h:42` + `src/qwen_moe_model.cpp`
    (`QwenMoeModelBase::Benchmark`)
  - `src/qwen_moe_exp_model.h:27` + `src/qwen_moe_exp_model.cpp`
    (`Qwen38FlashNextModel::Benchmark`)
  - `src/kimi_moe_model.h:26` + `src/kimi_moe_model.cpp`
    (`KimiK27CodeModel::Benchmark`)
  - `src/deepseek_moe_model.h:21` + `src/deepseek_moe_model.cpp`
    (`DeepSeekV4FlashModel::Benchmark`)
  - `src/glm_moe_flash_model.h:26` + `src/glm_moe_flash_model.cpp`
    (`Glm5NextFlashModel::Benchmark`)
  - `src/gemma_base_model.h:26` + `src/gemma_base_model.cpp`
    (`GemmaBaseModel::Benchmark`)
  - `include/lsxcommon/qwen_base_model.h:23` + `src/qwen_base_model.cpp`
    (`QwenBaseModel::Benchmark`)

- [ ] **Step 8: Verify no dangling references, build, test**

```bash
rg -n "BenchmarkResult|RunBenchmark|FillBenchmarkResult|RequireLoaded|IModel::Benchmark|::Benchmark\(" libs/ apps/ tests/ || true
cmake --preset linux-debug && cmake --build --preset linux-debug && ctest --preset linux-debug --output-on-failure
```
Expected: grep returns **no hits**; full fast suite builds and PASSES
(includes `test_model_suite_guard`, `test_model_independence_guard`,
`test_code_naming_guard`, `test_ctest_labels_guard`).

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "refactor: delete consumer-less IModel::Benchmark API (lsxbenchmark owns speed measurement)"
```

---

### Task 7: AGENTS.md + full verification sweep

**Files:**
- Modify: `AGENTS.md`

**Interfaces:**
- Consumes: the implemented state of Tasks 1-6.
- Produces: documentation matching the new CLI surface.

- [ ] **Step 1: Update the `benchmark_gpu` line in the lsxcommon structure
  block (AGENTS.md line ~262)** — replace:

```
  - `libs/lsxcommon/src/benchmark_gpu.cu` + `include/lsxcommon/benchmark_gpu.h` — `--benchmark` pipeline: NVFP4 ping-pong benchmark (init_hidden_states kernel, cuBLAS GEMM, dual-stream overlap, pinned staging), `lsxcommon::gpu::RunBenchmark()`, BenchmarkResult
```

with:

```
  - `libs/lsxcommon/src/model_runtime.cpp` — `runtime::RunCliBenchmark` (lsxbenchmark --bench): реальная генерация через IModel::Infer на args.fast_math_mode, отчёт tokens/t/s/ttft/resident
```

- [ ] **Step 2: Update the lsxfabric entry (AGENTS.md line ~264-266)** —
  replace:

```
- `apps/lsxfabric/` — B2B inference engine (`--localmodel`, `--validate`, `--benchmark`, `--testprompt`, `--selfcheck`, `--bit-exact`; CLI parse в `src/cli.{h,cpp}`)
```

with:

```
- `apps/lsxfabric/` — B2B inference engine, всегда максимальная скорость (`--localmodel`, `--validate`, `--testprompt`; CLI parse в `src/cli.{h,cpp}`); замер скорости и бит-точность переехали в lsxbenchmark
```

- [ ] **Step 3: Add the lsxbenchmark entry** right after the lsxcli block
  (AGENTS.md, after `apps/lsxcli/tests/` line):

```
- `apps/lsxbenchmark/` — замер скорости генерации + бит-точность (`--bench` — дефолтный режим, реальная генерация IModel::Infer; `--selfcheck` — Paris-проверка; `--bit-exact` — `FastMathMode::kOff` для активного режима; `--model`, `--prompt`, `--max-gen`; CLI parse в `src/cli.{h,cpp}`)
  - `apps/lsxbenchmark/tests/` — 5 test cases (CLI parser, label fast)
```

- [ ] **Step 4: Update the GPU Perf Auto-Tuning section (AGENTS.md
  line ~548-551)** — replace:

```
  нет: движок всегда гоняет per-GPU auto-политику; единственный численный флаг
  запуска — `--bit-exact` (lsxfabric), он маппится в `FastMathMode::kOff`
  модели.
```

with:

```
  нет: движок всегда гоняет per-GPU auto-политику; единственный численный флаг
  запуска — `--bit-exact` (lsxbenchmark), он маппится в `FastMathMode::kOff`
  модели.
```

- [ ] **Step 5: Full fast sweep + guards**

```bash
cmake --build --preset linux-debug && ctest --preset linux-debug --output-on-failure
```
Expected: all fast-labeled suites PASS (app suites, parser suites, guard
tests). Optionally dry-run the router: `cmake -DLSX_PRESET=linux-debug
-DLSX_LIST=ON -P tools/test_router.cmake`.

- [ ] **Step 6: Commit**

```bash
git add AGENTS.md
git commit -m "docs: lsxbenchmark owns speed + bit-exactness; lsxfabric max-speed surface"
```

---

## Verification Matrix (post-implementation)

| Concern | Command | Expected |
|---|---|---|
| Parser contracts | `ctest --preset linux-debug -R "test_lsxbenchmark_cli|test_fabric_cli"` | PASS |
| Runtime lifecycle | `ctest --preset linux-debug -R test_model_runtime` | PASS |
| Guard discipline | `ctest --preset linux-debug -R "guard|suite|naming"` | PASS |
| Full fast sweep | `ctest --preset linux-debug` | PASS |
| No dead API | `rg "BenchmarkResult|RunBenchmark" libs/ apps/ tests/` | no hits |
