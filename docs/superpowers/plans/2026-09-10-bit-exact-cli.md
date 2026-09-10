# Bit-Exact-Only CLI Numeric Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove all inference speed/performance flags from the lsxfabric launch parameters; keep a single `--bit-exact` flag forcing the deterministic scalar path for 100% answer verification.

**Architecture:** A pure testable CLI parser (`lsxfabric::ParseFabricCliArgs`) replaces ad-hoc parsing in `main()`. The lib-side plumbing (`GenerationCliArgs::fast_math` bool, `RunBenchmark` fast_math param, `RunCliCalibrate`) is deleted; the internal `FastMathMode` tri-state and `LsxFastMathScope` per-model policy stay untouched (model-independence invariant).

**Tech Stack:** C++20, CMake 3.25+ (CMakePresets, Ninja), Catch2 v3, abseil (printf-style `%`-formats only), spdlog.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-10-bit-exact-cli-design.md` — follow it verbatim.
- Do NOT touch: `FastMathMode` enum, `LsxFastMathScope`, `ModelInitConfig`, `gpu_perf.*`, any model file, `tests/test_gpu_perf.cpp`, `tests/test_model_runtime.cpp`, `tests/test_model_independence_guard.cpp`.
- New test: label `fast` in the same commit (guard `test_ctest_labels_guard`), TIMEOUT 120 s.
- `absl::StrFormat` — printf-style conversions only (`%s`/`%d`/`%u`/`%f`); no `{}` (see `tests/test_absl_format_guard.cpp`).
- All logging through spdlog wrappers (`lsxcommon::log`); no `std::cout`/`printf`.
- Build/verify on Linux: `cmake --preset linux-debug`, `cmake --build --preset linux-debug`, `ctest --preset linux-debug` (fast suite).
- English commit messages; prefixes `feat:`/`refactor:`/`docs:`/`test:`.

---

### Task 1: CLI parser with `--bit-exact` (TDD)

**Files:**
- Create: `apps/lsxfabric/src/cli.h`
- Create: `apps/lsxfabric/src/cli.cpp`
- Create: `apps/lsxfabric/tests/test_fabric_cli.cpp`
- Modify: `apps/lsxfabric/tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `lsxcommon::lsx_model_models_dir()` from `lsxcommon/path_registry.h` (already exists).
- Produces: `lsxfabric::FabricCliArgs { std::string model_path; std::string prompt = "The capital of France is"; int max_gen = 64; bool validate = false; bool benchmark = false; bool testprompt = false; bool selfcheck = false; bool bit_exact = false; }` and `lsxfabric::FabricCliArgs ParseFabricCliArgs(int argc, char *argv[])`. Task 2's main.cpp relies on exactly these names.

- [ ] **Step 1: Write the failing test** — create `apps/lsxfabric/tests/test_fabric_cli.cpp`:

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

TEST_CASE("fabric cli: defaults are auto-policy with no modes", "[lsxfabric][cli]") {
    const lsxfabric::FabricCliArgs a = Parse({"lsxfabric"});
    REQUIRE_FALSE(a.validate);
    REQUIRE_FALSE(a.benchmark);
    REQUIRE_FALSE(a.testprompt);
    REQUIRE_FALSE(a.selfcheck);
    REQUIRE_FALSE(a.bit_exact);
    REQUIRE(a.max_gen == 64);
    REQUIRE(a.prompt == "The capital of France is");
    REQUIRE(a.model_path.find("DeepSeek-V4-Flash") != std::string::npos);
}

TEST_CASE("fabric cli: --bit-exact is the single numeric-mode flag", "[lsxfabric][cli]") {
    const lsxfabric::FabricCliArgs a = Parse({"lsxfabric", "--bit-exact"});
    REQUIRE(a.bit_exact);
    REQUIRE_FALSE(a.validate);
    REQUIRE_FALSE(a.benchmark);
    REQUIRE_FALSE(a.testprompt);
    REQUIRE_FALSE(a.selfcheck);
}

TEST_CASE("fabric cli: removed speed/perf flags are not recognized", "[lsxfabric][cli]") {
    const lsxfabric::FabricCliArgs a = Parse(
        {"lsxfabric", "--fast-math", "--no-fast-math", "--calibrate"});
    REQUIRE_FALSE(a.bit_exact);
    REQUIRE_FALSE(a.benchmark);
    REQUIRE_FALSE(a.validate);
    REQUIRE_FALSE(a.testprompt);
    REQUIRE_FALSE(a.selfcheck);
}

TEST_CASE("fabric cli: kept flags parse", "[lsxfabric][cli]") {
    const lsxfabric::FabricCliArgs a = Parse(
        {"lsxfabric", "--localmodel", "/tmp/model_x", "--prompt", "hi",
         "--max-gen", "7", "--validate", "--benchmark", "--testprompt",
         "--selfcheck"});
    REQUIRE(a.model_path == "/tmp/model_x");
    REQUIRE(a.prompt == "hi");
    REQUIRE(a.max_gen == 7);
    REQUIRE(a.validate);
    REQUIRE(a.benchmark);
    REQUIRE(a.testprompt);
    REQUIRE(a.selfcheck);
}
```

- [ ] **Step 2: Register the test target** — append to `apps/lsxfabric/tests/CMakeLists.txt`:

```cmake
add_executable(test_fabric_cli test_fabric_cli.cpp ../src/cli.cpp)
target_include_directories(test_fabric_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(test_fabric_cli PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_fabric_cli POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_fabric_cli>"
    )
    add_custom_command(TARGET test_fabric_cli POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_fabric_cli>"
    )
endif()

add_test(NAME test_fabric_cli COMMAND test_fabric_cli)
set_tests_properties(test_fabric_cli PROPERTIES LABELS fast TIMEOUT 120)
```

- [ ] **Step 3: Run the test to verify it fails (red)**

Run: `cmake --build --preset linux-debug --target test_fabric_cli 2>&1 | tail -5`
Expected: compilation FAILS — fatal error: cli.h: No such file or directory.

- [ ] **Step 4: Implement the parser** — create `apps/lsxfabric/src/cli.h`:

```cpp
#pragma once

#include <string>

namespace lsxfabric {

// lsxfabric launch parameters. The engine always runs the per-GPU
// maximum-speed auto perf policy; the single numeric-mode switch is
// bit_exact (deterministic scalar path for 100% answer verification).
struct FabricCliArgs {
    std::string model_path;
    std::string prompt = "The capital of France is";
    int max_gen = 64;
    bool validate = false;
    bool benchmark = false;
    bool testprompt = false;
    bool selfcheck = false;
    bool bit_exact = false;
};

// Parses argv (argv[0] is skipped). Unknown tokens — including the removed
// --fast-math/--no-fast-math/--calibrate — are silently ignored.
FabricCliArgs ParseFabricCliArgs(int argc, char *argv[]);

} // namespace lsxfabric
```

Create `apps/lsxfabric/src/cli.cpp`:

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
        else if (std::strcmp(argv[i], "--benchmark") == 0)
            a.benchmark = true;
        else if (std::strcmp(argv[i], "--testprompt") == 0)
            a.testprompt = true;
        else if (std::strcmp(argv[i], "--selfcheck") == 0)
            a.selfcheck = true;
        else if (std::strcmp(argv[i], "--bit-exact") == 0)
            a.bit_exact = true;
        else if (std::strcmp(argv[i], "--max-gen") == 0 && i + 1 < argc)
            a.max_gen = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--prompt") == 0 && i + 1 < argc)
            a.prompt = argv[++i];
    }
    return a;
}

} // namespace lsxfabric
```

- [ ] **Step 5: Run the test to verify it passes (green)**

Run: `cmake --build --preset linux-debug --target test_fabric_cli && ctest --preset linux-debug -R test_fabric_cli --output-on-failure`
Expected: build OK; `1/1 Test #... test_fabric_cli ... Passed`.

- [ ] **Step 6: Commit**

```bash
git add apps/lsxfabric/src/cli.h apps/lsxfabric/src/cli.cpp apps/lsxfabric/tests/test_fabric_cli.cpp apps/lsxfabric/tests/CMakeLists.txt
git commit -m "feat: lsxfabric cli parser with single --bit-exact numeric flag"
```

---

### Task 2: Rewire main.cpp + slim lib plumbing (atomic)

**Files:**
- Modify: `apps/lsxfabric/src/main.cpp` (full rewrite of the flag plumbing)
- Modify: `libs/lsxcommon/include/lsxcommon/model_runtime.h:79-102`
- Modify: `libs/lsxcommon/src/model_runtime.cpp:180-208,224-260,316-369` (+ includes)
- Modify: `libs/lsxcommon/include/lsxcommon/benchmark_gpu.h:21-22`
- Modify: `libs/lsxcommon/src/benchmark_gpu.cpp:9-16`

**Interfaces:**
- Consumes: `lsxfabric::ParseFabricCliArgs` from Task 1.
- Produces: `GenerationCliArgs { model_path, prompt, max_gen_tokens, fast_math_mode = FastMathMode::kAuto }` (bool `fast_math` removed); `gpu::RunBenchmark(const std::string &model_path, int num_tokens = 2048)` (bool param removed); `RunCliCalibrate` deleted.

- [ ] **Step 1: Slim `GenerationCliArgs`** — in `libs/lsxcommon/include/lsxcommon/model_runtime.h` replace:

```cpp
struct GenerationCliArgs {
    std::string model_path;
    std::string prompt;
    int max_gen_tokens = 64;
    // Routes NVFP4 GEMMs through the native tensor-core path (sm_120a).
    // Tolerance-class numerics; off keeps the bit-exact scalar path.
    // Legacy opt-in: true forces kOn. Otherwise fast_math_mode decides
    // (kAuto = per-GPU perf policy from InitializeEngine).
    bool fast_math = false;
    FastMathMode fast_math_mode = FastMathMode::kAuto;
};
```

with:

```cpp
struct GenerationCliArgs {
    std::string model_path;
    std::string prompt;
    int max_gen_tokens = 64;
    // Numeric policy: kAuto = per-GPU maximum-speed perf policy;
    // kOff = bit-exact scalar path (--bit-exact, 100% verification).
    FastMathMode fast_math_mode = FastMathMode::kAuto;
};
```

and delete the lines:

```cpp
// Developer base-authoring run (GPU perf auto-tuning): A/B benchmarks the
// GEMM paths on the real model, prints the winning knobs and the C++ snippet
// to paste into the committed base table (gpu_perf.cpp BaseKnobs). Also
// persists the result to the vault perf cache. NOT an inference path.
int RunCliCalibrate(const std::string &model_path);
```

- [ ] **Step 2: Simplify `CreateCliModel` + CLI tails, delete `RunCliCalibrate`** — in `libs/lsxcommon/src/model_runtime.cpp`:

Replace `CreateCliModel` (lines 187-206) with:

```cpp
CliModel CreateCliModel(const std::string &model_path, FastMathMode mode)
{
    CliModel out;
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model_path;
    cfg.fast_math = false;
    cfg.fast_math_mode = mode;
    out.model = lsxcommon::LsxModelFactory::Create(cfg);
    if (!out.model || !out.model->Load()) {
        lsxcommon::log::error(
            absl::StrFormat("Failed to create/load model: %s", model_path));
        return out;
    }
    out.ok = true;
    return out;
}
```

In `RunCliGeneration` (line 228) and `RunCliSelfCheck` (line 257) replace:

```cpp
    CliModel m = CreateCliModel(args.model_path, args.fast_math
                                           ? FastMathMode::kOn
                                           : args.fast_math_mode);
```

with (both sites):

```cpp
    CliModel m = CreateCliModel(args.model_path, args.fast_math_mode);
```

Delete the entire `RunCliCalibrate` block (lines 316-369: the `// ── Developer calibration run (perf base-authoring) ──` comment through its closing brace).

Delete now-unused includes at the top of the file (keep `benchmark_gpu.h` — `FillBenchmarkResult` at line 166 uses `gpu::BenchmarkResult`):

```cpp
#include "lsxcommon/gpu_perf.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/perf_calibration.h"
```

- [ ] **Step 3: Drop the `RunBenchmark` fast_math param** — in `libs/lsxcommon/include/lsxcommon/benchmark_gpu.h` replace:

```cpp
BenchmarkResult RunBenchmark(const std::string &model_path, int num_tokens = 2048,
                             bool fast_math = false);
```

with:

```cpp
// Always follows the per-GPU maximum-speed auto perf policy.
BenchmarkResult RunBenchmark(const std::string &model_path,
                             int num_tokens = 2048);
```

In `libs/lsxcommon/src/benchmark_gpu.cpp` replace:

```cpp
BenchmarkResult RunBenchmark(const std::string &model_path, int num_tokens,
                             bool fast_math) {
    BenchmarkResult r;
    r.num_tokens = num_tokens;
    ModelInitConfig cfg;
    cfg.model_path = model_path;
    cfg.max_gen_tokens = num_tokens;
    cfg.fast_math = fast_math;
```

with:

```cpp
BenchmarkResult RunBenchmark(const std::string &model_path, int num_tokens) {
    BenchmarkResult r;
    r.num_tokens = num_tokens;
    ModelInitConfig cfg;
    cfg.model_path = model_path;
    cfg.max_gen_tokens = num_tokens;
```

- [ ] **Step 4: Rewire `apps/lsxfabric/src/main.cpp`** — full replacement:

```cpp
#include "lsxcommon/core.h"
#include <absl/strings/str_format.h>
#include "lsxcommon/common.h"
#include "lsxcommon/benchmark_gpu.h"
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
    lsxcommon::log::info(absl::StrFormat("Model path: %s", cli.model_path));
    const lsxcommon::FastMathMode fm = cli.bit_exact
                                           ? lsxcommon::FastMathMode::kOff
                                           : lsxcommon::FastMathMode::kAuto;
    if (cli.testprompt) {
        lsxcommon::runtime::GenerationCliArgs args;
        args.model_path = cli.model_path;
        args.prompt = cli.prompt;
        args.max_gen_tokens = cli.max_gen;
        args.fast_math_mode = fm;
        int rc = lsxcommon::runtime::RunCliGeneration(args);
        lsxcommon::Core::Shutdown();
        return rc;
    }
    if (cli.selfcheck) {
        lsxcommon::runtime::GenerationCliArgs args;
        args.model_path = cli.model_path;
        args.prompt = cli.prompt;
        args.max_gen_tokens = 1;
        args.fast_math_mode = fm;
        int rc = lsxcommon::runtime::RunCliSelfCheck(args);
        lsxcommon::Core::Shutdown();
        return rc;
    }
    if (cli.validate) {
        lsxcommon::log::info("Running validation...");
        auto result = lsxfabric::RunValidator(cli.model_path);
        lsxcommon::Core::Shutdown();
        return result.ok ? 0 : 1;
    }
    if (cli.benchmark) {
        lsxcommon::log::info("Running model benchmark...");
        auto result = lsxcommon::gpu::RunBenchmark(cli.model_path, 2048);
        if (!result.ok) {
            lsxcommon::log::error(absl::StrFormat("Benchmark failed: %s", result.error_msg));
        } else {
            lsxcommon::log::info("=== --benchmark Results ===");
            lsxcommon::log::info(absl::StrFormat("Tokens generated : %d", result.num_tokens));
            lsxcommon::log::info(absl::StrFormat("Layers processed : %d",
                         result.num_layers * result.num_tokens));
            lsxcommon::log::info(absl::StrFormat("Total time       : %.1f ms", result.total_time_ms));
            lsxcommon::log::info(absl::StrFormat("Avg per layer    : %.3f ms",
                         result.avg_bus_latency_ms));
            lsxcommon::log::info(absl::StrFormat("Tokens/sec       : %.1f t/s",
                         result.tokens_per_sec));
            lsxcommon::log::info(absl::StrFormat("DMA volume       : %d MiB",
                         static_cast<int>(result.total_bytes_transferred / (1024 * 1024))));
        }
        lsxcommon::Core::Shutdown();
        return result.ok ? 0 : 1;
    }
    lsxcommon::Core::Shutdown();
    return 0;
}
```

- [ ] **Step 5: Build everything and run the fast suite**

Run: `cmake --build --preset linux-debug && ctest --preset linux-debug --output-on-failure`
Expected: full build OK (lsxcommon, lsxfabric, all tests); fast suite green, including `test_fabric_cli`, `test_model_runtime`, `test_gpu_perf`, `test_lsxfabric`, `test_ctest_labels_guard`.

- [ ] **Step 6: Commit**

```bash
git add apps/lsxfabric/src/main.cpp libs/lsxcommon/include/lsxcommon/model_runtime.h libs/lsxcommon/src/model_runtime.cpp libs/lsxcommon/include/lsxcommon/benchmark_gpu.h libs/lsxcommon/src/benchmark_gpu.cpp
git commit -m "refactor: launch parameters keep only --bit-exact, drop speed/perf plumbing"
```

---

### Task 3: AGENTS.md sync + full verification

**Files:**
- Modify: `AGENTS.md`

**Interfaces:**
- Consumes: final CLI surface from Tasks 1-2.
- Produces: instruction file matching reality.

- [ ] **Step 1: Update AGENTS.md** — three edits:

(a) GPU Perf Auto-Tuning section — replace the line:

```
- Приоритет разрешения: env > CLI(kOn/kOff) > vault-кэш калибровки
  (`perf-cache/<gpu-key>.json`) > эвристика по паспорту > коммитимая база
  (`BaseKnobs` в `gpu_perf.cpp`). `LSX_PERF_AUTO=0` возвращает legacy-поведение
  (bit-exact дефолты). `LSX_TEST_FAST_MATH` форсирует TC (существующие тесты).
```

with:

```
- Приоритет разрешения: env > тестовый cli_mode(kOn/kOff в
  `ResolvePerfKnobs`, только тесты) > vault-кэш калибровки
  (`perf-cache/<gpu-key>.json`) > эвристика по паспорту > коммитимая база
  (`BaseKnobs` в `gpu_perf.cpp`). `LSX_PERF_AUTO=0` возвращает legacy-поведение
  (bit-exact дефолты). `LSX_TEST_FAST_MATH` форсирует TC (существующие тесты).
  CLI speed/perf-флагов нет: движок всегда гоняет per-GPU auto-политику;
  единственный численный флаг запуска — `--bit-exact` (lsxfabric), он
  маппится в `FastMathMode::kOff` модели.
```

(b) GPU Perf Auto-Tuning section — replace:

```
Записи коммитимой базы авторятся через `lsxfabric --calibrate <model>`
(реальные веса, A/B t/s, печатает сниппет для `BaseKnobs`).
```

with:

```
`RunCliCalibrate`/`lsxfabric --calibrate` удалены (2026-09, бит-точная
политика CLI): записи коммитимой базы авторятся вручную из сниппетов
авто-калибровки неизвестных GPU (vault `perf-cache/<gpu-key>.json`).
```

(c) Structure section — replace:

```
- `apps/lsxfabric/` — B2B inference engine (`--localmodel`, `--validate`, `--benchmark`, `--testprompt`, `--selfcheck`)
```

with:

```
- `apps/lsxfabric/` — B2B inference engine (`--localmodel`, `--validate`, `--benchmark`, `--testprompt`, `--selfcheck`, `--bit-exact`; CLI parse в `src/cli.{h,cpp}`)
```

- [ ] **Step 2: Full fast-suite verification**

Run: `cmake --build --preset linux-debug && ctest --preset linux-debug --output-on-failure`
Expected: 100% green (fast label; includes `test_absl_format_guard`, `test_code_naming_guard`, `test_ctest_labels_guard`).

- [ ] **Step 3: Commit**

```bash
git add AGENTS.md
git commit -m "docs: bit-exact-only launch policy in AGENTS.md"
```
