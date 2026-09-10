# Test Categories (fast / inference / perf) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans
> (subagent-driven development ЗАПРЕЩЁН в этом репо — см. AGENTS.md,
> Autonomous Mode). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Разделить 125 ctest-тестов на категории `fast` / `inference` / `perf`
(CTest LABELS), сделать базовый `ctest --preset` fast-only, добавить
label-пресеты, git-diff-роутер и guard-тест, унифицировать имена фикстур.

**Architecture:** CTest LABELS через три канонических блока
`set_tests_properties(... PROPERTIES LABELS <cat>)` в конце
`tests/CMakeLists.txt`; 15 новых + 5 изменённых testPresets в
`CMakePresets.json`; `tools/test_router.cmake` (script mode, `-D`-флаги)
классифицирует `git diff` и запускает `ctest -L`; дисциплину проверяет
host-only guard `tests/test_ctest_labels_guard.cpp` (TDD: каждая проверка
сначала RED).

**Tech Stack:** CMake 3.25+ (CTest LABELS, testPresets, script mode), Catch2
v3.16 (guard), gcc/MSVC без изменений.

**Spec:** `docs/superpowers/specs/2026-09-10-test-categories-design.md`

## Global Constraints

- Ровно один label на тест: `fast` | `inference` | `perf`; никаких файлов
  в `tests/` не переносим и не переименовываем, кроме 7 фикстур (Task 8).
- `perf` никогда не входит в автопрогон (роутер умеет только fast+inference).
- Существующие свойства тестов (`SKIP_RETURN_CODE 4`, `ENVIRONMENT`) не менять.
- Windows-ограничение inference-прогона: `-R "^(test_qwen|test_gemma)"`
  (применяет роутер, не пресеты).
- Коммиты: английский, префиксы `build:`/`feat:`/`refactor:`/`test:`/`docs:`.
- Рабочие команды: `cmake --preset linux-debug`,
  `cmake --build --preset linux-debug --target <t>`,
  `ctest --preset linux-debug ...`.

---

### Task 1: Guard-тест label-дисциплины (RED)

**Files:**
- Create: `tests/test_ctest_labels_guard.cpp`
- Modify: `tests/CMakeLists.txt` (в конец файла, после блока
  `test_model_independence_guard`)

**Interfaces:**
- Consumes: ничего (host-only, только Catch2 + std).
- Produces: `tests/test_ctest_labels_guard.cpp`, ctest-имя
  `test_ctest_labels_guard`; чистые функции `RegisteredTestNames`,
  `ParseLabelCalls`, `CheckLabelCoverage`, `CheckPerfSkip` (переиспользуются
  в Tasks 3/5/7 через добавление новых функций/TEST_CASE).

- [ ] **Step 1: Write the failing test**

Создать `tests/test_ctest_labels_guard.cpp` целиком:

```cpp
// CTest category-label discipline guard (host-only, zero deps beyond Catch2).
// Categories: fast | inference | perf. Spec:
// docs/superpowers/specs/2026-09-10-test-categories-design.md
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#ifndef LSX_SOURCE_DIR
#define LSX_SOURCE_DIR "."
#endif

namespace {

const std::set<std::string>& Categories()
{
    static const std::set<std::string> v = {"fast", "inference", "perf"};
    return v;
}

std::string ReadFileOrThrow(const fs::path& p)
{
    std::ifstream in(p);
    if (!in.is_open())
        throw std::runtime_error("cannot read " + p.string());
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::vector<std::string> RegisteredTestNames(const std::string& text)
{
    std::vector<std::string> names;
    static const std::regex add_re(
        R"(add_test\s*\(\s*NAME\s+([A-Za-z0-9_]+))");
    // foreach(<var> <items...>): every item is a registered test name.
    static const std::regex fe_re(
        R"(foreach\s*\(\s*\w+((?:\s+[A-Za-z0-9_]+)+)\s*\))");
    std::smatch m;
    auto begin = text.cbegin();
    while (std::regex_search(begin, text.cend(), m, add_re))
    {
        names.push_back(m[1]);
        begin = m.suffix().first;
    }
    begin = text.cbegin();
    while (std::regex_search(begin, text.cend(), m, fe_re))
    {
        std::istringstream is(m[1].str());
        std::string tok;
        while (is >> tok)
            names.push_back(tok);
        begin = m.suffix().first;
    }
    return names;
}

struct LabelCall
{
    std::vector<std::string> names;
    std::string label;
};

// Recognizes set_tests_properties(<names...> PROPERTIES LABELS <cat>) calls,
// single- or multi-line (parens body scan up to the first ')').
std::vector<LabelCall> ParseLabelCalls(const std::string& text)
{
    std::vector<LabelCall> calls;
    static const std::regex open_re(R"(set_tests_properties\s*\()");
    static const std::regex label_re(R"(PROPERTIES\s+LABELS\s+(\w+)\s*$)");
    static const std::regex props_re(R"(PROPERTIES\s+LABELS)");
    std::smatch m;
    auto begin = text.cbegin();
    while (std::regex_search(begin, text.cend(), m, open_re))
    {
        const auto body_begin = m.suffix().first;
        const auto body_end = std::find(body_begin, text.cend(), ')');
        const std::string body(body_begin, body_end);
        std::smatch lm;
        if (std::regex_search(body, lm, label_re))
        {
            LabelCall c;
            c.label = lm[1];
            std::smatch pm;
            std::regex_search(body, pm, props_re);
            std::istringstream is(body.substr(0, pm.position()));
            std::string tok;
            while (is >> tok)
                c.names.push_back(tok);
            calls.push_back(std::move(c));
        }
        begin = (body_end == text.cend()) ? body_end : body_end + 1;
    }
    return calls;
}

std::vector<std::string> CheckLabelCoverage(
    const std::vector<std::string>& tests, const std::vector<LabelCall>& calls)
{
    std::vector<std::string> v;
    std::map<std::string, std::string> label_of;
    for (const auto& c : calls)
    {
        if (!Categories().count(c.label))
            v.push_back("unknown label: " + c.label);
        for (const auto& n : c.names)
        {
            const auto it = label_of.find(n);
            if (it != label_of.end())
                v.push_back("duplicate label for " + n + " (" + it->second +
                            " and " + c.label + ")");
            else
                label_of.emplace(n, c.label);
        }
    }
    for (const auto& t : tests)
        if (!label_of.count(t))
            v.push_back("unlabeled test: " + t);
    for (const auto& [n, l] : label_of)
        if (std::find(tests.begin(), tests.end(), n) == tests.end())
            v.push_back("label for unknown test: " + n);
    return v;
}

std::vector<std::string> CheckPerfSkip(const std::vector<LabelCall>& calls,
                                       const std::string& text)
{
    std::vector<std::string> v;
    static const std::regex re(
        R"(set_tests_properties\s*\(([^)]*?)SKIP_RETURN_CODE\s+4)");
    for (const auto& c : calls)
    {
        if (c.label != "perf")
            continue;
        for (const auto& n : c.names)
        {
            bool found = false;
            std::smatch m;
            auto begin = text.cbegin();
            while (std::regex_search(begin, text.cend(), m, re))
            {
                std::istringstream is(m[1].str());
                std::string tok;
                while (is >> tok)
                    if (tok == n)
                    {
                        found = true;
                        break;
                    }
                if (found)
                    break;
                begin = m.suffix().first;
            }
            if (!found)
                v.push_back("perf test without SKIP_RETURN_CODE 4: " + n);
        }
    }
    return v;
}

} // namespace

TEST_CASE("ctest labels guard parser: synthetic fixtures",
          "[labels][guard][unit]")
{
    SECTION("unlabeled test is a violation")
    {
        CHECK(!CheckLabelCoverage({"t1"}, {}).empty());
    }
    SECTION("duplicate label is a violation")
    {
        CHECK(!CheckLabelCoverage(
                  {"t1"}, {{{"t1"}, "fast"}, {{"t1"}, "inference"}})
                  .empty());
    }
    SECTION("unknown label is a violation")
    {
        CHECK(!CheckLabelCoverage({"t1"}, {{{"t1"}, "perf2"}}).empty());
    }
    SECTION("phantom label (unknown test) is a violation")
    {
        CHECK(!CheckLabelCoverage({}, {{{"ghost"}, "fast"}}).empty());
    }
    SECTION("clean labeling passes")
    {
        CHECK(CheckLabelCoverage(
                  {"t1", "t2"}, {{{"t1"}, "fast"}, {{"t2"}, "perf"}})
                  .empty());
    }
    SECTION("multi-line label call parses")
    {
        const std::string cm = R"(set_tests_properties(
    t1
    t2
    PROPERTIES LABELS fast)
set_tests_properties(t3 PROPERTIES LABELS perf)
set_tests_properties(t3 PROPERTIES SKIP_RETURN_CODE 4
    ))";
        const auto calls = ParseLabelCalls(cm);
        REQUIRE(calls.size() == 2);
        CHECK(calls[0].label == "fast");
        CHECK(calls[0].names == std::vector<std::string>{"t1", "t2"});
        CHECK(calls[1].label == "perf");
        CHECK(CheckLabelCoverage({"t1", "t2", "t3"}, calls).empty());
        CHECK(CheckPerfSkip(calls, cm).empty());
    }
}

TEST_CASE("ctest labels guard: every test labeled exactly once",
          "[labels][guard]")
{
    const fs::path root(LSX_SOURCE_DIR);
    const std::string cmake_text =
        ReadFileOrThrow(root / "tests" / "CMakeLists.txt");
    const auto tests = RegisteredTestNames(cmake_text);
    INFO("registered tests: " << tests.size());
    REQUIRE(tests.size() >= 126);
    const auto calls = ParseLabelCalls(cmake_text);
    auto v = CheckLabelCoverage(tests, calls);
    std::string joined;
    for (const auto& s : v)
        joined += s + "\n";
    INFO("label violations:\n" << joined);
    REQUIRE(v.empty());
}

TEST_CASE("ctest labels guard: perf tests declare SKIP_RETURN_CODE 4",
          "[labels][guard]")
{
    const fs::path root(LSX_SOURCE_DIR);
    const std::string cmake_text =
        ReadFileOrThrow(root / "tests" / "CMakeLists.txt");
    const auto calls = ParseLabelCalls(cmake_text);
    auto v = CheckPerfSkip(calls, cmake_text);
    std::string joined;
    for (const auto& s : v)
        joined += s + "\n";
    INFO("perf skip violations:\n" << joined);
    REQUIRE(v.empty());
}
```

Зарегистрировать таргет в конце `tests/CMakeLists.txt` (после блока
`test_model_independence_guard`):

```cmake
# Guard: ctest category-label discipline (fast | inference | perf).
# Host-only static scan of tests/CMakeLists.txt (see spec 2026-09-10).
add_executable(test_ctest_labels_guard test_ctest_labels_guard.cpp)
target_link_libraries(test_ctest_labels_guard PRIVATE Catch2::Catch2WithMain)
file(TO_CMAKE_PATH "${CMAKE_SOURCE_DIR}" LSX_SRC_DIR_FWD_LABELS)
target_compile_definitions(test_ctest_labels_guard
    PRIVATE "LSX_SOURCE_DIR=\"${LSX_SRC_DIR_FWD_LABELS}\"")
add_test(NAME test_ctest_labels_guard COMMAND test_ctest_labels_guard)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cmake --build --preset linux-debug --target test_ctest_labels_guard
ctest --preset linux-debug -R test_ctest_labels_guard --output-on-failure
```

Expected: сборка успешна; тест FAIL на
`REQUIRE(v.empty())` с десятками строк `unlabeled test: ...` (весь файл ещё
без LABELS). Синтетический TEST_CASE `parser: synthetic fixtures` при этом
зелёный — он проверяет чистые функции.

- [ ] **Step 3: (RED зафиксирован, реализация в Task 2)**

- [ ] **Step 4: Commit**

```bash
git add tests/test_ctest_labels_guard.cpp tests/CMakeLists.txt
git commit -m "test: ctest category-label guard (RED baseline)"
```

---

### Task 2: Label-блоки fast/inference/perf (GREEN для Task 1)

**Files:**
- Modify: `tests/CMakeLists.txt` (три блока в самый конец файла)

**Interfaces:**
- Consumes: guard из Task 1 (проверяет `PROPERTIES LABELS`).
- Produces: каноническая разметка всех 126 тестов; список имён используется
  в Task 4 (пресеты) и Task 6 (роутер).

- [ ] **Step 1: Append label blocks**

Добавить в самый конец `tests/CMakeLists.txt`:

```cmake
# ── Test category labels (fast | inference | perf) ────────────────────────
# fast     — host-only: runs on every default ctest invocation.
# inference — CUDA/model contour: runs when the inference contour changes
#             (tools/test_router.cmake).
# perf     — real-weight perf baselines: manual command only (ctest -L perf).
# Discipline guard: tests/test_ctest_labels_guard.cpp
set_tests_properties(
    test_core
    test_fast_load
    test_geometry_guard
    test_gpu_perf
    test_schema_views
    test_deepseek_moe_loader
    test_expert_store
    test_qwen_dense_convert
    test_glm_moe_expert_cache
    test_model_integrity
    test_qwen_dense_converter_integrity
    test_glm_moe_convert
    test_glm_moe_flash_convert
    test_kimi_moe_convert
    test_model_factory
    test_generation_cli
    test_glm_moe_prompt_framing
    test_gemma_prompt_framing
    test_gemma_moe_convert
    test_model_runtime
    test_model_engine
    test_tokenizer
    test_mosaic_network
    test_mosaic_frontend_route
    test_token_translator
    test_network_translator_weld
    test_integrated_pipeline
    test_path_registry
    test_vram_estimate
    test_resource_governor
    test_hardware_profile
    test_lifecycle_plan
    test_runtime_tuner
    test_kimi_moe_tokenizer
    test_code_naming_guard
    test_absl_format_guard
    test_kimi_moe_expert_cache
    test_qwen_moe_convert
    test_qwen_moe_36_convert
    test_qwen_moe_exp_convert
    test_model_independence_guard
    test_ctest_labels_guard
    PROPERTIES LABELS fast)

set_tests_properties(
    test_sampling
    test_lsx_nvfp4_dispatch
    test_perf_knobs_apply
    test_blackwell_tc_mma
    test_lsx_kernels
    test_blackwell_micro_gemm
    test_nvfp4_pack
    test_deepseek_moe_inference_kernels
    test_deepseek_moe_moe
    test_deepseek_moe_arena
    test_qwen_moe_kernels
    test_qwen_moe_exp_kernels
    test_glm_moe_flash_kernels
    test_kimi_moe_int4_dequant
    test_kimi_moe_int4_real
    test_kimi_moe_router
    test_kimi_moe_attention
    test_kimi_moe_experts
    test_kimi_moe_tc_batch
    test_glm_moe_load_fast
    test_deepseek_moe_load_fast
    test_glm_moe_model_lifecycle
    test_qwen_dense_loader
    test_qwen_dense_model
    test_qwen_dense_27b_model
    test_qwen_dense_kv_cache
    test_qwen_dense_kv_parity
    test_qwen_dense_selfcheck
    test_qwen_dense_27b_selfcheck
    test_qwen_dense_norm
    test_qwen_dense_rotary
    test_qwen_dense_shuffle_reduce
    test_qwen_dense_mlp
    test_qwen_dense_nvfp4
    test_qwen_nvfp4_tile_gemm
    test_qwen_nvfp4_gemv
    test_qwen_dense_full_attn
    test_qwen_dense_full_attn_parallel
    test_qwen_dense_linear_attn
    test_qwen_dense_fused_mlp
    test_qwen_dense_fused_linear_attn
    test_qwen_moe_loader
    test_qwen_moe_model
    test_qwen_moe_selfcheck
    test_qwen_moe_36_model
    test_qwen_moe_36_selfcheck
    test_qwen_moe_exp_loader
    test_qwen_moe_exp_model
    test_qwen_moe_exp_selfcheck
    test_glm_moe_loader
    test_glm_moe_model
    test_glm_moe_selfcheck
    test_glm_moe_inference
    test_glm_moe_full_compute
    test_glm_moe_prefill_seqp
    test_glm_moe_prompt_output
    test_glm_moe_attention
    test_glm_moe_dense_compute
    test_glm_moe_layer_compute
    test_glm_moe_experts
    test_glm_moe_moe_router
    test_glm_moe_moe_compute
    test_glm_moe_flash_loader
    test_glm_moe_flash_model
    test_glm_moe_flash_selfcheck
    test_deepseek_moe_prompt_output
    test_deepseek_moe_selfcheck
    test_gemma_moe_loader
    test_gemma_moe_model
    test_gemma_moe_selfcheck
    test_gemma_moe_norm
    test_gemma_moe_rotary
    test_gemma_moe_attn
    test_gemma_moe_moe
    test_kimi_moe_loader
    test_kimi_moe_model
    test_kimi_moe_selfcheck
    test_kimi_moe_inference
    PROPERTIES LABELS inference)

set_tests_properties(
    test_blackwell_perf_glm
    test_blackwell_perf_dense
    test_blackwell_perf_moe_qwen
    test_blackwell_perf_moe_gemma
    test_blackwell_perf_deepseek
    test_kimi_moe_perf
    PROPERTIES LABELS perf)
```

- [ ] **Step 2: Run guard to verify it passes**

```bash
cmake --build --preset linux-debug --target test_ctest_labels_guard
ctest --preset linux-debug -R test_ctest_labels_guard --output-on-failure
```

Expected: PASS. Если guard ругается `unlabeled test: <name>` — имя из
foreach-цикла пропущено в списках; `label for unknown test` — опечатка.
Сверка количества: `ctest --preset linux-debug -N` должен показывать 126
тестов; `42 fast + 78 inference + 6 perf`.

- [ ] **Step 3: Cross-check label counts**

```bash
ctest --preset linux-debug -L "^fast$" -N | tail -2
ctest --preset linux-debug -L "^inference$" -N | tail -2
ctest --preset linux-debug -L "^perf$" -N | tail -2
```

Expected: 42 / 78 / 6 (строка «Total Tests: N»).

- [ ] **Step 4: Commit**

```bash
git add tests/CMakeLists.txt
git commit -m "test: label all 126 ctest tests (fast/inference/perf)"
```

---

### Task 3: Guard-проверка пресетов (RED)

**Files:**
- Modify: `tests/test_ctest_labels_guard.cpp`

**Interfaces:**
- Consumes: `Categories()`, `ReadFileOrThrow` (Task 1).
- Produces: функция `CheckPresets` (используется только guard'ом).

- [ ] **Step 1: Add the failing check**

В анонимный namespace `tests/test_ctest_labels_guard.cpp` добавить (после
`CheckPerfSkip`):

```cpp
std::vector<std::string> CheckPresets(const std::string& json_text)
{
    static const std::vector<std::string> kBase = {
        "windows-debug", "windows-release", "linux-debug",
        "linux-release", "linux-release-znver4"};
    static const std::vector<std::string> kSuffix = {
        "fast", "inference", "perf"};
    std::vector<std::string> v;
    static const std::regex label_re(R"("label"\s*:\s*"([^"]+)")");
    std::smatch m;
    auto begin = json_text.cbegin();
    while (std::regex_search(begin, json_text.cend(), m, label_re))
    {
        std::string inner = m[1].str();
        if (!inner.empty() && inner.front() == '^')
            inner.erase(0, 1);
        if (!inner.empty() && inner.back() == '$')
            inner.pop_back();
        if (!Categories().count(inner))
            v.push_back("CMakePresets.json: invalid label filter: " +
                        m[1].str());
        begin = m.suffix().first;
    }
    for (const auto& b : kBase)
        for (const auto& s : kSuffix)
            if (json_text.find("\"" + b + "-" + s + "\"") ==
                std::string::npos)
                v.push_back("CMakePresets.json: missing testPreset " + b +
                            "-" + s);
    return v;
}
```

И новый TEST_CASE (после perf-теста):

```cpp
TEST_CASE("ctest labels guard: presets reference valid labels",
          "[labels][guard]")
{
    const fs::path root(LSX_SOURCE_DIR);
    const std::string json_text =
        ReadFileOrThrow(root / "CMakePresets.json");
    auto v = CheckPresets(json_text);
    std::string joined;
    for (const auto& s : v)
        joined += s + "\n";
    INFO("preset violations:\n" << joined);
    REQUIRE(v.empty());
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cmake --build --preset linux-debug --target test_ctest_labels_guard
ctest --preset linux-debug -R test_ctest_labels_guard --output-on-failure
```

Expected: FAIL на `missing testPreset <base>-fast/inference/perf` (15 строк).

- [ ] **Step 3: Commit (RED)**

```bash
git add tests/test_ctest_labels_guard.cpp
git commit -m "test: preset label-filter guard (RED)"
```

---

### Task 4: testPresets fast/inference/perf (GREEN для Task 3)

**Files:**
- Modify: `CMakePresets.json` (массив `testPresets` целиком)

**Interfaces:**
- Consumes: labels из Task 2; `CheckPresets` из Task 3.
- Produces: 20 testPresets — 5 базовых (теперь fast-only) + по 3 варианта
  на каждый configurePreset.

- [ ] **Step 1: Replace the testPresets array**

Заменить массив `"testPresets"` в `CMakePresets.json` на:

```json
  "testPresets": [
    { "name": "windows-debug",   "configurePreset": "windows-debug",   "configuration": "Debug",   "output": { "outputOnFailure": true }, "filters": [ { "include": { "label": "^fast$" } } ] },
    { "name": "windows-debug-inference",   "configurePreset": "windows-debug",   "configuration": "Debug",   "output": { "outputOnFailure": true }, "filters": [ { "include": { "label": "^inference$" } } ] },
    { "name": "windows-debug-perf",   "configurePreset": "windows-debug",   "configuration": "Debug",   "output": { "outputOnFailure": true }, "filters": [ { "include": { "label": "^perf$" } } ] },
    { "name": "windows-debug-fast",   "configurePreset": "windows-debug",   "configuration": "Debug",   "output": { "outputOnFailure": true }, "filters": [ { "include": { "label": "^fast$" } } ] },
    { "name": "windows-release",   "configurePreset": "windows-release",   "configuration": "Release",   "output": { "outputOnFailure": true }, "filters": [ { "include": { "label": "^fast$" } } ] },
    { "name": "windows-release-inference",   "configurePreset": "windows-release",   "configuration": "Release",   "output": { "outputOnFailure": true }, "filters": [ { "include": { "label": "^inference$" } } ] },
    { "name": "windows-release-perf",   "configurePreset": "windows-release",   "configuration": "Release",   "output": { "outputOnFailure": true }, "filters": [ { "include": { "label": "^perf$" } } ] },
    { "name": "windows-release-fast",   "configurePreset": "windows-release",   "configuration": "Release",   "output": { "outputOnFailure": true }, "filters": [ { "include": { "label": "^fast$" } } ] },
    { "name": "linux-debug",   "configurePreset": "linux-debug",   "output": { "outputOnFailure": true }, "filters": [ { "include": { "label": "^fast$" } } ] },
    { "name": "linux-debug-inference",   "configurePreset": "linux-debug",   "output": { "outputOnFailure": true }, "filters": [ { "include": { "label": "^inference$" } } ] },
    { "name": "linux-debug-perf",   "configurePreset": "linux-debug",   "output": { "outputOnFailure": true }, "filters": [ { "include": { "label": "^perf$" } } ] },
    { "name": "linux-debug-fast",   "configurePreset": "linux-debug",   "output": { "outputOnFailure": true }, "filters": [ { "include": { "label": "^fast$" } } ] },
    { "name": "linux-release",   "configurePreset": "linux-release",   "output": { "outputOnFailure": true }, "filters": [ { "include": { "label": "^fast$" } } ] },
    { "name": "linux-release-inference",   "configurePreset": "linux-release",   "output": { "outputOnFailure": true }, "filters": [ { "include": { "label": "^inference$" } } ] },
    { "name": "linux-release-perf",   "configurePreset": "linux-release",   "output": { "outputOnFailure": true }, "filters": [ { "include": { "label": "^perf$" } } ] },
    { "name": "linux-release-fast",   "configurePreset": "linux-release",   "output": { "outputOnFailure": true }, "filters": [ { "include": { "label": "^fast$" } } ] },
    { "name": "linux-release-znver4",   "configurePreset": "linux-release-znver4",   "output": { "outputOnFailure": true }, "filters": [ { "include": { "label": "^fast$" } } ] },
    { "name": "linux-release-znver4-inference",   "configurePreset": "linux-release-znver4",   "output": { "outputOnFailure": true }, "filters": [ { "include": { "label": "^inference$" } } ] },
    { "name": "linux-release-znver4-perf",   "configurePreset": "linux-release-znver4",   "output": { "outputOnFailure": true }, "filters": [ { "include": { "label": "^perf$" } } ] },
    { "name": "linux-release-znver4-fast",   "configurePreset": "linux-release-znver4",   "output": { "outputOnFailure": true }, "filters": [ { "include": { "label": "^fast$" } } ] }
  ]
```

- [ ] **Step 2: Verify presets**

```bash
cmake --preset linux-debug >/dev/null && ctest --preset linux-debug -N | tail -2
ctest --preset linux-debug-inference -N | tail -2
ctest --preset linux-debug-perf -N | tail -2
ctest --preset linux-debug-fast -N | tail -2
ctest --build --preset linux-debug --target test_ctest_labels_guard && \
  ctest --preset linux-debug -R test_ctest_labels_guard --output-on-failure
```

Expected: базовый пресет и `-fast` = 42 теста; `-inference` = 78;
`-perf` = 6; guard PASS (CMake валидирует схему пресетов при configure —
ошибка схемы упала бы на первом шаге).

- [ ] **Step 3: Commit**

```bash
git add CMakePresets.json
git commit -m "build: label-filtered test presets (fast/inference/perf)"
```

---

### Task 5: Guard-проверка роутера (RED)

**Files:**
- Modify: `tests/test_ctest_labels_guard.cpp`

**Interfaces:**
- Consumes: `Categories()` не нужен (валидация одним регэкспом).
- Produces: функция `CheckRouter`.

- [ ] **Step 1: Add the failing check**

В анонимный namespace добавить (после `CheckPresets`):

```cpp
// tools/test_router.cmake declares: lsx_router_rule(<id> <label> <regex>)
// — one rule per line, label is fast|inference (perf is never auto-run).
std::vector<std::string> CheckRouter(const std::string& router_text)
{
    std::vector<std::string> v;
    static const std::regex rule_re(
        R"(lsx_router_rule\s*\(\s*(\w+)\s+(fast|inference)\s+(.*)\))");
    bool any = false;
    std::istringstream in(router_text);
    std::string line;
    while (std::getline(in, line))
    {
        const auto first = line.find_first_not_of(" \t");
        if (first == std::string::npos || line[first] == '#')
            continue;
        if (line.find("lsx_router_rule") == std::string::npos)
            continue;
        std::smatch m;
        if (std::regex_search(line, m, rule_re))
            any = true;
        else
            v.push_back("tools/test_router.cmake: malformed rule: " + line);
    }
    if (!any)
        v.push_back("tools/test_router.cmake: no lsx_router_rule entries");
    return v;
}
```

И TEST_CASE (после preset-теста):

```cpp
TEST_CASE("ctest labels guard: router rules valid", "[labels][guard]")
{
    const fs::path root(LSX_SOURCE_DIR);
    const std::string router_text =
        ReadFileOrThrow(root / "tools" / "test_router.cmake");
    auto v = CheckRouter(router_text);
    std::string joined;
    for (const auto& s : v)
        joined += s + "\n";
    INFO("router violations:\n" << joined);
    REQUIRE(v.empty());
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cmake --build --preset linux-debug --target test_ctest_labels_guard && \
  ctest --preset linux-debug -R test_ctest_labels_guard --output-on-failure
```

Expected: FAIL на `cannot read .../tools/test_router.cmake` (файла ещё нет).

- [ ] **Step 3: Commit (RED)**

```bash
git add tests/test_ctest_labels_guard.cpp
git commit -m "test: router rule guard (RED)"
```

---

### Task 6: git-diff-роутер tools/test_router.cmake (GREEN для Task 5)

**Files:**
- Create: `tools/test_router.cmake`

**Interfaces:**
- Consumes: labels из Task 2; testPresets из Task 4.
- Produces: команда запуска (из корня репо):

```
cmake -DLSX_PRESET=linux-debug -P tools/test_router.cmake
cmake -DLSX_PRESET=linux-debug -DLSX_BASE=origin/main -DLSX_LIST=ON -P tools/test_router.cmake
cmake -DLSX_PRESET=linux-debug -DLSX_ALL=ON -P tools/test_router.cmake
```

- [ ] **Step 1: Write the router**

Создать `tools/test_router.cmake` целиком:

```cmake
# git-diff-driven ctest category router (script mode, no build system).
# Labels: fast (always) + inference (when the inference contour changed).
# perf is NEVER auto-run: `ctest --preset <p> -L perf` is a manual command.
#
# Usage (from the repo root):
#   cmake -DLSX_PRESET=linux-debug [-DLSX_BASE=<ref>] [-DLSX_ALL=ON] \
#       [-DLSX_LIST=ON] -P tools/test_router.cmake
#   LSX_PRESET  required ctest preset name (e.g. linux-debug)
#   LSX_BASE    diff base ref, default origin/main (merge-base diff)
#   LSX_ALL=ON  force fast+inference regardless of the diff
#   LSX_LIST=ON print the selected labels and exit (no ctest run)
#
# Windows note: the inference sweep additionally applies
# -R "^(test_qwen|test_gemma)" per the AGENTS.md OS restriction.
cmake_minimum_required(VERSION 3.25)

if(NOT DEFINED LSX_PRESET)
    message(FATAL_ERROR
        "usage: cmake -DLSX_PRESET=<ctest-preset> [-DLSX_BASE=<ref>] "
        "[-DLSX_ALL=ON] [-DLSX_LIST=ON] -P tools/test_router.cmake")
endif()
if(NOT DEFINED LSX_BASE)
    set(LSX_BASE "origin/main")
endif()
if(NOT DEFINED LSX_ALL)
    set(LSX_ALL OFF)
endif()
if(NOT DEFINED LSX_LIST)
    set(LSX_LIST OFF)
endif()

get_filename_component(LSX_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

# ── Rule table (guard-checked by tests/test_ctest_labels_guard.cpp) ───────
# lsx_router_rule(<id> <label> <regex>) — FIRST match wins; order matters.
macro(lsx_router_rule id label regex)
    list(APPEND LSX_RULE_IDS "${id}")
    list(APPEND LSX_RULE_LABELS "${label}")
    list(APPEND LSX_RULE_REGEXES "${regex}")
endmacro()

set(LSX_RULE_IDS "")
set(LSX_RULE_LABELS "")
set(LSX_RULE_REGEXES "")

# 1. Any CUDA source anywhere → inference.
lsx_router_rule(cuda inference "\\.cuh?$")
# 2. lsxcommon: host-only fast-lexicon first, everything else → inference
#    (conservative default: a new unknown lsxcommon file is inference).
lsx_router_rule(lsxcommon_host fast "libs/lsxcommon/.*(tokenizer|unicode|pretokenizer|path_registry|schema|converter|_convert|orchestrator)")
lsx_router_rule(lsxcommon inference "libs/lsxcommon/")
# 3. Network layer, apps → fast only.
lsx_router_rule(network fast "libs/lsxnetwork/")
lsx_router_rule(apps fast "apps/")
# 4. Test infra changed → both sweeps (before the tooling rule, else
#    tests/CMakeLists.txt would be swallowed by CMakeLists\.txt$).
lsx_router_rule(tests inference "^tests/")
# 5. Repo-root build files, docs, tools, top-level *.md → fast only.
lsx_router_rule(tooling fast "^(CMakeLists\\.txt$|CMakePresets\\.json$|cmake/|docs/|tools/|[^/]+\\.md$)")
# 6. Everything else → fast.
lsx_router_rule(default fast ".")

set(LSX_LABELS "fast")
if(NOT LSX_ALL)
    execute_process(
        COMMAND git diff --name-only ${LSX_BASE}...HEAD
        WORKING_DIRECTORY "${LSX_REPO_ROOT}"
        OUTPUT_VARIABLE LSX_DIFF
        RESULT_VARIABLE LSX_DIFF_RC
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT LSX_DIFF_RC EQUAL 0 OR LSX_DIFF STREQUAL "")
        message(STATUS
            "test_router: no diff against ${LSX_BASE} — running fast only")
        set(LSX_DIFF "")
    endif()
    string(REPLACE "\n" ";" LSX_FILES "${LSX_DIFF}")
    foreach(f IN LISTS LSX_FILES)
        if(f STREQUAL "")
            continue()
        endif()
        list(LENGTH LSX_RULE_IDS LSX_N)
        set(i 0)
        while(i LESS LSX_N)
            list(GET LSX_RULE_LABELS ${i} lbl)
            list(GET LSX_RULE_REGEXES ${i} re)
            if(f MATCHES "${re}")
                if(lbl STREQUAL "inference")
                    list(FIND LSX_LABELS "inference" idx)
                    if(idx EQUAL -1)
                        list(APPEND LSX_LABELS "inference")
                    endif()
                endif()
                break()
            endif()
            math(EXPR i "${i} + 1")
        endwhile()
    endforeach()
else()
    list(APPEND LSX_LABELS "inference")
endif()

list(FIND LSX_LABELS "inference" LSX_HAS_INFERENCE)
list(JOIN LSX_LABELS "|" LSX_LABEL_RE)
message(STATUS "test_router: labels = ${LSX_LABELS}")
if(LSX_LIST)
    return()
endif()

find_program(LSX_CTEST_EXE ctest REQUIRED)
set(LSX_FAILED OFF)
if(WIN32)
    # Two sweeps: the qwen|gemma name filter must not shrink the fast run.
    execute_process(COMMAND ${LSX_CTEST_EXE} --preset ${LSX_PRESET}
        -L "^fast$" RESULT_VARIABLE LSX_RC)
    if(NOT LSX_RC EQUAL 0)
        set(LSX_FAILED ON)
    endif()
    if(LSX_HAS_INFERENCE GREATER -1)
        execute_process(COMMAND ${LSX_CTEST_EXE} --preset ${LSX_PRESET}
            -L "^inference$" -R "^(test_qwen|test_gemma)"
            RESULT_VARIABLE LSX_RC_INF)
        if(NOT LSX_RC_INF EQUAL 0)
            set(LSX_FAILED ON)
        endif()
    endif()
else()
    execute_process(COMMAND ${LSX_CTEST_EXE} --preset ${LSX_PRESET}
        -L "^(${LSX_LABEL_RE})$" RESULT_VARIABLE LSX_RC)
    if(NOT LSX_RC EQUAL 0)
        set(LSX_FAILED ON)
    endif()
endif()
if(LSX_FAILED)
    message(FATAL_ERROR "test_router: ctest sweep failed")
endif()
return()
```

- [ ] **Step 2: Run guard to verify it passes**

```bash
cmake --build --preset linux-debug --target test_ctest_labels_guard && \
  ctest --preset linux-debug -R test_ctest_labels_guard --output-on-failure
```

Expected: PASS.

- [ ] **Step 3: Smoke-check router --list**

```bash
cmake -DLSX_PRESET=linux-debug -DLSX_LIST=ON -P tools/test_router.cmake
```

Expected: `-- test_router: labels = fast` (нет диффа против origin/main) или
`labels = fast;inference` если дифф затрагивает контур. Ошибок CMake нет.

- [ ] **Step 4: Commit**

```bash
git add tools/test_router.cmake
git commit -m "feat: git-diff test router (fast+inference, perf manual)"
```

---

### Task 7: Guard-проверка имён tests/ (RED)

**Files:**
- Modify: `tests/test_ctest_labels_guard.cpp`

**Interfaces:**
- Consumes: ничего нового.
- Produces: функция `CheckTestsNaming`.

- [ ] **Step 1: Add the failing check**

В анонимный namespace добавить (после `CheckRouter`):

```cpp
// Every file in tests/ (top level) follows the test_* prefix: test_*.cpp,
// test_*.cu, test_*_fixture.h, test_*_test_fixture.h is NOT allowed.
std::vector<std::string> CheckTestsNaming(const fs::path& tests_dir)
{
    std::vector<std::string> v;
    static const std::regex ok(R"(test_.+\.(cpp|cu|h)$)");
    std::error_code ec;
    for (auto& e : fs::directory_iterator(tests_dir, ec))
    {
        if (!e.is_regular_file())
            continue;
        const std::string name = e.path().filename().string();
        if (name == "CMakeLists.txt")
            continue;
        if (!std::regex_match(name, ok))
            v.push_back("tests/ naming violation (expected test_*): " + name);
    }
    return v;
}
```

И TEST_CASE (в конец файла):

```cpp
TEST_CASE("ctest labels guard: tests/ file naming uniformity",
          "[labels][guard]")
{
    const fs::path root(LSX_SOURCE_DIR);
    auto v = CheckTestsNaming(root / "tests");
    std::string joined;
    for (const auto& s : v)
        joined += s + "\n";
    INFO("naming violations:\n" << joined);
    REQUIRE(v.empty());
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cmake --build --preset linux-debug --target test_ctest_labels_guard && \
  ctest --preset linux-debug -R test_ctest_labels_guard --output-on-failure
```

Expected: FAIL с 7 нарушениями — `<family>_test_fixture.h` (glm, deepseek,
gemma, qwen_dense, qwen_moe, qwen_moe_exp, kimi).

- [ ] **Step 3: Commit (RED)**

```bash
git add tests/test_ctest_labels_guard.cpp
git commit -m "test: tests/ naming-uniformity guard (RED)"
```

---

### Task 8: Переименование фикстур (GREEN для Task 7)

**Files:**
- Rename: `tests/glm_moe_test_fixture.h` → `tests/test_glm_moe_fixture.h`
- Rename: `tests/deepseek_moe_test_fixture.h` → `tests/test_deepseek_moe_fixture.h`
- Rename: `tests/gemma_moe_test_fixture.h` → `tests/test_gemma_moe_fixture.h`
- Rename: `tests/qwen_dense_test_fixture.h` → `tests/test_qwen_dense_fixture.h`
- Rename: `tests/qwen_moe_test_fixture.h` → `tests/test_qwen_moe_fixture.h`
- Rename: `tests/qwen_moe_exp_test_fixture.h` → `tests/test_qwen_moe_exp_fixture.h`
- Rename: `tests/kimi_moe_test_fixture.h` → `tests/test_kimi_moe_fixture.h`
- Modify: все файлы в `tests/`, включающие эти заголовки (~30 include)

**Interfaces:**
- Consumes: ничего.
- Produces: единообразное именование `tests/test_*`; фикс-паттерн для
  `CheckTestsNaming`.

- [ ] **Step 1: Rename fixtures with git mv**

```bash
git mv tests/glm_moe_test_fixture.h tests/test_glm_moe_fixture.h
git mv tests/deepseek_moe_test_fixture.h tests/test_deepseek_moe_fixture.h
git mv tests/gemma_moe_test_fixture.h tests/test_gemma_moe_fixture.h
git mv tests/qwen_dense_test_fixture.h tests/test_qwen_dense_fixture.h
git mv tests/qwen_moe_exp_test_fixture.h tests/test_qwen_moe_exp_fixture.h
git mv tests/qwen_moe_test_fixture.h tests/test_qwen_moe_fixture.h
git mv tests/kimi_moe_test_fixture.h tests/test_kimi_moe_fixture.h
```

(qwen_moe_exp переименовывается ДО qwen_moe — защита от ложного совпадения
при следующем шаге.)

- [ ] **Step 2: Update includes**

```bash
rg -l 'test_fixture\.h' tests/ | xargs sed -i \
  -e 's/glm_moe_test_fixture\.h/test_glm_moe_fixture.h/g' \
  -e 's/deepseek_moe_test_fixture\.h/test_deepseek_moe_fixture.h/g' \
  -e 's/gemma_moe_test_fixture\.h/test_gemma_moe_fixture.h/g' \
  -e 's/qwen_dense_test_fixture\.h/test_qwen_dense_fixture.h/g' \
  -e 's/qwen_moe_exp_test_fixture\.h/test_qwen_moe_exp_fixture.h/g' \
  -e 's/qwen_moe_test_fixture\.h/test_qwen_moe_fixture.h/g' \
  -e 's/kimi_moe_test_fixture\.h/test_kimi_moe_fixture.h/g'
rg -n 'test_fixture\.h' tests/ || true
```

Expected: последняя `rg` ничего не находит (0 совпадений).

- [ ] **Step 3: Rebuild affected targets and run guard**

```bash
cmake --build --preset linux-debug --target test_ctest_labels_guard \
  test_glm_moe_loader test_qwen_dense_model test_qwen_moe_convert \
  test_kimi_moe_model test_gemma_moe_convert test_deepseek_moe_loader
ctest --preset linux-debug -R test_ctest_labels_guard --output-on-failure
```

Expected: PASS (все 5 guard-TEST_CASE зелёные).

- [ ] **Step 4: Commit**

```bash
git add -A tests/
git commit -m "refactor: unify tests/ fixture naming to test_*_fixture.h"
```

---

### Task 9: Документация AGENTS.md + финальная верификация

**Files:**
- Modify: `AGENTS.md` (раздел `## Test`)

**Interfaces:**
- Consumes: всё предыдущее.
- Produces: задокументированные команды трёх категорий.

- [ ] **Step 1: Update the Test section**

В `AGENTS.md` сразу после строки `## Test` заменить содержимое раздела на:

```markdown
## Test

`BUILD_TESTING=ON` is included in every preset. Тесты разбиты на три
категории (CTest labels, guard: `test_ctest_labels_guard`):

| Label | Что это | Когда запускается |
|---|---|---|
| `fast` (42) | host-only тесты | каждый `ctest --preset` (дефолт) |
| `inference` (78) | CUDA/модельный контур | когда изменился контур (роутер) или вручную |
| `perf` (6) | perf-бенчмарки на реальных весах | только отдельной командой |

```bash
ctest --preset linux-debug                  # fast only (дефолт всех пресетов)
ctest --preset linux-debug-inference        # inference sweep
ctest --preset linux-debug-perf             # perf, отдельная команда
ctest -L "^(fast|inference|perf)$"          # полный прогон

# Автодетект по git diff (запускает fast + inference при изменении контура):
cmake -DLSX_PRESET=linux-debug -P tools/test_router.cmake
cmake -DLSX_PRESET=linux-debug -DLSX_LIST=ON -P tools/test_router.cmake  # dry-run
```

Дисциплина: новый тест обязан получить label в том же коммите (иначе красный
`test_ctest_labels_guard`). Правила классификации путей — таблица
`lsx_router_rule(...)` в `tools/test_router.cmake`.
```

Существующий Windows-абзац
(`**WINDOWS-ОГРАНИЧЕНИЕ**: под Windows запускать тесты ТОЛЬКО ...`)
оставить без изменений после новой таблицы.

- [ ] **Step 2: Full verification**

```bash
ctest --preset linux-debug -N | tail -1          # Total Tests: 42
ctest --preset linux-debug                       # fast зелёный
ctest --preset linux-debug-perf -N | tail -1     # Total Tests: 6
cmake -DLSX_PRESET=linux-debug -DLSX_ALL=ON -DLSX_LIST=ON -P tools/test_router.cmake
```

Expected: 42 / зелёный / 6 / `labels = fast;inference`.

- [ ] **Step 3: Commit**

```bash
git add AGENTS.md
git commit -m "docs: test category workflows in AGENTS.md"
```
