# FSISelfCheck Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, GPU-independent model-parity gate (`FSISelfCheck`) that captures golden answers for a fixed 3-prompt financial set from the original safetensors model and verifies the converted NVFP4 model reproduces them.

**Architecture:** A host-only, model-neutral `lsxcommon::fsi` module owns extraction, golden-JSON parsing and the verify harness. A Python capture script (HF transformers, generic + per-family overrides) produces per-model committed golden JSON. FSI is fully decoupled from `SelfCheck`: separate API, result type, env var, numeric mode (`kOff`), CLI flag and ctest targets.

**Tech Stack:** C++20, Catch2 v3.16.0, simdjson (already a private `lsxcommon` dep), CMake presets, Python 3 + transformers/torch (capture only).

## Global Constraints

- Source of truth: `docs/superpowers/specs/2026-09-26-fsi-selfcheck-design.md`. If the plan and spec disagree, the spec wins; update the spec first.
- Branch: `feat/fsi-selfcheck` (already checked out). Never commit on `main`.
- Logging only via `lsxcommon::log` (spdlog facade); no `std::cout`/`printf`.
- `absl::StrFormat` is printf-style (`%s/%d/%u/%f`); never `{}`. Concatenations use `absl::StrCat`.
- Do not add comments to code unless the surrounding file already documents that way; follow existing style.
- No new family/variant/role naming tokens. Host module files are family-neutral (`fsi_selfcheck.{h,cpp}`), not scanned by the naming guard.
- Every new ctest target gets a category label (`fast` | `inference`) in the same commit; FSI real-model targets get `SKIP_RETURN_CODE 4`.
- FSI must NOT call `IModel::SelfCheck`, `RunParisSelfCheck`, `selfcheck_passed`, or `LSX_SELFCHECK_EXPECT_HASH`.
- No hardcoded vault paths: use `lsxcommon::lsx_model_models_dir()` / `LSX_MODEL_ROOT` (spec §3.5, §10).
- Exit criteria per task: `cmake --build --preset linux-debug` clean + the named `ctest` green before commit.
- Offline/no-GPU host unit tests must run on a CPU-only machine (label `fast`).

---

## File Structure

Create:
- `libs/lsxcommon/include/lsxcommon/fsi_selfcheck.h` — FSI types + API.
- `libs/lsxcommon/src/fsi_selfcheck.cpp` — extraction, golden parse, harness.
- `tests/test_fsi_selfcheck.cpp` — host-only unit tests.
- `tools/fsi_prompts.json` — canonical 3-prompt set.
- `tools/fsi_capture.py` — capture + `--selftest`.
- `tests/fixtures/fsi/<model_type>.json` — committed goldens.
- `tests/test_<family>_fsi_selfcheck.cpp` — per-family real-model tests (Task 7).

Modify:
- `libs/lsxcommon/CMakeLists.txt` — add `src/fsi_selfcheck.cpp`.
- `libs/lsxcommon/include/lsxcommon/model_runtime.h` — declare `RunCliFsiSelfCheck`.
- `libs/lsxcommon/src/model_runtime.cpp` — implement `RunCliFsiSelfCheck`.
- `apps/lsxbenchmark/src/main.cpp`, `apps/lsxbenchmark/src/cli.{h,cpp}` — `--fsi-selfcheck`.
- `tests/CMakeLists.txt` — new targets + label lists.
- `tests/test_model_suite_guard.cpp` — `fsi` lifecycle stage.
- `AGENTS.md` — Structure + Test pointers.

---

## Task 1: FSI extraction + host test target

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/fsi_selfcheck.h`
- Create: `libs/lsxcommon/src/fsi_selfcheck.cpp`
- Create: `tests/test_fsi_selfcheck.cpp`
- Modify: `libs/lsxcommon/CMakeLists.txt` (source list)
- Modify: `tests/CMakeLists.txt` (target + `fast` label list)

**Interfaces:**
- Consumes: nothing (pure host code).
- Produces:
  - `enum class lsxcommon::fsi::FsiKind { kArithmetic, kComparison };`
  - `std::string lsxcommon::fsi::ExtractFsiFormula(FsiKind, std::string_view anchor, std::string_view raw);`

- [ ] **Step 1: Write the failing test**

Create `tests/test_fsi_selfcheck.cpp`:

```cpp
// FSISelfCheck host-only unit tests (extraction/golden/harness). No GPU.
#include <catch2/catch_test_macros.hpp>

#include <string>

#include <lsxcommon/fsi_selfcheck.h>

namespace fsi = lsxcommon::fsi;

TEST_CASE("fsi: arithmetic extraction normalizes to a bare formula",
          "[fsi][unit]")
{
    CHECK(fsi::ExtractFsiFormula(fsi::FsiKind::kArithmetic, "PV",
          "PV=CF*((1-exp(-r*n))/(1-exp(-r)))") ==
          "PV=CF*((1-exp(-r*n))/(1-exp(-r)))");
    CHECK(fsi::ExtractFsiFormula(fsi::FsiKind::kArithmetic, "PV",
          "```python\nPV = CF*((1-exp(-r*n))/(1-exp(-r)))\n```") ==
          "PV=CF*((1-exp(-r*n))/(1-exp(-r)))");
    CHECK(fsi::ExtractFsiFormula(fsi::FsiKind::kArithmetic, "PV",
          "PV=CF*((1-exp(-r*n))/(1-exp(-r))) This is the formula.") ==
          "PV=CF*((1-exp(-r*n))/(1-exp(-r)))");
    CHECK(fsi::ExtractFsiFormula(fsi::FsiKind::kArithmetic, "Balance_Alpha",
          "Balance_Alpha = (((100000000*Rate_A)-Slip_A)*Rate_B-Slip_B)*Rate_C-Slip_C") ==
          "Balance_Alpha=(((100000000*Rate_A)-Slip_A)*Rate_B-Slip_B)*Rate_C-Slip_C");
    CHECK(fsi::ExtractFsiFormula(fsi::FsiKind::kArithmetic, "PV", "").empty());
    CHECK(fsi::ExtractFsiFormula(fsi::FsiKind::kArithmetic, "PV",
          "no formula here").empty());
}

TEST_CASE("fsi: comparison extraction is bounded", "[fsi][unit]")
{
    CHECK(fsi::ExtractFsiFormula(fsi::FsiKind::kComparison, "Drawdown",
          "Drawdown>Threshold") == "Drawdown>Threshold");
    CHECK(fsi::ExtractFsiFormula(fsi::FsiKind::kComparison, "Drawdown",
          "Drawdown > Threshold") == "Drawdown>Threshold");
    CHECK(fsi::ExtractFsiFormula(fsi::FsiKind::kComparison, "Drawdown",
          "Drawdown>Threshold.") == "Drawdown>Threshold");
    CHECK(fsi::ExtractFsiFormula(fsi::FsiKind::kComparison, "Drawdown",
          "```\nDrawdown>Threshold\n```") == "Drawdown>Threshold");
    CHECK(fsi::ExtractFsiFormula(fsi::FsiKind::kComparison, "Drawdown",
          "no answer").empty());
}
```

- [ ] **Step 2: Add the CMake target and confirm the test fails to compile**

In `tests/CMakeLists.txt`, immediately before the `# ── Test category labels` block (search: `# ── Test category labels (fast | convert | inference | perf)`), add:

```cmake
# ── FSISelfCheck: host-only extraction/golden/harness unit tests (no GPU) ──
add_executable(test_fsi_selfcheck test_fsi_selfcheck.cpp)
target_link_libraries(test_fsi_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
add_test(NAME test_fsi_selfcheck COMMAND test_fsi_selfcheck)
```

Add `test_fsi_selfcheck` to the big `set_tests_properties(... PROPERTIES LABELS fast)` list (the one containing `test_model_runtime`).

Run:

```bash
cmake --preset linux-debug
cmake --build --preset linux-debug --target test_fsi_selfcheck
```

Expected: compile failure (`lsxcommon/fsi_selfcheck.h` not found).

- [ ] **Step 3: Create the header and the extraction implementation**

Create `libs/lsxcommon/include/lsxcommon/fsi_selfcheck.h`:

```cpp
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "lsxcommon/model.h"

namespace lsxcommon {
namespace fsi {

enum class FsiKind { kArithmetic, kComparison };

struct FsiGoldenCase {
    std::string id;
    FsiKind kind = FsiKind::kArithmetic;
    std::string anchor;
    std::vector<int32_t> input_ids;
    int max_new_tokens = 128;
    std::string expected;
    std::string raw;
};

struct FsiGolden {
    std::string model_type;
    std::vector<FsiGoldenCase> cases;
};

struct FsiCaseResult {
    std::string id;
    std::string expected;
    std::string extracted;
    bool passed = false;
    std::string error;
};

struct FsiResult {
    std::vector<FsiCaseResult> cases;
    bool all_passed = false;
    std::string error;
};

std::string ExtractFsiFormula(FsiKind kind, std::string_view anchor,
                              std::string_view raw);

bool ParseFsiGolden(std::string_view json, FsiGolden &out, std::string *error);
bool LoadFsiGoldenFile(const std::string &path, FsiGolden &out,
                       std::string *error);

using FsiDecodeFn = std::function<std::string(const std::vector<int32_t> &)>;

FsiResult RunFsiSelfCheck(IModel &model, const FsiGolden &golden);
FsiResult RunFsiSelfCheck(IModel &model, const FsiGolden &golden,
                          const FsiDecodeFn &decode);

}  // namespace fsi
}  // namespace lsxcommon
```

Create `libs/lsxcommon/src/fsi_selfcheck.cpp` with ONLY the extraction for now (golden/harness added in Tasks 2-3):

```cpp
#include "lsxcommon/fsi_selfcheck.h"

#include <absl/strings/ascii.h>
#include <cctype>

namespace lsxcommon {
namespace fsi {
namespace {

bool IsArithChar(char c)
{
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_' ||
           c == '+' || c == '-' || c == '*' || c == '/' || c == '(' ||
           c == ')' || c == '.';
}

bool IsComparisonOp(char c)
{
    return c == '<' || c == '>' || c == '=' || c == '!';
}

std::string Unfence(std::string_view raw)
{
    std::string s = absl::StripAsciiWhitespace(raw);
    const std::size_t open = s.find("```");
    if (open == std::string::npos) return s;
    std::size_t begin = open + 3;
    const std::size_t line_end = s.find('\n', begin);
    if (line_end != std::string::npos) {
        const std::string first = s.substr(begin, line_end - begin);
        bool lang = !first.empty();
        for (char c : first)
            if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' ||
                  c == '+' || c == '-'))
                lang = false;
        if (lang) begin = line_end + 1;
    }
    const std::size_t close = s.find("```", begin);
    const std::string inner = (close == std::string::npos)
                                  ? s.substr(begin)
                                  : s.substr(begin, close - begin);
    return absl::StripAsciiWhitespace(inner);
}

std::string FirstLineWith(const std::string &s, std::string_view anchor)
{
    std::size_t pos = 0;
    while (pos <= s.size()) {
        const std::size_t nl = s.find('\n', pos);
        const std::string line =
            (nl == std::string::npos) ? s.substr(pos)
                                      : s.substr(pos, nl - pos);
        if (line.find(anchor) != std::string::npos) return line;
        if (nl == std::string::npos) break;
        pos = nl + 1;
    }
    return s;
}

std::string Normalize(std::string_view s)
{
    std::string out;
    out.reserve(s.size());
    for (char c : s)
        if (!std::isspace(static_cast<unsigned char>(c))) out.push_back(c);
    while (!out.empty() && (out.back() == '.' || out.back() == ',' ||
                            out.back() == ';'))
        out.pop_back();
    return out;
}

} // namespace

std::string ExtractFsiFormula(FsiKind kind, std::string_view anchor,
                              std::string_view raw)
{
    const std::string line = FirstLineWith(Unfence(raw), anchor);
    if (kind == FsiKind::kComparison) {
        const std::size_t a = line.find(anchor);
        if (a == std::string::npos) return {};
        std::size_t i = a + anchor.size();
        while (i < line.size() && std::isspace((unsigned char)line[i])) ++i;
        const std::size_t op = i;
        while (i < line.size() && IsComparisonOp(line[i])) ++i;
        if (i == op) return {};
        while (i < line.size() && std::isspace((unsigned char)line[i])) ++i;
        if (i >= line.size() ||
            !(std::isalpha((unsigned char)line[i]) || line[i] == '_'))
            return {};
        while (i < line.size() &&
               (std::isalnum((unsigned char)line[i]) || line[i] == '_'))
            ++i;
        return Normalize(std::string_view(line).substr(a, i - a));
    }
    const std::size_t a = line.find(anchor);
    if (a == std::string::npos) return {};
    const std::size_t eq = line.find('=', a + anchor.size());
    if (eq == std::string::npos) return {};
    const std::size_t expr = eq + 1;
    std::size_t end = line.size();
    if (line.find('(', expr) != std::string::npos) {
        int depth = 0;
        bool started = false;
        for (std::size_t k = expr; k < line.size(); ++k) {
            const char c = line[k];
            if (c == '(') { ++depth; started = true; }
            else if (c == ')') {
                --depth;
                if (started && depth == 0) { end = k + 1; break; }
            }
        }
    } else {
        std::size_t k = expr;
        while (k < line.size() && IsArithChar(line[k])) ++k;
        end = k;
    }
    return Normalize(std::string_view(line).substr(a, end - a));
}

}  // namespace fsi
}  // namespace lsxcommon
```

Add `src/fsi_selfcheck.cpp` to the `add_library(lsxcommon STATIC ...)` source list in `libs/lsxcommon/CMakeLists.txt` (place it next to `src/model_runtime.cpp`).

- [ ] **Step 4: Build and run the test**

```bash
cmake --build --preset linux-debug --target test_fsi_selfcheck
ctest --preset linux-debug -R '^test_fsi_selfcheck$' --output-on-failure
```

Expected: PASS (all extraction checks).

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/fsi_selfcheck.h \
        libs/lsxcommon/src/fsi_selfcheck.cpp \
        libs/lsxcommon/CMakeLists.txt tests/test_fsi_selfcheck.cpp tests/CMakeLists.txt
git commit -m "feat: add FSI answer extraction with host unit tests"
```

---

## Task 2: Golden JSON parse / load

**Files:**
- Modify: `libs/lsxcommon/src/fsi_selfcheck.cpp`
- Modify: `tests/test_fsi_selfcheck.cpp`

**Interfaces:**
- Consumes: `FsiGolden`, `FsiGoldenCase`, `FsiKind` (Task 1).
- Produces:
  - `bool lsxcommon::fsi::ParseFsiGolden(std::string_view, FsiGolden &, std::string *error);`
  - `bool lsxcommon::fsi::LoadFsiGoldenFile(const std::string &, FsiGolden &, std::string *error);`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_fsi_selfcheck.cpp`:

```cpp
#include <vector>

namespace {

const char *kGoldenJson = R"JSON({
  "format": "logestix-fsi-golden",
  "version": 1,
  "model_type": "fake-model",
  "source": "fake",
  "runtime": "test",
  "prompts": [
    {
      "id": "FSI-TEST-01",
      "kind": "arithmetic",
      "anchor": "PV",
      "input_ids": [1, 2, 3],
      "max_new_tokens": 16,
      "expected": "PV=CF*((1-exp(-r*n))/(1-exp(-r)))",
      "raw": "PV=CF*((1-exp(-r*n))/(1-exp(-r)))"
    },
    {
      "id": "FSI-TEST-03",
      "kind": "comparison",
      "anchor": "Drawdown",
      "input_ids": [9],
      "max_new_tokens": 16,
      "expected": "Drawdown>Threshold",
      "raw": "Drawdown>Threshold"
    }
  ]
})JSON";

} // namespace

TEST_CASE("fsi: golden JSON parses all fields", "[fsi][unit]")
{
    fsi::FsiGolden g;
    std::string err;
    REQUIRE(fsi::ParseFsiGolden(kGoldenJson, g, &err));
    CHECK(err.empty());
    CHECK(g.model_type == "fake-model");
    REQUIRE(g.cases.size() == 2);
    CHECK(g.cases[0].id == "FSI-TEST-01");
    CHECK(g.cases[0].kind == fsi::FsiKind::kArithmetic);
    CHECK(g.cases[0].anchor == "PV");
    CHECK(g.cases[0].input_ids == std::vector<int32_t>{1, 2, 3});
    CHECK(g.cases[0].max_new_tokens == 16);
    CHECK(g.cases[1].kind == fsi::FsiKind::kComparison);
}

TEST_CASE("fsi: malformed golden is rejected with an error", "[fsi][unit]")
{
    fsi::FsiGolden g;
    std::string err;
    CHECK_FALSE(fsi::ParseFsiGolden("{}", g, &err));
    CHECK_FALSE(err.empty());
    CHECK_FALSE(fsi::ParseFsiGolden(
        R"({"format":"logestix-fsi-golden","model_type":"m","prompts":[]})",
        g, &err));
}
```

- [ ] **Step 2: Run and confirm failure**

```bash
cmake --build --preset linux-debug --target test_fsi_selfcheck
ctest --preset linux-debug -R '^test_fsi_selfcheck$' --output-on-failure
```

Expected: link/compile failure (`ParseFsiGolden` undefined).

- [ ] **Step 3: Implement parse + load**

In `libs/lsxcommon/src/fsi_selfcheck.cpp`, add includes at the top:

```cpp
#include <simdjson.h>
#include <fstream>
#include <sstream>
```

Append before the closing `}  // namespace fsi`:

```cpp
bool ParseFsiGolden(std::string_view json, FsiGolden &out, std::string *error)
{
    out = {};
    simdjson::ondemand::parser parser;
    simdjson::padded_string pad(json.data(), json.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(pad).get(doc) != simdjson::SUCCESS) {
        if (error) *error = "invalid JSON";
        return false;
    }
    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS) {
        if (error) *error = "root is not an object";
        return false;
    }
    std::string_view tag;
    if (root["format"].get_string().get(tag) != simdjson::SUCCESS ||
        tag != "logestix-fsi-golden") {
        if (error) *error = "missing/unknown format tag";
        return false;
    }
    std::string_view model_type;
    if (root["model_type"].get_string().get(model_type) != simdjson::SUCCESS ||
        model_type.empty()) {
        if (error) *error = "missing model_type";
        return false;
    }
    out.model_type = std::string(model_type);
    simdjson::ondemand::array prompts;
    if (root["prompts"].get_array().get(prompts) != simdjson::SUCCESS) {
        if (error) *error = "missing prompts array";
        return false;
    }
    for (auto element : prompts) {
        simdjson::ondemand::object o;
        if (element.get_object().get(o) != simdjson::SUCCESS) {
            if (error) *error = "prompt entry is not an object";
            return false;
        }
        FsiGoldenCase c;
        std::string_view sv;
        if (o["id"].get_string().get(sv) != simdjson::SUCCESS || sv.empty()) {
            if (error) *error = "prompt missing id";
            return false;
        }
        c.id = std::string(sv);
        if (o["kind"].get_string().get(sv) != simdjson::SUCCESS) {
            if (error) *error = c.id + ": missing kind";
            return false;
        }
        if (sv == "arithmetic") c.kind = FsiKind::kArithmetic;
        else if (sv == "comparison") c.kind = FsiKind::kComparison;
        else { if (error) *error = c.id + ": unknown kind"; return false; }
        if (o["anchor"].get_string().get(sv) != simdjson::SUCCESS ||
            sv.empty()) {
            if (error) *error = c.id + ": missing anchor";
            return false;
        }
        c.anchor = std::string(sv);
        simdjson::ondemand::array ids;
        if (o["input_ids"].get_array().get(ids) != simdjson::SUCCESS) {
            if (error) *error = c.id + ": missing input_ids";
            return false;
        }
        for (auto v : ids) {
            int64_t id = 0;
            if (v.get_int64().get(id) != simdjson::SUCCESS) {
                if (error) *error = c.id + ": non-integer input id";
                return false;
            }
            c.input_ids.push_back(static_cast<int32_t>(id));
        }
        if (c.input_ids.empty()) {
            if (error) *error = c.id + ": empty input_ids";
            return false;
        }
        int64_t mnt = 0;
        if (o["max_new_tokens"].get_int64().get(mnt) == simdjson::SUCCESS &&
            mnt > 0)
            c.max_new_tokens = static_cast<int>(mnt);
        if (o["expected"].get_string().get(sv) != simdjson::SUCCESS ||
            sv.empty()) {
            if (error) *error = c.id + ": missing expected";
            return false;
        }
        c.expected = std::string(sv);
        std::string_view raw;
        if (o["raw"].get_string().get(raw) == simdjson::SUCCESS)
            c.raw = std::string(raw);
        out.cases.push_back(std::move(c));
    }
    if (out.cases.empty()) {
        if (error) *error = "no prompts in golden";
        return false;
    }
    return true;
}

bool LoadFsiGoldenFile(const std::string &path, FsiGolden &out,
                       std::string *error)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        if (error) *error = "cannot open " + path;
        return false;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    const std::string text = ss.str();
    std::string local;
    if (!ParseFsiGolden(text, out, &local)) {
        if (error) *error = path + ": " + local;
        return false;
    }
    return true;
}
```

- [ ] **Step 4: Build and run**

```bash
cmake --build --preset linux-debug --target test_fsi_selfcheck
ctest --preset linux-debug -R '^test_fsi_selfcheck$' --output-on-failure
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/fsi_selfcheck.cpp tests/test_fsi_selfcheck.cpp
git commit -m "feat: parse committed FSI golden JSON"
```

---

## Task 3: `RunFsiSelfCheck` harness (decoder seam)

**Files:**
- Modify: `libs/lsxcommon/src/fsi_selfcheck.cpp`
- Modify: `tests/test_fsi_selfcheck.cpp`

**Interfaces:**
- Consumes: `ExtractFsiFormula`, `FsiGolden`/`FsiGoldenCase` (Tasks 1-2), `IModel`, `runtime::DecodeGeneratedText`.
- Produces:
  - `using FsiDecodeFn = std::function<std::string(const std::vector<int32_t> &)>;`
  - `FsiResult RunFsiSelfCheck(IModel &, const FsiGolden &);`
  - `FsiResult RunFsiSelfCheck(IModel &, const FsiGolden &, const FsiDecodeFn &);`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_fsi_selfcheck.cpp`:

```cpp
#include <shared_mutex>
#include <stdexcept>

#include <lsxcommon/inference_gpu.h>
#include <lsxcommon/safetensors.h>
#include <lsxcommon/tokenizer.h>

namespace {

class FakeFsiModel : public lsxcommon::IModel {
public:
    bool Load() override { return true; }
    void Unload() override {}
    lsxcommon::SafetensorsModel Inspect() override { return {}; }
    lsxcommon::gpu::InferenceResult Infer(
        const lsxcommon::ModelRequest &req) override
    {
        last_ = req;
        if (fail) {
            lsxcommon::gpu::InferenceResult r;
            r.ok = false;
            r.error_msg = "boom";
            return r;
        }
        lsxcommon::gpu::InferenceResult r;
        r.ok = true;
        r.gen_tokens = gen;
        return r;
    }
    lsxcommon::gpu::InferenceResult SelfCheck(
        const lsxcommon::ModelRequest &) override
    {
        throw std::runtime_error("FSI must not call SelfCheck");
    }
    const std::string &Type() const override
    {
        static const std::string t = "fake-model";
        return t;
    }
    lsxcommon::ArrowTokenizerDb &Tokenizer() override
    {
        static lsxcommon::ArrowTokenizerDb db{};
        return db;
    }
    std::shared_mutex &ResidencyMutex() override { return m_; }

    std::vector<int32_t> gen = {7};
    bool fail = false;
    lsxcommon::ModelRequest last_;
private:
    std::shared_mutex m_;
};

fsi::FsiGolden OneCaseGolden()
{
    fsi::FsiGolden g;
    fsi::FsiGoldenCase c;
    c.id = "FSI-TEST-01";
    c.kind = fsi::FsiKind::kArithmetic;
    c.anchor = "PV";
    c.input_ids = {1, 2};
    c.max_new_tokens = 8;
    c.expected = "PV=CF*((1-exp(-r*n))/(1-exp(-r)))";
    g.cases.push_back(std::move(c));
    return g;
}

} // namespace

TEST_CASE("fsi: harness passes on a matching answer and forwards the request",
          "[fsi][unit]")
{
    FakeFsiModel m;
    const fsi::FsiGolden g = OneCaseGolden();
    const auto decode = [](const std::vector<int32_t> &) {
        return std::string("PV=CF*((1-exp(-r*n))/(1-exp(-r)))");
    };
    const fsi::FsiResult r = fsi::RunFsiSelfCheck(m, g, decode);
    REQUIRE(r.all_passed);
    REQUIRE(r.cases.size() == 1);
    CHECK(r.cases[0].extracted == "PV=CF*((1-exp(-r*n))/(1-exp(-r)))");
    CHECK(m.last_.input_ids == std::vector<int32_t>{1, 2});
    CHECK(m.last_.max_gen_tokens == 8);
}

TEST_CASE("fsi: harness fails on a mismatching answer", "[fsi][unit]")
{
    FakeFsiModel m;
    const fsi::FsiGolden g = OneCaseGolden();
    const auto decode = [](const std::vector<int32_t> &) {
        return std::string("PV=wrong");
    };
    const fsi::FsiResult r = fsi::RunFsiSelfCheck(m, g, decode);
    CHECK_FALSE(r.all_passed);
    REQUIRE(r.cases.size() == 1);
    CHECK(r.cases[0].extracted == "PV=wrong");
}

TEST_CASE("fsi: harness propagates inference errors per case", "[fsi][unit]")
{
    FakeFsiModel m;
    m.fail = true;
    const fsi::FsiGolden g = OneCaseGolden();
    const auto decode = [](const std::vector<int32_t> &) { return std::string(); };
    const fsi::FsiResult r = fsi::RunFsiSelfCheck(m, g, decode);
    CHECK_FALSE(r.all_passed);
    REQUIRE(r.cases.size() == 1);
    CHECK(r.cases[0].error == "boom");
}
```

- [ ] **Step 2: Run and confirm failure**

```bash
cmake --build --preset linux-debug --target test_fsi_selfcheck
ctest --preset linux-debug -R '^test_fsi_selfcheck$' --output-on-failure
```

Expected: link failure (`RunFsiSelfCheck` undefined).

- [ ] **Step 3: Implement the harness**

In `libs/lsxcommon/src/fsi_selfcheck.cpp`, add includes:

```cpp
#include <absl/strings/str_format.h>
#include "lsxcommon/log.h"
#include "lsxcommon/model_runtime.h"
```

Append before the closing `}  // namespace fsi`:

```cpp
namespace {

FsiCaseResult RunFsiCase(IModel &model, const FsiGoldenCase &c,
                         const FsiDecodeFn &decode)
{
    FsiCaseResult r;
    r.id = c.id;
    r.expected = c.expected;
    ModelRequest req;
    req.input_ids = c.input_ids;
    req.max_gen_tokens = c.max_new_tokens;
    const gpu::InferenceResult ir = model.Infer(req);
    if (!ir.ok) {
        r.error = ir.error_msg.empty() ? "inference failed" : ir.error_msg;
        return r;
    }
    const std::string text = decode(ir.gen_tokens);
    r.extracted = ExtractFsiFormula(c.kind, c.anchor, text);
    r.passed = (r.extracted == c.expected);
    return r;
}

} // namespace

FsiResult RunFsiSelfCheck(IModel &model, const FsiGolden &golden,
                          const FsiDecodeFn &decode)
{
    FsiResult res;
    for (const FsiGoldenCase &c : golden.cases)
        res.cases.push_back(RunFsiCase(model, c, decode));
    res.all_passed = !res.cases.empty();
    for (const FsiCaseResult &c : res.cases) {
        lsxcommon::log::info(absl::StrFormat(
            "[fsi] %s expected='%s' extracted='%s' passed=%s",
            c.id.c_str(), c.expected.c_str(), c.extracted.c_str(),
            c.passed ? "true" : "false"));
        if (!c.passed) res.all_passed = false;
    }
    return res;
}

FsiResult RunFsiSelfCheck(IModel &model, const FsiGolden &golden)
{
    return RunFsiSelfCheck(
        model, golden,
        [&model](const std::vector<int32_t> &ids) {
            return runtime::DecodeGeneratedText(model.Tokenizer(), ids);
        });
}
```

- [ ] **Step 4: Build and run**

```bash
cmake --build --preset linux-debug --target test_fsi_selfcheck
ctest --preset linux-debug -R '^test_fsi_selfcheck$' --output-on-failure
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/fsi_selfcheck.cpp tests/test_fsi_selfcheck.cpp
git commit -m "feat: add FSI golden verify harness with injected decoder"
```

---

## Task 4: Prompt set + Python capture script

**Files:**
- Create: `tools/fsi_prompts.json`
- Create: `tools/fsi_capture.py`

**Interfaces:**
- Consumes: golden format (spec §5); `expected` canonical strings (spec §4).
- Produces: golden JSON consumed by `LoadFsiGoldenFile` (Task 2).

- [ ] **Step 1: Write the canonical prompt set**

Create `tools/fsi_prompts.json`:

```json
{
  "format": "logestix-fsi-prompts",
  "version": 1,
  "prompts": [
    {
      "id": "FSI-TEST-01",
      "title": "CONTINUOUS-COMPOUNDING-STRUCT",
      "kind": "arithmetic",
      "anchor": "PV",
      "expected": "PV=CF*((1-exp(-r*n))/(1-exp(-r)))",
      "body": "[CONTEXT]: FSI-Validation-Core-01 // Financial Structure\nDefine the mathematical execution block for the Present Value (PV) of an annuity-due asset under continuous compounding.\nParameters: CF = 2500000; n = 7; r = 0.0835.\nTask: Format the target PV calculation strictly as a single-line Python-compatible expression using exp() and the exact variable names provided.\nOutput Specification: Output ONLY the raw formula string. Do not use markdown blocks, do not include spaces inside the formula, and do not emit explanatory text.\nExpected Template: PV=CF*((1-exp(-r*n))/(1-exp(-r)))"
    },
    {
      "id": "FSI-TEST-02",
      "title": "ARBITRAGE-ALGORITHMIC-PATH",
      "kind": "arithmetic",
      "anchor": "Balance_Alpha",
      "expected": "Balance_Alpha=(((100000000*Rate_A)-Slip_A)*Rate_B-Slip_B)*Rate_C-Slip_C",
      "body": "[CONTEXT]: FSI-Validation-Core-02 // Deterministic Routing Loop\nMap the structural sequence for a 3-leg currency clearing loop with fixed slippage mitigation.\nInitial_State: Balance_Alpha = 100000000.\nLegs: Rate_A=1.083541; Slip_A=0.0005; Rate_B=83.152402; Slip_B=0.0005; Rate_C=0.011104; Slip_C=0.0005.\nTask: Formulate the continuous multi-leg balance mutation sequence in a single line of standard arithmetic operators without spaces.\nOutput Specification: Output ONLY the final math equation string solving for the terminal state of Balance_Alpha using the provided variable names. No text, no markdown blocks.\nExpected Template: Balance_Alpha=(((100000000*Rate_A)-Slip_A)*Rate_B-Slip_B)*Rate_C-Slip_C"
    },
    {
      "id": "FSI-TEST-03",
      "title": "BASEL-THRESHOLD-LOGIC",
      "kind": "comparison",
      "anchor": "Drawdown",
      "expected": "Drawdown>Threshold",
      "body": "[CONTEXT]: FSI-Validation-Core-03 // Boundary Decision Check\nConstruct the evaluation logic for a Basel-compliant threshold boundary breach.\nVariables: Drawdown = 14.999991; Threshold = 15.000000.\nTask: Formulate the exact conditional logical statement that checks if Drawdown strictly exceeds Threshold.\nOutput Specification: Output strictly the standard boolean comparison string without spaces. Do not evaluate numerically, do not include code blocks or text.\nExpected Template: Drawdown>Threshold"
    }
  ]
}
```

- [ ] **Step 2: Write the capture script with a real self-test**

Create `tools/fsi_capture.py`:

```python
#!/usr/bin/env python3
"""FSISelfCheck reference capture (separate from SelfCheck).

Captures the FSI golden answers from an original safetensors model with HF
transformers. The logestix side only verifies the converted model against the
committed golden (libs/lsxcommon/src/fsi_selfcheck.cpp). Run the self-test with
no model present to validate the prompt/formula set.
"""
import argparse
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = os.path.join(HERE, "fsi_prompts.json")

# Per-model capture overrides keyed by model type. Absent -> generic defaults.
OVERRIDES = {
    # "ornith-1.5-35b-a3b": {"trust_remote_code": True},
}
GENERIC = {"trust_remote_code": True, "max_new_tokens": 128,
           "torch_dtype": "auto", "device_map": "auto"}


def load_prompts():
    with open(PROMPTS, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    assert doc["format"] == "logestix-fsi-prompts", doc["format"]
    return doc["prompts"]


def normalize(s):
    out = "".join(ch for ch in s if not ch.isspace())
    return out.rstrip(".,;")


def extract(kind, anchor, raw):
    m = re.search(r"```", raw)
    text = raw
    if m:
        begin = m.end()
        nl = text.find("\n", begin)
        if nl != -1 and re.fullmatch(r"[A-Za-z0-9_+-]*", text[begin:nl]):
            begin = nl + 1
        close = text.find("```", begin)
        text = text[begin:] if close == -1 else text[begin:close]
        text = text.strip()
    line = text
    for candidate in text.splitlines():
        if anchor in candidate:
            line = candidate
            break
    if kind == "comparison":
        pat = re.compile(re.escape(anchor) + r"\s*[<>=!]+\s*[A-Za-z_]\w*")
        hit = pat.search(line)
        return normalize(hit.group(0)) if hit else ""
    a = line.find(anchor)
    if a == -1:
        return ""
    eq = line.find("=", a + len(anchor))
    if eq == -1:
        return ""
    expr = eq + 1
    if "(" in line[expr:]:
        depth = 0
        started = False
        end = len(line)
        for k in range(expr, len(line)):
            if line[k] == "(":
                depth += 1
                started = True
            elif line[k] == ")":
                depth -= 1
                if started and depth == 0:
                    end = k + 1
                    break
    else:
        end = expr
        while end < len(line) and re.match(r"[A-Za-z0-9_+\-*/().]", line[end]):
            end += 1
    return normalize(line[a:end])


def selftest():
    prompts = load_prompts()
    assert [p["id"] for p in prompts] == ["FSI-TEST-01", "FSI-TEST-02", "FSI-TEST-03"]
    for p in prompts:
        got = extract(p["kind"], p["anchor"], p["expected"])
        assert got == p["expected"], (p["id"], got)
    # Formula checks: FSI-03 is False at the stated boundary; both arithmetic
    # formulas evaluate to positive finite values with the stated parameters.
    assert (14.999991 > 15.000000) is False
    v1 = eval("CF*((1-exp(-r*n))/(1-exp(-r)))", {"__builtins__": {}},
             {"CF": 2500000, "n": 7, "r": 0.0835, "exp": math.exp})
    v2 = eval("(((100000000*Rate_A)-Slip_A)*Rate_B-Slip_B)*Rate_C-Slip_C",
             {"__builtins__": {}},
             {"Rate_A": 1.083541, "Slip_A": 0.0005, "Rate_B": 83.152402,
              "Slip_B": 0.0005, "Rate_C": 0.011104, "Slip_C": 0.0005})
    assert math.isfinite(v1) and v1 > 0
    assert math.isfinite(v2) and v2 > 0
    print("fsi selftest OK: pv=%.6f arb=%.6f" % (v1, v2))


def capture(model, model_type, out, max_new_tokens):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    opts = dict(GENERIC)
    opts.update(OVERRIDES.get(model_type, {}))
    opts["max_new_tokens"] = max_new_tokens
    prompts = load_prompts()

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=opts["trust_remote_code"])
    lm = AutoModelForCausalLM.from_pretrained(
        model, trust_remote_code=opts["trust_remote_code"],
        torch_dtype=opts["torch_dtype"], device_map=opts["device_map"])
    lm.eval()

    cases = []
    for p in prompts:
        prompt_text = p["body"]
        if getattr(tok, "chat_template", None):
            try:
                prompt_text = tok.apply_chat_template(
                    [{"role": "user", "content": p["body"]}],
                    tokenize=False, add_generation_prompt=True)
            except Exception as exc:  # noqa: BLE001
                print("chat_template failed, using raw prompt:", exc, file=sys.stderr)
        ids = tok(prompt_text, return_tensors="pt").input_ids.to(lm.device)
        with torch.no_grad():
            out_ids = lm.generate(ids, do_sample=False,
                                  max_new_tokens=max_new_tokens)
        gen = out_ids[0][ids.shape[1]:].tolist()
        raw = tok.decode(gen, skip_special_tokens=True)
        got = extract(p["kind"], p["anchor"], raw)
        if got != p["expected"]:
            raise SystemExit(
                "%s: extracted %r != expected %r (raw=%r)"
                % (p["id"], got, p["expected"], raw))
        cases.append({"id": p["id"], "kind": p["kind"], "anchor": p["anchor"],
                      "input_ids": ids[0].tolist(),
                      "max_new_tokens": max_new_tokens,
                      "expected": p["expected"], "raw": raw})

    golden = {"format": "logestix-fsi-golden", "version": 1,
              "model_type": model_type, "source": model,
              "runtime": "transformers %s / torch %s" % (transformers.__version__, torch.__version__),
              "prompts": cases}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(golden, fh, indent=2)
        fh.write("\n")
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--type")
    ap.add_argument("--out")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not (args.model and args.type and args.out):
        ap.error("--model, --type and --out are required unless --selftest")
    capture(args.model, args.type, args.out, args.max_new_tokens)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the self-test**

```bash
python3 tools/fsi_capture.py --selftest
```

Expected: `fsi selftest OK: pv=... arb=...`.

- [ ] **Step 4: Commit**

```bash
git add tools/fsi_prompts.json tools/fsi_capture.py
git commit -m "feat: add FSI prompt set and safetensors capture script"
```

---

## Task 5: Pilot capture + separate per-family FSI test + suite-guard stage

**Files:**
- Create: `tests/fixtures/fsi/qwen3.6-35b-a3b.json` (capture output)
- Create: `tests/test_qwen_moe_36_fsi_selfcheck.cpp`
- Modify: `tests/CMakeLists.txt` (target + `inference` label list)
- Modify: `tests/test_model_suite_guard.cpp` (`fsi` stage + skip check)

**Interfaces:**
- Consumes: `fsi::LoadFsiGoldenFile`, `fsi::RunFsiSelfCheck` (Tasks 2-3), `tools/fsi_capture.py` (Task 4).
- Produces: pattern for Task 7; `"fsi"` suite stage.

- [ ] **Step 1: Capture the Qwen3.6-35B-A3B golden**

The source safetensors is always present in the vault:

```bash
mkdir -p tests/fixtures/fsi
python3 tools/fsi_capture.py \
  --model /mnt/llm_models/models/Qwen3.6-35B-A3B \
  --type qwen3.6-35b-a3b \
  --out tests/fixtures/fsi/qwen3.6-35b-a3b.json
```

Expected: `wrote tests/fixtures/fsi/qwen3.6-35b-a3b.json`. Then inspect it:

```bash
python3 -c "import json;d=json.load(open('tests/fixtures/fsi/qwen3.6-35b-a3b.json'));print(d['model_type'],[p['id'] for p in d['prompts']],[p['expected'] for p in d['prompts']])"
```

If capture reports an extracted/expected mismatch, STOP and investigate the raw
output (the reference is not accepted). Do not hand-edit the golden.

- [ ] **Step 2: Write the failing per-family test**

Create `tests/test_qwen_moe_36_fsi_selfcheck.cpp`:

```cpp
// FSISelfCheck — Qwen3.6-35B-A3B: verifies the converted model reproduces the
// golden answers captured from the source safetensors model. Separate
// functionality from SelfCheck; SKIPs without converted weights or golden.
#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <string>

#include <lsxcommon/fsi_selfcheck.h>
#include <lsxcommon/model.h>
#include <lsxcommon/model_factory.h>
#include <lsxcommon/path_registry.h>

#ifndef LSX_FSI_GOLDEN_DIR
#define LSX_FSI_GOLDEN_DIR "tests/fixtures/fsi"
#endif

namespace {

constexpr const char *kModelType = "qwen3.6-35b-a3b";

std::filesystem::path Qwen36ModelRoot()
{
    if (const char *env = std::getenv("LSX_TEST_QWEN36_MODEL_PATH"))
        return std::filesystem::path(env);
    const char *src = std::getenv("LSX_TEST_QWEN36_SOURCE_PATH");
    return src ? std::filesystem::path(src)
               : std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
                     "Qwen3.6-35B-A3B";
}

std::filesystem::path Qwen36GoldenPath()
{
    if (const char *env = std::getenv("LSX_FSI_GOLDEN_DIR"))
        return std::filesystem::path(env) /
               (std::string(kModelType) + ".json");
    return std::filesystem::path(LSX_FSI_GOLDEN_DIR) /
           (std::string(kModelType) + ".json");
}

} // namespace

TEST_CASE("Qwen3.6-35B-A3B FSI SelfCheck: golden answers match the converted "
          "model", "[qwen36][model][fsi][real-model]")
{
    const std::filesystem::path root = Qwen36ModelRoot();
    if (!std::filesystem::exists(root / "logestix" / "pinned"))
        SKIP("Qwen3.6-35B-A3B converted model directory not present");
    const std::filesystem::path golden_path = Qwen36GoldenPath();
    if (!std::filesystem::exists(golden_path))
        SKIP("FSI golden not captured for qwen3.6-35b-a3b");

    lsxcommon::fsi::FsiGolden golden;
    std::string err;
    REQUIRE(lsxcommon::fsi::LoadFsiGoldenFile(golden_path.string(), golden,
                                              &err));
    REQUIRE(golden.model_type == kModelType);

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = root.string();
    cfg.fast_math_mode = lsxcommon::FastMathMode::kOff;
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Type() == kModelType);
    REQUIRE(model->Load());

    const lsxcommon::fsi::FsiResult result =
        lsxcommon::fsi::RunFsiSelfCheck(*model, golden);
    for (const lsxcommon::fsi::FsiCaseResult &c : result.cases)
        INFO("case " << c.id << " expected='" << c.expected
                     << "' extracted='" << c.extracted << "' error='"
                     << c.error << "'");
    REQUIRE(result.all_passed);

    model->Unload();
}
```

- [ ] **Step 3: Register the target and label**

In `tests/CMakeLists.txt`, immediately after the `test_qwen_moe_36_selfcheck` block (search: `add_test(NAME test_qwen_moe_36_selfcheck`), add:

```cmake
file(TO_CMAKE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fsi" LSX_FSI_GOLDEN_DIR_FWD)
add_executable(test_qwen_moe_36_fsi_selfcheck test_qwen_moe_36_fsi_selfcheck.cpp)
target_compile_definitions(test_qwen_moe_36_fsi_selfcheck PRIVATE
    LSX_FSI_GOLDEN_DIR="${LSX_FSI_GOLDEN_DIR_FWD}")
target_link_libraries(test_qwen_moe_36_fsi_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
if(WIN32)
    add_custom_command(TARGET test_qwen_moe_36_fsi_selfcheck POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_qwen_moe_36_fsi_selfcheck>")
    add_custom_command(TARGET test_qwen_moe_36_fsi_selfcheck POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_qwen_moe_36_fsi_selfcheck>")
endif()
add_test(NAME test_qwen_moe_36_fsi_selfcheck COMMAND test_qwen_moe_36_fsi_selfcheck)
set_tests_properties(test_qwen_moe_36_fsi_selfcheck PROPERTIES SKIP_RETURN_CODE 4)
```

Add `test_qwen_moe_36_fsi_selfcheck` to the `set_tests_properties(... PROPERTIES LABELS inference)` list (the one containing `test_qwen_moe_36_selfcheck`).

- [ ] **Step 4: Register the `fsi` suite stage**

In `tests/test_model_suite_guard.cpp`:

1. In every row of `Suites()` that has a `{"selfcheck", {...}}` stage, add the
   corresponding FSI stage after it. For the qwen3.6 row (and Task 7 rows):

```cpp
          {"selfcheck", {"test_qwen_moe_36_selfcheck"}},
          {"fsi", {"test_qwen_moe_36_fsi_selfcheck"}}}},
```

2. Change the skip check to cover `fsi` too. Replace:

```cpp
            if (std::string(st.stage) != "selfcheck") continue;
```

with:

```cpp
            const std::string stage = st.stage;
            if (stage != "selfcheck" && stage != "fsi") continue;
```

Also update that test's `TEST_CASE` name to
`"suite guard: selfcheck and fsi targets declare SKIP_RETURN_CODE 4"` and the
`INFO` label to `"stage skip violations:\n"`.

- [ ] **Step 5: Build and run**

```bash
cmake --preset linux-debug
cmake --build --preset linux-debug --target test_qwen_moe_36_fsi_selfcheck test_model_suite_guard
ctest --preset linux-debug -R 'test_qwen_moe_36_fsi_selfcheck|test_model_suite_guard' --output-on-failure
```

Expected: suite guard PASS; FSI test PASS on the real model (or SKIP if the
converted model is absent). If FSI FAILs, capture the per-case `INFO`
expected/extracted and STOP — a mismatch is a real parity finding, not a test
bug to paper over.

- [ ] **Step 6: Commit**

```bash
git add tests/fixtures/fsi/qwen3.6-35b-a3b.json \
        tests/test_qwen_moe_36_fsi_selfcheck.cpp tests/CMakeLists.txt \
        tests/test_model_suite_guard.cpp
git commit -m "feat: pilot FSISelfCheck on Qwen3.6-35B-A3B with golden capture"
```

---

## Task 6: `lsxbenchmark --fsi-selfcheck` CLI

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/model_runtime.h`
- Modify: `libs/lsxcommon/src/model_runtime.cpp`
- Modify: `apps/lsxbenchmark/src/cli.h` and `apps/lsxbenchmark/src/cli.cpp`
- Modify: `apps/lsxbenchmark/src/main.cpp`

**Interfaces:**
- Consumes: `fsi::LoadFsiGoldenFile`, `fsi::RunFsiSelfCheck`, `CliModel` (existing in `model_runtime.cpp`).
- Produces: `int runtime::RunCliFsiSelfCheck(const GenerationCliArgs &);` and the `--fsi-selfcheck` flag.

- [ ] **Step 1: Inspect the benchmark CLI struct**

Open `apps/lsxbenchmark/src/cli.h` and note the `BenchmarkCliArgs` fields (`model_path`, `selfcheck`, `bit_exact`, `max_gen`, `prompt`). Add a `bool fsi_selfcheck = false;` member and parse `--fsi-selfcheck` in `cli.cpp` exactly like the existing `--selfcheck` branch.

- [ ] **Step 2: Declare and implement the lifecycle helper**

In `libs/lsxcommon/include/lsxcommon/model_runtime.h`, after `RunCliSelfCheck`:

```cpp
int RunCliFsiSelfCheck(const GenerationCliArgs &args);
```

In `libs/lsxcommon/src/model_runtime.cpp`, add `#include "lsxcommon/fsi_selfcheck.h"` at the top, then after `RunCliSelfCheck`:

```cpp
int RunCliFsiSelfCheck(const GenerationCliArgs &args)
{
    lsxcommon::log::info("Running FSI self-check (golden parity)...");
    const char *dir = std::getenv("LSX_FSI_GOLDEN_DIR");
    if (!dir || !*dir) {
        lsxcommon::log::error(
            "LSX_FSI_GOLDEN_DIR is not set (directory with <model_type>.json)");
        return 1;
    }
    CliModel m = CreateCliModel(args.model_path, FastMathMode::kOff);
    if (!m.ok) return 1;
    const std::string golden_path =
        std::string(dir) + "/" + m.model->Type() + ".json";
    fsi::FsiGolden golden;
    std::string err;
    if (!fsi::LoadFsiGoldenFile(golden_path, golden, &err)) {
        lsxcommon::log::error(absl::StrFormat("FSI golden load failed: %s",
                                              err.c_str()));
        m.model->Unload();
        return 1;
    }
    const fsi::FsiResult result = fsi::RunFsiSelfCheck(*m.model, golden);
    lsxcommon::log::info("=== lsxbenchmark: fsi-selfcheck ===");
    for (const fsi::FsiCaseResult &c : result.cases)
        lsxcommon::log::info(absl::StrFormat(
            "%s  expected='%s'  extracted='%s'  %s", c.id.c_str(),
            c.expected.c_str(), c.extracted.c_str(),
            c.passed ? "PASS" : "FAIL"));
    lsxcommon::log::info(absl::StrFormat(
        "result   : %s", result.all_passed ? "PASS" : "FAIL"));
    m.model->Unload();
    return result.all_passed ? 0 : 1;
}
```

- [ ] **Step 3: Wire the flag**

In `apps/lsxbenchmark/src/cli.h` add `bool fsi_selfcheck = false;`. In `cli.cpp` add:

```cpp
else if (std::strcmp(argv[i], "--fsi-selfcheck") == 0)
    cli.fsi_selfcheck = true;
```

In `apps/lsxbenchmark/src/main.cpp`, extend the dispatch:

```cpp
        args.max_gen_tokens = (cli.selfcheck || cli.fsi_selfcheck) ? 1 : cli.max_gen;
        args.fast_math_mode = cli.bit_exact ? lsxcommon::FastMathMode::kOff
                                            : lsxcommon::FastMathMode::kAuto;

        if (cli.fsi_selfcheck)
            rc = lsxcommon::runtime::RunCliFsiSelfCheck(args);
        else if (cli.selfcheck)
            rc = lsxcommon::runtime::RunCliSelfCheck(args);
        else
            rc = lsxcommon::runtime::RunCliBenchmark(args);
```

(`RunCliFsiSelfCheck` forces `kOff` internally, so `--bit-exact` is not needed.)

- [ ] **Step 4: Build and run**

```bash
cmake --build --preset linux-debug --target lsxbenchmark
LSX_FSI_GOLDEN_DIR="$PWD/tests/fixtures/fsi" \
  ./build/linux-debug/bin/lsxbenchmark --fsi-selfcheck \
  --model /mnt/llm_models/models/Qwen3.6-35B-A3B
```

Expected: exit 0 and per-case `PASS`. (Adjust the binary path to the one printed
by the build; use the actual output dir if different.)

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/model_runtime.h \
        libs/lsxcommon/src/model_runtime.cpp \
        apps/lsxbenchmark/src/cli.h apps/lsxbenchmark/src/cli.cpp \
        apps/lsxbenchmark/src/main.cpp
git commit -m "feat: add lsxbenchmark --fsi-selfcheck"
```

---

## Task 7: Roll out to the remaining families

**Files:** for each row below, create `tests/test_<name>.cpp` (Task 5 template),
add the CMake target block, add the name to the `LABELS inference` list, add the
`fsi` stage to `Suites()`, and capture the golden.

**Interfaces:** identical to Task 5, parameterized by the table below.

The per-family test body is exactly the Task 5 template with three constants
substituted: `kModelType`, the model-root helper's env var + default directory,
and the `TEST_CASE` tag. Copy the existing vault-path helper shape from the
family's `test_<family>_selfcheck.cpp` (the default directory is authoritative)
but always resolve the default via `lsxcommon::lsx_model_models_dir()`.

| model_type (`kModelType`) | vault dir (default) | override env var | test file | ctest target |
|---|---|---|---|---|
| `glm-5.2` | `GLM-5.2` | `LSX_TEST_GLM_MODEL_PATH` | `tests/test_glm_moe_fsi_selfcheck.cpp` | `test_glm_moe_fsi_selfcheck` |
| `glm-5.3-flash` | `GLM-5.3-Flash` | `LSX_TEST_GLM_FLASH_ROOT` | `tests/test_glm_moe_flash_fsi_selfcheck.cpp` | `test_glm_moe_flash_fsi_selfcheck` |
| `deepseek-v4-flash` | `DeepSeek-V4-Flash` | `LSX_TEST_DEEPSEEK_MODEL_PATH` | `tests/test_deepseek_moe_fsi_selfcheck.cpp` | `test_deepseek_moe_fsi_selfcheck` |
| `deepseek-v4.1-flash` | `DeepSeek-V4.1-Flash` | `LSX_TEST_DEEPSEEK_V41_ROOT` | `tests/test_deepseek_moe_v41_fsi_selfcheck.cpp` | `test_deepseek_moe_v41_fsi_selfcheck` |
| `qwen3.5-9b` | `Qwen3.5-9B` | `LSX_TEST_QWEN_MODEL_PATH` | `tests/test_qwen_dense_fsi_selfcheck.cpp` | `test_qwen_dense_fsi_selfcheck` |
| `qwen3.8-27b` | `Qwen3.8-27B` | `LSX_TEST_QWEN27_MODEL_PATH` | `tests/test_qwen_dense_27b_fsi_selfcheck.cpp` | `test_qwen_dense_27b_fsi_selfcheck` |
| `qwen3.5-35b-a3b` | `Qwen3.5-35B-A3B` | `LSX_TEST_QWEN_MOE_MODEL_PATH` | `tests/test_qwen_moe_fsi_selfcheck.cpp` | `test_qwen_moe_fsi_selfcheck` |
| `qwen3.8-flash-next` | `Qwen3.8-Flash-Next` | `LSX_TEST_QWEN38_MODEL_DIR` | `tests/test_qwen_moe_exp_fsi_selfcheck.cpp` | `test_qwen_moe_exp_fsi_selfcheck` |
| `gemma-4-26b-a4b-it` | `gemma-4-26B-A4B-it` | `LSX_TEST_GEMMA_MODEL_PATH` | `tests/test_gemma_moe_fsi_selfcheck.cpp` | `test_gemma_moe_fsi_selfcheck` |
| `kimi-k2.7-code` | `Kimi-K2.7-Code` | `LSX_TEST_KIMI_MODEL_PATH` | `tests/test_kimi_moe_fsi_selfcheck.cpp` | `test_kimi_moe_fsi_selfcheck` |
| `nemotron-3.5-lightning-30b-a3b` | `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` | `LSX_TEST_NEMOTRON_ROOT` | `tests/test_nemotron_moe_fsi_selfcheck.cpp` | `test_nemotron_moe_fsi_selfcheck` |
| `gigachat3.5-432b-a28b-reasoning` | `GigaChat3.5-432B-A28B-Reasoning` | `LSX_TEST_GIGACHAT_ROOT` | `tests/test_gigachat_moe_fsi_selfcheck.cpp` | `test_gigachat_moe_fsi_selfcheck` |
| `ornith-1.5-35b-a3b` | `Ornith-1.5-35B-A3B-MLX` | `LSX_TEST_ORNITH_MODEL_PATH` | `tests/test_ornith_moe_fsi_selfcheck.cpp` | `test_ornith_moe_fsi_selfcheck` |
| `granite-4.2-30b` | `granite-4.2-30b` | `LSX_TEST_GRANITE_MODEL_PATH` | `tests/test_granite_dense_fsi_selfcheck.cpp` | `test_granite_dense_fsi_selfcheck` |

(Qwen3.6 is the pilot from Task 5; not repeated here.)

- [ ] **Step 1: For each row, capture the golden**

```bash
python3 tools/fsi_capture.py --model /mnt/llm_models/models/<vault dir> \
    --type <model_type> --out tests/fixtures/fsi/<model_type>.json
```

If the generic HF path cannot load a family (e.g. MLX export), add a row to
`OVERRIDES` in `tools/fsi_capture.py` (e.g. a `chat_template`/`trust_remote_code`
or dtype choice) and retry. If a family's reference genuinely diverges from the
template, STOP and consult the spec (the reference is not accepted).

- [ ] **Step 2: For each row, create the test + CMake target + label + suite stage**

Create the test file from the Task 5 template with the row's constants, add the
literal CMake block, add the target name to the `LABELS inference` list, and add
`{"fsi", {"<ctest target>"}}` after the family's `selfcheck` stage in
`test_model_suite_guard.cpp::Suites()`.

- [ ] **Step 3: Build, run the whole FSI set, and the guards**

```bash
cmake --preset linux-debug
cmake --build --preset linux-debug
ctest --preset linux-debug -R 'fsi_selfcheck|_guard' --output-on-failure
```

Expected: host unit test PASS; each real-model FSI test PASS or SKIP (absent
converted weights/golden); all guards PASS (naming, labels, suite).

- [ ] **Step 4: Commit (one commit per family or a batch of families)**

```bash
git add tests/test_*_fsi_selfcheck.cpp tests/fixtures/fsi/<model_type>.json \
        tests/CMakeLists.txt tests/test_model_suite_guard.cpp
git commit -m "test: add FSISelfCheck for <family>"
```

---

## Task 8: Documentation + final verification

**Files:**
- Modify: `AGENTS.md` (Structure + Test/Resources pointers)
- Modify: `docs/superpowers/specs/2026-09-26-fsi-selfcheck-design.md` if any
  as-built delta exists.

- [ ] **Step 1: Update `AGENTS.md`**

Add to the Structure section, near the SelfCheck mention:

```
FSISelfCheck (`libs/lsxcommon/{include,src}/fsi_selfcheck.{h,cpp}`, model-neutral
host module; capture `tools/fsi_capture.py` + `tools/fsi_prompts.json`; goldens
`tests/fixtures/fsi/`; per-family `test_<family>_fsi_selfcheck.cpp`, label
inference, SKIP 4) — a separate parity gate from SelfCheck: compares the
converted model's extracted answer to a golden captured from the source
safetensors model at `kOff`; unrelated to `IModel::SelfCheck`.
```

- [ ] **Step 2: Full verification**

```bash
cmake --build --preset linux-debug
ctest --preset linux-debug                                    # fast suite
ctest --preset linux-debug -R 'fsi_selfcheck' --output-on-failure
ctest --preset linux-debug -R '_guard' --output-on-failure
```

Expected: all green (FSI real-model tests may SKIP, which is success).

- [ ] **Step 3: Confirm decoupling (spec §3.5)**

```bash
git diff main --name-only | rg 'selfcheck' || true
rg -n "SelfCheck|selfcheck_passed|LSX_SELFCHECK_EXPECT_HASH|RunParisSelfCheck" \
   libs/lsxcommon/src/fsi_selfcheck.cpp libs/lsxcommon/include/lsxcommon/fsi_selfcheck.h \
   tests/test_*_fsi_selfcheck.cpp
```

Expected: the FSI module and FSI tests contain no SelfCheck symbol; the only
diff to existing SelfCheck files is none (they are untouched).

- [ ] **Step 4: Commit**

```bash
git add AGENTS.md docs/superpowers/specs/2026-09-26-fsi-selfcheck-design.md
git commit -m "docs: record FSISelfCheck in AGENTS and spec"
```

---

## Self-Review (author checklist)

- **Spec coverage:** §3.1 prompts (Task 4), §3.2 capture (Task 4), §3.3 verify (Task 3/5), §3.4 GPU independence (Task 3/5 `kOff`, decoder seam), §4 extraction (Task 1), §5 golden (Task 2), §6 API (Tasks 1-3) + CLI (Task 6), §7 Python (Task 4), §8 tests/guards (Tasks 1-5, 7-8), §3.5 decoupling (Task 8 step 3). All covered.
- **Placeholders:** none — every code step is complete; the Task 7 table supplies all per-family constants and the template is fully specified in Task 5.
- **Type consistency:** `FsiKind`/`FsiGolden`/`FsiGoldenCase`/`FsiCaseResult`/`FsiResult`/`FsiDecodeFn` and the three function signatures are identical across Tasks 1-3, 5-6.
