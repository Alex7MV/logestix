# DETSelfCheck Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **logestix override:** the repository's `AGENTS.md` FORBIDS Task subagents
> ("No parallel/subagents ... All codebase research, audits and edits are done by
> the main session directly"). Execute this plan **inline** with
> `executing-plans`; ignore the subagent-driven recommendation.

**Goal:** Add DETSelfCheck — a model-neutral gate that runs a fixed near-tie prompt set R times, requires byte-identical generated token streams across runs, and requires the extracted canonical token to match a per-model golden.

**Architecture:** A new model-neutral `lsxcommon::det` host-only surface (`det_selfcheck.{h,cpp}`) parallel to `fsi`/`code`. Per-family ctest targets call `RunDetSelfCheck` on the converted NVFP4 model at `FastMathMode::kOff`. A new `det` lifecycle stage in `test_model_suite_guard.cpp` covers all 15 families. A Python tool captures goldens from source safetensors.

**Tech Stack:** C++20, Catch2 v3.16, simdjson, abseil (`StrFormat`), CMake/Ninja presets, Python3 + HF transformers for capture. No new CUDA kernels.

## Global Constraints

- Language C++20; host-only DET files are NOT compiled by NVCC.
- Logging only via spdlog (`lsxcommon::log`); no `std::cout`/`printf`.
- Host C++ prefers `absl::`: `absl::StrFormat` is printf-style (`%s`), `{}` is literal.
- Headers reachable from `.cu`/`.cuh` must not include `<absl/container/*>`; `det_selfcheck.h` is host-only and includes none.
- New model-neutral files have NO family stem: `det_selfcheck.h` / `det_selfcheck.cpp`.
- Every per-family DET test: label `inference`, `SKIP_RETURN_CODE 4`, model created with `fast_math_mode = FastMathMode::kOff`.
- Suite/label/guard tests must be extended in the SAME commit as the targets.
- No CLI speed/perf flags; only `--det-selfcheck` is added.
- Commit only when the operator explicitly requests it (AGENTS.md).

---

### Task 1: DET extraction core (`ExtractDetAnswer`)

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/det_selfcheck.h`
- Create: `libs/lsxcommon/src/det_selfcheck.cpp`
- Create: `tests/test_det_selfcheck.cpp`
- Modify: `tests/CMakeLists.txt` (host unit test target + `fast` label list)

**Interfaces:**
- Consumes: nothing.
- Produces: `lsxcommon::det::DetKind`, `DetGoldenCase`, `DetGolden`, `DetCaseResult`, `DetResult`, `ExtractDetAnswer(DetKind, std::string_view)`, `ParseDetGolden`, `LoadDetGoldenFile`, `DetDecodeFn`, `RunDetSelfCheck`.

- [ ] **Step 1: Create the header with all public declarations**

`libs/lsxcommon/include/lsxcommon/det_selfcheck.h`:

```cpp
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "lsxcommon/model.h"

namespace lsxcommon {
namespace det {

enum class DetKind { kToken };

struct DetGoldenCase {
    std::string id;
    DetKind kind = DetKind::kToken;
    std::vector<int32_t> input_ids;
    int max_new_tokens = 64;
    bool ignore_eos = false;   // run the full token budget (no EOS stop)
    int stress_runs = 3;       // determinism repetition count (floored at 3)
    std::string expected;
    std::string raw;           // diagnostic
    std::string body;          // optional prompt body for the echo cut
};

struct DetGolden {
    std::string model_type;
    std::vector<DetGoldenCase> cases;
};

struct DetCaseResult {
    std::string id;
    std::string expected;
    std::string extracted;
    std::string raw;             // decoded run-0 text (diagnostic)
    int runs = 0;                // effective R
    bool deterministic = false;  // all runs' gen_tokens identical
    bool passed = false;
    std::string error;
};

struct DetResult {
    std::vector<DetCaseResult> cases;
    bool all_passed = false;
    std::string error;
};

std::string ExtractDetAnswer(DetKind kind, std::string_view raw);

bool ParseDetGolden(std::string_view json, DetGolden &out, std::string *error);
bool LoadDetGoldenFile(const std::string &path, DetGolden &out,
                       std::string *error);

using DetDecodeFn = std::function<std::string(const std::vector<int32_t> &)>;

DetResult RunDetSelfCheck(IModel &model, const DetGolden &golden);
DetResult RunDetSelfCheck(IModel &model, const DetGolden &golden,
                          const DetDecodeFn &decode,
                          int stress_runs_override = 0);

}  // namespace det
}  // namespace lsxcommon
```

- [ ] **Step 2: Write the failing extraction tests**

`tests/test_det_selfcheck.cpp`:

```cpp
// DETSelfCheck host-only unit tests (extraction/golden/harness). No GPU.
#include <catch2/catch_test_macros.hpp>

#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <lsxcommon/det_selfcheck.h>
#include <lsxcommon/inference_gpu.h>
#include <lsxcommon/safetensors.h>

namespace det = lsxcommon::det;

TEST_CASE("det: token extraction picks the first standalone uppercase token",
          "[det][unit]")
{
    CHECK(det::ExtractDetAnswer(det::DetKind::kToken, "PARADOX_ANTINOMY") ==
          "PARADOX_ANTINOMY");
    CHECK(det::ExtractDetAnswer(det::DetKind::kToken,
                                "The answer is PARADOX_ANTINOMY.") ==
          "PARADOX_ANTINOMY");
    CHECK(det::ExtractDetAnswer(det::DetKind::kToken,
                                "```\nCATASTROPHIC\n```") == "CATASTROPHIC");
    CHECK(det::ExtractDetAnswer(det::DetKind::kToken,
                                "thinking... BIT_ZERO maybe\n</think>\nBIT_ZERO") ==
          "BIT_ZERO");
    CHECK(det::ExtractDetAnswer(det::DetKind::kToken,
                                "**PARADOX_ANTINOMY**") == "PARADOX_ANTINOMY");
}

TEST_CASE("det: token extraction ignores lowercase and short tokens", "[det][unit]")
{
    CHECK(det::ExtractDetAnswer(det::DetKind::kToken, "the answer is a word")
              .empty());
    CHECK(det::ExtractDetAnswer(det::DetKind::kToken, "").empty());
    CHECK(det::ExtractDetAnswer(det::DetKind::kToken, "A").empty());
    // "DET" is a standalone uppercase run -> it is a valid token by contract.
    CHECK(det::ExtractDetAnswer(det::DetKind::kToken, "DET-TEST-01") == "DET");
}
```

- [ ] **Step 3: Register the test target and label it `fast`**

In `tests/CMakeLists.txt`, append near the FSISelfCheck host unit test block (after line ~4437):

```cmake
# ── DETSelfCheck: host-only extraction/golden/harness unit tests (no GPU) ──
add_executable(test_det_selfcheck test_det_selfcheck.cpp)
target_link_libraries(test_det_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
add_test(NAME test_det_selfcheck COMMAND test_det_selfcheck)
```

Add `test_det_selfcheck` to the `PROPERTIES LABELS fast` name list (before line 4500, e.g. after `test_code_selfcheck`):

```cmake
    test_code_selfcheck
    test_det_selfcheck
```

Also add `lsxcommon/src/det_selfcheck.cpp` to the `lsxcommon` source list. Find the file:
`grep -n "code_selfcheck.cpp" libs/lsxcommon/CMakeLists.txt` and add
`src/det_selfcheck.cpp` immediately after `src/code_selfcheck.cpp`.

- [ ] **Step 4: Run the tests to verify they fail**

Run: `cmake --build --preset linux-debug --target test_det_selfcheck`
Expected: link failure — `undefined reference to lsxcommon::det::ExtractDetAnswer`.

- [ ] **Step 5: Implement extraction in `det_selfcheck.cpp`**

`libs/lsxcommon/src/det_selfcheck.cpp` (initial content; later tasks append):

```cpp
#include "lsxcommon/det_selfcheck.h"

#include <absl/strings/ascii.h>
#include <absl/strings/str_format.h>
#include <simdjson.h>
#include <cctype>
#include <fstream>
#include <sstream>

#include "lsxcommon/log.h"
#include "lsxcommon/model_runtime.h"

namespace lsxcommon {
namespace det {
namespace {

std::string Unfence(std::string_view raw)
{
    std::string s(absl::StripAsciiWhitespace(raw));
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
    return std::string(absl::StripAsciiWhitespace(inner));
}

// Strip everything before the last </think> so a reasoning chain does not
// shadow the final answer (same rule as FSI/CODE).
std::string Preprocess(std::string_view raw)
{
    std::string text = Unfence(raw);
    const std::size_t close = text.rfind("</think>");
    if (close != std::string::npos) text = text.substr(close + 8);
    return text;
}

bool IsWordChar(char c)
{
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

// First standalone token matching [A-Z][A-Z0-9_]* with length >= 2.
std::string ExtractToken(std::string_view text)
{
    const std::size_t n = text.size();
    for (std::size_t i = 0; i < n; ++i) {
        const char c = text[i];
        if (c < 'A' || c > 'Z') continue;
        if (i > 0 && IsWordChar(text[i - 1])) continue;
        std::size_t j = i + 1;
        while (j < n) {
            const char d = text[j];
            const bool ok = (d >= 'A' && d <= 'Z') ||
                            (d >= '0' && d <= '9') || d == '_';
            if (!ok) break;
            ++j;
        }
        if (j - i < 2) { i = j; continue; }
        if (j < n && IsWordChar(text[j])) { i = j; continue; }
        return std::string(text.substr(i, j - i));
    }
    return {};
}

} // namespace

std::string ExtractDetAnswer(DetKind kind, std::string_view raw)
{
    const std::string text = Preprocess(raw);
    switch (kind) {
        case DetKind::kToken: return ExtractToken(text);
    }
    return {};
}

}  // namespace det
}  // namespace lsxcommon
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cmake --build --preset linux-debug --target test_det_selfcheck && ./build/linux-debug/tests/test_det_selfcheck`
Expected: all `[det][unit]` extraction assertions PASS.

- [ ] **Step 7: Commit (only if the operator has asked for commits)**

```bash
git add libs/lsxcommon/include/lsxcommon/det_selfcheck.h \
        libs/lsxcommon/src/det_selfcheck.cpp \
        libs/lsxcommon/CMakeLists.txt tests/test_det_selfcheck.cpp tests/CMakeLists.txt
git commit -m "feat: add DETSelfCheck extraction core"
```

---

### Task 2: DET golden parsing

**Files:**
- Modify: `libs/lsxcommon/src/det_selfcheck.cpp` (append `ParseDetGolden`, `LoadDetGoldenFile`)
- Modify: `tests/test_det_selfcheck.cpp` (append golden tests)

**Interfaces:**
- Consumes: `DetGolden`, `DetGoldenCase`, `DetKind` (Task 1).
- Produces: `ParseDetGolden(std::string_view, DetGolden&, std::string*)`, `LoadDetGoldenFile(const std::string&, DetGolden&, std::string*)`.

- [ ] **Step 1: Write the failing golden tests**

Append to `tests/test_det_selfcheck.cpp`:

```cpp
namespace {

const char *kGoldenJson = R"JSON({
  "format": "logestix-det-golden",
  "version": 1,
  "model_type": "fake-model",
  "source": "fake",
  "runtime": "test",
  "prompts": [
    {
      "id": "DET-TEST-01",
      "kind": "token",
      "input_ids": [1, 2, 3],
      "max_new_tokens": 32,
      "stress_runs": 4,
      "body": "[CONTEXT]: DET-Validation-Core-01",
      "expected": "PARADOX_ANTINOMY",
      "raw": "PARADOX_ANTINOMY"
    },
    {
      "id": "DET-TEST-03",
      "kind": "token",
      "input_ids": [9],
      "expected": "BIT_ZERO"
    }
  ]
})JSON";

} // namespace

TEST_CASE("det: golden JSON parses all fields and stress_runs default",
          "[det][unit]")
{
    det::DetGolden g;
    std::string err;
    REQUIRE(det::ParseDetGolden(kGoldenJson, g, &err));
    CHECK(err.empty());
    CHECK(g.model_type == "fake-model");
    REQUIRE(g.cases.size() == 2);
    CHECK(g.cases[0].id == "DET-TEST-01");
    CHECK(g.cases[0].kind == det::DetKind::kToken);
    CHECK(g.cases[0].input_ids == std::vector<int32_t>{1, 2, 3});
    CHECK(g.cases[0].max_new_tokens == 32);
    CHECK(g.cases[0].stress_runs == 4);
    CHECK(g.cases[0].body == "[CONTEXT]: DET-Validation-Core-01");
    CHECK(g.cases[0].expected == "PARADOX_ANTINOMY");
    CHECK(g.cases[1].stress_runs == 3);  // absent -> default
    CHECK(g.cases[1].max_new_tokens == 64);
}

TEST_CASE("det: malformed golden is rejected with an error", "[det][unit]")
{
    det::DetGolden g;
    std::string err;
    CHECK_FALSE(det::ParseDetGolden("{}", g, &err));
    CHECK_FALSE(err.empty());
    CHECK_FALSE(det::ParseDetGolden(
        R"({"format":"logestix-det-golden","model_type":"m","prompts":[]})",
        g, &err));
    CHECK_FALSE(det::ParseDetGolden(
        R"({"format":"logestix-det-golden","model_type":"m","prompts":[
            {"id":"x","kind":"nope","input_ids":[1],"expected":"T"}]})",
        g, &err));
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cmake --build --preset linux-debug --target test_det_selfcheck`
Expected: link failure — `undefined reference to ParseDetGolden`.

- [ ] **Step 3: Implement parsing**

Insert the following before the final `}  // namespace det` in
`libs/lsxcommon/src/det_selfcheck.cpp` (after `ExtractDetAnswer`):

```cpp
bool ParseDetGolden(std::string_view json, DetGolden &out, std::string *error)
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
        tag != "logestix-det-golden") {
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
        DetGoldenCase c;
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
        if (sv == "token") c.kind = DetKind::kToken;
        else { if (error) *error = c.id + ": unknown kind"; return false; }
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
        bool ign = false;
        if (o["ignore_eos"].get_bool().get(ign) == simdjson::SUCCESS)
            c.ignore_eos = ign;
        int64_t sr = 0;
        if (o["stress_runs"].get_int64().get(sr) == simdjson::SUCCESS) {
            if (sr < 1) {
                if (error) *error = c.id + ": stress_runs must be >= 1";
                return false;
            }
            c.stress_runs = static_cast<int>(sr);
        }
        if (o["expected"].get_string().get(sv) != simdjson::SUCCESS ||
            sv.empty()) {
            if (error) *error = c.id + ": missing expected";
            return false;
        }
        c.expected = std::string(sv);
        std::string_view raw;
        if (o["raw"].get_string().get(raw) == simdjson::SUCCESS)
            c.raw = std::string(raw);
        std::string_view body;
        if (o["body"].get_string().get(body) == simdjson::SUCCESS)
            c.body = std::string(body);
        out.cases.push_back(std::move(c));
    }
    if (out.cases.empty()) {
        if (error) *error = "no prompts in golden";
        return false;
    }
    return true;
}

bool LoadDetGoldenFile(const std::string &path, DetGolden &out,
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
    if (!ParseDetGolden(text, out, &local)) {
        if (error) *error = path + ": " + local;
        return false;
    }
    return true;
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cmake --build --preset linux-debug --target test_det_selfcheck && ./build/linux-debug/tests/test_det_selfcheck`
Expected: all golden parse assertions PASS.

- [ ] **Step 5: Commit (operator-gated)**

```bash
git add libs/lsxcommon/src/det_selfcheck.cpp tests/test_det_selfcheck.cpp
git commit -m "feat: parse DET golden"
```

---

### Task 3: DET harness (`RunDetSelfCheck` determinism)

**Files:**
- Modify: `libs/lsxcommon/src/det_selfcheck.cpp` (append echo cut, `RunDetCase`, `RunDetSelfCheck`)
- Modify: `tests/test_det_selfcheck.cpp` (fake-model tests)

**Interfaces:**
- Consumes: `DetGolden`, `DetCaseResult`, `DetResult`, `DetDecodeFn`, `ExtractDetAnswer`.
- Produces: `RunDetSelfCheck(model, golden)`, `RunDetSelfCheck(model, golden, decode, stress_runs_override=0)`.

- [ ] **Step 1: Write the failing harness tests**

Append to `tests/test_det_selfcheck.cpp`:

```cpp
namespace {

class FakeDetModel : public lsxcommon::IModel {
public:
    bool Load() override { return true; }
    void Unload() override {}
    lsxcommon::SafetensorsModel Inspect() override { return {}; }
    lsxcommon::gpu::InferenceResult Infer(
        const lsxcommon::ModelRequest &req) override
    {
        last_ = req;
        ++calls;
        if (fail) {
            lsxcommon::gpu::InferenceResult r;
            r.ok = false;
            r.error_msg = "boom";
            return r;
        }
        lsxcommon::gpu::InferenceResult r;
        r.ok = true;
        const std::size_t idx = sequence.empty()
                                    ? 0
                                    : std::min(calls - 1, sequence.size() - 1);
        r.gen_tokens = sequence.empty() ? std::vector<int32_t>{1}
                                        : sequence[idx];
        return r;
    }
    lsxcommon::gpu::InferenceResult SelfCheck(
        const lsxcommon::ModelRequest &) override
    {
        throw std::runtime_error("DET must not call SelfCheck");
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

    std::vector<std::vector<int32_t>> sequence;
    bool fail = false;
    int calls = 0;
    lsxcommon::ModelRequest last_;
private:
    std::shared_mutex m_;
};

det::DetGolden OneCaseGolden(int stress_runs = 3)
{
    det::DetGolden g;
    det::DetGoldenCase c;
    c.id = "DET-TEST-01";
    c.kind = det::DetKind::kToken;
    c.input_ids = {1, 2};
    c.max_new_tokens = 8;
    c.stress_runs = stress_runs;
    c.expected = "PARADOX_ANTINOMY";
    g.cases.push_back(std::move(c));
    return g;
}

std::string DecodeOne(const std::vector<int32_t> &) { return "PARADOX_ANTINOMY"; }

} // namespace

TEST_CASE("det: harness passes when runs are identical and answer matches",
          "[det][unit]")
{
    FakeDetModel m;
    const det::DetGolden g = OneCaseGolden();
    const det::DetResult r = det::RunDetSelfCheck(m, g, DecodeOne);
    REQUIRE(r.all_passed);
    REQUIRE(r.cases.size() == 1);
    CHECK(r.cases[0].deterministic);
    CHECK(r.cases[0].runs == 3);
    CHECK(r.cases[0].extracted == "PARADOX_ANTINOMY");
    CHECK(m.last_.input_ids == std::vector<int32_t>{1, 2});
    CHECK(m.last_.max_gen_tokens == 8);
    CHECK(m.calls == 3);
}

TEST_CASE("det: harness fails when a run diverges", "[det][unit]")
{
    FakeDetModel m;
    m.sequence = {{1}, {1}, {2}};
    const det::DetResult r = det::RunDetSelfCheck(m, OneCaseGolden(), DecodeOne);
    CHECK_FALSE(r.all_passed);
    REQUIRE(r.cases.size() == 1);
    CHECK_FALSE(r.cases[0].deterministic);
}

TEST_CASE("det: harness fails on an answer mismatch", "[det][unit]")
{
    FakeDetModel m;
    const auto decode = [](const std::vector<int32_t> &) {
        return std::string("CATASTROPHIC");
    };
    const det::DetResult r = det::RunDetSelfCheck(m, OneCaseGolden(), decode);
    CHECK_FALSE(r.all_passed);
    CHECK(r.cases[0].deterministic);
    CHECK(r.cases[0].extracted == "CATASTROPHIC");
}

TEST_CASE("det: harness propagates inference errors", "[det][unit]")
{
    FakeDetModel m;
    m.fail = true;
    const det::DetResult r = det::RunDetSelfCheck(m, OneCaseGolden(), DecodeOne);
    CHECK_FALSE(r.all_passed);
    REQUIRE(r.cases.size() == 1);
    CHECK(r.cases[0].error == "boom");
}

TEST_CASE("det: stress_runs override is floored at 3", "[det][unit]")
{
    FakeDetModel m;
    const det::DetResult r =
        det::RunDetSelfCheck(m, OneCaseGolden(1), DecodeOne, 1);
    CHECK(r.cases[0].runs == 3);
    CHECK(m.calls == 3);
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cmake --build --preset linux-debug --target test_det_selfcheck`
Expected: link failure — `undefined reference to RunDetSelfCheck`.

- [ ] **Step 3: Implement the harness**

Insert before the final `}  // namespace det` in `det_selfcheck.cpp` (after
`LoadDetGoldenFile`), noting `std::min` needs `<algorithm>` (add the include):

```cpp
namespace {

// A fragile reasoning model can fail to emit a turn-end token and echo the
// prompt after its answer. When the golden carries the prompt body, cut the
// decoded text at the body's prefix (after the reasoning close) so the echo
// does not bleed into the extracted answer.
std::string CutPromptEcho(std::string_view text, std::string_view body)
{
    if (body.empty()) return std::string(text);
    std::string s(text);
    const std::size_t close = s.rfind("</think>");
    if (close != std::string::npos) s = s.substr(close + 8);
    const std::size_t n = std::min<std::size_t>(32, body.size());
    const std::size_t pos = s.find(std::string_view(body).substr(0, n));
    if (pos != std::string::npos) s = s.substr(0, pos);
    return s;
}

DetCaseResult RunDetCase(IModel &model, const DetGoldenCase &c,
                         const DetDecodeFn &decode, int stress_runs_override)
{
    DetCaseResult r;
    r.id = c.id;
    r.expected = c.expected;
    int runs = (stress_runs_override > 0) ? stress_runs_override
                                          : c.stress_runs;
    if (runs < 3) runs = 3;
    r.runs = runs;

    ModelRequest req;
    req.input_ids = c.input_ids;
    req.max_gen_tokens = c.max_new_tokens;
    req.ignore_eos = c.ignore_eos;

    std::vector<int32_t> first;
    bool identical = true;
    for (int i = 0; i < runs; ++i) {
        const gpu::InferenceResult ir = model.Infer(req);
        if (!ir.ok) {
            r.error = ir.error_msg.empty() ? "inference failed" : ir.error_msg;
            return r;
        }
        if (i == 0) first = ir.gen_tokens;
        else if (ir.gen_tokens != first) identical = false;
    }
    r.deterministic = identical;
    const std::string text = decode(first);
    r.raw = text;
    r.extracted =
        ExtractDetAnswer(c.kind, CutPromptEcho(text, c.body));
    r.passed = r.deterministic && (r.extracted == c.expected);
    return r;
}

} // namespace

DetResult RunDetSelfCheck(IModel &model, const DetGolden &golden,
                          const DetDecodeFn &decode, int stress_runs_override)
{
    DetResult res;
    for (const DetGoldenCase &c : golden.cases)
        res.cases.push_back(
            RunDetCase(model, c, decode, stress_runs_override));
    res.all_passed = !res.cases.empty();
    for (const DetCaseResult &c : res.cases) {
        lsxcommon::log::info(absl::StrFormat(
            "[det] %s runs=%d deterministic=%s expected='%s' extracted='%s' "
            "passed=%s raw='%s'",
            c.id.c_str(), c.runs, c.deterministic ? "true" : "false",
            c.expected.c_str(), c.extracted.c_str(),
            c.passed ? "true" : "false", c.raw.c_str()));
        if (!c.passed) res.all_passed = false;
    }
    return res;
}

DetResult RunDetSelfCheck(IModel &model, const DetGolden &golden)
{
    return RunDetSelfCheck(
        model, golden,
        [&model](const std::vector<int32_t> &ids) {
            return runtime::DecodeGeneratedText(model.Tokenizer(), ids);
        });
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cmake --build --preset linux-debug --target test_det_selfcheck && ./build/linux-debug/tests/test_det_selfcheck`
Expected: all `[det][unit]` assertions PASS.

- [ ] **Step 5: Commit (operator-gated)**

```bash
git add libs/lsxcommon/src/det_selfcheck.cpp tests/test_det_selfcheck.cpp
git commit -m "feat: add DET determinism harness"
```

---

### Task 4: Prompt set + Python capture tool

**Files:**
- Create: `tools/det_prompts.json`
- Create: `tools/det_capture.py`

**Interfaces:**
- Produces: `tools/det_capture.py --selftest` (exit 0), and golden JSON files under `tests/fixtures/det/`.

- [ ] **Step 1: Write `tools/det_prompts.json`**

```json
{
  "format": "logestix-det-prompts",
  "version": 1,
  "prompts": [
    {
      "id": "DET-TEST-01",
      "title": "PARADOXICAL-LOGIC-ANCHOR",
      "kind": "token",
      "expected": "PARADOX_ANTINOMY",
      "body": "[CONTEXT]: DET-Validation-Core-01 // Self-Referential Resolution\nResolve the standard Epimenides boundary paradox under absolute zero-variance constraints.\nStatement: \"This explicit statement is completely false.\"\nTask: Determine the structural truth-value behavior condition of the statement above.\nOutput Specification: Output strictly one capitalized token: PARADOX_ANTINOMY. Do not evaluate textually, do not include whitespace or formatting tags.\nExpected Template: PARADOX_ANTINOMY"
    },
    {
      "id": "DET-TEST-02",
      "title": "SYNONYM-COLLAPSE-PRESSURE",
      "kind": "token",
      "expected": "CATASTROPHIC",
      "body": "[CONTEXT]: DET-Validation-Core-02 // Lexical Near-Tie Compression\nSelect the absolute premier, immutable system-level classification noun to describe an unrecoverable hardware state.\nCandidates: [FATAL, CRITICAL, TERMINAL, CATASTROPHIC]\nTask: Isolate the final vocabulary token strictly according to the highest alphabetical sort priority to suppress semantic drift.\nOutput Specification: Output ONLY the single chosen word from the candidates array without spaces, punctuation, or code gates.\nExpected Template: CATASTROPHIC"
    },
    {
      "id": "DET-TEST-03",
      "title": "QUANTUM-STATE-DETERMINISM",
      "kind": "token",
      "expected": "BIT_ZERO",
      "body": "[CONTEXT]: DET-Validation-Core-03 // Superposition Token Clamp\nEvaluate a simulated quantum superposition register state where Probability(State_0) perfectly equals Probability(State_1).\nConstraint: Internal random seed initialization is blocked; system execution mode is hard-max greedy.\nTask: Force the output sampler to resolve the symmetrical tie by emitting the strict primary hardware default register bit identifier.\nOutput Specification: Output exactly one uppercase character token string. Stop generation immediately.\nExpected Template: BIT_ZERO"
    }
  ]
}
```

- [ ] **Step 2: Write `tools/det_capture.py`**

Mirror `tools/code_capture.py` exactly in structure; the extraction mirror for
`kind == "token"` is a Python port of `ExtractToken`. Full file:

```python
#!/usr/bin/env python3
"""DETSelfCheck reference capture (determinism + near-tie gate).

Captures the DET golden answers from an original safetensors model with HF
transformers. The logestix side only verifies the converted model against the
committed golden (libs/lsxcommon/src/det_selfcheck.cpp). Run the self-test with
no model present to validate the prompt set and the extraction mirror.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = os.path.join(HERE, "det_prompts.json")

# Per-model capture overrides keyed by model type. Absent -> generic defaults.
OVERRIDES = {
    "granite-4.2-30b": {"enable_thinking": True, "low_effort": True},
    "qwen3.8-flash-next": {"loader": "qwen4_exp"},
}
GENERIC = {"trust_remote_code": True, "max_new_tokens": 64,
           "device": None, "dtype": None, "enable_thinking": False,
           "device_map": None, "loader": None}

_STRESS_RUNS = 3


def load_prompts():
    with open(PROMPTS, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    assert doc["format"] == "logestix-det-prompts", doc["format"]
    return doc["prompts"]


def _unfence(raw):
    s = raw.strip()
    open_at = s.find("```")
    if open_at == -1:
        return s
    begin = open_at + 3
    nl = s.find("\n", begin)
    import re
    if nl != -1 and re.fullmatch(r"[A-Za-z0-9_+-]*", s[begin:nl]):
        begin = nl + 1
    close = s.find("```", begin)
    inner = s[begin:] if close == -1 else s[begin:close]
    return inner.strip()


def _preprocess(raw):
    text = _unfence(raw)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text


def _is_word_char(c):
    return c.isalnum() or c == "_"


def _extract_token(text):
    n = len(text)
    i = 0
    while i < n:
        c = text[i]
        if not ("A" <= c <= "Z"):
            i += 1
            continue
        if i > 0 and _is_word_char(text[i - 1]):
            i += 1
            continue
        j = i + 1
        while j < n and (("A" <= text[j] <= "Z") or ("0" <= text[j] <= "9")
                         or text[j] == "_"):
            j += 1
        if j - i < 2:
            i = j
            continue
        if j < n and _is_word_char(text[j]):
            i = j
            continue
        return text[i:j]
    return ""


def extract(kind, raw):
    text = _preprocess(raw)
    if kind == "token":
        return _extract_token(text)
    raise ValueError("unknown kind: %r" % (kind,))


def selftest():
    prompts = load_prompts()
    assert [p["id"] for p in prompts] == ["DET-TEST-01", "DET-TEST-02",
                                          "DET-TEST-03"]
    for p in prompts:
        got = extract(p["kind"], p["expected"])
        assert got == p["expected"], (p["id"], got, p["expected"])
    assert extract("token", "The answer is CATASTROPHIC.") == "CATASTROPHIC"
    assert extract("token", "lowercase only") == ""
    print("det selftest OK: token templates valid")


def frame_prompt(tok, body, opts):
    prompt_text = body
    if getattr(tok, "chat_template", None):
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if opts.get("enable_thinking") is not None:
            template_kwargs["enable_thinking"] = opts["enable_thinking"]
        if opts.get("low_effort"):
            template_kwargs["low_effort"] = True
        try:
            prompt_text = tok.apply_chat_template(
                [{"role": "user", "content": body}], **template_kwargs)
        except TypeError:
            template_kwargs.pop("enable_thinking", None)
            prompt_text = tok.apply_chat_template(
                [{"role": "user", "content": body}], **template_kwargs)
        except Exception as exc:  # noqa: BLE001
            print("chat_template failed, using raw prompt:", exc,
                  file=sys.stderr)
            prompt_text = body
    return prompt_text


def _load_tokenizer(model, opts):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        model, trust_remote_code=opts["trust_remote_code"])
    if not getattr(tok, "chat_template", None):
        jinja = os.path.join(model, "chat_template.jinja")
        if os.path.exists(jinja):
            with open(jinja, encoding="utf-8") as fh:
                tok.chat_template = fh.read()
    return tok


def template_golden(model, model_type, out, max_new_tokens):
    import transformers

    opts = dict(GENERIC)
    opts.update(OVERRIDES.get(model_type, {}))
    opts["max_new_tokens"] = max_new_tokens
    tok = _load_tokenizer(model, opts)
    cases = []
    for p in load_prompts():
        prompt_text = frame_prompt(tok, p["body"], opts)
        ids = tok(prompt_text).input_ids
        got = extract(p["kind"], p["expected"])
        assert got == p["expected"], (p["id"], got, p["expected"])
        cases.append({"id": p["id"], "kind": p["kind"],
                      "input_ids": list(ids),
                      "max_new_tokens": max_new_tokens,
                      "stress_runs": _STRESS_RUNS, "body": p["body"],
                      "expected": p["expected"], "raw": ""})
    golden = {"format": "logestix-det-golden", "version": 1,
              "model_type": model_type,
              "source": "prompt-template (%s source not runnable locally)"
                        % model_type,
              "runtime": "tokenizer.json + chat_template.jinja "
                         "(transformers %s)" % transformers.__version__,
              "prompts": cases}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(golden, fh, indent=2)
        fh.write("\n")
    print("wrote", out)


def capture(model, model_type, out, max_new_tokens, device_map=None):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    opts = dict(GENERIC)
    opts.update(OVERRIDES.get(model_type, {}))
    opts["max_new_tokens"] = max_new_tokens
    if device_map:
        opts["device_map"] = device_map
    tok = _load_tokenizer(model, opts)
    load_kwargs = {"trust_remote_code": opts["trust_remote_code"]}
    if opts.get("dtype"):
        load_kwargs["dtype"] = opts["dtype"]
    if opts.get("device_map"):
        load_kwargs["device_map"] = opts["device_map"]
    lm = AutoModelForCausalLM.from_pretrained(model, **load_kwargs)
    lm.eval()
    device = opts.get("device")
    if (device is None and not opts.get("device_map")
            and torch.cuda.is_available()):
        nbytes = sum(p.numel() * p.element_size() for p in lm.parameters())
        free, _total = torch.cuda.mem_get_info()
        if nbytes < int(free * 0.9):
            device = "cuda"
    if device:
        lm = lm.to(device)
    print("device:", next(lm.parameters()).device)

    cases = []
    for p in load_prompts():
        prompt_text = frame_prompt(tok, p["body"], opts)
        ids = tok(prompt_text, return_tensors="pt").input_ids.to(lm.device)
        with torch.no_grad():
            out_ids = lm.generate(ids, do_sample=False,
                                  max_new_tokens=max_new_tokens)
        gen = out_ids[0][ids.shape[1]:].tolist()
        raw = tok.decode(gen, skip_special_tokens=True)
        got = extract(p["kind"], raw)
        if not got:
            raise SystemExit("%s: empty extraction (raw=%r)" % (p["id"], raw))
        if got != p["expected"]:
            print("WARNING %s: model answer differs from the prompt template: "
                  "%r != %r" % (p["id"], got, p["expected"]), file=sys.stderr)
        cases.append({"id": p["id"], "kind": p["kind"],
                      "input_ids": ids[0].tolist(),
                      "max_new_tokens": max_new_tokens,
                      "stress_runs": _STRESS_RUNS, "body": p["body"],
                      "expected": got, "raw": raw})
    golden = {"format": "logestix-det-golden", "version": 1,
              "model_type": model_type, "source": model,
              "runtime": "transformers %s / torch %s"
                         % (transformers.__version__, torch.__version__),
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
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--template-golden", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not (args.model and args.type and args.out):
        ap.error("--model, --type and --out are required unless --selftest")
    if args.template_golden:
        template_golden(args.model, args.type, args.out, args.max_new_tokens)
    else:
        capture(args.model, args.type, args.out, args.max_new_tokens,
                args.device_map)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the self-test**

Run: `python3 tools/det_capture.py --selftest`
Expected: `det selftest OK: token templates valid`

- [ ] **Step 4: Commit (operator-gated)**

```bash
git add tools/det_prompts.json tools/det_capture.py
git commit -m "feat: add DET prompt set and capture tool"
```

---

### Task 5: `lsxbenchmark --det-selfcheck` CLI

**Files:**
- Modify: `libs/lsxcommon/src/model_runtime.cpp` (add `RunCliDetSelfCheck`)
- Modify: `libs/lsxcommon/include/lsxcommon/model_runtime.h` (declare it)
- Modify: `apps/lsxbenchmark/src/cli.h` (add `det_selfcheck`)
- Modify: `apps/lsxbenchmark/src/cli.cpp` (parse `--det-selfcheck`)
- Modify: `apps/lsxbenchmark/src/main.cpp` (dispatch + exclusivity)
- Modify: `apps/lsxbenchmark/tests/test_lsxbenchmark_cli.cpp` (new assertions)

**Interfaces:**
- Consumes: `det::LoadDetGoldenFile`, `det::RunDetSelfCheck`.
- Produces: `int lsxcommon::runtime::RunCliDetSelfCheck(const GenerationCliArgs&)`.

- [ ] **Step 1: Write the failing CLI test additions**

Append to `apps/lsxbenchmark/tests/test_lsxbenchmark_cli.cpp` (adapt names to the
existing style; the file uses `Parse({...})` returning `BenchmarkCliArgs a`):

```cpp
TEST_CASE("lsxbenchmark cli: --det-selfcheck selects verification mode",
          "[cli]")
{
    const lsxbenchmark::BenchmarkCliArgs a =
        Parse({"lsxbenchmark", "--det-selfcheck"});
    REQUIRE(a.det_selfcheck);
    REQUIRE_FALSE(a.bench);
    REQUIRE_FALSE(a.selfcheck);
    REQUIRE_FALSE(a.fsi_selfcheck);
    REQUIRE_FALSE(a.code_selfcheck);
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cmake --build --preset linux-debug --target test_lsxbenchmark_cli`
Expected: compile error — no member `det_selfcheck`.

- [ ] **Step 3: Add the CLI field and parse**

`apps/lsxbenchmark/src/cli.h`: after `code_selfcheck` add:

```cpp
    bool det_selfcheck = false; // --det-selfcheck: DET determinism gate
```

`apps/lsxbenchmark/src/cli.cpp`: after the `--code-selfcheck` branch add:

```cpp
        } else if (std::strcmp(argv[i], "--det-selfcheck") == 0) {
            a.det_selfcheck = true;
```

- [ ] **Step 4: Declare and implement `RunCliDetSelfCheck`**

In `libs/lsxcommon/include/lsxcommon/model_runtime.h`, next to
`RunCliCodeSelfCheck`, add:

```cpp
int RunCliDetSelfCheck(const GenerationCliArgs &args);
```

In `libs/lsxcommon/src/model_runtime.cpp`, add the include
`#include "lsxcommon/det_selfcheck.h"` and insert after `RunCliCodeSelfCheck`:

```cpp
int RunCliDetSelfCheck(const GenerationCliArgs &args)
{
    lsxcommon::log::info(
        "Running DET self-check (determinism + near-tie golden parity)...");
    const char *dir = std::getenv("LSX_DET_GOLDEN_DIR");
    if (!dir || !*dir) {
        lsxcommon::log::error(
            "LSX_DET_GOLDEN_DIR is not set (directory with <model_type>.json)");
        return 1;
    }
    CliModel m = CreateCliModel(args.model_path, FastMathMode::kOff);
    if (!m.ok) return 1;
    const std::string golden_path =
        std::string(dir) + "/" + m.model->Type() + ".json";
    det::DetGolden golden;
    std::string err;
    if (!det::LoadDetGoldenFile(golden_path, golden, &err)) {
        lsxcommon::log::error(
            absl::StrFormat("DET golden load failed: %s", err.c_str()));
        m.model->Unload();
        return 1;
    }
    int override = 0;
    if (const char *sr = std::getenv("LSX_DET_STRESS_RUNS")) {
        const int v = std::atoi(sr);
        if (v > 0) override = v;
    }
    const det::DetResult result =
        det::RunDetSelfCheck(*m.model, golden, {}, override);
    lsxcommon::log::info("=== lsxbenchmark: det-selfcheck ===");
    for (const det::DetCaseResult &c : result.cases)
        lsxcommon::log::info(absl::StrFormat(
            "%s  runs=%d  deterministic=%s  expected='%s'  extracted='%s'  %s",
            c.id.c_str(), c.runs, c.deterministic ? "yes" : "no",
            c.expected.c_str(), c.extracted.c_str(),
            c.passed ? "PASS" : "FAIL"));
    lsxcommon::log::info(
        absl::StrFormat("result   : %s", result.all_passed ? "PASS" : "FAIL"));
    m.model->Unload();
    return result.all_passed ? 0 : 1;
}
```

Note: the empty `{}` decode argument is ambiguous against the 2-arg overload;
use the explicit lambda form instead:

```cpp
    const det::DetResult result = det::RunDetSelfCheck(
        *m.model, golden,
        [&m](const std::vector<int32_t> &ids) {
            return lsxcommon::runtime::DecodeGeneratedText(m.model->Tokenizer(),
                                                           ids);
        },
        override);
```

Ensure `<cstdlib>` (for `std::getenv`/`std::atoi`) is already included in
`model_runtime.cpp`; if not, add it.

- [ ] **Step 5: Dispatch in `apps/lsxbenchmark/src/main.cpp`**

In the verification branch, add `det_selfcheck` to the mode predicate and the
dispatch chain:

```cpp
            (cli.selfcheck || cli.fsi_selfcheck || cli.code_selfcheck ||
             cli.det_selfcheck)
```
```cpp
        if (cli.det_selfcheck)
            rc = lsxcommon::runtime::RunCliDetSelfCheck(args);
        else if (cli.code_selfcheck)
            rc = lsxcommon::runtime::RunCliCodeSelfCheck(args);
```

(Order the `if` chain so exactly one runs; the earlier `--code-selfcheck`
branch stays.)

- [ ] **Step 6: Run to verify it passes**

Run: `cmake --build --preset linux-debug --target test_lsxbenchmark_cli && ./build/linux-debug/bin/test_lsxbenchmark_cli`
Expected: all CLI assertions PASS.

- [ ] **Step 7: Commit (operator-gated)**

```bash
git add libs/lsxcommon/include/lsxcommon/model_runtime.h libs/lsxcommon/src/model_runtime.cpp \
        apps/lsxbenchmark/src/cli.h apps/lsxbenchmark/src/cli.cpp apps/lsxbenchmark/src/main.cpp \
        apps/lsxbenchmark/tests/test_lsxbenchmark_cli.cpp
git commit -m "feat: add lsxbenchmark --det-selfcheck"
```

---

### Task 6: Per-family tests, CMake wiring, and guards

**Files:**
- Create: `tests/test_<family>_det_selfcheck.cpp` × 15 (see list)
- Modify: `tests/CMakeLists.txt` (15 targets + `LSX_DET_GOLDEN_DIR` + `inference` label list)
- Modify: `tests/test_model_suite_guard.cpp` (15 `det` rows + SKIP check + comment)
- Modify: `tests/test_ctest_labels_guard.cpp` if it enumerates labels explicitly (verify; likely no change needed)

**Interfaces:**
- Consumes: `det::LoadDetGoldenFile`, `det::RunDetSelfCheck`, `LsxModelFactory`.
- Produces: ctest targets `test_<family>_det_selfcheck`.

- [ ] **Step 1: Create the 15 per-family test files**

Copy the template below per family, changing only `kModelType`, the test name,
the env-var name in `...ModelRoot()`, and the vault subdirectory. Families and
their `(kModelType, vault-dir, target-stem)`:

| family Type() | vault dir | target stem |
|---|---|---|
| `glm-5.2` | `glm-5.2` | `glm_moe` |
| `glm-5.3-flash` | `GLM-5.3-Flash` | `glm_moe_flash` |
| `deepseek-v4-flash` | `DeepSeek-V4-Flash` | `deepseek_moe` |
| `deepseek-v4.1-flash` | `DeepSeek-V4.1-Flash` | `deepseek_moe_v41` |
| `qwen3.5-9b` | `Qwen3.5-9B` | `qwen_dense` |
| `qwen3.8-27b` | `Qwen3.8-27B` | `qwen_dense_27b` |
| `qwen3.5-35b-a3b` | `Qwen3.5-35B-A3B` | `qwen_moe` |
| `qwen3.6-35b-a3b` | `Qwen3.6-35B-A3B` | `qwen_moe_36` |
| `ornith-1.5-35b-a3b` | `Ornith-1.5-35B-A3B` | `ornith_moe` |
| `qwen3.8-flash-next` | `Qwen3.8-Flash-Next` | `qwen_moe_exp` |
| `gemma-4-26b-a4b-it` | `gemma-4-26B-A4B-it` | `gemma_moe` |
| `kimi-k2.7-code` | `Kimi-K2.7-Code` | `kimi_moe` |
| `nemotron-3.5-lightning-30b-a3b` | `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` | `nemotron_moe` |
| `gigachat3.5-432b-a28b-reasoning` | `GigaChat3.5-432B-A28B-Reasoning` | `gigachat_moe` |
| `granite-4.2-30b` | `granite-4.2-30b` | `granite_dense` |

Template (`tests/test_<stem>_det_selfcheck.cpp`):

```cpp
// DETSelfCheck — <kModelType>: verifies the converted model is bit-deterministic
// across repeated greedy runs and reproduces the golden near-tie token.
// Separate functionality from SelfCheck; SKIPs without converted weights or
// golden.
#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <string>

#include <lsxcommon/det_selfcheck.h>
#include <lsxcommon/model.h>
#include <lsxcommon/model_factory.h>
#include <lsxcommon/path_registry.h>

#ifndef LSX_DET_GOLDEN_DIR
#define LSX_DET_GOLDEN_DIR "tests/fixtures/det"
#endif

namespace {

constexpr const char *kModelType = "<kModelType>";
constexpr const char *kModelDir = "<vault dir>";
constexpr const char *kEnv = "LSX_TEST_<UPPER>_MODEL_PATH";

std::filesystem::path ModelRoot()
{
    if (const char *env = std::getenv(kEnv))
        return std::filesystem::path(env);
    return std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
           kModelDir;
}

std::filesystem::path GoldenPath()
{
    if (const char *env = std::getenv("LSX_DET_GOLDEN_DIR"))
        return std::filesystem::path(env) /
               (std::string(kModelType) + ".json");
    return std::filesystem::path(LSX_DET_GOLDEN_DIR) /
           (std::string(kModelType) + ".json");
}

} // namespace

TEST_CASE("<kModelType> DET SelfCheck: deterministic near-tie golden",
          "[det][real-model]")
{
    const std::filesystem::path root = ModelRoot();
    if (!std::filesystem::exists(root / "logestix" / "model.index.json"))
        SKIP("<kModelType> converted model directory not present");
    const std::filesystem::path golden_path = GoldenPath();
    if (!std::filesystem::exists(golden_path))
        SKIP("DET golden not captured for <kModelType>");

    lsxcommon::det::DetGolden golden;
    std::string err;
    REQUIRE(lsxcommon::det::LoadDetGoldenFile(golden_path.string(), golden,
                                              &err));
    REQUIRE(golden.model_type == kModelType);

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = root.string();
    cfg.model_type = kModelType;
    cfg.fast_math_mode = lsxcommon::FastMathMode::kOff;
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Type() == kModelType);
    REQUIRE(model->Load());

    const lsxcommon::det::DetResult result =
        lsxcommon::det::RunDetSelfCheck(*model, golden);
    for (const lsxcommon::det::DetCaseResult &c : result.cases)
        INFO("case " << c.id << " runs=" << c.runs
                     << " deterministic=" << c.deterministic
                     << " expected='" << c.expected << "' extracted='"
                     << c.extracted << "' raw='" << c.raw << "' error='"
                     << c.error << "'");
    REQUIRE(result.all_passed);

    model->Unload();
}
```

- [ ] **Step 2: Add the 15 CMake targets**

In `tests/CMakeLists.txt`, after the CODESelfCheck block (ends line ~3719),
append one 13-line block per family (adjusting name/path). For granite:

```cmake
# ── DETSelfCheck (separate from SelfCheck/FSI/CODE): golden determinism ──
file(TO_CMAKE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/det" LSX_DET_GOLDEN_DIR_FWD)
add_executable(test_granite_dense_det_selfcheck test_granite_dense_det_selfcheck.cpp)
target_compile_definitions(test_granite_dense_det_selfcheck PRIVATE
    LSX_DET_GOLDEN_DIR="${LSX_DET_GOLDEN_DIR_FWD}")
target_link_libraries(test_granite_dense_det_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
if(WIN32)
    add_custom_command(TARGET test_granite_dense_det_selfcheck POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_granite_dense_det_selfcheck>")
    add_custom_command(TARGET test_granite_dense_det_selfcheck POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_granite_dense_det_selfcheck>")
endif()
add_test(NAME test_granite_dense_det_selfcheck COMMAND test_granite_dense_det_selfcheck)
set_tests_properties(test_granite_dense_det_selfcheck PROPERTIES SKIP_RETURN_CODE 4)
```

Repeat for all 15 stems. Then add all 15 names to the `PROPERTIES LABELS
inference` list (after `test_granite_dense_code_selfcheck`, before the closing
`PROPERTIES LABELS inference)` line):

```cmake
    test_glm_moe_det_selfcheck
    test_glm_moe_flash_det_selfcheck
    test_deepseek_moe_det_selfcheck
    test_deepseek_moe_v41_det_selfcheck
    test_qwen_dense_det_selfcheck
    test_qwen_dense_27b_det_selfcheck
    test_qwen_moe_det_selfcheck
    test_qwen_moe_36_det_selfcheck
    test_ornith_moe_det_selfcheck
    test_qwen_moe_exp_det_selfcheck
    test_gemma_moe_det_selfcheck
    test_kimi_moe_det_selfcheck
    test_nemotron_moe_det_selfcheck
    test_gigachat_moe_det_selfcheck
    test_granite_dense_det_selfcheck
```

- [ ] **Step 3: Extend the suite guard**

In `tests/test_model_suite_guard.cpp`:

1. Update the lifecycle comment (lines 8 and 16): `... -> selfcheck -> fsi ->
   code -> det` and mention `det`.
2. In `Suites()`, add `{"det", {"test_<stem>_det_selfcheck"}}` after every
   `code` stage (15 rows), e.g. for granite:
   ```cpp
   {"code", {"test_granite_dense_code_selfcheck"}},
   {"det", {"test_granite_dense_det_selfcheck"}}}},
   ```
3. In `CheckSelfcheckSkip`, extend the stage filter:
   ```cpp
   if (stage != "selfcheck" && stage != "fsi" &&
       stage != "code" && stage != "det") continue;
   ```
4. Update the test-case name at line ~509 from
   `"suite guard: selfcheck, fsi and code targets declare "` to include `det`.

- [ ] **Step 4: Build the guards and a per-family target**

Run:
```bash
cmake --preset linux-debug
cmake --build --preset linux-debug --target test_model_suite_guard test_ctest_labels_guard test_granite_dense_det_selfcheck
./build/linux-debug/tests/test_model_suite_guard
./build/linux-debug/tests/test_ctest_labels_guard
```
Expected: both guards PASS. The granite DET test builds (it will SKIP at runtime
until a golden exists).

- [ ] **Step 5: Commit (operator-gated)**

```bash
git add tests/test_*_det_selfcheck.cpp tests/CMakeLists.txt tests/test_model_suite_guard.cpp
git commit -m "test: add per-family DET selfcheck targets and suite stage"
```

---

### Task 7: Capture pilot goldens and run the real-model gate

**Files:**
- Create: `tests/fixtures/det/<model_type>.json` (one per runnable family)

**Interfaces:**
- Consumes: `tools/det_capture.py`, converted vault models.

- [ ] **Step 1: Capture a template golden for every family (fast, always possible)**

For each family, run the tokenizer-only fallback so the golden exists:

```bash
python3 tools/det_capture.py --template-golden \
    --model /mnt/llm_models/models/<source-dir> \
    --type <model_type> --out tests/fixtures/det/<model_type>.json
```

- [ ] **Step 2: Replace with model-defined goldens where the source runs locally**

For a runnable source (start with `granite-4.2-30b` and `Qwen3.6-35B-A3B`):

```bash
python3 tools/det_capture.py \
    --model /mnt/llm_models/models/granite-4.2-30b \
    --type granite-4.2-30b --out tests/fixtures/det/granite-4.2-30b.json
```

A non-empty extraction is required; a `WARNING` about a template difference is
acceptable (model-defined golden wins).

- [ ] **Step 3: Run the gate on the converted models**

```bash
cmake --build --preset linux-debug
ctest --preset linux-debug-inference -R "_det_selfcheck" --output-on-failure
```

Expected: PASS on converted models with goldens; SKIP where absent. Any
`deterministic=no` or extraction mismatch is a **real finding** — record it in
`docs/DET_SELFCHECK_FINDINGS.md`; do not loosen the harness.

- [ ] **Step 4: Verify the CLI on one model**

```bash
LSX_DET_GOLDEN_DIR=tests/fixtures/det \
  ./build/linux-debug/bin/lsxbenchmark --model /mnt/llm_models/models/granite-4.2-30b \
  --det-selfcheck
```
Expected: exit 0 and a `result   : PASS` line.

- [ ] **Step 5: Commit (operator-gated)**

```bash
git add tests/fixtures/det/
git commit -m "test: capture DET goldens for runnable families"
```

---

### Task 8: Documentation and final guard sweep

**Files:**
- Create: `docs/DET_SELFCHECK_FINDINGS.md`
- Create: `.opencode/skills/det-selfcheck/SKILL.md`
- Modify: `AGENTS.md` (add DET to the `lsxcommon` structure paragraph and to the skill list)
- The spec already exists at `docs/superpowers/specs/2026-09-27-det-selfcheck-design.md`.

- [ ] **Step 1: Write `docs/DET_SELFCHECK_FINDINGS.md`**

Record, per family: capture feasibility, the capture recipe/overrides, the
observed `deterministic` result, and any resolved/unresolved divergence. Seed it
with the pilot results from Task 7.

- [ ] **Step 2: Write the skill `.opencode/skills/det-selfcheck/SKILL.md`**

Mirror the `fsi-selfcheck` skill's structure: overview, decoupling invariants
table, prompt set, capture workflow, golden schema, test/CMake/guard wiring,
extraction contract, common mistakes, verification commands.

- [ ] **Step 3: Update `AGENTS.md`**

- In the `libs/lsxcommon/` structure paragraph, add a sentence for
  `det_selfcheck.{h,cpp}` (host-only, model-neutral; capture
  `tools/det_capture.py` + `tools/det_prompts.json`; goldens
  `tests/fixtures/det/<type>.json`; per-family `test_<family>_det_selfcheck.cpp`,
  label `inference`, SKIP 4; stage `det`; `lsxbenchmark --det-selfcheck`;
  env `LSX_DET_GOLDEN_DIR`/`LSX_DET_STRESS_RUNS`).
- In the "Structure" model list section, reference the new skill
  `det-selfcheck`.

- [ ] **Step 4: Final guard and unit sweep**

Run:
```bash
cmake --build --preset linux-debug
ctest --preset linux-debug -R "test_det_selfcheck|test_model_suite_guard|test_ctest_labels_guard|test_lsxbenchmark_cli"
ctest --preset linux-debug-inference -R "_det_selfcheck"
```
Expected: host unit tests + both guards + CLI test PASS; per-family DET targets
PASS/SKIP as expected.

- [ ] **Step 5: Commit (operator-gated)**

```bash
git add docs/DET_SELFCHECK_FINDINGS.md .opencode/skills/det-selfcheck AGENTS.md
git commit -m "docs: document DETSelfCheck and its skill"
```

---

## Self-Review

**Spec coverage:** §3.1 prompt set → Task 4; §3.2 verify contract → Task 3; §3.3
numeric mode → Tasks 5/6; §3.4 decoupling → enforced structurally by Tasks 1–3, 6;
§4 extraction → Task 1; §5 golden schema → Tasks 2, 4; §6 API → Tasks 1–3; §6.1
CLI → Task 5; §7 capture → Task 4; §8.1 unit tests → Tasks 1–3; §8.2 per-family →
Task 6; §8.3 ordering → task order; §9 risks → documented in spec/findings; §10
resources → Task 7; §11 DoD → Tasks 1–8.

**Placeholder scan:** no TBD/TODO; every code step carries real code. The only
parameterized item is the per-family template in Task 6, which is fully specified
with a table for every family.

**Type consistency:** `DetKind::kToken`, `DetGoldenCase::stress_runs`,
`DetCaseResult::{runs,deterministic}` and
`RunDetSelfCheck(..., int stress_runs_override=0)` are used identically in Tasks
1, 2, 3, 5, 6.
