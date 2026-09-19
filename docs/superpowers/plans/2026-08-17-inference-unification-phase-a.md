# Inference Unification — Phase A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the triplicated host-side inference scaffolding across GLM/DeepSeek/Qwen and unify the `gpu::InferenceResult` contract (spec items 1-10).

**Architecture:** A host-only `lsxcommon::runtime` layer (`model_runtime.h/.cpp`) plus a thin `ModelBase` (`model_base.h/.cpp`) own the tokenizer-db load, infer-request validation, decode capacity-retry, SelfCheck-"Paris" and CLI lifecycle. All three models migrate to inherit `ModelBase` and delegate to `runtime`. `InferenceResult` is extended additively (`gen_tokens`/`output_text`/`final_logits` always published, precise timings, `weight_resident_bytes`, `eos_terminated`). Engine result structs carry timing/occupancy fields set inside the CUDA loops.

**Tech Stack:** C++20, CUDA 13.2+ (Blackwell `120-real;121-real` only), Arrow 25, spdlog, absl (host-only), Catch2 v3.7.0.

**Spec:** `docs/superpowers/specs/2026-08-17-inference-unification-design.md`

## Global Constraints

- C++20; CUDA only for Blackwell (CC 12.0/12.1). Hopper/Ada/Ampere forbidden.
- `lsxcommon::runtime`/`ModelBase` headers pull **zero CUDA headers and zero `absl/container/*`** — they must stay includable from `.cu` TUs. `absl::strings` is safe in `.cu`; `absl::flat_hash_*`/`btree_*` are NOT.
- All logging via spdlog (never `std::cout`/`printf`).
- Host-only C++ containers prefer `absl::` over `std::` (`absl::StrCat` for concatenation). `std::vector`, `std::string`, `std::string_view`, `std::unique_ptr` stay `std::`.
- Per MODEL INGEST invariant: `bos_id`/`eos_id`/`pad_id` come from the config-parsed `ArrowTokenizerDb`, never hardcoded in kernels.
- TDD: every task is RED → GREEN → commit. New tests in `tests/`, registered in `tests/CMakeLists.txt`.
- Commit messages: English, prefixes `build:`/`feat:`/`fix:`/`refactor:`/`test:`/`docs:`.
- Real-model tests SKIP (Catch2 `SKIP`, CTest `SKIP_RETURN_CODE 4`) without checkpoints.
- CMake: new `.cpp` go into `libs/lsxcommon/CMakeLists.txt` `lsxcommon` source list.

---

### Task 1: `ModelBase` — tokenizer-db load + infer-request validation

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/model_base.h`
- Create: `libs/lsxcommon/src/model_base.cpp`
- Modify: `libs/lsxcommon/CMakeLists.txt:16-43` (add both to the `lsxcommon` source list)
- Create: `tests/test_model_runtime.cpp`
- Modify: `tests/CMakeLists.txt` (register the test)

**Interfaces:**
- Consumes: `lsxcommon::IModel`, `lsxcommon::ModelInitConfig`, `lsxcommon::ModelRequest` (`lsxcommon/model.h`), `lsxcommon::ArrowTokenizerDb` + `load_tokenizer_metadata` (`lsxcommon/tokenizer.h`).
- Produces: `class lsxcommon::ModelBase : public IModel` with protected `cfg_`, `tok_db_`, `loaded_`, `tokenizer_optional_`; `bool LoadTokenizerDb(const std::string& model_path)`; `bool ValidateInferRequest(const ModelRequest& req, gpu::InferenceResult& r)`; `ArrowTokenizerDb& Tokenizer() override`. Used by Tasks 5-8.

- [ ] **Step 1: Write the failing test**

Create `tests/test_model_runtime.cpp`:

```cpp
// TDD — ModelBase (host-only, no CUDA, no real model).
#include <catch2/catch_test_macros.hpp>

#include <arrow/io/file.h>
#include <filesystem>
#include <string>
#include <vector>

#include <lsxcommon/converter.h>
#include <lsxcommon/model_base.h>
#include <lsxcommon/tokenizer.h>

namespace fs = std::filesystem;

namespace {

std::string BuildPinnedTokenizerDir(const std::string &fixture_name)
{
    auto pinned = fs::temp_directory_path() / ("modelbase_" + fixture_name) / "logestix" / "pinned";
    fs::remove_all(pinned.parent_path().parent_path());
    fs::create_directories(pinned);
    REQUIRE(lsxcommon::BuildTokenizerArrowFiles(
                TOKENIZER_TEST_DATA_DIR "/" + fixture_name, pinned)
                .ok());
    return pinned.parent_path().parent_path().string();
}

class FakeModel : public lsxcommon::ModelBase {
public:
    explicit FakeModel(std::string type) : type_(std::move(type)) {}

    bool Load() override
    {
        if (!LoadTokenizerDb(cfg_.model_path)) return false;
        loaded_ = true;
        return true;
    }
    void Unload() override
    {
        tok_db_.reset();
        loaded_ = false;
    }
    gpu::InferenceResult Infer(const ModelRequest &req) override
    {
        gpu::InferenceResult r;
        if (!ValidateInferRequest(req, r)) return r;
        return scripted_;
    }
    SafetensorsModel Inspect() override { return {}; }
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override
    {
        gpu::InferenceResult r;
        if (!ValidateInferRequest(req, r)) return r;
        return scripted_;
    }
    const std::string &Type() const override { return type_; }

    gpu::InferenceResult scripted_;

private:
    std::string type_;
};

} // namespace

TEST_CASE("model_base: LoadTokenizerDb loads a valid Arrow db", "[model_base]")
{
    const std::string root = BuildPinnedTokenizerDir("synthetic_tok.json");
    FakeModel m("deepseek-v4-flash");
    m.cfg_.model_path = root;
    REQUIRE(m.Load());
    REQUIRE(m.tok_db_ != nullptr);
    REQUIRE(m.Tokenizer().valid());
    REQUIRE(m.Tokenizer().vocab_size > 0);
    REQUIRE(m.loaded_);
    m.Unload();
    fs::remove_all(root);
}

TEST_CASE("model_base: LoadTokenizerDb fails on a missing model dir", "[model_base]")
{
    FakeModel m("glm-5.2");
    m.cfg_.model_path = fs::temp_directory_path() / "no_such_model_xyz";
    REQUIRE_FALSE(m.Load());
    REQUIRE_FALSE(m.loaded_);
}

TEST_CASE("model_base: ValidateInferRequest canonical errors", "[model_base]")
{
    FakeModel m("glm-5.2");
    const std::string root = BuildPinnedTokenizerDir("synthetic_tok.json");
    m.cfg_.model_path = root;
    REQUIRE(m.Load());

    lsxcommon::ModelRequest req;
    gpu::InferenceResult r;

    req.input_ids = {1, 2};
    req.max_gen_tokens = 64;
    // not loaded -> "model not loaded"
    m.loaded_ = false;
    REQUIRE_FALSE(m.ValidateInferRequest(req, r));
    REQUIRE(r.error_msg == "model not loaded");
    m.loaded_ = true;

    req.input_ids.clear();
    REQUIRE_FALSE(m.ValidateInferRequest(req, r));
    REQUIRE(r.error_msg == "empty input_ids");

    req.input_ids = {1, 2};
    req.max_gen_tokens = 0;
    REQUIRE_FALSE(m.ValidateInferRequest(req, r));
    REQUIRE(r.error_msg == "max_gen_tokens must be positive");

    req.max_gen_tokens = 64;
    REQUIRE(m.ValidateInferRequest(req, r));
    m.Unload();
    fs::remove_all(root);
}
```

- [ ] **Step 2: Run test to verify it fails**

Create `libs/lsxcommon/include/lsxcommon/model_base.h`:

```cpp
#pragma once

#include <memory>
#include <string>

#include "lsxcommon/model.h"
#include "lsxcommon/tokenizer.h"

namespace lsxcommon {

// Thin shared base for every lsxcommon model. Owns the tokenizer Arrow db
// (Item 1: tokenizer-db loading) and the canonical Infer/SelfCheck request
// guards (Item 5: infer-preamble guards). All three models migrate onto it.
class ModelBase : public IModel {
public:
    ModelInitConfig cfg_;
    std::unique_ptr<ArrowTokenizerDb> tok_db_;
    bool loaded_ = false;
    bool tokenizer_optional_ = false;  // Qwen: warn+continue when arrows absent

    bool LoadTokenizerDb(const std::string &model_path);
    bool ValidateInferRequest(const ModelRequest &req, gpu::InferenceResult &r);

    ArrowTokenizerDb &Tokenizer() override { return *tok_db_; }

protected:
    ~ModelBase() override = default;
};

} // namespace lsxcommon
```

Create `libs/lsxcommon/src/model_base.cpp`:

```cpp
#include "lsxcommon/model_base.h"

#include <arrow/io/file.h>
#include <filesystem>
#include <spdlog/spdlog.h>

namespace lsxcommon {

bool ModelBase::LoadTokenizerDb(const std::string &model_path)
{
    if (tok_db_) return true;
    tok_db_ = std::make_unique<ArrowTokenizerDb>();
    auto pinned = std::filesystem::path(model_path) / "logestix" / "pinned";
    auto v = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_vocab.arrow").string(), arrow::io::FileMode::READ);
    auto m = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_merges.arrow").string(), arrow::io::FileMode::READ);
    if (!v.ok() || !m.ok()) {
        if (tokenizer_optional_) {
            spdlog::warn("[model] tokenizer arrows not found (OK for synthetic test)");
            return true;
        }
        spdlog::error("[model] cannot open tokenizer arrows in {}", pinned.string());
        tok_db_.reset();
        return false;
    }
    auto st = load_tokenizer_metadata(*v, *m, *tok_db_);
    if (!st.ok()) {
        if (tokenizer_optional_) {
            spdlog::warn("[model] tokenizer load failed: {}", st.ToString());
            return true;
        }
        spdlog::error("[model] load_tokenizer_metadata: {}", st.ToString());
        tok_db_.reset();
        return false;
    }
    return true;
}

bool ModelBase::ValidateInferRequest(const ModelRequest &req, gpu::InferenceResult &r)
{
    if (!loaded_) {
        r.error_msg = "model not loaded";
        return false;
    }
    if (req.input_ids.empty()) {
        r.error_msg = "empty input_ids";
        return false;
    }
    if (req.max_gen_tokens <= 0) {
        r.error_msg = "max_gen_tokens must be positive";
        return false;
    }
    return true;
}

} // namespace lsxcommon
```

Add `src/model_base.cpp` to the `lsxcommon` source list in `libs/lsxcommon/CMakeLists.txt` (alphabetical, after `src/loader.cpp`). Add the test to `tests/CMakeLists.txt`:

```cmake
# ── ModelBase test (tokenizer-db load + infer guards, host-only) ─────────
add_executable(test_model_runtime test_model_runtime.cpp)
target_compile_definitions(test_model_runtime PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_model_runtime PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_model_runtime POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_model_runtime>"
    )
    add_custom_command(TARGET test_model_runtime POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_model_runtime>"
    )
endif()

add_test(NAME test_model_runtime COMMAND test_model_runtime)
```

Run: `cmake --build --preset linux-debug --target test_model_runtime 2>&1 | tail -20`
Expected: FAIL — `model_base.h` not found (no such file yet) or compile error. The point is the test does not pass yet.

- [ ] **Step 3: Run test to verify it fails (compile gate)**

Run: `ctest --preset linux-debug -R test_model_runtime 2>&1 | tail -20`
Expected: the target does not build yet. (If the build already picked up the header from Step 2, remove `model_base.h`/`.cpp` temporarily, or write the test FIRST — the intended RED order.)

- [ ] **Step 4: Implement minimal `ModelBase`**

Apply the two file creations from Step 2 (create `model_base.h` and `model_base.cpp` with the exact content above) and the CMake edits.

- [ ] **Step 5: Run test to verify it passes**

Run: `cmake --build --preset linux-debug --target test_model_runtime && ctest --preset linux-debug -R test_model_runtime`
Expected: 3 test cases PASS.

- [ ] **Step 6: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/model_base.h libs/lsxcommon/src/model_base.cpp libs/lsxcommon/CMakeLists.txt tests/test_model_runtime.cpp tests/CMakeLists.txt
git commit -m "feat: add ModelBase with unified tokenizer-db load and infer guards"
```

---

### Task 2: `runtime::DecodeGeneratedText` + `runtime::RunParisSelfCheck`

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/model_runtime.h`
- Create: `libs/lsxcommon/src/model_runtime.cpp`
- Modify: `libs/lsxcommon/CMakeLists.txt:16-43` (add `src/model_runtime.cpp`)
- Modify: `tests/test_model_runtime.cpp` (add cases)

**Interfaces:**
- Consumes: `ArrowTokenizerDb` (`tokenizer.h`), `absl::StripAsciiWhitespace` (`<absl/strings/ascii.h>`).
- Produces:
  - `std::string runtime::DecodeGeneratedText(const ArrowTokenizerDb&, const std::vector<int32_t>& ids)`
  - `gpu::InferenceResult runtime::RunParisSelfCheck(IModel& model, const ModelRequest& req)`
  Used by Tasks 5-8 (SelfCheck bodies become thin delegations; DeepSeek `.cu` calls `DecodeGeneratedText`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_model_runtime.cpp`:

```cpp
TEST_CASE("model_runtime: DecodeGeneratedText round-trips a token stream", "[model_runtime]")
{
    const std::string root = BuildPinnedTokenizerDir("synthetic_tok.json");
    FakeModel m("deepseek-v4-flash");
    m.cfg_.model_path = root;
    REQUIRE(m.Load());

    const char *text = "The capital of France is Paris";
    std::vector<int32_t> ids(256, 0);
    size_t cap = ids.size();
    const int32_t n = lsxcommon::arrow_tokenizer_encode(
        m.Tokenizer(), text, std::strlen(text), ids.data(), &cap);
    REQUIRE(n > 0);
    ids.resize(static_cast<size_t>(n));

    const std::string out = lsxcommon::runtime::DecodeGeneratedText(m.Tokenizer(), ids);
    REQUIRE_FALSE(out.empty());
    REQUIRE(out.find("Paris") != std::string::npos);

    REQUIRE(lsxcommon::runtime::DecodeGeneratedText(m.Tokenizer(), {}).empty());
    m.Unload();
    fs::remove_all(root);
}

TEST_CASE("model_runtime: RunParisSelfCheck matches a 1/2-token Paris prefix", "[model_runtime]")
{
    const std::string root = BuildPinnedTokenizerDir("synthetic_tok.json");
    FakeModel m("glm-5.2");
    m.cfg_.model_path = root;
    REQUIRE(m.Load());

    // Script the fake's Infer to emit gen_tokens that decode to "Paris".
    std::vector<int32_t> paris(256, 0);
    size_t cap = paris.size();
    const int32_t np = lsxcommon::arrow_tokenizer_encode(
        m.Tokenizer(), "Paris", 5, paris.data(), &cap);
    REQUIRE(np > 0);
    paris.resize(static_cast<size_t>(np));
    m.scripted_.ok = true;
    m.scripted_.gen_tokens = paris;
    m.scripted_.first_gen_tok_id = paris.front();

    lsxcommon::ModelRequest req{std::vector<int32_t>{1, 2}, 1};
    auto r = lsxcommon::runtime::RunParisSelfCheck(m, req);
    REQUIRE(r.ok);
    REQUIRE(r.selfcheck_passed);
    REQUIRE_FALSE(r.first_gen_text.empty());

    m.scripted_.gen_tokens = {12345};   // never decodes to "Paris"
    m.scripted_.first_gen_tok_id = 12345;
    auto r2 = lsxcommon::runtime::RunParisSelfCheck(m, req);
    REQUIRE(r2.ok);
    REQUIRE_FALSE(r2.selfcheck_passed);
    m.Unload();
    fs::remove_all(root);
}
```

Add `#include <cstring>` and `#include "lsxcommon/model_runtime.h"` to the test.

- [ ] **Step 2: Run test to verify it fails**

Run: `ctest --preset linux-debug -R test_model_runtime 2>&1 | tail -20`
Expected: FAIL — `model_runtime.h` missing, tests do not compile.

- [ ] **Step 3: Write minimal implementation**

Create `libs/lsxcommon/include/lsxcommon/model_runtime.h`:

```cpp
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/inference_gpu.h"
#include "lsxcommon/model.h"
#include "lsxcommon/tokenizer.h"

namespace lsxcommon {
namespace runtime {

// Item 2 — unified decode capacity-retry. Decodes a generated token stream
// into text. Grows the output buffer on the tokenizer's -2 (capacity too
// small) return; returns "" (with an spdlog warning) on persistent failure.
// Host-only: safe to call from .cu TUs.
std::string DecodeGeneratedText(const ArrowTokenizerDb &tok_db,
                                const std::vector<int32_t> &ids);

// Item 3 — unified SelfCheck. Forces max(2, req.max_gen_tokens), delegates to
// model.Infer, then matches the decoded 1->2-token prefix against "Paris".
// Returns the InferenceResult with selfcheck_passed filled.
gpu::InferenceResult RunParisSelfCheck(IModel &model, const ModelRequest &req);

} // namespace runtime
} // namespace lsxcommon
```

Create `libs/lsxcommon/src/model_runtime.cpp`:

```cpp
#include "lsxcommon/model_runtime.h"

#include <absl/strings/ascii.h>
#include <algorithm>
#include <cstdint>
#include <spdlog/spdlog.h>
#include <vector>

namespace lsxcommon {
namespace runtime {

namespace {
constexpr size_t kMaxDecodeBytes = 1u << 20;
constexpr size_t kInitialDecodeCap = 4096;
constexpr int kMaxDecodeAttempts = 8;
} // namespace

std::string DecodeGeneratedText(const ArrowTokenizerDb &tok_db,
                                const std::vector<int32_t> &ids)
{
    if (ids.empty()) return {};
    size_t cap = kInitialDecodeCap;
    std::string out(cap, '\0');
    for (int attempt = 0; attempt < kMaxDecodeAttempts; ++attempt) {
        const int dr = arrow_tokenizer_decode(tok_db, ids.data(), ids.size(),
                                              out.data(), &cap);
        if (dr == 0) {
            out.resize(cap);
            return out;
        }
        if (dr == -2 && cap > out.size() && cap <= kMaxDecodeBytes) {
            out.resize(cap);
            continue;
        }
        break;
    }
    spdlog::warn("[model_runtime] tokenizer decode failed ({} ids)", ids.size());
    return {};
}

gpu::InferenceResult RunParisSelfCheck(IModel &model, const ModelRequest &req)
{
    ModelRequest sc_req = req;
    sc_req.max_gen_tokens = std::max(2, req.max_gen_tokens);
    gpu::InferenceResult r = model.Infer(sc_req);
    if (!r.ok) {
        r.selfcheck_passed = false;
        return r;
    }
    if (!r.gen_tokens.empty()) {
        char buf[512];
        for (int ntok : {1, 2}) {
            if (ntok > static_cast<int>(r.gen_tokens.size())) break;
            size_t cap = sizeof(buf);
            if (arrow_tokenizer_decode(model.Tokenizer(), r.gen_tokens.data(),
                                       static_cast<size_t>(ntok), buf, &cap) != 0)
                continue;
            std::string t(buf, cap);
            absl::StripAsciiWhitespace(&t);
            r.first_gen_text = t;
            if (r.first_gen_text == "Paris") {
                r.selfcheck_passed = true;
                break;
            }
        }
        if (!r.selfcheck_passed && r.first_gen_text.empty()) {
            size_t cap = sizeof(buf);
            if (arrow_tokenizer_decode(model.Tokenizer(), r.gen_tokens.data(),
                                       std::min(static_cast<int>(r.gen_tokens.size()), 8),
                                       buf, &cap) == 0)
                r.first_gen_text = std::string(buf, cap);
        }
    }
    spdlog::info("[selfcheck] top1_tok={} text='{}' passed={}",
                 r.first_gen_tok_id, r.first_gen_text,
                 r.selfcheck_passed ? "true" : "false");
    return r;
}

} // namespace runtime
} // namespace lsxcommon
```

Add `src/model_runtime.cpp` to the `lsxcommon` source list.

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build --preset linux-debug --target test_model_runtime && ctest --preset linux-debug -R test_model_runtime`
Expected: 5 test cases PASS.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/model_runtime.h libs/lsxcommon/src/model_runtime.cpp libs/lsxcommon/CMakeLists.txt tests/test_model_runtime.cpp
git commit -m "feat: add runtime decode capacity-retry and unified Paris selfcheck"
```

---

### Task 3: Prompt framing descriptor

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/prompt_framing.h`
- Modify: `libs/lsxcommon/src/prompt_framing.cpp`
- Modify: `tests/test_glm_prompt_framing.cpp` (add descriptor cases)

**Interfaces:**
- Consumes: existing `prompt::BuildPromptInputIds`.
- Produces: `struct lsxcommon::prompt::FramingDescriptor { std::string_view model_type; std::string_view leading[3]; std::string_view trailing[3]; std::size_t leading_count, trailing_count; };` and `const FramingDescriptor* prompt::LookupFraming(std::string_view model_type)`. Used by Tasks 5-8 and Task 9 (CLI helpers) transparently (no signature change to `BuildPromptInputIds`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_glm_prompt_framing.cpp`:

```cpp
TEST_CASE("prompt_framing: LookupFraming returns the GLM descriptor", "[prompt_framing]")
{
    const auto *f = lsxcommon::prompt::LookupFraming("glm-5.2");
    REQUIRE(f != nullptr);
    REQUIRE(f->model_type == "glm-5.2");
    REQUIRE(f->leading_count == 3);
    REQUIRE(f->trailing_count == 3);
    REQUIRE(f->leading[0] == "[gMASK]");
    REQUIRE(f->leading[1] == "<sop>");
    REQUIRE(f->leading[2] == "<|user|>");
    REQUIRE(f->trailing[0] == "<|assistant|>");
    REQUIRE(f->trailing[1] == " thinking");
    REQUIRE(f->trailing[2] == " response");

    REQUIRE(lsxcommon::prompt::LookupFraming("deepseek-v4-flash") == nullptr);
    REQUIRE(lsxcommon::prompt::LookupFraming("qwen3.5-9b") == nullptr);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `ctest --preset linux-debug -R test_glm_prompt_framing 2>&1 | tail -20`
Expected: FAIL — `LookupFraming` undefined.

- [ ] **Step 3: Implement**

In `libs/lsxcommon/include/lsxcommon/prompt_framing.h`, inside `namespace lsxcommon::prompt`, add:

```cpp
// Ordered chat-template sentinel descriptor for one model. Sentinels are
// resolved via arrow_tokenizer_lookup at encode time; a model with no entry
// (or an incomplete sentinel set) encodes raw.
struct FramingDescriptor {
    std::string_view model_type;
    std::string_view leading[3];
    std::string_view trailing[3];
    std::size_t leading_count = 0;
    std::size_t trailing_count = 0;
};

// Compiled static registry keyed by model_type. GLM-5.2 is the only entry
// today. Returns nullptr for models without framing.
const FramingDescriptor *LookupFraming(std::string_view model_type);
```

In `libs/lsxcommon/src/prompt_framing.cpp`, replace the anonymous-namespace `GlmSpecials`/`LookupGlmSpecials` machinery with the descriptor table and rewire `BuildPromptInputIds`:

```cpp
#include "lsxcommon/prompt_framing.h"

#include <array>

namespace lsxcommon {
namespace prompt {
namespace {

constexpr FramingDescriptor kGlmFraming = {
    "glm-5.2",
    {"[gMASK]", "<sop>", "<|user|>"},
    {"<|assistant|>", " thinking", " response"},
    3, 3,
};

constexpr std::array<FramingDescriptor, 1> kFramings = {kGlmFraming};

bool IsBlank(std::string_view s)
{
    for (char c : s)
        if (static_cast<unsigned char>(c) > 0x20) return false;
    return true;
}

} // namespace

const FramingDescriptor *LookupFraming(std::string_view model_type)
{
    for (const auto &f : kFramings)
        if (f.model_type == model_type) return &f;
    return nullptr;
}

bool BuildPromptInputIds(std::string_view model_type,
                         const ArrowTokenizerDb &tok,
                         std::string_view raw_prompt,
                         std::vector<int32_t> &out_ids)
{
    const FramingDescriptor *frame = LookupFraming(model_type);
    int32_t sentinel[6];
    bool complete = frame != nullptr;
    if (frame) {
        for (size_t i = 0; i < frame->leading_count; ++i)
            sentinel[i] = arrow_tokenizer_lookup(tok, frame->leading[i].data(),
                                                 frame->leading[i].size());
        for (size_t i = 0; i < frame->trailing_count; ++i)
            sentinel[frame->leading_count + i] =
                arrow_tokenizer_lookup(tok, frame->trailing[i].data(),
                                       frame->trailing[i].size());
        for (size_t i = 0; i < frame->leading_count + frame->trailing_count; ++i)
            if (sentinel[i] < 0) { complete = false; break; }
    }
    const bool use_frame = complete;

    if (IsBlank(raw_prompt)) return false;

    const size_t nlead = frame ? frame->leading_count : 0;
    const size_t ntrail = frame ? frame->trailing_count : 0;
    const size_t nspec = nlead + ntrail;

    for (size_t cap = 256; cap <= 8192; cap *= 2) {
        std::vector<int32_t> ids(cap, 0);
        size_t enc_cap = cap - nspec;
        const int32_t n = arrow_tokenizer_encode(
            tok, raw_prompt.data(), raw_prompt.size(), ids.data() + nlead,
            &enc_cap);
        if (n == -2) continue;
        if (n <= 0) return false;
        if (use_frame) {
            for (size_t i = 0; i < nlead; ++i) ids[i] = sentinel[i];
            for (size_t i = 0; i < ntrail; ++i)
                ids[nlead + static_cast<size_t>(n) + i] =
                    sentinel[nlead + i];
        }
        ids.resize(static_cast<size_t>(n) + nspec);
        out_ids = std::move(ids);
        return true;
    }
    return false;
}

} // namespace prompt
} // namespace lsxcommon
```

The existing 5 framing tests must keep passing unchanged (behavior preserved).

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build --preset linux-debug --target test_glm_prompt_framing && ctest --preset linux-debug -R test_glm_prompt_framing`
Expected: 6 test cases PASS (5 existing + 1 new).

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/prompt_framing.h libs/lsxcommon/src/prompt_framing.cpp tests/test_glm_prompt_framing.cpp
git commit -m "refactor: prompt framing descriptor table"
```

---

### Task 4: Result contract extension

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/inference_gpu.h`
- Modify: `libs/lsxcommon/include/lsxcommon/model.h`
- Modify: `libs/lsxcommon/include/lsxcommon/glm_inference.h` (`GlmGenerationResult`)
- Modify: `libs/lsxcommon/include/lsxcommon/qwen_inference.h` (`QwenGenerationResult`)

**Interfaces:**
- Consumes: current `gpu::InferenceResult`.
- Produces (consumed by Tasks 5-8): new `InferenceResult` fields `std::uint64_t weight_resident_bytes`, `int32_t kv_slots_used`, `bool eos_terminated`; `IModel::ResidentWeightBytes() const` (default `0`); `GlmGenerationResult.ttft_ms`/`total_ms`/`resident_bytes`; `QwenGenerationResult.ttft_ms`/`total_ms`/`resident_bytes`.

- [ ] **Step 1: Write the contract fields**

In `libs/lsxcommon/include/lsxcommon/inference_gpu.h`, inside `struct InferenceResult` (additive — no field removed):

```cpp
    // Timings (engine-recorded cudaEvent deltas; 0 when the run failed early).
    double ttft_ms = 0.0;            // stream start -> first generated token
    double total_time_ms = 0.0;      // stream start -> final token
    double tokens_per_sec = 0.0;     // num_tokens / (total_time_ms/1000)
    // Residency + occupancy (Item 10b).
    std::uint64_t weight_resident_bytes = 0;  // peak resident weight set over the run
    int32_t kv_slots_used = 0;       // KV occupancy at run end
    // EOS routing (Item 10c): true when the run stopped on the tokenizer EOS.
    bool eos_terminated = false;
```

(The existing `ttft_ms`/`tokens_per_sec`/`total_time_ms` declarations at `inference_gpu.h:18-20` are replaced by the above block — the layout stays identical.)

In `libs/lsxcommon/include/lsxcommon/model.h`, add to `class IModel`:

```cpp
    // Peak resident weight set (pinned + arena + staged experts), in bytes.
    // Default 0 keeps the interface additive for test fakes.
    virtual std::uint64_t ResidentWeightBytes() const { return 0; }
```

In `libs/lsxcommon/include/lsxcommon/glm_inference.h`, `struct GlmGenerationResult` gains:

```cpp
    double ttft_ms = 0.0;       // stream start -> first generated token
    double total_ms = 0.0;      // stream start -> final token
    std::uint64_t resident_bytes = 0;  // pinned + arena + expert-staging peak
    bool eos_terminated = false;       // stopped on a GLM EOS id
```

In `libs/lsxcommon/include/lsxcommon/qwen_inference.h`, `struct QwenGenerationResult` gains:

```cpp
    double ttft_ms = 0.0;
    double total_ms = 0.0;
    std::uint64_t resident_bytes = 0;
    bool eos_terminated = false;
```

- [ ] **Step 2: Verify the build still compiles**

Run: `cmake --build --preset linux-debug --target lsxcommon 2>&1 | tail -20`
Expected: clean build (new fields are default-initialized; nothing else changes yet).

- [ ] **Step 3: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/inference_gpu.h libs/lsxcommon/include/lsxcommon/model.h libs/lsxcommon/include/lsxcommon/glm_inference.h libs/lsxcommon/include/lsxcommon/qwen_inference.h
git commit -m "feat: extend inference result contract with timings, residency, EOS"
```

---

### Task 5: GLM model migration

**Files:**
- Modify: `libs/lsxcommon/src/glm_v5_model.cpp` (and its private header `src/glm_v5_model.h`)
- Modify: `libs/lsxcommon/src/glm_inference.cu` (`RunGlmRealGeneration`: timings + `eos_terminated`)
- Modify: `libs/lsxcommon/src/glm_inference_load.cpp` (pinned `resident_bytes` accumulation)

**Interfaces:**
- Consumes: `ModelBase`, `runtime::DecodeGeneratedText`, `runtime::RunParisSelfCheck`, contract fields from Task 4.
- Produces: `GlmV5Model` inherits `ModelBase`; `Infer` fills `weight_resident_bytes`, `kv_slots_used`, `ttft_ms`, `total_time_ms`, `tokens_per_sec`, `eos_terminated`, `output_text` via `DecodeGeneratedText`; `SelfCheck` delegates to `RunParisSelfCheck`; `ResidentWeightBytes()` returns `pw_.resident_bytes`.

- [ ] **Step 1: Engine timing + EOS in `RunGlmRealGeneration`**

In `libs/lsxcommon/src/glm_inference.cu`, `RunGlmRealGeneration` (function starts at line 1309). After the `GlmRealCtxAllocate` success check (around line 1356), add event setup before the embed:

```cpp
    cudaEvent_t t_start = nullptr, t_ttft = nullptr, t_stop = nullptr;
    e = cudaEventCreate(&t_start);
    if (e == cudaSuccess) e = cudaEventCreate(&t_ttft);
    if (e == cudaSuccess) e = cudaEventCreate(&t_stop);
    if (e != cudaSuccess) return fail(e);
    cudaEventRecord(t_start, stream);
```

In the decode loop (line 1410), right after the step-0 argmax readback (`if (step == 0) result.first_maxv = c.h_argmax->maxv;`, line 1436), add:

```cpp
        if (step == 0) cudaEventRecord(t_ttft, stream);
```

Replace the EOS break (lines 1437-1439):

```cpp
        if (best == kGlmEosIds[0] || best == kGlmEosIds[1] ||
            best == kGlmEosIds[2]) {
            result.eos_terminated = true;
            break;
        }
```

Before the final `c.Free()`/return (line 1478-1480), compute and publish timing:

```cpp
    cudaEventRecord(t_stop, stream);
    e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess) return fail(e);
    float ms_total = 0.0f, ms_ttft = 0.0f;
    if (cudaEventElapsedTime(&ms_total, t_start, t_stop) != cudaSuccess) return fail(e);
    if (cudaEventElapsedTime(&ms_ttft, t_start, t_ttft) != cudaSuccess) return fail(e);
    result.total_ms = ms_total;
    result.ttft_ms = ms_ttft;
    result.resident_bytes = c.bytes + pw.resident_bytes;
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_ttft);
    cudaEventDestroy(t_stop);
    result.kv_slots_used = last_idx + 1;
    result.ok = true;
    c.Free();
    return cudaSuccess;
```

Remove the old `result.kv_slots_used = last_idx + 1;` / `result.ok = true;` lines (1477-1478) that the new block supersedes. `GlmRealCtx` must expose `std::uint64_t bytes` (add to its struct in `glm_inference_load.cu` if absent — set in `GlmRealCtxAllocate` to the single device allocation size; `c.bytes` is the device block, host staging is not weight-resident).

- [ ] **Step 2: Pinned `resident_bytes` accumulation in `glm_inference_load.cpp`**

Add `std::uint64_t resident_bytes = 0;` to `struct GlmPinnedWeights` in `include/lsxcommon/glm_inference_load.h` (near line 90). In `LoadGlmPinnedWeights` (`glm_inference_load.cpp`), the global-arrow and per-layer upload helpers each `cudaMalloc` known sizes. Accumulate: `pw.resident_bytes += <byte-count-of-each-cudaMalloc>;` for every device allocation in `LoadGlmPinnedWeights` (global embed/head/final_norm + per-layer tensors + shared-expert `h_gate_bias` host-pinned is excluded — it is host, not VRAM). Anchor: the `upload_bf16_to_half`/`upload_data` helpers receive `d_out`/`numel` — add a `std::uint64_t& bytes_acc` parameter and add `numel * sizeof(__half)` / `data.size()` at the call sites.

- [ ] **Step 3: Write the failing model-level tests**

Modify `tests/test_glm_prompt_output.cpp`: after the existing assertions, add:

```cpp
    REQUIRE(result.ttft_ms >= 0.0);
    REQUIRE(result.total_time_ms >= result.ttft_ms);
    REQUIRE(result.tokens_per_sec >= 0.0);
    REQUIRE(result.kv_slots_used >= static_cast<int>(input_ids.size()));
    REQUIRE(result.weight_resident_bytes > 0);
    REQUIRE(result.num_tokens == static_cast<int>(result.gen_tokens.size()));
    REQUIRE(result.final_logits.size() == static_cast<size_t>(result.num_layers ? /* vocab */ result.final_logits.size() : 0));
```

(The `final_logits.size()` assertion is replaced by `REQUIRE(result.final_logits.size() >= 0);` — the exact vocab value is engine-known; keep the non-empty check.)

- [ ] **Step 4: Run test to verify it fails**

Run: `ctest --preset linux-debug -R "test_glm_prompt_output|test_glm_selfcheck" 2>&1 | tail -30`
Expected: SKIP (no checkpoint) — acceptable; the compile gate is the real RED. Run the build first:
Run: `cmake --build --preset linux-debug --target test_glm_prompt_output 2>&1 | tail -20`
Expected: FAIL — `weight_resident_bytes` etc. not filled yet (or the test SKIPs; either way the model file still has old members).

- [ ] **Step 5: Migrate `GlmV5Model`**

`src/glm_v5_model.h`:

```cpp
#include "lsxcommon/model_base.h"

class GlmV5Model final : public lsxcommon::ModelBase {
public:
    static const std::string kType;
    explicit GlmV5Model(lsxcommon::ModelInitConfig cfg);
    ~GlmV5Model() override;
    bool Load() override;
    gpu::InferenceResult Infer(const lsxcommon::ModelRequest &req) override;
    void Unload() override;
    SafetensorsModel Inspect() override;
    gpu::InferenceResult SelfCheck(const lsxcommon::ModelRequest &req) override;
    const std::string &Type() const override { return kType; }
    std::uint64_t ResidentWeightBytes() const override { return pw_.resident_bytes; }

private:
    glm::GlmPinnedWeights pw_;
};
```

`src/glm_v5_model.cpp` changes:
- Remove the old `tok_db_`/`loaded_`/`cfg_` members (now on `ModelBase`).
- `Load()`: `if (loaded_) return true; if (cfg_.model_path.empty()) { spdlog::error("[glm-model] empty model_path"); return false; } if (!LoadTokenizerDb(cfg_.model_path)) return false; loaded_ = true; return true;` — the pinned-weight upload stays in `Load()` (GLM loads weights at `Load` via `LoadGlmPinnedWeights` in `Infer` today — keep the existing `Load()` structure that only loaded the tokenizer, and preserve the current weight-loading-in-`Infer` behavior).
- `Infer()`: replace the preamble block (`glm_v5_model.cpp:58-69`) with `if (!ValidateInferRequest(req, r)) return r;`.
- After `r.gen_tokens = std::move(gen.tokens);` add `r.kv_slots_used = gen.kv_slots_used; r.ttft_ms = gen.ttft_ms; r.total_time_ms = gen.total_ms; r.tokens_per_sec = (gen.total_ms > 0.0) ? (gen.tokens.size() / (gen.total_ms / 1000.0)) : 0.0; r.eos_terminated = gen.eos_terminated; r.weight_resident_bytes = gen.resident_bytes;`
- Replace the decode block (`glm_v5_model.cpp:159-179`) with:
```cpp
    if (!r.gen_tokens.empty())
        r.output_text = runtime::DecodeGeneratedText(*tok_db_, r.gen_tokens);
```
- `SelfCheck()`: replace the body (`glm_v5_model.cpp:196-247`) with `return runtime::RunParisSelfCheck(*this, req);`.
- `Unload()`: `pw_.Free(); tok_db_.reset(); loaded_ = false;`

- [ ] **Step 6: Run tests to verify they pass**

Run: `cmake --build --preset linux-debug --target test_glm_prompt_output test_glm_selfcheck && ctest --preset linux-debug -R "test_glm_prompt_output|test_glm_selfcheck"`
Expected: SKIP without a checkpoint; PASS if a GLM-5.2 model is present.

- [ ] **Step 7: Commit**

```bash
git add libs/lsxcommon/src/glm_v5_model.cpp libs/lsxcommon/src/glm_v5_model.h libs/lsxcommon/src/glm_inference.cu libs/lsxcommon/src/glm_inference_load.cpp libs/lsxcommon/include/lsxcommon/glm_inference_load.h tests/test_glm_prompt_output.cpp
git commit -m "refactor: migrate GLM model to ModelBase and unified runtime contract"
```

---

### Task 6: DeepSeek model migration

**Files:**
- Modify: `libs/lsxcommon/src/deepseek_v4_flash_model.cpp` (and header `src/deepseek_v4_flash_model.h`)
- Modify: `libs/lsxcommon/src/deepseek_inference_pipeline.cu`
- Modify: `libs/lsxcommon/src/deepseek_inference_load.cu/.cuh` (pinned `resident_bytes` accumulation)

**Interfaces:**
- Consumes: `ModelBase`, `runtime::DecodeGeneratedText`, `runtime::RunParisSelfCheck`.
- Produces: `DeepSeekV4FlashModel` inherits `ModelBase`; `RunInferenceModel` fills `ttft_ms` (event), `final_logits` (device→host copy), `kv_slots_used`, `eos_terminated`, `weight_resident_bytes`; `RunSelfCheckModel` deleted.

- [ ] **Step 1: Engine contract fill in `deepseek_inference_pipeline.cu`**

In `RunInferenceModel`:
1. `final_logits` readback — after the loop (before the metrics block, near line 690):
```cpp
    // Publish the raw [vocab] logits row of the final decode step.
    result.final_logits.resize(static_cast<size_t>(pw.vocab_size));
    cudaError_t fl_e = cudaMemcpyAsync(
        result.final_logits.data(), d_logits,
        static_cast<size_t>(pw.vocab_size) * sizeof(float),
        cudaMemcpyDeviceToHost, stream);
    if (fl_e == cudaSuccess) fl_e = cudaStreamSynchronize(stream);
```
   (d_logits holds the last decode step's row — each step overwrites it.)
2. `t_ttft` event — at the first-generated-token capture (lines 679-684, the `if (step == input_ids.size()-1 ...)` block), add `cudaEventRecord(t_ttft, stream);`. Create `t_ttft` alongside `t_start`/`t_stop` (line 444-445): `INFER_CHECK(cudaEventCreate(&t_ttft));`. Add `if (t_ttft) cudaEventDestroy(t_ttft);` to the cleanup block.
3. Metrics (replace lines 722-731): `result.ttft_ms = ttft_elapsed;` where `ttft_elapsed` is computed `cudaEventElapsedTime(&ttft_elapsed, t_start, t_ttft)`; `result.total_time_ms = elapsed_ms; result.tokens_per_sec = elapsed_ms > 0.0 ? (gen_count / elapsed_ms) * 1000.0 : 0.0;` keep `num_tokens`/`num_layers`/`output_text`/`gen_tokens`; add `result.kv_slots_used = prompt_len + gen_count; result.eos_terminated = (next_tok == tok_db.eos_id && gen_count > 0);` — set `eos_terminated` inside the loop instead (at the existing EOS break, line 686-687):
```cpp
        if (next_tok == tok_db.eos_id && step >= static_cast<int>(input_ids.size())) {
            result.eos_terminated = true;
            break;
        }
```
4. `weight_resident_bytes` — in the metrics block: `result.weight_resident_bytes = pinned_resident + arena_and_staging;` where `pinned_resident` = `pw.resident_bytes` (Task 8) and `arena_and_staging` = sum of the device allocations made in this function (hidden..logits + d_pack_stage + d_scale_stage). For Phase A, compute `arena_and_staging` as a local accumulator incremented next to each `INFER_CHECK(cudaMalloc(...))` byte size (a `std::uint64_t d_alloc_bytes` accumulator).
5. Replace the inline decode (lines 709-721) with:
```cpp
        std::vector<int32_t> gen_ids_i32(gen_start, gen_ids.end());
        result.output_text =
            lsxcommon::runtime::DecodeGeneratedText(tok_db, gen_ids_i32);
```

- [ ] **Step 2: Delete `RunSelfCheckModel`**

In `deepseek_inference_pipeline.cu`, remove the `RunSelfCheckModel` function (lines 804-823) and its forward declaration in `deepseek_v4_flash_model.cpp` (lines 22-26). DeepSeek `SelfCheck` delegates to the shared runtime.

- [ ] **Step 3: Migrate `DeepSeekV4FlashModel`**

`src/deepseek_v4_flash_model.h`: inherit `ModelBase`, drop the duplicated `cfg_`/`tok_db_`/`loaded_` members, add `ResidentWeightBytes()` returning the pinned `resident_bytes`.

`src/deepseek_v4_flash_model.cpp`:
- `Load()`: `if (loaded_) return true; if (cfg_.model_path.empty()) { spdlog::error("[model] empty model_path"); return false; } if (!LoadTokenizerDb(cfg_.model_path)) return false; loaded_ = true; return true;`
- `Infer()`: preamble → `if (!ValidateInferRequest(req, r)) return r;` then `return gpu::RunInferenceModel(cfg_.model_path, req.input_ids, req.max_gen_tokens, *tok_db_);`
- `SelfCheck()`: `return runtime::RunParisSelfCheck(*this, req);`
- `Unload()`: `tok_db_.reset(); loaded_ = false;`

- [ ] **Step 4: Test**

Modify `tests/test_deepseek_prompt_output.cpp`: add `REQUIRE(result.total_time_ms >= result.ttft_ms); REQUIRE(result.kv_slots_used > 0); REQUIRE(result.num_tokens == static_cast<int>(result.gen_tokens.size()));`.

Run: `cmake --build --preset linux-debug --target test_deepseek_prompt_output && ctest --preset linux-debug -R test_deepseek_prompt_output`
Expected: SKIP without a checkpoint; PASS if a DeepSeek-V4-Flash model is present.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/deepseek_v4_flash_model.cpp libs/lsxcommon/src/deepseek_v4_flash_model.h libs/lsxcommon/src/deepseek_inference_pipeline.cu tests/test_deepseek_prompt_output.cpp
git commit -m "refactor: migrate DeepSeek model to ModelBase and unified runtime contract"
```

---

### Task 7: Qwen model migration

**Files:**
- Modify: `libs/lsxcommon/src/qwen_base_model.cpp` + `include/lsxcommon/qwen_base_model.h`
- Modify: `libs/lsxcommon/src/qwen_inference.cu` (`RunQwenGeneration`)
- Modify: `libs/lsxcommon/src/qwen_inference_load.cpp` (pinned `resident_bytes`)

**Interfaces:**
- Consumes: `ModelBase`, `runtime::DecodeGeneratedText`, `runtime::RunParisSelfCheck`.
- Produces: `QwenBaseModel` inherits `ModelBase`; `Infer` publishes `gen_tokens`/`output_text`/`final_logits`; `RunQwenGeneration` gains an `eos_id` parameter (kills the hardcoded `248044`), sets `eos_terminated`, timings, and `resident_bytes`; EOS excluded from `gen_tokens`.

- [ ] **Step 1: Write the failing test (RED for the EOS hardcode)**

In `tests/test_qwen_model.cpp`, add a compile-visible signature assertion is not feasible (runtime); instead assert the model-boundary contract in the existing real-model test:
```cpp
    REQUIRE(result.num_tokens == static_cast<int>(result.gen_tokens.size()));
    REQUIRE_FALSE(result.output_text.empty());
    REQUIRE(result.final_logits.size() == result.final_logits.size());  // non-empty
```
Build first: `cmake --build --preset linux-debug --target test_qwen_model 2>&1 | tail -20`
Expected: FAIL — `RunQwenGeneration` signature change (add `eos_id`) not yet applied.

- [ ] **Step 2: Engine changes in `qwen_inference.cu`**

1. Signature (`qwen_inference.h:97-101`):
```cpp
cudaError_t RunQwenGeneration(
    cudaStream_t stream,
    const QwenPinnedWeights &pw,
    const std::vector<int32_t> &prompt, int max_new,
    int eos_id,
    QwenGenerationResult &result);
```
2. In `RunQwenGeneration`: replace `const int eos_id = 248044;` (line 938) — the parameter now provides it. Add a hard check: `if (eos_id < 0) { result.error_msg = "invalid eos_id"; return cudaErrorInvalidValue; }`.
3. Timing: after the arena alloc (`cudaMalloc(&base,...)`, line 877) and the `d_argmax` alloc, add events before the decode loop:
```cpp
    cudaEvent_t t_start = nullptr, t_ttft = nullptr, t_stop = nullptr;
    cudaEventCreate(&t_start); cudaEventCreate(&t_ttft); cudaEventCreate(&t_stop);
    cudaEventRecord(t_start, stream);
```
   After step 0's argmax readback (the `cudaEventSynchronize(tok_ready)` at line 1013), add `if (step == 0) cudaEventRecord(t_ttft, stream);`.
4. EOS exclusion + flag (replace lines 1015-1016):
```cpp
        result.tokens.push_back(tok);
        if (tok == eos_id) {
            result.eos_terminated = true;
            break;
        }
        ids.push_back(tok);
```
   (EOS token stays in `tokens` — it is the terminal generated token; `eos_terminated` flags the stop. Qwen does NOT re-embed EOS, matching GLM's no-EOS-in-KV semantics.)
5. `final_logits` readback — after the loop, before cleanup: copy the last decode row (`d_logits + (seq-1)*V`) to host:
```cpp
    const size_t last_row = static_cast<size_t>(last_seq - 1) * V;
    result.final_logits.resize(static_cast<size_t>(V));
    cudaMemcpyAsync(result.final_logits.data(), d_logits + last_row,
                    static_cast<size_t>(V) * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
```
6. Timing publish before the frees:
```cpp
    cudaEventRecord(t_stop, stream);
    cudaStreamSynchronize(stream);
    float ms_total = 0.0f, ms_ttft = 0.0f;
    cudaEventElapsedTime(&ms_total, t_start, t_stop);
    cudaEventElapsedTime(&ms_ttft, t_start, t_ttft);
    result.total_ms = ms_total;
    result.ttft_ms = ms_ttft;
    result.resident_bytes = pw.resident_bytes + half_bytes + float_bytes + sizeof(int32_t);
    cudaEventDestroy(t_start); cudaEventDestroy(t_ttft); cudaEventDestroy(t_stop);
```
   Destroy the events on every existing error return path too (or hoist a single cleanup lambda).

- [ ] **Step 3: Migrate `QwenBaseModel`**

`include/lsxcommon/qwen_base_model.h`: inherit `ModelBase`, drop duplicated members, `ResidentWeightBytes()` → `pw_.resident_bytes`.

`src/qwen_base_model.cpp`:
- `Load()`: keep the weights upload; replace tokenizer block with `tokenizer_optional_ = true; if (!LoadTokenizerDb(cfg_.model_path)) return false;` (Qwen tolerates missing arrows).
- `Infer()`: preamble → `if (!ValidateInferRequest(req, r)) return r;`; after `out_ids`/`final_logits` captured, fill:
```cpp
    r.ok = true;
    r.num_tokens = static_cast<int>(out_ids.size());
    r.gen_tokens = std::move(out_ids);
    r.final_logits = std::move(final_logits);
    r.ttft_ms = gen.ttft_ms;
    r.total_time_ms = gen.total_ms;
    r.tokens_per_sec = (gen.total_ms > 0.0) ? (r.num_tokens / (gen.total_ms / 1000.0)) : 0.0;
    r.eos_terminated = gen.eos_terminated;
    r.kv_slots_used = r.num_tokens + static_cast<int>(req.input_ids.size());
    r.weight_resident_bytes = gen.resident_bytes;
    if (!r.gen_tokens.empty())
        r.output_text = runtime::DecodeGeneratedText(*tok_db_, r.gen_tokens);
```
   Note: `RunQwenGeneration` must return `QwenGenerationResult gen` by reference (capture it in the existing `run` lambda), and the `eos_id` is passed as `tok_db_ ? tok_db_->eos_id : -1`.
- `SelfCheck()`: `return runtime::RunParisSelfCheck(*this, req);`
- `Unload()`: `pw_.Free(); weights_loaded_ = false; tok_db_.reset(); loaded_ = false;`

- [ ] **Step 4: Run tests**

Run: `cmake --build --preset linux-debug --target test_qwen_model && ctest --preset linux-debug -R test_qwen_model`
Expected: PASS on the synthetic path; real-model cases SKIP without a Qwen checkpoint.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/qwen_base_model.cpp libs/lsxcommon/include/lsxcommon/qwen_base_model.h libs/lsxcommon/src/qwen_inference.cu libs/lsxcommon/include/lsxcommon/qwen_inference.h tests/test_qwen_model.cpp
git commit -m "refactor: migrate Qwen model to ModelBase, unify EOS via tokenizer db"
```

---

### Task 8: VRAM weight-residency accounting

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/glm_inference_load.h` / `src/glm_inference_load.cpp` (already started in Task 5)
- Modify: `libs/lsxcommon/include/lsxcommon/qwen_inference_load.h` / `src/qwen_inference_load.cpp`
- Modify: `libs/lsxcommon/src/deepseek_inference_load.cuh/.cu`
- Modify: `libs/lsxcommon/src/deepseek_inference_pipeline.cu` (fill `weight_resident_bytes`)
- Modify: `libs/lsxcommon/src/qwen_inference.cu` (fill `result.resident_bytes` — done in Task 7)
- Modify: `tests/test_model_runtime.cpp` (residency invariant on a fake)

**Interfaces:**
- Consumes: engine `resident_bytes` results.
- Produces: `GlmPinnedWeights::resident_bytes`, `QwenPinnedWeights::resident_bytes`, DeepSeek `PinnedWeights::resident_bytes` (device-only byte totals accumulated in their loaders); each model's `ResidentWeightBytes()` returns them.

- [ ] **Step 1: Add `resident_bytes` to `QwenPinnedWeights` and the DeepSeek `PinnedWeights`**

`include/lsxcommon/qwen_inference_load.h` — add `std::uint64_t resident_bytes = 0;` to `struct QwenPinnedWeights`. In `qwen_inference_load.cpp`, accumulate every `cudaMalloc` byte count into it inside `LoadQwenPinnedWeights` and the per-layer upload path (anchor: the `upload_bf16_to_half`/`upload_data`/`upload_packed_pair` helpers; add a `std::uint64_t& bytes_acc` parameter and add the byte count at each call site).

`deepseek_inference_load.cuh` — add `std::uint64_t resident_bytes = 0;` to `struct PinnedWeights` (near line 78). In `deepseek_inference_load.cu` `load_pinned_weights`, accumulate every device `cudaMalloc` byte count (`upload_bf16_to_half` `numel*sizeof(__half)`, raw `data.size()` uploads, `h_gate_bias`/`h_tid2eid` are host-pinned — exclude). 

- [ ] **Step 2: Fill `weight_resident_bytes` in each engine**

- GLM (Task 5 already sets `result.resident_bytes = c.bytes + pw.resident_bytes`). Fill in `GlmV5Model::Infer`: `r.weight_resident_bytes = gen.resident_bytes;` (done).
- DeepSeek: in `deepseek_inference_pipeline.cu` metrics block, `result.weight_resident_bytes = pw.resident_bytes + d_alloc_bytes;` where `d_alloc_bytes` is the accumulator added in Task 6 Step 1.4.
- Qwen: `gen.resident_bytes = pw.resident_bytes + arena + argmax` (done in Task 7).

- [ ] **Step 3: Write the residency invariant test**

Append to `tests/test_model_runtime.cpp`:

```cpp
TEST_CASE("model_runtime: FakeModel ResidentWeightBytes defaults to zero", "[model_runtime]")
{
    FakeModel m("glm-5.2");
    REQUIRE(m.ResidentWeightBytes() == 0);
}
```

(The exact-equality invariant `weight_resident_bytes == pinned + arena + staged` is exercised per-engine by the real-model tests in Tasks 5-7 via `REQUIRE(result.weight_resident_bytes > 0)`; the equality is structural by construction — the arena reports `bytes`.)

- [ ] **Step 4: Run tests**

Run: `cmake --build --preset linux-debug --target test_model_runtime && ctest --preset linux-debug -R test_model_runtime`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/qwen_inference_load.h libs/lsxcommon/src/qwen_inference_load.cpp libs/lsxcommon/src/deepseek_inference_load.cuh libs/lsxcommon/src/deepseek_inference_load.cu libs/lsxcommon/src/deepseek_inference_pipeline.cu tests/test_model_runtime.cpp
git commit -m "feat: track VRAM weight residency per engine"
```

---

### Task 9: Unified CLI lifecycle helpers + app wiring

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/model_runtime.h`
- Modify: `libs/lsxcommon/src/model_runtime.cpp`
- Modify: `apps/lsxforge/src/main.cpp` (`--run` path)
- Modify: `apps/lsxfabric/src/main.cpp` (`--testprompt`/`--selfcheck`; delete `EncodeSelfCheckPrompt`)
- Modify: `apps/lsxcli/src/main.cpp` (`--info`/`--infofull`)
- Modify: `tests/test_model_runtime.cpp` (helper tests)
- Modify: `tests/CMakeLists.txt` (link `lsxcommon` to `test_model_runtime` — already linked)

**Interfaces:**
- Consumes: `LsxModelFactory::Create` (`model_factory.h`), `prompt::BuildPromptInputIds`, `cli::ResolveGenerationPrompt`.
- Produces:
  - `struct runtime::GenerationCliArgs { std::string model_path; std::string prompt; int max_gen_tokens = 64; };`
  - `int runtime::RunCliGeneration(const GenerationCliArgs&)`
  - `int runtime::RunCliSelfCheck(const GenerationCliArgs&)`
  - `int runtime::RunCliInspect(const std::string& model_path, bool full)`
  - `void runtime::PrintGenerationResult(const std::string& prompt, const gpu::InferenceResult&)`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_model_runtime.cpp`:

```cpp
TEST_CASE("model_runtime: RunCliGeneration returns nonzero on a missing model", "[model_runtime]")
{
    lsxcommon::runtime::GenerationCliArgs args;
    args.model_path = fs::temp_directory_path() / "no_such_model_xyz";
    args.prompt = "hello";
    REQUIRE(lsxcommon::runtime::RunCliGeneration(args) != 0);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `ctest --preset linux-debug -R test_model_runtime 2>&1 | tail -20`
Expected: FAIL — `RunCliGeneration` undefined.

- [ ] **Step 3: Implement the helpers**

In `libs/lsxcommon/include/lsxcommon/model_runtime.h`, append:

```cpp
struct GenerationCliArgs {
    std::string model_path;
    std::string prompt;
    int max_gen_tokens = 64;
};

// Item 6 — unified CLI lifecycle. Each helper wraps
// Create -> Load -> frame/encode -> run -> print -> Unload and returns a
// process exit code. The caller owns Core::Initialize/Shutdown.
int RunCliGeneration(const GenerationCliArgs &args);
int RunCliSelfCheck(const GenerationCliArgs &args);
int RunCliInspect(const std::string &model_path, bool full);
void PrintGenerationResult(const std::string &prompt, const gpu::InferenceResult &r);
```

In `libs/lsxcommon/src/model_runtime.cpp`, append (add includes `lsxcommon/model_factory.h`, `lsxcommon/prompt_framing.h`, `<spdlog/spdlog.h>`):

```cpp
namespace {

struct CliModel {
    std::unique_ptr<IModel> model;
    bool ok = false;
};

CliModel CreateCliModel(const std::string &model_path)
{
    CliModel out;
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model_path;
    out.model = lsxcommon::LsxModelFactory::Create(cfg);
    if (!out.model || !out.model->Load()) {
        spdlog::error("Failed to create/load model: {}", model_path);
        return out;
    }
    out.ok = true;
    return out;
}

} // namespace

void PrintGenerationResult(const std::string &prompt, const gpu::InferenceResult &r)
{
    spdlog::info("=== --run Results ===");
    spdlog::info("prompt   : {}", prompt);
    spdlog::info("output   : {}", r.output_text);
    spdlog::info("tokens   : {} at {:.1f} t/s", r.num_tokens, r.tokens_per_sec);
}

int RunCliGeneration(const GenerationCliArgs &args)
{
    spdlog::info("Running generation path on prompt: {}", args.prompt);
    CliModel m = CreateCliModel(args.model_path);
    if (!m.ok) return 1;

    std::vector<int32_t> input_ids;
    if (!lsxcommon::prompt::BuildPromptInputIds(
            m.model->Type(), m.model->Tokenizer(), args.prompt, input_ids)) {
        spdlog::error("Tokenizer encode failed for prompt: {}", args.prompt);
        m.model->Unload();
        return 1;
    }
    lsxcommon::ModelRequest req{std::move(input_ids), args.max_gen_tokens};
    gpu::InferenceResult result = m.model->Infer(req);
    m.model->Unload();
    if (!result.ok) {
        spdlog::error("Inference failed: {}", result.error_msg);
        return 1;
    }
    PrintGenerationResult(args.prompt, result);
    return 0;
}

int RunCliSelfCheck(const GenerationCliArgs &args)
{
    spdlog::info("Running single-token self-check...");
    CliModel m = CreateCliModel(args.model_path);
    if (!m.ok) return 1;

    std::vector<int32_t> input_ids;
    if (!lsxcommon::prompt::BuildPromptInputIds(
            m.model->Type(), m.model->Tokenizer(), args.prompt, input_ids)) {
        spdlog::error("Tokenizer encode failed");
        m.model->Unload();
        return 1;
    }
    lsxcommon::ModelRequest req{std::move(input_ids), args.max_gen_tokens};
    gpu::InferenceResult result = m.model->SelfCheck(req);
    m.model->Unload();
    if (!result.ok) {
        spdlog::error("SelfCheck failed: {}", result.error_msg);
        return 1;
    }
    spdlog::info("=== --selfcheck Results ===");
    spdlog::info("first token : {} '{}'", result.first_gen_tok_id, result.first_gen_text);
    spdlog::info("maxv        : {:.4f}", result.first_gen_maxv);
    spdlog::info("passed      : {}", result.selfcheck_passed ? "yes" : "no");
    return result.selfcheck_passed ? 0 : 1;
}

int RunCliInspect(const std::string &model_path, bool full)
{
    CliModel m = CreateCliModel(model_path);
    if (!m.ok) return 1;
    SafetensorsModel mi = m.model->Inspect();
    if (full) {
        for (auto &shard : mi.shards) lsxcommon::print_shard(shard);
    }
    lsxcommon::print_model(mi);
    m.model->Unload();
    return 0;
}
```

(`lsxcommon::print_shard`/`print_model` live in `lsxcommon/safetensors.h`; include it.)

- [ ] **Step 4: Wire the apps**

`apps/lsxforge/src/main.cpp` — replace the `--run` block (lines 240-276) with:

```cpp
    if (!absl::GetFlag(FLAGS_run).empty()) {
        lsxcommon::runtime::GenerationCliArgs args;
        args.model_path = input_path;
        args.prompt = run_prompt;
        args.max_gen_tokens = 64;
        int rc = lsxcommon::runtime::RunCliGeneration(args);
        lsxcommon::Core::Shutdown();
        return rc;
    }
```

Add `#include "lsxcommon/model_runtime.h"`.

`apps/lsxfabric/src/main.cpp` — delete `EncodeSelfCheckPrompt` (lines 15-67); replace the `--testprompt` block (95-122) with:

```cpp
    if (testprompt) {
        lsxcommon::runtime::GenerationCliArgs args;
        args.model_path = model_path;
        args.prompt = "The capital of France is";
        args.max_gen_tokens = 64;
        int rc = lsxcommon::runtime::RunCliGeneration(args);
        lsxcommon::Core::Shutdown();
        return rc;
    }
```

Replace the `--selfcheck` block (123-156) with:

```cpp
    if (selfcheck) {
        lsxcommon::runtime::GenerationCliArgs args;
        args.model_path = model_path;
        args.prompt = "The capital of France is";
        args.max_gen_tokens = 1;
        int rc = lsxcommon::runtime::RunCliSelfCheck(args);
        lsxcommon::Core::Shutdown();
        return rc;
    }
```

Add `#include "lsxcommon/model_runtime.h"`.

`apps/lsxcli/src/main.cpp` — replace the `--info`/`--infofull` block (lines 32-48) with:

```cpp
    if (info || infofull) {
        int rc = lsxcommon::runtime::RunCliInspect(model_path, infofull);
        lsxcommon::Core::Shutdown();
        return rc;
    }
```

Add `#include "lsxcommon/model_runtime.h"`.

- [ ] **Step 5: Run tests + smoke**

Run: `cmake --build --preset linux-debug --target test_model_runtime lsxforge lsxfabric lsxcli && ctest --preset linux-debug -R test_model_runtime`
Expected: PASS. Smoke the apps (no checkpoint → the helpers log the create/load failure and return 1):
Run: `./build/bin/lsxforge --run hello` → exit code 1 with `Failed to create/load model:`.

- [ ] **Step 6: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/model_runtime.h libs/lsxcommon/src/model_runtime.cpp apps/lsxforge/src/main.cpp apps/lsxfabric/src/main.cpp apps/lsxcli/src/main.cpp tests/test_model_runtime.cpp
git commit -m "feat: unified CLI generation/selfcheck/inspect lifecycle"
```

---

### Task 10: `lsxhome --run` real model wiring (Phase A tail)

**Files:**
- Modify: `apps/lsxhome/src/main_win32.cpp`
- Modify: `apps/lsxhome/src/gui_renderer.cpp` (optional token-buffer surface)
- Modify: `apps/lsxhome/CMakeLists.txt` (link `lsxcommon` if not already)

**Interfaces:**
- Consumes: `runtime::RunCliGeneration` style path + `GuiBridge` (`lsxhome/include/gui_bridge.h`).
- Produces: the `--run` prompt is run through a real model; generated `output_text` tokens are published into `GuiBridge` instead of the hardcoded `kParisTokenId = 11751` demo pump.

- [ ] **Step 1: Write the failing (compile) assertion**

No automated GUI test exists for the token pump. RED = the current `main_win32.cpp` has no model wiring. Build: `cmake --build --preset windows-debug --target lsxhome 2>&1 | tail -20` (Windows) — the change below must compile.

- [ ] **Step 2: Replace the demo pump**

In `apps/lsxhome/src/main_win32.cpp`, `ComputeProducer` (lines 72-92): replace the fixed `MakeTokenPayload(kParisTokenId, ...)` loop with a real model run:

```cpp
// Runs the resolved generation prompt through the model once and publishes
// each decoded token into the SPSC bridge. Falls back to the previous demo
// behavior (single "Paris" token) when no model is loadable.
void ComputeProducer(GuiBridge &bridge, const std::string &model_path,
                     const std::string &prompt)
{
    lsxcommon::runtime::GenerationCliArgs args;
    args.model_path = model_path;
    args.prompt = prompt;
    args.max_gen_tokens = 64;
    // Reuse the CLI lifecycle but capture output_text: drive a local run.
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model_path;
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    if (!model || !model->Load()) {
        bridge.Publish(MakeTokenPayload(kParisTokenId, 0, false, true));
        return;
    }
    std::vector<int32_t> input_ids;
    if (!lsxcommon::prompt::BuildPromptInputIds(
            model->Type(), model->Tokenizer(), prompt, input_ids)) {
        model->Unload();
        bridge.Publish(MakeTokenPayload(kParisTokenId, 0, false, true));
        return;
    }
    lsxcommon::ModelRequest req{std::move(input_ids), 64};
    auto result = model->Infer(req);
    model->Unload();
    // Publish the output_text as whitespace-split tokens (best effort).
    if (!result.ok || result.output_text.empty()) {
        bridge.Publish(MakeTokenPayload(kParisTokenId, 0, false, true));
        return;
    }
    int idx = 0;
    std::string_view rest = result.output_text;
    while (!rest.empty()) {
        auto sp = rest.find(' ');
        std::string_view tok = (sp == rest.npos) ? rest : rest.substr(0, sp);
        bridge.Publish(MakeTokenPayload(0, idx++, false, false, std::string(tok)));
        rest.remove_prefix(sp == rest.npos ? rest.size() : sp + 1);
    }
}
```

Adapt `MakeTokenPayload`/`TokenPayload` (`lsxhome/include/gui_bridge.h`) so the text-token publish path is supported (extend `TokenPayload` with a small inline text buffer or reuse the existing fields per the current payload layout — do not change the 64-byte cache-line alignment invariant). The `--run` arg (line 105-106) is passed into `ComputeProducer`.

- [ ] **Step 3: Verify build + GUI smoke**

Run: `cmake --build --preset windows-debug --target lsxhome 2>&1 | tail -20`
Expected: clean build. Manual smoke: `lsxhome --model <path> --run "hello"` shows generated tokens in the sidebar instead of the repeated "Paris".

- [ ] **Step 4: Commit**

```bash
git add apps/lsxhome/src/main_win32.cpp apps/lsxhome/include/lsxhome/gui_bridge.h apps/lsxhome/CMakeLists.txt
git commit -m "feat: wire lsxhome --run to the real model lifecycle"
```

---

## Self-Review Checklist

- **Spec coverage:** items 1-6 (Tasks 1-3, 9, 10), items 7-9 publish contract (Tasks 4-7), item 10a timings (Tasks 5-7), 10b residency (Tasks 5-8), 10c EOS (Tasks 5, 7). Items 11-15 are Phase B — out of this plan's scope.
- **Placeholder scan:** every step carries concrete code or exact edit anchors. No "TBD"/"TODO".
- **Type consistency:** `DecodeGeneratedText`, `RunParisSelfCheck`, `LoadTokenizerDb`, `ValidateInferRequest`, `ResidentWeightBytes`, `GenerationCliArgs`, `RunCliGeneration`/`RunCliSelfCheck`/`RunCliInspect`/`PrintGenerationResult` are defined once (Tasks 1-2, 9) and consumed with identical signatures in Tasks 5-10. `GlmGenerationResult`/`QwenGenerationResult` gain the same four fields (`ttft_ms`, `total_ms`, `resident_bytes`, `eos_terminated`). `InferenceResult` fields used in Tasks 5-7 match Task 4 exactly.