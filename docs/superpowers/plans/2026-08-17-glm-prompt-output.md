# GLM-5.2 Meaningful `--run` Generation Output — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make GLM-5.2 produce a meaningful, tokenizer-decoded response (`output_text`) on the `lsxforge --run "PROMPT"` generation path, verified by TDD.

**Architecture:** Three small changes: (A) a host-only framing helper `lsxcommon::prompt::BuildPromptInputIds` that wraps GLM-5.2 prompts in the chat template `[gMASK]<sop><|user|>{prompt}<|assistant|><think></think>`; (B) `GlmV5Model::Infer` decodes `gen.tokens` into `output_text` via `arrow_tokenizer_decode` (capacity-retry, DeepSeek pattern); (C) `lsxforge --run` routes the resolved prompt through the helper. Tests: host-only framing tests on a synthetic GLM tokenizer fixture, plus a real-model test asserting `output_text` contains `"Paris"` for `"The capital of France is"`.

**Tech Stack:** C++20, CMake/Ninja (preset `linux-debug`), Catch2 v3.7.0, Apache Arrow IPC, simdjson tokenizer fixtures. No new dependencies.

## Global Constraints

- C++20 (project pins C++20; C++23 allowed only via GCC 15 defaults, do not change `CMAKE_CXX_STANDARD`).
- All logging via spdlog — no `std::cout`/`printf`.
- No comments in code unless they document an invariant (existing code style).
- GLM-5.2 special-token framing is ONLY the non-thinking mode template
  `[gMASK]<sop><|user|>{prompt}<|assistant|><think></think>` (exact strings: `[gMASK]`, `<sop>`, `<|user|>`, `<|assistant|>`, `<think>`, `</think>`).
- GLM never auto-prepends BOS; the helper must NOT add one.
- `gen.tokens` from `RunGlmRealGeneration` contains ONLY generated tokens (prompt excluded).
- Real-model tests SKIP (Catch2 `SKIP`, CTest `SKIP_RETURN_CODE 4`) when `/mnt/llm_models/models/GLM-5.2/logestix/pinned` is absent.
- `lsxhome` is out of scope — it now lives in its own repository.
- Commit messages: English, prefixed `feat:` / `test:` / `docs:`.

---
---

### Task 1: Host-only framing tests (RED)

**Files:**
- Create: `tests/data/synthetic_glm_tok.json` (generated from `tests/data/synthetic_tok.json`)
- Create: `tests/test_glm_prompt_framing.cpp`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `lsxcommon::BuildTokenizerArrowFiles` (declared in `lsxcommon/converter.h`), `lsxcommon::load_tokenizer_metadata`, `lsxcommon::arrow_tokenizer_encode` / `arrow_tokenizer_lookup` (`lsxcommon/tokenizer.h`).
- Produces: a test-side fixture `synthetic_glm_tok.json` whose tokenizer vocab contains the six GLM special tokens at ids 268–273 (vocab_size 274).

- [ ] **Step 1: Generate the GLM tokenizer fixture**

Run from the repo root (`python3` is available on Ubuntu 26.04):

```bash
python3 - <<'PY'
import json
src = 'tests/data/synthetic_tok.json'
with open(src) as f:
    d = json.load(f)
glm = ['[gMASK]', '<sop>', '<|user|>', '<|assistant|>', '<think>', '</think>']
next_id = max(d['model']['vocab'].values()) + 1
for i, tok in enumerate(glm):
    _id = next_id + i
    d['added_tokens'].append({
        "id": _id, "content": tok, "single_word": False,
        "lstrip": False, "rstrip": False, "normalized": False, "special": True,
    })
    d['model']['vocab'][tok] = _id
with open('tests/data/synthetic_glm_tok.json', 'w') as f:
    json.dump(d, f, ensure_ascii=False)
PY
```

Expected: no output; `tests/data/synthetic_glm_tok.json` is created. Verify:

```bash
python3 -c "import json; d=json.load(open('tests/data/synthetic_glm_tok.json')); print(d['added_tokens'][-1]); print(max(d['model']['vocab'].values())+1)"
```

Expected: the last added token is `</think>` with id 273, and `max(id)+1 == 274`.

Note: the parser in `tokenizer_convert.cpp` requires `added_tokens` to precede `"model"` in the JSON stream; `json.dump` preserves key order from `json.load`, so this holds automatically.

- [ ] **Step 2: Write the failing test `tests/test_glm_prompt_framing.cpp`**

```cpp
// TDD — GLM-5.2 prompt framing helper (host-only, no CUDA, no real model).
#include <catch2/catch_test_macros.hpp>

#include <arrow/io/file.h>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <lsxcommon/converter.h>
#include <lsxcommon/prompt_framing.h>
#include <lsxcommon/tokenizer.h>

namespace fs = std::filesystem;

namespace {

struct DbFiles {
    fs::path pinned;
    lsxcommon::ArrowTokenizerDb db;
};

DbFiles BuildDb(const std::string &fixture_name)
{
    DbFiles out;
    out.pinned = fs::temp_directory_path() / ("logestix_pf_" + fixture_name);
    fs::remove_all(out.pinned);
    fs::create_directories(out.pinned);
    REQUIRE(lsxcommon::BuildTokenizerArrowFiles(
                TOKENIZER_TEST_DATA_DIR "/" + fixture_name, out.pinned)
                .ok());
    auto v = arrow::io::MemoryMappedFile::Open(
        (out.pinned / "tokenizer_vocab.arrow").string(),
        arrow::io::FileMode::READ);
    REQUIRE(v.ok());
    auto m = arrow::io::MemoryMappedFile::Open(
        (out.pinned / "tokenizer_merges.arrow").string(),
        arrow::io::FileMode::READ);
    REQUIRE(m.ok());
    REQUIRE(lsxcommon::load_tokenizer_metadata(*v, *m, out.db).ok());
    return out;
}

std::vector<int32_t> RawEncode(const lsxcommon::ArrowTokenizerDb &db,
                               const char *text)
{
    std::vector<int32_t> ids(256, 0);
    size_t cap = ids.size();
    const int32_t n = lsxcommon::arrow_tokenizer_encode(
        db, text, std::strlen(text), ids.data(), &cap);
    REQUIRE(n > 0);
    ids.resize(static_cast<size_t>(n));
    return ids;
}

} // namespace

TEST_CASE("prompt_framing: GLM chat-template framing order", "[prompt_framing]")
{
    DbFiles f = BuildDb("synthetic_glm_tok.json");
    const auto &db = f.db;
    REQUIRE(db.vocab_size == 274);

    const int32_t gmask = lsxcommon::arrow_tokenizer_lookup(db, "[gMASK]", 7);
    const int32_t sop   = lsxcommon::arrow_tokenizer_lookup(db, "<sop>", 5);
    const int32_t usr   = lsxcommon::arrow_tokenizer_lookup(db, "<|user|>", 8);
    const int32_t asst  = lsxcommon::arrow_tokenizer_lookup(db, "<|assistant|>", 13);
    const int32_t think = lsxcommon::arrow_tokenizer_lookup(db, "<think>", 7);
    const int32_t ethink = lsxcommon::arrow_tokenizer_lookup(db, "</think>", 8);
    REQUIRE(gmask >= 0 && sop >= 0 && usr >= 0 && asst >= 0 &&
            think >= 0 && ethink >= 0);

    const char *prompt = "The capital of France is";
    const std::vector<int32_t> raw = RawEncode(db, prompt);

    std::vector<int32_t> expected = {gmask, sop, usr};
    expected.insert(expected.end(), raw.begin(), raw.end());
    expected.push_back(asst);
    expected.push_back(think);
    expected.push_back(ethink);

    std::vector<int32_t> ids;
    REQUIRE(lsxcommon::prompt::BuildPromptInputIds("glm-5.2", db, prompt, ids));
    REQUIRE(ids == expected);
    REQUIRE(ids.size() == expected.size());

    f.db = lsxcommon::ArrowTokenizerDb{};
    fs::remove_all(f.pinned);
}

TEST_CASE("prompt_framing: non-GLM model type is encoded raw", "[prompt_framing]")
{
    DbFiles f = BuildDb("synthetic_tok.json");
    const char *prompt = "hello world";
    const std::vector<int32_t> expected = RawEncode(f.db, prompt);

    std::vector<int32_t> ids;
    REQUIRE(lsxcommon::prompt::BuildPromptInputIds(
        "deepseek-v4-flash", f.db, prompt, ids));
    REQUIRE(ids == expected);

    f.db = lsxcommon::ArrowTokenizerDb{};
    fs::remove_all(f.pinned);
}

TEST_CASE("prompt_framing: GLM falls back to raw when specials missing", "[prompt_framing]")
{
    DbFiles f = BuildDb("synthetic_tok.json");
    const char *prompt = "hello world";
    const std::vector<int32_t> expected = RawEncode(f.db, prompt);

    std::vector<int32_t> ids;
    REQUIRE(lsxcommon::prompt::BuildPromptInputIds("glm-5.2", f.db, prompt, ids));
    REQUIRE(ids == expected);

    f.db = lsxcommon::ArrowTokenizerDb{};
    fs::remove_all(f.pinned);
}

TEST_CASE("prompt_framing: empty prompt is rejected", "[prompt_framing]")
{
    DbFiles f = BuildDb("synthetic_tok.json");
    std::vector<int32_t> ids(4, 0);
    REQUIRE_FALSE(lsxcommon::prompt::BuildPromptInputIds("glm-5.2", f.db, "", ids));
    f.db = lsxcommon::ArrowTokenizerDb{};
    fs::remove_all(f.pinned);
}

TEST_CASE("prompt_framing: long prompt grows past the initial capacity", "[prompt_framing]")
{
    DbFiles f = BuildDb("synthetic_glm_tok.json");
    std::string long_text;
    for (int i = 0; i < 1200; ++i) long_text += "ab ";

    std::vector<int32_t> ids;
    REQUIRE(lsxcommon::prompt::BuildPromptInputIds(
        "glm-5.2", f.db, long_text, ids));
    REQUIRE(ids.size() > 256);
    REQUIRE(ids[0] == lsxcommon::arrow_tokenizer_lookup(f.db, "[gMASK]", 7));
    REQUIRE(ids[1] == lsxcommon::arrow_tokenizer_lookup(f.db, "<sop>", 5));
    REQUIRE(ids[2] == lsxcommon::arrow_tokenizer_lookup(f.db, "<|user|>", 8));

    f.db = lsxcommon::ArrowTokenizerDb{};
    fs::remove_all(f.pinned);
}
```

- [ ] **Step 3: Register the test in `tests/CMakeLists.txt`**

Append after the `test_generation_cli` block (after line 367):

```cmake
# ── GLM-5.2 prompt framing helper test (host-only, synthetic tokenizer) ────
add_executable(test_glm_prompt_framing test_glm_prompt_framing.cpp)
target_compile_definitions(test_glm_prompt_framing PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_glm_prompt_framing PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_glm_prompt_framing POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_glm_prompt_framing>"
    )
    add_custom_command(TARGET test_glm_prompt_framing POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_glm_prompt_framing>"
    )
endif()

add_test(NAME test_glm_prompt_framing COMMAND test_glm_prompt_framing)
```

- [ ] **Step 4: Reconfigure and build the test to verify it FAILS**

```bash
cmake --preset linux-debug
cmake --build --preset linux-debug --target test_glm_prompt_framing
```

Expected: COMPILE FAILURE — `test_glm_prompt_framing.cpp` includes `lsxcommon/prompt_framing.h`, which does not exist yet (`fatal error: lsxcommon/prompt_framing.h: No such file or directory`). This is the RED.

- [ ] **Step 5: Commit the red test**

```bash
git add tests/data/synthetic_glm_tok.json tests/test_glm_prompt_framing.cpp tests/CMakeLists.txt
git commit -m "test: add host-only GLM prompt framing tests (RED)"
```

---
---

### Task 2: Implement the framing helper (GREEN for Task 1)

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/prompt_framing.h`
- Create: `libs/lsxcommon/src/prompt_framing.cpp`
- Modify: `libs/lsxcommon/CMakeLists.txt`

**Interfaces:**
- Consumes: `ArrowTokenizerDb`, `arrow_tokenizer_lookup`, `arrow_tokenizer_encode` from `lsxcommon/tokenizer.h`.
- Produces: `bool lsxcommon::prompt::BuildPromptInputIds(std::string_view model_type, const ArrowTokenizerDb &tok, std::string_view raw_prompt, std::vector<int32_t> &out_ids)` — the exact signature used by `tests/test_glm_prompt_framing.cpp`, `tests/test_glm_prompt_output.cpp` (Task 3) and `apps/lsxforge/src/main.cpp` (Task 5).

- [ ] **Step 1: Write the header `libs/lsxcommon/include/lsxcommon/prompt_framing.h`**

```cpp
#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "lsxcommon/tokenizer.h"

namespace lsxcommon {
namespace prompt {

// Frames a raw generation prompt into model-specific input token ids.
// For model_type == "glm-5.2" the GLM chat-template framing for the
// non-thinking generation mode is applied:
//   [gMASK]<sop><|user|>{prompt}<|assistant|><think></think>
// (falls back to the raw BPE encode when any of those special tokens is
// missing from the tokenizer vocab). Any other model type is encoded raw.
// Returns true and fills out_ids on success; false when the tokenizer
// rejects the prompt (or it is empty).
bool BuildPromptInputIds(std::string_view model_type,
                         const ArrowTokenizerDb &tok,
                         std::string_view raw_prompt,
                         std::vector<int32_t> &out_ids);

} // namespace prompt
} // namespace lsxcommon
```

- [ ] **Step 2: Write the implementation `libs/lsxcommon/src/prompt_framing.cpp`**

```cpp
#include "lsxcommon/prompt_framing.h"

namespace lsxcommon {
namespace prompt {
namespace {

constexpr std::string_view kGlmType = "glm-5.2";

struct GlmSpecials {
    int32_t gmask = -1, sop = -1, user = -1, asst = -1, think = -1, endthink = -1;
    bool complete() const
    {
        return gmask >= 0 && sop >= 0 && user >= 0 && asst >= 0 &&
               think >= 0 && endthink >= 0;
    }
};

GlmSpecials LookupGlmSpecials(const ArrowTokenizerDb &tok)
{
    GlmSpecials s;
    s.gmask    = arrow_tokenizer_lookup(tok, "[gMASK]", 7);
    s.sop      = arrow_tokenizer_lookup(tok, "<sop>", 5);
    s.user     = arrow_tokenizer_lookup(tok, "<|user|>", 8);
    s.asst     = arrow_tokenizer_lookup(tok, "<|assistant|>", 13);
    s.think    = arrow_tokenizer_lookup(tok, "<think>", 7);
    s.endthink = arrow_tokenizer_lookup(tok, "</think>", 8);
    return s;
}

} // namespace

bool BuildPromptInputIds(std::string_view model_type,
                         const ArrowTokenizerDb &tok,
                         std::string_view raw_prompt,
                         std::vector<int32_t> &out_ids)
{
    const bool glm = (model_type == kGlmType);
    const GlmSpecials specials = glm ? LookupGlmSpecials(tok) : GlmSpecials{};
    const bool frame = glm && specials.complete();

    // Capacity-grow loop: 256 -> 8192 ids. GLM framing reserves 6 leading
    // ids for the chat-template sentinels; encode may return -2 (capacity too
    // small), in which case the buffer is doubled and the prompt re-encoded.
    for (size_t cap = 256; cap <= 8192; cap *= 2) {
        std::vector<int32_t> ids(cap, 0);
        const int32_t head = frame ? 3 : 0;
        size_t enc_cap = cap - (frame ? 6u : 0u);
        const int32_t n = arrow_tokenizer_encode(
            tok, raw_prompt.data(), raw_prompt.size(), ids.data() + head,
            &enc_cap);
        if (n == -2) continue;
        if (n <= 0) return false;   // empty or rejected prompt
        if (frame) {
            ids[0] = specials.gmask;
            ids[1] = specials.sop;
            ids[2] = specials.user;
            ids[3 + static_cast<size_t>(n)] = specials.asst;
            ids[4 + static_cast<size_t>(n)] = specials.think;
            ids[5 + static_cast<size_t>(n)] = specials.endthink;
        }
        ids.resize(static_cast<size_t>(n) + (frame ? 6u : 0u));
        out_ids = std::move(ids);
        return true;
    }
    return false;
}

} // namespace prompt
} // namespace lsxcommon
```

- [ ] **Step 3: Register the new source in `libs/lsxcommon/CMakeLists.txt`**

In the `add_library(lsxcommon STATIC ...)` block (lines 16-42), add `src/prompt_framing.cpp` after `src/path_registry.cpp` (line 30):

```cmake
    src/path_registry.cpp
    src/prompt_framing.cpp
    src/models/lsx_orchestrator.cpp
```

- [ ] **Step 4: Reconfigure, build, and run the test to verify it PASSES**

```bash
cmake --preset linux-debug
cmake --build --preset linux-debug --target test_glm_prompt_framing
ctest --preset linux-debug -R "^test_glm_prompt_framing$" --output-on-failure
```

Expected: all 5 test cases pass (`All tests passed`, exit code 0).

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/prompt_framing.h \
        libs/lsxcommon/src/prompt_framing.cpp \
        libs/lsxcommon/CMakeLists.txt
git commit -m "feat: add GLM-5.2 prompt framing helper (GREEN)"
```

---
---

### Task 3: Real-model `output_text` test (RED)

**Files:**
- Create: `tests/test_glm_prompt_output.cpp`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `lsxcommon::LsxModelFactory`, `lsxcommon::ModelInitConfig`, `lsxcommon::ModelRequest`, `lsxcommon::IModel` (`lsxcommon/model.h`, `lsxcommon/model_factory.h`), `lsxcommon::lsx_model_models_dir()` (`lsxcommon/path_registry.h`), `BuildPromptInputIds` (Task 2).
- Produces: the acceptance gate — `GlmV5Model::Infer` must fill `output_text` (implemented in Task 4).

- [ ] **Step 1: Write the failing test `tests/test_glm_prompt_output.cpp`**

```cpp
// TDD — GLM-5.2 meaningful prompt output (Model Ingestion Protocol step 8).
// Real-model test: SKIPs when no GLM-5.2 checkpoint is present (CTest
// SKIP_RETURN_CODE 4, same as the other GLM real-weight tests).
//
// Contract under test (GlmV5Model::Infer):
//   - the GLM chat-template framed "The capital of France is" must generate a
//     non-empty tokenizer-decoded output_text,
//   - output_text must contain "Paris" (bit-exact determinism of the greedy
//     decode is covered by test_glm_selfcheck),
//   - r.ok == true && !r.gen_tokens.empty().
#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include <lsxcommon/model.h>
#include <lsxcommon/model_factory.h>
#include <lsxcommon/path_registry.h>
#include <lsxcommon/prompt_framing.h>

namespace {

std::filesystem::path GlmModelPath()
{
    const char *env = std::getenv("LSX_TEST_GLM_MODEL_PATH");
    return env ? std::filesystem::path(env)
               : std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
                     "GLM-5.2";
}

} // namespace

TEST_CASE("GLM-5.2 Infer: meaningful output_text on the test prompt",
          "[glm][model][prompt][real-model]")
{
    const std::filesystem::path model = GlmModelPath();
    if (!std::filesystem::exists(model / "logestix" / "pinned"))
        SKIP("GLM real model directory not present");

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model.string();
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Type() == "glm-5.2");
    REQUIRE(m->Load());

    const char *prompt = "The capital of France is";
    std::vector<int32_t> ids;
    REQUIRE(lsxcommon::prompt::BuildPromptInputIds(
        m->Type(), m->Tokenizer(), prompt, ids));
    REQUIRE(ids.size() > 6);

    lsxcommon::ModelRequest req;
    req.input_ids = ids;
    req.max_gen_tokens = 8;

    auto r = m->Infer(req);
    REQUIRE(r.ok);
    REQUIRE(!r.gen_tokens.empty());
    REQUIRE(!r.output_text.empty());
    REQUIRE(r.output_text.find("Paris") != std::string::npos);

    m->Unload();
}
```

- [ ] **Step 2: Register the test in `tests/CMakeLists.txt`**

Append after the `test_glm_selfcheck` block (after line 329):

```cmake
# ── GLM-5.2 meaningful-output test via --run (Infer output_text) ───────────
add_executable(test_glm_prompt_output test_glm_prompt_output.cpp)
target_compile_definitions(test_glm_prompt_output PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_glm_prompt_output PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_glm_prompt_output POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_glm_prompt_output>"
    )
    add_custom_command(TARGET test_glm_prompt_output POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_glm_prompt_output>"
    )
endif()

add_test(NAME test_glm_prompt_output COMMAND test_glm_prompt_output)
set_tests_properties(test_glm_prompt_output PROPERTIES SKIP_RETURN_CODE 4)
```

- [ ] **Step 3: Reconfigure, build, and run to verify it FAILS**

```bash
cmake --preset linux-debug
cmake --build --preset linux-debug --target test_glm_prompt_output
ctest --preset linux-debug -R "^test_glm_prompt_output$" --output-on-failure
```

Expected: FAILURE — `REQUIRE(!r.output_text.empty())` fails because
`GlmV5Model::Infer` currently leaves `output_text` empty (red). If the GLM
checkpoint is absent the test SKIPs (exit 4, reported as skipped — in that
case proceed; the GREEN proof is the host-only test plus code review).

- [ ] **Step 4: Commit the red test**

```bash
git add tests/test_glm_prompt_output.cpp tests/CMakeLists.txt
git commit -m "test: add GLM-5.2 meaningful output_text test (RED)"
```

---
---

### Task 4: Decode `output_text` in `GlmV5Model::Infer` (GREEN for Task 3)

**Files:**
- Modify: `libs/lsxcommon/src/glm_v5_model.cpp:145-155`

**Interfaces:**
- Consumes: `arrow_tokenizer_decode` (`lsxcommon/tokenizer.h`, already included), `*tok_db_` member.
- Produces: `gpu::InferenceResult.output_text` filled for GLM-5.2 (non-empty when `gen_tokens` is non-empty and decodes successfully).

- [ ] **Step 1: Edit the result-filling block in `GlmV5Model::Infer`**

Replace the current block (lines 145-155):

```cpp
    r.ok = true;
    r.num_tokens = static_cast<int>(gen.tokens.size());
    r.num_layers = glm::kGlmNumLayers;
    // output_text stays empty: full tokenizer decode is out of scope for the
    // raw hidden/logit pass; only the decoded id stream is published.
    if (!gen.tokens.empty()) {
        r.first_gen_tok_id = gen.tokens[0];
        r.first_gen_maxv = gen.first_maxv;
    }
    r.gen_tokens = std::move(gen.tokens);
    r.final_logits = std::move(gen.final_logits);
    cleanup();
    return r;
```

with:

```cpp
    r.ok = true;
    r.num_tokens = static_cast<int>(gen.tokens.size());
    r.num_layers = glm::kGlmNumLayers;
    if (!gen.tokens.empty()) {
        r.first_gen_tok_id = gen.tokens[0];
        r.first_gen_maxv = gen.first_maxv;
    }
    r.gen_tokens = std::move(gen.tokens);
    r.final_logits = std::move(gen.final_logits);

    // Decode the generated token stream into output_text (gen.tokens carries
    // only generated tokens, never the prompt). Capacity-retry mirrors the
    // DeepSeek pipeline; a decode failure leaves output_text empty but keeps
    // r.ok true — generation succeeded, decode is best-effort.
    if (!r.gen_tokens.empty()) {
        std::string out(4096, '\0');
        size_t cap = 4096;
        int dr = ::lsxcommon::arrow_tokenizer_decode(
            *tok_db_, r.gen_tokens.data(), r.gen_tokens.size(),
            out.data(), &cap);
        if (dr == -2) {   // *cap now reports the required size; retry once
            out.resize(cap);
            size_t cap2 = cap;
            dr = ::lsxcommon::arrow_tokenizer_decode(
                *tok_db_, r.gen_tokens.data(), r.gen_tokens.size(),
                out.data(), &cap2);
            cap = cap2;
        }
        if (dr == 0) {
            out.resize(cap);
            r.output_text = std::move(out);
        } else {
            spdlog::warn("[glm-model] tokenizer decode failed: {}", dr);
        }
    }

    cleanup();
    return r;
```

- [ ] **Step 2: Rebuild and run the real-model test to verify it PASSES**

```bash
cmake --build --preset linux-debug --target test_glm_prompt_output
ctest --preset linux-debug -R "^test_glm_prompt_output$" --output-on-failure
```

Expected: PASS — `output_text` non-empty and contains `"Paris"` (the GLM
checkpoint is present at `/mnt/llm_models/models/GLM-5.2`).

- [ ] **Step 3: Run the GLM determinism stress test to confirm no regression**

```bash
ctest --preset linux-debug -R "^test_glm_selfcheck$" --output-on-failure
```

Expected: PASS (50 cycles, bit-exact `final_logits`). This guards that the
added host-side decode did not perturb the device generation path.

- [ ] **Step 4: Commit**

```bash
git add libs/lsxcommon/src/glm_v5_model.cpp
git commit -m "feat: decode GLM-5.2 output_text in Infer (GREEN)"
```

---
---

### Task 5: Wire `lsxforge --run` through the framing helper

**Files:**
- Modify: `apps/lsxforge/src/main.cpp:250-261` (plus the include block)

**Interfaces:**
- Consumes: `BuildPromptInputIds` (Task 2), `model->Type()` / `model->Tokenizer()` (`lsxcommon::IModel`).
- Produces: `lsxforge --run` end-to-end meaningful GLM response — the CLI goal.

- [ ] **Step 1: Add the include**

In `apps/lsxforge/src/main.cpp`, after `#include "lsxcommon/model_factory.h"` (line 5), add:

```cpp
#include "lsxcommon/prompt_framing.h"
```

- [ ] **Step 2: Replace the raw encode block**

Replace the block (currently lines 250-261):

```cpp
        std::vector<int32_t> input_ids(512, 0);
        size_t cap = input_ids.size();
        int32_t n_ids = lsxcommon::arrow_tokenizer_encode(
            model->Tokenizer(), run_prompt.data(), run_prompt.size(),
            input_ids.data(), &cap);
        if (n_ids < 0) {
            spdlog::error("Tokenizer encode failed: {}", n_ids);
            model->Unload();
            lsxcommon::Core::Shutdown();
            return 1;
        }
        input_ids.resize(static_cast<size_t>(n_ids));
```

with:

```cpp
        std::vector<int32_t> input_ids;
        if (!lsxcommon::prompt::BuildPromptInputIds(
                model->Type(), model->Tokenizer(), run_prompt, input_ids)) {
            spdlog::error("Tokenizer encode failed for prompt: {}",
                          run_prompt);
            model->Unload();
            lsxcommon::Core::Shutdown();
            return 1;
        }
```

- [ ] **Step 3: Reconfigure, build, and smoke-test**

```bash
cmake --preset linux-debug
cmake --build --preset linux-debug --target lsxforge
```

Expected: `lsxforge` builds clean.

Smoke test with the real GLM model (inference is resource-critical — run alone):

```bash
build/linux-debug/bin/lsxforge --run "The capital of France is" \
  --input "$HOME/models/GLM-5.2" 2>/dev/null | rg "output"
```

(If the `--input` path differs on your box, use the absolute path from
`lsxcommon::lsx_model_models_dir()`; on this dev box it is
`/mnt/llm_models/models/GLM-5.2`.) Expected: the `output` line contains
`Paris` and is non-empty — the meaningful response.

- [ ] **Step 4: Run the full targeted test set**

```bash
ctest --preset linux-debug -R "test_glm_prompt_framing|test_glm_prompt_output|test_generation_cli" --output-on-failure
```

Expected: all pass (host-only framing, real-model output, and the unchanged
generation-CLI normalization).

- [ ] **Step 5: Commit**

```bash
git add apps/lsxforge/src/main.cpp
git commit -m "feat: frame GLM-5.2 prompts in lsxforge --run path"
```

---
---

## Self-Review

**Spec coverage:**
- Component A (framing helper) → Task 2; its tests → Task 1.
- Component B (output_text decode) → Task 4; its test → Task 3.
- Component C (lsxforge wiring) → Task 5.
- `tests/data/synthetic_glm_tok.json` fixture → Task 1 Step 1.
- SKIP_RETURN_CODE 4 + TOKENIZER_TEST_DATA_DIR registry → Tasks 1 & 3.
- Acceptance criteria (non-empty output_text; "Paris"; SKIP without model) → Tasks 3 & 4.

**Placeholder scan:** No TBD/TODO; every step carries exact code/commands.

**Type consistency:** `BuildPromptInputIds(std::string_view model_type, const ArrowTokenizerDb &tok, std::string_view raw_prompt, std::vector<int32_t> &out_ids)` defined in Task 2 and used identically in Tasks 1, 3, 5. `ModelRequest{input_ids, max_gen_tokens}` matches `lsxcommon/model.h:20-23`. `arrow_tokenizer_decode(*tok_db_, ...)` matches `tokenizer.h:58-60`.

## Final Verification (run after all tasks)

```bash
cmake --build --preset linux-debug
ctest --preset linux-debug -R "test_glm_prompt_framing|test_glm_prompt_output|test_generation_cli|test_glm_selfcheck" --output-on-failure
```
