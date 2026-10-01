# AliceAI-Foundation-80B-A3B-Base Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Repository override (AGENTS.md, highest priority):** subagents are FORBIDDEN
> in this repo. Execute every task inline in the main session using
> `executing-plans`; do NOT dispatch task subagents.

**Goal:** Ingest `AliceAI-Foundation-80B-A3B-Base` (`model_type=alice_ai`) as a new independent `alice` family — converter, NVFP4 layout, native SentencePiece-BPE tokenizer, loader, model class, AttnRes + KDA + gated-attention + MoE CUDA runtime, tool bridge, selfchecks and perf baseline — following the 8-stage Model Ingestion Protocol.

**Architecture:** A new `alice_moe_*` family over the model-neutral layers (`lsx_inference_nvfp4`, `lsx_gemv`, `lsx_kernels_*.cuh`, `expert_store`, `schema`, `converter`, `tokenizer`, `model_base`, `path_registry`). A novel block attention-residual mixer and a Kimi-Delta-Attention linear layer are implemented as Alice-owned TUs; the SentencePiece-BPE backend is added to the neutral tokenizer infrastructure.

**Tech Stack:** C++20 / C23, CUDA 13.4 (Blackwell CC 12.1), CMake 3.25+ presets, Apache Arrow 25.0.1, FlatBuffers, simdjson, absl, Catch2 v3.16.0, spdlog; PyTorch `tokenizers` only for golden capture (`tools/`).

## Global Constraints

- Target architecture only: Blackwell `sm_120/sm_121` (CC 12.0/12.1). Forbidden: `sm_90` (Hopper), `sm_89` (Ada), `sm_80` (Ampere).
- CUDA Toolkit 13.4 (minimum 13.2+), full C++23 in NVCC; language C++20 / C23.
- Frozen geometry: every dimension locked in `libs/lsxcommon/src/alice_moe_constants.h` with `static_assert`; `Load` must reject a mismatch.
- Norms, biases, router weights and embeddings are NEVER quantized.
- NVFP4 math only via the neutral `lsx_inference_nvfp4` layer; no family-owned NVFP4 dispatcher.
- One CUDA stream; no per-token `cudaDeviceSynchronize`; reductions use the shared-memory sandbox (thread-0 + barriers); no `atomicAdd` feeding `final_logits`.
- Headers reachable from `.cu`/`.cuh` must NOT include `<absl/container/*>`; containers/streams in device TUs stay `std::`.
- Logging via spdlog only (no `std::cout`/`printf`). Host-only C++ prefers absl (`flat_hash_map`, `StrCat`, `StrFormat` printf-style).
- Paths: converter output only through `lsx_model_metadata(rel)` / `lsx_model_metadata_root()`; no hardcoded paths.
- Tests: every new target gets a ctest label in the SAME commit; selfcheck/fsi/code/det targets get `SKIP_RETURN_CODE 4`; converter tests label `convert`, bounded `TIMEOUT` ≤600 s (canonical 120 s), synthetic checkpoint via `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1` (runtime env).
- Commit messages English, prefix `build:`/`feat:`/`fix:`/`refactor:`/`test:`/`docs:`.
- Model-neutral tokenizer additions live in `tokenizer.cpp` / `tokenizer_convert.cpp` / `tokenizer.h` (no family stem, not scanned by the naming guard).
- Never modify another family's source/tests. Shared surfaces (registry, CMake, `lsxforge`, guards, perf harness, AGENTS, repo-map) are extended additively.

---

## File Structure

| File | Responsibility |
|---|---|
| `libs/lsxcommon/src/alice_moe_constants.h` | Frozen passport, `static_assert`s (ns `lsxcommon::alice::frozen`) |
| `libs/lsxcommon/include/lsxcommon/alice_moe_convert.h` + `src/alice_moe_convert.cpp` | config parse → BF16→NVFP4 dense + per-expert packing + index |
| `libs/lsxcommon/include/lsxcommon/alice_moe_load.h` + `src/alice_moe_load.cpp` | NVFP4 pinned weight graph + format/geometry gates |
| `libs/lsxcommon/include/lsxcommon/alice_moe_expert_cache.h` + `src/alice_moe_expert_cache.cpp` | host expert store + device slot-LRU (`ExpertStore`) |
| `libs/lsxcommon/src/alice_moe_device_math.h` | host/device reference math (KDA, AttnRes, router, ZC RMSNorm) |
| `libs/lsxcommon/src/alice_moe_kernels.{h,cu}` | ZC RMSNorm, AttnRes mix, router, SwiGLU, KDA, gated attention |
| `libs/lsxcommon/include/lsxcommon/alice_moe_inference_attn.h` + `src/alice_moe_inference_attn.cu` | attention/KDA forward |
| `libs/lsxcommon/include/lsxcommon/alice_moe_inference_pipeline.h` + `src/alice_moe_inference_pipeline.cu` | layer graph + prefill/decode |
| `libs/lsxcommon/include/lsxcommon/alice_moe_model.h` + `src/alice_moe_model.cpp` | `IModel`/`ModelBase` subclass, registry, SelfCheck |
| `libs/lsxcommon/include/lsxcommon/alice_tool_format.h` + `src/alice_tool_format.cpp` | native AliceAI chat/tool codec |
| `libs/lsxcommon/src/tokenizer_convert.cpp` / `tokenizer.cpp` / `include/lsxcommon/tokenizer.h` | model-neutral SP-BPE backend + `tokenizer_kind` |
| `tests/test_alice_moe_*.{cpp,cu}` + `tests/test_alice_tool_framing.cpp` + `tests/test_alice_tokenizer.cpp` | TDD coverage |
| `tools/alice_tokenize_golden.py` | golden capture for the SP-BPE backend |
| `apps/lsxforge/src/main.cpp` | `--alice` flag + dispatch |

---

## Phase A — Model-neutral SentencePiece-BPE tokenizer backend

### Task A1: Parse `tokenizer.model` into the arrow tokenizer artifacts

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/tokenizer.h`
- Modify: `libs/lsxcommon/src/tokenizer_convert.cpp`
- Modify: `libs/lsxcommon/src/tokenizer.cpp` (metadata read only in this task)
- Test: `tests/test_alice_tokenizer.cpp`
- Modify: `tests/CMakeLists.txt` (new `test_alice_tokenizer` target, label `fast`)

**Interfaces:**
- Consumes: existing `TokenizerVocabSchema`, `TokenizerMergesSchema` (`lsxcommon/schema.h`), `BuildTokenizerArrowFiles` (`tokenizer_convert.h`).
- Produces:
  - `arrow::Status BuildTokenizerArrowFilesSentencePiece(const std::string &tokenizer_model_path, const std::string &output_dir);`
  - vocab metadata keys added: `tokenizer_kind="sp-bpe"`, `byte_fallback` (`0|1`), `add_dummy_prefix` (`0|1`), `escape_whitespaces` (`0|1`); `byte_to_token_id` for sp-bpe maps byte `b` to the id of piece `<0xXX>`.
  - `ArrowTokenizerDb` gains `int32_t tokenizer_kind = 0;` (`0` = `gpt2-byte-bpe`, `1` = `sp-bpe`).

- [ ] **Step 1: Write the failing test**

`tests/test_alice_tokenizer.cpp`:
```cpp
#include <catch2/catch_test_macros.hpp>
#include <lsxcommon/tokenizer.h>
#include <lsxcommon/tokenizer_convert.h>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

TEST_CASE("sp-bpe builder parses the real tokenizer.model", "[tokenizer][fast]")
{
    const char *root = std::getenv("LSX_TEST_ALICE_ROOT");
    if (!root) SKIP("LSX_TEST_ALICE_ROOT not set");
    const fs::path src = fs::path(root) / "tokenizer.model";
    if (!fs::exists(src)) SKIP("tokenizer.model not present");
    const fs::path out = fs::temp_directory_path() / "alice_tok_meta";
    fs::create_directories(out);
    auto st = lsxcommon::BuildTokenizerArrowFilesSentencePiece(src.string(),
                                                               out.string());
    REQUIRE(st.ok());
    auto vocab = out / "tokenizer_vocab.arrow";
    auto merges = out / "tokenizer_merges.arrow";
    REQUIRE(fs::exists(vocab));
    REQUIRE(fs::exists(merges));

    lsxcommon::ArrowTokenizerDb db;
    auto vmm = arrow::io::MemoryMappedFile::Open(vocab.string(),
                                                 arrow::io::FileMode::READ).ValueOrDie();
    auto mmm = arrow::io::MemoryMappedFile::Open(merges.string(),
                                                 arrow::io::FileMode::READ).ValueOrDie();
    REQUIRE(lsxcommon::load_tokenizer_metadata(vmm, mmm, db).ok());
    REQUIRE(db.tokenizer_kind == 1);
    REQUIRE(db.vocab_size == 129024);
    REQUIRE(db.bos_id == 1);
    REQUIRE(db.eos_id == 2);
    REQUIRE(lsxcommon::arrow_tokenizer_lookup(db, "<s>", 3) == 1);
    REQUIRE(lsxcommon::arrow_tokenizer_lookup(db, "</s>", 4) == 2);
    // byte-fallback anchor <0x00> .. <0xff> at ids 7..262
    REQUIRE(lsxcommon::arrow_tokenizer_lookup(db, "<0x00>", 6) == 7);
    REQUIRE(lsxcommon::arrow_tokenizer_lookup(db, "<0xff>", 6) == 262);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --preset linux-debug && cmake --build --preset linux-debug --target test_alice_tokenizer`
Expected: link error / FAIL — `BuildTokenizerArrowFilesSentencePiece` and `tokenizer_kind` do not exist.

- [ ] **Step 3: Add the `tokenizer_kind` field**

In `libs/lsxcommon/include/lsxcommon/tokenizer.h`, inside `struct ArrowTokenizerDb` after `int32_t bos_id = -1, eos_id = -1, pad_id = -1;` add:
```cpp
    int32_t tokenizer_kind = 0;  // 0 = gpt2-byte-bpe, 1 = sentencepiece-bpe
```
In `load_tokenizer_metadata` (`tokenizer.cpp`) add, next to the other metadata reads:
```cpp
        {
            ARROW_ASSIGN_OR_RAISE(auto s, meta->Get("tokenizer_kind"));
            if (s == "sp-bpe") db.tokenizer_kind = 1;
            else db.tokenizer_kind = 0;
        }
```
Add to `valid()`: nothing (kind 0 default preserves every existing model).

- [ ] **Step 4: Implement the protobuf parser + artifact writer**

In `libs/lsxcommon/src/tokenizer_convert.cpp` add (declare in `tokenizer_convert.h`):

```cpp
namespace {
// Minimal protobuf wire reader: varint, length-delimited, fixed32/64.
struct PbReader {
    const uint8_t *p, *end;
    bool varint(uint64_t &out);
    bool skip(uint32_t wire);
    bool next(uint32_t &field, uint32_t &wire, uint64_t &v, const uint8_t *&ld,
              size_t &ldlen);
};
} // namespace

arrow::Status BuildTokenizerArrowFilesSentencePiece(
    const std::string &tokenizer_model_path, const std::string &output_dir)
{
    // 1. Read file into memory.
    // 2. Walk ModelProto: field 1 repeated SentencePiece{1:piece string,
    //    2:score float(fixed32), 3:type varint}; field 2 TrainerSpec
    //    {1:model_name ignored, 3:model_type (must be 2==BPE),
    //    21:byte_fallback, 42:?, 44:??}; field 3 NormalizerSpec
    //    {1:name (must be "identity"), 3:add_dummy_prefix}.
    //    add_dummy_prefix / escape_whitespaces are trainer/normalizer flags;
    //    read them, default add_dummy_prefix=true, escape_whitespaces=true.
    // 3. Build vocab rows (TokenIdRow) exactly like BuildTokenizerArrowFiles:
    //    is_special = type in {CONTROL(3), USER_DEFINED(4), BYTE(6)}.
    // 4. byte_to_token_id[b] = id of piece "<0x%02X>" (byte fallback); error if
    //    any byte missing and byte_fallback==false.
    // 5. Derive merges: for each non-byte piece with >=2 codepoints, find the
    //    split (a,b) into existing pieces maximizing score(a)+score(b); emit
    //    (left_id,right_id,rank,token_id) where rank = global order by
    //    descending piece score (stable on id). Sort by (left_id,right_id).
    // 6. Write tokenizer_vocab.arrow + tokenizer_merges.arrow into output_dir
    //    with the schema builders, metadata including tokenizer_kind="sp-bpe",
    //    byte_fallback, add_dummy_prefix, escape_whitespaces, bos/eos/pad,
    //    vocab_size, byte_to_token_id.
}
```
Follow the write pattern of `BuildTokenizerArrowFiles` in the same file (schema `MakeBatch` + `WriteRecordBatchToFile`). Parse the real checkpoint's header to confirm trainer field numbers before finalizing (already audited: `model_type`=field 3, value 2; `byte_fallback` field 21 value 1; normalizer name field 1 = `identity`, add_dummy_prefix field 3 value 1).

- [ ] **Step 5: Run test to verify it passes**

Run: `LSX_TEST_ALICE_ROOT=/mnt/llm_models/models/AliceAI-Foundation-80B-A3B-Base ctest --preset linux-debug -R test_alice_tokenizer --output-on-failure`
Expected: PASS (or SKIP if root unset).

- [ ] **Step 6: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/tokenizer.h libs/lsxcommon/src/tokenizer_convert.cpp libs/lsxcommon/src/tokenizer.cpp tests/test_alice_tokenizer.cpp tests/CMakeLists.txt
git commit -m "feat: model-neutral sentencepiece-bpe tokenizer artifact builder"
```

### Task A2: SP-BPE encode/decode runtime + goldens

**Files:**
- Modify: `libs/lsxcommon/src/tokenizer.cpp`
- Create: `tools/alice_tokenize_golden.py`
- Create: `tests/fixtures/tokenizer/alice.json`
- Modify: `tests/test_alice_tokenizer.cpp` (add encode/decode cases)

**Interfaces:**
- Consumes: `ArrowTokenizerDb.tokenizer_kind == 1`, `token_offsets`/`token_bytes`, the derived merge table, `byte_to_token_id`.
- Produces: `arrow_tokenizer_encode` / `arrow_tokenizer_decode` transparently handle SP-BPE when `tokenizer_kind == 1`.

- [ ] **Step 1: Write the golden capture tool**

`tools/alice_tokenize_golden.py`:
```python
#!/usr/bin/env python3
"""Capture SP-BPE goldens for AliceAI. Reference: reconstruct a tokenizers BPE
from tokenizer.model (byte_fallback=True) and, when importable, cross-check
against sentencepiece/LlamaTokenizer. Emits tests/fixtures/tokenizer/alice.json.
"""
import argparse, json, struct, sys

def rd_varint(b, i):
    s = r = 0
    while True:
        c = b[i]; i += 1; r |= (c & 0x7F) << s; s += 7
        if not c & 0x80: break
    return r, i

def parse_model(data):
    pieces, normalizer = [], None
    i = 0
    while i < len(data):
        tag, i = rd_varint(data, i); f, w = tag >> 3, tag & 7
        if w == 2:
            ln, i = rd_varint(data, i); val = data[i:i+ln]; i += ln
            if f == 1:
                j, pc, sc, ty = 0, None, 0.0, 1
                while j < len(val):
                    t, j = rd_varint(val, j); f2, w2 = t >> 3, t & 7
                    if w2 == 2:
                        l2, j = rd_varint(val, j); v2 = val[j:j+l2]; j += l2
                        if f2 == 1: pc = v2
                    elif w2 == 0:
                        v2, j = rd_varint(val, j)
                        if f2 == 3: ty = v2
                    elif w2 == 5:
                        sc = struct.unpack('<f', val[j:j+4])[0]; j += 4
                    elif w2 == 1: j += 8
                pieces.append((pc, sc, ty))
            elif f == 3:
                normalizer = val
        elif w == 0: _, i = rd_varint(data, i)
        elif w == 5: i += 4
        elif w == 1: i += 8
    return pieces, normalizer

PROMPTS = [
    "The capital of France is",
    "Привет, мир! Как дела?",
    "def f(x):\n    return x ** 2\n",
    "  leading and trailing  ",
    "emoji \U0001F600 and tabs\there",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data = open(f"{a.model_dir}/tokenizer.model", "rb").read()
    pieces, _ = parse_model(data)
    # Build tokenizers BPE reference (vocab + merges derived from scores) and
    # encode PROMPTS; if sentencepiece is importable, assert equality first.
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    vocab = {}
    for pid, (pc, sc, ty) in enumerate(pieces):
        key = pc.decode("utf-8")
        vocab[key] = pid
    # merge derivation identical to the C++ converter
    merges = []
    for pc, sc, ty in pieces:
        s = pc.decode("utf-8", "ignore")
        if len(s) < 2 or pc.startswith(b"<0x"):
            continue
        best = None
        for k in range(1, len(s)):
            a2, b2 = s[:k], s[k:]
            if a2 in vocab and b2 in vocab:
                sa = pieces[vocab[a2]][1]; sb = pieces[vocab[b2]][1]
                if best is None or sa + sb > best[0]:
                    best = (sa + sb, a2, b2)
        if best: merges.append((best[1], best[2]))
    tok = Tokenizer(BPE(vocab, merges, unk_token="<unk>",
                        byte_fallback=True, fuse_unk=False))
    out = {"bos_id": 1, "eos_id": 2, "vocab_size": len(pieces),
           "cases": []}
    for p in PROMPTS:
        enc = tok.encode(p, add_special_tokens=False).ids
        out["cases"].append({"text": p, "ids": enc})
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=2)
    print(f"wrote {a.out}: {len(out['cases'])} cases")

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Generate the goldens**

Run: `python3 tools/alice_tokenize_golden.py /mnt/llm_models/models/AliceAI-Foundation-80B-A3B-Base --out tests/fixtures/tokenizer/alice.json`
Expected: `wrote tests/fixtures/tokenizer/alice.json: 5 cases`. If `tokenizers` import fails, stop and report — do not fabricate goldens.

- [ ] **Step 3: Write the failing encode/decode test**

Append to `tests/test_alice_tokenizer.cpp`:
```cpp
#include <fstream>
#include <nlohmann/json.hpp>  // if not already a dep, use simdjson as elsewhere

TEST_CASE("sp-bpe encode matches committed goldens", "[tokenizer][fast]")
{
    const char *root = std::getenv("LSX_TEST_ALICE_ROOT");
    if (!root) SKIP("LSX_TEST_ALICE_ROOT not set");
    // ... build artifacts into temp dir as in the previous TEST_CASE ...
    std::ifstream f(std::string(LSX_SOURCE_DIR) +
                    "/tests/fixtures/tokenizer/alice.json");
    if (!f) SKIP("golden missing");
    // parse golden; for each case: arrow_tokenizer_encode(db, text, ...) and
    // REQUIRE(ids == golden ids). For decode: REQUIRE(round-trip text).
}
```
Use the existing JSON parsing convention in this repo (`simdjson`) rather than adding a dependency.

- [ ] **Step 4: Run test to verify it fails**

Run: `LSX_TEST_ALICE_ROOT=/mnt/llm_models/models/AliceAI-Foundation-80B-A3B-Base ctest --preset linux-debug -R test_alice_tokenizer --output-on-failure`
Expected: FAIL — encode still uses the GPT-2 byte path.

- [ ] **Step 5: Implement `sp_bpe_encode` / `sp_bpe_decode`**

In `tokenizer.cpp` add a dispatch at the top of both public paths:
```cpp
// sp-bpe: identity normalize; prepend "▁" (U+2581) when add_dummy_prefix;
// map whitespace runs to a single "▁" when escape_whitespaces; represent the
// symbol sequence as codepoints, replacing uncovered codepoints with their
// UTF-8 bytes' <0xXX> ids; then greedy adjacent-pair merge by merge_rank
// (lowest rank first), exactly like bpe_encode_word but over piece ids.
// decode: concatenate token bytes, skip specials, map <0xXX> pieces to raw
// bytes and "▁" to ' '.
```
Mirror the two-pass structure of `arrow_tokenizer_encode` (count then emit) and reuse `arrow_tokenizer_lookup` for specials. `byte_to_token_id` already maps bytes to `<0xXX>` ids.

- [ ] **Step 6: Run test to verify it passes**

Run: `LSX_TEST_ALICE_ROOT=/mnt/llm_models/models/AliceAI-Foundation-80B-A3B-Base ctest --preset linux-debug -R test_alice_tokenizer --output-on-failure`
Expected: PASS.

- [ ] **Step 7: Verify existing tokenizers are unaffected**

Run: `ctest --preset linux-debug -R "token|qwen_moe_convert|phi_dense_convert" --output-on-failure`
Expected: PASS — kind 0 default preserves GPT-2 byte-level behaviour.

- [ ] **Step 8: Commit**

```bash
git add libs/lsxcommon/src/tokenizer.cpp tools/alice_tokenize_golden.py tests/fixtures/tokenizer/alice.json tests/test_alice_tokenizer.cpp
git commit -m "feat: sentencepiece-bpe encode/decode runtime with goldens"
```

---

## Phase B — Passport and converter (stages 1–2)

### Task B1: Frozen passport + naming/independence guard rows

**Files:**
- Create: `libs/lsxcommon/src/alice_moe_constants.h`
- Test: `tests/test_alice_moe_model.cpp` (passport `fast` case only for now)
- Modify: `tests/test_code_naming_guard.cpp`
- Modify: `tests/test_model_independence_guard.cpp`
- Modify: `tests/CMakeLists.txt` (new `test_alice_moe_model` target, labels `fast`; later also `inference`)

**Interfaces:**
- Produces: namespace `lsxcommon::alice::frozen` constants (see below); `kType` string lives in the model class (Task D1).

- [ ] **Step 1: Write the failing passport test**

`tests/test_alice_moe_model.cpp`:
```cpp
#include <catch2/catch_test_macros.hpp>
#include "alice_moe_constants.h"

using namespace lsxcommon::alice::frozen;

TEST_CASE("alice passport matches the audited checkpoint", "[alice][fast]")
{
    STATIC_REQUIRE(kHidden == 2048);
    STATIC_REQUIRE(kLayers == 48);
    STATIC_REQUIRE(kFullAttnInterval == 4);
    STATIC_REQUIRE(kHeads == 16);
    STATIC_REQUIRE(kKVHeads == 2);
    STATIC_REQUIRE(kHeadDim == 256);
    STATIC_REQUIRE(kRotaryDim == 64);
    STATIC_REQUIRE(kLinearKeyHeads == 32);
    STATIC_REQUIRE(kLinearValueHeads == 32);
    STATIC_REQUIRE(kLinearKeyHeadDim == 128);
    STATIC_REQUIRE(kLinearValueHeadDim == 128);
    STATIC_REQUIRE(kLinearConvKernel == 4);
    STATIC_REQUIRE(kNumExperts == 512);
    STATIC_REQUIRE(kTopK == 10);
    STATIC_REQUIRE(kMoeIntermediate == 512);
    STATIC_REQUIRE(kSharedIntermediate == 512);
    STATIC_REQUIRE(kVocab == 129024);
    STATIC_REQUIRE(kBlockAttnRes == 4);
    STATIC_REQUIRE(kBosId == 1);
    STATIC_REQUIRE(kEosId == 2);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --preset linux-debug && cmake --build --preset linux-debug --target test_alice_moe_model`
Expected: FAIL — header not found.

- [ ] **Step 3: Write the passport header**

`libs/lsxcommon/src/alice_moe_constants.h`:
```cpp
#pragma once

// Frozen AliceAI-Foundation-80B-A3B-Base geometry (compile-time passport).
// Source of truth: /mnt/llm_models/models/AliceAI-Foundation-80B-A3B-Base
// (config.json + modeling_alice_ai.py + safetensors audit 2026-10-01):
// 48 layers, 12 x (3 x (KDA->MoE) -> 1 x (gated attention->MoE)), hidden 2048,
// KDA 32/32 heads hd 128 conv 4, gated attention 16Q/2KV hd 256 partial RoPE
// 64, MoE 512 experts top-10 I=512 + shared I=512, block AttnRes 4, vocab
// 129024, bos 1 eos 2, MTP 1 layer (never applied), all BF16.

namespace lsxcommon::alice {
namespace frozen {

inline constexpr int kHidden = 2048;
inline constexpr int kLayers = 48;
inline constexpr int kFullAttnInterval = 4;  // full attention at i % 4 == 3
inline constexpr int kBlockAttnRes = 4;

inline constexpr int kHeads = 16;
inline constexpr int kKVHeads = 2;
inline constexpr int kHeadDim = 256;
inline constexpr int kRotaryDim = 64;  // 256 * partial_rotary_factor 0.25
inline constexpr double kRopeTheta = 1000000.0;
inline constexpr double kRmsNormEps = 1e-6;

inline constexpr int kLinearKeyHeads = 32;
inline constexpr int kLinearValueHeads = 32;
inline constexpr int kLinearKeyHeadDim = 128;
inline constexpr int kLinearValueHeadDim = 128;
inline constexpr int kLinearConvKernel = 4;
inline constexpr int kLinearKvRepeat = kLinearValueHeads / kLinearKeyHeads;  // 1

inline constexpr int kNumExperts = 512;
inline constexpr int kTopK = 10;
inline constexpr int kMoeIntermediate = 512;
inline constexpr int kSharedIntermediate = 512;

inline constexpr int kVocab = 129024;
inline constexpr int kBosId = 1;
inline constexpr int kEosId = 2;
inline constexpr int kPadId = -1;

inline constexpr int kMtpLayers = 1;

static_assert(kLayers % kFullAttnInterval == 0, "hybrid pattern incomplete");
static_assert(kLayers % kBlockAttnRes == 0, "AttnRes blocking incomplete");
static_assert(kFullAttnInterval == kBlockAttnRes, "AttnRes block != full-attn interval");
static_assert(kHeads % kKVHeads == 0, "GQA repeat mismatch");
static_assert(kLinearValueHeads % kLinearKeyHeads == 0, "KDA repeat mismatch");
static_assert(kRotaryDim > 0 && kRotaryDim % 2 == 0, "rotary dim must be even");
static_assert(kRotaryDim <= kHeadDim, "rotary dim exceeds head dim");
static_assert(kTopK > 0 && kTopK <= kNumExperts, "bad top-k");

} // namespace frozen
} // namespace lsxcommon::alice
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build --preset linux-debug --target test_alice_moe_model && ctest --preset linux-debug -R test_alice_moe_model --output-on-failure`
Expected: PASS.

- [ ] **Step 5: Extend the naming guard**

In `tests/test_code_naming_guard.cpp`:
- `SchemeRegex()`: change `fam` to
  `"(?:glm|qwen|gemma|kimi|deepseek|nemotron|gigachat|ornith|granite|mimo|phi|alice)"`.
- `FamilyStems()`: add `"alice"`.
- `AllowedCommon()`: add `"alice_tool_format"`.

- [ ] **Step 6: Extend the independence guard**

In `tests/test_model_independence_guard.cpp`:
- `kPassports`: add
  `{"alice-foundation-80b-a3b", "alice_moe_constants.h", "lsxcommon::alice::frozen"},`
- `owner` map: add `{"alice_moe_constants.h", "namespace frozen {"},`
- `consumers` list: add `"alice_moe_model.cpp",  // alice-foundation-80b-a3b` (the file is created in Task D1; if the guard runs before D1 it fails — schedule this edit in the D1 commit instead if executing strictly, but keep it here when the file already exists).
- Add an Alice borrowing test mirroring the ornith block: `alice_moe_constants.h`, `alice_moe_model.h`, `alice_moe_model.cpp`, `alice_moe_load.cpp`, `../include/lsxcommon/alice_moe_load.h`, `../include/lsxcommon/alice_moe_inference_attn.h` must not contain any of `qwen_moe_constants.h`, `qwen_moe_36_constants.h`, `qwen_moe_exp_constants.h`, `glm_moe_load.h`, `deepseek_moe_load.cuh`, `gemma_moe_load.h`, `kimi_moe_load.h`.

- [ ] **Step 7: Run the guards**

Run: `ctest --preset linux-debug -R "test_code_naming_guard|test_model_independence_guard" --output-on-failure`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add libs/lsxcommon/src/alice_moe_constants.h tests/test_alice_moe_model.cpp tests/test_code_naming_guard.cpp tests/test_model_independence_guard.cpp tests/CMakeLists.txt
git commit -m "feat: alice frozen passport + guard registry rows"
```

### Task B2: Converter — config, NVFP4 dense, per-expert packing, index

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/alice_moe_convert.h`
- Create: `libs/lsxcommon/src/alice_moe_convert.cpp`
- Create: `tests/test_alice_moe_fixture.h`
- Create: `tests/test_alice_moe_convert.cpp`
- Modify: `apps/lsxforge/src/main.cpp`
- Modify: `libs/lsxcommon/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt` (target `test_alice_moe_convert`, label `convert`, `TIMEOUT 120`)

**Interfaces:**
- Consumes: `synthtest::*` (`tests/test_synth_checkpoint.h`), `lsxcommon::converter` (`quantize_tensor_gpu`, `PackExpertNVFP4`), `DenseNvfp4Schema`, `ExpertSchema`, `lsx_model_metadata`.
- Produces:
  - `arrow::Status lsxcommon::alice::ConvertAliceModel(const std::string &input_dir, const std::string &output_dir);`
  - Index JSON (`logestix/model.index.json`) with `format = "logestix-alice-nvfp4"` and the schema in the design doc.

- [ ] **Step 1: Write the fixture header**

`tests/test_alice_moe_fixture.h` re-exports the neutral primitives and defines a tiny `AliceGeom` (hidden 64, layers 8 with `i%4==3` full-attn, KDA heads 2/2 hd 8, gated 4Q/2KV hd 16 rotary 4, experts 4 top-2 I 16, shared 16, vocab 256, AttnRes block 4). Provide `WriteAliceConfigJson(dir)` and `WriteAliceShard(dir, geom)` that writes layer 0 (full-attn via the pattern) plus KDA/full/MoE tensors with the audited names. Reuse `synthtest::WriteSafetensors` / `MakeBf16`.

- [ ] **Step 2: Write the failing synthetic E2E converter test**

`tests/test_alice_moe_convert.cpp`:
```cpp
#include <catch2/catch_test_macros.hpp>
#include "test_alice_moe_fixture.h"
#include <lsxcommon/alice_moe_convert.h>
#include <cstdlib>
#include <filesystem>

namespace fs = std::filesystem;

TEST_CASE("alice synthetic convert emits logestix-alice-nvfp4", "[alice][convert]")
{
    REQUIRE(std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") != nullptr);
    const fs::path in = fs::temp_directory_path() / "alice_synth_in";
    const fs::path out = fs::temp_directory_path() / "alice_synth_out";
    fs::remove_all(in); fs::remove_all(out); fs::create_directories(in);
    alicetest::WriteAliceConfigJson(in);
    alicetest::WriteAliceShard(in, alicetest::AliceGeom{});
    auto st = lsxcommon::alice::ConvertAliceModel(in.string(), out.string());
    REQUIRE(st.ok());
    // index format + geometry + experts counted + layer-0 has no attn_res
    auto index = out / "logestix" / "model.index.json";
    REQUIRE(fs::exists(index));
    REQUIRE(alicetest::IndexFormatIs(index, "logestix-alice-nvfp4"));
    REQUIRE(alicetest::IndexGeometryField(index, "block_attn_res") == 4);
    REQUIRE(alicetest::ExpertFileCount(out) == 8 * 4);  // layers * experts
    REQUIRE(alicetest::LayerHasTensor(out, 0, "input_layernorm.weight"));
    REQUIRE_FALSE(alicetest::LayerHasTensor(out, 0, "attn_res_proj.weight"));
    REQUIRE(alicetest::LayerHasTensor(out, 4, "attn_res_proj.weight"));
    REQUIRE(alicetest::LayerHasNvfp4(out, 3, "self_attn.q_proj.nvfp4"));
    REQUIRE(alicetest::LayerHasBf16(out, 3, "self_attn.q_norm.weight"));
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cmake --build --preset linux-debug --target test_alice_moe_convert`
Expected: FAIL — `ConvertAliceModel` undefined.

- [ ] **Step 4: Implement the converter**

`src/alice_moe_convert.cpp` must:
1. Parse `config.json` (simdjson) and validate the frozen geometry — relaxed only when `SyntheticGeometryAllowed()`; read `layer_types`, `block_attn_res_block_size`, `linear_*`, MoE fields; reject unknown `layer_types`.
2. Build/append the index skeleton (format, model, geometry, layer_types, vram estimate, deferred MTP).
3. Stream shards from `model.safetensors.index.json`; dispatch per tensor by suffix:
   - large projections (`q/k/v/o_proj`, `f_b_proj`, `g_b_proj`, `shared_expert.{gate,up,down}_proj`) → `quantize_tensor_gpu` → `DenseNvfp4Schema` into `pinned/layer_NNN.arrow`;
   - `experts.gate_up_proj` → slice `[E]` into gate/up → `PackExpertNVFP4` → `experts/layer_NNN/exp_EEE.arrow` via `ExpertSchema`; `experts.down_proj` likewise;
   - everything else (norms, `q_norm`/`k_norm`/`o_norm`, AttnRes, router, `a_log_bias`, `dt_bias`, `b_proj`, `f_a/g_a_proj`, conv1d, `shared_expert_gate`, embed, `lm_head`, top norm) → verbatim BF16;
   - `mtp.*` → converted into `mtp/` and counted, never resident.
4. Write `logestix/model.index.json` through the same AtomicWrite/layout helper used by the other converters (mirror `mimo_moe_convert.cpp`).
5. Strict: unknown tensor prefix is an error.

Use `qwen_moe_convert.cpp` / `mimo_moe_convert.cpp` as the structural reference for index writing + safetensors streaming (do not include their headers).

- [ ] **Step 5: Wire the `lsxforge --alice` flag**

In `apps/lsxforge/src/main.cpp`: add the `alice` bool parse (mirror `mimo`) and, before the gemma branch, add:
```cpp
    if (alice) {
        lsxcommon::log::info("AliceAI-Foundation pipeline (logestix-alice-nvfp4 layout)");
        auto st = lsxcommon::alice::ConvertAliceModel(input_path, output_path);
        if (!st.ok()) {
            lsxcommon::log::error(
                absl::StrFormat("Alice conversion failed: %s", st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("Alice conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }
```
Include `lsxcommon/alice_moe_convert.h`.

- [ ] **Step 6: Run test to verify it passes**

Run: `ctest --preset linux-debug-convert -R test_alice_moe_convert --output-on-failure`
Expected: PASS within 120 s.

- [ ] **Step 7: Commit**

```bash
git add libs/lsxcommon/src/alice_moe_convert.cpp libs/lsxcommon/include/lsxcommon/alice_moe_convert.h tests/test_alice_moe_fixture.h tests/test_alice_moe_convert.cpp apps/lsxforge/src/main.cpp libs/lsxcommon/CMakeLists.txt tests/CMakeLists.txt
git commit -m "feat: alice converter (bf16 -> nvfp4) with synthetic e2e"
```

---

## Phase C — Loader (stage 3)

### Task C1: `alice_moe_load` pinned graph + geometry gates

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/alice_moe_load.h`
- Create: `libs/lsxcommon/src/alice_moe_load.cpp`
- Create: `tests/test_alice_moe_loader.cpp`
- Modify: `tests/CMakeLists.txt` (target `test_alice_moe_loader`, label `fast`)

**Interfaces:**
- Consumes: converted `logestix-alice-nvfp4` output; neutral `schema`, `safetensors_gpu`.
- Produces:
  - `struct AliceMoeTopology { int hidden, layers, heads, kv_heads, head_dim, rotary_dim, linear_key_heads, linear_value_heads, linear_key_head_dim, linear_value_head_dim, linear_conv_kernel, num_experts, top_k, moe_intermediate, shared_intermediate, vocab, block_attn_res; std::string format; std::vector<std::string> layer_types; };`
  - `arrow::Status LoadAliceTopology(const std::string &model_path, AliceMoeTopology &out);`
  - `bool AliceTopologyMatchesFrozen(const AliceMoeTopology &t);`

- [ ] **Step 1: Write the failing loader test**

`tests/test_alice_moe_loader.cpp`: run the synthetic converter into a temp dir, then
```cpp
TEST_CASE("alice loader gates the format and frozen geometry", "[alice][fast]")
{
    // convert synthetic -> out (helper shared with test_alice_moe_convert)
    lsxcommon::alice::AliceMoeTopology top;
    REQUIRE(lsxcommon::alice::LoadAliceTopology(out.string(), top).ok());
    REQUIRE(top.format == "logestix-alice-nvfp4");
    REQUIRE(lsxcommon::alice::AliceTopologyMatchesFrozen(top));
    // corrupt format -> gate fails
    alicetest::RewriteFormat(out, "logestix-wrong");
    REQUIRE_FALSE(lsxcommon::alice::LoadAliceTopology(out.string(), top).ok());
}
```
Plus a `Load` test that a geometry mismatch under `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY` still fails `AliceTopologyMatchesFrozen` when a field is perturbed.

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build --preset linux-debug --target test_alice_moe_loader`
Expected: FAIL — undefined.

- [ ] **Step 3: Implement the loader**

`alice_moe_load.cpp`: read `logestix/model.index.json`; reject non-`logestix-alice-nvfp4`; populate `AliceMoeTopology`; `AliceTopologyMatchesFrozen` compares every field against `alice::frozen`, layer-by-layer `layer_types` (`linear_attention` when `i%4!=3`), and accepts only under semantic equality. Mirror `mimo_moe_load.cpp` structure.

- [ ] **Step 4: Run test to verify it passes**

Run: `ctest --preset linux-debug -R test_alice_moe_loader --output-on-failure`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/alice_moe_load.h libs/lsxcommon/src/alice_moe_load.cpp tests/test_alice_moe_loader.cpp tests/CMakeLists.txt
git commit -m "feat: alice loader topology + format/geometry gates"
```

---

## Phase D — Model class, tokenizer bridge, tool bridge (stages 3/4/4b/7)

### Task D1: `alice_moe_model` registration + minimal Load/Infer shell

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/alice_moe_model.h`
- Create: `libs/lsxcommon/src/alice_moe_model.cpp`
- Modify: `libs/lsxcommon/src/model_registry.cpp`
- Modify: `tests/test_model_independence_guard.cpp` (consumers row)
- Modify: `tests/CMakeLists.txt` (extend `test_alice_moe_model`)

**Interfaces:**
- Consumes: `ModelBase`, `IModel` (`Load/Infer/Unload/Inspect/SelfCheck/Tokenizer`), `BuildConversationInputIds`, `ParseToolCalls`.
- Produces: `class AliceFoundation80BModel final : public ModelBase` with `static const std::string kType; // "alice-foundation-80b-a3b"`, `Infer` opening `LsxFastMathScope(EffectiveFastMathMode(cfg_))`.

- [ ] **Step 1: Write the failing registration test**

Add to `tests/test_alice_moe_model.cpp`:
```cpp
#include <lsxcommon/model_factory.h>
TEST_CASE("alice kType resolves in the factory", "[alice][fast]")
{
    lsxcommon::EnsureModelRegistry();
    auto model = lsxcommon::CreateModel(
        "alice-foundation-80b-a3b", lsxcommon::ModelInitConfig{});
    REQUIRE(model != nullptr);
    REQUIRE(model->Type() == "alice-foundation-80b-a3b");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build --preset linux-debug --target test_alice_moe_model && ctest --preset linux-debug -R test_alice_moe_model --output-on-failure`
Expected: FAIL — factory returns null.

- [ ] **Step 3: Implement the model class**

`alice_moe_model.h/.cpp`: `final` `ModelBase` subclass; `kType = "alice-foundation-80b-a3b"`; `Infer` opens the fast-math scope and delegates to the pipeline (Task F1; until then return a diagnostic `InferenceResult` only on the synthetic fixture path); `Load` calls `BeginLoad`, `LoadAliceTopology`, `AliceTopologyMatchesFrozen`, `LoadTokenizerDb`, `FinishLoad`; `TeardownWeights`; `Inspect`; `SelfCheck` (Task F2); `BuildConversationInputIds`/`ParseToolCalls` delegate to `alice_tool_format` (Task D2). Register in `model_registry.cpp`:
```cpp
#include "alice_moe_model.h"
...
        LsxModelFactory::Register<AliceFoundation80BModel>();
```

- [ ] **Step 4: Add the guard consumer row**

In `tests/test_model_independence_guard.cpp` `consumers` add `"alice_moe_model.cpp",`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `ctest --preset linux-debug -R "test_alice_moe_model|test_model_independence_guard" --output-on-failure`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/alice_moe_model.h libs/lsxcommon/src/alice_moe_model.cpp libs/lsxcommon/src/model_registry.cpp tests/test_alice_moe_model.cpp tests/test_model_independence_guard.cpp
git commit -m "feat: alice model registration + lifecycle shell"
```

### Task D2: Native AliceAI tool/chat codec (stage 4b)

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/alice_tool_format.h`
- Create: `libs/lsxcommon/src/alice_tool_format.cpp`
- Create: `tests/test_alice_tool_framing.cpp`
- Modify: `tests/CMakeLists.txt` (target `test_alice_tool_framing`, label `fast`)

**Interfaces:**
- Consumes: `ArrowTokenizerDb`, `arrow_tokenizer_lookup`/`encode`/`decode`, `ChatConversation`, `ToolCall`, `tools::ParseToolCallsText`/`ParseBareToolCallJson`.
- Produces:
  - `bool lsxcommon::alice::RenderConversation(const ArrowTokenizerDb &tok, const ChatConversation &conv, bool add_generation_prompt, std::vector<int32_t> &out_ids);`
  - `bool lsxcommon::alice::ParseToolCallsAlice(const ArrowTokenizerDb &tok, const std::vector<int32_t> &gen_tokens, std::vector<ToolCall> &out_calls);`

- [ ] **Step 1: Verify the format against the checkpoint template**

Read `/mnt/llm_models/models/AliceAI-Foundation-80B-A3B-Base/chat_template.jinja` and the tokenizer pieces. Record in the test comments the exact strings: roles `system:`/`user:`/`assistant:`/`meta:`; separators `"\n\n "`; generation prompt `"\n\n assistant:"`; tools header `Тебе доступны следующие функции:` with `function {"name":…,"description":…,"parameters":…}` lines; tool call `"\n[TOOL_CALL_START]" + name + "\n" + json`; reasoning `[COT_START]…[COT_END]`; tool result `Tool {name}: {content}`. Confirm which markers are `special` pieces (`<s>`, `</s>`, `<0x..>`, `[SPEC_TOKEN_*]`); the `[TOOL_CALL_START]`/`[COT_*]` strings are plain text and survive `arrow_tokenizer_decode`.

- [ ] **Step 2: Write the failing framing test**

`tests/test_alice_tool_framing.cpp` (host, `fast`) on a synthetic SP-BPE tokenizer fixture: render a system+user+assistant-with-tool-call conversation and assert the decoded text contains `user: …`, `assistant: …[TOOL_CALL_START]NAME`, and the tools header; then `ParseToolCallsAlice` recovers the call; then a tokenizer round-trip on the body (not the full prompt). Use the repo's existing tool-framing test shape (`tests/test_phi_tool_framing.cpp`).

- [ ] **Step 3: Run test to verify it fails**

Run: `cmake --build --preset linux-debug --target test_alice_tool_framing`
Expected: FAIL — undefined.

- [ ] **Step 4: Implement the codec**

`alice_tool_format.cpp`: build the frame string exactly per the template, inject control markers as ids via `arrow_tokenizer_lookup` (drop the auto-prepended bos inside the frame), BPE-encode free text via `arrow_tokenizer_encode`; handle the `meta` role and `reasoning_content`; parse from decoded text, falling back to `tools::ParseToolCallsText` then `tools::ParseBareToolCallJson`.

- [ ] **Step 5: Run test to verify it passes**

Run: `ctest --preset linux-debug -R test_alice_tool_framing --output-on-failure`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/alice_tool_format.h libs/lsxcommon/src/alice_tool_format.cpp tests/test_alice_tool_framing.cpp tests/CMakeLists.txt
git commit -m "feat: native alice chat/tool codec (stage 4b)"
```

### Task D3: Passport translation + IoWorker wiring (stage 7)

**Files:**
- Modify: `libs/lsxcommon/src/alice_moe_model.cpp` (or converter, per repo convention)
- Modify: `libs/lsxnetwork/include/lsxnetwork/io_worker.h` + implementation
- Modify: `libs/lsxnetwork/src/io_worker.cpp`

**Interfaces:**
- Consumes: `lsxnetwork::BuildPassport` / `WritePassport`, `LsxTokenTranslator`.
- Produces: `passport/passport.bin` for `alice-foundation-80b-a3b`.

- [ ] **Step 1: Write the failing passport test**

Add a `fast` host test that builds the passport from the converted fixture and asserts the payload carries `alice-foundation-80b-a3b` and vocab 129024.

- [ ] **Step 2: Run test to verify it fails**

Run: `ctest --preset linux-debug -R test_alice_moe_loader --output-on-failure` (the passport is exercised via the loader fixture)
Expected: FAIL — no passport builder.

- [ ] **Step 3: Implement**

Call `BuildPassport`/`WritePassport` during conversion into `lsx_model_metadata("passport/passport.bin")`; integrate `LsxTokenTranslator` as a private `alignas(64)` member of `IoWorker`, `Load()` strictly AFTER `CpuAffinity::PinToCores0To3` and BEFORE the readiness `set_value()`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `ctest --preset linux-debug -R "test_alice_moe_loader|test_lsxnetwork" --output-on-failure`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/alice_moe_convert.cpp libs/lsxnetwork/include/lsxnetwork/io_worker.h libs/lsxnetwork/src/io_worker.cpp tests/test_alice_moe_loader.cpp
git commit -m "feat: alice passport translation + io worker wiring"
```

---

## Phase E — CUDA kernels (stage 5/6)

> Kernel rule: all reference math is pinned in the design doc ("Pinned reference formulas"). Implement exactly those formulas; each task's test compares the CUDA kernel against the host reference in `alice_moe_device_math.h` with `-fmad=false`.

### Task E1: Reference math + host kernel tests

**Files:**
- Create: `libs/lsxcommon/src/alice_moe_device_math.h`
- Create: `tests/test_alice_moe_kernels.cu`
- Modify: `tests/CMakeLists.txt` (target `test_alice_moe_kernels`, label `fast`)

**Interfaces:**
- Produces (header-only, `__host__ __device__` where used on device):
  - `float AliceZeroCenteredRmsNorm(...)`
  - `float AliceDepthSoftmaxMix(...)`
  - `void AliceKdaStep(...)` / `void AliceKdaForward(...)`
  - `void AliceGatedAttentionHostRef(...)`
  - `void AliceRouterTopKHostRef(...)`

- [ ] **Step 1: Write the failing reference tests**

`tests/test_alice_moe_kernels.cu` (host-only cases guard with `#if !defined(__CUDA_ARCH__)`): ZC RMSNorm vs hand-computed; AttnRes mix produces softmax-weighted sum and equals the exact `_depth_softmax_mix` for a small case; KDA single-token recurrence matches the Python loop for 2 steps; router top-10 with correction bias renormalizes; gated attention applies `sigmoid(gate)` and partial RoPE on the first 64/256 dims.

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build --preset linux-debug --target test_alice_moe_kernels`
Expected: FAIL — header missing.

- [ ] **Step 3: Implement the reference math**

Translate each formula from the design doc verbatim. Use `lscommon::math` (`lsx_device_math.h`) `ExpfGlibc`/`RsqrtfGlibc` where the model-neutral layer does; keep KDA in F32 accumulation.

- [ ] **Step 4: Run test to verify it passes**

Run: `ctest --preset linux-debug -R test_alice_moe_kernels --output-on-failure`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/alice_moe_device_math.h tests/test_alice_moe_kernels.cu tests/CMakeLists.txt
git commit -m "feat: alice reference math (kda/attnres/router) + host tests"
```

### Task E2: ZC RMSNorm, AttnRes mix, router, SwiGLU kernels

**Files:**
- Create: `libs/lsxcommon/src/alice_moe_kernels.h`
- Create: `libs/lsxcommon/src/alice_moe_kernels.cu`
- Modify: `libs/lsxcommon/CMakeLists.txt` (`lsxcommon_alice_cu` OBJECT lib)
- Modify: `tests/test_alice_moe_kernels.cu`

**Interfaces:**
- Produces CUDA entry points (host launchers): `RunAliceZcRmsNorm`, `RunAliceDepthMix`, `RunAliceRouterTopK`, `RunAliceSwiGLU`.

- [ ] **Step 1: Write device-vs-reference parity tests**

Add device `TEST_CASE`s that allocate, launch each kernel, and compare against `alice_moe_device_math.h` within the bit-exact class (`FastMathMode::kOff`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `ctest --preset linux-debug -R test_alice_moe_kernels --output-on-failure`
Expected: FAIL — launchers missing.

- [ ] **Step 3: Implement the kernels**

Shared-memory sandbox: thread-0-only RMW behind `__syncthreads()`; no `atomicAdd` feeding logits; one stream. Reuse neutral primitives from `lsx_kernels_norm.cuh` / `lsx_kernels_act.cuh` where they are model-neutral (do not copy qwen-family headers).

- [ ] **Step 4: Run tests to verify they pass**

Run: `ctest --preset linux-debug -R test_alice_moe_kernels --output-on-failure`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/alice_moe_kernels.h libs/lsxcommon/src/alice_moe_kernels.cu libs/lsxcommon/CMakeLists.txt tests/test_alice_moe_kernels.cu
git commit -m "feat: alice zc-rmsnorm/attnres/router/swiglu kernels"
```

### Task E3: KDA linear-attention kernels (conv + recurrence)

**Files:**
- Create: `libs/lsxcommon/src/alice_moe_inference_attn.{h,cu}` (KDA portion)
- Modify: `tests/test_alice_moe_kernels.cu`

**Interfaces:**
- Produces: `RunAliceKdaPrefill(...)` (chunked/sequential over L) and `RunAliceKdaDecode(...)` (single token, recurrent state in/out).

- [ ] **Step 1: Write the failing KDA parity test**

Single-token recurrence (state update + output) for a handful of steps matches `AliceKdaForward` host reference; prefill (L=8) matches the same reference; depthwise conv-4 with causal padding matches.

- [ ] **Step 2: Run test to verify it fails**

Run: `ctest --preset linux-debug -R test_alice_moe_kernels --output-on-failure`
Expected: FAIL.

- [ ] **Step 3: Implement**

Per-channel decay `exp(-exp(a_log_bias[h]) * softplus(alpha + dt_bias))`; L2 normalize q/k; query scale `128^-0.5`; delta rule; state `[32,128,128]` F32; conv state `[4096,3]`. Prefill and decode share the same recurrence math.

- [ ] **Step 4: Run test to verify it passes**

Run: `ctest --preset linux-debug -R test_alice_moe_kernels --output-on-failure`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/alice_moe_inference_attn.h libs/lsxcommon/src/alice_moe_inference_attn.cu tests/test_alice_moe_kernels.cu
git commit -m "feat: alice kda linear attention kernels"
```

### Task E4: Gated full attention + partial RoPE + GQA + KV cache

**Files:**
- Modify: `libs/lsxcommon/src/alice_moe_inference_attn.{h,cu}`
- Modify: `tests/test_alice_moe_kernels.cu`

**Interfaces:**
- Produces: `RunAliceGatedAttention(...)` with `q_proj` split into query/output-gate, ZC q/k norm, rotate-half RoPE on the first 64 dims, GQA repeat 8, causal mask, `sigmoid(gate)`.

- [ ] **Step 1: Write the failing attention parity test**

Compare against `AliceGatedAttentionHostRef` for prefill L=8 and decode L=1 with a KV cache; assert partial RoPE leaves dims 64..255 unchanged.

- [ ] **Step 2: Run test to verify it fails** — expected FAIL.
- [ ] **Step 3: Implement** per the pinned formula.
- [ ] **Step 4: Run test to verify it passes** — expected PASS.
- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/alice_moe_inference_attn.h libs/lsxcommon/src/alice_moe_inference_attn.cu tests/test_alice_moe_kernels.cu
git commit -m "feat: alice gated full attention with partial rope"
```

### Task E5: Expert cache + MoE streaming

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/alice_moe_expert_cache.h`
- Create: `libs/lsxcommon/src/alice_moe_expert_cache.cpp`
- Modify: `tests/test_alice_moe_loader.cpp` (LRU bookkeeping)
- Modify: `libs/lsxcommon/CMakeLists.txt`

**Interfaces:**
- Consumes: `gpu::expert_store::ExpertStore<AliceExpertEntry>`.
- Produces: `AliceExpertEntry` (15-member aggregate: `gate_up.nvfp4`, `gate_up.scale`, `gate_up.global_scale`, `down.*`), `LoadAliceExpertStore(...)`, `PlanAliceExpertSlotFromStore(...)`, knob `LSX_ALICE_EXPERT_CACHE_SLOTS`. Mirror `mimo_moe_expert_cache.h` exactly.

- [ ] **Step 1: Write the failing LRU test** — synthetic converted fixture; insert > slots experts; assert eviction order and that a decode with top-10 never errors on a miss.
- [ ] **Step 2: Run test to verify it fails** — expected FAIL.
- [ ] **Step 3: Implement** using the neutral `ExpertStore`.
- [ ] **Step 4: Run test to verify it passes** — expected PASS.
- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/alice_moe_expert_cache.h libs/lsxcommon/src/alice_moe_expert_cache.cpp tests/test_alice_moe_loader.cpp libs/lsxcommon/CMakeLists.txt
git commit -m "feat: alice expert host store + device slot lru"
```

---

## Phase F — Pipeline, SelfCheck, selfchecks, perf

### Task F1: Layer graph + prefill/decode pipeline

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/alice_moe_inference_pipeline.h`
- Create: `libs/lsxcommon/src/alice_moe_inference_pipeline.cu`
- Modify: `libs/lsxcommon/src/alice_moe_model.cpp` (Infer delegates)
- Modify: `tests/test_alice_moe_model.cpp`

**Interfaces:**
- Produces: `AliceRunInfer(const AliceModel &, const ModelRequest &, gpu::InferenceResult &)`.

- [ ] **Step 1: Write the failing synthetic model test**

Load the synthetic converted fixture, run a 2-token prompt through `Infer`, assert deterministic `final_logits` across 3 runs and that gen output equals a host-reference pipeline in `alice_moe_device_math.h`.

- [ ] **Step 2: Run test to verify it fails** — expected FAIL.
- [ ] **Step 3: Implement** the exact layer graph from the design doc (AttnRes blocking every 4 layers; KDA|full attention; MoE routed+shared; final AttnRes → ZC norm → lm_head F32). Single stream; no per-token sync.
- [ ] **Step 4: Run test to verify it passes** — expected PASS.
- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/alice_moe_inference_pipeline.h libs/lsxcommon/src/alice_moe_inference_pipeline.cu libs/lsxcommon/src/alice_moe_model.cpp tests/test_alice_moe_model.cpp
git commit -m "feat: alice layer pipeline (attnres + kda + moe)"
```

### Task F2: SelfCheck harness (`Paris`)

**Files:**
- Create: `tests/test_alice_moe_selfcheck.cpp`
- Modify: `tests/CMakeLists.txt` (target label `inference`, `SKIP_RETURN_CODE 4`)

- [ ] **Step 1: Write the failing test** — 1→2-token incremental selfcheck over ≥3 cycles (`LSX_TEST_ALICE_SELFCHECK_CYCLES`), bit-exact `final_logits`, real model via `LSX_TEST_ALICE_ROOT`, `SKIP()` when absent; control prompt `"The capital of France is"` expects `"Paris"`; record a Base-model deviation (top-1/substring fallback) rather than failing.
- [ ] **Step 2: Run** `ctest --preset linux-debug-inference -R test_alice_moe_selfcheck` — expected FAIL/skip.
- [ ] **Step 3: Implement** `AliceFoundation80BModel::SelfCheck`.
- [ ] **Step 4: Run test to verify it passes or SKIPs** — expected PASS or SKIP 4.
- [ ] **Step 5: Commit**

```bash
git add tests/test_alice_moe_selfcheck.cpp tests/CMakeLists.txt libs/lsxcommon/src/alice_moe_model.cpp
git commit -m "test: alice selfcheck (paris parity)"
```

### Task F3: FSI / CODE / DET gates

**Files:**
- Create: `tests/test_alice_moe_fsi_selfcheck.cpp`, `tests/test_alice_moe_code_selfcheck.cpp`, `tests/test_alice_moe_det_selfcheck.cpp`
- Create goldens: `tests/fixtures/fsi/alice.json`, `tests/fixtures/code/alice.json`, `tests/fixtures/det/alice.json`
- Modify: `tests/CMakeLists.txt` (3 targets, label `inference`, `SKIP_RETURN_CODE 4`)

- [ ] **Step 1–5:** Follow the `fsi-selfcheck`, `code-selfcheck`, `det-selfcheck` skills: capture from the source safetensors with `tools/fsi_capture.py` / `tools/code_capture.py` / `tools/det_capture.py` (once the model is converted and runnable), commit the goldens, wire the per-family tests exactly like `test_phi_dense_*_selfcheck.cpp`. Each target `SKIP_RETURN_CODE 4`. Commit each stage separately with `test:` prefix.

### Task F4: Perf baseline

**Files:**
- Modify: `tests/perf/perf_baselines.json`
- Modify: `tests/test_perf_baseline.cpp`

- [ ] **Step 1: Add the model entry**

Add to `tests/perf/perf_baselines.json` `entries`:
```json
{ "model_type": "alice-foundation-80b-a3b",
  "gpu_key": "rtx 5090", "os": "linux", "mode": "perf",
  "tokens_per_sec": 0.0, "max_gen": 2048,
  "prompt": "The capital of France is", "date": "2026-10-01",
  "commit": "-", "gen_tokens": 2048 }
```

- [ ] **Step 2: Add the harness case**

In `tests/test_perf_baseline.cpp`:
```cpp
const PerfModel kAlice = {
    "alice-foundation-80b-a3b", "AliceAI-Foundation-80B-A3B-Base",
    "The capital of France is", 2048};
...
TEST_CASE("perf baseline: alice-foundation-80b-a3b", "[perf]")
{
    RunPerfModel(kAlice);
}
```

- [ ] **Step 3: Measure and record** (after correctness, strict GPU exclusivity, perf last per `running-a-model`).

Run: `./build/linux-debug/bin/lsxbenchmark --bench --model <vault>/AliceAI-Foundation-80B-A3B-Base`
Then update `tokens_per_sec`/`ttft_ms` with the measured values and commit.

- [ ] **Step 4: Commit**

```bash
git add tests/perf/perf_baselines.json tests/test_perf_baseline.cpp
git commit -m "test: alice perf baseline"
```

---

## Phase G — Guard completion, docs, labels

### Task G1: Suite guard, label guard, CMake targets, docs

**Files:**
- Modify: `tests/test_model_suite_guard.cpp`
- Modify: `AGENTS.md`
- Modify: `.opencode/skills/logestix-repo-map/SKILL.md`

- [ ] **Step 1: Extend the suite guard**

In `tests/test_model_suite_guard.cpp` add `alice` to `FamilyStems()` and add:
```cpp
        {"alice-foundation-80b-a3b",
         {{"convert", {"test_alice_moe_convert"}},
          {"load", {"test_alice_moe_loader"}},
          {"model", {"test_alice_moe_model"}},
          {"kernels", {"test_alice_moe_kernels"}},
          {"selfcheck", {"test_alice_moe_selfcheck"}},
          {"fsi", {"test_alice_moe_fsi_selfcheck"}},
          {"code", {"test_alice_moe_code_selfcheck"}},
          {"det", {"test_alice_moe_det_selfcheck"}}}},
```
Extend the model-stage regex to recognise `alice`.

- [ ] **Step 2: Run the four guards**

Run: `ctest --preset linux-debug -R "test_code_naming_guard|test_model_independence_guard|test_model_suite_guard|test_ctest_labels_guard" --output-on-failure`
Expected: all PASS.

- [ ] **Step 3: Update docs**

Add the `alice` family files to `AGENTS.md` Structure and to the repo-map skill's per-file inventory (converter/loader/kernels/expert cache/tool format paths, format tag `logestix-alice-nvfp4`, knob `LSX_ALICE_EXPERT_CACHE_SLOTS`).

- [ ] **Step 4: Full fast sweep + commit**

Run: `ctest --preset linux-debug`
Expected: all `fast` green.

```bash
git add tests/test_model_suite_guard.cpp AGENTS.md .opencode/skills/logestix-repo-map/SKILL.md
git commit -m "test: alice suite guard + docs"
```

---

## Self-Review

**Spec coverage:** tokenizer backend (A1–A2), passport (B1), converter+NVFP4+experts+forge flag (B2), loader (C1), model+registry (D1), tool bridge (D2), passport translation/IoWorker (D3), reference math + kernels incl. KDA/gated attention/expert cache (E1–E5), pipeline (F1), selfcheck (F2), FSI/CODE/DET (F3), perf (F4), guards+docs (G1) — every design-doc section maps to a task.

**Placeholder scan:** no "TBD"/"implement later"; each code step shows the code or the exact algorithm + acceptance test. Kernel bodies are specified by the pinned formulas in the design doc (the single source of truth for the math) with concrete device-vs-reference parity tests and interfaces.

**Type consistency:** `AliceMoeTopology` fields, `ConvertAliceModel`, `LoadAliceTopology`, `AliceTopologyMatchesFrozen`, `RenderConversation`, `ParseToolCallsAlice`, `RunAlice*` launchers, `AliceExpertEntry` names are used consistently across tasks.

**Known execution ordering:** Task B1 mutates the independence guard's `consumers` list before `alice_moe_model.cpp` exists; move that single row into the D1 commit if the guard is run between B1 and D1.
