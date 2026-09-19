# Zero-Alloc, Zero-Copy Arrow Binary Tokenizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the runtime `tokenizer.json` dependency and the heap-allocating `BpeTokenizer` with a zero-allocation, zero-copy BPE tokenizer that reads two memory-mapped Arrow files (`tokenizer_vocab.arrow`, `tokenizer_merges.arrow`) produced at build time by `lsxforge --tokenizer`.

**Architecture:** Two Arrow files per model under `logestix/pinned/`: a 3-column vocab batch (rows: `token_id`, `token_bytes` LargeBinary, `is_special` Int8) plus schema metadata (`bos_id/eos_id/pad_id/vocab_size/byte_to_token_id`), and a 4-column merges batch sorted by `(merge_left, merge_right)`: `merge_left`, `merge_right`, `merge_rank`, `merge_token_id` plus `merge_count` metadata. The runtime loads both via `arrow::io::MemoryMappedFile` + `load_tokenizer_metadata`, then encodes with a zero-alloc streaming pre-tokenizer (faithful GPT-2/fancy-regex semantics, Unicode classes from generated `unicode_tables.inc`) + stack-array BPE (binary search over the sorted merge table, min-rank selection). `merge_token_id` is always resolved by the converter via vocab lookup of `bytes(A) ++ bytes(B)` — never by formula.

**Tech Stack:** C++20, Arrow IPC 25.0 (mmap zero-copy), simdjson (build-time converter only), Catch2 v3 (tests), PowerShell (UCD table generator), CUDA 13.3 (integration only)

---

## Design Notes (verified against two reference implementations)

The pre-tokenizer is a `Sequence` of three `Split` pretokenizers, each passing its word list to the next:

1. `\p{N}{1,3}` Isolated — greedily isolate number runs in pieces of ≤ 3 consecutive `\p{N}` codepoints.
2. `[一-龥぀-ゟ゠-ヿ]+` Isolated — isolate CJK ideograph (U+4E00–U+9FA5) and Hiragana/Katakana (U+3040–U+309F, U+30A0–U+30FF) runs.
3. Main GPT-2 regex, 6 alternatives in priority order, leftmost match, greedy-with-backtracking:
   - alt1 `[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+`
   - alt2 `[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+`
   - alt3 ` ?[\p{P}\p{S}]+[\r\n]*`
   - alt4 `\s*[\r\n]+`
   - alt5 `\s+(?!\S)`
   - alt6 `\s+`

Semantics pinned empirically (matches the Rust `tokenizers` 0.13.4-rc1 and `@huggingface/tokenizers` 0.1.3 references). These are the exact matcher rules the streaming splitter must implement:

- **alt1**: one ASCII punct from `!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~` (the full ASCII `\p{P}` set) followed by ≥1 ASCII letter `[A-Za-z]`. `"!abc"` → word `"!abc"`; `"a?x"` → `"a"` + `"?x"`.
- **alt2**: `[\p{L}\p{M}]+` (≥1 Unicode letter/mark run), optionally preceded by ONE codepoint that is **not** `\r`/`\n`/L/P/S — that "other" class includes whitespace. **This prefix is the space-glue mechanism**: a single whitespace char followed by a letter attaches to the word. `" a"` → ONE word `" a"`; after alt5 already emitted `" "`, the rest `" x"` is ONE word via the prefix consuming the second space. If the prefix matches a non-L/M codepoint but no L/M follows, the alternative fails entirely.
- **alt3**: optional single U+0020 (only `' '`, not `\t`/`\n`) before a run of ≥1 Unicode P/S, then optional trailing ASCII `\r`/`\n`. `"  !"` run handling: `"a !"` → `"a"` + `" !"`.
- **alt4**: `\s*[\r\n]+` — whitespace run (Unicode White_Space) that ends in ≥1 `\r`/`\n`; equivalently match `[pos, last_crlf+1)` where `last_crlf` is the last `\r`/`\n` in the run. `"\r\n"` → one word (BPE splits it into two byte tokens because no merge produces `\r\n`); `"\n\n"` → one word → merges to token 271.
- **alt5**: `\s+(?!\S)`. For a whitespace run `[pos, k)` (Unicode White_Space): if `k == text_len` the lookahead sees end-of-string → match the whole run `[pos, k)`; else the next char is non-whitespace so the lookahead fails and the engine backtracks to `[pos, k-1)` (lookahead then sees whitespace) — valid only when `k-1 > pos`; a single whitespace char followed by non-whitespace fails alt5.
- **alt6**: `\s+` — only reached when alt5 failed, i.e. a single whitespace char followed by a non-whitespace, non-L/M, non-P/S char (an "Other" control char). Matches `[pos, k)`.
- **Gap (Isolated) semantics**: the main regex is applied left-to-right (find_iter); any non-matching run between matches — and before the first / after the last match — becomes one word. Concretely: `"a\x01"` → `"a"` + `"\x01"`; `"a\x01\x02b"` → `"a"` + `"\x01"` + `"\x02b"`.
- **Splits 1/2 boundaries**: the main regex never sees `\p{N}` or CJK-run codepoints, so the matcher stops L/M runs and gaps at numbers and CJK chars. `"1abc"` → `"1"` + `"abc"`; `"a中b"` → `"a"` + `"中"` + `"b"`. Number runs yield ≤3 number codepoints per word: `"12345"` → `"123"` + `"45"`.
- **ByteLevel** (`use_regex=false`, `add_prefix_space=false`) is a no-op in the raw-byte domain.

Reference goldens (captured from both reference implementations, source of truth):

| input | tokens (real gpt2 model) | tokens (synthetic fixture) |
|---|---|---|
| `"hello world"` | — | `[262, 267]` |
| `"hello"` | — | `[262]` |
| `"  x"` | `[223, 1527]` | `[35, 35, 123]` |
| `" a"` | `[260]` | `[35, 100]` |
| `"a "` | `[67, 223]` | `[100, 35]` |
| `"a b"` | `[67, 291]` | `[100, 35, 101]` |
| `"a  b"` | `[67, 223, 291]` | `[100, 35, 35, 101]` |
| `"x  y  z"` | `[90, 223, 383, 223, 961]` | `[123, 35, 35, 124, 35, 35, 125]` |
| `"trailing  "` | `[19660, 7400, 262]` | `[119, 117, 100, 108, 111, 108, 113, 106, 35, 35]` |
| `"  leading spaces"` | `[223, 6646, 13564]` | `[35, 35, 111, 104, 100, 103, 108, 113, 106, 35, 118, 115, 100, 102, 104, 118]` |

The word splits implied by the real-model goldens (requiring merges like `"Ġ x"` rank 1268, `"Ġ a"` rank 1, `"Ġ b"` to fuse within one word): `"  x"` → `[" ", " x"]`; `"a  b"` → `["a", " ", " b"]`; `" a"` → `[" a"]`; `"trailing  "` → `["trailing", "  "]`; `"x  y  z"` → `["x", " ", " y", " ", " z"]`; `"  leading spaces"` → `[" ", " leading", " spaces"]`. The streaming splitter in Task 5 reproduces exactly these.

**BOS prepend:** the Arrow encoder prepends `bos_id` to every result (same behavior as the legacy `BpeTokenizer::Encode`), so the synthetic encode goldens below carry a leading `0`.

## File Structure

```
tools/
└── generate_unicode_tables.ps1     NEW — pinned UCD 16.0.0 → unicode_tables.inc
libs/lsxcommon/
├── include/lsxcommon/
│   ├── tokenizer.h                 MODIFY — keep BpeTokenizer; add ArrowTokenizerDb + arrow API
│   ├── pretokenizer.h              NEW — WordStream streaming splitter decls
│   ├── unicode.h                   NEW — CharClass + classify() decls
│   ├── byte_unicode.h              NEW — GPT-2 byte↔Unicode table + key encode/decode
│   ├── schema.h                    MODIFY — VocabRow/MergeRow + TokenizerVocabSchema/TokenizerMergesSchema
│   └── converter.h                 MODIFY — BuildTokenizerArrowFiles decl
├── src/
│   ├── unicode_tables.inc          NEW — generated range tables (committed)
│   ├── unicode.cpp                 NEW — classify() over the tables
│   ├── pretokenizer.cpp            NEW — WordStream (single-pass 3-split matcher)
│   ├── tokenizer_convert.cpp       NEW — BuildTokenizerArrowFiles (simdjson converter)
│   ├── schema.cpp                  MODIFY — new schema classes
│   ├── tokenizer.cpp               MODIFY — keep BpeTokenizer; add loader + encode + decode
│   ├── inference_pipeline.cu       MODIFY — mmap + load_tokenizer_metadata + new encode/decode
│   └── CMakeLists.txt              MODIFY — add unicode.cpp, pretokenizer.cpp, tokenizer_convert.cpp
apps/lsxforge/
└── src/main.cpp                    MODIFY — --tokenizer <model_path> CLI mode
tests/
├── data/synthetic_tok.json         NEW — synthetic fixture (moved from repo root)
├── test_tokenizer.cpp              NEW — all tokenizer tests
└── CMakeLists.txt                  MODIFY — add test_tokenizer target + data dir
```

---

### Task 1: Unicode classification tables

**Files:**
- Create: `tools/generate_unicode_tables.ps1`
- Create: `libs/lsxcommon/include/lsxcommon/unicode.h`
- Create: `libs/lsxcommon/src/unicode.cpp`
- Create: `libs/lsxcommon/src/unicode_tables.inc` (generated)
- Modify: `libs/lsxcommon/CMakeLists.txt`
- Test: `tests/test_tokenizer.cpp` (new file, unicode section)

- [ ] **Step 1: Write the failing unicode spot-check test**

Create `tests/test_tokenizer.cpp`:

```cpp
#include <catch2/catch_test_macros.hpp>
#include <lsxcommon/unicode.h>

TEST_CASE("unicode classification spot checks", "[tokenizer][unicode]")
{
    using lsxcommon::unicode::CharClass;
    CHECK(lsxcommon::unicode::classify(0x41) == CharClass::Letter);       // 'A'
    CHECK(lsxcommon::unicode::classify(0x61) == CharClass::Letter);       // 'a'
    CHECK(lsxcommon::unicode::classify(0x4E2D) == CharClass::Letter);     // 中
    CHECK(lsxcommon::unicode::classify(0x9FA6) == CharClass::Letter);     // 龦 (L, outside CJK split range)
    CHECK(lsxcommon::unicode::classify(0x0301) == CharClass::Mark);       // combining acute
    CHECK(lsxcommon::unicode::classify(0x30) == CharClass::Number);       // '0'
    CHECK(lsxcommon::unicode::classify(0x00B2) == CharClass::Number);     // ² (No)
    CHECK(lsxcommon::unicode::classify(0x21) == CharClass::Punct);        // '!'
    CHECK(lsxcommon::unicode::classify(0x20AC) == CharClass::Symbol);     // €
    CHECK(lsxcommon::unicode::classify(0x20) == CharClass::Whitespace);   // space
    CHECK(lsxcommon::unicode::classify(0x00A0) == CharClass::Whitespace); // NBSP
    CHECK(lsxcommon::unicode::classify(0x0A) == CharClass::Whitespace);   // \n
    CHECK(lsxcommon::unicode::classify(0x09) == CharClass::Whitespace);   // \t
    CHECK(lsxcommon::unicode::classify(0x0001) == CharClass::Other);      // control char
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"; cmake --build build --config Debug --target test_tokenizer 2>&1 | Select-Object -Last 5`
Expected: FAIL — `lsxcommon/unicode.h: No such file or directory`

- [ ] **Step 3: Write the generator script**

Create `tools/generate_unicode_tables.ps1`:

```powershell
$ErrorActionPreference = "Stop"
param(
    [string]$UcdVersion = "16.0.0",
    [string]$OutFile = "libs/lsxcommon/src/unicode_tables.inc"
)

$base = "https://www.unicode.org/Public/$UcdVersion/ucd"
$gcat = Join-Path $env:TEMP "ucd_DerivedGeneralCategory.txt"
$prop = Join-Path $env:TEMP "ucd_PropList.txt"
Invoke-WebRequest -Uri "$base/extracted/DerivedGeneralCategory.txt" -OutFile $gcat
Invoke-WebRequest -Uri "$base/PropList.txt" -OutFile $prop

function Parse-Ranges($path, [scriptblock]$filter) {
    $ranges = New-Object System.Collections.Generic.List[object]
    foreach ($line in Get-Content $path) {
        $trim = $line.Trim()
        if ($trim -eq "" -or $trim.StartsWith("#")) { continue }
        $code = ($line -split "#")[0].Trim()
        if ($code -notmatch '^([0-9A-Fa-f]+)(?:\.\.([0-9A-Fa-f]+))?\s*;\s*(\S+)') { continue }
        $lo = [Convert]::ToUInt32($Matches[1], 16)
        $hi = if ($Matches[2]) { [Convert]::ToUInt32($Matches[2], 16) } else { $lo }
        $cat = $Matches[3]
        if (& $filter $cat) { $ranges.Add(@($lo, $hi)) }
    }
    $sorted = @($ranges | Sort-Object { $_[0] })
    $merged = New-Object System.Collections.Generic.List[object]
    foreach ($r in $sorted) {
        if ($merged.Count -gt 0 -and $r[0] -le $merged[$merged.Count - 1][1] + 1) {
            if ($r[1] -gt $merged[$merged.Count - 1][1]) { $merged[$merged.Count - 1][1] = $r[1] }
        } else { $merged.Add(@($r[0], $r[1])) }
    }
    return $merged
}

$L  = Parse-Ranges $gcat { param($c) $c -in @('Lu','Ll','Lt','Lm','Lo') }
$M  = Parse-Ranges $gcat { param($c) $c -in @('Mn','Mc','Me') }
$N  = Parse-Ranges $gcat { param($c) $c -in @('Nd','Nl','No') }
$P  = Parse-Ranges $gcat { param($c) $c -in @('Pc','Pd','Ps','Pe','Pi','Pf','Po') }
$S  = Parse-Ranges $gcat { param($c) $c -in @('Sm','Sc','Sk','So') }
$WS = Parse-Ranges $prop { param($c) $c -eq 'White_Space' }

function Emit-Ranges($name, $ranges) {
    $lines = "constexpr CodeRange $name[] = {"
    foreach ($r in $ranges) {
        $lines += "`n    0x{0:X}, 0x{1:X}," -f $r[0], $r[1]
    }
    $lines += "`n};"
    $lines += "`nconstexpr size_t k$name" + "Count = sizeof($name) / sizeof($name[0]);"
    return $lines
}

$header = @"
// Generated by tools/generate_unicode_tables.ps1 from Unicode $UcdVersion UCD.
// DO NOT EDIT.
#ifndef LOGESTIX_UNICODE_TABLES_INC
#define LOGESTIX_UNICODE_TABLES_INC
#include <cstddef>
#include <cstdint>
namespace lsxcommon {
namespace unicode {
struct CodeRange { uint32_t first; uint32_t last; };

"@

$body = @(
    (Emit-Ranges "kLetterRanges" $L),
    (Emit-Ranges "kMarkRanges" $M),
    (Emit-Ranges "kNumberRanges" $N),
    (Emit-Ranges "kPunctRanges" $P),
    (Emit-Ranges "kSymbolRanges" $S),
    (Emit-Ranges "kWhitespaceRanges" $WS)
)

$footer = @"

} // namespace unicode
} // namespace lsxcommon
#endif // LOGESTIX_UNICODE_TABLES_INC
"@

$content = $header + ($body -join "`n`n") + $footer
Set-Content -Path $OutFile -Value $content -Encoding ascii
Write-Host "wrote $OutFile ($($L.Count) L, $($M.Count) M, $($N.Count) N, $($P.Count) P, $($S.Count) S, $($WS.Count) WS ranges)"
```

- [ ] **Step 4: Run the generator**

Run: `powershell -ExecutionPolicy Bypass -File tools/generate_unicode_tables.ps1`
Expected: prints `wrote libs/lsxcommon/src/unicode_tables.inc (...)`; the file exists and contains the 6 `CodeRange` arrays.

- [ ] **Step 5: Write `unicode.h`**

```cpp
#pragma once

#include <cstdint>

namespace lsxcommon {
namespace unicode {

enum class CharClass { Letter, Mark, Number, Punct, Symbol, Whitespace, Other };

CharClass classify(uint32_t cp);
bool is_letter(uint32_t cp);
bool is_mark(uint32_t cp);
bool is_number(uint32_t cp);
bool is_punct(uint32_t cp);
bool is_symbol(uint32_t cp);
bool is_whitespace(uint32_t cp);

} // namespace unicode
} // namespace lsxcommon
```

- [ ] **Step 6: Write `unicode.cpp`**

```cpp
#include "lsxcommon/unicode.h"
#include "unicode_tables.inc"

namespace lsxcommon {
namespace unicode {
namespace {

bool in_ranges(uint32_t cp, const CodeRange *ranges, size_t count)
{
    size_t lo = 0, hi = count;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (ranges[mid].last < cp) lo = mid + 1;
        else if (ranges[mid].first > cp) hi = mid;
        else return true;
    }
    return false;
}

} // namespace

bool is_letter(uint32_t cp)     { return in_ranges(cp, kLetterRanges, kLetterRangesCount); }
bool is_mark(uint32_t cp)       { return in_ranges(cp, kMarkRanges, kMarkRangesCount); }
bool is_number(uint32_t cp)     { return in_ranges(cp, kNumberRanges, kNumberRangesCount); }
bool is_punct(uint32_t cp)      { return in_ranges(cp, kPunctRanges, kPunctRangesCount); }
bool is_symbol(uint32_t cp)     { return in_ranges(cp, kSymbolRanges, kSymbolRangesCount); }
bool is_whitespace(uint32_t cp) { return in_ranges(cp, kWhitespaceRanges, kWhitespaceRangesCount); }

CharClass classify(uint32_t cp)
{
    if (is_letter(cp))   return CharClass::Letter;
    if (is_mark(cp))     return CharClass::Mark;
    if (is_number(cp))   return CharClass::Number;
    if (is_punct(cp))    return CharClass::Punct;
    if (is_symbol(cp))   return CharClass::Symbol;
    if (is_whitespace(cp)) return CharClass::Whitespace;
    return CharClass::Other;
}

} // namespace unicode
} // namespace lsxcommon
```

- [ ] **Step 7: Add sources to CMake and register the test target**

Edit `libs/lsxcommon/CMakeLists.txt` — add `src/unicode.cpp` to the `lsxcommon` STATIC list (keep alphabetical):

```cmake
add_library(lsxcommon STATIC
    src/converter.cpp
    src/core.cpp
    src/common.cpp
    src/loader.cpp
    src/schema.cpp
    src/tokenizer.cpp
    src/unicode.cpp
)
```

Append to `tests/CMakeLists.txt`:

```cmake
# ── Zero-alloc tokenizer test ─────────────────────────────────────────
add_executable(test_tokenizer test_tokenizer.cpp)
target_compile_definitions(test_tokenizer PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_tokenizer PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_tokenizer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_tokenizer>"
    )
    add_custom_command(TARGET test_tokenizer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_tokenizer>"
    )
endif()

add_test(NAME test_tokenizer COMMAND test_tokenizer)
```

- [ ] **Step 8: Run test to verify it passes**

Run: `$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"; cmake --build build --config Debug --target test_tokenizer && & build/bin/Debug/test_tokenizer.exe "[tokenizer][unicode]"`
Expected: PASS — all 14 spot checks pass.

- [ ] **Step 9: Commit**

```bash
git add tools/generate_unicode_tables.ps1 libs/lsxcommon/include/lsxcommon/unicode.h libs/lsxcommon/src/unicode.cpp libs/lsxcommon/src/unicode_tables.inc libs/lsxcommon/CMakeLists.txt tests/CMakeLists.txt tests/test_tokenizer.cpp
git commit -m "feat: add Unicode class tables (UCD 16.0.0) for tokenizer splitter"
```

---

### Task 2: GPT-2 byte↔Unicode table + tokenizer schemas

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/byte_unicode.h`
- Modify: `libs/lsxcommon/include/lsxcommon/schema.h`
- Modify: `libs/lsxcommon/src/schema.cpp`
- Test: `tests/test_tokenizer.cpp` (schema section)

- [ ] **Step 1: Write the failing schema round-trip test**

Append to `tests/test_tokenizer.cpp`:

```cpp
#include <lsxcommon/schema.h>
#include <arrow/api.h>
#include <cstring>
#include <filesystem>

namespace fs = std::filesystem;

TEST_CASE("TokenizerVocabSchema round-trip", "[tokenizer][schema]")
{
    std::vector<lsxcommon::VocabRow> rows;
    rows.push_back({0, {'<', '|', 'b', 'o', 's', '|', '>'}, 1});
    rows.push_back({1, {}, 1});
    rows.push_back({2, {}, 1});
    for (int b = 3; b < 259; ++b)
        rows.push_back({b, {static_cast<uint8_t>(b - 3)}, 0});

    lsxcommon::TokenizerVocabMeta meta;
    meta.bos_id = 0; meta.eos_id = 1; meta.pad_id = 2; meta.vocab_size = 259;
    for (int i = 0; i < 256; ++i) meta.byte_to_token_id[i] = i + 3;

    auto batch_res = lsxcommon::TokenizerVocabSchema::MakeBatch(rows, meta);
    REQUIRE(batch_res.ok());
    auto batch = *batch_res;
    REQUIRE(batch->ValidateFull().ok());

    auto unpack_res = lsxcommon::TokenizerVocabSchema::UnpackBatch(batch);
    REQUIRE(unpack_res.ok());
    auto &out = *unpack_res;
    REQUIRE(out.size() == 259);
    CHECK(out[0].is_special == 1);
    CHECK(out[35].token_id == 35);
    CHECK(out[35].is_special == 0);
    REQUIRE(out[35].token_bytes.size() == 1);
    CHECK(out[35].token_bytes[0] == 0x20);

    auto path = fs::temp_directory_path() / "logestix_tok_vocab_test.arrow";
    REQUIRE(lsxcommon::WriteRecordBatchToFile(batch, path).ok());
    auto read_res = lsxcommon::ReadRecordBatchFromFile(path);
    REQUIRE(read_res.ok());
    auto meta2 = (*read_res)->schema()->metadata();
    REQUIRE(meta2 != nullptr);
    auto bos_res = meta2->Get("bos_id");
    REQUIRE(bos_res.ok());
    CHECK(*bos_res == "0");
    auto btt_res = meta2->Get("byte_to_token_id");
    REQUIRE(btt_res.ok());
    CHECK(btt_res->rfind("3,", 0) == 0);
    fs::remove(path);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"; cmake --build build --config Debug --target test_tokenizer 2>&1 | Select-Object -Last 5`
Expected: FAIL — `VocabRow` / `TokenizerVocabSchema` not declared.

- [ ] **Step 3: Write `byte_unicode.h`**

```cpp
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace lsxcommon {
namespace byte_unicode {

// GPT-2 byte→Unicode table (standard construction algorithm):
// bytes 33..126, 161..172, 174..255 self-map; the remaining 68 bytes map to
// U+0100..U+0143 in order of first appearance.
inline const std::array<char32_t, 256> &gpt2_byte_table()
{
    static const std::array<char32_t, 256> t = [] {
        std::array<char32_t, 256> m{};
        for (int i = 33; i <= 126; ++i)  m[static_cast<size_t>(i)] = static_cast<char32_t>(i);
        for (int i = 161; i <= 172; ++i) m[static_cast<size_t>(i)] = static_cast<char32_t>(i);
        for (int i = 174; i <= 255; ++i) m[static_cast<size_t>(i)] = static_cast<char32_t>(i);
        int n = 0;
        for (int b = 0; b < 256; ++b)
            if (m[static_cast<size_t>(b)] == 0)
                m[static_cast<size_t>(b)] = static_cast<char32_t>(0x100 + n++);
        return m;
    }();
    return t;
}

inline char32_t byte_to_unicode(int b)
{
    return gpt2_byte_table()[static_cast<size_t>(b) & 0xFF];
}

// Returns 0..255, or -1 if `cp` is not a byte-level char.
inline int unicode_to_byte(char32_t cp)
{
    const auto &t = gpt2_byte_table();
    for (int b = 0; b < 256; ++b)
        if (t[static_cast<size_t>(b)] == cp) return b;
    return -1;
}

// Encode raw bytes into the byte-unicode key string (UTF-8).
inline std::string bytes_to_key(const uint8_t *data, size_t n)
{
    std::string s;
    s.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        char32_t c = byte_to_unicode(data[i]);
        if (c < 0x80) {
            s.push_back(static_cast<char>(c));
        } else if (c < 0x800) {
            s.push_back(static_cast<char>(0xC0 | (c >> 6)));
            s.push_back(static_cast<char>(0x80 | (c & 0x3F)));
        } else {
            s.push_back(static_cast<char>(0xE0 | (c >> 12)));
            s.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
            s.push_back(static_cast<char>(0x80 | (c & 0x3F)));
        }
    }
    return s;
}

// Decode a byte-unicode key string back to raw bytes. Returns false if any
// codepoint is unmappable (i.e. the key is a special token, not byte-level).
inline bool key_to_bytes(const std::string &key, std::vector<uint8_t> &out)
{
    out.clear();
    out.reserve(key.size());
    for (size_t i = 0; i < key.size();) {
        char32_t cp = 0;
        size_t len = 0;
        uint8_t b0 = static_cast<uint8_t>(key[i]);
        if (b0 < 0x80) { cp = b0; len = 1; }
        else if ((b0 & 0xE0) == 0xC0) { cp = b0 & 0x1F; len = 2; }
        else if ((b0 & 0xF0) == 0xE0) { cp = b0 & 0x0F; len = 3; }
        else { cp = b0 & 0x07; len = 4; }
        if (i + len > key.size()) return false;
        for (size_t j = 1; j < len; ++j)
            cp = (cp << 6) | (static_cast<uint8_t>(key[i + j]) & 0x3F);
        int b = unicode_to_byte(cp);
        if (b < 0) return false;
        out.push_back(static_cast<uint8_t>(b));
        i += len;
    }
    return true;
}

} // namespace byte_unicode
} // namespace lsxcommon
```

- [ ] **Step 4: Add schema declarations to `schema.h`**

Add `#include <array>` to the includes, then append before the closing `namespace lsxcommon`:

```cpp
struct VocabRow {
    int32_t token_id = 0;
    std::vector<uint8_t> token_bytes;
    int8_t is_special = 0;
};

struct TokenizerVocabMeta {
    int32_t bos_id = -1;
    int32_t eos_id = -1;
    int32_t pad_id = -1;
    int32_t vocab_size = 0;
    std::array<int32_t, 256> byte_to_token_id{};
};

class TokenizerVocabSchema {
public:
    static constexpr int kIdxTokenId = 0;
    static constexpr int kIdxTokenBytes = 1;
    static constexpr int kIdxIsSpecial = 2;

    static std::shared_ptr<arrow::Schema> Make(const TokenizerVocabMeta &meta);
    static arrow::Result<std::shared_ptr<arrow::RecordBatch>>
    MakeBatch(const std::vector<VocabRow> &rows, const TokenizerVocabMeta &meta);
    static arrow::Result<std::vector<VocabRow>>
    UnpackBatch(const std::shared_ptr<arrow::RecordBatch> &batch);
};

struct MergeRow {
    int32_t merge_left = 0;
    int32_t merge_right = 0;
    int32_t merge_rank = 0;
    int32_t merge_token_id = 0;
};

class TokenizerMergesSchema {
public:
    static constexpr int kIdxMergeLeft = 0;
    static constexpr int kIdxMergeRight = 1;
    static constexpr int kIdxMergeRank = 2;
    static constexpr int kIdxMergeTokenId = 3;

    static std::shared_ptr<arrow::Schema> Make(int32_t merge_count);
    static arrow::Result<std::shared_ptr<arrow::RecordBatch>>
    MakeBatch(const std::vector<MergeRow> &rows, int32_t merge_count);
    static arrow::Result<std::vector<MergeRow>>
    UnpackBatch(const std::shared_ptr<arrow::RecordBatch> &batch);
};
```

- [ ] **Step 5: Implement in `schema.cpp`**

Add `#include <array>` and `#include <cstdlib>` to the includes, then append before the "File I/O helpers" comment block:

```cpp
// ── Tokenizer schemas ───────────────────────────────────────────────────────

namespace {

std::string SerializeByteToTokenId(const std::array<int32_t, 256> &b)
{
    std::string s;
    s.reserve(256 * 8);
    for (int i = 0; i < 256; ++i) {
        if (i) s.push_back(',');
        s += std::to_string(b[static_cast<size_t>(i)]);
    }
    return s;
}

std::array<int32_t, 256> DeserializeByteToTokenId(const std::string &s)
{
    std::array<int32_t, 256> out{};
    const char *p = s.c_str();
    for (int i = 0; i < 256; ++i) {
        char *end = nullptr;
        long v = std::strtol(p, &end, 10);
        out[static_cast<size_t>(i)] = static_cast<int32_t>(v);
        p = end;
        if (i < 255 && *p == ',') ++p;
    }
    return out;
}

} // namespace

std::shared_ptr<arrow::Schema> TokenizerVocabSchema::Make(const TokenizerVocabMeta &meta)
{
    auto kv = arrow::KeyValueMetadata::Make({
        "bos_id", "eos_id", "pad_id", "vocab_size", "byte_to_token_id"
    }, {
        std::to_string(meta.bos_id),
        std::to_string(meta.eos_id),
        std::to_string(meta.pad_id),
        std::to_string(meta.vocab_size),
        SerializeByteToTokenId(meta.byte_to_token_id)
    });
    return arrow::schema({
        arrow::field("token_id", arrow::int32()),
        arrow::field("token_bytes", arrow::large_binary()),
        arrow::field("is_special", arrow::int8()),
    }, kv);
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>>
TokenizerVocabSchema::MakeBatch(const std::vector<VocabRow> &rows,
                                const TokenizerVocabMeta &meta)
{
    auto schema = Make(meta);
    const int64_t n = static_cast<int64_t>(rows.size());

    arrow::Int32Builder id_builder;
    arrow::LargeBinaryBuilder bytes_builder;
    arrow::Int8Builder spec_builder;
    ARROW_RETURN_NOT_OK(id_builder.Reserve(n));
    ARROW_RETURN_NOT_OK(bytes_builder.Reserve(n));
    ARROW_RETURN_NOT_OK(spec_builder.Reserve(n));

    for (const auto &r : rows) {
        ARROW_RETURN_NOT_OK(id_builder.Append(r.token_id));
        ARROW_RETURN_NOT_OK(bytes_builder.Append(
            r.token_bytes.data(), static_cast<int64_t>(r.token_bytes.size())));
        ARROW_RETURN_NOT_OK(spec_builder.Append(r.is_special));
    }

    ARROW_ASSIGN_OR_RAISE(auto id_arr, id_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto bytes_arr, bytes_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto spec_arr, spec_builder.Finish());

    auto batch = arrow::RecordBatch::Make(schema, n, {id_arr, bytes_arr, spec_arr});
    diag::ValidateBatchBuffers(*batch, "TokenizerVocabSchema::MakeBatch result");
    return batch;
}

arrow::Result<std::vector<VocabRow>>
TokenizerVocabSchema::UnpackBatch(const std::shared_ptr<arrow::RecordBatch> &batch)
{
    if (!batch || batch->num_columns() != 3)
        return arrow::Status::Invalid("Expected 3-column tokenizer vocab RecordBatch");

    auto ids = std::dynamic_pointer_cast<arrow::Int32Array>(batch->column(kIdxTokenId));
    auto bytes = std::dynamic_pointer_cast<arrow::LargeBinaryArray>(batch->column(kIdxTokenBytes));
    auto specs = std::dynamic_pointer_cast<arrow::Int8Array>(batch->column(kIdxIsSpecial));
    if (!ids || !bytes || !specs)
        return arrow::Status::TypeError("tokenizer vocab column type mismatch");

    const int64_t n = batch->num_rows();
    std::vector<VocabRow> rows;
    rows.resize(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        auto &r = rows[static_cast<size_t>(i)];
        r.token_id = ids->Value(i);
        r.is_special = specs->Value(i);
        auto v = bytes->GetView(i);
        r.token_bytes.assign(v.begin(), v.end());
    }
    return rows;
}

std::shared_ptr<arrow::Schema> TokenizerMergesSchema::Make(int32_t merge_count)
{
    auto kv = arrow::KeyValueMetadata::Make({"merge_count"}, {std::to_string(merge_count)});
    return arrow::schema({
        arrow::field("merge_left", arrow::int32()),
        arrow::field("merge_right", arrow::int32()),
        arrow::field("merge_rank", arrow::int32()),
        arrow::field("merge_token_id", arrow::int32()),
    }, kv);
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>>
TokenizerMergesSchema::MakeBatch(const std::vector<MergeRow> &rows, int32_t merge_count)
{
    auto schema = Make(merge_count);
    const int64_t n = static_cast<int64_t>(rows.size());

    arrow::Int32Builder left_builder, right_builder, rank_builder, token_builder;
    ARROW_RETURN_NOT_OK(left_builder.Reserve(n));
    ARROW_RETURN_NOT_OK(right_builder.Reserve(n));
    ARROW_RETURN_NOT_OK(rank_builder.Reserve(n));
    ARROW_RETURN_NOT_OK(token_builder.Reserve(n));

    for (const auto &r : rows) {
        ARROW_RETURN_NOT_OK(left_builder.Append(r.merge_left));
        ARROW_RETURN_NOT_OK(right_builder.Append(r.merge_right));
        ARROW_RETURN_NOT_OK(rank_builder.Append(r.merge_rank));
        ARROW_RETURN_NOT_OK(token_builder.Append(r.merge_token_id));
    }

    ARROW_ASSIGN_OR_RAISE(auto left_arr, left_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto right_arr, right_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto rank_arr, rank_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto token_arr, token_builder.Finish());

    auto batch = arrow::RecordBatch::Make(schema, n,
        {left_arr, right_arr, rank_arr, token_arr});
    diag::ValidateBatchBuffers(*batch, "TokenizerMergesSchema::MakeBatch result");
    return batch;
}

arrow::Result<std::vector<MergeRow>>
TokenizerMergesSchema::UnpackBatch(const std::shared_ptr<arrow::RecordBatch> &batch)
{
    if (!batch || batch->num_columns() != 4)
        return arrow::Status::Invalid("Expected 4-column tokenizer merges RecordBatch");

    auto left = std::dynamic_pointer_cast<arrow::Int32Array>(batch->column(kIdxMergeLeft));
    auto right = std::dynamic_pointer_cast<arrow::Int32Array>(batch->column(kIdxMergeRight));
    auto rank = std::dynamic_pointer_cast<arrow::Int32Array>(batch->column(kIdxMergeRank));
    auto token = std::dynamic_pointer_cast<arrow::Int32Array>(batch->column(kIdxMergeTokenId));
    if (!left || !right || !rank || !token)
        return arrow::Status::TypeError("tokenizer merges column type mismatch");

    const int64_t n = batch->num_rows();
    std::vector<MergeRow> rows;
    rows.resize(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        auto &r = rows[static_cast<size_t>(i)];
        r.merge_left = left->Value(i);
        r.merge_right = right->Value(i);
        r.merge_rank = rank->Value(i);
        r.merge_token_id = token->Value(i);
    }
    return rows;
}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"; cmake --build build --config Debug --target test_tokenizer && & build/bin/Debug/test_tokenizer.exe "[tokenizer][schema]"`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/byte_unicode.h libs/lsxcommon/include/lsxcommon/schema.h libs/lsxcommon/src/schema.cpp tests/test_tokenizer.cpp
git commit -m "feat: add GPT-2 byte table and tokenizer Arrow schemas"
```

---

### Task 3: Converter — `BuildTokenizerArrowFiles`

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/converter.h`
- Create: `libs/lsxcommon/src/tokenizer_convert.cpp`
- Move: `synthetic_tok.json` → `tests/data/synthetic_tok.json`
- Modify: `libs/lsxcommon/CMakeLists.txt`
- Test: `tests/test_tokenizer.cpp` (convert section)

- [ ] **Step 1: Move the synthetic fixture**

Run: `New-Item -ItemType Directory -Force -Path tests\data | Out-Null; Move-Item -Force synthetic_tok.json tests\data\synthetic_tok.json`

- [ ] **Step 2: Write the failing converter test**

Append to `tests/test_tokenizer.cpp`:

```cpp
#include <lsxcommon/converter.h>

TEST_CASE("BuildTokenizerArrowFiles from synthetic fixture", "[tokenizer][convert]")
{
    auto pinned = fs::temp_directory_path() / "logestix_tok_synth";
    fs::remove_all(pinned);
    fs::create_directories(pinned);

    auto st = lsxcommon::BuildTokenizerArrowFiles(
        TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json", pinned);
    REQUIRE(st.ok());
    REQUIRE(fs::exists(pinned / "tokenizer_vocab.arrow"));
    REQUIRE(fs::exists(pinned / "tokenizer_merges.arrow"));

    auto v = lsxcommon::ReadRecordBatchFromFile(pinned / "tokenizer_vocab.arrow");
    REQUIRE(v.ok());
    CHECK((*v)->num_rows() == 268);

    auto m = lsxcommon::ReadRecordBatchFromFile(pinned / "tokenizer_merges.arrow");
    REQUIRE(m.ok());
    CHECK((*m)->num_rows() == 9);

    auto meta = (*v)->schema()->metadata();
    REQUIRE(meta != nullptr);
    REQUIRE(meta->Get("vocab_size").ok());
    CHECK(*meta->Get("vocab_size") == "268");
    REQUIRE(meta->Get("bos_id").ok());
    CHECK(*meta->Get("bos_id") == "0");
    REQUIRE(meta->Get("byte_to_token_id").ok());
    CHECK(meta->Get("byte_to_token_id")->rfind("3,", 0) == 0);

    auto merges = lsxcommon::TokenizerMergesSchema::UnpackBatch(*m);
    REQUIRE(merges.ok());
    REQUIRE(merges->size() == 9);
    for (size_t i = 1; i < merges->size(); ++i) {
        auto &a = (*merges)[i - 1];
        auto &b = (*merges)[i];
        CHECK((a.merge_left < b.merge_left ||
               (a.merge_left == b.merge_left && a.merge_right < b.merge_right)));
    }
    // "h e" (ids 107/104) -> "he" = 259, ranks follow input order
    bool found_he = false;
    for (auto &r : *merges) {
        if (r.merge_left == 107 && r.merge_right == 104) {
            CHECK(r.merge_token_id == 259);
            CHECK(r.merge_rank == 0);
            found_he = true;
        }
    }
    CHECK(found_he);
    fs::remove_all(pinned);
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"; cmake --build build --config Debug --target test_tokenizer 2>&1 | Select-Object -Last 5`
Expected: FAIL — `BuildTokenizerArrowFiles` not declared.

- [ ] **Step 4: Declare the converter function**

Append to `libs/lsxcommon/include/lsxcommon/converter.h` before the closing namespace (the file already includes `<arrow/api.h>`, `<filesystem>`, `<string>`):

```cpp
arrow::Status BuildTokenizerArrowFiles(
    const std::string &tokenizer_json_path,
    const std::filesystem::path &pinned_dir);
```

- [ ] **Step 5: Write `tokenizer_convert.cpp`**

```cpp
#include "lsxcommon/converter.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/byte_unicode.h"
#include <arrow/api.h>
#include <simdjson.h>
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace lsxcommon {

arrow::Status BuildTokenizerArrowFiles(const std::string &tokenizer_json_path,
                                       const std::filesystem::path &pinned_dir)
{
    std::ifstream f(tokenizer_json_path, std::ios::binary | std::ios::ate);
    if (!f) return arrow::Status::IOError("cannot open ", tokenizer_json_path);
    const auto sz = f.tellg();
    f.seekg(0);
    std::string json(static_cast<size_t>(sz), '\0');
    f.read(json.data(), sz);

    simdjson::padded_string pad(json.data(), json.size());
    simdjson::ondemand::parser parser;
    simdjson::ondemand::document doc;
    if (parser.iterate(pad).get(doc) != simdjson::SUCCESS)
        return arrow::Status::Invalid("tokenizer: JSON parse failed");
    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS)
        return arrow::Status::Invalid("tokenizer: root is not an object");

    simdjson::ondemand::object model_obj;
    if (root["model"].get_object().get(model_obj) != simdjson::SUCCESS)
        return arrow::Status::Invalid("tokenizer: no model object");

    std::unordered_map<std::string, int32_t> key_to_id;
    std::unordered_map<int32_t, std::string> id_to_key;
    {
        simdjson::ondemand::object vocab_obj;
        if (model_obj["vocab"].get_object().get(vocab_obj) != simdjson::SUCCESS)
            return arrow::Status::Invalid("tokenizer: no model.vocab object");
        for (auto entry : vocab_obj) {
            std::string_view key;
            int64_t id;
            if (entry.unescaped_key().get(key) != simdjson::SUCCESS) continue;
            if (entry.value().get_int64().get(id) != simdjson::SUCCESS) continue;
            auto k = std::string(key);
            auto i = static_cast<int32_t>(id);
            key_to_id[k] = i;
            id_to_key[i] = std::move(k);
        }
    }

    std::unordered_set<int32_t> special_ids;
    std::unordered_map<int32_t, std::string> added_content;
    bool has_added = false;
    {
        simdjson::ondemand::array added_arr;
        if (root["added_tokens"].get_array().get(added_arr) == simdjson::SUCCESS) {
            has_added = true;
            for (auto at : added_arr) {
                simdjson::ondemand::object at_obj;
                if (at.get_object().get(at_obj) != simdjson::SUCCESS) continue;
                std::string_view content;
                int64_t id = -1;
                bool special = false;
                at_obj["content"].get_string().get(content);
                at_obj["id"].get_int64().get(id);
                if (at_obj["special"].get_bool().get(special) != simdjson::SUCCESS) special = false;
                if (id < 0) continue;
                auto i = static_cast<int32_t>(id);
                auto k = std::string(content);
                added_content[i] = k;
                key_to_id[k] = i;
                id_to_key[i] = k;
                if (special) special_ids.insert(i);
            }
        }
    }

    // Fallback specials for byte-level models without added_tokens.
    if (!has_added) {
        special_ids.insert(0);
        special_ids.insert(1);
        special_ids.insert(2);
    }

    int32_t bos_id = 0, eos_id = 1, pad_id = 2;
    {
        auto find_special = [&](const char *needle) -> int32_t {
            for (auto &[id, k] : added_content)
                if (k.find(needle) != std::string::npos) return id;
            return -1;
        };
        int32_t b = find_special("begin");
        if (b < 0) b = find_special("sentence");
        if (b >= 0) bos_id = b;
        int32_t e = find_special("end");
        if (e >= 0) eos_id = e;
        int32_t p = find_special("pad");
        if (p >= 0) pad_id = p;
    }

    // byte_to_token_id via vocab lookup of the GPT-2 byte table.
    std::array<int32_t, 256> byte_to_token_id{};
    for (int b = 0; b < 256; ++b) {
        uint8_t b8 = static_cast<uint8_t>(b);
        std::string key = byte_unicode::bytes_to_key(&b8, 1);
        auto it = key_to_id.find(key);
        if (it == key_to_id.end())
            return arrow::Status::Invalid("tokenizer: byte token missing in vocab: '", key, "'");
        byte_to_token_id[static_cast<size_t>(b)] = it->second;
    }

    // Merge table via vocab lookup only.
    std::vector<MergeRow> merges;
    {
        simdjson::ondemand::array merges_arr;
        if (model_obj["merges"].get_array().get(merges_arr) == simdjson::SUCCESS) {
            int64_t rank = 0;
            for (auto m : merges_arr) {
                std::string_view sv;
                if (m.get_string().get(sv) != simdjson::SUCCESS) continue;
                auto s = std::string(sv);
                auto sp = s.find(' ');
                if (sp == std::string::npos) continue;
                std::string A = s.substr(0, sp);
                std::string B = s.substr(sp + 1);

                auto itL = key_to_id.find(A);
                auto itR = key_to_id.find(B);
                if (itL == key_to_id.end() || itR == key_to_id.end())
                    return arrow::Status::Invalid("tokenizer: merge operand missing in vocab: '", s, "'");

                std::vector<uint8_t> bytesA, bytesB;
                if (!byte_unicode::key_to_bytes(A, bytesA) ||
                    !byte_unicode::key_to_bytes(B, bytesB))
                    return arrow::Status::Invalid("tokenizer: merge operand not byte-level: '", s, "'");

                std::vector<uint8_t> merged_bytes = bytesA;
                merged_bytes.insert(merged_bytes.end(), bytesB.begin(), bytesB.end());
                std::string merged_key =
                    byte_unicode::bytes_to_key(merged_bytes.data(), merged_bytes.size());
                auto itM = key_to_id.find(merged_key);
                if (itM == key_to_id.end())
                    return arrow::Status::Invalid("tokenizer: merge product missing in vocab: '",
                                                  merged_key, "'");

                MergeRow r;
                r.merge_left = itL->second;
                r.merge_right = itR->second;
                r.merge_rank = static_cast<int32_t>(rank++);
                r.merge_token_id = itM->second;
                merges.push_back(r);
            }
        }
    }
    std::sort(merges.begin(), merges.end(), [](const MergeRow &a, const MergeRow &b) {
        if (a.merge_left != b.merge_left) return a.merge_left < b.merge_left;
        return a.merge_right < b.merge_right;
    });

    // Vocab rows indexed by token_id.
    int32_t vocab_size = 0;
    for (auto &[id, k] : id_to_key) vocab_size = std::max(vocab_size, id + 1);

    std::vector<VocabRow> vocab_rows;
    vocab_rows.resize(static_cast<size_t>(vocab_size));
    for (int32_t id = 0; id < vocab_size; ++id) {
        auto it = id_to_key.find(id);
        if (it == id_to_key.end())
            return arrow::Status::Invalid("tokenizer: missing vocab id ", std::to_string(id));
        auto &row = vocab_rows[static_cast<size_t>(id)];
        row.token_id = id;

        std::vector<uint8_t> bytes;
        bool special = special_ids.count(id) > 0;
        if (!byte_unicode::key_to_bytes(it->second, bytes)) {
            // Unmappable chars are only legal for special tokens.
            if (!special)
                return arrow::Status::Invalid("tokenizer: non-special token not byte-level: '",
                                              it->second, "'");
            row.is_special = 1;
            row.token_bytes.assign(
                reinterpret_cast<const uint8_t *>(it->second.data()),
                reinterpret_cast<const uint8_t *>(it->second.data()) + it->second.size());
        } else {
            row.is_special = special ? 1 : 0;
            row.token_bytes = std::move(bytes);
        }
    }

    TokenizerVocabMeta meta;
    meta.bos_id = bos_id;
    meta.eos_id = eos_id;
    meta.pad_id = pad_id;
    meta.vocab_size = vocab_size;
    meta.byte_to_token_id = byte_to_token_id;

    auto vbatch = TokenizerVocabSchema::MakeBatch(vocab_rows, meta);
    if (!vbatch.ok()) return vbatch.status();
    auto mbatch = TokenizerMergesSchema::MakeBatch(merges,
        static_cast<int32_t>(merges.size()));
    if (!mbatch.ok()) return mbatch.status();

    ARROW_RETURN_NOT_OK(WriteRecordBatchToFile(*vbatch, pinned_dir / "tokenizer_vocab.arrow"));
    ARROW_RETURN_NOT_OK(WriteRecordBatchToFile(*mbatch, pinned_dir / "tokenizer_merges.arrow"));
    return arrow::Status::OK();
}

} // namespace lsxcommon
```

- [ ] **Step 6: Add `src/tokenizer_convert.cpp` to CMake and run the test**

Edit `libs/lsxcommon/CMakeLists.txt` — add `src/tokenizer_convert.cpp` to the `lsxcommon` STATIC list (keep alphabetical):

```cmake
add_library(lsxcommon STATIC
    src/converter.cpp
    src/core.cpp
    src/common.cpp
    src/loader.cpp
    src/schema.cpp
    src/tokenizer.cpp
    src/tokenizer_convert.cpp
    src/unicode.cpp
)
```

Run: `$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"; cmake --build build --config Debug --target test_tokenizer && & build/bin/Debug/test_tokenizer.exe "[tokenizer][convert]"`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tests/data/synthetic_tok.json libs/lsxcommon/include/lsxcommon/converter.h libs/lsxcommon/src/tokenizer_convert.cpp libs/lsxcommon/CMakeLists.txt tests/test_tokenizer.cpp
git commit -m "feat: add tokenizer.json -> Arrow converter (lsxcommon BuildTokenizerArrowFiles)"
```

---

### Task 4: Host loader — `ArrowTokenizerDb` + `load_tokenizer_metadata`

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/tokenizer.h` (keep `BpeTokenizer`, append Arrow API)
- Modify: `libs/lsxcommon/src/tokenizer.cpp` (keep legacy impl, append loader + stubs)
- Test: `tests/test_tokenizer.cpp` (load section)

`BpeTokenizer` must stay until Task 8 swaps `inference_pipeline.cu`; removing it earlier would break the `lsxcommon_cu` build.

- [ ] **Step 1: Write the failing loader test**

Append to `tests/test_tokenizer.cpp`:

```cpp
#include <lsxcommon/tokenizer.h>
#include <arrow/io/file.h>

static fs::path BuildSyntheticDb(lsxcommon::ArrowTokenizerDb &db)
{
    auto pinned = fs::temp_directory_path() / "logestix_tok_synth_db";
    fs::remove_all(pinned);
    fs::create_directories(pinned);
    auto st = lsxcommon::BuildTokenizerArrowFiles(
        TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json", pinned);
    REQUIRE(st.ok());
    auto v = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_vocab.arrow").string(), arrow::io::FileMode::READ);
    REQUIRE(v.ok());
    auto m = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_merges.arrow").string(), arrow::io::FileMode::READ);
    REQUIRE(m.ok());
    auto lst = lsxcommon::load_tokenizer_metadata(*v, *m, db);
    REQUIRE(lst.ok());
    return pinned;
}

TEST_CASE("load_tokenizer_metadata from synthetic files", "[tokenizer][load]")
{
    lsxcommon::ArrowTokenizerDb db;
    auto pinned = BuildSyntheticDb(db);

    CHECK(db.valid());
    CHECK(db.vocab_size == 268);
    CHECK(db.merge_count == 9);
    CHECK(db.bos_id == 0);
    CHECK(db.eos_id == 1);
    CHECK(db.pad_id == 2);

    CHECK(db.byte_to_token_id[0x20] == 35);
    CHECK(db.byte_to_token_id[0x78] == 123);
    CHECK(db.byte_to_token_id[0x68] == 107);

    CHECK(db.token_offsets != nullptr);
    CHECK(db.token_bytes != nullptr);
    CHECK(db.token_offsets[35 + 1] - db.token_offsets[35] == 1);
    CHECK(db.token_bytes[db.token_offsets[35]] == 0x20);

    CHECK(db.is_special[0] == 1);
    CHECK(db.is_special[1] == 1);
    CHECK(db.is_special[2] == 1);
    CHECK(db.is_special[35] == 0);

    CHECK(db.merge_left != nullptr);
    CHECK(db.merge_right != nullptr);
    CHECK(db.merge_rank != nullptr);
    CHECK(db.merge_token_id != nullptr);
    // merges sorted by (left,right): row 0 is "\u0120 w" = (35,122), rank 4
    // (fixture merge #5); "h e" = (107,104) is rank 0 (fixture merge #1).
    // NOTE: 'w' = id 122 (byte->token = byte+3); the plan draft said 124.
    CHECK(db.merge_left[0] == 35);
    CHECK(db.merge_right[0] == 122);
    CHECK(db.merge_rank[0] == 4);
    CHECK(db.merge_token_id[0] == 263);
    int he_idx = -1;
    for (int i = 0; i < db.merge_count; ++i) {
        if (db.merge_left[i] == 107 && db.merge_right[i] == 104) { he_idx = i; break; }
    }
    REQUIRE(he_idx >= 0);
    CHECK(db.merge_rank[he_idx] == 0);
    CHECK(db.merge_token_id[he_idx] == 259);

    lsxcommon::ArrowTokenizerDb bad;
    CHECK_FALSE(bad.valid());

    db = lsxcommon::ArrowTokenizerDb{};   // release mmap handles before deleting files
    fs::remove_all(pinned);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"; cmake --build build --config Debug --target test_tokenizer 2>&1 | Select-Object -Last 5`
Expected: FAIL — `ArrowTokenizerDb` / `load_tokenizer_metadata` not declared.

- [ ] **Step 3: Append the Arrow API to `tokenizer.h`**

Add to the includes at the top of `libs/lsxcommon/include/lsxcommon/tokenizer.h`:

```cpp
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <cstddef>
```

Then append after the `BpeTokenizer` class (before the closing `namespace lsxcommon`):

```cpp
constexpr size_t kMaxWordBytes = 512;

struct ArrowTokenizerDb {
    std::shared_ptr<arrow::io::MemoryMappedFile> vocab_mmap;
    std::shared_ptr<arrow::io::MemoryMappedFile> merges_mmap;
    const uint8_t *vocab_file_base = nullptr;   // ptr − file_base offset math
    int64_t vocab_file_size = 0;

    const int64_t *token_offsets = nullptr;     // LargeBinary offsets, n = vocab_size + 1
    const uint8_t *token_bytes = nullptr;       // value buffer base
    const uint8_t *is_special = nullptr;        // Int8 column
    int32_t vocab_size = 0;

    const int32_t *merge_left = nullptr;
    const int32_t *merge_right = nullptr;
    const int32_t *merge_rank = nullptr;
    const int32_t *merge_token_id = nullptr;
    int32_t merge_count = 0;

    int32_t bos_id = -1, eos_id = -1, pad_id = -1;
    int32_t byte_to_token_id[256];
    std::string error_msg;

    bool valid() const;
};

arrow::Status load_tokenizer_metadata(
    const std::shared_ptr<arrow::io::MemoryMappedFile> &vocab_mmap,
    const std::shared_ptr<arrow::io::MemoryMappedFile> &merges_mmap,
    ArrowTokenizerDb &db);

void set_active_tokenizer_db(const ArrowTokenizerDb *db);

// Returns token count on success. Negative on error:
//   -1 invalid db / no active db, -2 output capacity too small,
//   -3 a word exceeds kMaxWordBytes. *max_out_len is capacity in, count out.
int32_t arrow_tokenizer_encode(const char *text, size_t text_len,
                               int32_t *out_tokens, size_t *max_out_len);

int32_t arrow_tokenizer_encode(const ArrowTokenizerDb &db, const char *text,
                               size_t text_len, int32_t *out_tokens,
                               size_t *max_out_len);

// Returns 0 on success (specials skipped, raw bytes appended),
// -1 invalid db, -2 output capacity too small. *max_out_len is in/out.
int32_t arrow_tokenizer_decode(const ArrowTokenizerDb &db, const int32_t *ids,
                               size_t n_ids, char *out_text,
                               size_t *max_out_len);
```

- [ ] **Step 4: Append the loader + stubs to `tokenizer.cpp`**

Add to the includes at the top of `libs/lsxcommon/src/tokenizer.cpp`:

```cpp
#include "lsxcommon/schema.h"
#include <arrow/ipc/reader.h>
#include <cstdlib>
#include <limits>
```

Then append after the existing `BpeTokenizer::Decode` implementation (before the closing `namespace lsxcommon`):

```cpp
bool ArrowTokenizerDb::valid() const
{
    return vocab_mmap && merges_mmap && token_offsets && token_bytes && is_special &&
           vocab_size > 0 && merge_left && merge_right && merge_rank &&
           merge_token_id && merge_count > 0;
}

namespace {

const ArrowTokenizerDb *g_active_db = nullptr;

} // namespace

void set_active_tokenizer_db(const ArrowTokenizerDb *db)
{
    g_active_db = db;
}

arrow::Status load_tokenizer_metadata(
    const std::shared_ptr<arrow::io::MemoryMappedFile> &vocab_mmap,
    const std::shared_ptr<arrow::io::MemoryMappedFile> &merges_mmap,
    ArrowTokenizerDb &db)
{
    auto impl = [&]() -> arrow::Status {
        if (!vocab_mmap || !merges_mmap)
            return arrow::Status::Invalid("tokenizer: null MemoryMappedFile");
        db = ArrowTokenizerDb{};
        db.vocab_mmap = vocab_mmap;
        db.merges_mmap = merges_mmap;

        ARROW_ASSIGN_OR_RAISE(int64_t v_size, vocab_mmap->GetSize());
        db.vocab_file_size = v_size;
        ARROW_ASSIGN_OR_RAISE(auto base_buf, vocab_mmap->ReadAt(0, 1));
        db.vocab_file_base = base_buf->data();

        auto v_reader_res = arrow::ipc::RecordBatchFileReader::Open(vocab_mmap.get());
        if (!v_reader_res.ok()) return v_reader_res.status();
        if ((*v_reader_res)->num_record_batches() == 0)
            return arrow::Status::Invalid("tokenizer: vocab file has no batches");
        auto v_batch_res = (*v_reader_res)->ReadRecordBatch(0);
        if (!v_batch_res.ok()) return v_batch_res.status();
        auto v_batch = *v_batch_res;

        auto tids = std::dynamic_pointer_cast<arrow::Int32Array>(
            v_batch->column(TokenizerVocabSchema::kIdxTokenId));
        auto tbytes = std::dynamic_pointer_cast<arrow::LargeBinaryArray>(
            v_batch->column(TokenizerVocabSchema::kIdxTokenBytes));
        auto tspec = std::dynamic_pointer_cast<arrow::Int8Array>(
            v_batch->column(TokenizerVocabSchema::kIdxIsSpecial));
        if (!tids || !tbytes || !tspec)
            return arrow::Status::TypeError("tokenizer: vocab column type mismatch");

        const int64_t n = v_batch->num_rows();
        if (n <= 0 || n > INT32_MAX)
            return arrow::Status::Invalid("tokenizer: bad vocab row count");

        auto in_range = [&](const uint8_t *p, int64_t len) {
            return p >= db.vocab_file_base &&
                   p + len <= db.vocab_file_base + db.vocab_file_size;
        };

        auto offsets_buf = tbytes->data()->buffers[1];
        if (!offsets_buf || !in_range(offsets_buf->data(), offsets_buf->size()))
            return arrow::Status::Invalid("tokenizer: offsets buffer not backed by mmap");
        if (offsets_buf->size() != (n + 1) * static_cast<int64_t>(sizeof(int64_t)))
            return arrow::Status::Invalid("tokenizer: offsets length != vocab_size+1");
        auto values_buf = tbytes->value_data();
        if (!values_buf || !in_range(values_buf->data(), values_buf->size()))
            return arrow::Status::Invalid("tokenizer: values buffer not backed by mmap");
        auto spec_buf = tspec->data()->buffers[1];
        if (!spec_buf || !in_range(spec_buf->data(), spec_buf->size()))
            return arrow::Status::Invalid("tokenizer: is_special buffer not backed by mmap");

        db.vocab_size = static_cast<int32_t>(n);
        db.token_offsets = tbytes->raw_value_offsets();
        db.token_bytes = values_buf->data();
        db.is_special = reinterpret_cast<const uint8_t *>(tspec->raw_values());

        auto meta = v_batch->schema()->metadata();
        if (!meta) return arrow::Status::Invalid("tokenizer: vocab schema has no metadata");
        auto get_i32 = [&](const char *key, int32_t &out) -> arrow::Status {
            ARROW_ASSIGN_OR_RAISE(auto s, meta->Get(key));
            char *end = nullptr;
            long v = std::strtol(s.c_str(), &end, 10);
            if (end == s.c_str())
                return arrow::Status::Invalid("tokenizer: bad metadata ", key);
            out = static_cast<int32_t>(v);
            return arrow::Status::OK();
        };
        ARROW_RETURN_NOT_OK(get_i32("bos_id", db.bos_id));
        ARROW_RETURN_NOT_OK(get_i32("eos_id", db.eos_id));
        ARROW_RETURN_NOT_OK(get_i32("pad_id", db.pad_id));
        ARROW_RETURN_NOT_OK(get_i32("vocab_size", db.vocab_size));
        if (db.vocab_size != static_cast<int32_t>(n))
            return arrow::Status::Invalid("tokenizer: vocab_size metadata != row count");

        // byte_to_token_id (256 comma-separated ints)
        {
            ARROW_ASSIGN_OR_RAISE(auto btt, meta->Get("byte_to_token_id"));
            const char *p = btt.c_str();
            for (int i = 0; i < 256; ++i) {
                char *end = nullptr;
                long v = std::strtol(p, &end, 10);
                if (end == p)
                    return arrow::Status::Invalid("tokenizer: bad byte_to_token_id metadata");
                db.byte_to_token_id[i] = static_cast<int32_t>(v);
                p = end;
                if (i < 255 && *p == ',') ++p;
            }
        }

        // ---- merges file ----
        ARROW_ASSIGN_OR_RAISE(int64_t m_size, merges_mmap->GetSize());
        ARROW_ASSIGN_OR_RAISE(auto m_base_buf, merges_mmap->ReadAt(0, 1));
        const uint8_t *m_file_base = m_base_buf->data();
        const int64_t m_file_size = m_size;

        auto m_reader_res = arrow::ipc::RecordBatchFileReader::Open(merges_mmap.get());
        if (!m_reader_res.ok()) return m_reader_res.status();
        if ((*m_reader_res)->num_record_batches() == 0)
            return arrow::Status::Invalid("tokenizer: merges file has no batches");
        auto m_batch_res = (*m_reader_res)->ReadRecordBatch(0);
        if (!m_batch_res.ok()) return m_batch_res.status();
        auto m_batch = *m_batch_res;

        auto mleft = std::dynamic_pointer_cast<arrow::Int32Array>(
            m_batch->column(TokenizerMergesSchema::kIdxMergeLeft));
        auto mright = std::dynamic_pointer_cast<arrow::Int32Array>(
            m_batch->column(TokenizerMergesSchema::kIdxMergeRight));
        auto mrank = std::dynamic_pointer_cast<arrow::Int32Array>(
            m_batch->column(TokenizerMergesSchema::kIdxMergeRank));
        auto mtok = std::dynamic_pointer_cast<arrow::Int32Array>(
            m_batch->column(TokenizerMergesSchema::kIdxMergeTokenId));
        if (!mleft || !mright || !mrank || !mtok)
            return arrow::Status::TypeError("tokenizer: merges column type mismatch");

        const int64_t mn = m_batch->num_rows();
        if (mn <= 0 || mn > INT32_MAX)
            return arrow::Status::Invalid("tokenizer: bad merges row count");

        auto m_in_range = [&](const uint8_t *p, int64_t len) {
            return p >= m_file_base && p + len <= m_file_base + m_file_size;
        };
        for (int c = 0; c < 4; ++c) {
            auto buf = m_batch->column(c)->data()->buffers[1];
            if (!buf || !m_in_range(buf->data(), buf->size()))
                return arrow::Status::Invalid("tokenizer: merges column buffer not backed by mmap");
        }

        auto m_meta = m_batch->schema()->metadata();
        if (!m_meta) return arrow::Status::Invalid("tokenizer: merges schema has no metadata");
        {
            ARROW_ASSIGN_OR_RAISE(auto s, m_meta->Get("merge_count"));
            char *end = nullptr;
            long v = std::strtol(s.c_str(), &end, 10);
            if (end == s.c_str() || v != mn)
                return arrow::Status::Invalid("tokenizer: merge_count metadata != row count");
        }

        db.merge_count = static_cast<int32_t>(mn);
        db.merge_left = mleft->raw_values();
        db.merge_right = mright->raw_values();
        db.merge_rank = mrank->raw_values();
        db.merge_token_id = mtok->raw_values();

        return arrow::Status::OK();
    };
    auto st = impl();
    if (!st.ok()) db.error_msg = st.ToString();
    return st;
}

// Stubs — real implementations land in Tasks 6 and 7.
int32_t arrow_tokenizer_encode(const char *, size_t, int32_t *, size_t *)
{
    return -1;
}

int32_t arrow_tokenizer_encode(const ArrowTokenizerDb &, const char *,
                               size_t, int32_t *, size_t *)
{
    return -1;
}

int32_t arrow_tokenizer_decode(const ArrowTokenizerDb &, const int32_t *,
                               size_t, char *, size_t *)
{
    return -1;
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"; cmake --build build --config Debug --target test_tokenizer && & build/bin/Debug/test_tokenizer.exe "[tokenizer][load]"`
Expected: PASS.

- [ ] **Step 6: Verify the full build still compiles (BpeTokenizer still present)**

Run: `$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"; cmake --build build --config Debug 2>&1 | Select-Object -Last 5`
Expected: build succeeds (all apps + lsxcommon_cu still compile against the unchanged `BpeTokenizer`).

- [ ] **Step 7: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/tokenizer.h libs/lsxcommon/src/tokenizer.cpp tests/test_tokenizer.cpp
git commit -m "feat: add zero-copy Arrow tokenizer loader (ArrowTokenizerDb, load_tokenizer_metadata)"
```

---

### Task 5: Streaming pre-tokenizer — `WordStream`

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/pretokenizer.h`
- Create: `libs/lsxcommon/src/pretokenizer.cpp`
- Modify: `libs/lsxcommon/CMakeLists.txt`
- Test: `tests/test_tokenizer.cpp` (pretokenizer section)

`WordStream` is the zero-alloc single-pass splitter that reproduces the three-split `Sequence` from Design Notes (`\p{N}{1,3}` Isolated → CJK-run Isolated → main 6-alternative regex with gap words). It decodes UTF-8 on the fly, classifies via `lsxcommon::unicode::classify` (Task 1), and emits byte spans into the caller's buffer. No allocation anywhere.

- [ ] **Step 1: Write the failing pre-tokenizer test**

Append to `tests/test_tokenizer.cpp`:

```cpp
#include <lsxcommon/pretokenizer.h>
#include <vector>
#include <string>

namespace {

std::vector<std::string> SplitWords(const std::string &text)
{
    lsxcommon::WordStream ws(text.data(), text.size());
    std::vector<std::string> out;
    lsxcommon::WordSpan w;
    while (ws.next(w))
        out.emplace_back(w.begin, w.length);
    CHECK(ws.error() == 0);
    return out;
}

} // namespace

TEST_CASE("WordStream GPT-2 word splits", "[tokenizer][pretokenizer]")
{
    using V = std::vector<std::string>;
    CHECK(SplitWords("hello world") == V{"hello", " world"});
    CHECK(SplitWords("  x") == V{" ", " x"});
    CHECK(SplitWords("a  b") == V{"a", " ", " b"});
    CHECK(SplitWords(" a") == V{" a"});
    CHECK(SplitWords("a ") == V{"a", " "});
    CHECK(SplitWords("trailing  ") == V{"trailing", "  "});
    CHECK(SplitWords("x  y  z") == V{"x", " ", " y", " ", " z"});
    CHECK(SplitWords("  leading spaces") == V{" ", " leading", " spaces"});
    CHECK(SplitWords("a b") == V{"a", " b"});
}

TEST_CASE("WordStream CRLF, numbers, CJK, gaps", "[tokenizer][pretokenizer]")
{
    using V = std::vector<std::string>;
    CHECK(SplitWords("\r\n") == V{"\r\n"});
    CHECK(SplitWords("\n\n") == V{"\n\n"});
    CHECK(SplitWords("1abc") == V{"1", "abc"});
    CHECK(SplitWords("a中b") == V{"a", "中", "b"});
    CHECK(SplitWords("12345") == V{"123", "45"});
    CHECK(SplitWords("a\x01") == V{"a", "\x01"});
    CHECK(SplitWords("a\x01\x02b") == V{"a", "\x01", "\x02b"});
    CHECK(SplitWords("!abc") == V{"!abc"});
    CHECK(SplitWords("a?x") == V{"a", "?x"});
    CHECK(SplitWords("a !") == V{"a", " !"});
}

TEST_CASE("WordStream rejects malformed UTF-8", "[tokenizer][pretokenizer]")
{
    lsxcommon::WordStream ws("a\xffz", 3);
    lsxcommon::WordSpan w;
    REQUIRE(ws.next(w));                     // "a"
    CHECK(std::string(w.begin, w.length) == "a");
    CHECK_FALSE(ws.next(w));                 // stops at bad byte
    CHECK(ws.error() != 0);
}
```

Note: `SplitWords` lives in an anonymous namespace and is reused by later tasks; the encode/decode sections may reference `BuildSyntheticDb` (Task 4).

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"; cmake --build build --config Debug --target test_tokenizer 2>&1 | Select-Object -Last 5`
Expected: FAIL — `lsxcommon/pretokenizer.h` not found.

- [ ] **Step 3: Declare `WordStream` in `pretokenizer.h`**

Create `libs/lsxcommon/include/lsxcommon/pretokenizer.h`:

```cpp
#pragma once
#include <cstddef>
#include <cstdint>

namespace lsxcommon {

// Zero-copy byte span referencing the input text passed to WordStream.
struct WordSpan {
    const char *begin = nullptr;
    size_t length = 0;
};

// Single-pass streaming pre-tokenizer. Reproduces the GPT-2 3-split Sequence:
//   1. \p{N}{1,3} Isolated               (number runs, chunks of <= 3 codepoints)
//   2. [Han|Hiragana|Katakana]+ Isolated (U+4E00-U+9FA5, U+3040-U+309F, U+30A0-U+30FF)
//   3. main 6-alternative regex + Isolated gap words
// next() returns WordSpans pointing into the input buffer - no allocation,
// no copies, valid UTF-8 only. On a malformed UTF-8 sequence next() returns
// false and error() is set to 1.
class WordStream {
public:
    WordStream(const char *text, size_t length);

    bool next(WordSpan &word);
    uint32_t error() const { return error_; }

private:
    bool next_number(WordSpan &word);   // split 1 (<= 3 \p{N} codepoints)
    bool next_cjk(WordSpan &word);      // split 2 (one CJK-run word)
    bool next_main(WordSpan &word);     // split 3 (one alt match or one gap word)

    const char *const begin_;
    const char *const end_;
    size_t pos_ = 0;
    uint32_t error_ = 0;
};

} // namespace lsxcommon
```

- [ ] **Step 4: Implement the matcher in `pretokenizer.cpp`**

Create `libs/lsxcommon/src/pretokenizer.cpp`:

```cpp
#include "lsxcommon/pretokenizer.h"
#include "lsxcommon/unicode.h"
#include <cstring>

namespace lsxcommon {

namespace {

constexpr uint32_t kInvalid = 0xFFFFFFFFu;

inline bool is_ascii_punct(uint32_t cp)
{
    if (cp > 0x7E) return false;
    static const char kPunct[] = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
    return std::strchr(kPunct, static_cast<char>(cp)) != nullptr;
}

inline bool is_ascii_letter(uint32_t cp)
{
    return (cp >= 0x41 && cp <= 0x5A) || (cp >= 0x61 && cp <= 0x7A);
}

inline bool is_cjk_run(uint32_t cp)
{
    return (cp >= 0x4E00 && cp <= 0x9FA5) ||
           (cp >= 0x3040 && cp <= 0x309F) ||
           (cp >= 0x30A0 && cp <= 0x30FF);
}

// Decodes one UTF-8 codepoint at p (0 < n bytes available). Returns kInvalid
// and sets *len = 1 on any malformed sequence (overlong, surrogate, out of
// range, bad continuation). All other returns are valid Unicode.
uint32_t decode_utf8(const char *p, size_t n, size_t *len)
{
    const uint8_t *s = reinterpret_cast<const uint8_t *>(p);
    uint8_t b0 = s[0];
    if (b0 < 0x80) { *len = 1; return b0; }
    if ((b0 & 0xE0) == 0xC0) {
        if (n < 2 || (s[1] & 0xC0) != 0x80 || b0 < 0xC2) goto bad;
        *len = 2;
        return (uint32_t)((b0 & 0x1F) << 6) | (s[1] & 0x3F);
    }
    if ((b0 & 0xF0) == 0xE0) {
        if (n < 3 || (s[1] & 0xC0) != 0x80 || (s[2] & 0xC0) != 0x80) goto bad;
        uint32_t cp = (uint32_t)((b0 & 0x0F) << 12) | ((uint32_t)(s[1] & 0x3F) << 6) | (s[2] & 0x3F);
        if (cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF)) goto bad;
        *len = 3;
        return cp;
    }
    if ((b0 & 0xF8) == 0xF0) {
        if (n < 4 || (s[1] & 0xC0) != 0x80 || (s[2] & 0xC0) != 0x80 || (s[3] & 0xC0) != 0x80) goto bad;
        uint32_t cp = (uint32_t)((b0 & 0x07) << 18) | ((uint32_t)(s[1] & 0x3F) << 12) |
                      ((uint32_t)(s[2] & 0x3F) << 6) | (s[3] & 0x3F);
        if (cp < 0x10000 || cp > 0x10FFFF) goto bad;
        *len = 4;
        return cp;
    }
bad:
    *len = 1;
    return kInvalid;
}

} // anonymous namespace

WordStream::WordStream(const char *text, size_t length)
    : begin_(text), end_(text + length)
{
}

bool WordStream::next_number(WordSpan &word)
{
    size_t p = pos_;
    size_t count = 0;
    while (p < static_cast<size_t>(end_ - begin_) && count < 3) {
        size_t cl = 0;
        uint32_t c2 = decode_utf8(begin_ + p, static_cast<size_t>(end_ - (begin_ + p)), &cl);
        if (c2 == kInvalid || unicode::classify(c2) != CharClass::Number) break;
        p += cl;
        ++count;
    }
    word = WordSpan{begin_ + pos_, p - pos_};
    pos_ = p;
    return true;
}

bool WordStream::next_cjk(WordSpan &word)
{
    size_t p = pos_;
    while (p < static_cast<size_t>(end_ - begin_)) {
        size_t cl = 0;
        uint32_t c2 = decode_utf8(begin_ + p, static_cast<size_t>(end_ - (begin_ + p)), &cl);
        if (c2 == kInvalid || !is_cjk_run(c2)) break;
        p += cl;
    }
    word = WordSpan{begin_ + pos_, p - pos_};
    pos_ = p;
    return true;
}

// ---- main 6-alternative matcher ----
namespace {

// L/M run in bytes starting at a known-L/M first codepoint at 'start'.
// NOTE: breaks at CJK-run codepoints: the CJK split runs before the main
// regex (Isolated), so CJK chars are hard word boundaries even though they
// are \p{L}. Also note CharClass lives in lsxcommon::unicode, so all
// occurrences below read `unicode::CharClass::Letter` etc.
size_t consume_lm(const char *begin, const char *end, size_t start, size_t first_len)
{
    size_t run = first_len;
    size_t p = start + first_len;
    while (p < static_cast<size_t>(end - begin)) {
        size_t cl = 0;
        uint32_t c2 = decode_utf8(begin + p, static_cast<size_t>(end - (begin + p)), &cl);
        if (c2 == kInvalid) break;
        if (is_cjk_run(c2)) break;
        if (unicode::classify(c2) != CharClass::Letter &&
            unicode::classify(c2) != CharClass::Mark) break;
        run += cl;
        p += cl;
    }
    return run;
}

// Maximal Unicode whitespace run; on return *k is the byte offset after the
// last whitespace codepoint and *prev_start the start of that codepoint.
// *saw_invalid is set if decoding failed mid-run (run stops there).
void ws_run(const char *begin, const char *end, size_t pos, size_t *k, size_t *prev_start)
{
    size_t p = pos;
    size_t prev = pos;
    while (p < static_cast<size_t>(end - begin)) {
        size_t cl = 0;
        uint32_t c2 = decode_utf8(begin + p, static_cast<size_t>(end - (begin + p)), &cl);
        if (c2 == kInvalid || unicode::classify(c2) != CharClass::Whitespace) break;
        prev = p;
        p += cl;
    }
    *k = p;
    *prev_start = prev;
}

} // anonymous namespace

// Tries alt1..alt6 in priority order at 'pos'. On success sets *match_len
// (bytes) and returns true. Returns false when no alternative matches (the
// caller then emits a gap word). Malformed UTF-8 -> returns false with the
// stream error_ set by the caller on the next top-level decode; the matcher
// itself stops runs at kInvalid but never consumes beyond it.
static bool match_main(const char *begin, const char *end, size_t pos, size_t *match_len)
{
    size_t n = static_cast<size_t>(end - begin);
    if (pos >= n) return false;
    size_t cp_len = 0;
    uint32_t cp = decode_utf8(begin + pos, n - pos, &cp_len);
    if (cp == kInvalid) return false;

    // alt1: [ascii punct][ascii letter]+
    if (is_ascii_punct(cp)) {
        size_t p = pos + cp_len;
        size_t run = 0;
        while (p < n) {
            size_t cl = 0;
            uint32_t c2 = decode_utf8(begin + p, n - p, &cl);
            if (c2 == kInvalid || !is_ascii_letter(c2)) break;
            run += cl;
            p += cl;
        }
        if (run > 0) { *match_len = cp_len + run; return true; }
    }

    // alt2: [not \r\n \p{L} \p{P} \p{S}]? [\p{L}\p{M}]+
    {
        bool cp_lm = unicode::classify(cp) == CharClass::Letter ||
                     unicode::classify(cp) == CharClass::Mark;
        bool cp_other = cp != 0x0D && cp != 0x0A &&
                        unicode::classify(cp) != CharClass::Letter &&
                        unicode::classify(cp) != CharClass::Punct &&
                        unicode::classify(cp) != CharClass::Symbol;
        if (cp_lm) {
            *match_len = consume_lm(begin, end, pos, cp_len);
            return true;
        }
        if (cp_other) {
            size_t p = pos + cp_len;
            if (p < n) {
                size_t cl = 0;
                uint32_t c2 = decode_utf8(begin + p, n - p, &cl);
                if (c2 != kInvalid && (unicode::classify(c2) == CharClass::Letter ||
                                       unicode::classify(c2) == CharClass::Mark)) {
                    *match_len = cp_len + consume_lm(begin, end, p, cl);
                    return true;
                }
            }
            // prefix matched but no L/M follows -> alt2 fails, keep trying
        }
    }

    // alt3: ' '? [\p{P}\p{S}]+ [\r\n]*
    {
        size_t start = pos;
        if (cp == 0x20) start = pos + cp_len;
        size_t p = start;
        size_t ps = 0;
        bool has_ps = false;
        while (p < n) {
            size_t cl = 0;
            uint32_t c2 = decode_utf8(begin + p, n - p, &cl);
            if (c2 == kInvalid) break;
            if (unicode::classify(c2) != CharClass::Punct &&
                unicode::classify(c2) != CharClass::Symbol) break;
            ps += cl;
            p += cl;
            has_ps = true;
        }
        if (has_ps) {
            size_t crlf = 0;
            while (p < n) {
                uint8_t b = static_cast<uint8_t>(begin[p]);
                if (b != 0x0D && b != 0x0A) break;
                crlf += 1;
                p += 1;
            }
            *match_len = (start - pos) + ps + crlf;
            return true;
        }
    }

    // alt4: \s*[\r\n]+  (whitespace run ending in >= 1 \r or \n)
    if (unicode::classify(cp) == CharClass::Whitespace) {
        size_t k = 0, prev_start = 0;
        ws_run(begin, end, pos, &k, &prev_start);
        size_t last_crlf_end = static_cast<size_t>(-1);
        size_t p = pos;
        while (p < k) {
            size_t cl = 0;
            uint32_t c2 = decode_utf8(begin + p, n - p, &cl);
            if (c2 == kInvalid) break;
            if (c2 == 0x0D || c2 == 0x0A) last_crlf_end = p + cl;
            p += cl;
        }
        if (last_crlf_end != static_cast<size_t>(-1)) {
            *match_len = last_crlf_end - pos;
            return true;
        }
    }

    // alt5: \s+(?!\S)  (backtrack one codepoint when followed by non-ws)
    if (unicode::classify(cp) == CharClass::Whitespace) {
        size_t k = 0, prev_start = 0;
        ws_run(begin, end, pos, &k, &prev_start);
        if (k >= n) { *match_len = k - pos; return true; }   // lookahead sees EOF
        if (prev_start > pos) { *match_len = prev_start - pos; return true; }
        // single whitespace codepoint followed by non-whitespace -> alt5 fails
    }

    // alt6: \s+  (only reached when alt5 failed)
    if (unicode::classify(cp) == CharClass::Whitespace) {
        size_t k = 0, prev_start = 0;
        ws_run(begin, end, pos, &k, &prev_start);
        *match_len = k - pos;
        return true;
    }

    return false;
}

bool WordStream::next_main(WordSpan &word)
{
    size_t n = static_cast<size_t>(end_ - begin_);
    size_t len = 0;
    if (match_main(begin_, end_, pos_, &len)) {
        word = WordSpan{begin_ + pos_, len};
        pos_ += len;
        return true;
    }

    // Gap: extend until the next number / CJK-run / main-regex match.
    size_t gap_start = pos_;
    size_t p = pos_;
    while (p < n) {
        size_t cl = 0;
        uint32_t c2 = decode_utf8(begin_ + p, n - p, &cl);
        if (c2 == kInvalid) { error_ = 1; break; }
        if (unicode::classify(c2) == CharClass::Number || is_cjk_run(c2)) break;
        size_t mlen = 0;
        if (match_main(begin_, end_, p, &mlen)) break;
        p += cl;
    }
    word = WordSpan{begin_ + gap_start, p - gap_start};
    pos_ = p;
    return true;
}

bool WordStream::next(WordSpan &word)
{
    if (error_) return false;
    size_t n = static_cast<size_t>(end_ - begin_);
    if (pos_ >= n) return false;

    size_t cp_len = 0;
    uint32_t cp = decode_utf8(begin_ + pos_, n - pos_, &cp_len);
    if (cp == kInvalid) { error_ = 1; return false; }

    if (unicode::classify(cp) == CharClass::Number) return next_number(word);
    if (is_cjk_run(cp)) return next_cjk(word);
    return next_main(word);
}

} // namespace lsxcommon
```

This implements exactly the Design Notes matcher: alt1..alt6 priority order, alt2 space-glue (the "other" prefix), alt3 single-U+0020 + P/S run, alt4 last-CR/LF match, alt5 codepoint-aware backtracking, alt6 plain run, and Isolated gap words stopping at numbers / CJK runs / the next main-regex match.

- [ ] **Step 5: Add the source to CMake**

Append `src/pretokenizer.cpp` to the `lsxcommon` STATIC source list in `libs/lsxcommon/CMakeLists.txt` (alphabetical order, after `src/loader.cpp`):

```cmake
add_library(lsxcommon STATIC
    src/converter.cpp
    src/core.cpp
    src/common.cpp
    src/loader.cpp
    src/pretokenizer.cpp
    src/schema.cpp
    src/tokenizer.cpp
)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"; cmake --build build --config Debug --target test_tokenizer && & build/bin/Debug/test_tokenizer.exe "[tokenizer][pretokenizer]"`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/pretokenizer.h libs/lsxcommon/src/pretokenizer.cpp libs/lsxcommon/CMakeLists.txt tests/test_tokenizer.cpp
git commit -m "feat: add zero-alloc streaming WordStream pre-tokenizer (GPT-2 splits)"
```

---

### Task 6: BPE encoder — `arrow_tokenizer_encode`

**Files:**
- Modify: `libs/lsxcommon/src/tokenizer.cpp`
- Test: `tests/test_tokenizer.cpp` (encode section)

Replaces the `return -1;` stubs from Task 4 (they live in `tokenizer.cpp` after `load_tokenizer_metadata`). Encoding is two-pass over a `WordStream`: pass 1 counts the total token count (checking `kMaxWordBytes` and capacity), pass 2 writes `bos_id` then each word's stack-array BPE segmentation. No heap allocation in the hot path — a word is at most `kMaxWordBytes` bytes, so its token array fits a fixed 512-entry stack array.

The BPE word algorithm (per Design Notes): initialize each byte to `db.byte_to_token_id[byte]`, then repeatedly scan adjacent pairs, binary-search the sorted `(merge_left, merge_right)` table for each pair, and merge the pair with the **lowest** `merge_rank` into its `merge_token_id`. Stop when no adjacent pair is in the table. `merge_token_id` is used directly (never recomputed).

- [ ] **Step 1: Write the failing encode test**

Append to `tests/test_tokenizer.cpp` (reuses `BuildSyntheticDb` and the `#include <lsxcommon/tokenizer.h>` from the load section):

```cpp
#include <cstring>
#include <vector>

TEST_CASE("arrow_tokenizer_encode synthetic goldens", "[tokenizer][encode]")
{
    lsxcommon::ArrowTokenizerDb db;
    auto pinned = BuildSyntheticDb(db);

    struct Case { const char *text; std::vector<int32_t> ids; };
    const Case cases[] = {
        {"hello world", {0, 262, 267}},
        {"hello", {0, 262}},
        {"  x", {0, 35, 35, 123}},
        {" a", {0, 35, 100}},
        {"a ", {0, 100, 35}},
        {"a b", {0, 100, 35, 101}},
        {"a  b", {0, 100, 35, 35, 101}},
        {"x  y  z", {0, 123, 35, 35, 124, 35, 35, 125}},
        {"trailing  ", {0, 119, 117, 100, 108, 111, 108, 113, 106, 35, 35}},
        {"  leading spaces",
         {0, 35, 35, 111, 104, 100, 103, 108, 113, 106, 35, 118, 115, 100, 102, 104, 118}},
    };
    for (const auto &c : cases) {
        std::vector<int32_t> out(64, -7);
        size_t cap = out.size();
        int32_t n = lsxcommon::arrow_tokenizer_encode(
            db, c.text, std::strlen(c.text), out.data(), &cap);
        REQUIRE(n == static_cast<int32_t>(c.ids.size()));
        CHECK(cap == c.ids.size());
        CHECK(std::equal(out.begin(), out.begin() + n, c.ids.begin()));
    }

    fs::remove_all(pinned);
}

TEST_CASE("arrow_tokenizer_encode error codes", "[tokenizer][encode]")
{
    lsxcommon::ArrowTokenizerDb db;
    auto pinned = BuildSyntheticDb(db);

    // capacity too small -> -2, *max_out_len reports the required count
    std::vector<int32_t> small(2, -7);
    size_t cap = small.size();
    int32_t r = lsxcommon::arrow_tokenizer_encode(db, "hello world", 11, small.data(), &cap);
    CHECK(r == -2);
    CHECK(cap == 3);

    // a word longer than kMaxWordBytes -> -3 (single L-run word of 600 'a')
    std::string long_word(600, 'a');
    std::vector<int32_t> big(700, -7);
    cap = big.size();
    r = lsxcommon::arrow_tokenizer_encode(db, long_word.data(), long_word.size(), big.data(), &cap);
    CHECK(r == -3);

    // default-constructed db -> -1
    lsxcommon::ArrowTokenizerDb bad;
    std::vector<int32_t> out(16, -7);
    cap = out.size();
    r = lsxcommon::arrow_tokenizer_encode(bad, "x", 1, out.data(), &cap);
    CHECK(r == -1);

    // active-db overload
    lsxcommon::set_active_tokenizer_db(&db);
    std::vector<int32_t> out2(16, -7);
    cap = out2.size();
    r = lsxcommon::arrow_tokenizer_encode("hello", 5, out2.data(), &cap);
    CHECK(r == 2);
    CHECK(out2[0] == 0 && out2[1] == 262);
    lsxcommon::set_active_tokenizer_db(nullptr);
    cap = out2.size();
    r = lsxcommon::arrow_tokenizer_encode("hello", 5, out2.data(), &cap);
    CHECK(r == -1);

    fs::remove_all(pinned);
}
```

Note: the goldens were pinned from reference implementations against `tests/data/synthetic_tok.json` (fixture merge chains: `h e→he(259)→hel(260)→hell(261)→hello(262)` and `Ġ w→Ġw(263)→Ġwo(264)→Ġwor(265)→Ġworl(266)→Ġworld(267)`; token 35 = U+0120 Ġ = byte 0x20). If a single case mismatches after a real implementation, re-run the reference tokenizer on the fixture before "fixing" the test.

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"; cmake --build build --config Debug --target test_tokenizer && & build/bin/Debug/test_tokenizer.exe "[tokenizer][encode]"`
Expected: FAIL — encode returns `-1` (stub).

- [ ] **Step 3: Implement the encoder in `tokenizer.cpp`**

Replace the three stubs appended in Task 4 (keep `g_active_db`, `set_active_tokenizer_db`, and `valid()`):

```cpp
namespace {

// Returns the number of merged tokens for one word, or:
//   -1  a byte has no token mapping (db malformed)
//   -3  word longer than kMaxWordBytes
// Fills toks[0..n) with the merged token ids.
int32_t bpe_encode_word(const ArrowTokenizerDb &db, const char *w, size_t wlen,
                        int32_t *toks)
{
    if (wlen > kMaxWordBytes) return -3;

    for (size_t i = 0; i < wlen; ++i) {
        int32_t t = db.byte_to_token_id[static_cast<uint8_t>(w[i])];
        if (t < 0) return -1;
        toks[i] = t;
    }
    size_t n = wlen;

    while (n > 1) {
        int best_i = -1;
        int32_t best_rank = INT32_MAX;
        int32_t best_tok = -1;
        for (size_t i = 0; i + 1 < n; ++i) {
            int32_t left = toks[i], right = toks[i + 1];
            // binary search over merges sorted by (merge_left, merge_right)
            int lo = 0, hi = db.merge_count;
            while (lo < hi) {
                int mid = (lo + hi) / 2;
                if (db.merge_left[mid] < left ||
                    (db.merge_left[mid] == left && db.merge_right[mid] < right))
                    lo = mid + 1;
                else
                    hi = mid;
            }
            if (lo < db.merge_count && db.merge_left[lo] == left &&
                db.merge_right[lo] == right && db.merge_rank[lo] < best_rank) {
                best_i = static_cast<int>(i);
                best_rank = db.merge_rank[lo];
                best_tok = db.merge_token_id[lo];
            }
        }
        if (best_i < 0) break;
        toks[best_i] = best_tok;
        for (size_t i = static_cast<size_t>(best_i) + 1; i + 1 < n; ++i)
            toks[i] = toks[i + 1];
        --n;
    }
    return static_cast<int32_t>(n);
}

} // anonymous namespace

int32_t arrow_tokenizer_encode(const ArrowTokenizerDb &db, const char *text,
                               size_t text_len, int32_t *out_tokens,
                               size_t *max_out_len)
{
    if (!db.valid()) return -1;
    if (text_len == 0) text = "";

    // Pass 1: count.
    size_t total = (db.bos_id >= 0) ? 1 : 0;
    {
        WordStream ws(text, text_len);
        WordSpan w;
        int32_t stack[kMaxWordBytes];
        while (ws.next(w)) {
            int32_t n = bpe_encode_word(db, w.begin, w.length, stack);
            if (n < 0) return n;                    // -1 or -3
            total += static_cast<size_t>(n);
        }
        if (ws.error()) return -1;                  // malformed UTF-8
    }
    if (total > *max_out_len) { *max_out_len = total; return -2; }

    // Pass 2: emit bos + each word.
    size_t o = 0;
    if (db.bos_id >= 0) out_tokens[o++] = db.bos_id;
    {
        WordStream ws(text, text_len);
        WordSpan w;
        int32_t stack[kMaxWordBytes];
        while (ws.next(w)) {
            int32_t n = bpe_encode_word(db, w.begin, w.length, stack);
            for (int32_t i = 0; i < n; ++i) out_tokens[o++] = stack[i];
        }
    }
    *max_out_len = o;
    return static_cast<int32_t>(o);
}

int32_t arrow_tokenizer_encode(const char *text, size_t text_len,
                               int32_t *out_tokens, size_t *max_out_len)
{
    if (!g_active_db) return -1;
    return arrow_tokenizer_encode(*g_active_db, text, text_len, out_tokens, max_out_len);
}
```

`INT32_MAX` comes from `<limits>` + `<cstdlib>`, both already added at the top of `tokenizer.cpp` in Task 4 Step 4 — no new includes needed.

- [ ] **Step 4: Run test to verify it passes**

Run: `$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"; cmake --build build --config Debug --target test_tokenizer && & build/bin/Debug/test_tokenizer.exe "[tokenizer][encode]"`
Expected: PASS.

- [ ] **Step 5: Run the full tokenizer suite (load + convert still green)**

Run: `$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"; cmake --build build --config Debug --target test_tokenizer && & build/bin/Debug/test_tokenizer.exe "[tokenizer]"`
Expected: PASS (previous sections unaffected; `bpe_encode_word` is a new private function, `load_tokenizer_metadata` untouched).

- [ ] **Step 6: Commit**

```bash
git add libs/lsxcommon/src/tokenizer.cpp tests/test_tokenizer.cpp
git commit -m "feat: add stack-array BPE encoder (arrow_tokenizer_encode)"
```

---

### Task 7: Decoder — `arrow_tokenizer_decode`

**Files:**
- Modify: `libs/lsxcommon/src/tokenizer.cpp`
- Test: `tests/test_tokenizer.cpp` (decode section)

Raw-byte decoder: each token id's byte span is `[token_offsets[id], token_offsets[id+1])` into the mmap'd `token_bytes` buffer; concatenating the spans of the non-special ids reproduces the original UTF-8 text. Special tokens are skipped (this is what makes `decode(encode(text)) == text`, since encode prepends `bos_id`). Two passes: count then copy, so the capacity check happens before any write.

- [ ] **Step 1: Write the failing decode test**

Append to `tests/test_tokenizer.cpp`:

```cpp
TEST_CASE("arrow_tokenizer_decode round-trip and specials", "[tokenizer][decode]")
{
    lsxcommon::ArrowTokenizerDb db;
    auto pinned = BuildSyntheticDb(db);

    // round-trip: bos (special) is skipped, raw bytes come back
    {
        std::vector<int32_t> ids(64, -7);
        size_t cap = ids.size();
        int32_t n = lsxcommon::arrow_tokenizer_encode(
            db, "hello world", 11, ids.data(), &cap);
        REQUIRE(n == 3);
        std::string buf(64, '\0');
        size_t ocap = buf.size();
        int32_t r = lsxcommon::arrow_tokenizer_decode(
            db, ids.data(), static_cast<size_t>(n), buf.data(), &ocap);
        CHECK(r == 0);
        CHECK(ocap == 11);
        CHECK(std::string(buf.data(), ocap) == "hello world");
    }

    // special token in the middle is skipped
    {
        const int32_t ids[] = {100, 0, 101};
        std::string buf(64, '\0');
        size_t ocap = buf.size();
        int32_t r = lsxcommon::arrow_tokenizer_decode(db, ids, 3, buf.data(), &ocap);
        CHECK(r == 0);
        CHECK(ocap == 2);
        CHECK(std::string(buf.data(), ocap) == "ab");
    }

    // two spaces decode back to "  x"
    {
        const int32_t ids[] = {0, 35, 35, 123};
        std::string buf(64, '\0');
        size_t ocap = buf.size();
        int32_t r = lsxcommon::arrow_tokenizer_decode(db, ids, 4, buf.data(), &ocap);
        CHECK(r == 0);
        CHECK(ocap == 3);
        CHECK(std::string(buf.data(), ocap) == "  x");
    }

    fs::remove_all(pinned);
}

TEST_CASE("arrow_tokenizer_decode error codes", "[tokenizer][decode]")
{
    lsxcommon::ArrowTokenizerDb db;
    auto pinned = BuildSyntheticDb(db);

    // capacity too small -> -2, reports required count
    {
        const int32_t ids[] = {0, 262, 267};
        std::string buf(2, '\0');
        size_t ocap = buf.size();
        int32_t r = lsxcommon::arrow_tokenizer_decode(db, ids, 3, buf.data(), &ocap);
        CHECK(r == -2);
        CHECK(ocap == 11);
    }

    // out-of-range id -> -1
    {
        const int32_t ids[] = {999};
        char buf[16];
        size_t ocap = 16;
        CHECK(lsxcommon::arrow_tokenizer_decode(db, ids, 1, buf, &ocap) == -1);
    }

    // default-constructed db -> -1
    {
        lsxcommon::ArrowTokenizerDb bad;
        const int32_t ids[] = {0};
        char buf[16];
        size_t ocap = 16;
        CHECK(lsxcommon::arrow_tokenizer_decode(bad, ids, 1, buf, &ocap) == -1);
    }

    fs::remove_all(pinned);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"; cmake --build build --config Debug --target test_tokenizer && & build/bin/Debug/test_tokenizer.exe "[tokenizer][decode]"`
Expected: FAIL — decode returns `-1` (stub).

- [ ] **Step 3: Implement the decoder in `tokenizer.cpp`**

Replace the last `return -1;` stub (keep the encoder from Task 6 above it):

```cpp
int32_t arrow_tokenizer_decode(const ArrowTokenizerDb &db, const int32_t *ids,
                               size_t n_ids, char *out_text, size_t *max_out_len)
{
    if (!db.valid()) return -1;

    // Pass 1: count raw bytes, skipping specials.
    size_t total = 0;
    for (size_t i = 0; i < n_ids; ++i) {
        int32_t id = ids[i];
        if (id < 0 || id >= db.vocab_size) return -1;
        if (db.is_special[id]) continue;
        total += static_cast<size_t>(db.token_offsets[id + 1] - db.token_offsets[id]);
    }
    if (total > *max_out_len) { *max_out_len = total; return -2; }

    // Pass 2: copy raw bytes.
    size_t o = 0;
    for (size_t i = 0; i < n_ids; ++i) {
        int32_t id = ids[i];
        if (db.is_special[id]) continue;
        int64_t off = db.token_offsets[id];
        int64_t len = db.token_offsets[id + 1] - off;
        std::memcpy(out_text + o, db.token_bytes + off, static_cast<size_t>(len));
        o += static_cast<size_t>(len);
    }
    *max_out_len = o;
    return 0;
}
```

Add `#include <cstring>` to the top of `tokenizer.cpp` if not already present (needed for `std::memcpy`).

- [ ] **Step 4: Run test to verify it passes**

Run: `$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"; cmake --build build --config Debug --target test_tokenizer && & build/bin/Debug/test_tokenizer.exe "[tokenizer]"`
Expected: PASS (all sections: unicode, schema, convert, load, pretokenizer, encode, decode).

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/tokenizer.cpp tests/test_tokenizer.cpp
git commit -m "feat: add raw-byte decoder (arrow_tokenizer_decode)"
```

---

### Task 8: Wire it up — `lsxforge --tokenizer`, pipeline swap, remove `BpeTokenizer`

**Files:**
- Modify: `apps/lsxforge/src/main.cpp` — add `--tokenizer <model_path>` mode
- Modify: `libs/lsxcommon/src/inference_pipeline.cu` — use `ArrowTokenizerDb` + Arrow encode/decode
- Modify: `libs/lsxcommon/include/lsxcommon/tokenizer.h` — remove `BpeTokenizer`
- Modify: `libs/lsxcommon/src/tokenizer.cpp` — remove `BpeTokenizer` implementation

This is the final integration task: the design's stated goal ("replace the runtime `tokenizer.json` dependency and the heap-allocating `BpeTokenizer`") is completed here. The `lsxcommon_cu` OBJECT library's only tokenizer user is `inference_pipeline.cu`, so after this swap nothing references `BpeTokenizer` and it can be deleted.

- [x] **Step 1: Add `--tokenizer` mode to `lsxforge`**

In `apps/lsxforge/src/main.cpp` add `#include <filesystem>` to the includes (currently only available transitively), then:

In `main`, next to `bool shared_only = false;` add a flag and parse `--tokenizer <model_path>`:

```cpp
    bool tokenizer_only = false;
    ...
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--input") == 0 && i + 1 < argc)
            input_path = argv[++i];
        else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc)
            output_path = argv[++i];
        else if (std::strcmp(argv[i], "--shared-only") == 0)
            shared_only = true;
        else if (std::strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
            tokenizer_only = true;
            input_path = argv[++i];
        }
    }
```

After the `shared_only` block, before the generic conversion:

```cpp
    if (tokenizer_only) {
        namespace fs = std::filesystem;
        auto tokenizer_json = fs::path(input_path) / "tokenizer.json";
        auto pinned_dir = fs::path(input_path) / "logestix" / "pinned";
        fs::create_directories(pinned_dir);
        spdlog::info("Building tokenizer Arrow files from {}", tokenizer_json.string());
        auto st = lsxcommon::BuildTokenizerArrowFiles(tokenizer_json.string(), pinned_dir);
        if (!st.ok()) {
            spdlog::error("Tokenizer build failed: {}", st.ToString());
            lsxcommon::Core::Shutdown();
            return 1;
        }
        spdlog::info("Tokenizer Arrow files complete -> {}", pinned_dir.string());
        lsxcommon::Core::Shutdown();
        return 0;
    }
```

`BuildTokenizerArrowFiles` is already declared in `lsxcommon/converter.h` (included at the top of `main.cpp`). Usage: `lsxforge --tokenizer <model_path>` writes `tokenizer_vocab.arrow` + `tokenizer_merges.arrow` under `<model_path>/logestix/pinned/`.

- [x] **Step 2: Build and smoke-test `lsxforge --tokenizer`**

Run: `$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"; cmake --build build --config Debug --target lsxforge 2>&1 | Select-Object -Last 5`
Expected: build succeeds.

Run:
```powershell
$t = Join-Path $env:TEMP "lsxforge_tok_smoke"
Remove-Item -Recurse -Force $t -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $t | Out-Null
Copy-Item tests/data/synthetic_tok.json "$t\tokenizer.json"
& build/bin/Debug/lsxforge.exe --tokenizer $t
Test-Path "$t\logestix\pinned\tokenizer_vocab.arrow"
Test-Path "$t\logestix\pinned\tokenizer_merges.arrow"
```
Expected: prints "Tokenizer Arrow files complete", both files exist.

- [x] **Step 3: Swap `inference_pipeline.cu` to the Arrow tokenizer**

`inference_pipeline.cu` includes `"lsxcommon/tokenizer.h"`, so `ArrowTokenizerDb` / `load_tokenizer_metadata` / `arrow_tokenizer_encode` / `arrow_tokenizer_decode` are visible (the header gains the `arrow` includes in Task 4). Replace the load + tokenize block (currently lines ~104-125):

```cpp
    // 1. Load tokenizer (mmap the Arrow files; tok_db keeps them alive for the run)
    ::lsxcommon::ArrowTokenizerDb tok_db;
    {
        auto pinned = std::filesystem::path(mp) / "logestix" / "pinned";
        auto v = arrow::io::MemoryMappedFile::Open(
            (pinned / "tokenizer_vocab.arrow").string(), arrow::io::FileMode::READ);
        auto m = arrow::io::MemoryMappedFile::Open(
            (pinned / "tokenizer_merges.arrow").string(), arrow::io::FileMode::READ);
        if (!v.ok() || !m.ok()) {
            result.error_msg = "Failed to open tokenizer Arrow files in " + pinned.string();
            return result;
        }
        auto st = ::lsxcommon::load_tokenizer_metadata(*v, *m, tok_db);
        if (!st.ok()) {
            result.error_msg = "Failed to load tokenizer metadata: " + st.ToString();
            return result;
        }
    }

    // 2. Tokenize hardcoded prompt
    const char *prompt = "The capital of France is";
    std::vector<int32_t> input_ids(512);
    size_t in_cap = input_ids.size();
    int32_t n_ids = ::lsxcommon::arrow_tokenizer_encode(
        tok_db, prompt, std::strlen(prompt), input_ids.data(), &in_cap);
    if (n_ids < 0) {
        result.error_msg = "Tokenizer encode failed (code " + std::to_string(n_ids) + ")";
        return result;
    }
    input_ids.resize(static_cast<size_t>(n_ids));
    std::fprintf(stderr, "[inference] prompt=%zu tokens: ", input_ids.size());
    for (size_t i = 0; i < input_ids.size(); ++i)
        std::fprintf(stderr, "%d ", static_cast<int>(input_ids[i]));
    std::fprintf(stderr, "\n");
```

Replace the per-step decode + EOS check (currently lines ~364-375):

```cpp
        // Store generated token (always — gen_ids[0] used as first gen-step input)
        gen_ids.push_back(next_tok);

        std::fprintf(stderr, "[inference] step=%d tok=%d", step, next_tok);
        char tok_buf[512];
        size_t tok_cap = sizeof(tok_buf);
        std::vector<int32_t> single_tok = {static_cast<int32_t>(next_tok)};
        if (::lsxcommon::arrow_tokenizer_decode(
                tok_db, single_tok.data(), single_tok.size(), tok_buf, &tok_cap) == 0)
            std::fprintf(stderr, " '%.*s'", static_cast<int>(tok_cap), tok_buf);
        std::fprintf(stderr, " maxv=%.4f\n", max_val);

        if (next_tok == tok_db.eos_id && step >= static_cast<int>(input_ids.size()))
            break;
```

`tok_buf` is sized `kMaxWordBytes` (512): a single token is at most one 512-byte word, so decode cannot return `-2` here.

Replace the final decode (currently lines ~388-391):

```cpp
        auto gen_start = gen_ids.begin() + std::min(prompt_len, static_cast<int>(gen_ids.size()));
        std::vector<int32_t> gen_ids_i32(gen_start, gen_ids.end());
        size_t out_cap = 4096;
        std::string output(out_cap, '\0');
        int32_t dr = ::lsxcommon::arrow_tokenizer_decode(
            tok_db, gen_ids_i32.data(), gen_ids_i32.size(), output.data(), &out_cap);
        if (dr == -2) {   // *out_cap now reports the required size; retry once
            output.resize(out_cap);
            size_t cap2 = out_cap;
            dr = ::lsxcommon::arrow_tokenizer_decode(
                tok_db, gen_ids_i32.data(), gen_ids_i32.size(), output.data(), &cap2);
            out_cap = cap2;
        }
        if (dr == 0) output.resize(out_cap);
        else output.clear();
```

Keep the metric computations as-is; `result.output_text = output;` now receives the Arrow-decoded text.

- [x] **Step 4: Remove `BpeTokenizer`**

In `libs/lsxcommon/include/lsxcommon/tokenizer.h`, delete the entire `class BpeTokenizer { ... };` block (the private `Impl` + members and all 5 public methods). The file now holds only the Arrow API appended in Task 4 (`kMaxWordBytes`, `ArrowTokenizerDb`, `load_tokenizer_metadata`, `set_active_tokenizer_db`, `arrow_tokenizer_encode`, `arrow_tokenizer_decode`). Keep the includes (the Arrow API needs `<arrow/api.h>`, `<arrow/io/file.h>`, `<cstddef>`).

In `libs/lsxcommon/src/tokenizer.cpp`, delete the `struct BpeTokenizer::Impl`, the constructor/destructor/move members, `BpeTokenizer::Load`, `BpeTokenizer::Encode`, and `BpeTokenizer::Decode` (the whole first block appended originally in the testprompt plan). `load_tokenizer_metadata` + the Arrow encode/decode from Tasks 4/6/7 remain. Unused includes (e.g. simdjson) may stay — they do not affect the build.

- [x] **Step 5: Full build (all targets incl. `lsxcommon_cu`)**

Run: `$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"; cmake --build build --config Debug 2>&1 | Select-Object -Last 8`
Expected: everything compiles — `lsxcommon`, `lsxcommon_cu` (incl. `inference_pipeline.cu`), all 4 apps, all test targets. No `BpeTokenizer` references remain:

Run: `rg -n "BpeTokenizer|tokenizer\.json" libs apps tests`
Expected: no matches in `libs/`, `apps/`, `tests/` (only `docs/` and `third_party/` may match).

- [x] **Step 6: Run the full test suite**

Run: `$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"; ctest --test-dir build --output-on-failure -C Debug`
Expected: all tests PASS (test_core, test_loader, test_tokenizer, plus the per-app placeholder tests).

- [x] **Step 7: Commit**

```bash
git add apps/lsxforge/src/main.cpp libs/lsxcommon/src/inference_pipeline.cu libs/lsxcommon/include/lsxcommon/tokenizer.h libs/lsxcommon/src/tokenizer.cpp
git commit -m "feat: integrate mmap Arrow tokenizer (lsxforge --tokenizer, swap inference pipeline, drop BpeTokenizer)"
```

