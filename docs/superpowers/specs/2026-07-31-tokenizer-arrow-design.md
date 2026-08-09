# Zero-Alloc, Zero-Copy Arrow Binary Tokenizer

## Overview

Replace the runtime `tokenizer.json` dependency and the host-side heap
allocation tax of the current `BpeTokenizer` (simdjson + `std::string` +
`std::unordered_map`) with a zero-allocation, zero-copy BPE tokenizer that
operates directly over two memory-mapped Arrow files produced at build time by
the converter (`lsxforge`).

The runtime encode path forbids dynamic strings, hash maps, and dynamic array
allocations. Everything reads through flat host pointers into the mmap'd
columnar payloads.

## Context Facts (DeepSeek-V4-Flash)

- Vocab: **128,000** tokens, dense ids `0..127999`.
- Merges: **127,741** byte-level BPE rules.
- Byte-level BPE (GPT-2 style): 3 specials at ids 0–2
  (`<|beginofsentence|>`, `<|endofsentence|>`, `<|pad|>`), ids 3..258 are the
  256 byte-level tokens, the rest are merge products.
- Pre-tokenizer: chained Splits `\p{N}{1,3}` → CJK punctuation run → main
  GPT-2 regex (6 alternatives) → ByteLevel (`use_regex=false`,
  `add_prefix_space=false`). Decoder: ByteLevel (`add_prefix_space=true`).
- `base_token_offset = vocab_size − merge_count = 259` holds for this model,
  but the design **must not rely on any linear id↔merge-order relation**.

## Design Decisions (from brainstorming)

1. **Pair-rule storage**: BPE merge rules are pairs `(left, right) → rank`.
   A flat rank array cannot answer "is (a,b) mergeable and at what rank?" (the
   byte-concatenation shortcut is provably wrong). The merge pair table is
   stored explicitly as sorted columns.
2. **`merge_token_id` via Vocab Lookup** — never by formula. The converter
   resolves each merge product by looking up the concatenated raw byte string
   in the vocab (`bytes(A) ++ bytes(B)`). The only invariant checked: that
   lookup must succeed. No `base_token_offset` anywhere in the design.
3. **Faithful GPT-2 regex pre-tokenizer** (zero-alloc scanner), not a
   simplified splitter.
4. **Arrow decode included** so `tokenizer.json` can be fully removed from the
   runtime (the pipeline calls Decode for generated output).
5. **Two sibling Arrow files** (one schema per file, Arrow constraint), not a
   combined batch in `shared_experts.arrow`.
6. **Approach A** for the encode engine: direct mmap + binary search over the
   sorted `(left,right)` rule table + linear min-rank merge scan on fixed stack
   arrays. No heap, no shadow copies.

## On-Disk Layout

Two files under `<model_path>/logestix/pinned/`.

### `tokenizer_vocab.arrow`

One record batch, 128,000 rows sorted by `token_id`.

| Column | Type | Notes |
|---|---|---|
| `token_id` | Int32 | dense 0..127999 |
| `token_bytes` | LargeBinary | raw model bytes; each int64 offset kept to an 8-byte multiple (converter pads between tokens) so `ptr = base + offset` is 8-byte aligned |
| `is_special` | Int8 | 1 for ids 0,1,2 (and any added specials); else 0 |

Schema metadata (kv pairs, parsed once at load):
`bos_id`, `eos_id`, `pad_id`, `vocab_size`, and `byte_to_token_id` — the
256-int byte→initial-token-id table serialized as a string (e.g.
comma-separated). This is the only byte-level-transform artifact the runtime
needs; it avoids O(log V) reverse lookups per input byte.

### `tokenizer_merges.arrow`

One record batch, 127,741 rows sorted by `(merge_left, merge_right)`.

| Column | Type | Notes |
|---|---|---|
| `merge_left` | Int32 | left token id |
| `merge_right` | Int32 | right token id |
| `merge_rank` | Int32 | creation order (0 = first merge; authoritative from `tokenizer.json` merges array) |
| `merge_token_id` | Int32 | the token created by this merge (resolved by vocab lookup) |

Schema metadata: `merge_count`.

### Deliberate omissions

- The 256-entry byte↔Unicode table and the Unicode class ranges
  (`\p{N}`, `\p{L}`, `\p{M}`, `\p{P}`, `\p{S}`, `\s`) are **hardcoded const
  data in the splitter source** — deterministic for the GPT-2 byte-level
  tokenizer and not model-specific. The regex splitter logic is code; vocab,
  merges, and byte-map are data.

## Converter Ingestion (`lsxforge --tokenizer <model_path>`)

Build-time tool (`simdjson` + heap are acceptable here; the zero-alloc mandate
applies to the runtime).

1. Parse `tokenizer.json`: `model.vocab` (128k entries, key = byte-level string,
   value = id), `model.merges` (127,741 lines `"A B"`), `added_tokens`.
2. Reconstruct the standard GPT-2 byte_unicode table via its construction
   algorithm (bytes 33..126 & 161..172 & 174..255 self-map; the remainder map
   to U+0100+). Map every vocab key char back to its raw byte to obtain
   `token_bytes`. Any unmappable char → hard error (defensive).
3. Derive `byte_to_token_id[256]` = `vocab[table[b]]`; embed in vocab metadata.
4. Build the merge table via **vocab lookup only**:
   - `left = vocab_lookup(bytes(A))`, `right = vocab_lookup(bytes(B))`
   - `merge_token_id = vocab_lookup(bytes(A) ++ bytes(B))`
   - `merge_rank = i` (position in the merges array)
   - invariant: every `bytes(A) ++ bytes(B)` lookup must succeed, else fail
     loudly with the offending merge line.
   - sort rows by `(left, right)`.
5. Detect specials from `added_tokens` by content substring (`begin`, `end`,
   `pad`); fall back to ids 0,1,2 if absent.
6. Write both Arrow files via `WriteRecordBatchToFile`; pad `token_bytes`
   offsets to 8-byte multiples.

Batch-building code lives in `lsxcommon` (`src/tokenizer_convert.cpp`,
declared in `converter.h`) so Catch2 tests can call it directly; `lsxforge`
is the CLI wrapper.

## Host Metadata Loader

### `ArrowTokenizerDb`

```cpp
struct ArrowTokenizerDb {
    std::shared_ptr<arrow::io::MemoryMappedFile> vocab_mmap;
    std::shared_ptr<arrow::io::MemoryMappedFile> merges_mmap;
    const uint8_t* vocab_file_base = nullptr;   // ptr − file_base offset math
    int64_t        vocab_file_size  = 0;

    const int64_t* token_offsets = nullptr;     // LargeBinary offsets, n = vocab_size+1
    const uint8_t* token_bytes   = nullptr;     // value buffer (8-byte aligned)
    const uint8_t* is_special    = nullptr;     // Int8 column
    int32_t        vocab_size    = 0;

    const int32_t* merge_left      = nullptr;
    const int32_t* merge_right     = nullptr;
    const int32_t* merge_rank      = nullptr;
    const int32_t* merge_token_id  = nullptr;
    int32_t        merge_count     = 0;

    int32_t bos_id = -1, eos_id = -1, pad_id = -1;
    int32_t byte_to_token_id[256];
    std::string error_msg;

    bool valid() const;   // all pointers non-null and counts > 0
};

arrow::Status load_tokenizer_metadata(
    const std::shared_ptr<arrow::io::MemoryMappedFile>& vocab_mmap,
    const std::shared_ptr<arrow::io::MemoryMappedFile>& merges_mmap,
    ArrowTokenizerDb& db);
```

`load_tokenizer_metadata` (same pattern as `load_shared_experts_metadata`):

1. `ReadAt(0,1)` → `vocab_file_base` (zero-copy).
2. `RecordBatchFileReader::Open` → batch 0 of each file; cast to
   `LargeBinaryArray` / `Int32Array` / `Int8Array`.
3. Raw pointers via `raw_value_offsets()` / `raw_values()` /
   `value_data()->data()`; exact offsets via `ptr − file_base`
   (`Buffer::offset()` is private in Arrow 25).
4. Zero-copy validation: every payload must lie in
   `[file_base, file_base + file_size)`.
5. Parse schema metadata: `bos_id`/`eos_id`/`pad_id`/`vocab_size`/
   `byte_to_token_id` (vocab file), `merge_count` (merges file).
6. Validate: `vocab_size > 0`, `merge_count > 0`, all `byte_to_token_id`
   values in `[0, vocab_size)`, offsets length == vocab_size+1. A monotonic
   `(left,right)` scan over the merge table is optional debug validation.

## Zero-Alloc Encode (`arrow_tokenizer_encode`)

### API

```cpp
// Exact signature from the task spec; uses the process-global "active" db.
int32_t arrow_tokenizer_encode(const char* text, size_t text_len,
                               int32_t* out_tokens, size_t* max_out_len);

// db-parameterized variant (tests, multi-threaded use).
int32_t arrow_tokenizer_encode(const ArrowTokenizerDb& db, const char* text,
                               size_t text_len, int32_t* out_tokens,
                               size_t* max_out_len);
```

Return semantics: `>= 0` = token count; `-1` invalid args; `-2` output buffer
too small (`*max_out_len` ← required capacity); `-3` a pre-tokenized word
exceeds `MAX_WORD_BYTES`. `*max_out_len` is in/out (capacity → written).
BOS-prepend preserved (as in the current `BpeTokenizer::Encode`).

### Pre-tokenizer

One-pass zero-alloc scanner with the exact semantics of the chained Splits:

1. Decode UTF-8 → classify each codepoint into
   {N, L, M, P, S, whitespace, other} via hardcoded const Unicode ranges.
2. Combined matcher with HF semantics (leftmost match, alternation priority,
   Isolated segmentation) for:
   - `\p{N}{1,3}` (isolated number runs),
   - CJK punctuation `[\p{P}\p{S}]+` over CJK ranges,
   - main regex, 6 alternatives:
     `[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+` |
     `[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+` |
     ` ?[\p{P}\p{S}]+[\r\n]*` | `\s*[\r\n]+` | `\s+(?!\S)` | `\s+`.
   ByteLevel step (`use_regex=false`) is a no-op in the raw-byte domain.
3. Each matched word feeds BPE. `MAX_WORD_BYTES = 512`; longer word → `-3`.

### BPE per word (Approach A)

- Fixed stack arrays only; initialize `sym_id[j] = byte_to_token_id[word[j]]`.
- Merge loop:
  - scan all adjacent pairs; for each, binary search `(left,right)` in the
    sorted merge table; track the pair with minimal `merge_rank`;
  - no mergeable pair → done; else `sym_id[pos] = merge_token_id`, shift
    left, length −1.
- Complexity O(n²·log M), n ≤ 512; typical words ≤ 15 symbols.
- Final `sym_id[0..n-1]` appended to `out_tokens` under the capacity check.

Zero-alloc guarantee: stack + const tables + mmap pointers only; no
`std::string`, `std::unordered_map`, or `new` in the encode path.

## Decode

```cpp
int32_t arrow_tokenizer_decode(const ArrowTokenizerDb& db, const int32_t* ids,
                               size_t n_ids, char* out_text, size_t* max_out_len);
```

For each id: skip if out of range or `is_special[id]`; otherwise copy
`[offsets[id], offsets[id+1])` raw bytes to the output. `*max_out_len` is
in/out with the same error semantics as encode. No byte↔Unicode table needed
at runtime (token_bytes are raw).

## Integration & Cleanup

- New files: `include/lsxcommon/tokenizer.h`, `src/tokenizer.cpp`,
  `src/tokenizer_convert.cpp` (decls in `converter.h`).
- `tokenizer.h`/`tokenizer.cpp` replace the legacy `BpeTokenizer` content.
- `tokenizer_convert.cpp` added to the `lsxcommon` STATIC sources.
- `lsxforge`: `--tokenizer <model_path>` writes
  `pinned/tokenizer_vocab.arrow` + `pinned/tokenizer_merges.arrow`.
- `inference_pipeline.cu`: open both mmaps → `load_tokenizer_metadata` →
  local `ArrowTokenizerDb`; replace `BpeTokenizer::Encode/Decode` call sites;
  set the process-global active db for the exact-signature entry point.
- Legacy `BpeTokenizer` and runtime `tokenizer.json` reading removed entirely.
  simdjson remains linked only for the build-time converter.

## Testing

1. **Converter round-trip (synthetic):** small tokenizer (256 bytes + a few
   merges + specials) → build batches → `load_tokenizer_metadata` → verify
   pointers, values, `byte_to_token_id`, `is_special`.
2. **Merge-table invariants:** sorted by `(left,right)`; `merge_rank` matches
   input order; `merge_token_id` == vocab lookup of the concatenated bytes.
3. **Golden encode (synthetic):** fixed strings → exact id sequences computed
   by hand from the synthetic merges.
4. **Golden encode (real data):** ~10 strings, exact ids captured once from a
   reference implementation during development and committed as test data. If
   no reference tool is available, fall back to synthetic goldens + structural
   checks over the real vocab.
5. **Error paths:** `-1` null args, `-2` small buffer (`*max_out_len` updated),
   `-3` word > 512 bytes.
6. **Zero-alloc by construction:** encode path uses stack/const/mmap only —
   verified by review plus a source scan test forbidding
   `std::string`/`std::unordered_map`/`new` in encode functions.
7. **Integration:** all existing ctest suites pass; `--testprompt` runs with
   the new tokenizer; grep confirms `tokenizer.json` is not read at runtime.
