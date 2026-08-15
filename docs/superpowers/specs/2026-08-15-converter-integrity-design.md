# Converter Integrity, Tokenizer, and Output Verification Pipeline

## Status: Draft

## Problem

All three model converters — `lsxcommon::qwen::ConvertQwenModel`
(`libs/lsxcommon/src/qwen_convert.cpp`), `lsxcommon::glm::ConvertGlmModel`
(`libs/lsxcommon/src/glm_convert.cpp`), and the generic `lsxcommon::Converter`
(`libs/lsxcommon/src/converter.cpp`, DeepSeek-style `logestix-v1` layout) —
silently produce broken models on damaged or incomplete input:

- Failed tensor reads only `spdlog::warn` and **skip** the tensor (silent data
  loss). See `qwen_convert.cpp:415`, `converter.cpp:739-758`, `glm_convert.cpp:434-452`.
- `config.json` geometry is never cross-checked against actual tensor shapes.
- `tokenizer.json` is optional: a missing file or a failed
  `BuildTokenizerArrowFiles` only warns and conversion "succeeds" without a
  usable tokenizer (`qwen_convert.cpp:497-502`, `converter.cpp:1266-1278`;
  GLM is the exception — it hard-fails the build but still accepts a missing file).
- Nothing verifies the converted output: `model.index.json`, arrow files,
  record counts, or tokenizer round-trip.

Goal: the converter must (1) validate the integrity of the source model before
writing anything, (2) build the binary (Arrow) tokenizer as a mandatory step and
prove it works, and (3) guarantee the output is a converted, loadable, working
model. Implemented with TDD.

## Solution

A shared verification library `lsxcommon::integrity` (`model_integrity.h/cpp`)
holding the reusable mechanics, invoked by all three converters. Each converter
keeps its own topology knowledge (which tensors a layer type requires, which
geometry fields to compare) and delegates the diffing/reading/verification to
the shared module. All integrity problems classified as *critical* abort
conversion with `arrow::Status`. Non-critical findings (extra tensors,
unclassified tensors, benign layout drift) are logged as warnings only.

No public converter signature changes: `ConvertQwenModel`,
`ConvertGlmModel`, and `Converter::Run()` keep their existing signatures; all
new behavior is internal.

## Architecture

```
lsxforge (--qwen / --glm / generic, --verify)
   │
   ▼
ConverterA::Run ─┐             ┌─ tokenizer: BuildAndVerifyTokenizer
ConverterB::Run ─┼──► integrity::  ├─ source:  VerifyAllTensorReads
ConverterC::Run ─┘             │            CheckTensorCoverage
                               │            CheckLayerContiguity
                               │            CheckGeometryFields
                               └─ output:  VerifyConvertedModel / VerificationReport
```

### Shared module API (`lsxcommon::integrity`)

```cpp
namespace lsxcommon::integrity {

// Which special tokens a model's tokenizer must carry.
struct TokenizerSpecialIds {
    bool bos_required = true;
    bool eos_required = true;
    bool pad_required = false;
};

// (1) Build binary tokenizer arrows (mandatory), then verify:
//     - re-mmap tokenizer_vocab.arrow + tokenizer_merges.arrow,
//     - load through load_tokenizer_metadata(),
//     - validate required bos/eos/pad against required_specials,
//     - byte-exact encode→decode round-trip on two hardcoded samples
//       (ASCII "The capital of France is" + a Unicode sample).
arrow::Status BuildAndVerifyTokenizer(
    const std::string &tokenizer_json_path,
    const std::filesystem::path &pinned_dir,
    const TokenizerSpecialIds &required_specials);

// (2) Source integrity: raw-read every tensor at its header-claimed offsets and
//     confirm byte count == t.byte_size(). Empty/short/OOB read hard-fails and
//     names shard + tensor. Pure CPU, runs before any output is written.
arrow::Status VerifyAllTensorReads(
    const std::filesystem::path &input_path,
    const SafetensorsModel &model);

// Coverage: missing = expected − present → hard error naming each; extra =
// present − expected → spdlog::warn only. `what` is used in error text.
arrow::Status CheckTensorCoverage(
    const std::vector<std::string> &expected,
    const std::vector<std::string> &present,
    const std::string &what);

// Layers must be a full contiguous range 0..expected_layers-1.
arrow::Status CheckLayerContiguity(int expected_layers, int tensor_max_layer);

// Geometry: ordered {name, expected-from-config} pairs vs actual-from-shapes.
// Any mismatch hard-fails naming field, expected, and actual.
arrow::Status CheckGeometryFields(
    const std::vector<std::pair<std::string, int64_t>> &expected);

// (3) Output verification report.
struct VerificationReport {
    bool ok = false;
    std::string error;
    int64_t pinned_tensor_count = 0;
    size_t arrow_files_checked = 0;
};

// Reopens every produced arrow under <output_root> (pinned/*.arrow,
// experts/**/*.arrow when present), runs RecordBatch::ValidateFull(), sums
// record counts, and cross-checks against total_pinned_tensors /
// total_experts declared by model.index.json. Dispatches on the index'
// "format" tag ("logestix-qwen" | "logestix-glm" | "logestix-v1").
arrow::Result<VerificationReport> VerifyConvertedModel(
    const std::filesystem::path &output_root);

// Resolves the index by probing output_root/logestix/model.index.json (Qwen
// layout) then output_root/model.index.json (GLM/DeepSeek layout). The pinned
// dir is always parent_of_index/pinned.

} // namespace lsxcommon::integrity
```

`negative status`: shared helpers return `arrow::Status::Invalid(...)` /
`arrow::Status::IOError(...)` with an explicit message; the converters return
that status up the call chain unchanged so lsxforge reports the precise cause.

## Data flow

1. `scan_directory()` → existing empty-model check.
2. Parse `config.json` (now **mandatory** for all three converters; missing or
   unparseable aborts). It remains the authoritative geometry source where a
   converter already reads it (Qwen) and is added for GLM and generic.
3. `VerifyAllTensorReads()` — fail before touching the output directory.
4. `CheckTensorCoverage()` — expected names generated by each converter from
   its config; diffs the classified present names (MTP/visual/unclassified
   tensors are excluded from `present`).
5. `CheckLayerContiguity()`.
6. `CheckGeometryFields()` — config-declared vs tensor-derived.
7. Convert (existing quantization + write arrow) — tensor reads now use the
   verified reader as well, so a failing read cannot leave partial output.
8. `BuildAndVerifyTokenizer()` — mandatory.
9. `VerifyConvertedModel()` — hard-fail the conversion if any check fails.

`lsxforge --verify --input <converted_root>` runs only step 9 standalone (no
CUDA needed) against an already-converted model and exits 0/1.

## Source integrity checks

- **Reads**: replace the existing warn+skip read paths with the shared verified
  reader. An empty/short/OOB read is now a hard error naming shard and tensor.
- **`config.json` mandatory**: all three converters read it; missing or
  unparseable aborts. Qwen's shape-inference fallback is removed (kept function
  body unreachable or deleted).
- **Geometry cross-check** (`CheckGeometryFields`): each converter supplies
  `{config-declared, tensor-derived}` pairs — Qwen: `num_hidden_layers` vs
  `max_layer+1`, `hidden_size` vs embed `[1]`, `vocab_size` vs embed `[0]`,
  `intermediate_size` vs `mlp.gate_proj [1]`, head dims; GLM: `num_layers`,
  `hidden_size`, `experts_per_layer` vs tensor-derived counts; DeepSeek:
  `num_layers`, `hidden_size`, `vocab_size`, `experts_per_layer`.
- **Coverage** (`CheckTensorCoverage`): per-converter expected-name generation:
  - Qwen: fixed suffix sets per layer driven by `layer_types`
    (`linear_attention`: in_proj_qkv/z/b/a, out_proj, conv1d, norm, A_log,
    dt_bias; `full_attention`: q/k/v/o_proj, q_norm, k_norm; both: input_layernorm,
    post_attention_layernorm, mlp gate/up/down) + globals (embed_tokens,
    lm_head, norm).
  - GLM: dense / MoE / indexer layer templates + router + shared_experts +
    every routed expert index `0..experts_per_layer-1`.
  - DeepSeek: attention tensors + the full expert grid per layer.
  - Missing are hard errors; extra/unexpected are warnings.
- **Layer contiguity** (`CheckLayerContiguity`): config `num_layers` vs the
  maximum classified layer index, catching holes like layers 0,1,3.

Non-goal for this pass: dtype whitelists (NVFP4 quantization already accepts
BF16 and F32; dtype rules are YAGNI here).

## Tokenizer pipeline

`BuildAndVerifyTokenizer` runs at the end of every conversion:

1. `BuildTokenizerArrowFiles()` unchanged; any error aborts (mandatory).
2. Re-open `tokenizer_vocab.arrow` / `tokenizer_merges.arrow` via
   MemoryMappedFile and `load_tokenizer_metadata()`; failure aborts.
3. Validate `bos_id`/`eos_id`/`pad_id` against `required_specials`; a required
   special missing (id < 0) aborts.
4. Byte-exact round-trip on `"The capital of France is"` and a Unicode sample:
   encode must return ≥1 token and no negative code; decode must return the
   source bytes exactly. Synthetic tokenizers cover all 256 GPT-2 bytes, so
   byte-exact round-trip is a valid strict guarantee.

`required_specials` is the sole parameter beyond the file paths; the two
round-trip samples are hardcoded inside `BuildAndVerifyTokenizer`.
Qwen/glm/deepseek all declare `bos+eos required, pad optional`.

## Output verification

`VerifyConvertedModel(output_root)`:

1. Resolves `model.index.json` by probing
   `output_root/logestix/model.index.json` (Qwen layout) then falling back to
   `output_root/model.index.json` (GLM/DeepSeek layout). The pinned dir is
   derived as `parent_of_index/pinned`.
2. Parses the index and dispatches on `"format"` in
   `{"logestix-qwen", "logestix-glm", "logestix-v1"}`, plus the declared
   `total_pinned_tensors` / `total_experts`.
3. Every produced arrow file under the pinned dir reopens via
   `ReadRecordBatchFromFile` and passes `RecordBatch::ValidateFull()` — a
   format-agnostic IPC integrity check (stronger than schema-class dispatch,
   no per-converter coupling).
3. Summed pinned record count equals `total_pinned_tensors`; when present,
   expert arrows total equals `total_experts`. Mismatch aborts.
4. Tokenizer arrows are present (re-checked here).
5. Returns `VerificationReport{ok, error, pinned_tensor_count, arrow_files_checked}`
   and hard-fails the surrounding conversion on any false.

`lsxforge --verify` surfaces a concise spdlog summary and exit code 0/1.

## Converter wiring

Identical 9-step shape for all three; only topology/geometry/extras differ.

| Step | Qwen | GLM-5.2 | Generic (DeepSeek) |
|---|---|---|---|
| 1. Scan shards | existing | existing | existing |
| 2. `config.json` mandatory | already parses — missing becomes error; shape-infer fallback removed | **add config parse** | **add config parse** |
| 3. `VerifyAllTensorReads` | shared | shared | shared |
| 4. `CheckTensorCoverage` | layer_types driven | dense/MoE/indexer templates + expert indices 0..E-1 | attention + expert grid |
| 5. `CheckLayerContiguity` | num_hidden_layers | num_layers (79) | num_layers |
| 6. `CheckGeometryFields` | hidden/vocab/intermediate/heads | hidden/heads/experts_per_layer | hidden/vocab/layers/experts |
| 7. Convert | verified read path | verified read path | verified read path |
| 8. `BuildAndVerifyTokenizer` | bos+eos required | bos+eos required | bos+eos required |
| 9. `VerifyConvertedModel` | `logestix-qwen` | `logestix-glm` | `logestix-v1` |

## Files

### New files

| File | Purpose |
|------|---------|
| `libs/lsxcommon/include/lsxcommon/model_integrity.h` | Public `integrity::` API |
| `libs/lsxcommon/src/model_integrity.cpp` | Implementation |
| `tests/test_model_integrity.cpp` | Shared-helper CPU unit tests |
| `tests/test_qwen_converter_integrity.cpp` | Qwen converter integration tests |

### Changed files

| File | Change |
|------|--------|
| `libs/lsxcommon/src/qwen_convert.cpp` | Wire steps 2-9; config mandatory; verified reads |
| `libs/lsxcommon/src/glm_convert.cpp` | Add config parse; wire steps 2-9; verified reads |
| `libs/lsxcommon/src/converter.cpp` | Add config parse; wire steps 2-9; verified reads |
| `apps/lsxforge/src/main.cpp` | `--verify` flag |
| `libs/lsxcommon/CMakeLists.txt` | Add `model_integrity.cpp` |
| `tests/CMakeLists.txt` | Add 2 test targets + `TOKENIZER_TEST_DATA_DIR` where missing; POST_BUILD arrow dll copies |

### Unchanged

All CUDA kernels, loaders, `tokenizer_convert.cpp` (only wrapped by the new
verify layer), schema classes, model classes, and CLIs other than the `--verify`
flag.

## Tests (TDD)

Written first, red → implement → green.

### A. Shared helpers — `tests/test_model_integrity.cpp` (CPU-only)

- `CheckGeometryFields`: equal pairs pass; a mismatch fails naming expected vs actual.
- `CheckTensorCoverage`: missing → error listing names; extra → OK (warn).
- `CheckLayerContiguity`: complete range passes; a hole fails.
- `VerifyTokenizerRoundTrip`: byte-exact round-trip using `tests/data/synthetic_tok.json`;
  empty-token output fails.
- Special-id validation: bos+eos present passes; missing bos fails.

### B. Qwen integration — `tests/test_qwen_converter_integrity.cpp`

Reuses `qwen_test_fixture.h`; test data tokenizer copied into the fixture dir.

1. Happy path converts and `VerifyConvertedModel` passes with matching counts.
2. Missing `config.json` → abort.
3. Malformed `config.json` → abort.
4. Config `hidden_size` mismatches tensors → abort.
5. Delete layer-0 `mlp.down_proj.weight` from safetensors → abort, names it.
6. Layer hole (drop layer 2, keep 0,1,3) → abort.
7. Truncated safetensors shard → abort on `VerifyAllTensorReads`.
8. Missing `tokenizer.json` → abort.
9. Broken `tokenizer.json` (invalid JSON) → abort.
10. Tokenizer without required `bos` special → abort.
11. Tampered `total_pinned_tensors` in index.json → `VerifyConvertedModel` fails.
12. Truncated `global.arrow` → offline `--verify` fails.

### C. GLM + DeepSeek wiring

One happy-path + one abort-path (missing tensor / missing tokenizer) each,
reusing the existing GLM and DeepSeek test fixtures.

### Verification cadence

- Each test: `cmake --build --preset <debug>` → targeted ctest → red → implement → green.
- Full pass: `ctest --preset windows-debug`, then `ctest -R glm` and
  `ctest -R deepseek`.

## Open questions

- GLM and DeepSeek already derive topology from tensor scans; the newly added
  `config.json` parse must find the right geometry keys for those artifacts
  (verified against the real model dirs during implementation).