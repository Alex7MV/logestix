# GLM-5.2: Meaningful `--run` Generation Output

## Overview

The `--run "PROMPT"` generation path in `lsxforge` (commit `7101a54`) loads a
model, encodes the prompt, runs `IModel::Infer` and prints
`result.output_text`. For GLM-5.2 this output is currently always empty:
`GlmV5Model::Infer` publishes only the raw `gen_tokens` id stream and the
`final_logits` row, leaving `output_text` undecoded (see
`glm_v5_model.cpp:148-149`). As a consequence, no meaningful response is ever
generated for a GLM-5.2 test prompt.

This spec makes GLM-5.2 produce a meaningful response on the `--run` path via
TDD:

- `GlmV5Model::Infer` fills `output_text` by tokenizer-decoding the generated
  token stream (the prompt tokens are not part of it).
- A shared host-only framing helper in `lsxcommon` applies the GLM-5.2
  chat-template wrapping `[gMASK]<sop><|user|>{prompt}<|assistant|><think></think>`
  (non-thinking generation mode) so a bare prompt yields a coherent answer.
- `lsxforge --run` routes the resolved prompt through the helper.

## Acceptance Criteria

- `GlmV5Model::Infer` returns a non-empty `output_text` (decoded without
  errors) whenever generation succeeds.
- For the test prompt `"The capital of France is"` (GLM chat-template framed)
  the generated `output_text` contains `"Paris"`. Determinism is guaranteed by
  the bit-exact SelfCheck contract already covered by `test_glm_selfcheck`
  (`gen_tokens[0]` decodes to `"Paris"`).
- Real-model test SKIPs (CTest `SKIP_RETURN_CODE 4`) when no GLM-5.2
  checkpoint is present, matching the existing GLM real-weight tests.

## Architecture

### Component A: Prompt framing helper

New host-only module (no NVCC, no CUDA):

- `libs/lsxcommon/include/lsxcommon/prompt_framing.h`
- `libs/lsxcommon/src/prompt_framing.cpp`

```cpp
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

Implementation rules:

- Special tokens resolved via `arrow_tokenizer_lookup` (`[gMASK]`, `<sop>`,
  `<|user|>`, `<|assistant|>`, `<think>`, `</think>`). If any is absent from
  the vocab, encode the raw prompt (graceful fallback — same policy as
  `EncodeSelfCheckPrompt` in `apps/lsxfabric/src/main.cpp`).
- Encode with `arrow_tokenizer_encode` through a capacity-grow loop: start at
  256 ids, double up to 8192 on return code `-2` (capacity too small). GLM
  framing reserves 6 slots ahead of the prompt tokens.
- GLM never auto-prepends BOS; the helper does not add one.
- Empty/blank prompt or negative encode result (other than `-2`) → `false`.
- Registered in `libs/lsxcommon/CMakeLists.txt` `lsxcommon` source list.

### Component B: `output_text` decode in `GlmV5Model::Infer`

`libs/lsxcommon/src/glm_v5_model.cpp`, inside `Infer()` after the generation
result is collected (currently lines 145-155):

- Keep `r.first_gen_tok_id` / `r.first_gen_maxv` population.
- Decode `gen.tokens` with `arrow_tokenizer_decode` into `r.output_text`,
  mirroring the capacity-retry pattern of
  `deepseek_inference_pipeline.cu:703-715` (first call with a 4096 buffer, on
  `-2` resize to the reported required size and retry).
- `gen.tokens` contains only generated tokens (prompt excluded); the decode
  skips specials and appends raw bytes.
- Decode failure leaves `output_text` empty but keeps `r.ok = true` (generation
  succeeded; decode is best-effort), with an spdlog warning.

### Component C: `lsxforge --run` wiring

`apps/lsxforge/src/main.cpp`, generation path (currently lines 250-261):

- Replace the direct `arrow_tokenizer_encode` call with
  `lsxcommon::prompt::BuildPromptInputIds(model->Type(), model->Tokenizer(),
  run_prompt, input_ids)`.
- On `false` → spdlog error, `model->Unload()`, `Core::Shutdown()`, exit 1
  (same error path as today).
- Add `#include "lsxcommon/prompt_framing.h"`.

## Data Flow

```
lsxforge --run "PROMPT"
 → ResolveGenerationPrompt (bare --run → blueprint prompt)
 → LsxModelFactory::Create → model->Load()
 → prompt::BuildPromptInputIds(model->Type(), tok, prompt)
     → glm-5.2: [gMASK]<sop><|user|>PROMPT<|assistant|><think></think>
     → other: raw ids
 → model->Infer(req)
     → RunGlmRealGeneration → gen.tokens (generated only)
     → output_text = arrow_tokenizer_decode(gen.tokens)   [NEW]
 → print result.output_text (was empty for GLM)
```

## Error Handling

| Path | Behavior |
|---|---|
| `BuildPromptInputIds` empty/blank prompt | returns `false` → lsxforge logs error, exit 1 |
| `BuildPromptInputIds` encode capacity | grow loop (256 → 8192) before giving up |
| `BuildPromptInputIds` missing GLM specials | raw fallback (still succeeds) |
| `arrow_tokenizer_decode` capacity | retry with reported required size |
| `arrow_tokenizer_decode` failure | `output_text` empty, `r.ok` stays `true`, spdlog warning |
| GLM inference capacity (`prompt+gen > 1024`) | unchanged; `RunGlmRealGeneration` already rejects |

## Testing (TDD)

### Test 1 — host-only `tests/test_prompt_framing.cpp`

No CUDA, no real model. Uses `BuildTokenizerArrowFiles` +
`load_tokenizer_metadata` (pattern of `test_tokenizer.cpp:247-264`).

New fixture `tests/data/synthetic_glm_tok.json` — copy of `synthetic_tok.json`
with the six GLM special tokens added to `added_tokens` and the `vocab` map
(fresh ids past the existing 268).

Test cases:

1. GLM framing: `"The capital of France is"` →
   `[gmask, sop, usr, ...prompt_ids..., asst, think, endthink]`; length equals
   raw-encode length + 6; the six sentinel ids sit at positions 0,1,2 and
   `n`, `n+1`, `n+2` (checked via `arrow_tokenizer_lookup`).
2. Non-GLM type (`"deepseek-v4-flash"`) → raw encode, no framing.
3. GLM + tokenizer without the specials (`synthetic_tok.json`) → raw fallback.
4. Empty prompt → `false`.
5. Long prompt (> 256 tokens) → success (capacity-grow loop exercised).

### Test 2 — real-model `tests/test_glm_prompt_output.cpp`

SKIPs (Catch2 `SKIP`, CTest `SKIP_RETURN_CODE 4`) when the GLM-5.2 checkpoint
is absent. Model path resolution identical to `test_glm_selfcheck.cpp`
(`LSX_TEST_GLM_MODEL_PATH` env override, else `lsx_model_models_dir()/"GLM-5.2"`).

- Load model via `LsxModelFactory`, assert `Type() == "glm-5.2"`.
- `BuildPromptInputIds(m->Type(), m->Tokenizer(), "The capital of France is", ids)`.
- `ModelRequest req{ids, 8}` → `m->Infer(req)`.
- Assert `result.ok`, `!result.output_text.empty()`,
  `result.output_text.find("Paris") != npos`, `!result.gen_tokens.empty()`.
- `m->Unload()`.

### Test registry (`tests/CMakeLists.txt`)

Both tests follow the `test_glm_selfcheck` pattern: `add_executable` +
`TOKENIZER_TEST_DATA_DIR` (host-only test needs the fixture) +
`Catch2::Catch2WithMain` + WIN32 arrow DLL POST_BUILD copy + `add_test`.
`test_glm_prompt_output` gets `set_tests_properties(... SKIP_RETURN_CODE 4)`.

### TDD order (RED → GREEN)

1. RED: add both tests + CMake wiring; build → `test_prompt_framing` fails to
   compile (`prompt_framing.h` missing); `test_glm_prompt_output` fails on
   empty `output_text` (or SKIPs if no checkpoint).
2. GREEN: implement Component A, Component B, Component C; run
   `ctest -R "test_prompt_framing|test_glm_prompt_output|test_generation_cli"`.

## Files Changed

| File | Change |
|---|---|
| `libs/lsxcommon/include/lsxcommon/prompt_framing.h` | new |
| `libs/lsxcommon/src/prompt_framing.cpp` | new |
| `libs/lsxcommon/CMakeLists.txt` | add `prompt_framing.cpp` |
| `libs/lsxcommon/src/glm_v5_model.cpp` | `output_text` decode in `Infer` |
| `apps/lsxforge/src/main.cpp` | use `BuildPromptInputIds` in `--run` path |
| `tests/data/synthetic_glm_tok.json` | new fixture |
| `tests/test_prompt_framing.cpp` | new |
| `tests/test_glm_prompt_output.cpp` | new |
| `tests/CMakeLists.txt` | register both tests |

## Out of Scope

- `lsxfabric --testprompt` — untouched (still hardcoded prompt; no output print).
- Auto-framing inside `GlmV5Model::Infer` — rejected (violates the
  caller-owns-input-ids contract).
- Extending `generation_cli` — rejected (keeps string-only CLI utilities free
  of tokenizer/model dependencies).
