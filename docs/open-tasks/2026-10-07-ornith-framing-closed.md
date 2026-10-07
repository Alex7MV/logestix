# Ornith-1.5-35B-A3B-MLX conversion + prompt-framing bug - closed 2026-10-07

Snapshot of the work session that converted `Ornith-1.5-35B-A3B-MLX` and found
(and fixed) a real prompt-framing defect in the Ornith family. The item was
never open in `docs/OPEN_TASKS.md`; this file is the investigation trail, kept
because the wrong hypotheses and how they were ruled out are the valuable part.

## Summary

- **Conversion** `lsxforge --ornith`: 40 layers x 256 experts, 613 pinned
  tensors, ~19 GiB, no errors.
- **Selfchecks**: Paris **RED on first run** (the model emitted junk:
  `The capital of  (  /  /  1  1 `). FSI/CODE/DET **SKIP** (no goldens).
- **Root cause**: `OrnithMoeModel::BuildPromptInputIds` rendered a hand-rolled
  ChatML frame that *omits the leading system turn*. This reasoning fine-tune
  degenerates (repetition collapse) on a frame that opens at the user turn; the
  chat/tool path worked only because the conversation carried a system message.
- **Fix**: render the single turn through the native Qwen renderer with a
  system + user conversation (mirrors `NemotronModelBase`), so
  `BuildPromptInputIds` always opens `<|im_start|>system…`; also inject a default
  system turn in `BuildConversationInputIds` when the caller omitted one.
- **After the fix**: Paris **PASS** (48 assertions, 3 cases, deterministic
  `logits_hash` across cycles); the synthetic Ornith suites stay green.

## Investigation (in order)

The failing selfcheck output was a repetition collapse. The model also produced
garbage through `lsxforge --run` and `lsxfabric --testprompt` (both use
`RunCliGeneration` -> `BuildPromptInputIds`), while the in-test tool round-trip
(`BuildConversationInputIds`) produced a coherent reasoning trace + tool call.
So the weights were fine and the defect was in the prompt frame.

Hypotheses tested and rejected:

1. **Corrupted frame tokens.** Logged the runtime `BuildPromptInputIds` ids:
   `248045 846 198 760 6511 314 9338 369 248046 198 248045 74455 198 248068 271 248069 271`.
   This is **byte-identical** to the HF tokenization of the checkpoint's own
   `chat_template.jinja` (no-system, `enable_thinking=false`). The frame was
   correct, so the tokenization was not the cause.
2. **Auto-prepended BOS.** `arrow_tokenizer_encode` prepends `db.bos_id` when
   `>= 0`, and the sibling `AppendQwenEncoded` strips it while the Ornith helper
   did not. But this checkpoint's `bos_token_id` is **None** (`db.bos_id < 0`),
   so no BOS was ever injected — a no-op.
3. **Missing thinking.** Opening `<think>` instead of the empty
   `<think>\n\n</think>\n\n` changed the failure shape but did not fix it.

Hypothesis that held: **the leading system turn is required.** Adding
`<|im_start|>system\n…<|im_end|>\n` before the user turn produced exactly the
expected control answer `The capital of France is **Paris**.` — with a persona
(`"You are a helpful assistant."`) and with an **empty** system content alike,
so it is the turn *role*, not any persona text, that matters.

## Root cause

`OrnithMoeModel::BuildPromptInputIds` built the frame by hand and skipped the
system turn (the Qwen template only emits one when the conversation provides
it). The Ornith-1.5 RL harness always injects a system turn, so the no-system
frame is out of distribution and collapses. `BuildConversationInputIds` was
unaffected only because the tool round-trip / product callers pass a system
message.

## Fix

`libs/lsxcommon/src/ornith_moe_model.cpp`:

- `BuildPromptInputIds` now builds `{system, user}` and renders it through
  `RenderQwenConversation(..., enable_thinking=false)` + `AppendQwenChatIds`,
  the same renderer the conversation path uses; the hand-rolled
  `AppendOrnithEncoded` helper is deleted.
- `BuildConversationInputIds` copies the conversation and prepends a default
  `{"system", "You are a helpful assistant."}` when the caller omitted it.

`tests/test_ornith_moe_selfcheck.cpp`: the real-weight case now asserts the
frame opens `[<|im_start|>, system, …]`, so a future regression fails loudly
without needing a human to read the generated text.

## Verification

- `LSX_TEST_ORNITH_MODEL_PATH=<vault> test_ornith_moe_selfcheck`: **PASS**,
  48 assertions / 3 cases, `logits_hash` identical across cycles.
- `test_ornith_moe_{model,loader,convert,kernels}` (synthetic): green.
- FSI/CODE/DET: still **SKIP** (no goldens; see open debt below).

## Open debt (not closed here)

- Ornith FSI/CODE/DET goldens remain absent: the MLX checkpoint uses
  `language_model.*` key names that HF transformers 5.15.1 does not load
  (recorded in `FSI_SELFCHECK_FINDINGS.md`), so the source cannot be run to
  capture a model-defined golden.
