# DeepSeek-V4.1-Flash — open items

### A2. `deepseek-v4.1-flash`: a cancelled request poisons the **next** request

- **Status: fix applied 2026-11-07, verification pending on the real weights.**
  The fix is in `deepseek_moe_v41_inference_pipeline.cu`: the first-token capture
  now happens BEFORE the abort break (so `ev.ttft` is always recorded when there
  is a token), `cudaEventElapsedTime` results are checked, and a request cancelled
  before its first token reports `ttft_ms = 0` while CONSUMING any pending error
  (not as an `error_msg` — `test_streaming_assert` requires `ok` for a cancel).
  The other half of the item, `ModelRoot()` pointing at the retired
  `/mnt/llm_models` path, is gone by construction: the shared driver resolves the
  root through the manifest's `vault_env` + `lsx_model_models_dir()`, so the case
  can no longer skip silently.
  **This item was unobservable for a while**, which is why it sat: the shared
  selfcheck driver dropped its cancel assertion during the manifest migration
  (§G5). It is restored, and the case now runs.
  Measured 2026-11-04 on real weights (294 GB tree in the vault):
  `LSX_TEST_DEEPSEEK_V41_ROOT=/mnt/fast_storage/huggingface/DeepSeek-V4.1-Flash`
  `./tests/test_deepseek_moe_v41_selfcheck` — the Paris case passes (3 cycles
  bit-exact, `logits_hash 0ab7077b529eaa3b`, top-1 11111), then the
  cooperative-cancel case fails: `mid-generation abort: compressor state reset
  failed` (27 assertions, 26 passed).
- The case had **never run**: `ModelRoot()` in
  `tests/test_deepseek_moe_v41_selfcheck.cpp` defaults to the retired vault
  `/mnt/llm_models/models/DeepSeek-V4.1-Flash`, so both cases SKIP silently on
  this host. Same class as a golden that was never captured — the target is
  green in CI because it never starts.
- **Root cause (measured**, with a temporary probe logging the entry
  last-error and the two `cudaEventElapsedTime` results in
  `libs/lsxcommon/src/deepseek_moe_v41_inference_pipeline.cu`**)**:
  1. A pre-set abort breaks on the first emitted token, and the
     `if (gs->Aborted()) break;` sits **above** the
     `if (!first_gen_captured && step == prompt_len - 1)` block, so `ev.ttft`
     is never recorded for that request.
  2. `cudaEventElapsedTime(ev.start, ev.ttft)` then returns
     `cudaErrorInvalidResourceHandle`. The result is unchecked, so the request
     still reports `ok=true` (which is why the cancel case passes and the
     failure appears elsewhere).
  3. CUDA's last-error state is **per thread** and is only cleared by a read,
     so the next request's first `cudaGetLastError()` returns it — and the
     first thing the pipeline does is `LaunchFillF32` on the compressor state,
     which then reports `compressor state reset failed`. The context is
     healthy: the very next launch in the same loop returns `cudaSuccess`, so
     this is a stale-error leak, not a fault (`cudaErrorInvalidResourceHandle`,
     not `700`).
- **Fix (small, but it needs its own verification run):** record `ev.ttft` and
  the `first_gen_*` fields before the abort break, check both
  `cudaEventElapsedTime` results, and point `ModelRoot()` at the vault
  convention (`lsx_model_models_dir()`, as the qwen tests do) so the case
  cannot skip silently again. Then re-run the cancel case plus the tool
  round-trip.
- **Why the class matters beyond this family:** any driver that runs two
  requests in one process and leaves an unchecked API error behind fails the
  second request at an unrelated site — and the shared abort gate
  (`tests/test_streaming_assert.h`) only protects the families that actually
  reach it.
- The FSI/CODE goldens for this model stay RED for the reasons already recorded
  in §Q (1/3 and 2/3 on the E4M3FN conversion); they are not affected by this
  item.

### Q. DeepSeek-V4.1-Flash FSI/CODE goldens committed model-defined; FSI 1/3, CODE 1/3 (2026-10-07, re-measured 2026-11-07)

The DeepSeek-V4.1-Flash source ships its own FP8/FP4 tilelang tensor-parallel
stack (target 2×80 GB) and cannot run locally, so FSI/CODE had always SKIPped.
Following the GLM-5.3-Flash method the model itself supplies the reference: the
DeepSeek-V4.1-Flash assistant answered the prompts (answers = the
`Expected Templates`) and those answers plus `encoding/encoding.py` thinking
framing are committed as the goldens
(`tests/fixtures/{fsi,code}/deepseek-v4.1-flash.json`). Both tests now RUN
instead of SKIP (a deliberate deviation from "commit only if the converted model
passes").

Both are RED on the E4M3FN NVFP4 conversion (`kOff`):

- **Re-measured 2026-11-07 on binaries rebuilt AFTER the `total_steps` fix**
  (`prompt_len - 1 + max_gen`, which removed an extra generated token): FSI **1/3**
  (01 pass, 02/03 fail), CODE **1/3** (02 pass), DET **2/3** (03 fails). Identical
  to the pre-fix grid, so the off-by-one changes the decode length and the Paris
  hash (`b759bc84b820b4de`) but not the parity verdicts. Worth recording because
  the *first* attempt at this measurement was taken on binaries built before the
  fix and reported the same numbers — right answer, no evidence. Rebuilding the
  three targets and re-running was the only way to tell.
- **FSI 1/3** — FSI-01: the thinking CoT does not finish within the 512-token
  budget, so the first-anchor-line extractor latches onto the truncated prompt
  echo `PV=CF*(1-e`; FSI-02 yields the initial-state echo
  `Balance_Alpha=100000000`; FSI-03 passes.
- **CODE 1/3, re-measured 2026-11-07** (this section said 2/3 and that
  "CODE-02/03 pass"): CODE-01: the model's CoT declares the balanced template
  `{"root":{"nodes":{"leaf":true}}}` "invalid", miscounts the braces, and emits
  an object missing its final `}` (extractor empty, `extracted=''`); CODE-02
  passes; **CODE-03 also fails** — expected `\b[0-9A-F]{4}\b`, extracted
  `\b[0-9A-F]{4}\b?`, one stray `?`. Confirmed on **both** binaries: the
  pre-refactor release build of 2026-11-06 17:09 (no shared driver in it) and the
  current debug build give the identical 1/3 with the identical two reds, so this
  is a stale record and not a migration regression. The extra `?` is the kind of
  near-miss a form-only extractor reports as a fail; it is not a truncation
  artifact (it survives a 512..2048 budget sweep with `ignore_eos`).

This is the `numeric-contract.md` §6 class (a reasoning/requantization divergence
below the format error), not a discrete kernel defect — the same profile as the
2026-09-27 sweep, unchanged by the E4M3FN codec.

**Budget experiment (2026-10-07):** raising the think-CoT budget from 512 to 2048
does **not** rescue either gate, and was reverted. FSI is budget-independent: the
FSI-01 decoded text is byte-identical at 512 and 2048 (the model echoes the
prompt and then emits nothing useful), so FSI-01/02 are not truncation artifacts.
CODE is *sensitive* to the budget under `ignore_eos`: at 2048 the model
degenerates past its answer and the last-`</think>` extractor flips CODE-03 from
PASS to a malformed `\b[0-9A-F]{4}b` (CODE-01 stays empty). The goldens keep
`max_new_tokens=512`, the value that captures the answer before the degeneration.

**Next step:** obtain a hardware-faithful reference from the release's own
`inference/` stack on 2×80 GB; if any case is a genuine converted-model
regression rather than a budget/extractor artifact, raise the precision of the
responsible tensor and re-convert.
