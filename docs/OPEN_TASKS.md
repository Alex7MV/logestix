# Open tasks

**Rule:** only genuinely open work lives here. When an item is resolved it
moves to `docs/open-tasks/<date>-<slug>.md` and leaves a row in the Archive
table at the bottom. Index: [`open-tasks/README.md`](open-tasks/README.md).

Nothing here is speculative — each entry carries the measured value or error
code that produced it.

---

## A. Real defects — GPU faults

### A1. `test_qwen_moe_kernels` + `test_ornith_moe_kernels`: `cudaErrorIllegalAddress`
- **Status: open, needs the 5060 Ti box.** Not reproducible on the Linux
  RTX 5090 host: both tests pass (865 assertions / 13 cases each) and
  `compute-sanitizer --tool memcheck` reports 0 errors for each. Either a
  Windows/5060 Ti-specific fault or one already removed by the Windows build
  repair. **Do not change a kernel until it reproduces on that box.**
- `cudaErrorIllegalAddress` (700). It is **not** the first thing that goes
  wrong: once a kernel faults, the context is poisoned and every later CUDA
  call in the process returns 700, so the bulk of the failure list is cascade.
  Triage from the *first* failing `REQUIRE`, not the last.
- Also carries a latency budget failure at `tests/test_qwen_moe_kernels.cu:388`
  (`us < 20.0`, measured 31.24) — that part is the GPU-keyed-budget question
  settled in the 2026-10-23 archive (`open-tasks/2026-10-23-doc-hygiene-and-non-defect-records.md`
  § B): the number is machine-specific, so it needs that card, not a patch.
- **Next step:** same workflow that resolved the GigaChat router.
  ```
  compute-sanitizer --tool memcheck --launch-timeout 120 <test>.exe
  ```
  Add `-lineinfo` to the owning object library to get `file:line`, otherwise
  the report only gives `kernel+0xNNN`. Note `TcGemmDispatch`
  (`libs/lsxcommon/src/lsx_tc_gemm.cu:891`) already rejects non-CC-12, and this
  GPU *is* CC 12, so the `major != 12` guard is not what is firing here.

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

## R. gemma-4-31b-it (dense): FSI-01 RED (the TC partial-M bug it was found by is closed, 2026-10-10)

The dense 31B member of the Gemma-4 family was ingested (`gemma_dense_*`,
format tag `logestix-gemma-dense`, `lsxforge --gemma4dense`). Real-weight
results: SelfCheck PASS (`Paris.`, bit-exact across stress cycles), CODE 3/3,
DET 3/3, FSI 2/3. Open items:

- **FSI-TEST-01 RED (stays red).** Source (golden):
  `PV=2500000*((1-exp(-0.0835*7))/(1-exp(-0.0835)))`; the converted model
  repeats the prompt template `PV=CF*((1-exp(-r*n))/(1-exp(-r)))`. The fork is
  at step 2 (after `PV=`), and it was re-measured in the engine on 2026-10-24
  after the MoE member's fix (pre-softcap, teacher-forced): src `2` 55.5 vs
  `CF` 42.75, converted `CF` 46.1 vs `2` 43.4.
  **The 26B residual recipe does not transfer and no partial placement fits.**
  Measured in the engine (each variant = a 65 s re-conversion + probe):
  `o`+`down` planes (the offline-simulation winner) -> margin `CF` +6.3 at
  22 GiB; `q,k,v,o`+`down` -> `CF` +10.4 at 25 GiB; all linears -> 33.7 GiB,
  OOM. Adding precision moved *away* from the source at this pair, i.e. the
  required upgrade is broader than the card. The offline HF simulation is only
  qualitative here: its all-4-bit baseline already sat on the source's side of
  the fork (its error is smaller than the engine's), which is why
  "`down`+`o` suffices" did not transfer. The dense residual plumbing was
  therefore reverted -- the tree is back to the 4-bit-only 17 GiB format.
  Getting this gate green needs ~8 bits on every linear (~33.7 GiB resident),
  i.e. a 48 GiB card or a different block format. Full trail:
  `open-tasks/2026-10-24-gemma4-moe-residual-plane-closed.md` §5.
- **Scalar prefill is a choice, not a workaround.** The shared TC M-tile
  defect this gate was originally found by is fixed (2026-10-10, archived);
  gemma keeps the scalar path because running it through TC moves the first
  token and breaks the tool round-trip —
  [`open-tasks/2026-10-10-neutral-moe-and-tc-mtile-closed.md`](open-tasks/2026-10-10-neutral-moe-and-tc-mtile-closed.md).
- **Perf: 47.3 t/s (perf) / 41.0 t/s (r3)** at 2048 tokens on
  `linux-release-znver4-aocc` (16.5 GiB resident; TTFT 337 ms, scalar prefill).
  Profile-guided decode work took it 40.6 -> 47.5: shuffle RMSNorm
  (11.4 -> 3.5 us/call), tied-head logits through the neutral warp-per-row GEMV
  (308 -> ~670 GB/s), unsplit attention at short context.
- **Rejected experiments (each measured, then reverted):**
  - *CUDA-graph replay of the decode step.* Implemented (capture-safe device
    position + KV-write kernel; `LSX_GEMMA_DENSE_CUDA_GRAPH`), capture
    succeeded, but it is SLOWER: 46.8 vs 53.8 t/s at 1024 and 47.6 -> 46.1 at
    2048. The decode is kernel-bound, not launch-bound, and a captured step
    needs a FIXED split count (the position advances in-graph), which
    over-splits short contexts. Default off. The device-pos/KV-kernel refactor
    it required was reverted with it.
  - *Warp-per-head decode attention.* 43.1 vs 47.6 t/s (only ~1k threads in
    flight). Reverted.
- **Comparison to the other dense model, Qwen3.8-27B** (same release tree,
  same prompt, `--max-gen 2048`): 54.9 t/s, but it is 48/64 *linear-attention*
  (GatedDeltaNet) layers, so its per-token work is far lower, and it reads a
  smaller pinned set; gemma-4-31B is 60 full/sliding-window attention layers
  and ~7% more weight bytes (16.1 vs ~15 GiB). Effective HBM bandwidth is
  comparable (gemma ~818 GB/s vs qwen ~884 GB/s). The ~13% gap is layer-type +
  bytes, not a defect; at `--max-gen 1024` they are at parity (53.8 vs 54.1).

## R2. Kimi-K2.7-Code — audit closure (2026-10-26)

The 2026-08 adaptation was re-audited against the checkpoint's own reference and
four blockers were found and fixed (details and every number:
`docs/open-tasks/2026-10-26-kimi-audit-verification.md`):

- **RoPE pairing** was half-split; the vendor de-interleaves before
  `rotate_half`, i.e. adjacent pairs. Cross-position dots differed by O(1).
- **Attention temperature** dropped the YaRN `mscale^2` factor (0.0722 vs 0.1447).
- **Encoder prepended a `[BOS]`** the reference tokenizer never emits, and the
  **stop id was `[EOS]` (163585)** instead of `config.json eos_token_id` =
  `<|im_end|>` (163586), so generation never cut at the end of a turn.
- **tiktoken merge reconstruction was unsound** (9/14 corpus cases diverged);
  the vocabulary itself is the merge table, and added tokens are matched
  literally. Two shared-pretokenizer defects came out of the same gate (Han vs
  the letter alternative, the `'s/'t/'re…` contraction suffix).

Tool-calling now uses the checkpoint's native grammar (stage 4b), the converter
classifies and counts ballast (`skipped_visual_tensors`,
`skipped_shape_tensors`, `mtp_*`) and fails on unknown tensors, and the model
was re-converted (23040 experts, 542 GB). Real-weight gates: SelfCheck Paris ×3
bit-exact; layer oracle relL2 0.0034 (layer 0) → 0.1226 (layer 60), cos ≥ 0.9925.

Open items for this family:

- **K1 — golden debt (FSI/CODE/DET).** Declared, not captured, and now
  **measured as infeasible on this host**: with `compressed-tensors` 0.19 and
  `tiktoken` 0.14 installed, `AutoModelForCausalLM.from_pretrained(...,
  device_map="cpu", attn_implementation="eager")` loads the config, applies the
  quantizer to 69 120 modules and then runs "Decompressing model": it
  materializes bf16 copies of the INT4 experts — 187 GB at 8 % of the module
  count, i.e. **~2.0 TB extrapolated against 750 GB of RAM**. The 581 GB source
  alone mmap-fits (page cache), the decompressed form does not. A vendor-stack
  capture (vLLM/SGLang `--tool-call-parser kimi_k2`, per the checkpoint's
  deployment guide) needs multi-GPU hardware, not this box. The three targets
  therefore SKIP (tree present, golden absent) and the rows stay in
  `kGoldenDebt`; the layer oracle is the parity instrument that *is* available
  (`docs/open-tasks/2026-10-26-kimi-audit-verification.md`). Do **not** derive a
  golden from the engine.
- **K2 — async decode overlap.** The per-step D2H + event wait is not a "park":
  the host needs the token id to stream it and to honour abort, so it is a real
  dependency. The old note about a "cross-cycle divergence on a fully-async
  decode" could not be reproduced after the numeric fixes
  (`LSX_KIMI_ASYNC_STEP=1` compares 6 cycles, identical hashes), and the
  diagnostic path itself is still host-serialized (`cudaStreamSynchronize`).
  Pipelining the readback one step behind is a throughput item, not correctness.
- **K3 — perf rows seeded 2026-10-26 (resolved, archived).** The 2026-09-14
  `perf` row (5.1 t/s) carried no evidence and predated the conversion; it was
  replaced through the harness with **`perf` 4.86 / `r3` 5.04 t/s**. The -5% is
  the numeric fixes changing the routed expert sets, not a regression (the old
  row had no protocol evidence to compare against). Numbers, power and TTFT:
  [`open-tasks/2026-10-26-kimi-audit-verification.md`](open-tasks/2026-10-26-kimi-audit-verification.md).
- **K4 — KV capacity.** `kKimiMaxSlots = 1024` against a 262 144 window; a
  long-context gate is out of reach on one 5090 (the KV cache is 512+64 bf16 per
  token per layer). Unchanged, documented.

## K. Golden-debt: the declared list, and why it is not a hard failure (2026-10-06)

Vector 2 of the three "will it recur" holes is now closed. `test_model_suite_guard`
carries a new case that pairs every `Suites()` row against `tests/fixtures/{fsi,code,det}`
and is machine-enforced in **both** directions:

- a parity golden that is missing **and not declared** in `kGoldenDebt` fails;
- a `kGoldenDebt` row whose golden now exists fails too, so the list cannot outlive
  the problem and rot;
- a `kGoldenDebt` row matching no live gap fails, so renames do not leave dead entries.

Current state: **39 declared debt entries** across thirteen model rows that have
none (or only some) of the three goldens. That number is the honest coverage figure, and
until now it was only obtainable by diffing two directories by hand.

The design choice worth recording: this is **declared debt, not a hard requirement**.
A hard requirement would paint nine families red on the next fast run and demand
capturing goldens for every one — including models with no local safetensors, which
is hours of work plus vault access we do not have here. Declared debt gets the
property that actually matters (a new family cannot be added quietly without
goldens, and the remaining gap is enumerable from test output) without pretending the
existing debt is smaller than it is. A missing golden stays a SKIP; what changes is
that it can no longer be invisible.

The last single-stage entry here, `granite-4.2-30b:code`, closed 2026-11-03
(its golden had existed since 2026-10-07; archived with the inference defect
found alongside it). The mechanism remains: a family can look fully wired across
all four parity stages and still have a hole in one of them.

**A rejected golden set (2026-10-25), kept here as evidence.** Three
`nemotron-3.5-lightning-30b-a3b-nvfp4` goldens were committed and reverted the
same day. They were not a source capture: the `input_ids` were
Qwen3.8-Flash-Next ids (up to **248069** against this model's **131072**-entry
vocabulary — `convert_ids_to_tokens(248045)` returns `None`, and decoding the
row with the Qwen tokenizer reproduces the prompt exactly), the `expected`
values equalled that family's goldens verbatim in all nine cases, and `source`
was a descriptive sentence rather than the model path both capture tools write.
The three gates therefore failed with an empty `raw` (out-of-vocabulary ids →
immediate turn close), and the commit also left `test_model_suite_guard` red
("debt row whose golden now exists" ×3). No fix is available locally: the
checkpoint is `modelopt 0.44.0rc5` `MIXED_PRECISION` with `W4A16_NVFP4` experts
plus an FP8 KV cache, and the installed transformers ships **no** modelopt
quantizer, so the source cannot be executed for an oracle; the bf16 passport is
not in the vault. The debt rows stay. The lesson matches the gemma-4-26b-a4b-it
case (archived 2026-10-24): a golden that is
not the source's own generation is not a parity oracle, and a golden with the
wrong tokenizer cannot even be a framing test.

## R3. DeepSeek-V4-Pro-0813 — ingestion landed, parity/perf pending (2026-10-26)

The family is registered end to end (convert -> load -> model -> kernels ->
selfcheck -> fsi -> code -> det -> perf), and the frozen passport is
61 layers / 7168 hidden / 384x3072 experts / 129280 vocab with the
`128,128,4,128,4,...` compression grid. Two items remain open:

- **P1 — SelfCheck is green on real weights (2026-10-26).** The 892.7 GB
  source converted cleanly in place (838 GB tree: 27 GB pinned + 811 GB
  experts, `deferred.mtp = 7009`; `incomplete 0`). `test_..._pro_selfcheck`:
  Paris x3 with an identical `final_logits` hash, streaming order exact, and
  both abort contracts green; the real-weight tool round-trip returned
  `get_weather {"city":"Paris"}`. Getting there exposed two non-obvious
  defects: the HC residual exceeds the FP16 range at depth (the source runs
  BF16/FP8), so `hc_pre`/`hc_head` now publish an **FP32** combined vector
  (`d_single_f32`) before the scale-invariant RMSNorm; and `LayerWeights::Free`
  skipped the separately-allocated shared-expert blobs (~4 GiB) on the
  arena-backed path, so a second `Load()` in one process OOM'd — both fixed.
- **P2 — parity goldens are template-derived and the gates now run (2026-10-26).**
  `tools/{fsi,code,det}_capture.py --template-golden/--template-only` frame
  each prompt with Pro's own `encoding/encoding_dsv4.py` (chat mode) and record
  the prompt's authored Expected Template as `expected`; `source` says the TP-4
  FP8 runner cannot execute here, so these are **not** source-parity goldens
  (same status as DeepSeek-V4.1). Verdicts: **FSI 2/3**, **CODE 3/3**,
  **DET 3/3** (3 runs each, deterministic). FSI-02 is RED for a real
  instruction-following reason: the prompt asks for the symbolic form and the
  model inlines the numeric constants
  (`((100000000*1.083541-0.0005)*83.152402-...)`); FSI-01/03 pass. That is an
  open model-behavior defect, not a kernel defect, and it is not "fixed" by
  loosening the golden.
- **P3 — perf is seeded through the harness (2026-10-26).** Running
  `test_perf_baseline "perf baseline: deepseek-v4-pro-0813"` on
  `linux-release-znver4-aocc` (RTX 5090, 2048-token protocol, one process per
  mode) wrote both rows: **perf 0.77 t/s** (TTFT 8987 ms, 114 W avg) and
  **r3 0.79 t/s** (TTFT 9023 ms, 116 W avg), 28.0 GiB resident. The number is
  host-store bound, as expected: every token streams 61 x 6 experts (~13.5 GB)
  from the 811 GB mmap tree; the 5-slot device cache (177 MiB, all the free
  VRAM the 26.7 GiB resident set leaves) contributes almost nothing. A pinned
  arena is not an option (822 GiB vs 750 GB RAM). This is the honest ceiling
  for this card, not a tuneable regression.

Residency note for the next operator: do NOT turn on
`LSX_DEEPSEEK_PRO_PIN=1` on this box. The expert tree is 822 GiB against
750 GB of RAM; the loader caps an opt-in pool at 64 GiB for exactly that
reason, and the default (mmap + `LSX_DEEPSEEK_PRO_MAX_OPEN`, 4096) is the
bounded path.

## Q. DeepSeek-V4.1-Flash FSI/CODE goldens committed model-defined; FSI 1/3, CODE 1/3 (2026-10-07, re-measured 2026-11-07)

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

## R4. MiniMax-M3 (~428B/23B active) — ingestion landed (2026-10-27/28)

Branch `feat/ingest-minimax-m3`. New family `minimax` (GQA + MiniMax Sparse
Attention indexer + sigmoid-router MoE, 60L / 6144H / 64Q-4KV / 128HD,
128 experts top-4 + ungated 3072 shared expert, layers 0-2 dense). Frozen
passport `minimax_moe_constants.h`; converter `lsxforge --minimax`; loader,
kernels, pipeline, model, native tool codec and all eight lifecycle stages
landed. Four guard registries extended (naming family token + tool-format
allowlist, `kPassports`/`consumers`/`kLifecycle`, `Suites()` + tool verdict,
perf `add_test` + `TEST_CASE`).

**Conversion (real, in place).** 796 GB BF16 source -> **229 GB** tree in
1 h 45 min: 10.5 GiB pinned (FP16 embed+head 4.9 GiB, NVFP4 attention/dense/
shared, F32 router + bias, FP16 norms) + 217 GiB routed experts (`57 x 128`
ExpertSchema files). `skipped_visual_tensors = 523` (vision tower + projector,
counted); no `mtp.*` in the tree. Expert pass is expert-major (w1/w2/w3 joined
across shards) so the source is never resident. `LSX_MINIMAX_SKIP_EXPERTS=1`
re-converts only the dense side (byte-identical index) when the expert store is
already on disk.

**Real-weight verdicts (RTX 5090, 32 GB, one process at a time).**
- SelfCheck: **PASS** — "The capital of France is **Paris**." ×3 stress cycles,
  `logits_hash 579a4a12cdedc279` stable (bit-exact), streaming order exact.
- CODE **3/3**, DET **3/3** (3 runs each, byte-identical `gen_tokens`).
- FSI **2/3** — FSI-01 RED: the prompt's Expected Template is
  `PV=CF*((1-exp(-r*n))/(1-exp(-r)))` and the model answers
  `PV=CF*(1-exp(-r*n))/(1-exp(-r))*exp(r)` (the annuity-**due** form, an extra
  factor). That is a template/form deviation of the same class as
  gemma-4-31b FSI-01 and deepseek-v4-pro FSI-02, not a near-tie flip: the
  golden is NOT rewritten.
- Goldens are **template-derived** (`--template-only` / `--template-golden`):
  the 428B BF16 source has no 32 GB-card execution path (the vendor stack is
  TP-8 over the full tree), so `source` says so in each golden. Same status as
  DeepSeek-V4.1/Pro.

**Defects found and fixed while landing it** (all pinned by tests):
1. The residual add ran `<<<1, 256>>>` for a 6144-wide row, so only the first
   256 hidden dims accumulated — the model produced token salad. Now
   `<<<(H+255)/256, 256>>>` (`test_minimax_moe_model` determinism covers it).
2. `cudaEventElapsedTime` on the never-recorded ttft event left a sticky
   `cudaErrorInvalidResourceHandle` that poisoned the NEXT request ("embed
   (invalid resource handle)"). Guarded; the abort case pins it.
3. An already-aborted request still emitted one token (the abort was polled
   only after `Emit`). Now polled before sampling: pre-set abort => 0 tokens.
4. Tokenizer bridge, two defects: `base_vocab_size` was computed over
   `key_to_id` (which already contained the added tokens) so `literal_ids` was
   empty and every control marker (`]~b]`, `]<]minimax[>[`) was BPE-encoded as
   raw bytes; and the pre-tokenizer was the legacy GPT-2 3-split, which splits
   CJK runs (the checkpoint uses the Llama-3/o200k pattern). Both fixed:
   `TokenizerSpecialPolicy.pretokenizer = 1` selects new `tokenizer_kind = 3`
   (`Llama3WordStream`, exact Lu/Ll/Lt case runs, no CJK split) and
   `base_vocab_size` is now the model-vocab rank space. `tokenizer_kind = 3`
   reuses the MERGE-RANK table (not tiktoken's id-order rule: HF BPE ids are
   not merge-ordered). Pinned by `test_minimax_moe_tokenizer_parity` — 14
   adversarial cases (CJK, CamelCase, contractions, multi-space, digits,
   markers) captured from the checkpoint's own tokenizer: exact.
5. Precision: `embed_tokens`/`lm_head` MUST stay FP16. At 4-bit the head's
   ~9% row error moves a logit by ~3 while this model's first-token margin is
   0.4; with the head 4-bit the answer was token salad, with the head FP16 the
   engine's top-5 reproduced the reference's ids and order. Recorded rule for
   the family (phi-4 / gemma-4 precedent).

**Numeric evidence (external oracle).** `tools/minimax_reference.py` is a host
re-implementation over the source safetensors (loads only the dense graph + the
routed experts actually selected, ~25 GB RAM) and
`tools/minimax_hf_layer_oracle.py` runs the CHECKPOINT'S OWN modeling code for
one layer on the meta device. Layer 0 agrees with the HF oracle to NVFP4
quantization error, and the per-layer cos/relL2 vs the source is 0.9998/0.034
at layer 0 -> 0.965-0.98/0.21-0.27 at layers 40-59 (the residual stream grows
~50x over 60 layers, which is what amplifies 4-bit body error). The reference's
greedy continuation is "The capital of France is **Paris**." — the same text
the engine produces.

**Perf (honest numbers, stage 4 debt).** Measured on this box: ~0.31 s per
step token-by-token (179-token prefill + 16 generated = ~1 min per selfcheck
cycle), dominated by pageable H2D expert staging (6.9 GB/token: 228 expert
stages; nsys: 78% of CUDA API time in `cudaMemcpyAsync`, 80% of kernel time
was the serial router before it was parallelized to one warp per expert).
Rows seeded through the harness: see `tests/perf/perf_baselines.json`
(`perf` / `r3`).

Open work, explicitly:
- P1. Perf: the decode step is PCIe/host-store bound. Real options, in order:
  overlap expert staging with compute (dual-stream prefetch, the neutral layer
  already has the machinery), raise the slot count with a pinned chunk pool,
  and only then consider quantization-side tricks. The canonical 2048-token
  protocol is recorded as a baseline so future work has a reference.
- P2. MSA beyond the block budget: up to `topk_blocks*block_size` = 2048 tokens
  every key block is selected, so the canonical protocol runs on the neutral
  coalesced attention kernel. Past that the two-pass masked kernel engages; it
  is correct but reads the whole KV (no sparse speedup yet).
- P3. FSI-01 stays RED (see above). Not a golden to rewrite.

## R5. Hy-Memory-30B-A3B (~30B / ~3B active) — ingestion landed (2026-10-30)

Branch `feat/ingest-hy-memory-30b`. New family `hy` (GQA + plain per-head
QK-RMSNorm + full 128-dim RoPE + sigmoid router with an `expert_bias` + one
gated shared expert; 48L / 2048H / 32Q-4KV / 128HD, 128 experts top-8 at
intermediate 768, dense SwiGLU MLP on layer 0). Frozen passport
`hy_moe_constants.h`; converter `lsxforge --hy`; loader, kernels, pipeline,
model, native tool codec and all eight lifecycle stages landed. Guard
registries extended (naming family token + tool-format allowlist,
`kPassports`/`consumers`/`kLifecycle`, `Suites()` + tool verdict, perf
`add_test` + `TEST_CASE`).

**Stage-0 findings that were load-bearing.** (a) `config.json` carries BOTH
`moe_intermediate_size` (1536, a vestigial default inherited from Hy3) and
`expert_hidden_dim` (768); the tensors are 768 wide, so the converter pins 768
and rejects a 1536-wide tree. (b) The vocabulary has **no CR byte token**
(`"\r"` encodes to no token at all in the reference) and no single-byte token
for the 13 UTF-8-illegal lead bytes → `TokenizerSpecialPolicy::
drop_unmapped_bytes` maps them to -1 and the encoder skips them. (c) Its 818
added tokens include **plain (special:false)** control markers, which the
reference `AddedVocabulary` matches literally; the kind-0 encoder now does too
when the vocab metadata carries `base_vocab_size` (only this family writes it,
so every existing family's tokenization is byte-identical — the full fast
suite stays green).

**Conversion (real, in place).** 60.1 GB BF16 source -> **17 GB** tree in
~6 s of expert pass + dense side: 1.5 GiB pinned (FP16 embed+head 0.94 GiB,
NVFP4 attention/dense/shared, router + bias raw, FP16 norms), 24 MiB router,
120 MiB shared, 15 GiB routed experts (`47 x 128` ExpertSchema files, 6016).
No vision tower and no MTP in the tree (`num_nextn_predict_layers = 0`), so
`skipped_visual_tensors = 0`. Expert pass is expert-major per (layer, expert)
across shards; `LSX_HY_SKIP_EXPERTS=1` re-converts the dense side only.

**Real-weight verdicts (RTX 5090, 32 GB, one process at a time).**
- SelfCheck: **PASS** — "The capital of France is Paris." (token 628), logit
  max 25.2172, `logits_hash 4472c7d335d0bf7d` stable over stress cycles;
  streaming order exact; cancel before step 1 publishes nothing.
- FSI **3/3**, CODE **3/3**, DET **3/3** (3 runs each, byte-identical
  `gen_tokens`), all against goldens captured from the **source safetensors**
  (`tools/{fsi,code,det}_capture.py` with `loader: "hy_v3"`). The source oracle
  is `tools/hy_v3_oracle.py`: the published checkpoint is per-expert
  (`mlp.experts.E.{gate,up,down}_proj`, `mlp.shared_mlp.*`,
  `mlp.router.gate`), which the installed `HYV3ForCausalLM` does not accept, so
  the oracle assembles the fused `gate_up_proj`/`down_proj` 3-D tensors and
  sets `moe_intermediate_size = expert_hidden_dim` before constructing the
  config. FSI-02's expected answer is the **numeric** form the source
  produced, not the prompt's symbolic template — source parity, not a template
  match (the opposite convention from DeepSeek-Pro/MiniMax, whose sources
  cannot run here).
- Tool calling (stage 4b) verified on real weights: the model answers the
  round-trip prompt with a native `<tool_calls>\n<tool_call>get_weather<tool_sep>`
  block and `ParseToolCalls` returns `get_weather {"city":"Paris"}`.
- Tokenizer parity **17/17** cases (`tools/hy_tokenizer_capture.py`) and chat
  framing **14/14** cases (`tools/hy_chat_framing_capture.py`: control prompt,
  FSI/CODE/DET bodies, multi-turn, tools, tool round-trip) reproduce the
  checkpoint tokenizer + `chat_template.jinja` id-for-id, including the
  Python-`tojson` tool block.
- Perf rows seeded through the harness at the canonical 2048-token protocol:
  **perf 86.69 t/s / r3 75.08 t/s**, TTFT 193.8 / 227.3 ms, 16.5 GiB resident
  (all routed experts resident; no expert cache, no host store).

Open work, explicitly:
- P1. Perf. ~2.0 GB of active weights per token (of which 0.5 GB is the FP16
  lm_head) against a 1531 GB/s probe is a **~1.3 ms / ~750 t/s roofline**; we
  run at 87 t/s, i.e. ~12% of it. The router-sync hypothesis was tested and
  REFUTED: the ids-streaming expert GEMV (`LsxExpertPtrRow` device tables, no
  per-layer D2H) landed and moved the row 86.7 -> 87.2 t/s, so the 47
  `cudaEventSynchronize` calls were not the bottleneck. What remains is the
  weight-streaming GEMV itself (48 attention + 47x3 expert-batch + shared
  launches per token, ~2 GB streamed at ~180 GB/s effective) — the next step is
  an nsys decode profile under `perf-tuning-a-model`, not more plumbing.
- P2. TTFT is token-by-token prefill (correctness first). A batched prefill is
  the next step if TTFT matters for the memory-writer workload.
- P3. `LSX_HY_DEBUG=1` dumps prompt ids + top-5 logits of the first decoded
  step (diagnostic only, stderr) for bisecting against the oracle.
## T. Mistral-Small-4-119B-2603-NVFP4 ingestion — later stages (2026-10-11)

### T.1 Re-ingestion and full lifecycle verification (2026-11-06)

Branch `feat/mistral-small-4-reingest`. The converted vault tree had been lost
(`<model>/logestix/` was absent while 24 other families still had theirs), so
every real-weight claim about this family was unverifiable. Re-ingested from the
intact 66 GB source and ran the whole lifecycle.

- **Re-converted** (`lsxforge --mistral`, 76 s): `logestix-mistral-nvfp4`,
  36 layers × 128 experts = 4608 expert arrows, 222 vision tensors stripped AND
  counted, tokenizer arrows written (131072 vocab / 269443 merges).
- **Defect found and fixed: the index carried no `vram` block.** Every other
  family writes one, and an absent block does not fail - `ParseVramBlock`
  returns false and the consumers (the placement plan via
  `IModel::VramRequirements`, `lsxcli --info`) silently saw an all-zero
  profile, which is indistinguishable from "this model needs no VRAM". The
  family never overrode `VramRequirements` at all, so even a tree with a block
  would have reported nothing. Fixed in `mistral_moe_convert`/`_model` with
  tests written first; re-converted tree now reports min 4 GiB / optimal 50 GiB
  / max 111 GiB at ctx 1048576. Two conversions of detail are load-bearing: the
  inventory is the tensors the *artifact* holds (stripped ballast and the
  activation-only `input_global_scale` companions excluded), and `is_nvfp4`
  stays null because the source is already NVFP4 - re-deriving a 4-bit size
  would halve every tensor a second time.
- **Gates, all on real weights** (AOCC Release tree, 2026-11-06):
  `test_mistral_moe_selfcheck` PASS, Paris ×3 with byte-identical
  full-vocabulary logits (`logits_hash cdfd7309bb141772`, top-1 6993);
  native tool round-trip green (`get_weather{"city": "Paris"}` → parsed
  `get_weather {"city": "Paris"}`); `lsxfabric --validate` loads in 1053 ms
  (re-measured 2026-11-07: 1161 ms — ±10% run to run, so read it as "about a
  second") with 4 715 087 872 B resident and 1398/4608 expert slots — both
  re-measured identically; convert/loader/model/kernels/tool_framing all green
  (`_model` is a synthetic fixture by design, so it pins wiring, not the
  checkpoint); fast suite 138/138 (written as 137/137 here until 2026-11-07 —
  `744f2c4c` added a guard target and updated the report's verdict but not this
  line. The count is stable across the migration: the pre-migration ctest oracle
  has 140 `fast` targets and today's tree has 140, with one swap — the naming
  guard out, `test_manifest_generator_selftest` in).
- **Perf re-seeded through the harness**: perf **45.77 t/s** (was 45.04),
  r3 **49.82 t/s** (was 41.57), TTFT 525/509 ms, DRAM probe 1532.73 GB/s
  recorded in the same run. `r3` moving up 20% is the strict path now
  benefiting from the same tree/commit; it is a measured row, not a claim.
- **The three gates were RUN on 2026-11-06 with template-derived goldens, and
  measured.** They do not pass, and the measurement is now a fact rather than a
  SKIP. Capture used `tools/{fsi,code,det}_capture.py --template-*` with the
  engine-exact frame (`BOS + [INST]…[/INST]`, no vendor system turn):
  **FSI 1/3, CODE 0/3, DET 0/3** (DET `deterministic=true tokens_ok=true` on all
  three prompts — three byte-identical runs each). The model is coherent and
  on-topic but off-format: FSI-03 returns exactly `Drawdown>Threshold`, FSI-01
  answers `PV=CF*(1-exp(-r*n))` (the annuity-due form, missing the geometric
  denominator), FSI-02 opens with prose. CODE opens every answer with an
  intro line ("Here's a valid JSON schema…") and CODE-03 degenerates into a
  repeated phrase. DET-02 reproduces the behaviour recorded for the ORIGINAL
  ingestion — it reasons *to* the answer ("The word that starts with the
  character 'C' and ends with the character 'C'") without emitting the token —
  which is a reproducibility signal across both conversions.
  **The goldens are still NOT committed**, for the reason the earlier decision
  gave: with a template-derived (not source-captured) expected value they assert
  nothing about parity that the Paris SelfCheck does not already cover, and
  three permanently-red targets train people to ignore red. The rows stay in
  `kGoldenDebt`; what changed is that the debt is now measured.
- **Two capture-tool defects were found and fixed while running them.**
  `tools/fsi_capture.py` had no `chat_template` override for this model (code
  and det had it) AND never applied `opts["chat_template"]` at all — the first
  FSI run therefore used the checkpoint's own template, which injects
  `default_system_message` when no system turn is present, and all three prompts
  answered "It seems like you're asking about a specific dates, you can use
  tools to fetch…" (0/3, a pure framing artefact). With the override applied and
  honoured the same three prompts answer on-topic, so the 0/3 was the harness,
  not the model.
- **Why no source-faithful golden: the reference stack cannot consume this
  checkpoint.** When the debt was declared the source was "unrunnable"
  (`params.json`, no `config.json`, no `model_type`). That part is now stale —
  `transformers` 5.15.1 ships `Mistral4ForCausalLM`, its defaults match this
  checkpoint exactly, and a synthesized `config.json`
  (`tools/mistral_small4_oracle.py::mistral_small4_config_dict`) makes
  `AutoConfig` construct and `from_pretrained` read all 13 shards. Two rope keys
  are load-bearing in that synthesis and each fails differently: without
  `partial_rotary_factor` the rotary embedding is built at the full head width
  (128) against a 64-wide `q_rot` (`size of tensor a (32) must match b (64)`),
  and without `llama_4_scaling_beta` the Llama-4 logit temperature multiplies
  `None` (`unsupported operand type(s) for *: 'NoneType' and 'Tensor'`).
  **The real blocker is the tensor naming and expert layout, not the config.**
  The consolidated checkpoint names are `layers.N.attention.wq_a`,
  `layers.N.attention_norm`, `tok_embeddings`, `output`, `norm` and **one index
  per expert** (`layers.N.experts.E.w{1,2,3}`), while the class expects
  `model.layers.N.self_attn.q_a_proj`, `model.layers.N.input_layernorm`,
  `model.embed_tokens`, `lm_head`, `model.norm` and the **fused stacked**
  `model.layers.N.mlp.experts.{gate_up_proj,down_proj}`. A plain
  `from_pretrained` therefore reports **50 missing / 3118 unexpected** keys and
  silently runs a randomly-initialized 119B model; its forward pass then emits
  NaN logits (first NaN at `model.layers.0.mlp.shared_experts.down_proj` with a
  zero input — garbage weights, not a numeric defect). `load()` in the new tool
  REFUSES that model (`allow_missing=False` by default) precisely so nobody
  captures a golden from random weights. Closing the debt needs a
  name+repack mapping (the vendor `mistral-inference` loader is the reference
  implementation, and the expert unfusing is the hard half), or a
  de-consolidated bf16 HF export, at which point the capture scripts need no
  change beyond `loader: "mistral_small4"` and the deletion of the
  `unrunnable` override in `tools/{fsi,code,det}_capture.py`.
- **The checkpoint's own data was cleared while investigating that.** A host
  dequant of `layers.0.shared_experts.w2` gives min/max ±0.277 with
  `weight_global_scale` as a DIVISOR (independently confirming the converter
  rule fixed on 2026-10-12), and all 202 899 456 E4M3 block-scale elements of
  layer 0 contain **zero** NaN. So the NaN was the reference stack, never the
  weights — which is a positive statement about the converted tree.

- **Branch** `feat/mistral-small-4-119b`; design
  `docs/superpowers/specs/2026-10-11-mistral-small-4-119b-ingestion-design.md`.
- **Landed (verified, green):** stages 0–5 + 4b. Stage 1 converter
  (`mistral_moe_convert.{h,cpp}`, `lsxforge --mistral`, synthetic `convert`
  test); stage 3 loader (`mistral_moe_load.{h,cpp}`, synthetic loader test);
  stage 4b tool codec (`mistral_moe_tool_format.{h,cpp}`, framing test); stage 5
  MLA+MoE decode (`mistral_moe_inference.{h,cu}` + `mistral_moe_model.{h,cpp}`,
  registered `mistral-small-4-119b`), kernels + model tests green. Frozen
  passport `mistral_moe_constants.h`. Four guard registries extended
  (`test_model_independence_guard` `kPassports`/`consumers`/`kLifecycle`,
  `test_model_suite_guard` `Suites()` + tool verdict + golden debt,
  `test_perf_baseline_logic` mid list, `test_code_naming_guard` family token).
  Fast suite 93/93; `test_mistral_*` (convert/loader/tool/kernels/model) green.
- **Real-weight conversion: done.** `lsxforge --mistral` converted the vault
  tree in place (`<model>/logestix/`, 66 GiB, 4608 experts, 222 vision tensors
  stripped, tokenizer arrows written) in ~75 s. The loader then loads it
  (70 s: dense BF16 graph 4.7 GiB + 56 GiB pinned expert store). Two real bugs
  were fixed on the way:
  1. `weight_global_scale` is a **divisor** (`w = nibble*scale/gs`), not the
     repo's multiply — the converter now stores `1/weight_global_scale`.
  2. A NaN logit on finite real inputs first looked like the neutral
     `RunMoERouterTopK` seq==1 dots path. It was not: the NaN was that FP16 SwiGLU overflow
     arriving in the *next* layer's `d_post`, which is the router's input. With that fixed the neutral device router runs this family
     at its real geometry (E=128/H=4096/top-4) in both numeric modes and the
     host workaround is gone. The investigation did surface a genuine coverage
     gap — the seq==1 decode router had no reference test at all — now pinned
     across six geometries in `test_qwen_moe_kernels`. Details:
     [`open-tasks/2026-10-12-mistral-small-4-decode-defects.md`](open-tasks/2026-10-12-mistral-small-4-decode-defects.md).
- **The decode defects found on real weights, five of them plus three
  follow-ups (2026-10-12) — archived.** Each was isolated against a host NVFP4
  reference built from the source safetensors, and each is pinned by a
  mutation-checked regression case. The headline: the shared expert was run
  through the BF16 GEMV (44x under-contribution, `rms 0.0080` vs `0.3495`), the
  real NaN was an FP16 SwiGLU overflow at layer 32 (`|silu(g)*u| = 2.5e5`
  against the 65504 ceiling), the residual itself reaches ~2.4e5 so `d_h` is
  FP32, `~MistralSession` leaked the whole dense graph per request, and the
  converter's `"model"` field normalised to a name the registry could not
  resolve by path. Two attributions were wrong before they were right: the
  "neutral router NaN" was that FP16 overflow downstream, not the router (the
  investigation did expose a real coverage gap — the seq==1 decode router had no
  reference test at all — now pinned across six geometries), and the layer-4
  "outlier" was a genuine massive-activation expert, confirmed against the host.
  Full trail:
  [`open-tasks/2026-10-12-mistral-small-4-decode-defects.md`](open-tasks/2026-10-12-mistral-small-4-decode-defects.md).
- **Still open — the three FSI/CODE/DET golden gates, with the capture now
  understood.** (The old heading said "the four S* golden gates"; there is no
  S1..S4 scheme anywhere in this file, the body and
  `test_model_suite_guard.cpp` both enumerate exactly these three, and the body
  already said "three permanently-red gates" -- the count in the heading was
  the only stale part.)
  Capture tooling improved first: `tools/fsi_capture.py` gained `--template-only`
  (the source is unrunnable), and `det`/`code` gained a `chat_template` override
  in `OVERRIDES`. Findings:
  1. **The source cannot run.** `AutoConfig.from_pretrained` fails — the
     checkpoint has `params.json`, no `config.json`, no `model_type`, and
     NVFP4-packed experts. So a template-derived golden is the only option, and
     it cannot establish *parity*; it can only assert the converted path's own
     answers.
  2. **The vendor default system prompt suppresses every answer gate.** It
     instructs "when the user's request requires up-to-date or specific data, you
     must use the available tools". With it, all nine prompts answer "It seems
     like you're asking about a specific date, you can use tools to fetch ...".
     The model is *complying*; the engine's own `RenderText` does not inject
     that turn, so capturing with a `chat_template` override (BOS + `[INST]`...,
     no system turn) is what matches what we run.
  3. **Re-framed, the converted model is coherent but off-format, and the two
     numeric modes disagree on these prompts.** DET determinism PASSES
     (`runs=3 deterministic=true`, byte-identical) and answers are on-topic —
     DET-02 correctly reasons "The word that starts with the character 'C' and
     ends with the character 'C'" toward `CATASTROPHIC` — but no prompt emits the
     canonical token/formula, several collapse into repetition loops, and the
     auto and bit-exact modes diverge (auto returns `.`, bit-exact returns a
     full sentence) on DET-01.
  **Decision: the goldens are NOT committed.** Shipping three permanently-red
  gates trains people to ignore red, and with the source unrunnable a
  template-derived golden asserts nothing about parity that the Paris SelfCheck
  does not already cover more cheaply. FSI/CODE/DET remain declared debt in
  `test_model_suite_guard`.

- **Unblocked 2026-10-23: the operator will supply a runnable source.** The
  decision "template-derived or nothing" is retired; the goldens are to be
  captured from a model that actually runs. Two artefact forms are acceptable,
  and the amount of code each needs is known:

  1. **An HF-format checkpoint** (`config.json` with `model_type`, bf16
     safetensors, tokenizer) - e.g. a de-consolidated bf16 export. **Zero code
     changes**: `tools/{fsi,code,det}_capture.py` already fall through to
     `AutoModelForCausalLM`, and the `mistral-small-4-119b` override exists only
     to mark the *consolidated* layout as unrunnable.
  2. **The consolidated NVFP4 checkpoint plus the vendor stack**
     (`mistral-inference` + `mistral_common`, neither installed here). Needs a
     `loader: "mistral"` branch in each capture script - the `qwen4_exp` and
     `qwen3_5` branches are the precedent (load, `eval()`, expose
     `device`/`generate`), roughly 15 lines each. The golden's `input_ids` must
     still come from this repo's tokenizer arrows (`tokenizer.json` +
     `chat_template.jinja` are present in the vault), so a vendor tokenizer
     difference cannot silently move the prompt.
  3. **Not acceptable: the converted model's own answers.** That is
     self-certification (`docs/OPEN_ENDEDNESS.md`) and the reason this gate was
     left SKIPping rather than shipped red.

  Acceptance, all three stages: the golden's `source` field names a real model
  run (not "prompt-template"), the `"unrunnable"` override is deleted from
  `tools/fsi_capture.py` in the same commit, and
  `test_mistral_moe_{fsi,code,det}_selfcheck` run (not SKIP) - red or green is
  then a real finding either way. `code_capture.py`/`det_capture.py` carry the
  same override block, so the deletion is three one-line edits.

## S. Arrow IPC residuals (2026-11-01)

Nothing here is a correctness defect: the audit closed A-01..A-07 and the
full-vault run (29 trees, 310 158 files) is green. These are measured-cost
decisions, not yet-worth-it work.

### S1. Multi-expert per-file IPC packing — measure first
- **Open, instrument-first.** Packing N experts into one IPC file (N record
  batches, `ReadRecordBatch(i)`) divides inode/mmap/open/footer churn by N
  (~23k expert files per large tree) and changes nothing about the payload
  bytes read. It requires a format change plus re-conversion of all 29 trees
  (~4 TB) and pack-granular LRU bookkeeping.
- **Before any of that:** measure the `ExpertStore::ParseFile` share of the
  per-token cost on a high-miss model (DeepSeek-V4-Pro streams ~13.5 GB/token;
  Kimi does 277 misses/token). Migrate only if parse is material (>=10-20% of
  the step); for resident families it is already a one-time load cost.

### S2. Write-path wall-time A/B
- **Open, low value.** The copy elimination is pinned by a deterministic
  tracked-pool budget (peak 2x -> 1x payload; 0 growing reallocations), which
  is stronger than a noisy timing. A wall-time A/B needs a second build of the
  pre-change commit in a worktree; do it only if a converter run looks
  memcpy-bound.

### S3. Windows: atomic artifact writes
- **Open, platform-gated.** `WriteRecordBatchToFile` writes a temp file and
  renames; on Windows `std::filesystem::rename` cannot replace an existing
  file, so a remove-then-rename fallback is compiled under `_WIN32`. It has
  never run on the Windows preset (this host is Linux).

### S4. Shared-expert size contract is not validated
- **Open, hardening.** `SharedExpertSchema` carries no encoding tag. Measured
  reality: DeepSeek-V4-Pro/V4-Flash store shared experts as raw F8_E4M3 with
  E8M0 128x128 tile scales (32 768 elements/scale), the NVFP4 families as
  per-16 E4M3. The schema-level check was removed on 2026-11-01 because it
  rejected the real trees; only offsets/ranges/global-scale are validated.
  A truncated shared-expert payload is therefore caught late (or not at all).
- **Fix shape:** write the shared-expert encoding/block into
  `model.index.json` (e.g. `"shared_expert": {"encoding": "nvfp4", "scale_block": 16}`
  or `{"encoding": "fp8_e8m0_tile", "tile": 128}`) and have
  `load_shared_experts_metadata` / `LoadSharedExpertBatchViews` enforce it.

### S5. Routed-expert proportional truncation
- **Open, hardening.** `ExpertSchema` has no `shape`, so
  `CheckExpertScaleCount` infers `numel = packed*2` from the payload itself.
  It rejects an inconsistent truncation (packed and scales disagree) but not a
  *proportional* one (both halved). The loader knows the frozen geometry
  (`I`/`H` per family) and could cross-check `packed_bytes == ceil(I*H/2)`.
- **Fix shape:** per-family geometry check in the loader (no format change), or
  store the logical shape in the expert record (format change).

### S6. Full-vault audit is manual
- **Open, tooling.** `test_arrow_tree_integrity` skips without
  `LSX_ARROW_AUDIT_ROOT` (a SKIP is not a pass), and the exhaustive
  `LSX_ARROW_AUDIT_FULL=1` run (29 trees, ~4 TB, 783 s) exceeds the `convert`
  label's 600 s cap, so it cannot live in CTest as-is. `lsxforge --verify` runs
  the same verifier but is a manual post-conversion step.
- **Fix shape:** a nightly/job that mounts the vault and runs the full verifier,
  plus wiring `--verify` into the conversion wrapper so a converted tree is
  never published unverified.

### S7. The audit-matrix coverage rule is keyed off a hand-written list
- **Open, instrument.** `test_arrow_usage_guard` enforces R-A8 ("every adapted
  model has a row") as `by_model[model] == 1` for each entry of `kModels` plus
  `rows.size() == kModels.size()`. Both sides are hand-maintained, so a model
  present in **neither** is invisible and the rule silently passes over it —
  which is exactly what happened to `Kimi-K3.1-Pro` until 2026-11-05 (added in
  `0f919929`, after the variant had already landed with real-weight gates).
  The finding cost one audit, not a defect: the guard was not wrong about the
  rows it knew.
- **Fix shape:** derive the model set from a machine-checked registry instead of
  restating it — `test_model_suite_guard.cpp::Suites()` already enumerates every
  ingested model and is extended in the same commit as the model files, so the
  Arrow guard could assert `kModels ⊇ Suites()` (name mapping aside) and stop
  depending on a second list. Until then: adding a model means touching both,
  and the audit's own coverage claim rests on the list being complete.

## U. CUDA 13.5 adoption — landed, one measured negative (2026-11-05)

Toolkit 13.4.92 → **13.5.37** (driver stays 13.2). Design and the full
measurement: `docs/superpowers/specs/2026-11-05-cuda-13.5-adoption-design.md`,
plan `docs/superpowers/plans/2026-11-05-cuda-13.5-adoption.md`.

**Landed.** (a) A build-toolkit **stamp** (`lsx_cuda_toolkit_stamp.txt`) — the
tree really was half-converted: 1250 of 1460 objects were 13.4 headers linked
against a 13.5 runtime and every build step succeeded, because CUDA's headers
carry *package* mtimes older than the objects. Configure now refuses the swap and
`test_cuda_toolkit_guard` (moved to label `fast` — it was the only instrument that
saw it, and `ctest --preset linux-debug` never ran it) catches a mixed tree.
(b) Device codegen arch lives in ONE expanded place: 30 test targets had their own
`--generate-code=arch=compute_120,code=sm_120` literal, so the arch was in 31
places and would have stayed non-family on any bump. (c) The AOCC preset pinned
only the C++ compiler, so a fresh release configure died in uriparser with
`-Qunused-arguments`; `CMAKE_C_COMPILER` is now AOCC's clang.

**U1 — REJECTED on measurement: family-specific targets are neutral here.**
`sm_120f` really does change the SASS — the plain build compiles the NVFP4 decode
to `F2FP.BF16.F32` (FP32 emulation), the family build to
`F2FP.BF16.E2M1.UNPACK_B` (384 in the gemma TU alone) — and the sm_120f cubin
runs on the 13.2 driver. It does not move the wall clock: qwen3.5-9b, 2048-token
protocol, GPU exclusive, three runs per arm gave control 171.7/171.7/171.79,
family-arch 171.71, family-arch + hardware decode 171.70/171.70/171.70. The
spread inside the control arm (168.99 on its first cold run) exceeds any
difference between arms. Cause: every NVFP4 decode path in this repository is
weight-streaming bound — the GEMV waits on HBM/page cache, not on the converter,
which is §R17 P6 seen from the other side. The numeric half was removed rather
than kept with tests behind it; `-DLSX_CUDA_ARCH=120f;121f` remains a
one-variable experiment for a compute-bound case (a long prefill is the only
plausible one). **Not worth re-deriving** unless the decode stops being
byte-bound.

**U2 — free win, attributed.** qwen3.5-9b `r3` 147.00 → **170.00 t/s** (harness
row) on the same weights. This is the 13.5 toolkit, not our code: the control arm
shows it. Other families' `r3` rows predate 13.5 and will re-seed upward on their
next harness run — do not read those as regressions.

**U3 — driver re-probed after the 2026-11-05 update: nothing opened that helps.**
The driver moved `595.99.02` (API 13.2) → **`615.71.09` (API 13.4)**; the toolkit
did not change (13.5.37). The runtime is therefore one minor ABOVE the driver,
which is supported by CUDA's minor-version compatibility but means 13.5-only
APIs are absent at runtime. Measured delta of `libcuda`'s exported `cu*` surface:

| item | 13.2 driver | 13.4 driver |
|---|---|---|
| `cuMemGetLocationInfo` | absent | **present** |
| `cuGraphAddNode_v3` | absent | **present** |
| `cuDeviceGetCliqueInfo` / `GetCliqueCount` | absent | **present** |
| `cuTensorMapSetAllocationClass` (TMA large class) | absent | still absent |
| `cuGraphAddMemcpyBatchNode` / `...NodeGetParams` | absent | still absent |
| `cuMemsetBatchAsync`, `cuFabricCpyBatchAsync` | absent | still absent |
| `cuMemcpyBatchAsync` (`_v2`), `cuCheckpointProcess*` | present | present |

None of the three that opened is a lever here: the clique API is multi-GPU
fabric, `cuGraphAddNode_v3` adds graph node kinds we do not need, and
`cuMemGetLocationInfo` answers "which device backs this UVa page" for a store we
map through the page cache. **Correction to what this entry said before:** the
memcpy-batch **graph node** was described as "the shape §R17 P6 needs" — that was
wrong. P6 needs batched copies overlapped with compute, and the stream API
`cuMemcpyBatchAsync` has been there all along; only the graph-node convenience is
driver-gated. P6 is **not** blocked on a driver update.

Two things that used to be inferences are now measurements on a driver that can
answer:

- **Oversized shared memory buys nothing on this card, now measured.**
  `MAX_OVERSIZED_SHARED_MEMORY_PER_BLOCK` resolves and equals
  `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN` (101 376) — the attribute is queryable on
  this driver, and there is no headroom above the ordinary opt-in limit.
- **Locality domains are useless here.** `LOCALITY_DOMAIN_COUNT = 1`
  (`..._MULTIPROCESSOR_COUNT = 170`): one domain, nothing to partition across.

So §U remains a record, not a work item, and the only real lever for the
host-store-bound families is still §R17 P6.

## R14. Mistral-Large-4-1T-A52B (1T-A52B) — ingestion landed, four gates green on template goldens (2026-11-02)

Branch `feat/mistral-large4-ingest`. A new `mistral` variant (`large4`,
`kType = mistral-large-4-1t`, format tag `logestix-mistral-large4-nvfp4`):
72L / 6144H / 64 MLA heads without Q-LoRA (kv_lora 512, qk_nope 64 + qk_rope
64, v 128), 128 routed experts top-4 + one gated shared expert, a dense layer 0,
and five components no other family has: **hybrid RoPE/NoPE** (every 4th layer
is NoPE and its 64 rotary dims become plain key dims), **LeRoPE** (a learned
per-frequency `inv_freq *= exp(alpha)` on top of YaRN θ=1e4 / factor 2048 /
original window 512), **ssmax** (per-head, position-dependent query scaling
`s[h]*log(pos+1)+b[h]`), a **DSA Lightning Indexer** (32 heads × 128, top-2048,
44 full layers, 28 reusing the previous full layer's mask) and **cross-layer
KV sharing** (layers 39,43,…,71 have no `kv_a` and read layer 35's latent).

**Measured on real weights** (988 GB FP8-E4M3 block-128 source → 555 GB
NVFP4 tree, 16.1 GB pinned + 547 GB mmap-streamed experts; RTX 5090, CUDA 13.4):

| gate | verdict |
|---|---|
| SelfCheck | PASS ×3, bit-exact (`logits_hash=747bf131f1ac557f`, top-1 `Paris` 13125) |
| tokenizer parity | 50/50 against the checkpoint's own `tokenizer.json` (new `tokenizer_kind` 12: o200k case runs, RIGHT-grouped digits) |
| tool round-trip | PASS (`get_weather {"city": "Paris"}` through the native `[TOOL_CALLS]name[ARGS]` codec) |
| FSI | 3/3 (template-derived golden) |
| CODE | 3/3 (template-derived golden) |
| DET | 3/3 (byte-identical token streams over 3 runs) |

**Open, explicitly.**

- L1. The first CODE/FSI runs read as red for the *wrong* reason and the
  instrument gap is now closed: this model reasons INLINE (its `[THINK]` /
  `[/THINK]` markers are special tokens, so the decoder drops them), and the
  extractors had no boundary to work with. CODE-TEST-03 looked like a duplicated
  regex and FSI-02 like a truncated equation; both were lines from the reasoning
  chain, and both final answers are correct. The fix is opt-in and default-off
  (`answer_start_token_id` cuts by the special id in all three goldens,
  `prefer_last_answer` in FSI) — no other family's verdict changed. With it,
  CODE is 3/3 and FSI 3/3. Kept here because the trap generalizes: a
  reasoning model whose markers are special tokens needs an id-based cut, not a
  text-based one.
- L2. The parity goldens are **template-derived**: the 988 GB FP8 source is a
  multi-node FP8 runner that cannot execute on this 32 GB card, so no
  model-defined golden can be captured. `expected` comes from the prompt set's
  own Expected Template; the source is stated in the golden JSON.
  `tools/mistral_large4_template_goldens.py` reproduces them. This is the
  sanctioned deviation (GLM-5.3-Flash / GigaChat / DeepSeek-Pro precedent), not
  a pass-by-construction: the model must still produce the template's answer.
- L3. FSI-02 originally read as red for the *wrong* reason: the model reasons
  inline (no `[THINK]` markers, so no boundary survives the decoder's
  special-token drop) and `SelectAnswerLine` picked the reasoning line
  `Balance_Alpha = 100000000`. Two opt-in instrumentation knobs were added
  (`answer_start_token_id` in all three goldens, `prefer_last_answer` in FSI) —
  default-off, so no other family changed. With a 2048-token budget (the chain
  is ~700-1000 tokens) it answers the Expected Template exactly.
- L4. DSA beyond `index_topk` (2048) is implemented (indexer key cache,
  exact device top-k, masked gather attention) and covered by kernel-level host
  references, but the canonical gates run at ≤2048 positions where every key is
  selected, so the masked path has no real-weight verdict yet. The 1M context
  the card cannot hold (latent KV ≈ 84 GB at 1M; 0.3 GB at 2048) is likewise
  unmeasured.
- L5. Perf: measured with the AOCC release harness at the canonical 2048-token protocol: **1.04 t/s (perf) / 1.03 t/s (r3)**, TTFT 5.7 s (`tests/perf/perf_baselines.json`), host-store bound
  (4 experts × 71 layers × 3 projections of 6144² at ~0.53 B/param ≈ 17 GB
  streamed per token) plus one slot-table D2H per MoE layer (the same contract
  as Mistral-Small-4). See §R14 perf line.
- L6. The expert device cache is 120 slots of 9216 on this card (16.1 GB pinned
  leaves ~13 GB); the working set per token is 284 expert touches, so the
  streaming is the throughput ceiling.

## M. Test-registration migration (CMake boilerplate, 2026-11-03)

`cmake/LsxTestHelpers.cmake` now provides `lsx_add_test()` /
`lsx_add_cuda_test()` / `lsx_copy_arrow_dlls()`; `LSX_CUDA_ARCH` is the single
device-codegen arch and `lsx_dep_includes` is the only dependency-include
propagator (no `${CMAKE_BINARY_DIR}/_deps/...` at a call site).

Done in the first pass (verified by `ctest --show-only=json-v1` property
equality: 493 tests, 0 name/property diffs, plus a full build and the `fast` +
kernel-inference suites): 212 Windows Arrow-DLL blocks replaced by
`lsx_copy_arrow_dlls`; 130 `_deps` call-site paths removed (37 include blocks
now link `lsx_dep_includes`); the 14-test qwen CUDA loop and the 10 guard tests
use the helpers.

**Remaining, mechanical:** ~390 host registrations still spell out
`add_executable` / `target_link_libraries` / `add_test` /
`set_tests_properties`. Migrate them family by family in the same commit as
their next change, keeping the JSON property diff at zero (the helper already
covers LINK/INCLUDE/DEFINE/LABEL/TIMEOUT/SKIP/ENV and the DLL copy; add a
property to the helper only when a target genuinely needs one, e.g.
`RESOURCE_LOCK`).

## G1. `RESOURCE_LOCK "gpu"` does not serialise two ctest processes (2026-11-07)

**Status: open — a documentation/process gap, not a code defect.**

Every GPU target carries `RESOURCE_LOCK "gpu"`, and `AGENTS.md` states the rule
"one GPU workload at a time". Both are true *inside one ctest invocation*: CTest
implements a resource lock as a scheduler constraint among the tests of a single
`ctest` run. Two concurrent `ctest` processes do not coordinate at all — there is
no lock file, so both schedules believe they own the GPU.

Measured on 2026-11-07: a background parity sweep (`glm_moe_flash`, `gigachat`,
`deepseek_v41`, `minimax_moe_flash`, `mistral`, `deepseek`) was holding
**30 272 MiB of 32 607** while a second ctest ran `test_granite_switch_*`. The
DET gate then failed on DET-03 with `deterministic=false`, `extracted=''` — an
inference error under allocation pressure, not a numeric divergence. Re-run with
the card otherwise idle, DET-03 passed (`deterministic=true`,
`extracted='BIT_ZERO'`).

The failure mode is worse than a red test: the reporter said
`deterministic=false`, which points at a numerics/instability bug, and the actual
cause was VRAM exhaustion in a *different process*. That half is fixed — the DET
reporter now prints `inference-error error='…'` when `Infer` returns not-ok,
instead of folding it into the determinism verdict
(`libs/lsxcommon/src/det_selfcheck.cpp`; FSI/CODE already carried the per-case
error, and the shared gate driver now attaches per-case verdicts to its assertion
too).

What is **not** fixed, and is a decision rather than a patch:

- A cross-process lock would have to be a file (`flock` on a path in
  `lsx_cache_dir()`), taken by every GPU target before it allocates. That is a
  real mechanism, but it also means a crashed test leaves a stale lock, and every
  target would need to opt in — worth doing only if unattended parallel runs
  become normal.
- The cheap alternative is the one now written into `AGENTS.md`: a GPU-labelled
  run starts only after `nvidia-smi --query-compute-apps` is empty. That needs no
  code and no lock lifecycle, and it is what the human operator already does.

Do not "fix" this by removing `RESOURCE_LOCK`: within one invocation it is
load-bearing (the perf sweep would otherwise run concurrently with the gates in
the same `ctest -j`).

**Fixed on the same day, and the numbers that forced it.** The lock was not
merely per-process, it was also mostly *absent*. `AGENTS.md` claimed "every perf
target takes `RESOURCE_LOCK "gpu"`"; the configured tree said otherwise:

| label | locked before | locked after | total |
|---|---|---|---|
| `inference` | 54 | **289** | 297 |
| `perf` | 8 | **37** | 44 |
| `convert` | 8 | 36 | 38 |
| `fast` | 10 | 10 | 140 |

`STAGE_LOCKS` in `tools/gen_model_manifest.py` was empty for every stage except
where the hand-written CMake happened to set a property, so 263 of 279
GPU-labelled targets ran unprotected. Every model stage allocates on the GPU
(convert quantises there, load uploads, model/kernels/selfcheck/fsi/code/det/perf
decode), so the table is now `"gpu"` for all nine stages, with one exception:
`fast` means host-only by definition and keeps the lock off — otherwise a
140-target fast sweep would serialise on a lock it never needed.

Reproducing the change against the pre-migration oracle:
`--diff-cmake` reports **292 differences, and every one of them is this field**
(284 `resource_lock: null -> "gpu"` plus the 8 new `granite-switch-4.1-8b`
targets that the 2026-11-06 snapshot predates). No label, timeout, skip code or
environment moved. That is the check that this was a targeted change and not a
registration drift.

The 17 hand-written GPU targets that were also unprotected are now locked too
(8 `perf`, 8 `inference`, 1 `convert`), so the configured tree reads:

| label | locked | total |
|---|---|---|
| `inference` | 297 | 297 |
| `perf` | 44 | 44 |
| `convert` | 37 | 38 |
| `fast` | 10 | 140 |

The one unlocked `convert` target is `test_arrow_tree_integrity`, allowlisted as
host-only **with a checkable reason** (0 occurrences of `cuda` in the file: it is
an mmap walk over the Arrow vault). The guard requires a reason for every
allowlist entry rather than a bare name, because "we think it is cheap" is how a
device call gets added later without anyone revisiting the exception.

**What the guard cannot see (open).** `test_ctest_labels_guard` parses literal
`set_tests_properties(<name> ...)` and `lsx_add_*_test(...)` calls. Names that
appear only inside a *group* list — `set_tests_properties(a b c PROPERTIES
LABELS inference)` — or inside a `foreach(qt ...)` registration are invisible to
that scan. That is exactly how the 8 `inference` targets above stayed both
unlabelled-by-the-scan and unlocked: they are registered through
`foreach(qt test_lsx_kernels)` / a group properties call. They are fixed, but the
*detection* still has the hole, and a new target added the same way would be
neither labelled nor locked with the guard's blessing. Closing it means reading
the configured tree (`ctest --show-only=json-v1`) or generating a lock-audit
manifest at configure time — a mechanism decision, not a patch.

## G2. A sibling model can be "covered" by a test that verifies its parent (2026-11-07)

**Status: open — one instance found and recorded, one instance fixed.**

Found while ingesting `granite-switch-4.1-8b`, which is the first family member
added *after* its sibling in the manifest era. A `models/*.json` row claims
targets by name, and `test_model_suite_guard` checks that a claimed target
EXISTS — not that it tests the model that claims it. So a source with a
hard-coded model name can sit under two rows and verify only one of them, while
every guard stays green. That is the shape of the migration's own recurring
defect (a SKIP naming a neighbour), one level up.

Measured: 25 test sources are claimed by more than one manifest. Four contain a
literal `model_type` string:

| source | claimed by | verdict |
|---|---|---|
| `test_granite_switch_adapter_parity.cpp` | 30b, 8b | **was a defect** — fixed: parameterized via `LSX_GSW_MODEL`, with the 30B as the `#ifndef` default and a separate 8B target |
| `test_granite_switch_model.cpp` | 30b, 8b | **was a defect** — same fix (it also `dynamic_cast` to `GraniteSwitch41_30BModel`, which is how the 8B would have failed at runtime instead of at review) |
| `test_granite_switch_tokenizer_parity.cpp` | 30b, 8b | **was a defect** — same fix; the golden cases stay shared because the two tokenizers are md5-identical |
| `test_kimi_k3_constants.cpp` | kimi-k3, kimi-k3.1-pro | **open** — see below |

Also in the capture tools, which is worse because a golden carries the mistake
forward: `granite_switch_adapter_capture.py` and `granite_switch_tool_capture.py`
wrote `"model_type": "granite-switch-4.1-30b"` as a literal. Both now derive it
from the checkpoint directory (the same rule the C++ converter uses), and the
parity test rejects a golden whose `model_type` is not its own — so a
mis-attributed capture fails loudly instead of being compared as if it were
right.

### G2a. `kimi-k3.1-pro` claims a target named after its sibling — name, not coverage

`kimi-k3.1-pro` lists `test_kimi_k3_constants` in its `model` stage. The name says
`k3`, and `test_kimi_k3_constants.cpp` does assert `kModelType == "kimi-k3"`, so
the first reading — recorded here earlier the same day — was "the Pro is covered by
its parent's test". **That reading was wrong**, and refuting it is cheap:

- `tests/CMakeLists.txt` compiles that binary from **two** sources,
  `test_kimi_k3_constants.cpp` *and* `test_kimi_pro_constants.cpp`;
- `--list-tests` shows 12 cases, six of them `kimi-k3.1-pro passport: …`
  (registry/format identity, text tower geometry, Lightning Indexer, derived
  scales, the DSA table = MLA table);
- all 12 pass (615 assertions), and the Pro cases assert
  `kModelType == "kimi-k3.1-pro"` / `kFormatTag == "logestix-kimi-k31-pro-nvfp4"`
  from `kimi_moe_pro_constants.h`.

So the coverage is real. What is genuinely wrong is the **name**: a target called
`test_kimi_k3_constants` sitting in the Pro's row is exactly the shape that made
three granite tests verify their parent, and the next reader will draw the same
wrong conclusion this entry did — from the name alone, without `--list-tests`.

**Not renamed here on purpose.** The honest split is one target per passport: the
k3 cases belong to `test_kimi_k3_constants.cpp`, the Pro cases to
`test_kimi_pro_constants.cpp`, each registered under its own model. That is a small
refactor touching both manifests, `tests/CMakeLists.txt` and the generated
fragment — mechanical, but it should be done deliberately rather than renamed into
a second false impression.

The lesson kept from this entry: **a target's name is not evidence about what it
asserts.** Check `--list-tests` and the source list before recording a coverage gap.

### G2b. The guard, written — and why it checks only the vault env

`test_model_suite_guard` gained *"a shared test source cannot name a sibling as
itself"* plus a non-vacuity case (`the sibling-identity rule bites`). It builds
`source -> claiming models` from the generated registry (25 sources are claimed by
more than one manifest) and flags a source that names a sibling's **vault
environment variable**, or hard-codes a sibling as `cfg.model_type` /
`kModelType`. A literal is accepted when it is the default arm of an
`#ifndef`-guarded macro, i.e. the parameterised form a shared source is supposed to
use (`test_granite_switch_model.cpp`).

Two candidate signals were **rejected after measuring them**, and the reasons
matter for whoever extends this:

- **the `model_type` string**: a sibling test legitimately compares two passports —
  `test_granite_switch_prepare.cpp` asserts that the 8B passport *rejects* a 30B
  topology, so it names both on purpose;
- **the vault directory name**: the same string is a natural name for a test
  fixture directory. The first version of this guard flagged its own fixtures
  (`test_granite_switch_convert.cpp` builds `/tmp/…/granite-switch-4.1-8b`), which
  is how the signal was retired — a check that cannot tell identity from data
  produces false positives and gets ignored.

G2a is the case this guard deliberately does **not** catch: the target's *name*
is misleading while the assertions are correct. That is a naming discipline, not a
mechanical rule — the only mechanical check available would be "a source may only
assert its own `kModelType`", which would forbid the two-source binary that is
actually right.

## G4. A converted tree can be built in `/tmp` and then vanish, leaving a green matrix row (2026-11-07)

**Status: open — vault hygiene, found while re-running parity gates.**

`gigachat3.5-432b-a28b-reasoning` is listed in `docs/SELFCHECK_MATRIX.md` §1 with
SelfCheck/FSI/CODE/DET all green (measured 2026-10-31) and
`docs/model-reports/2026-10-31-GigaChat3.5-432B-A28B-Reasoning.md` exists. On
2026-11-07 its three parity gates SKIP, and the SKIP is *correct*:
`/mnt/fast_storage/huggingface/GigaChat3.5-432B-A28B-Reasoning/` contains the
source checkpoint and a `lsxforge_runtime.1.log` whose writes all point at
`/mnt/fast_storage/tmp/gigachat_reconvert/logestix/...` — the tree was converted
**outside the vault**, and the tmp directory has since been cleaned. No
`logestix/model.index.json` exists anywhere for it.

So the row is not reproducible from the vault, and nothing in the tree records
that. Two lessons, both cheap to act on:

- **Convert into the vault, or record where the tree lives.** A conversion whose
  output root is a temp directory produces a tree that passes gates today and
  disappears tomorrow; the gates then SKIP, which reads as "not applicable"
  rather than "the artifact is gone".
- **The manifest cannot express this.** `golden_debt` is about goldens, and the
  goldens *are* present for gigachat; what is missing is the converted tree, which
  is vault state. `docs/OPEN_TASKS.md` is the only place that can say it, which is
  why it is here.

Re-converting GigaChat costs a 432B/28B-active pass over ~850 GB of source and
needs the disk headroom the host currently does not have (94% full, 878 GB free).
Treat it as a scheduling decision, not a bug fix.

## G5. The manifest migration dropped the cooperative-cancel gate for 8 models and flattened the selfcheck probe (2026-11-07)

**Status: closed — restored as a declared hook.** Recorded because it is the
second coverage regression this refactor caused (the first is §G2), and because
nothing in the suite could see it.

Eight per-family selfchecks called `ExpectCooperativeCancelStops` — deepseek
v4-flash / v4.1-flash / v4-pro, glm-5.2, glm-5.3-flash, ornith, qwen3.8-flash-next,
xing — and v4.1 additionally called `ExpectStreamMatchesTokens`. The single
shared driver (`tests/test_selfcheck_driver.cpp`) reproduced only the Paris case
plus the two hooks it declared (`tool_roundtrip`, `auto_path_determinism`), so for
those eight models **the cancel gate stopped running** while every target stayed
green. A gate that no longer runs is indistinguishable from a gate that passes:
the target still exists, still has its label, still reports Passed.

This is what §A2 had been blocked on. §A2 records a real defect
(`deepseek-v4.1-flash`: a cancelled request poisons the next one) whose *only*
witness was that cancel case — and after the migration the failure could no longer
be observed at all, which would have read as "fixed".

**Fixed:** a third declared hook, `cooperative_cancel`, in the manifest and the
driver. Enabled for the eight models that lost it plus both granite-switch
members (whose decode loop polls the flag). A model that does not declare it gets
`SUCCEED("cooperative_cancel hook not declared …")`, the same shape as the other
two hooks, so opting out is explicit rather than silent.

**The same migration flattened the probe itself (found next, same day).** The
cancel hook was the visible loss; underneath it the driver also dropped four knobs
each file had chosen for itself: the prompt framing (17 files fed
`BuildPromptInputIds`, 18 fed raw BPE), the generation budget (2 for most,
16/24/32/64/192/256 for the rest), the math mode (14 pinned `FastMathMode::kOff`,
21 left the kAuto default), and the extra "raw greedy continuation" assertion
(12 files had it). The first three feed the recorded `logits_hash`, so the uniform
driver changed the hash for nearly every family — phi-4 went `cf51b9bbd5c913dd` →
`c0955d0e9967e4af`, ornith `c3fe24e9dbc55b1e` → `8ea28e0ee80bc323` — and two
families' *gates* went red (phi answered "The capital" instead of "Paris" under a
framed probe; the driver's added continuation assertion is simply wrong for a
framed family, ornith answers "The capital" and should). All four are now manifest
fields — `selfcheck_frame_prompt`, `selfcheck_max_gen`, `selfcheck_bit_exact` —
transcribed from the deleted files, and the continuation assertion was dropped
because the canonical contract is `SelfCheck`.

**The oracle that settles it:** the release tree's binaries are dated
2026-11-06 17:08–17:09 and contain none of the new driver's strings, so they are
pre-refactor. Replaying four of them against the restored driver reproduces the old
hashes bit-exactly — phi-4 `cf51b9bbd5c913dd`, ornith `c3fe24e9dbc55b1e`,
qwen3.5-9b `a2af669548e59c73`, xing `53e18f6dd77a64a2`, same top-1 and `n`. Two of
the four were RED under the uniform driver. A hash mismatch is only actionable
against such an oracle; without one, "the hash changed" and "the weights changed"
are indistinguishable.

**What this class teaches, again:** the migration's own tests check that a
*declared* target exists and runs; they cannot see coverage that was never
declared, and they cannot see a *parameter* that was never declared. The only
defence is transcribing the old files line by line — assertions **and** their
constants — which is what should have happened at the time. §G2 has the same shape
one level up.

## G6. A ctest target can report **Passed** for a gate that never loaded a weight (2026-11-07)

**Status: fixed in the selfcheck driver; the general rule is worth keeping.**

Catch2 returns `AllTestsSkippedExitCode` (4) only when **every** case in the binary
skipped, and `SKIP_RETURN_CODE 4` is what turns that into ctest's `***Skipped`. The
selfcheck driver's three "hook not declared" branches were `SUCCEED`, so for a model
with no converted tree the binary reported:

    test cases: 4 | 1 passed | 3 skipped

The single "pass" was an undeclared hook — no weights, no vault, no model. The
binary exited 0 and ctest printed **Passed**. A sweep over the 37 SelfCheck targets
therefore reported 37 green when only 27 had executed the Paris case; the other 10
(the models whose converted tree is absent — `beam`, `deepseek-v4-pro-0813`,
`flint`, `gigachat`, `glm-5.2`, `ling`, `mimo`, `mistral-large-4-1t`,
`nemotron-3.5-lightning-30b-a3b`, `step-5-preview`) had skipped everything that
matters.

**Why it is not visible from the guard side:** `test_ctest_labels_guard` enforces
that a skip-capable target *sets* `SKIP_RETURN_CODE 4`; it cannot see that the
binary's own bookkeeping keeps the skip from reaching the exit code. The property
is per-BINARY, not per-test: one always-green case anywhere in the file defeats it.

**Fix:** the three branches `SKIP` instead of `SUCCEED`, so a weight-less model
exits 4 and ctest says `Skipped`. An undeclared hook was never a pass — it is "not
applicable to this model", which is what a skip means. A model that DOES load
weights still reports `Passed` with its hook cases skipped, which is correct: the
Paris case ran.

**The transferable rule:** in a Catch2 binary, "green" only means "nothing failed",
and a case that opts out with `SUCCEED` is indistinguishable from a case that
passed. Any driver with declared optional cases must use `SKIP` for them, or count
its own cases explicitly.

## G7. A relative fixture-dir fallback makes a gate SKIP while its golden sits in the tree (2026-11-07)

**Status: fixed; guarded.**

`tests/test_granite_switch_model.cpp` resolves its tool-round-trip golden with the
macro `LSX_GRANITE_SWITCH_FIXTURE_DIR`, whose `#ifndef` fallback is the **relative**
path `"tests/fixtures/granite_switch"`. `test_granite_switch_model` and
`test_granite_switch_8b_model` never received that macro from CMake (the other
granite-switch targets did), so under ctest — which runs the binary with CWD =
`build/linux-debug` — the path resolved to a directory that does not exist there and
the case reported

    SKIPPED: granite-switch tool golden not captured

while `tests/fixtures/granite_switch/tool_roundtrip.json` sat in the source tree.
The lifecycle gate requires "a native tool round-trip", so for both members of this
family that requirement was **not being verified at all** — and nothing was red. The
8B's stage-9 report quotes the real-weight line (`pinned weights loaded: 40 layers
(6088178432 bytes resident)`), which was captured by running the binary **by hand
from the repository root**, where the relative path happens to work. Reproducible by
accident of CWD is not reproducible.

**Fix:** the absolute fixture dir is now passed to both targets. Verified by
re-running through ctest: the case loads 40 layers (6 088 178 432 bytes) for the 8B
and 64 layers (17 416 338 176 bytes) for the 30B, and both binaries report 6/6 cases
with 136 assertions where they previously reported 5 passed + 1 skipped, 126.

**The class, and why the guard is the fix rather than the two defines.** 13
directory macros in the tree carried a relative fallback; 11 were correctly defined
by CMake and 2 were not, and the failure mode is silent in every case. Two guards
now make a repeat loud:

- `infra guard: a fixture dir macro is never satisfied by its relative fallback`
  scans every `#ifndef X / #define X "<relative>"` and requires each target that
  compiles that source to pass `X`. It has non-vacuity assertions (>= 5 macros and
  >= 100 `add_executable` blocks) because an empty scan passes everything — proven
  by removing one define and watching it name `test_granite_switch_model compiles
  test_granite_switch_model.cpp without defining LSX_GRANITE_SWITCH_FIXTURE_DIR`.
- `tests/test_parity_gate.cpp` no longer has per-kind fallbacks at all: the three
  `LSX_{FSI,CODE,DET}_GOLDEN_DIR` macros collapsed into one `LSX_GOLDEN_DIR`, with
  `#error` when absent. The kind is a string and cannot be compared in `#if`, so
  three per-kind macros forced three "define it or fall back" branches — the
  fallback was the design's fault, not a reviewer's. The runtime env overrides keep
  their documented per-kind names.

**Why this is the third G-class item with the same shape:** G5 (a declared hook that
stopped running), G6 (a case that skipped while the target said Passed) and G7 (a
path that resolved differently under ctest) are all "the instrument reported absence
as success". None of them was caught by a red target; all three were caught by
asking what a green number actually measured.

## G8. A DEGRADED `perf` verdict hid the `r3` verdict entirely (2026-11-07)

**Status: fixed.** Found while re-checking the mistral-small-4 report, and it is the
fourth item in this file with the same shape: an instrument reporting *absence* as
if it were a result.

`tests/test_perf_baseline.cpp` measures two numeric modes per model in one loop —
`perf` (kAuto) and `r3` (kOff, bit-exact) — and judges each with
`CheckAgainstBaseline`. The DEGRADED verdict used `FAIL(...)`. In Catch2 v3 `FAIL`
is a **fatal** assertion (`ResultDisposition::Normal`), unlike `FAIL_CHECK`
(`ContinueOnFailure`), so the first degraded mode aborted the test case and the
second mode was **never measured**. Verified: `mistral-small-4-119b` printed one
`weights loaded` line and one verdict; after the change it prints two of each and
reports both (`perf` 95.9%, `r3` 96.3% of their references).

The consequence while it lasted: any model whose `perf` mode regressed had an
*unverified* `r3` row, and the report/catalogue quote both numbers as if measured.
The test still fails after the fix — `FAIL_CHECK` records the failure and the run
exits non-zero — it just finishes the pair first.

**Why the guards did not catch it:** `test_perf_baseline_logic` tests `JudgePerf`
and `JudgePerfLen`, i.e. the decision, not how many modes a failing decision lets
reach the judge.

### G8a. `mistral-small-4-119b` reads 95.9% / 96.3% of its recorded throughput (open)

The other half of the same run: re-measuring the committed tree eleven days later
gives 43.89-44.31 t/s (`perf`) and 47.99 t/s (`r3`) against references of 45.77 and
49.82 — stable to under 1% over six runs and across both modes, so a systematic
~4%, not noise. **Thermal and power throttling are ruled out by measurement**: a
cold-card (45 °C start) run and an immediately-following hot-card run give the same
verdict with 46/51 °C, 141-145 W mean against a 600 W cap, and the harness's own
DRAM oracle reads 99.4% (1524 vs 1533 GB/s). Nothing in `mistral_moe_*` or the
shared kernels changed since the reference was written. `granite-4.2-30b` shows the
same shape (96.7%); `phi-4` passes.

**The reference has deliberately NOT been re-seeded.** Two candidate causes, in the
order they should be chased:

1. **The reference is a ratcheted best-of sample.** The harness re-seeds on any
   improvement, so a recorded row is the best observed run, not a median, and the
   pass band is only 3%. This row replaced 45.04, and `r3` replaced 41.57 — a 20%
   jump on the same tree, which is itself evidence that this measurement is
   unstable. If that is the cause, the fix is in the harness's statistics (record a
   median of N, or widen the band for a family whose spread is measured), not in
   the model.
2. **The capacity oracle is disabled for this entry.** `expert_cache_slots` is
   present in 2 of 67 entries; `mistral-small-4-119b` is not one of them, so
   `baseline_slots` is 0 and `AttributeDegrade` cannot return `kCapacity` for the
   family most sensitive to it — 4 608 routed experts streamed through 1 398 device
   slots, with the capacity derived from ambient free VRAM at load time. Backfilling
   the field on a conforming run restores the oracle.

Do not re-seed until one of those is settled; a wrong re-seed would erase the only
evidence if the ~4% is real.

## R18. Dory1.0-4B (4B, looped nGPT) — ingestion landed, one engine lesson on
direction-carrying residuals (2026-11-10)

Branch `feat/dory1-4b`. New family (`model_type: dory`, kType `dory-1.0-4b`):
a **looped latent-recurrent normalized-GPT** decoder, not a transformer with
pre-norms. All nine lifecycle stages ran on the real 7.95 GB checkpoint and are
green: SelfCheck Paris ×3 bit-exact (`f100088895adb12d`, top-1 6993), FSI 3/3,
CODE 3/3, DET 3/3, native tool round-trip, perf seeded through the harness at
**94.2 t/s** (`r3`) / 93.1 (`perf`) 2048 tokens, 4.83 GiB resident. Report:
`docs/model-reports/2026-11-10-NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16.md`.

### R18a. The auto (TC fast-math) policy silently produces non-source tokens

**Status: fixed by pinning `FastMathMode::kOff` in `DoryDenseModel::Infer`**, the
same remedy granite-4.2 and granite-switch already carry.

Symptom: the converted model answered `# Prompt\n\nWrite a short story about a
man who is a professional photographer…` to "The capital of France is" framed as
a chat, while the strict path answers `The user is asking a simple factual
question … Paris` — fluent, plausible, and unlike anything the source produces.
The SelfCheck does not see it because it runs at `kOff`; the FSI/CODE/DET gates
do not see it because they also pin `kOff`. Only a `RunCliGeneration` /
`lsxfabric --testprompt` invocation at the production policy exposes it, which is
the same blind spot `granite-4.2-30b` had.

Root cause, measured with a scratch GEMV harness (deleted after use): the
Blackwell W4A4 `mxf4nvf4` path is **tolerance-class per GEMM** — relL2 against
the scalar reference is 0.062 at `seq=1`, 0.113 at 3, 0.182 at 8, 0.192 at 16,
0.199 at 32 (K=2560, N=6144). For a magnitude-carrying transformer that is
invisible. For Dory it is not, because **every mixer output is L2-normalized
onto the unit sphere**: a few percent of relative error in a projection becomes
a *direction* error of the residual stream at the very next layer, and the
2 × 40 recurrent passes compound it. The residual NVFP4 plane plus the strict
scalar route is what makes the 4-bit body track the source (first 64 tokens of a
long-form prompt agree with the source oracle for 43, then one near-tie forks).

The general lesson, worth applying before the next ingestion: "auto policy" is
safe for a family whose residual stream carries magnitude, and unsafe for one
whose residual stream carries direction. Check what the architecture normalizes
before accepting a tolerance-class kernel route.

Pinned by `test_dory_dense_model.cpp::"dory pins the strict routing (TC
fast-math cannot change the output)"`, which drives the process-global routing to
the TC path before `Infer` and requires the strict token stream bit-for-bit.

### R18b. A reasoning-always model needs the mode named per gate

The checkpoint's template defaults to `enable_thinking=true` and both vendor
serving recipes enable a reasoning parser. Two of the three parity probes are
mode-sensitive:

- DET-TEST-02 (letter match) answers `CRITICAL` with reasoning off and
  `CATASTROPHIC` with it on — the latter is the correct answer, so the DET golden
  is captured in reasoning mode (2048-token budget, which is what lets all three
  probes close their `<think>` block);
- FSI and CODE are captured with reasoning off: this model's trace on those
  prompts is long and self-repeating and does not close the block inside 2048
  tokens, so the extraction would fall back inside the reasoning text.

The engine frames with reasoning ON by default (the served mode). Both capture
modes are named in the capture scripts' `OVERRIDES` comments and in
`docs/SELFCHECK_MATRIX.md`; this is a recorded deviation, not a silent one.

## G9. `test_gpu_perf`'"'"'s bandwidth-probe case is flaky under a PARALLEL fast
sweep (2026-11-10)

**Status: open. Pre-existing, not caused by any model branch** — reproduced with
every `dory` target excluded from the run.

`tests/test_gpu_perf.cpp` case *"memory-bandwidth probe releases its device
buffers"* (added in `d56305d3`) asserts

```
free_before >= free_after        // after MeasureMemoryBandwidthGbs()
```

to prove the probe does not leak its 2 × 512 MiB working set. The assertion is a
*cross-process* comparison: `cudaMemGetInfo` reports the whole device, so any
other process that releases memory during the probe window makes `free_after`
larger than `free_before` and the case fails while proving nothing. Measured on
`linux-debug`, `-j 24`, three consecutive sweeps of the 140 non-dory fast
targets: **fail, pass, fail**. Isolated (5 consecutive single-process runs):
5/5 pass, and the underlying probe is fine — the failure message shows
`free before 29543 MiB, after 29843 MiB`, i.e. free memory went UP by 300 MiB.

Two candidate fixes, in order of preference:

1. **Assert the leak, not the delta.** The probe's contract is "no 512 MiB
   buffer survives". Measuring that needs the process's own held bytes, not the
   device's free bytes — e.g. `cudaDeviceGetLimit`/a pool snapshot, or a
   `cudaMemGetInfo` pair taken around alloc/free of a same-sized sentinel buffer
   inside the case. Attribute the slack to the sentinel, not to the world.
2. Give the case `RESOURCE_LOCK "gpu"` so it never shares the card with another
   target of the same ctest. This hides the cross-process window without making
   it measurable — and it costs a lock on a label that deliberately stays
   parallel (`test_ctest_labels_guard` exempts `fast` for exactly that reason),
   so it would just move the problem to whatever runs next.

Do not "fix" it by widening `kSlack`: 300 MiB of foreign churn would still fail,
and a slack large enough to absorb it also hides the 1 GiB leak the case exists
to catch.

## R19. Inkling-2-Small (594B / 17B active) — stages 0-3 landed, engine pending (2026-11-14)

Branch `feat/inkling-2-small`. New family (`kType = inkling-2-small`, format tag
`logestix-inkling-2-small-nvfp4`): 48 MoE layers, hidden 4608, 48 Q / 8 KV heads
at head_dim 128, **no RoPE anywhere** (a learned relative-position bank of
d_rel 16 profiles over 1024 backward distances replaces it), 40 sliding-window
layers (1024) + 8 full-attention layers at `(layer+1)%6==0`, per-head learned
**L2 clipping** (q 16√2, k weight-free, rel 8.0) instead of QK-norm, **gated
attention with learned per-head sinks**, four short convolutions per layer
(post-attention, post-MLP, and k/v with static+per-token taps on the full
layers), an **AttnRes depth mixer** of block 6 (single-query for sliding layers
and MLP inputs, three queries on the full layers for separate q/k/v inputs, one
`output_query` for the final norm), 384 routed experts top-8 (sigmoid scoring,
selection-only FP32 bias, renormalised then ×8.0), softcapped SwiGLU
(gate 4.0, up 25.0), final logit softcap 30.0, vocab 201024 sliced to 200058,
tiktoken BPE (kind 2).

Stage 0 record and the plan: `docs/open-tasks/2026-11-14-inkling-2-small-ingest.md`.

**Measured on the real checkpoint** (367 GB source: 24 shards +
`mtp.safetensors`; RTX 5090 host, 2026-11-14):

| item | value |
|---|---|
| converted artifact | **312 GB** under `<vault>/logestix` (18 432 routed-expert records + 996 pinned tensors) |
| routed experts | 40 layers passed through **byte-verbatim** NVFP4 (per-expert global scale `amax/2688`), 8 layers (40-47) dequantized from MXFP8/32 and requantized to NVFP4 |
| stripped ballast | 422 vision tensors + the 5-layer MTP file (`mtp.safetensors`, not in the index) |
| conversion wall time | ~18 min on this host (single process) |
| dense plane | ~4.7 GiB NVFP4 + ~190 MiB FP16 router/side tensors |
| active read per token | ~7.1 GB (8 experts × 48 layers), so ~0.5-2 t/s is the expected band |

**Gates landed and green**

- `test_inkling_moe_constants` (fast, 186 assertions): the frozen passport.
- `test_inkling_moe_convert` (convert, synthetic checkpoint, 349 assertions):
  converter E2E, including the byte-verbatim expert pass-through with the
  per-expert global scale and the MXFP8 -> NVFP4 requant path.
- `test_inkling_moe_loader` (load, synthetic tree + the real converted index).

**Open, explicitly.**

- I1. **PARITY IS RED on real weights (2026-11-14).** The engine loads the 594B
  tree, frames the control prompt through the checkpoint's own chat template and
  decodes; it continues the `Thinking effort level: 0.9` line instead of
  answering Paris, with a flat logit distribution. **Nothing in this row may be
  read as a pass.**
  Seven real defects were found and fixed on the way: BF16 read as an FP16 bit
  pattern (every norm, clip scale, conv tap, sink, relative bank and router
  weight); a scratch-arena overlap that wrote the dynamic conv taps into the
  AttnRes block stack on every full layer; the relative-position states aliased
  onto the conv scratch; the recurrent conv state not reset between requests;
  the post-MLP conv state overlapping the post-attention one; and BOTH
  `embed`/`unembed` and the dense projections being requantized to NVFP4 when the
  release keeps them at BF16 (measured cost 4.6 deg on the embedding and 9.5%
  relative L2 on the projections — fixing them is what turned garbage into
  English).
  The artifact now carries the release's own precision and is verified
  tensor-by-tensor against the source (996/996 pinned tensors exact; NVFP4
  experts byte-exact), and the engine is verified against the AUTHORITATIVE
  implementation: `tools/inkling_transformers_crosscheck.py` loads the converted
  artifact into `transformers.models.inkling2` and gets **identical top-5 logits
  and 1e-3 per-layer agreement**, both at the tiny fixture geometry and at the
  real model's ratios (GQA group 6, top-8 experts, two AttnRes blocks).
  The remaining suspects, ranked, are in
  `docs/open-tasks/2026-11-14-inkling-2-small-ingest.md` § "What remains"; the
  first is the eight MXFP8 expert layers, which are the only part of the artifact
  not at the release's precision (requantized to 4 bits, measured 0.094).
- I2. `mtp.safetensors` is a **separate 3.4 GB file that the index does not
  list**; a converter that trusts `model.safetensors.index.json` silently
  ignores the 5 MTP layers (correct) but must not assume the directory only
  holds what the index names.
- I3. The expert codec is **mixed and undocumented in `config.json`**: only
  `hf_quant_config.json` says layers 0-39 are NVFP4 and 40-47 MXFP8. A
  converter that assumes one codec decodes a sixth of the stack as garbage.
  The converter now cross-checks the observed dtype against that file and
  fails on a mismatch.
- I4. `attn_res.attn_queries.L` is indexed by **`layer-1`** (layer 0 has no
  attention mixture) while `attn_res.mlp_queries.L` is indexed by layer. The
  artifact re-indexes both to the owning layer.
- I5. `config.json`'s `eos_token_id` is `<|content_model_end_sampling|>`
  (200006), not `<|endoftext|>` (199999); both stop. `bos_token_id` is absent
  and there is no pad id.
- I6. The `transformers` port that is authoritative for this family is
  `models/inkling2` (5.23.0+). The earlier `models/inkling` module is a
  **different** architecture (no sinks, no L2 clip, scale `1/head_dim` instead
  of `head_dim**-0.5`, no dynamic convolutions, no AttnRes) and copying from it
  produces plausible but wrong logits.

## Archive

Resolved items moved out of this file on 2026-10-02, preserved with their full
investigation trail:

| Item | Outcome |
|---|---|
| 2026-11-07 | Granite-Switch 4.1 8B FSI-01: red on a form fork, closed with a MEASURED residual-plane placement | The 8B sibling's FSI-01 answered the formula evaluated (`PV=2500000*…`) where the source answers it symbolically — every constant correct, so a form fork rather than a corrupted computation. Teacher-forcing showed it was not a tie: source margin +6.88 logits for `CF`, converted −1.27, an 8.2-logit swing. Bug checklist first (head `global_scale` applied, `Free()` complete, records decode back at relL2 0.094–0.095 on layers 0/20/39) — so, 4-bit body error. Four trees were then converted and probed rather than assuming gemma's placement: none −1.27 (red), **attn +0.9 GB +1.58 (green, shipped)**, mlp +3.3 GB +2.19, attn+mlp +4.2 GB +7.14. `attn` is the cheapest that clears AND leaves the biggest tensors at 4 bits (MLP = 79% of the layer's bytes). Resident 5.14 → 6.09 GB, profile min 5.51 → 6.45 GB, index records `"residual": "attn"`, knob is convert-time (`LSX_GSW_RESIDUAL`). FSI 3/3, SelfCheck ×3 `b7c1aa453d83a86f` (re-measured 2026-11-07 as `930dd592ea5f5834` when the generation budget became a declared per-family value), DET 2/3 = the source's own `CRITICAL`. Two instrument defects recorded: the probe read `final_logits` (published after the decode loop, one position late) and reported the wrong winner, and the `det` verdict folded an inference error into `deterministic=false` while embedding raw newlines into a one-line record. Legacy 4-bit-only trees still load (`ad9b0c4940bb27a9`). — [`open-tasks/2026-11-07-granite-switch-fsi01-residual.md`](open-tasks/2026-11-07-granite-switch-fsi01-residual.md) |
| 2026-11-06 | AliceAI-Foundation-80B-A3B: an inverted allocator predicate silently disabled generation for three days | All four gates were red with `alice workspace alloc failed` — and the pre-migration binary failed identically, so it was not the manifest work. The recorded `cudaError_t` and buffer identity showed `cudaMallocAsync -> cudaSuccess (0), free 27399 of 32202 MiB`: nothing had failed. 397ab361 rewrote the workspace macro's ternary arms from `cudaError_t` constants to `true`/`false` while clearing 42 `-Wint-in-bool-context` sites, inverting a chain that reads `if (A() || A() || …)`: the first *successful* malloc aborted the workspace, so alice decoded zero tokens from 2026-11-03 19:33. It stayed invisible because the message discarded the error and the free-VRAM number, because the gate driver's per-case `INFO` left scope before the aggregate `REQUIRE` (Catch2 printed `false` alone), and because the gates carry label `inference`, which no default sweep runs. Fixed with a named `WsAllocFailed()` predicate, a diagnostic naming buffer/bytes/error/free VRAM, and `ExplainCases()` in the gate driver; pinned by *"alice workspace allocator polarity"*, proven red by inverting the predicate. Post-fix SelfCheck Paris ×3 `logits_hash 0bf456024cc05255` (top-1 828), FSI/CODE/DET 3/3 at 230/221/183 s, tool round-trip green; the hash moved from `f900547d25be7914` (2026-11-03) at an unchanged top-1 id, unattributed between 397ab361's dropped `__restrict__` sites and the CUDA 13.5 bump. — [`open-tasks/2026-11-06-alice-workspace-polarity.md`](open-tasks/2026-11-06-alice-workspace-polarity.md) |
| 2026-11-03 | Dense-trio consolidation, phase 1: granite / granite-switch / phi now run one neutral `lsxcommon::dense` kernel layer | The three dense families had drifted into three near-copies of the same decode kernels. Phase 1 moved RMSNorm, full-dim RoPE, causal GQA attention (with the split-KV flash-decode path), fused split-QKV and fused SwiGLU into `lsx_dense_inference.{h,cu}`; the algorithm is the one Granite-4.2 always ran, moved verbatim (`tools/dedup_provenance.py` proves it), and every family's layer driver stays family-named. No golden moved. — [`open-tasks/2026-11-03-dense-trio-neutral-layer-closed.md`](open-tasks/2026-11-03-dense-trio-neutral-layer-closed.md) |
| 2026-11-03 | Warning backlog closed: a clean build is 246 lines, none of them ours | Scoping `-Wall -Wextra`/`-Wpedantic` to our own directories dropped a clean build from 51 408 warning lines to 484; this session cleared the remaining 238 across 170 sites / 83 files **without silencers**. The load-bearing find: 39 in-place activation/norm calls aliased `__restrict__` parameters, i.e. formal UB, fixed by dropping the qualifier. Plus fail-closed `SimpleAtoi`/simdjson config reads, a dangling test reference, `memcpy` into `__half`, 19 dead functions. — [`open-tasks/2026-11-03-warning-backlog-closed.md`](open-tasks/2026-11-03-warning-backlog-closed.md) |
| 2026-11-03 | AliceAI-Foundation-80B-A3B re-ingested: two accounting defects closed | The converted tree had not survived on this host, so the whole lifecycle was re-run — which exposed two defects no consumer had exercised: the emitted VRAM profile counted the routed experts as BF16 **and** device-resident (`min_bytes` 162 GB, `max_bytes == optimal_bytes`), and `resident_bytes` under-reported the uploaded dense graph 8x (1 187 vs 4 231 MiB) — the number that drives the governor lease. Both fixed test-first. SelfCheck Paris ×3 bit-exact (`f900547d25be7914`), FSI/CODE/DET 3/3, tool round-trip, perf re-seeded 45.70/45.88 t/s. — [`open-tasks/2026-11-03-alice-reingest-closed.md`](open-tasks/2026-11-03-alice-reingest-closed.md) |
| 2026-10-28 | MiniMax-M3 (428B/23B) ingested end to end | First 400B-class ingestion on this box: 796 GB BF16 → 229 GB NVFP4 in 1 h 45 m, all eight protocol stages with real-weight verification. SelfCheck ×3 bit-exact, CODE 3/3, DET 3/3, FSI 2/3 (FSI-01 template-form red). Three engine defects (1-block residual add, sticky TTFT event, a pre-set abort emitting a token) and a two-part tokenizer defect (added tokens never literal-matched; GPT-2 pre-tokenizer instead of Llama-3/o200k) were found and pinned; embed/head must stay FP16 because 4-bit head noise (~3 logits) against a 0.4 first-token margin flips the answer. — [`open-tasks/2026-10-28-minimax-m3-ingest.md`](open-tasks/2026-10-28-minimax-m3-ingest.md) |
| 2026-11-05 | Kimi decode rate: the "4x gap" was prose, the real defect was the expert window; eviction no longer drains the stream | The reported 4x slowdown of `kimi-k3.1-pro` against `kimi-k3` never existed: K3's "~2 s/token" was committed at 04:16:28, i.e. **before** that same log's first cycle at 04:24:38 (K3's own cycles are 434.6/426.3 s per 64 tokens = 6.66-6.79 s/token), and at identical binary, geometry, window and 1472-expert working set the two models measured **6.95 vs 6.98 s/token — ratio 1.00**. What the prose was hiding was a shared-`ExpertStore` defect: `max_open` 512 against 1472 experts (92 MoE layers x top-16, each layer visited once per token in ascending order) is a cyclic scan with a provably **zero** hit rate, so every expert was re-opened, re-mapped, re-parsed and evicted. Fixed Kimi-locally: window clamped up to the working-set floor with a 65 536 default, and `warm_on_load=false` (the eager pre-parse walks experts in path order, cost 156 s and bought nothing) — **6.95 -> 2.96 (K3) / 3.23-3.35 (Pro) s/token** at a 5 s load, bit-exact throughout. Then the eviction half of the same loop: `EnsureResident` called `cudaStreamSynchronize` on every eviction (20.3 ms per eviction against 20.2 ms of unrelated queued work); an evicted mapping is now **retired with a CUDA event** and released once the copies ahead of it have executed — **0.07 ms** (282x), 6.95 -> 6.26 s/token at a 512-slot window. Two negative results are recorded: page-cache prefetch of each layer's top-k cost 7.16 s/token (18% worse) and `fadvise(WILLNEED)` was a wash, both removed rather than shipped as knobs; and the first version of the eviction fix **regressed mistral-small-4 by 6.6%** (45.04 -> 42.06 t/s, caught by `test_perf_baseline` with a clean probe, counterfactual-passing build proving the regression was ours) because it queried every retired event - the events complete FIFO on one stream, so the front check alone is O(1). What remains is filed as §R17 **P6**: overlap the next layer's staging with this layer's compute — [`open-tasks/2026-11-05-kimi-k3-decode-window-and-eviction.md`](open-tasks/2026-11-05-kimi-k3-decode-window-and-eviction.md) |
| 2026-11-04 | qwen3.5-35b-a3b verified on real weights; two benchmark rows had been stale in opposite directions | The member was sitting in `SELFCHECK_MATRIX.md` §3 ("golden present, **no converted tree** — SKIP") although its 19 GB tree had been in the vault since 2026-10-01 and `kGoldenDebt` never exempted it — so all four gates were *required* to be green and none had ever been run. Measured serially on the real weights: SelfCheck Paris ×3 bit-exact (`logits_hash f4c91be5f2206d56`, top-1 11751), FSI/CODE/DET 3/3 each, DET `deterministic=true`. Both of its benchmark rows were dated 2026-10-06, i.e. before the neutral routed-MoE migration, the overflow-proof batched SwiGLU (`8e34314`) and the 2026-10-23 strict-mode fix — its siblings were re-seeded at each of those moments and this one was skipped twice. `perf` was **DEGRADED** at 96.14% with a clean probe (1528.54 vs 1531.68 GB/s, so not the environment): three runs measured 163.54 / 159.59 / 159.81 t/s and the row now carries the harness's own write, 170.10 → **159.81**, on the same band as its two 35B-A3B siblings on identical code (qwen3.6 161.97, ornith 160.25) — the accepted cost `8e34314` itself records as "Qwen3.6 166.7 → 161.3". No parent-commit bisect was run: the residual is inside the 2.4% run-to-run spread, and the sibling triple plus the change's own A/B carry the attribution. `r3` was 13.78, a **superseded** value — 5.6× below what the same binary does (77.3 t/s by `lsxbenchmark --bench --bit-exact`, TTFT 64.16 ms), i.e. a five-fold regression would have passed unseen; re-seeded through the harness 13.78 → **77.56**. Also recorded the missing `qwen_moe.router.decode` budget for `rtx 5090` (11.7931/11.7947/11.7955 µs over three observations, max 11.7955, budget 15.924 = 1.35× pad), so the Release case now judges (260 assertions) instead of `SKIP`ping — the instrument that keeps the ~30 µs deep-unrolled variant out — [`open-tasks/2026-11-04-qwen35-parity-and-open-tasks-hygiene.md`](open-tasks/2026-11-04-qwen35-parity-and-open-tasks-hygiene.md) |
| 2026-11-04 | §R16 and §J moved out of the live list; two gate documents corrected | The live file was carrying two fully-resolved sections, which its own rule forbids: **§J** (gemma-4-26b-a4b-it FSI/CODE), resolved 2026-10-24 and already archived, and **§R16** (qwen3.8-flash-next), whose last residual closed in this branch — its text moved verbatim into the lifecycle trail (§6). With §R16 gone, §A1 is the only qwen item left (the Windows/5060 Ti kernel fault no measurement on this host can chase) and its latency-budget half is now separated from the fault on this GPU. The same audit found `deepseek-v4.1-flash` also misfiled as "no converted tree" (294 GB tree present) and corrected the matrix summary counts to a stated counting rule (SelfCheck 22/22 rows, FSI 58/63, CODE 62/63, DET 62/63 over §1) plus the `Reproduce` block, and both `{DET,CODE}_SELFCHECK_FINDINGS.md` claims that the qwen3.5-35b-a3b source is absent from the vault — [`open-tasks/2026-11-04-qwen35-parity-and-open-tasks-hygiene.md`](open-tasks/2026-11-04-qwen35-parity-and-open-tasks-hygiene.md) |
| 2026-11-04 | qwen3.8-flash-next §R16 item A closed: expert staging moved onto the copy engine, chunk depth follows the layer's miss count | The last §R16 residual is closed. Residency staging no longer runs on the compute stream: it rides `copy_stream` with a per-expert `ev_uploads[k]` and an `ev_front` slot-reuse fence, and the routed TC chain gained a neutral entry point (`moe::RunMoERoutedTcBatchStreamed`) that runs the K slots in ascending chunks, each waiting **only** on its own uploads, so the next chunk's H2D overlaps this chunk's TC work. Depth is chosen from the miss count (`chunks = (n_miss >= 2) ? 2 : 1`) because a chunk's compute can only hide another chunk's copy — with fewer than two misses the second batch launch (K=5 tiles per launch instead of 10) is pure cost. Bit-exact by construction (the fp32 accumulate order into `d_acc` is unchanged; `LsxTcBatchSplitK` scratch is slot-disjoint), `logits_hash 2e3bcd090441ad65` unchanged. Measured against HEAD at identical slots/hit rate: **−1.55%** on `perf` (47 679 → 46 938 ms) and **−1.47%** at a forced 4319-slot capacity oracle (hit 0.862 → 0.745), `r3` −0.07% because under `kOff` the batched chain is not used at all — the honest reason one row does not move. The **instrument** was corrected first: the previous attribution reported 0.8% overlap from a scan that only matched the kernel starting before each copy; an interval intersection gives 17.7%. What is left is structural — the router host readback leaves the compute stream empty when the copies are issued, and the copies already run at 54.5-55.6 GB/s (the PCIe Gen5 ceiling) — so the largest serial host in the step is now the hyper-connection block, 6.97 of 18.5 ms/step of kernel time. Also fixed in passing: an early `chunks <= 1` path skipped the upload waits entirely, a race for any caller passing a `copy_stream`; one chunk now means "serialize", the test covers 1/2/3, SelfCheck ×3 + tool round-trip are green and `ctest -L fast` is 134/134. Rows re-seeded through the harness 43.39 / 33.91 t/s — [`open-tasks/2026-11-04-qwen38-expert-staging.md`](open-tasks/2026-11-04-qwen38-expert-staging.md) |
| 2026-11-04 | qwen family framing: the two "3.8 deltas" were family-wide | The previous session classified `render_content\|trim` and `{{- tool \| tojson }}` as qwen3.8-only and carried them behind policy flags. Diffing the six checkpoints' `chat_template.jinja` shows both lines are **identical in every qwen member** (4 trim sites + 1 tojson site each); 3.8's real deltas are `reasoning_effort`, the `preserve_thinking` default and an empty-system-turn guard. So the shared codec emitted the compact tool JSON and untrimmed content for every member, and 3.8 trimmed tool names/descriptions that the template renders verbatim (`"  Get the weather  \n"` byte for byte in the oracle). Fix: trim and tojson are unconditional (the flags are deleted — no template makes them configurable), trim is scoped to message content via `esc_content` vs `esc_verbatim`, and `suppress_empty_system` joins `QwenRenderPolicy` as the third genuine per-family delta. Four new oracle-pinned cases + two missing real-weight tool round-trips (qwen3.5-9b, qwen3.5-35b-a3b — neither had one, the instrument gap that let this ship); all six members' round-trips green on the new wire and every SelfCheck `logits_hash` unchanged, `ctest -L fast` 134/134. The gates could never have seen it: FSI/CODE/DET consume pre-tokenised `input_ids`, so the renderer is not in their path — [`open-tasks/2026-11-04-qwen-template-fidelity.md`](open-tasks/2026-11-04-qwen-template-fidelity.md) |
| 2026-11-04 | qwen3.8-flash-next full lifecycle re-verified; two perf-harness instrument defects fixed (§R16 then closed the one perf residual; the section moved out of the live file) | Every stage re-run on the real 336 GB checkpoint (164 GB tree): SelfCheck Paris ×3 bit-exact (`2e3bcd090441ad65`), FSI/CODE/DET 3/3, EOS + cross-instance gate, native tool round-trip; `ctest -L fast` 134/134. Stage 4b found the 3.8 `chat_template.jinja` differs from 3.5/3.6 in byte-level ways (closed think block on every assistant history turn; jinja `tojson` tool serialisation; `render_content|trim`) — carried by a new `QwenRenderPolicy` so the older members keep their exact wire, pinned against the template rendered by transformers **and** against the exact policy the model renders with (the `trim` delta was first wired into the renderer and left off for the model). The `|trim` delta is shared with 3.5/3.6 — latent for every qwen member, invisible to the gates because the capture prompts carry no surrounding whitespace — and those members keep the untrimmed wire until their goldens are re-verified (§R16). Two **instrument** defects were under-reporting the whole perf suite and were fixed test-first: `MeasureMemoryBandwidthGbs` held two 512 MiB buffers in `thread_local` statics for the process lifetime (the probe runs before the model in `test_perf_baseline`, so every capacity-bound family from 2026-10-23 measured with 1 GiB less VRAM — 4962 → 5350 slots here; the 49 affected rows self-heal on their next run because an increase is upserted), and the generation arena/embed statics were charged twice on the harness's second `Load()` (`SizeExpertCacheSlots` + `GenContextResidentBytes`). Rows re-seeded through the harness: **42.98** perf / **33.88** r3 t/s (old 44.85 / 26.47 removed first); the residual is capacity, measured by a forced-slot sweep (4962 → 41.40, 5350 → 43.00, 5750 → 44.15, 6000 → OOM) and now expressible as `PerfCause::kCapacity` with `expert_cache_slots` on every row — [`open-tasks/2026-11-04-qwen38-flash-next-lifecycle.md`](open-tasks/2026-11-04-qwen38-flash-next-lifecycle.md) |
| 2026-11-03 | granite-4.2-30b full lifecycle re-verification: §K `code` debt cleared, TC-overflow defect fixed | Re-converted the real 54.5 GiB checkpoint to a 15.3 GiB `logestix-granite` tree and ran every stage. Real defect found TDD: the default auto/TC NVFP4 path degenerates on the 306-token tool prompt (`_with_n_comprehension…`). External oracle: the BF16 checkpoint on CPU emits `get_weather {"city":"Paris"}` for the exact reference ids, and the converted model under `FastMathMode::kOff` reproduces it token-for-token — the TC activation range is exceeded (same class as granite-switch). `Infer` now pins `kOff`. Gates green: SelfCheck Paris x3 bit-exact (`c165e80fde6401cf`), FSI/CODE/DET 3/3, tool round-trip; perf re-seeded 76.79/76.47 t/s at 2048 (old 2026-09-22 rows removed first; the perf row is ~1.4% below its old 77.86 because it now takes the strict path). §K `granite-4.2-30b:code` was the last single-stage golden-debt entry — resolved (golden since 2026-10-07) — [`open-tasks/2026-11-03-granite-4.2-30b-lifecycle.md`](open-tasks/2026-11-03-granite-4.2-30b-lifecycle.md) |
| 2026-11-01 | Arrow IPC audit closed; glm-5.3-flash arena re-tried and rejected on measurement | Full audit of every converted model's Arrow path: owning `UnpackBatch`/`DecodeEntry` validation (SIGSEGV on a foreign shared-scale list), view payload/size checks, atomic artifact writes, zero-copy write path (`MakeBatchView`, reserved builders), gemma views, shared verifier + `lsxforge --verify`. Exhaustive `LSX_ARROW_AUDIT_FULL=1`: 29 trees / 310 158 files / 0 problems / 783 s — the full run caught that DeepSeek shared experts are raw F8_E4M3 with E8M0 128x128 tiles, not NVFP4/16 (per-16 kept for routed only). Converted the three missing trees (Qwen3.5-35B-A3B 217 s/19 GB, DeepSeek-V4.1-Flash 228 s/294 GB, DeepSeek-V4-Flash 124 s/157 GB), all verified. glm-5.3-flash neutral-store trial on real weights: SelfCheck/FSI/CODE/DET green, but 4.10 t/s at 4096 slots (57 GB) and 7.23 t/s with all 12 096 mapped (156 GB) against the arena's 12.79 t/s (171 GB) — pageable-mmap H2D; reverted, arena retained. Residuals moved to S1-S6 — [`open-tasks/2026-11-01-arrow-audit-and-arena-decision.md`](open-tasks/2026-11-01-arrow-audit-and-arena-decision.md) |
| 2026-10-31 | MiniMax-M3.1-Flash (428B/23B active) ingested as a **sibling** of minimax-m3 | New passport `minimax_moe_flash_constants.h` (theta 1e7, MSA on every layer, no MTP), shared converter with an MXFP8-dense + ModelOpt-NVFP4-expert source path (experts verbatim, divisor global scale inverted; dense E8M0 tile fed to the neutral quantizer), `lsxforge --minimax_flash`, model class `minimax-m3.1-flash`, and a `MinimaxRenderPolicy` for the M3.1 template (adaptive thinking, `<model_id>8bb8149fe</model_id>`, developer blocks, batched tool results). The parent loader/kernels/pipeline are reused; the loader now derives the dispatch boundaries and every member cross-checks them + the RoPE theta against its passport. Real conversion 236 GB -> 227 GB in ~6 min; 10.5 GiB pinned + 607 expert slots. All gates green on real weights, serial: SelfCheck PASS x3 bit-exact, FSI 3/3, CODE 3/3, DET 3/3 (template-derived goldens: the 428B QAT source has no 32 GB execution path), tool round-trip green. Perf re-seeded by the harness: **6.23/6.18 t/s** (perf/r3) at 2048 tokens, TTFT 48.8/49.2 s — [`open-tasks/2026-10-31-minimax-m31-flash-ingest.md`](open-tasks/2026-10-31-minimax-m31-flash-ingest.md) |
| 2026-10-24 | gemma-4-26b-a4b-it FSI/CODE closed: two defects, one recipe | §J resolved. **(a)** The NVFP4 logits kernel dropped the tied table's per-tensor global scale, so every logit was 7.01x hot and the soft cap saturated — greedy hid it (argmax is scale-invariant), the logit row/temperature semantics did not; pinned by a `gs = 0.25` CPU-table case (the latency test passed `gs = 1`, which is how it shipped). **(b)** The remaining flips were accumulated 4-bit body error on template near-ties: engine vs source cos 0.9992 after L0 and 0.960-0.965 at L29, identical router top-8, head error only ~1 logit while the body owned 3.46 rms — so FP16 edges were the wrong lever. Recipe ablation (validated: all-q4 reproduced the engine's flips): attn+MLP at 8 bits with the **routed experts still 4-bit** clears every flip, i.e. a second NVFP4 plane (`<name>.res`, `acc=true`) on attention + dense MLP only. All four gates now green (SelfCheck, FSI/CODE/DET 3/3), 15 GiB tree; throughput re-seeded by the harness with power/TTFT evidence 156.7/156.5 -> **128.25/128.34** t/s (the doubled attn+MLP read). The dense 31B sibling stays red on FSI-01: the same recipe does not transfer and no partial placement fits (§R) — [`open-tasks/2026-10-24-gemma4-moe-residual-plane-closed.md`](open-tasks/2026-10-24-gemma4-moe-residual-plane-closed.md) |
| 2026-10-24 | phi-4 DET-TEST-02 closed: 4-bit body + residual plane + FP16 edges | The gate was a genuine 1.1-logit near-tie (source `C` 30.75 / `CR` 29.625), and the flip was **not** the quantizer being bad: our NVFP4 reconstructs weights as well as NF4 (relL2 9.5% vs 9.1%) and, with FP16 edges, matches it on perplexity (13.88 vs 13.85, source 13.48), top-1 and a 35-prompt near-tie battery (7/9 each). A per-layer ablation showed the error is spread over all 40 layers, so no high-precision subset helps; a BF16 LM head alone still answered `CR` by 0.34. Fix: every projection carries a **second NVFP4 plane with the residual** (decoded with `acc=true`, same codec/schema/kernels) and `embed_tokens`/`lm_head` stay FP16 — effective ~8-bit weights at 2x the read. All four parity gates green (Paris ×3 bit-exact, FSI/CODE/DET 3/3), 16.6 GiB resident, 71 t/s at 2048 tokens. Also fixed the VRAM profile (it ignored the residual plane) and re-seeded the perf rows 130.91/130.31 -> 71.06/71.16 with the harness's own evidence — [`open-tasks/2026-10-24-phi-4-perf-mode-closed.md`](open-tasks/2026-10-24-phi-4-perf-mode-closed.md) |
| 2026-10-24 | phi-4 lifecycle re-audit: continuation X2, the kAuto perf row, two instrument defects | The vault had no converted artifact, so phi-4 was re-converted and every stage re-run. **Continuation (audit X2) had never been ported from granite**: prefill hard-coded `pos0 = 0` and reset `kv_rows`, so a continuation wrote K/V over cache row 0 — fixed, pinned by a fresh-prefill parity case. The `perf`-mode row (134.67, 2026-10-01) was **not a code regression**: the E4M3FN block-scale decode (2026-10-06, after the row) cost 1.2-1.5% in the streaming-GEMV inner loop, measured by an A/B against the pre-codec decode; a bit-identical single-return rewrite with an FMA subnormal lands at pow2 parity (app 133.2-133.7 = 98.9-99.3% of the row). Compiler (GCC == AOCC), warm-up, NVML, cooldown and artifact were each ruled out by measurement; the harness's own run-to-run spread on this model is ~2%. Rows re-seeded by the harness with evidence: perf **134.67 -> 130.72 -> 130.91**, r3 **128.68 -> 129.68 -> 130.31**; the target passes both modes. Instruments: the phi decode-shape bandwidth budget existed but had **no row** (silent SKIP) and now covers the real `qkv` shape, and `test_kernel_budget.h` no longer truncates hand-formatted multi-line entries on read or rewrite. Also fixed: a Debug perf run rewrote the tracked `__gpu_probe__` oracle before SKIPping — [`open-tasks/2026-10-24-phi-4-perf-mode-closed.md`](open-tasks/2026-10-24-phi-4-perf-mode-closed.md) |
| 2026-10-24 | The `model-metadata/` vault block is deleted | Only two things still lived there and neither belonged in the model vault: the passport (deleted just above) and the per-GPU perf calibration cache (empty; reachable only from an explicit calibration). `lsx_model_metadata{,_root}` and `kModelMetadataDir` are gone; the calibration moved to `lsx_cache_path` — `LSX_CACHE_DIR`, else `$XDG_CACHE_HOME/logestix` / `~/.cache/logestix` (`%LOCALAPPDATA%\logestix\cache`), temp fallback. The DeepSeek loaders' `manifests/model.index.json` vault fallback was removed too (nothing ever wrote such a file). `test_path_registry` now pins "cache root is never inside the vault"; `test_gpu_perf` asserts the same for the cache path. The directory itself was removed from the vault. |
| 2026-10-24 | Stage 7 retired: the token-translation passport is deleted | It had no consumer (`TranslateGlmToDeepSeek*` was called only from its two tests; the single `IoWorker` construction sat in an uncalled orchestrator method), the shipped handoff tokenizes with the target model's own tokenizer so ids never crossed vocabularies, and the converter's write went to a shared `model-metadata/passport/passport.bin` that the test suite had overwritten with a 268-token synthetic fixture. Removed: `lsx_token_translator.{h,cpp}`, `IoWorker`'s translator + passport parameter, `tools/lsx_token_packer.cpp`, two tests, `BuildFlashPassportTokens` and the converter write; `ingesting-a-model` stage 7 is now "nothing to do" and AGENTS.md points at the retirement. `IoWorker` stays as the plain pinned worker — [`open-tasks/2026-10-23-glm-5.3-flash-audit-closed.md`](open-tasks/2026-10-23-glm-5.3-flash-audit-closed.md) |
| 2026-10-24 | §U closed: GLM-5.3-Flash session scratch has a budget | The expert cache sized itself from a fixed 3 GiB reserve while the session's DSA KV scales with the context (0.69 MiB/position), so a ~2900-token prompt died with `out of memory` before the first token. `GlmFlashSessionScratchBytes(max_ctx)` now reports embed + KDA + DSA KV + indexer + arena (the indexer's span list is the allocator's, and its 11 allocations became one), and the driver reserves it before allocating, calling the neutral `moe::EnsureMoEExpertCacheHeadroom` to re-allocate the cache with fewer slots (64-slot floor) when short. Real weights at 2907 positions without the old `LSX_FLASH_SLOTS` workaround: 3414 MiB reserved, cache 1544 -> 1491, correct answer; the same session also closed the `vram` block (the `--vram_recompute` prefix dispatch was running the GLM-5.2 estimator, ~28x too small) and built the DSA selection oracle (source-shard host replica, 0 rank disagreements over 512 pools at slot 2304) — [`open-tasks/2026-10-23-glm-5.3-flash-audit-closed.md`](open-tasks/2026-10-23-glm-5.3-flash-audit-closed.md) |
| 2026-10-23 | GLM-5.3-Flash audited end to end and re-converted | `§P` **retracted**: the FSI red was a 128-token budget on a reasoning checkpoint — at 512 all three prompts pass with their canonical answers (the recorded "reproduces at 512 and 2048" claim is false on this conversion), so the golden's budget moved to match CODE/DET's 512 and no expectation was rewritten. Three real defects fixed in the family: the single-prompt path bypassed the chat template (raw BPE default), the renderer had drifted from `chat_template.jinja` (newlines after every turn, tool calls on the wrong line, tools block without the template's layout) and the cooperative-cancel counter incremented after the abort check, so a cancelled request reported `ok=false`/`unknown error`. SelfCheck now owns the family's answer window (the model closes an empty `<think>` before answering). New instrument: the real inventory is classified and the passport reconciled against `config.json` (39,522 assertions). Real weights: Paris ×3 bit-exact, tool round-trip, streaming/cancel, FSI/CODE/DET 3/3 each. Perf was then attributed and re-seeded (perf 14.89 -> 12.79, r3 12.50 -> 12.27): the whole drop is the expert hit rate following the *generated text*, i.e. the framing change, not a code regression — [`open-tasks/2026-10-23-glm-5.3-flash-audit-closed.md`](open-tasks/2026-10-23-glm-5.3-flash-audit-closed.md) |
| 2026-10-23 | §I closed (and §H with it): the non-split attention dot was a half-row bug, not FMA | `AttnHalfDotRow` consumed two `__half2` per step under a `hd >> 2` bound, so every score dot covered half the head (64 of 128); X16 (`f7cc177`) fixed the bound for the split path, and the 2026-10-06 non-split experiment predates it, which is the whole of the "compiler/FMA" story. The helper now sits outside the anonymous namespace and the non-split kernel calls it: Qwen3.6-35B-A3B SelfCheck/FSI/CODE/DET, Qwen3.5-9B SelfCheck, Alice and Ornith FSI/CODE/DET and gemma-MoE DET all pass (gemma FSI/CODE keep the known §J divergences). Alice, which only ever reached the non-split kernel, gains **+2.0%** (interleaved A/B: 45.6/45.4 vs 44.5/44.6 t/s) - [`open-tasks/2026-10-23-non-split-attn-dot-closed.md`](open-tasks/2026-10-23-non-split-attn-dot-closed.md) |
| 2026-10-23 | Doc hygiene: the non-defect records moved out (B, D1, F, G, E, V2) | These were not open work: `B` is the resolved GPU-keyed-budget mechanism, `D1` was closed as not-a-regression (batched expert GEMV 1.34-1.44x in isolation, no pipeline cost), `F` measured the opposite of a ramp (Xing falls 27% / 42% across a session), `G` is the completed vectorisation ledger (qwen +4.8%, gemma +3.0%, gigachat 0%; the non-split half remains §I), `E`'s last subsection (E4) closed the same day, and `V2` was refuted (`max_open` bounds RSS monotonically, 3.2 / 7.0 / 31.1 GB at cap 64 / 512 / 8192). Kept for their method notes, not as tasks — [`open-tasks/2026-10-23-doc-hygiene-and-non-defect-records.md`](open-tasks/2026-10-23-doc-hygiene-and-non-defect-records.md) |
| 2026-10-23 | E4 closed: opt-in sampling end to end | `SamplingParams` in `ModelRequest`; a deterministic device sampler (exact radix top-k, conservative 4096-bin top-p, counter-based RNG) behind `SamplerFor`, with greedy bit-identical and every fused-argmax family keeping its fast path via `ISampler::IsGreedy()`. Wire v3 carries the fields and v2 requests still decode as greedy; TaskMosaic/Fabric plumb them; a guard scans every `Infer` for `SamplerFor(req.sampling)`. `test_sampling` pins the distribution, the top-k set and seed reproducibility — [`open-tasks/2026-10-23-sampling-neutral-session-and-environment-oracle.md`](open-tasks/2026-10-23-sampling-neutral-session-and-environment-oracle.md) |
| 2026-10-23 | S/5b closed: KV session is neutral | `KvCache::Allocate`/`EnsureCapacity` take `attn::KvGeometry`; `QwenKvCache`/`QwenGenerationResult` became `attn::KvCache`/`attn::GenerationResult`; the per-request shallow pinned-weight view is gone; KV parity stays bit-exact — [`open-tasks/2026-10-23-sampling-neutral-session-and-environment-oracle.md`](open-tasks/2026-10-23-sampling-neutral-session-and-environment-oracle.md) |
| 2026-10-23 | §T cost closed: hoisted routed prequant | Batched SwiGLU prescale+quant for all K slots (three launches per layer instead of per slot), bit-identical to the per-slot chain (`memcmp` test that also asserts the branch ran). Mistral 41.3 -> 42.2 t/s; `LSX_MOE_TC_HOIST=0` keeps the old path. The three Mistral goldens remain declared debt — [`open-tasks/2026-10-23-sampling-neutral-session-and-environment-oracle.md`](open-tasks/2026-10-23-sampling-neutral-session-and-environment-oracle.md) |
| 2026-10-23 | §W closed: the environment oracle | A 512 MiB DRAM-bound copy probe (a 64 MiB probe measured L2 speed and was rejected) is recorded as a synthetic baseline entry; `AttributeDegrade` classifies a low model reading as ENVIRONMENT only when the probe is also down by at least as much. Perf suite re-seeded on 2026-10-23: the 10-23 strict-mode re-seeds reached the rows that had not been re-run (alice r3 10.14 -> 44.25, mistral r3 14.68 -> 41.57, qwen3.8-flash-next r3 6.40 -> 26.47, gemma-31b r3 41.00 -> 47.21, phi-4 r3 122.26 -> 128.68), nemotron-nvfp4 has baselines (127.27 / 120.35) — [`open-tasks/2026-10-23-sampling-neutral-session-and-environment-oracle.md`](open-tasks/2026-10-23-sampling-neutral-session-and-environment-oracle.md) |
| 2026-10-23 | r3/bit-exact restored to streaming GEMV; qwen3.6 closed | The strict mode had been made correct by disabling `gemv_large`, which dropped qwen3.5-9b r3 from 111 to 19.3 t/s. The strict scope now pins the large-N and <=32 MiB streaming GEMV as its canonical route (`LsxBitExactMode`), ignores `LSX_GEMV_LARGE`/`LSX_GEMV_MAX_MIB`, and routes qkv/z through it for every H. Re-seeded r3 references: qwen3.5-9b **111.06 -> 147.0**, qwen3.8-27b **39.03 -> 52.27**, qwen3.6-35B-A3B **13.32 -> 73.39**. SelfCheck/FSI/CODE/DET for all three plus Ornith are green; the TC path stays off. — [`open-tasks/2026-10-23-residual-closures.md`](open-tasks/2026-10-23-residual-closures.md) |
| 2026-10-23 | Ornith perf baseline re-seeded (documented cost) | 167.06 -> **160.20 t/s** on two cooled runs. The reference predates the neutral MoE migration and the overflow-proof batched SwiGLU (`8e34314`), whose own note records the same ~4% cost (Qwen3.6 166.7 -> 161.3); this is the accepted correctness fix, not a regression. r3 was then re-seeded 12.70 -> **72.02**. A router launch-geometry experiment (128 -> 64 threads) was neutral (160.5 vs 160.0) and reverted; `ornith.router.decode` now has a 14.0 us max budget row (11.84 us observed) instead of SKIPping. — [`open-tasks/2026-10-23-residual-closures.md`](open-tasks/2026-10-23-residual-closures.md) |
| 2026-10-23 | C1 closed: GPU budget tests serialized | `RESOURCE_LOCK "gpu"` added to all kernel-budget suites (plus Ornith, now budget-bearing) in `tests/CMakeLists.txt`; a `ctest -j` sweep can no longer overlap two latency/bandwidth samples. The measured flake was contention: five concurrent copies failed 4/5, 80 solo runs were green. Best-of-5 remains the second layer. — [`open-tasks/2026-10-23-residual-closures.md`](open-tasks/2026-10-23-residual-closures.md) |
| 2026-10-23 | D2 closed: Xing decode 123.2 -> **146.65 perf / 123.92 r3** | Expert-batch GEMV now uses the dense kernel's four 128-bit activation loads (bit-identical; gate/up 13.5 -> 9.24 us, down 12.95 -> 9.05 us); the second, unread router projection was deleted (6.4 us x 38/token); mHC gates+collapse merged with the 57 KB prefetch under the Sinkhorn (9.10 -> 7.27 us, bit-identical); RMSNorm moved from a cold-latency-bound scalar 256 block to float4/512 (7.99 -> 1.75 us in situ). All SelfCheck/FSI/CODE/DET and the fast suite pass. Found and fixed a harness defect: `test_perf_baseline` never initialized the logging core, so it measured the synchronous-logger path (134.4 vs 146.6 canonical); other families' rows will re-seed upward on their next run. The 2048 profile now puts attention split-K at 26.8% (~84 MB KV/token, roofline); remaining work is KV compression and prefill, not decode serialization. — [`open-tasks/2026-10-23-residual-closures.md`](open-tasks/2026-10-23-residual-closures.md) |
| 2026-10-23 | Granite tool round-trip and the qwen fused-linear test fixtures | §U closed: the RED was prompt framing, not generation arithmetic — the native `granite_tool_format` renderer/parser (commit `de0b160`) restores `get_weather`, and `test_granite_dense_selfcheck` passes with `logits_hash=3d1604aa75c99a1f` unchanged. The `test_qwen_dense_fused_linear_attn` failures were fixture bugs: the synthetic E4M3 scale cycle included `0x7F`, the reserved NaN; the host z reference added the projection twice; and the tiny full-layer geometry used `gs=0.5`, overflowing FP16 after the NaN was removed. All 43,675 assertions now pass. — [`open-tasks/2026-10-23-residual-closures.md`](open-tasks/2026-10-23-residual-closures.md) |
| 2026-10-22 | Inference audit — memory/lifecycle P1s and the cheap hardening closed | Mistral/Xing continuation no longer writes past the KV cache and all four families share one continuation contract (Mistral's `last_tok`, Granite's `pos0 = kv_rows`); fused w13 writes the up half to `d_yg + I` (real OOB in Mistral, memcheck-clean now); the E1 depth counter is process-global and cross-TU-tested; `--bit-exact` no longer routes through `gemv_large`; six busy-polls became bounded `WaitEventChecked`; the split-K partials scratch is per-stream; RPATH, `head_dim`/GDN validation and the prequant double-activation fallback fixed; §V corrected. Fast 98/98 without `LD_LIBRARY_PATH`; dense parity green except the known phi DET-02 / gemma FSI/CODE reds. All X1-X16 audit items are closed (second pass: GLM embed epoch, per-stream scratch, Arrow view range checks, split-KV dot bound) — [`open-tasks/2026-10-22-inference-audit-closed.md`](open-tasks/2026-10-22-inference-audit-closed.md) |
| 2026-10-17 | Sequential-dispatch invariant made machine-checked (E1) + alice aligned to the fleet logits contract (E2) | **E1:** with no parallel dispatch planned, the process-global kernel-routing toggles' single-dispatch assumption is now enforced — `LsxFastMathScope` keeps a depth counter and logs a loud violation on overlap, pinned by two guard cases (one spawns a second thread to prove the counter is shared state, so the guard is not vacuous). **E2:** alice now publishes the LAST logits row like every other family (it published the FIRST). The original worry — that this would move alice's DET/FSI goldens — was wrong: those harnesses read `gen_tokens`, never `final_logits`, so nothing was re-captured. Real-weight alice selfcheck/DET/FSI/CODE all green. [`open-tasks/2026-10-17-lifecycle-e1-e2-closed.md`](open-tasks/2026-10-17-lifecycle-e1-e2-closed.md) |
| 2026-10-17 | mimo uploaded weights through a host-side BF16→FP16 pass (E3) | `UploadHalf` decoded every BF16 element into a host `std::vector<__half>` and issued a synchronous pageable `cudaMemcpy` on the DEFAULT stream — `(void)st;` ignored the model's stream. Routed to the existing model-neutral device kernel `fastload::Bf16ToHalfInPlace` (the one GLM and the DeepSeek fast loader use), stream-ordered, with no host staging buffer. Bit-exact: the kernel is the same arithmetic the host loop ran, and `test_deepseek_moe_load_fast.cu` already pins it against a host reference including ragged tails. `UploadBf16`/`UploadF32` had the same `(void)st;` defect with no conversion involved and became stream-ordered too - [`open-tasks/2026-10-17-mimo-host-bf16-upload-closed.md`](open-tasks/2026-10-17-mimo-host-bf16-upload-closed.md) |
| 2026-10-17 | gemma-MoE synthetic decode failed at once and reported `ok=true` (E1e) | `RunGemmaGeneration` ended by assigning the last-step logits read-back status back to `e`, so any error raised inside the decode loop was overwritten by a successful copy: `ok=true`, empty `error_msg`, and an answer truncated at the step the loop died on. The loop itself was correct. Combined with a synthetic fixture whose `moe_intermediate_size` was 16 while the routed-GEMV chain requires K % 32 == 0, the synthetic MoE decode had never actually run. Fixed both: the decode status is preserved and the copy is best-effort; the fixture intermediate is a documented 32-multiple (the dense one already carried that comment). `test_gemma_moe_model` now carries the full cooperative-cancel gate. This is what exposed it: every gemma-MoE synthetic assertion was satisfied by one token - [`open-tasks/2026-10-17-gemma-moe-swallowed-decode-error-closed.md`](open-tasks/2026-10-17-gemma-moe-swallowed-decode-error-closed.md) |
| 2026-10-17 | A second model instance decoded from the previous one's fused GEMM weights (E1d) | Four process-lifetime caches in the shared neutral layers (`sh_cache`, `g_fw_cache`, `g_tma_cache`, `LsxTcFusedOutputScratch`) are keyed by a weight's DEVICE ADDRESS and never invalidated; CUDA recycles addresses across model lifetimes, so an instance created after another was unloaded inherited entries built from the previous model's weights. Measured: the same request answered with a 5-token EOS on a fresh instance and 32 tokens of unrelated text on the next one. Fixed with a monotonic weight epoch bumped by `ModelBase` on every weight lifetime change; rebuilding is value-identical, so no golden moved. The TMA map was the sharpest case - a stale hit loads through a freed address. `test_qwen_moe_exp_eos_selfcheck` gained the cross-instance gate - [`open-tasks/2026-10-17-qwen38-instance-state-closed.md`](open-tasks/2026-10-17-qwen38-instance-state-closed.md) |
| 2026-10-17 | Lifecycle audit: E1a/E1b/E1c closed | Cooperative cancel now pinned for all 22 `kLifecycle` families via a shared gate (pre-set + callback + control); the guard's abort marker now requires a poll and every row names the TU owning its generation loop. **Two real defects found by the work**: a cancel poisoned the NEXT qwen3.8-flash-next request (sticky `cudaEventElapsedTime` on an unrecorded event), and that driver emitted one bulk drain at the end - now per-step, with early EOS stop. E1b resolved by documentation: the shared lock orders `Infer` against `Unload()` only, so eight family comments and the guard rationale were corrected to name the one-`Infer`-per-instance precondition. It also surfaced §E1d, a separate determinism defect, fixed the same day - [`open-tasks/2026-10-17-lifecycle-abort-contract-closed.md`](open-tasks/2026-10-17-lifecycle-abort-contract-closed.md) |
| 2026-10-14 | qwen3.8-flash-next: the arena is required (V1, negative) | The bounded pinned window buys nothing when the decode working set is the whole expert set: views cap 4096 = 10.9 GB but 9.2 t/s, pinned window cap 2048 the same, and pinning everything costs 127 GB for 29 t/s. `cudaHostRegister` over file-backed MAP_SHARED views is refused. qwen3.8 stays on its arena (63.6 GB, 32.0 t/s, re-measured) - [`open-tasks/2026-10-14-qwen38-arena-required-negative.md`](open-tasks/2026-10-14-qwen38-arena-required-negative.md) |
| 2026-10-14 | Batched TC MoE swiglu overflowed FP16 | `moe_swiglu_quant_batch_kernel` stored `silu(gate)*up` to FP16 before quantizing, so any product above 65504 became `+inf` and saturated the block scale; the per-slot path had been fixed for this long before, the batched path never had. Stage-by-stage the GEMM was bit-exact (`nd=0/32`) — the bug was the swiglu between `w13` and `w2` (`h=inf`). Fixed with a per-slot power-of-two prescale folded back in the reduce; both kernel cases green with assertions untouched (998/877), real Qwen3.6 batch-on/off now identical, gates still pass. Also repaired the `LoadMoEExpertStore` line `bad0792` had deleted - [`open-tasks/2026-10-14-moe-batch-swiglu-overflow-closed.md`](open-tasks/2026-10-14-moe-batch-swiglu-overflow-closed.md) |
| B1 — expert pinning not implemented on Windows | platform-gated in `test_kimi_moe_loader.cpp` |
| D1 — no test drove `RunGlmFlashGeneration` | real-model generation case added |
| D2 — hardcoded POSIX temp paths in tests | `std::filesystem::temp_directory_path()` |
| E1 — 14 test executables were one binary | per-`.cu` object libraries |
| G1 — toolkit mismatch caught only after the build | configure-time check in `CMakeLists.txt` |
| A2 — MiMo wrong top-1 | closed 2026-10-02: no implementation bug, the checkpoint is bad (vendor oracle matches) — [`open-tasks/2026-10-02-mimo-closed.md`](open-tasks/2026-10-02-mimo-closed.md) |
| 2026-10-14 | Routed-expert host storage: the arena retired for qwen MoE / Ornith / Mistral | `MoEExpertRecord` reduced to immutable geometry, payload moved to the neutral `ExpertStore`, six-segment staging from mmap views; qwen3.6 17.5 → 7.0 GB at unchanged 141 t/s, qwen3.8 was tried on the neutral store and REVERTED to its arena — the decode working set is the whole expert set, so a bounded pinned window gives no speedup (9.2 t/s) and pinning everything costs 127 GB against the arena's 63.6 GB; `test_expert_store_guard` locks what did migrate - [`open-tasks/2026-10-14-expert-store-arena-retired.md`](open-tasks/2026-10-14-expert-store-arena-retired.md) |
| 2026-10-10 | Neutral routed-MoE layer + ornith de-duplication + the shared TC M-tile defect | `TcGemmKernel` wrote only rows `[0,16)` of its 64-row M tile (`wm` computed, never used) - fixed in all four row computations, TC shape sweep extended to seq 17/18/33/64; `ornith_moe_inference.cu` (2182 lines, 97% copy, already drifted) deleted in favour of the neutral `lsx_moe_inference` layer, expert-store loader cluster de-duplicated across qwen/glm-flash/ornith (-2776 lines); Ornith fused-expert split and gemma `final_logits` now read back by tests - [`open-tasks/2026-10-10-neutral-moe-and-tc-mtile-closed.md`](open-tasks/2026-10-10-neutral-moe-and-tc-mtile-closed.md) |
| Inference-conformance review (9 deviations) | all fixed in dedicated commits |
| "Closed in this session" list | 15 items, all closed |
| D — KV cache layout `[head][pos][dim]` | closed 2026-10-05, **negative**: implemented, measured, rejected. Head-major reads K/V 6-10% faster in isolation and is 2-6% slower end to end, so pos-major ships; the reusable stride layer, the neutral layout knob, a bit-exact parity gate and a streaming-read ceiling probe all shipped. A related `kAttnSplits = 4` finding (+10% at ctx 8191) was rejected on bit-identity. Two pre-existing defects fixed: a test iteration that verified nothing, and a wrong host pitch in the KV readback |
| L/M/N — NVFP4 E4M3FN codec, Nemotron-NVFP4 ingestion, granite tool round-trip | closed 2026-10-07 — [`open-tasks/2026-10-07-nemotron-nvfp4-and-codec-closed.md`](open-tasks/2026-10-07-nemotron-nvfp4-and-codec-closed.md) |
| Ornith-1.5-35B-A3B-MLX — prompt-framing defect (`BuildPromptInputIds` dropped the required leading system turn) | found + fixed 2026-10-07 — [`open-tasks/2026-10-07-ornith-framing-closed.md`](open-tasks/2026-10-07-ornith-framing-closed.md) |

Full text: [`open-tasks/2026-09-30-cuda-review-windows-bringup.md`](open-tasks/2026-09-30-cuda-review-windows-bringup.md)
and [`open-tasks/2026-10-02-mimo-closed.md`](open-tasks/2026-10-02-mimo-closed.md).

Added 2026-10-05:
[`open-tasks/2026-10-05-xing-kv-layout-closed.md`](open-tasks/2026-10-05-xing-kv-layout-closed.md)
— KV cache layout, closed negative.

Added 2026-10-07:
[`open-tasks/2026-10-07-nemotron-nvfp4-and-codec-closed.md`](open-tasks/2026-10-07-nemotron-nvfp4-and-codec-closed.md)
— the NVFP4 block-scale codec is now the complete E4M3FN table (mantissa +
subnormals, bit-identical to the NVIDIA source); the
NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 ingestion is done and its real-weight
SelfCheck passes (router-load dtype fix); the granite-4.2-30b tool round-trip is
recovered.

Added 2026-10-07:
[`open-tasks/2026-10-07-ornith-framing-closed.md`](open-tasks/2026-10-07-ornith-framing-closed.md)
— converting Ornith-1.5-35B-A3B-MLX surfaced a real prompt-framing defect: the
hand-rolled `BuildPromptInputIds` omitted the leading system turn this reasoning
fine-tune requires, so generation (Paris selfcheck, `--run`, `--testprompt`)
collapsed into repetition. The frame now goes through the native Qwen renderer
(mirrors Nemotron), the conversation path injects a default system turn, and the
real-weight selfcheck asserts the frame shape. FSI/CODE/DET goldens remain
declared debt.

## Maintenance rule

When an item in this file is resolved, move it to `docs/open-tasks/<date>-<slug>.md`
(one file per work session, dated the day the work happened), replace it here
with a one-line row in the Archive table above, and keep only genuinely open
work in `docs/OPEN_TASKS.md`. The index is
[`open-tasks/README.md`](open-tasks/README.md). This keeps the live file short
enough to read at a glance, which is its only purpose.
## R15. Kimi-K3 (2.8T / ~104B active) — ingestion landed; SelfCheck green, parity runs pending (2026-11-03)

Branch `feat/kimi-k3`. A new `kimi` variant (`k3`, `kType = kimi-k3`, format
tag `logestix-kimi-k3-nvfp4`): 93 layers (dense SiTU MLP on layer 0, MoE
elsewhere), 69 **KDA** linear-attention layers interleaved with 24 **gated
NoPE MLA** layers (0-based `{3,7,…,91,92}`), **AttnRes** block 12, **Stable
LatentMoE** (hidden→3584 latent + RMSNorm, 896 experts top-16, 2 shared experts
at 6144), SiTU-GLU (β=4, linear β=25), 7168H, vocab 163840.

**Measured on real weights** (1.5 TB MXFP4 source → 1.5 TB NVFP4 tree: 23 GB
pinned + 8.9 GB streamed plane + 1.4 TB routed experts; RTX 5090, CUDA 13.4):

| gate | verdict |
|---|---|
| SelfCheck (XTML-framed) | PASS x3 cycles bit-exact: `The answer is Paris.`, `logits_hash=5c1e227265d6a27b` on all three, top-1 1008 |
| tokenizer parity | 23/23 against the checkpoint's `tiktoken.model` (`tokenizer_kind 2`) |
| tool codec | byte-exact against `encoding_k3.py` frames (synthetic tokenizer) |
| FSI / CODE / DET | goldens committed (template-derived); **not yet run** on real weights |
| decode rate | **2.13 s/token (perf) / 2.04 (r3)** — harness rows `0.47` / `0.49` t/s at 2048 tokens, seeded 2026-11-06. Earlier figures on this row were log-derived prose: 2.96–2.97 at the shipped configuration (expert window 65 536, no eager warm, 64-token cycles) and 6.95 at the old 512-slot window against a 1472-expert working set — see §R17 P5 and its archive entry `open-tasks/2026-11-05-kimi-k3-decode-window-and-eviction.md` |
| perf row | **seeded 2026-11-06**: `perf` 0.47 t/s, `r3` 0.49 t/s, 2048 tokens, TTFT 131.9 / 102.1 s |

**Open, explicitly.**

- K1. FSI/CODE/DET were not executed to completion. At the measured decode rate
  (see K3 below) the budgets (512–2048 tokens × 3 prompts, DET × 3 stress runs)
  are multi-hour runs; the goldens and tests are in place and SKIP (4) until
  then. Do **not** read the matrix row as green.
- K2. The parity goldens are **template-derived** (`tools/kimi_k3_template_goldens.py`):
  the 1.5 TB source needs `fla-core` and a multi-GPU runner, so no
  model-defined golden can be captured here. `expected` is the prompt set's own
  Expected Template; the frame is the checkpoint's own XTML
  (`encoding_k3.py`), and `answer_start_token_id` cuts at `<|sep|>` (163589).
  Kimi-K3 answers inside its thinking channel, so a text-based cut would read
  the reasoning as the answer — the same trap as Mistral-Large-4 (L1/L3).
- K3. Perf rows are unseeded. The decode path streams 16 experts × 92 layers
  (~27 GB) per token from the 1.4 TB host store plus the 9.5 GB latent/shared
  plane; expect ~0.2–0.4 t/s. Seed through the harness at the canonical
  2048-token protocol before quoting a number.
- K4. `A_log` is read **per channel** (the checkpoint ships 128 entries =
  `head_dim`), while the `fla` reference indexes it per head (96). The
  checkpoint shape is authoritative and SelfCheck is green with it, but a
  source-side comparison (impossible here) is the only definitive check.
- K5. The latent/shared plane is streamed (~103 MiB/layer/step) because pinning
  it needs ~34 GiB and does not fit this card. There is deliberately no
  "pin it all" knob: it cannot succeed here, and a knob that only works on
  unseen hardware is untested code.
- K6. Context is capped at 4096 (`LSX_KIMI_K3_MAX_CONTEXT`). The 1 M-token
  window the release targets needs ~84 GB of latent KV at 24 layers and is
  out of scope for this card.
- K7. A load is flaky under VRAM pressure: with ~8 GiB free before `Load`, the
  24.5 GB pinned graph fails mid-graph (`layer N ... upload failed`, logged).
  The model needs a card that can hold the pinned set plus ~2 GiB headroom; a
  clean "not enough VRAM" pre-check is a follow-up.

## R17. Kimi-K3.1-Pro (2.8T / ~104B active) — Gated DSA variant landed, parity sweep is debt (2026-11-05)

Branch `feat/kimi-k31-pro`. Sibling variant of Kimi-K3 (`kimi-k3.1-pro`, files
`kimi_moe_pro_*`, format tag `logestix-kimi-k31-pro-nvfp4`): the same KimiLinear
tower (93 layers, 69 KDA + 24 gated NoPE MLA, AttnRes 12, Stable LatentMoE, SiTU)
with the 24 full-attention layers replaced by **Gated DSA** — a Lightning
Indexer per full-attention layer selecting `index_topk = 2048` keys per query.
Landed (stages 1–8 of the Model Ingestion Protocol): frozen passport,
converter (`lsxforge --kimi_k31_pro`, wrapper-level variant gate both ways),
loader (five BF16 indexer tensors per DSA layer pinned as FP16, never quantized,
plus the `index_norm_eps` cross-check), the four indexer kernels + selectable MLA
attention, the pipeline's indexer key cache with the dormant-at-`n_pos <= 2048`
DSA branch, `KimiK31ProModel` + registry + the lifecycle contract, and the four
real-weight targets with their template-derived goldens and the perf row.
Stage 9 landed too, in place at `/mnt/fast_storage/huggingface/Kimi-K3.1-Pro/logestix`
(1.5 TB source → **1.5 TB tree** in 17 min: 1.4 TB routed-expert arrows, 23 GB
pinned graph, 8.9 GB streamed latent/shared plane, `total_experts 82432`, 2120
pinned + 460 streamed tensors, 165 vision + 3 projector tensors stripped and
counted). On the real weights: **SelfCheck Paris ×3 bit-exact**
(`logits_hash 98ec1bbb372c9181`, top-1 40) and the **native XTML tool
round-trip** `get_weather {"city":"Paris"}` — both green in
`test_kimi_pro_selfcheck`. Model report:
`docs/model-reports/2026-11-05-Kimi-K3.1-Pro.md`.

**Open, explicitly.**

- P1. The indexer's arithmetic is **reconstructed**, not verified. The
  checkpoint ships no modeling code (`configuration_kimi_k3.py` only), so the
  semantics come from the tensor shapes, `dsa_config` and the repository's prior
  art (`deepseek_moe_pro` CSA, `mistral_moe_large4` DSA). The three open points
  are enumerated in the design (§3.2): the indexer's input is taken to be the
  post-`input_norm` hidden (the DeepSeek contour; the alternative reading is the
  raw AttnRes mixture and would be a one-line change), `index_skip_topk_offset`/
  `index_topk_freq` are not implemented as knobs, and the top-k does **not**
  force the current position in. None of them is reachable at the canonical
  2048-token protocol, where selection is dormant — which is also why no gate
  covers them: **do not read the variant's green gates as DSA verification.**
- P2. The **kimi-k3 synthetic fixture is input-invariant**: two different
  prompts produce bit-equal logits from the converted 4-layer tree (measured on
  the plain Kimi-K3 tree as well, so this predates the variant). Every
  model-stage assertion in the family is therefore a smoke assertion — finite,
  deterministic, lifecycle-correct — and cannot observe a wrong attention key
  set. The DSA branch is instead pinned by exact step/key counts plus a
  score-vector/selection comparison against an independent top-k, and by the
  kernel-level host references. Making the fixture input-sensitive (e.g. a
  scale that keeps the residual from being dominated by a constant block) is the
  instrument this family is missing — `docs/OPEN_ENDEDNESS.md` §"which
  instrument is missing".
- P3. **PERF HALF CLOSED (2026-11-06); the FSI / CODE / DET half is still debt.**
  All three kimi families now have harness rows in `tests/perf/perf_baselines.json`,
  recorded through `test_perf_baseline` (never by hand), 2048 tokens, the recorded
  prompt, GPU-exclusive:

  | model | perf (kAuto) | r3 (kOff) | s/token (perf / r3) | TTFT (perf / r3) |
  |---|---|---|---|---|
  | `kimi-k2.7-code` | 5.05 t/s | 5.04 t/s | 0.20 / 0.20 | 1.0 s / 1.1 s |
  | `kimi-k3` | 0.47 t/s | 0.49 t/s | 2.13 / 2.04 | 131.9 s / 102.1 s |
  | `kimi-k3.1-pro` | 0.50 t/s | 0.51 t/s | 2.00 / 1.96 | 165.1 s / 92.3 s |

  Two things this changed. (a) K3 and Pro had **no rows at all** before — every
  number quoted for them until now was log-derived, and the entry said so. (b) The
  prose figures they replace (2.96 s/token for K3, 3.23–3.35 for Pro, P5) are
  ~1.5x apart from the harness rows on the same tree and the same binary; the
  difference is cache state and TTFT accounting, which is why the rows are now
  the quotable form and the prose is not. Also note **Pro measured faster than
  K3** (2.00 vs 2.13 s/token), which removes the last trace of P5's premise: the
  Pro is not the slower of the two.
  The FSI / CODE / DET cells remain `not run` — that half of the sweep is
  unchanged and still owes ~7 h of exclusive GPU at the measured rate. The cost
  is measured, not guessed, and it shrank again when the rows above replaced the
  prose rate: recomputed at the **harness-measured 2.0 s/token** for this tree
  (window 65 536, no eager warm — see P5; the figures below were first written at
  the pre-fix 0.14 t/s, then at 3.2–3.35):

  | target | generations × tokens | wall clock at 2.0 s/token |
  |---|---|---|
  | `test_kimi_pro_fsi_selfcheck` | 3 × 2048 | ~3.4 h (+load) |
  | `test_kimi_pro_code_selfcheck` | 3 × 512 | ~0.9 h (+load) |
  | `test_kimi_pro_det_selfcheck` | 3 × 3 × 512 | ~2.6 h (+load) |

  (`test_perf_baseline_kimi_k31_pro` used to be the fourth row of this table; it is
  **done** — its two modes are in the table at the top of this entry.)

  so that sweep is ~7 h of exclusive GPU on this host at the now-measured
  2.0 s/token (it was ~32 h at the pre-fix rate, ~14.7–15.3 h when these figures
  were first written as prose, and ~11 h at the 3.2–3.35 estimate). FSI was
  actually started (release tree) and stopped 97 min in,
  still inside FSI-TEST-01; the interruption made the harness emit three empty
  generations, and **an in-flight kill is not a verdict** — the cells stay
  `not run`. The ctest budgets still describe the pre-fix worst case (FSI
  54 000 s, CODE 18 000 s, DET 43 200 s, perf 43 200 s), which is now ~4x
  generous rather than tight; leave them, since a budget may only be wrong in
  the safe direction. To close this item, run each target alone (GPU exclusivity)
  with `LSX_FSI_GOLDEN_DIR`/`LSX_CODE_GOLDEN_DIR`/`LSX_DET_GOLDEN_DIR` pointing
  at `tests/fixtures/<stage>`, then let the harness seed
  `tests/perf/perf_baselines.json` — never by hand. Note what a green row would
  and would not mean: the goldens are **template-derived** (P4), and no gate
  observes the indexer (P1/P2).
- P4. **The FSI/CODE/DET goldens are template-derived**, payload-identical to
  `kimi-k3`'s (same XTML frames, same prompt-set Expected Template values,
  `tools/kimi_k3_template_goldens.py --model kimi-k3.1-pro`): the 1.5 TB MXFP4
  source needs `fla-core` and a multi-GPU runner, so no source oracle can be
  captured on this card. A green row pins the converted path against the
  template, never source parity.
- P5. **RESOLVED (2026-11-05) — archived.** The "4x slower than Kimi-K3" was
  never a regression: that baseline was prose committed before the run it
  described (K3's own log says 6.66–6.79 s/token, and at identical binary,
  geometry and window the two models measured 6.95 vs 6.98 — ratio 1.00). The
  defect it hid was the shared `ExpertStore`'s window: `max_open` 512 against a
  1472-expert working set (92 MoE layers × top-16, each visited once per token),
  i.e. a cyclic scan with a provably **zero** hit rate. Fixed Kimi-locally —
  clamp up to the working-set floor plus a default of 65 536, and
  `warm_on_load = false` because the eager pre-parse walks experts in path order
  and cost 156 s for nothing — which is **6.95 → 2.96 (K3) / 3.23–3.35 (Pro)
  s/token** at ~5 s load, bit-exact *as measured then*; the 2026-11-06 harness
  rows are the current rate (**K3 2.13/2.04**, **Pro 2.00/1.96** s/token — see
  P3's table), so treat the figures in this bullet as history. P5a closed with it: an evicted slot is now
  retired with a CUDA event instead of `cudaStreamSynchronize` (**20.3 → 0.07 ms**
  per eviction; 6.95 → 6.26 at a 512-slot window). Full trail — the sweep table,
  the two prefetch designs measured and rejected, and the 6.6% mistral-small-4
  regression this fix itself introduced and how it was found:
  [`open-tasks/2026-11-05-kimi-k3-decode-window-and-eviction.md`](open-tasks/2026-11-05-kimi-k3-decode-window-and-eviction.md).
- P6. **Pinned staging ring: built, tested, bit-exact — and NEUTRAL on this host
  (2026-11-05).** The entry used to say the remaining lever was to "stage layer
  N+1 while layer N computes". That is impossible: the expert ids of a layer are a
  function of the previous layer's output, so there is nothing to stage until the
  layer that would have been overlapped has finished. What was built instead is
  the other structural change — a pinned staging ring in the shared
  `ExpertStore` (persistent pread pool + per-slot DMA events, layer-wide
  prefetch), off by default, with `kimi-k3.1-pro` wired behind
  `LSX_EXPERT_STAGE_RING`. Design and full measurement table:
  [`superpowers/specs/2026-11-05-expert-stage-ring-design.md`](superpowers/specs/2026-11-05-expert-stage-ring-design.md).

  Measured, and this is why it is not switched on: the host-side copy cost drops
  62% (118.0 → 45.3 s of `cudaMemcpyAsync` inside a 32-token run) and the wall
  does not move (119.4–121.2 s with the ring against 122.5 s without). Two
  reasons, both measured. (a) This host keeps 665 GB of 750 GB in buff/cache, so
  the pageable path already copies at **23.7 GB/s**, not the 5.2 GB/s of the cold
  probe — pinning cannot beat a copy the cache already serves. (b) Expert staging
  is a minority of the traffic: **61.8 GiB/token goes H2D, routed experts are
  25.5 GiB (41%), and 36.4 GiB/token is the streamed latent/shared/dense plane**,
  which never enters `ExpertStore`. A perfect expert path therefore touches at
  most 41% of the copy time.

  So the lever this entry named was mis-scoped, and the honest remainder is the
  59%, not the 41%. Two follow-ups, both leaving the ring parked:
  - the **canonical 2048-token rows were not run** for either arm (~2.1 h each);
    the cells are `not run`, not a pass. They are what a future reader needs
    before quoting any of this;
  - the cold-cache win (26.9 ms vs 54.8 ms per layer, probe on real files) is
    **unverified end to end**, because this host's cache holds the tree. GPUDirect
    Storage would remove the host copy from the 59% instead of optimising the 41%
    — `nvidia_fs`/`libcufile` are still absent here.

## R6. Recursive Flint (39.8B-A2.5B) — ingestion landed, expert precision + goldens open (2026-10-30)

**Landed.** Family `flint` (`logestix-flint-nvfp4`) is registered and runs on the
real 75 GB checkpoint converted in place (`/mnt/fast_storage/huggingface/flint`,
22 GB tree, 12 032 routed experts resident in per-layer device slabs). All
stages of the Model Ingestion Protocol exist with their tests: converter
(synthetic, `convert`), loader (`fast`), kernels (`inference`), model lifecycle,
tokenizer (`tokenizer_kind` 4 — the 3-split digit/CJK + o200k case-run
pre-tokenizer; **parity 27/27 against the checkpoint's own `tokenizer.json`**),
native tool codec (**byte-identical to `chat_template.jinja`** on 5 captured
cases, including nested/parallel calls) and SelfCheck (**Paris ×3, bit-exact
`logits_hash`**, 49 layers / 22 266 MiB resident).

**What the checkpoint forced into the engine (measured, not stylistic).**
Flint carries *massive activations*: the source (BF16, `tools/flint_oracle.py`)
reaches ~1.4e4 in the residual at layer 48 and one layer amplifies a
29-magnitude input ~300x (gate projection 5.3e4, SwiGLU product ~1.5e8). FP16
cannot hold that, so this family keeps
- the residual stream in **FP32** (`RunFlintEmbedF32`, `RunFlintNormF32In`,
  `RunFlintAddHalfToF32`, `RunFlintAddF32`), and
- the FFN intermediates in range by **power-of-two scaling**
  (`kFfnInScale` 64 into the gate/up GEMV, `1/8192` out of the SwiGLU,
  restored on the accumulate/combine), so every projection still runs the FP16
  NVFP4 kernels.

Without this, layer 47 produced NaN on any prompt whose earlier tokens spiked
(the router then read uninitialised expert ids — that path is now also
defensive: a NaN score falls back to the lowest untouched ids instead of an
out-of-bounds shared read).

**Open: routed-expert precision.** A second NVFP4 plane (w − dequant(base),
`acc=true`) was added on the dense side — attention q/k/v/o, the dense-MLP
prefix and the shared expert (`"residual": "attn+mlp+shared"`, +343 records).
Measured against the CPU oracle on a 160-token tool prompt, the first-step
top-1 logit moved from -2.10 to -1.06 (prefix 56: -3.70 → -0.99). The remaining
divergence is the **routed experts at 4 bits**: top-1 agrees at prefix 8/16/32
and flips at ~64 (engine `4067` vs source `3705`). An 8-bit expert
representation needs +21 GB (4-bit set 21.3 GB + its residual plane), and the
card has ~9 GB free after the 22.3 GB resident tree, so the placement that would
clear it does not fit — the same conclusion as gemma-4-31b dense, measured
rather than assumed. Options for a later session: (a) stream the expert residual
plane from a host mmap with a slot LRU (≈282 MB/token at the observed routing
density), (b) a bigger card, (c) accept 4-bit and keep the parity rows declared.

**Golden debt.** FSI/CODE/DET goldens are *not* captured: the only oracle is the
CPU reference (`tools/flint_oracle.py`, 3-5 s/token), and the capture set is 9
prompts × 256-512 tokens plus DET's 3 repeat runs — hours. The oracle itself is
committed and was used to validate the engine (control prompt: 32 greedy tokens
token-for-token identical). The three targets therefore SKIP, with rows in
`test_model_suite_guard.cpp`'s `kGoldenDebt`.

**Perf.** No baseline row yet: `test_perf_baseline` requires a Release tree and
the `linux-release-znver4-aocc` rebuild was out of this session's budget. Debug
observation only: ~0.4-0.5 s/token at 160-token prefill (greedy), i.e. ~2 t/s —
the correctness-first path (token-by-token prefill, per-head serial softmax with
the sink column) has had no perf pass. Do not seed the row by hand.

## R7. Ling-3.1-flash (560B-A25B) — ingestion landed, parity goldens are declared debt (2026-10-30)

**Landed.** Family `ling` (`logestix-ling-nvfp4`) on the real 535 GiB
`BailingMoeV3ForCausalLM` checkpoint converted in place
(`/mnt/fast_storage/huggingface/Ling-3.1-flash`, **298 GB** tree: 10 GiB pinned
dense side + 308 GB routed-expert arrows). Geometry: 72 main layers + MTP
layer 72 (stripped and counted, `deferred.mtp = 1555`), 63 KDA + 9 gated MLA
(every 8th layer), 68 MoE layers × 512 routed experts top-8 (group-limited
sigmoid + `expert_bias`, renorm ×2.5) and one ungated shared expert, vocab
157 184. The 34 816 routed experts stream from the neutral mmap `ExpertStore`
through a device slot LRU (≈16-17 GiB, 2053-2182 slots).

**What the checkpoint forced into the engine (measured).**
1. **Per-expert global scales.** Every expert carries its own three NVFP4
   `global_scale`s; the first loader pass filled `MoEExpertRecord.gs*` from a
   single probed expert and the output was incoherent English/Chinese
   ("Theodiech"). Visiting all experts in the geometry pass fixed it — the
   control prompt now reasons and answers "The capital of France is **Paris**".
2. **Official SwiGLU limits.** `share_expert_swiglu_limit_list` is 10 for layers
   64..71 and `expert_swiglu_limit_list` is 10 for layers 65..71. The
   reference HF `modeling_bailing_moe_v3.py` ignores them, but SGLang's
   `BailingMLP` applies `min(silu(gate), L) * clamp(up, -L, L)`; without the
   shared clamp the layer-64 activation jumps ~5x and the answer degrades. Both
   clamps are implemented (`RunLingSwiGLU` + the neutral routed FFN's optional
   `swiglu_limit`), and the HF `expert_swiglu_limit_list` is indexed per layer.
3. **Tokenizer kind 5.** The checkpoint's pre-tokenizer is a single letter run
   with an optional prefix, ONE digit per number word and apostrophe
   contractions (`'(?i:[sdmt]|ll|ve|re)|...`), not the o200k case-run split.
   `LingWordStream` + `tokenizer_kind = 5`; parity **27/27** against the
   checkpoint's own tokenizer (golden captured by
   `tools/ling_tokenizer_capture.py`).
4. **Scratch zeroing.** The per-request arena and router slots are zeroed before
   use; a reused `cudaMalloc` block could otherwise surface stale/NaN bits as a
   router `eid = -1` on the second model load in one process.

**Gates.** SelfCheck: **Paris ×3, bit-exact `final_logits`** (kOff, 10 762 MiB
pinned + 17 GiB expert cache), the kAuto token stream is stable across reloads
(raw logits are the TC tolerance class), and the real-weight native tool
round-trip returns `get_weather {"city":"Paris"}`. Tokenizer parity 27/27, tool
framing 51 assertions, converter/loader/kernels/model stages green. Perf rows
seeded through the harness at the canonical 2048-token protocol:
**2.77 t/s (perf) / 2.41 t/s (r3)**, TTFT ≈ 10.3 s, 2048 generated tokens
(`tests/perf/perf_baselines.json`).

**Golden debt (declared).** FSI/CODE/DET cannot be captured from the source on
this card: the 560B BF16/FP8 model needs 8×H200 and the FLA KDA kernels are
GPU-only. A CPU float32 reference over the source safetensors exists for
single-token probes (`/tmp` scratch during this session: pos-0 top-5 and the
20-token prompt's next-token top-5), but a full 9-prompt capture set is hours
per prompt. The three targets therefore SKIP with rows in `kGoldenDebt`
(`ling:fsi`, `ling:code`, `ling:det`); a future capture must come from the
source (SGLang with the official clamps preferred), never from the engine.

## R8. Beam-1.0-501B-A23B (501B-A23B) — ingestion landed, parity goldens are declared debt (2026-10-31)

**Landed.** Independent family `beam` (`logestix-beam-nvfp4`) on the real
934 GB `BeamForCausalLM` checkpoint converted in place
(`/mnt/fast_storage/huggingface/Beam-1.0-501B-A23B`, **265 GB** tree: 6.49 GiB
pinned dense graph + the device expert cache, and 12 544 routed-expert arrows
under `experts/layer_{3..51}/`). Geometry: 52 layers (3 dense + 49 MoE × 256
routed experts top-8, no shared expert), 5120H / 48 Q · 8 KV · 128 HD / vocab
151 936. The experts stream from the neutral mmap `ExpertStore` with
`first_moe_layer = 3` (`LSX_BEAM_EXPERT_MAX_OPEN`, default 4096) through a
device slot LRU (`LSX_BEAM_EXPERT_CACHE_SLOTS`, default 512 ≈ 11 GiB).

**What the checkpoint forced into the engine (from
`transformers/models/beam/modeling_beam.py`, read before writing any kernel).**
1. **Gated GQA.** `self_attn.gate_proj` [NH·HD,H] plus **one bias per head**
   (`gate_bias` [48]); `attn_out *= sigmoid(gate + bias[head])` BEFORE `o_proj`.
2. **Four sandwich norms + depth-scaled residual.** `idx = 2*(layer+1)` for the
   attention branch and `idx+1` for the MLP branch:
   `out = branch/sqrt(idx) + residual*sqrt((idx-1)/idx)`, each scaled operand
   rounded before the add.
3. **Hybrid attention.** Sliding layers (43) use rotate-half RoPE over the full
   128 head dims (theta 10000, window 512 INCLUDING the current token); the 9
   full layers {5,11,…,47,51} run **NoPE** with an SSMax query scale
   `(1 + (pos+1)/2048)^0.1` applied to q in FP32.
4. **Group-limited sigmoid router.** F32 logits; sigmoid scores;
   `e_score_correction_bias` only for CHOICE; group score = sum of the two best
   corrected scores in each of 8 groups; keep the top 4 groups; top-8 by
   corrected score; weights = the PLAIN sigmoid scores renormalized, ×2.5 after
   aggregation. No shared expert.
5. **Final logit softcap 30** (`logits/30 → tanh → ×30`, F32).
6. **Per-head q/k RMSNorm** (weight form) before RoPE/SSMax; SwiGLU product in
   FP32.
7. **Tokenizer `kind = 6`** (Qwen2/3 single-run pattern: one digit per number
   word, no case/CJK split) with the Harmony control markers as
   `special:false` added tokens (matched literally). Tool wire: modified
   Harmony, `to={name}` + `<channel|>analysis <|constrain|>xml` +
   `<rfl:parameter name="k">` bodies.

**Two defects the real weights exposed (both fixed).**
- `config.json` carries `eos_token_id: null`; the real stop list lives in
  `generation_config.json` as `[151674, 151675]`. Reading only the first id
  (`<|return|>`) made the model **repeat a tool call forever**: `<|call|>`
  (151675) never terminated generation. The converter now carries the whole
  list, and the index/loader/model propagate it.
- The sampler can reach reserved embedding rows (ids ≥ the tokenizer's largest
  added id). A single such id made the *whole* token-stream decode fail, so a
  perfectly valid native tool call parsed as "no call". `ParseBeamToolCalls`
  now drops ids outside the vocabulary before decoding.

**Gates.** SelfCheck: **Paris ×3, bit-exact `logits_hash`** (kOff, 17 445 MiB
resident = 6.49 GiB dense + 11 GiB expert cache); the real-weight native tool
round-trip returns `get_weather {"city":"Paris"}`. Tokenizer parity 30/30,
tool framing, converter/loader/kernels/model stages green. FSI/CODE/DET are
**declared golden debt**: the 934 GB BF16 source does not fit in this host's
750 GB RAM, so no source-oracle capture is possible (rows in `kGoldenDebt`:
`beam:fsi`, `beam:code`, `beam:det`). The instruments used instead are the
host-reference kernel parity suite (norms, RoPE/SSMax, router selection,
window bounds, gate, depth residual, softcap) and SelfCheck/determinism.

**Perf.** Seeded through the harness on the Release/AOCC tree at the canonical
2048-token protocol (`tests/perf/perf_baselines.json`): **1.61 t/s (perf) /
1.60 t/s (r3)**, TTFT ≈27.1 s, 116.97 W / 119.22 W average on the 1531 GB/s
DRAM probe. That is the correctness-first path: token-by-token prefill/decode,
a per-MoE-layer router readback (49 host round-trips per step, no copy-stream
overlap), and a 512-slot expert cache; the release build is ~2.6x the debug
number. The next levers are the ids-streaming/no-handshake decode path used by
qwen/ling and an expert prefetch copy stream — both are engine work, not
correctness gaps.

## R10. Granite-Switch 4.1 30B (granite dense + 2 aLoRA) — ingestion landed 2026-10-31

Composed checkpoint `ibm-granite/granite-switch-4.1-30b` (revision 4.1.1):
Granite 4.1 30B dense decoder + `guardian-core` (rank 16, q/k/v/o) and
`decision` (rank 64, q/k/v/o + MLP) aLoRA adapters + control-token routing +
token exchange. Format tag `logestix-granite-switch`, `kType`
`granite-switch-4.1-30b`, converted in place with `lsxforge --granite_switch`.
Frozen geometry is 64 decoder layers (config says 66 = 64 + 2 switch cache
slots), 4096H / 32Q / 8KV / 128HD / 32768I / vocab 100355, tied embeddings,
multipliers 12 / 0.175 / 16.

Real-weight results (`linux-debug`, RTX 5090, 2026-10-31):

- **SelfCheck PASS** — Paris, 3 stress cycles, bit-exact logits hash
  `fdc632042c30919c`; 16.22 GiB resident.
- **FSI 3/3, CODE 3/3.**
- **DET 2/3 — DET-TEST-02 RED.** The prompt template's Expected Template is
  `CATASTROPHIC`; the **source** CPU-BF16 oracle itself answers `CRITICAL`
  (`tools/det_capture.py` logged
  `source answer differs from the prompt template: 'CRITICAL' != 'CATASTROPHIC'`).
  The converted engine answers `CRITICAL` too, deterministically across 3 runs,
  i.e. this is a source-vs-prompt disagreement, not a conversion divergence and
  not a near-tie a residual plane can move toward the template. The golden is
  left as captured (template-defined by design); the red row is recorded in
  `docs/SELFCHECK_MATRIX.md` and `docs/DET_SELFCHECK_FINDINGS.md`.
- **Adapter parity 3/3** against the CPU-BF16 oracle: base / guardian-core /
  decision first tokens match (`The` / `{"` / `A`), routing is 0 / 1 / 2, and
  the native chat renderer reproduces the source `input_ids` token-for-token for
  all three modes (`test_granite_switch_adapter_parity`).
- **Perf (AOCC release, 2048-token protocol): 69.28 t/s (perf) / 70.07 t/s
  (r3)**, TTFT 49 / 191 ms, 552.95 W average — `tests/perf/perf_baselines.json`.
  The perf row was re-confirmed after the numeric-policy fix below.
- **TC fast-math is disabled for this family (correctness).** The checkpoint's
  `embedding_multiplier=12` / `residual_multiplier=0.175` produces activations
  up to ~6e3 (measured at layer 17 in BF16). The NVFP4 TC path quantizes
  *activations* to E2M1 with E4M3 block scales; its representable ceiling is
  ~2.7e3, so long prompts clamp. Evidence: the 197-token tool prompt on the TC
  path decoded `x = x e^{x}` instead of the source `<tool_call>...`; the same
  prompt on the FP16-activation path reproduces the source tool call exactly
  (input ids token-for-token, `test_granite_switch_model`). `Infer` therefore
  pins `FastMathMode::kOff` for this family (still consulting
  `EffectiveFastMathMode(cfg_)` for the independence contract). kOff is not a
  throughput loss here: 70.07 t/s r3 vs 69.28 Auto on the measured rows.

Open items / declared scope:

- **DET-TEST-02** as above: red because the source does not follow the prompt's
  Expected Template. Closing it needs either a different prompt (shared prompt
  set — not a per-model change) or a source that answers `CATASTROPHIC`.
- **ASR / `<|audio|>` is out of scope.** The 470M Granite Speech encoder is a
  separate artifact and is not present in this checkpoint; the tokenizer's
  `<|audio|>` added token is carried but never decoded.
- **Long context is card-bound.** KV at the advertised 131072 tokens is
  ~34.4 GiB by itself (64 layers x 131072 x 1024 x 2 B x 2), which does not fit
  beside the 16.2 GiB weight set on a 32 GiB card. ~32k tokens is the practical
  arithmetic ceiling; the perf row is the canonical 2048-token protocol.
- The switch codebook is stored (raw F32) for fidelity but not uploaded: the
  reference coded-memory retrieval is exactly the prefix scan over control
  tokens (verified against `granite_switch.hf.switch.multi` on random
  sequences), and token exchange is the checkpoint's LUT.

## R9. Step-5-Preview (600B-A27B) — ingestion landed; DSA >4096 and parity goldens open (2026-10-31)

**What landed.** Independent family `step` (`kType = step-5-preview`, format tag
`logestix-step5-nvfp4`), converted in place from the 606 GB FP8 release
(`stepfun-ai/Step-5-Preview`, 24 text shards + `model-mtp3*` + `model-vit*`)
into a 317 GB tree: 8.8 GiB pinned dense graph + a 307 GB routed-expert store
(30 976 arrows of 10.12 MiB) streamed through the neutral mmap `ExpertStore`,
`first_moe_layer = 3` (`LSX_STEP_EXPERT_MAX_OPEN`, default 4096) and a device
slot LRU (`LSX_STEP_EXPERT_CACHE_SLOTS`, default sized from free VRAM; 1482-1583
slots ≈ 15 GiB on this card). Kernels: FP32 residual stream with zero-centered
`(1+w)` RMSNorm, per-head QK norm, NEOX-paired partial RoPE (192-dim sliding /
64-dim full, theta 1e4 / 1e7), per-head sigmoid output gate, windowed GQA
decode attention, sigmoid + selection-only-bias router (top-8, renorm, ×3.0),
clamped SwiGLU (`min(silu(g),7) * clamp(u,-7,7)`) for the routed experts and the
gated shared expert combined in FP32.

**Geometry traps the source forces (each was a candidate for a silently wrong
token).**
1. **Zero-centered norms.** `Step5RMSNorm` is `x/rms(x) * (1 + w)` with F32
   weights; the checkpoint's weights are centred on zero, so reading them as a
   plain scale multiplies every activation by ~0. First run of the real weights
   emitted newlines; the fix took the Paris gate to green.
2. **`moe_layer_list` is 3..90, and layer 91 is DENSE.** The dense prefix is
   layers 0..2 *and* a trailing dense-MLP full-attention layer 91; the neutral
   `ExpertStore` only knows a contiguous dense prefix, so the loader bounds the
   store at the last MoE layer + 1 (91) and the flat expert index stays
   `layer*experts+e`.
3. **Three expert layers are BF16** (`quantization_config.modules_to_not_convert`
   → layers 88..90, no `weight_scale_inv`). The converter accepts both an FP8
   stack + 128x128 F32 scale grid (GPU dequant) and a BF16 stack (verbatim).
4. **The expert stacks are FUSED** `[352,1536,4096]` / `[352,4096,1536]`; the
   converter splits them per expert into the neutral 9-column `ExpertSchema`.
5. **Partial RoPE span is per layer** and the table is HALVED (pair
   `(d, d+span/2)`, NEOX). The `rope_scaling` llama3 block (`factor 1.0`) is the
   identity and is not applied.
6. **Per-head attention gate runs BEFORE `o_proj`**, from the same normed
   hidden the QKV projections use.
7. **Final RMSNorm narrows the FP32 residual to BF16 first** — part of the
   reference arithmetic, implemented in its own kernel.
8. **MTP layers 92..94 and the 667-tensor vision tower are stripped+counted**
   (`deferred`: `skipped_mtp_tensors = 51`, `skipped_visual_tensors = 667`,
   `num_mtp_layers_stripped = 3`).

**Open item A — the DSA indexer runs only as the dense-equivalent case.** The
Lightning Indexer (`sparse_indexer_{q,k,z,w}`, region 8, proxy 256, top-512
regions) selects at most `topk*region = 4096` tokens per query. At or below that
bound every region is selected, so the dense causal path IS the reference path;
the engine therefore runs dense attention and refuses a request larger than
4096 tokens with an explicit error. The indexer weights ARE converted
(`indexer/layer_N.arrow`, 184 tensors, 834 MB) and the four open items are:
the FP8 region-summary compression, the radix top-k selection with its
zero-group tie-break, the region gather in the attention kernel, and a
region-summary cache in the session. This is the difference between the model's
1M-token claim and the 4096-token supported context (`kMaxContextTokens`).

**Open item B — parity goldens are declared debt.** The reference stack
(`inference/model.py` + `kernel.py`) is the vendor's multi-GPU FP8/Triton path;
the model card's deployment table starts at 4× B300 / 8× H200, so the 606 GB
source cannot be executed on one RTX 5090 and no FSI/CODE/DET golden can be
captured from it (`kGoldenDebt`: `step-5-preview:{fsi,code,det}`). Instruments
used instead: the kernel parity suite against host references (zero-centered
norm + BF16 narrowing, per-head QK norm, partial RoPE, router selection and
coefficients, clamped SwiGLU, windowed attention, FP32 combine) and the
real-weight SelfCheck/determinism/tool round-trip.

**Gates as measured (2026-10-31).** SelfCheck: **Paris x3, bit-exact
`logits_hash = 63b0f8b2fcf56d8a`**, 128 896 logits, kOff, 8.6 GiB pinned +
15.0 GiB expert cache; the kAuto determinism case (8 tokens, two loads) is
byte-identical. Real-weight native tool round-trip: the model emits
`<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>`
and the parser returns `{"city":"Paris"}` (80 generated tokens,
`</think>` then the call). Converter/loader/kernels/model/tool stages green
(`test_step_moe_{convert,loader,kernels,model,tool_framing}`).

**Non-defect found and fixed by the determinism gate.** The attention block
initially took the RoPE span from the frozen passport (192/64) while the
synthetic fixture's head is 32 dims, so the rope kernel wrote past the head into
neighbouring arena buffers and the model was not bit-reproducible. The span is
now clamped to the actual `head_dim` (`min(frozen_span, HD)`); production
geometry is unchanged (192/64 ≤ 192).

**Perf.** Seeded through the harness on the Release/AOCC tree at the canonical
2048-token protocol (`tests/perf/perf_baselines.json`, `step-5-preview`):
**1.82 t/s (perf) / 1.83 t/s (r3)**, TTFT ≈14.5 s, 127.63 W / 129.86 W average
(145.47 W / 147.72 W peak) on the DRAM probe. Both numeric classes land on the
same number because the path is host-store bound (~6.6 GB of expert payload per
token from the 307 GB tree; 8.8 GiB pinned + ~15 GiB expert cache), not compute
bound. The correctness-first shape is token-by-token prefill/decode and a router
readback per MoE layer (88 host round-trips per step), with no prefetch copy
stream. Next levers: the ids-streaming/no-handshake expert path and an expert
prefetch stream (engine work, not correctness).

## R11. Hy-SimulMT-30B-A3B (~30B / ~3B active) — ingestion landed (2026-10-31)

Branch `feat/ingest-hy-simulmt-30b`. `Hy-SimulMT` is a **sibling of
Hy-Memory-30B-A3B**: `HYV3SpeechForCausalLM` with the same `hy_v3` text
backbone (18 673 text tensors, identical names/shapes, the same
`expert_hidden_dim=768`, 48L / 2048H / 32Q-4KV / 128 experts top-8) plus a
24-layer speech encoder. Frozen passport `hy_moe_simulmt_constants.h`
(`hy-simulmt-30b-a3b`, `logestix-hy-simulmt-nvfp4`); converter wrapper
`hy_moe_simulmt_convert.{h,cpp}` over the shared spec-driven pipeline;
`HySimulMTModel` reuses the parent's loader, kernels and generation pipeline.

**Stage-0 facts that were load-bearing.**
- The 403 `audio_tower.*` tensors (660 MB, a separate shard) are ballast:
  stripped AND counted (`skipped_visual_tensors = 403`). Speech input / speech
  output is out of scope for this engine; only the text decoder is served.
- `tokenizer.json` is **byte-identical to Hy-Memory's** (sha256
  `ad96ef9f641b79bb…`): same 120 000 BPE / 818 added tokens, same specials; the
  Hy-Memory tokenizer policy and its 17/17 parity capture apply unchanged.
- The SimulMT chat template's text branch matches Hy-Memory's renderer except
  for ONE literal: its tool-instruction example closes with `</tool_calls>`
  where Hy-Memory has `</tool_call>`. `HyRenderPolicy` now carries that delta;
  the SimulMT framing capture reproduces the checkpoint template id-for-id
  (14/14 cases, `tests/fixtures/hy/chat_framing_simulmt.json`).
- `generation_config.json` declares **two stop ids** (`[120025, 120008]`:
  `<eos>` and `<|hy_EOT|>`). The shared converter now writes
  `eos_token_ids` and the pipeline stops on any of them; Hy-Memory's legacy
  index (single eos) is unchanged.

**Conversion (real, in place).** 60.8 GB BF16 source → **17 GB** tree: 1.5 GiB
pinned dense graph + 15 GiB routed experts (6016 ExpertSchema files), 403
ballast tensors skipped, tokenizer arrows built into `logestix/pinned`.
`LSX_HY_SKIP_EXPERTS=1` re-converts the dense side only (used for the eos fix).

**Serving contract (discovered on the source, then implemented).** The
checkpoint is a **direction-bound zh<->en simultaneous translator**. Without the
official directive the decoder only summarises/transcribes its input; with it
(`SimulInfer/simulinfer/translation.py INSTRUCTIONS`) it translates. The engine
prepends the checkpoint's own directive and auto-selects the direction from the
source script (Han → zh-en, otherwise en-zh), so a plain text request is a
translation request in both directions. This is the family's single-prompt
`BuildPromptInputIds` / conversation rendering contract.

**Gates as measured (RTX 5090, 32 GB).**
- SelfCheck **PASS** — the canonical English sentence is en→zh for this model,
  so the family probe asks the same fact in the trained direction: source
  "法国的首都是巴黎。" translates to "The capital of France is Paris."
  (top1 628, `logits_hash 44b846591fd64cb9`), bit-exact over 3 cycles at kOff;
  kAuto determinism is byte-identical across two loads.
- Real-weight **bidirectional serving test**: "今天天气很好。" → "The weather is
  very nice today." (zh→en); "What is the capital of France?" → "什么是法国的
  首都？" (en→zh).
- FSI / CODE / DET are **declared golden debt**
  (`kGoldenDebt: hy-simulmt-30b-a3b:{fsi,code,det}`). The source oracle runs
  (`tools/hy_v3_oracle.py` with the `audio_tower.*` skip; the text backbone
  loads in `HYV3ForCausalLM`), and the measured behaviour is that the model
  summarises/translates these QA bodies instead of answering: FSI-TEST-02 and
  DET-TEST-01 emit EOT immediately, CODE-TEST-01/02 emit a prose summary,
  FSI-TEST-01 loses the required exact expression (spaces + translated label).
  No source-faithful golden can be extracted, so the instrument is SelfCheck +
  the bidirectional test, not a rewritten golden.
- Tokenizer parity is inherited from the byte-identical file; chat framing
  14/14 against the SimulMT template.
- Perf rows seeded through the harness at the canonical 2048-token protocol:
  **perf 87.57 t/s / r3 74.85 t/s**, TTFT 510.1 / 616.1 ms, 16.5 GiB resident
  (all routed experts resident; the same backbone and the same roofline as
  Hy-Memory).

**Open work, explicitly.**
- P1. The speech encoder (mel → 24-layer encoder → adapter → audio
  placeholders, and the streaming S2ST loop) is not implemented; audio input
  and speech output are not served. This is the difference between the
  checkpoint's product behaviour and the text-translator surface shipped here.
- P2. The shared perf debt of the `hy` backbone applies unchanged (weight
  streaming GEMV at ~87 t/s against a ~750 t/s roofline; see §R5 P1).
- P3. Tool calling is framing-complete (the native codec is verified) but
  semantically out of distribution for a translator; no real-weight tool
  round-trip is claimed.

## R12. GigaChat3.5-432B-A28B-Reasoning — real-weight lifecycle complete (2026-10-31)

Branch `feat/model-GigaChat3.5-432B-A28B-Reasoning`. The code landed in the
2026-09/10 sessions; this session executed the **real-weight half** of the
lifecycle at the current vault path (`/mnt/fast_storage/huggingface`) and closed
two stage-4 defects. Report:
`docs/model-reports/2026-10-31-GigaChat3.5-432B-A28B-Reasoning.md`.

**Stage-4 defects found and fixed (same commit as the tests).**
- `ConvertGigaChatModel` never built the tokenizer arrows (the only family that
  did not) — it left `"tokenizer": "fixture"` in the index and pushed stage 4
  to a separate operator step. It now calls `BuildAndVerifyTokenizer` and writes
  `"tokenizer": "arrow-bpe"`; the synthetic fixture writes a real
  `tokenizer.json` + `tokenizer_config.json` and the converter test pins
  `tokenizer_kind == 3`, `bos/eos/pad`, and a byte-exact encode/decode.
- The `--tokenizer` policy defaulted to the legacy kind-0 split, which
  mis-splits mixed-case identifiers (`getUserName`: `get|UserName` vs the
  checkpoint's `get|User|Name`). GigaChat is the Llama-3/o200k 7-alternative
  case-run BPE, so both the converter and `lsxforge --tokenizer --gigachat`
  now select `pretokenizer = 1` (kind 3) and a 30-case golden captured from the
  checkpoint's own `tokenizer.json` passes 30/30
  (`tests/fixtures/gigachat/tokenizer_parity.jsonl`,
  `tools/gigachat_tokenizer_capture.py`, `test_gigachat_moe_tokenizer_parity`,
  claimed by the gigachat row's `load` stage).

**Conversion (real, in place).** 416 GB FP8-block-128 source → **241 GB**
tree: 9.7 GB pinned dense + 219 GB routed experts (9472 ExpertSchema files,
14 × 16 GiB pinned chunks in ~7.2 s warm) + 13 GB MTP stored-not-applied
(144 tensors). `model.index.json` reports 40 layers, 9472 experts, 1009 pinned
tensors.

**Gates as measured (RTX 5090, 32 GB, `linux-release-znver4-aocc`).**
- SelfCheck **PASS**: "The capital of France is" → **Paris** (top-1 19904),
  Paris ×3 bit-exact at `logits_hash 2743ee1c26f859d5`, TTFT budget green after
  the cold-store page-cache eviction.
- Native tool round-trip (real weights): `get_weather {"city":"Paris"}` through
  the chatml-v5 GCML codec.
- FSI **3/3**, CODE **3/3**, DET **3/3** (`deterministic=true`, 3 runs/prompt)
  against **template-derived** goldens. They are honest template goldens
  (`source` = prompt-template + the unrunnable reason, `raw` empty in code/det)
  captured with the checkpoint's `chat_template.jinja` and a **1536-token**
  budget: at 512 the FSI-02 reasoning trace is still open when the budget runs
  out, and extraction then latches onto a prompt-echo line
  (`Balance_Alpha=100000000`) — a truncation artifact, not a wrong answer. The
  two capture tools gained an `add_special_tokens` override because the
  GigaChat template spells `<s>` literally (otherwise the ids carry a double
  BOS).
- Tokenizer parity 30/30 (kind 3, bos 1, eos 2, pad 2) against the tree's own
  arrows.
- Perf rows re-seeded through the harness at the canonical 2048-token protocol:
  **perf 11.32 t/s / r3 11.10 t/s**, TTFT 3905 / 3975 ms, 189 / 192 W,
  26.2 GiB resident. The previous `r3 3.47` row (2026-10-06) does not reproduce
  on this tree — the strict class now lands ~2% under `perf`, the expected shape
  for a store-bound model. Both rows were re-seeded through the harness twice
  (417 s each) with identical values, so the new number is measured, not
  inferred; the old row is history, not an explanation.

**Open work, explicitly.**
- L1. Decode is dominated by expert H2D (~3.7 GB/token at the PCIe Gen5
  ceiling; cache hit rate ~49-51% at the auto slot count). The attribution and
  the measured ceiling are in `docs/GIGACHAT_SPEED_AUDIT.md`; nothing is
  filed as a defect because the model fits only as a streamed expert set.
- L2. FSI/CODE/DET goldens are template-derived, not source parity. The source
  is the vendor's multi-GPU FP8 stack and the BF16 body is 864 GB > 750 GB host
  RAM, so no oracle can run here. A future host with enough RAM could replace
  them with source captures; the goldens are labelled so the provenance is
  never mistaken for source parity.
- L3. Session ceiling is 4096 tokens (`max_seq_`); the 262k context in
  `config.json` needs ~34 GiB of MLA KV alone.
- L4. MTP layers 40-42 are stored and counted, never applied.

## R13. Nemotron-3.5-Lightning-30B-A3B-NVFP4 — lifecycle verified, **NoPE defect fixed** (2026-11-01)

Branch `feat/nemotron-nvfp4-lifecycle`. The family (BF16 passport + NVFP4
sibling) was already ingested; this session **ran the lifecycle on real weights
for the first time** and found a real semantic defect the green suite could not
see.

**The defect: full-attention layers were wrong — they are NoPE.** The
checkpoint's own reference class (`transformers` 5.15.1
`NemotronHAttention`) has **no rotary module**: it never builds cos/sin, has no
`rotary_emb`, and ignores the `position_ids` it is handed. Position in this
hybrid comes from the 23 Mamba-2 layers; `rope_theta` /
`partial_rotary_factor` in `config.json` are vestigial. The engine applied
full-dimension `rotate_half` RoPE to q and k anyway, so every attention layer
injected a position-dependent rotation the weights were never trained with.

Why the old suite hid it: RoPE is the identity at `pos = 0` (cos 1, sin 0), and
the Paris SelfCheck is a ~7-token prompt — the model is robust enough to answer
it with a mild perturbation. The corruption is context-length dependent:

| pos | per-layer relL2 vs the BF16 reference, engine-with-RoPE |
|---|---|
| 0 | 0.09 (pure 4-bit) |
| 3 | 0.28 |
| 43 | 0.60 |
| 284 | **1.20** (cos 0.38 at L51), jumps of 2x at every attention layer |

Symptoms: any tool-framed prompt (~300 tokens) produced HTML-tag soup and no
tool call; the real-weight tool round-trip failed. Fix: delete the two
`RunNemotronRope` calls and the now-dead kernel + its unit test; the NoPE fact
is recorded in both constants headers and next to the attention launch.
After the fix, with the same converted tree:

- per-layer relL2 vs the BF16 tree reference at pos 284: **0.003 (L0), 0.011
  (L5), 0.23 (L51)**, cos ≥ 0.973 — pure quantization accumulation;
- greedy generation matches the source reference **24/24 tokens** on the tool
  prompts (was 0/24), 24/24 on the plain chat prompt, 13/24 on one free-form
  prompt before a near-tie;
- real-weight tool round-trip: `get_weather {"city":"Paris"}`.

**The instrument that found it (new, keep it).**
`tools/nemotron_nvfp4_oracle.py` undoes the ModelOpt encodings (W4A16_NVFP4
`weight`/`weight_scale`/`weight_scale_2`, per-tensor FP8, BF16 passthrough) with
torch and assembles a CPU BF16 `NemotronHForCausalLM` — the source oracle this
family never had. `--from-tree`-equivalent evidence was produced by decoding the
converted `logestix/pinned/*.arrow` back to BF16 (same script's codec): that
tree reference reproduces the *source* reference token-for-token, which is how
"the weights are fine, the runtime is wrong" was separated from "the conversion
is lossy". 8 s per 16-token CPU generation on 160 cores, so it is a usable gate
instrument, not an experiment.

**Gates as measured (RTX 5090, 32 GB, `linux-release-znver4-aocc`).**
- SelfCheck **PASS**: Paris ×3 bit-exact, `logits_hash fe3b05c75a2033ba`
  (changes from `e5543669c00386f6` — the NoPE fix legitimately moves the logits).
- Tokenizer parity **22/22** (`test_nemotron_moe_tokenizer_parity`, new;
  `tools/nemotron_tokenizer_capture.py` + `tests/fixtures/nemotron/`), kind 9.
- Tool calling (real weights): green, above.
- Perf rows re-seeded through the harness at the canonical 2048-token protocol:
  **perf 127.55 t/s / r3 120.39 t/s**, TTFT 84 / 89 ms, 242 / 241 W. RoPE was
  negligible, so this is a re-seed, not a speed-up.
- FSI/CODE/DET remain declared debt, now with measurements instead of a
  hypothesis: the **source** answers FSI 3/3 per the Expected Template
  (`PV=CF*((1-exp(-r*n))/(1-exp(-r)))`, `Balance_Alpha=…`,
  `Drawdown>Threshold`), so a golden is capturable — but the converted tree
  does not reproduce them (FSI-01 appends `*exp(r)`, FSI-02/03 emit fenced /
  repeatedly-closed bodies), and the source echoes the prompt body on
  CODE-TEST-01 and DET-TEST-01/02, so those two sets have no source-faithful
  answer to record.

**Open, explicitly.**
- N1. FSI parity is red on the converted tree. The measured mechanism is the
  4-bit body (attention projections sit at 9.5% relL2 vs source; the vendor
  kept them BF16 and the Mamba projections FP8). The documented lever is the
  residual plane (attn + Mamba projections at ~8 effective bits, routed experts
  left 4-bit) — the gemma-4-26b recipe — at roughly +1.1 GiB resident and a
  doubled attn+Mamba read. Not attempted here: it changes the tree every
  existing baseline was seeded on and the family is a perf-frontier row.
- N2. The mamba expert store probes `layer_0..` / every non-MoE layer at load
  (3712 failed mmap opens, debug-logged) because `ExpertStore::Load` does not
  consult `moe_index_of_layer`. Harmless (entries stay empty, the compact
  device id map is correct) and ~30 ms per load; noted, not fixed.

## R19. Nemotron-3-Ultra-550B-A55B — lifecycle complete, two Ultra-only defects found (2026-11-11)

Branch `feat/nemotron-ultra-550b-ingest`. A **separate lineage** from
`nemotron-3.5-lightning-30b-a3b` (operator decision): own frozen passport, converter,
index/loader, expert store, kernels, pipeline and tool codec under
`nemotron_moe_ultra_*` / `nemotron_ultra_*`. No 30B file was edited. Report:
`docs/model-reports/2026-11-11-NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16.md`.

**Geometry reality vs the first read of `config.json`.** Three things a reader gets
wrong, all caught by the real-weight probe and not by the green synthetic suite:

- `num_hidden_layers` is **absent**; depth is `len(layers_block_type)` (=108),
  exactly as `NemotronHConfig` derives it.
- `n_group` (routing) is **1**; the 8 in the same file is `n_groups` (SSM state
  groups). The passport initially froze `kRouterGroups = 8` from the wrong field.
  The grouped-router implementation stays (it is tested at n_group 1, 4 and 8) but
  this checkpoint's group stage is the identity.
- `generation_config.json` holds the real stop set `[2, 11]`; `config.json` holds
  the scalar `2`.

**The two defects the synthetic suite could not see (both fixed, both Ultra-only).**

- **U1 — top-22 exceeds the batched expert GEMV's contract.** 
  `RunLsxNvfp4GemmSeq1GemmVExpertBatch` accepts at most **8** experts per launch;
  the 30B sibling routes to top-6, so copying its call returns
  `cudaErrorInvalidValue` on the first MoE layer. Launches are chunked by 8 now.
- **U2 — the latent bridge aliases its own input.** With `moe_latent_size = 2048`
  the routed up-projection writes `topk*I = 112 640` halfs into the buffer the GEMV
  is still reading the 2048-wide latent from, so chunk 0 clobbers its input. The
  30B cannot hit it (no latent bridge: its expert input is `normed`). Symptom:
  coherent-looking garbage (`'إن order'`), residual L2 growing 1.4 → 14 129 over
  108 layers, and the head faithfully reproducing that wrong state from the
  checkpoint's own BF16 weights. Fixed with a dedicated `s.latent16` buffer.

**Measured (RTX 5090, 32 GB, `linux-release-znver4-aocc`).**

- SelfCheck **PASS**: Paris ×3 bit-exact, `logits_hash bd82916c6550a2df`, top-1 6993.
- Tool round-trip: `get_weather {"city":"Paris"}`.
- Slice oracle (`tools/nemotron_ultra_oracle.py`, real layers 0–7 + real
  embed/norm_f/head): engine vs reference relL2 **3e-4 (L0), 2.4e-3 (L1, latent
  MoE), 7e-3 (L4), 3e-2 (L5)**. It must be a CONTIGUOUS prefix — with layers
  0,1,7 the attention block's input differs by five layers and the comparison is
  meaningless.
- CODE 3/3, DET 3/3 (byte-identical over three runs), tokenizer 22/22.
- Perf: **1.77 t/s** (r3 1.75), TTFT 13.4 s, 139 W at the canonical 2 048-token
  protocol. 661 of 55 296 expert slots; 12.5 GB of expert traffic per token.
- Conversion: 292 GB, 24 576 experts, 1 040 MTP tensors cut, **peak host residency
  512 expert records** (per-layer streaming; the whole pool would be 290 GB).

**Open, explicitly.**

- **O1 — FSI-01 is a real generation error, excluded in writing.** The model
  answers `PV=CF*((1-exp(-r*n))/(1-exp(-r)))*exp(r)`: it appends `exp(r)` to the
  expression that is *already* the annuity-due form, landing 8.7 % high at
  r = 0.0835. The 30B sibling appends the same factor (§R13), so it is a
  family-level habit, not a conversion artefact. The committed FSI golden keeps
  FSI-02/03 (both pass) and records FSI-01's exact output under
  `excluded_cases`; the matrix row therefore reads 2/3, not 3/3.
- **O2 — no full-model source reference.** The 1.1 TB BF16 tree cannot be
  assembled in 750 GB, so the parity goldens are template-derived (recorded in
  every fixture's `source`) and source fidelity rests on the slice oracle's 8
  layers. Extending the slice is the documented lever (each 8-layer step with 3
  MoE layers costs ~65 GB of host RAM).
- **O3 — conversion tooling.** `lsxforge --nemotron_ultra --layers a:b` converts a
  window and marks the tree partial; a partial tree is refused by the loader even
  under `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1`. Resume works by re-running a later
  window (the index lists the union of layers on disk).
- **O4 — the Ultra converter validates descriptors but does not call
  `integrity::VerifyAllTensorReads`**, which reads all 51 023 tensors into heap
  vectors — a second full 1.1 TB pass. If another family ever needs the same, the
  helper wants a scoped form rather than being skipped ad hoc.

## R20. Nemotron-3.5-Lightning-30B-A3B (BF16) — first lifecycle run, one loader defect, parity red with a measured mechanism (2026-11-13)

Branch `feat/nemotron-lightning-bf16-lifecycle`. The BF16 sibling of the ingested
NVFP4 member had a manifest row, a registered `kType`, a leaf class, a converter
and eleven ctest targets — and no evidence: no converted tree, every gate a SKIP,
no goldens, `tool_roundtrip: false`, perf rows from 2026-09-16. This session
converted it and ran every stage on real weights. Report:
`docs/model-reports/2026-11-13-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.md`.

**Sibling, not a new lineage.** Identical frozen geometry and the same
`logestix-nemotron-nvfp4` output layout, so `NemotronModelBase`, the loader, the
kernels, the expert cache and the tool codec carry it unchanged; a second
passport for the same widths is what `test_model_independence_guard` forbids.

**The defect: the BF16 loader accepted the NVFP4 sibling's tree.** Both
distributions write the same format tag, so only the index's `model` field
separates them — and `NemotronNvfp4Model` has enforced that since ingestion while
`NemotronHModel` did not. A mixed-up tree would have loaded and then verified
against the other model's goldens. Fixed with `frozen::kModelName` plus a check
in `LoadAndValidateIndex`; the synthetic fixture's index writer now emits the
converter's string (it wrote `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`, which
no converter emits — that is why nothing caught the gap).
Test: "nemotron Load rejects a tree built for the other distribution", verified
RED with the check disabled.

**Conversion, measured.** 44 s from page cache → 18 GB (2.2 GB pinned / 107
Arrow files, 16 GB experts / 23 dirs × 128), 2944 routed experts, 355 pinned
tensors, 270 MTP tensors cut and counted. Per-tensor relL2 against the BF16
source (decoding the written record, `tools/nemotron_bf16_quant_probe.py`):
attention 0.0947, Mamba 0.0950, shared expert 0.0943, routed experts 0.0946;
`lm_head` and `embeddings` are BF16 passthrough (0.000000) — this distribution's
head is NOT quantized, unlike the vendor NVFP4 tree. An ideal E2M1 grid with an
exact float block scale measures 0.0937–0.0939 on the same tensors, so the codec
sits on the 4-bit grid bound and E4M3 block-scale rounding costs ~1 % relative.

**Gates as measured (RTX 5090, 32 GB, `linux-release-znver4-aocc`).**
- SelfCheck **PASS**: Paris ×3 bit-exact, `logits_hash 0dba89277c5d1250`, top-1 6993.
- Tokenizer parity **22/22** on the real vault (290 assertions).
- Tool round-trip **PASS** on real weights: `get_weather {"city":"Paris"}`;
  the manifest's `tool_roundtrip` is now `true`.
- Perf re-seeded through the harness at 2048 tokens: **127.00 t/s / r3 120.32**,
  TTFT 85 / 89 ms, 250 W mean. The NVFP4 sibling reads 127.55 / 120.39 — the
  expected agreement for the same layout.
- **FSI 2/3, CODE 2/3, DET 1/3** (DET determinism 3/3).

**Parity provenance is better than the sibling's, and that is what makes the red
informative.** This checkpoint loads in `transformers` 5.15.1 directly, so all
three goldens in `tests/fixtures/{fsi,code,det}/nemotron-3.5-lightning-30b-a3b.json`
record the vault path as `source` — the source was **executed** and answers all
nine cases exactly as the prompt templates specify (after 1358 / 734 / 363
thinking tokens; `transformers` 5.15.1 / torch 2.13.0+cu130, CPU bf16). No oracle
script, no template derivation. Alignment is mechanical: the goldens are captured
with `enable_thinking=True`, and their `input_ids` decode **byte-exactly** to the
served frame pinned by `tests/fixtures/nemotron/fsi01_frame.txt` (first id 10,
no bos — `add_bos_token = false`). `tests/test_nemotron_moe_model.cpp` "FSI-capture
frame equals the served frame" fails if the renderer and the template drift (a
thinking-off mutation was verified to fail it).

**The measured mechanism (why 02/03 are red).** Across the three parity runs every
case ran to its **token budget** — the decoded raw ends mid-repetition at the
truncation point and the wall time matches a full-budget decode (FSI-02 at 2048
took 38 s) — while the source stops at 363/734/1358 tokens. Where it reaches the answer it reaches it *correctly* and then
repeats it: FSI-02 logs 15 copies of
`</think>Balance_Alpha=(((100000000*Rate_A)-Slip_A)*Rate_B-Slip_B)*Rate_C-Slip_C`,
CODE-02 logs `</think>m=l+((h-l)/2)` repeatedly. The extractor's last-line
preference therefore lands on the budget-truncated copy (`...Rate_B-S`, `PARAD`)
or rejects the repetition (`''`). **The verdict is budget-sensitive** — at 2048
FSI-03 passes, at 3072 it fails and FSI-02 extracts `Balance_Alpha=(((` — which is
why the row is debt and not a pass. Two other generation paths in the same build
(the tool round-trip and the Paris probe) terminate normally, so "the decode loop
never advances" is not supported; whether the non-termination is only 4-bit body
noise on the `</think>` vs `<|im_end|>` argmax is **not established**.

**The instrument that cannot see what it is asked (keep this).** A grep for
`<|im_end|>` in a decoded raw can never prove that a model failed to emit eos:
`arrow_tokenizer_decode` drops special tokens, so a clean stop is invisible in
the text as well. The observable that does work is the repetition running to the
budget boundary, plus the wall time. The one prompt where the answer alone was
ambiguous (`FSI-01`) is also the one that passes — do not generalise from it.

**Open, explicitly.**
- N1. The residual-plane lever (`gemma-4-26b` recipe: attn + Mamba projections at
  ~8 effective bits, routed experts left 4-bit, ≈ +1.1 GiB resident, doubled
  attn+Mamba read) is the documented next step. Not taken: it changes the tree,
  re-seeds every perf row and makes the two distributions incomparable — an
  operator decision on 2026-11-13 to measure before changing the tree. Trigger:
  any claim that this family's parity can be green on this card without it.
- N2. Session state after a case that exhausts its budget: in one 3072-budget run
  the case following a cap-exhausted case logged `raw=''` while generating ~3000
  tokens (a decode of a non-empty tail returned an empty string); the same case
  in isolation is non-empty. `Infer` resets the session per call, so this is not
  the obvious cache-reuse bug — reproduce it before believing it.
- N3. The capture budget must exceed the SOURCE's own need: at the tools' 384-token
  default all three FSI traces were truncated inside the reasoning and the
  extraction recorded the prompt's own `Expected Template:` line as the answer
  (a tautology). The recorded budget is 2048, measured from the source's
  363–1358. Any future re-capture must check `</think>` is present in the raw.

## Model manifest migration (2026-11-06) — what it did not cover

The registry moved from six hand-maintained projections to `models/*.json`.
Closed: the guard registries, the ctest registration, the factory list, the
parity-gate and selfcheck test files, the `AGENTS.md` model narrative
(1109 -> 164 lines) and the `README.md` inventory.

Still open, by construction:

- **Build wiring is still per model.** `add_executable`, the CUDA OBJECT
  libraries and per-target includes/defines stay in `tests/CMakeLists.txt`: only
  10 of 421 targets follow a convention, so a generated build graph would be a
  worse copy of the same information. Revisit only if a re-architecture makes
  the wiring conventional.
- **`format_tag` is declared for 8 of 35 models.** The rest have no
  machine-readable source (no `kFormatTag` constant; the tag lives inside a
  converter's index writer). The catalogue prints "not declared" rather than a
  guess. Closing it means exposing each converter's tag as a constant.
- **Per-model goldens and perf rows are still captured by hand.** The manifest
  names them and records debt; it cannot produce them.
- **Approach C (family scaffolder) is not built.** Decided after this work, on
  the measure "lines a fresh family needs that are not converter/kernel logic".

### Stage-9 reports missing for 20 models

`docs/MODELS.md` links every model's report where one exists (15 of 35) and prints
**not written** where it does not — the line is stated rather than omitted, so a
missing report cannot read as "not applicable". The 20 without one:

`deepseek-v4-flash`, `deepseek-v4-pro-0813`, `deepseek-v4.1-flash`, `flint`, `gemma-4-26b-a4b-it`, `gemma-4-31b-it`, `glm-5.2`, `glm-5.3-flash`, `hy-memory-30b-a3b`, `kimi-k2.7-code`, `ling`, `mimo-v2.6-flash-rl`, `minimax-m3`, `nemotron-3.5-lightning-30b-a3b`, `ornith-1.5-35b-a3b`, `phi-4`, `qwen3.5-35b-a3b`, `qwen3.5-9b`, `qwen3.6-35b-a3b`, `xing4.0-29b-a4b`

The gap is not cosmetic: for these models the per-model *reasoning* (why a family
pins `FastMathMode::kOff`, why its residual plane exists, what the measured
bottleneck is) lives only in `AGENTS.md`'s git history and in `docs/open-tasks/`
sessions, not in a document a reader can be pointed at. `models/*.json:report`
now carries a validation rule — a declared path must exist — so writing one is a
one-line manifest change.

### The ctest oracle is a dated snapshot, not a live target list

`tools/dump_ctest_manifest.py` and `tests/fixtures/ctest/2026-11-06-baseline.json`
(511 targets captured from the pre-migration `linux-debug` tree) are kept as the
migration's evidence, **not** as a gate and not as documentation of the current
target set.

- **What they prove.** `python3 tools/gen_model_manifest.py --root . --diff-cmake
  tests/fixtures/ctest/2026-11-06-baseline.json` compares the emitted model
  registration against what CMake produced *before* the manifests existed: 421 of
  the 511 targets are model targets (the other 90 are guard/kernel/infrastructure
  tests). Today it prints `compared 421 targets, 0 difference(s)`, which is the
  only record that the transcription into `models/*.json` was exact rather than
  approximately exact.
- **CI never runs it, and should not.** The snapshot describes the tree on
  2026-11-06. The first model added after that date makes the check report
  `manifest target absent from the baseline` and exit 1 — verified by deleting
  one target from a copy of the baseline, which yields
  `compared 420 targets, 1 difference(s)` and exit 1. That is expected growth,
  not a regression, so the check cannot live in the suite.
- **Do not re-capture it in place.** Re-dumping today yields 511 targets again, but
  not the same ones: `test_parity_selfcheck_naming_guard` is gone and
  `test_manifest_generator_selftest` (this migration's own self-test) is new.
  The model-target projection happens to be identical (421, 0 differences), so
  overwriting the file would look harmless and would silently destroy the only
  record of the tree as it was *before* the manifests. A snapshot is meaningful
  only as the "before" of a *future* migration, captured before its first edit.
- **Live authority is elsewhere.** If a current registration check is ever
  needed, it is the generated `model_targets.cmake` plus `test_manifest_sync`,
  not this JSON. The fixture answers "did the migration lose a target or change
  a label/timeout/env?", a question that has one date.
