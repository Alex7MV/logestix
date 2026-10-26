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

## J. gemma-4-26b-a4b-it FSI/CODE — RESOLVED 2026-10-24 (archive)

All four parity gates are green on the MoE member (SelfCheck, FSI 3/3, CODE
3/3, DET 3/3) after two fixes: the NVFP4 logits kernel was dropping the tied
table's per-tensor global scale (logits 7.01x hot, fixed in `76a13afc`, pinned
by a `gs != 1` case in `test_gemma_moe_moe`), and the body needed a second
NVFP4 plane on the attention projections + dense MLP (routed experts stay
4-bit). Measured root cause, the recipe ablation, the implementation and the
re-seeded throughput (128.25 t/s perf / 128.34 r3, was ~157) --
[`open-tasks/2026-10-24-gemma4-moe-residual-plane-closed.md`](open-tasks/2026-10-24-gemma4-moe-residual-plane-closed.md).

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
- **Shared TC defect: RESOLVED 2026-10-10** - `TcGemmKernel` wrote only
  rows `[0,16)` of its 64-row tile (`wm` was computed and never used); all
  four row computations fixed, TC shape sweep extended to seq 17/18/33/64.
  gemma keeps its scalar prefill **by choice** now, because running it
  through TC moves the first token and breaks the tool round-trip. Full
  trail: [`open-tasks/2026-10-10-neutral-moe-and-tc-mtile-closed.md`](open-tasks/2026-10-10-neutral-moe-and-tc-mtile-closed.md).
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
- **K3 — perf rows seeded 2026-10-26 (resolved).** The 2026-09-14 `perf` row
  (5.1 t/s) carried no evidence (`ttft_ms 0`, `power_samples 0`) and had never
  been re-measured after the conversion. It was removed and re-recorded through
  the harness (Release/AOCC tree, 2048-token protocol, pinned 542 GB store):
  **`perf` 4.86 t/s** (ttft 1043.8 ms, avg 296.5 W, peak 396.4 W, 20979
  samples) and **`r3` 5.04 t/s** (ttft 1069.9 ms, avg 304.0 W, peak 395.7 W).
  The -5% against the stale reference is the numeric fixes changing the routed
  expert sets (different per-token cache working sets), and the old row had no
  protocol evidence to compare against. Re-seed is deliberate; the numbers are
  the harness's, not hand-edited.
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

Current state: **25 declared debt entries** across the nine families that have none
(or only some) of the three goldens. That number is the honest coverage figure, and
until now it was only obtainable by diffing two directories by hand.

The design choice worth recording: this is **declared debt, not a hard requirement**.
A hard requirement would paint nine families red on the next fast run and demand
capturing goldens for every one — including models with no local safetensors, which
is hours of work plus vault access we do not have here. Declared debt gets the
property that actually matters (a new family cannot be added quietly without
goldens, and the remaining gap is enumerable from test output) without pretending the
existing debt is smaller than it is. A missing golden stays a SKIP; what changes is
that it can no longer be invisible.

`granite-4.2-30b:code` is the one entry here that is a single stage rather than a
whole family, which is worth noticing: a family can look fully wired across all four
parity stages and still have a hole in one of them.

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
not in the vault. The debt rows stay. The lesson matches §J: a golden that is
not the source's own generation is not a parity oracle, and a golden with the
wrong tokenizer cannot even be a framing test.

## Q. DeepSeek-V4.1-Flash FSI/CODE goldens committed model-defined; both RED (2026-10-07)

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

- **FSI 1/3** — FSI-01: the thinking CoT does not finish within the 512-token
  budget, so the first-anchor-line extractor latches onto the truncated prompt
  echo `PV=CF*(1-e`; FSI-02 yields the initial-state echo
  `Balance_Alpha=100000000`; FSI-03 passes.
- **CODE 2/3** — CODE-01: the model's CoT declares the balanced template
  `{"root":{"nodes":{"leaf":true}}}` "invalid", miscounts the braces, and emits
  an object missing its final `}` (extractor empty); CODE-02/03 pass.

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

## T. Mistral-Small-4-119B-2603-NVFP4 ingestion — later stages (2026-10-11)

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
  2. The neutral `RunMoERouterTopK` seq==1 dots path produced a NaN logit on
     finite real inputs (host f64 of the same data is finite, max 7.97), which
     fed a NaN through the MoE. The family now routes on the **host**
     (`gemv::RunLxGemmVec` BF16 + softmax/top-K in `mistral_moe_inference.cu`).
     This may be a real defect in the neutral seq==1 router path — worth a
     separate look (it is not exercised here by any other family at E=128).
- **Real-weight decode: fixed (2026-10-12).** `test_mistral_moe_selfcheck`
  passes (Paris + tool round-trip + 3 bit-exact stress cycles); `--bench`
  completes 2048 tokens at **14.68 t/s** (Release, `r3` and `perf` both;
  the `linux-debug` preset is 45 t/s but a Debug binary is not a valid perf
  record — `test_perf_baseline` SKIPs outside `NDEBUG`); SelfCheck passes in **both** the auto
  and `--bit-exact` (`FastMathMode::kOff`) modes. The layer-4 "outlier" that
  this entry previously reported was a **measurement artefact** — the expert is
  a genuine massive-activation expert (host reference at layer 4 confirms
  routed max 411.9 vs the 403.8 the GPU produced). Five real defects, each
  isolated against a host NVFP4 reference built from the source safetensors:
  1. **Shared expert through the BF16 GEMV.** `attn::RunAttnMLP` reinterpreted
     the shared expert's E2M1 nibbles as BF16: shared rms 0.0080 vs the host
     reference 0.3495 (**44x** under-contribution). Fixed by running the
     neutral NVFP4 seq==1 GEMV (`MistralSharedExpert`, the Xing pattern —
     `RunMoESharedCombine` is gated and does not apply, this model's shared
     expert is ungated). Regression test
     `test_mistral_moe_kernels` "matches the host NVFP4 reference" (mutation
     checked: reverting the scale makes it fail).
  2. **Autodetect name mismatch.** The converter wrote
     `"model": "Mistral-Small-4-119B-2603"`, which `detect_model_type`
     normalizes to `mistral-small-4-119b-2603` — not the registered kType, so
     `lsxbenchmark`/`lsxfabric` could not resolve the model by path at all.
  3. **Expert-cache VRAM over-commit.** A hand-rolled budget with a 2 GiB
     reserve resolved 1838 slots (~25 GiB) and the 2048-token bench OOMed on
     the KV arena. Now the shared `lifecycle::ExpertCacheSlots` (8 GiB
     calibrated reserve + the `expert_slots_pct` knob) → 1382 slots.
  4. **FP16 SwiGLU overflow (the actual NaN).** This model carries massive
     activations: layer 32 expert 67 reaches `|silu(g)*u| = 2.5e5` against the
     FP16 ceiling of 65504 — exactly 4 values overflow, and the inf propagated
     through the down projection into the residual. The scalar (non-TC) MoE
     path stores that product in FP16. `RunMoERoutedFFN` and
     `MistralSharedExpert` now evaluate the product in FP32 and store it under
     a power-of-two prescale, folded back in the FP32 accumulator
     (`SwigluF32Kernel`/`SwigluAbsMaxKernel`/`SwigluAlphaKernel`/
     `SwigluStoreScaledKernel` in `lsx_kernels_act.cuh`, plus
     `moe_coeff_add_scaled_kernel` / `acc_add_half_scaled_kernel`). It is a
     **provable no-op for every existing model** (alpha == 1 whenever the
     product fits FP16, and the value expression is the same F32 silu). The TC
     path was already immune — it quantizes from FP32 — which is why only
     `--bit-exact` diverged. Regression test: "survives an FP16-overflowing
     SwiGLU" (mutation checked).
  5. **FP16 residual + session leak.** The residual itself reaches ~2.4e5 at
     layer 32 (host reference), past FP16 range, so `d_h` is now FP32 (the
     norms and the `d_post` hand-off are unchanged). And `~MistralSession` now
     frees — without it every `Infer` leaked the dense graph + expert cache and
     the *second* request OOMed. (`FreeMistralSession` resets the pointers in
     place; `s = MistralSession{}` recursed infinitely once the destructor
     owned teardown.)
- **CLOSED — the "neutral router NaN" was a mis-attribution, not a defect
  (2026-10-12).** This entry previously recorded
  `RunMoERouterTopK`'s seq==1 path as returning a NaN logit on finite real
  inputs. It does not. The NaN was the shared-expert FP16 overflow propagating
  into the *next* layer's `d_post`, which is the router's input — i.e. a
  downstream symptom of defect 1, measured before that defect was found. With
  the shared expert fixed, the neutral device router runs this family on real
  weights at its real geometry (E=128, H=4096, top-4) in **both** numeric
  modes; the host-side router workaround has been removed.
  The investigation did surface a **real coverage gap**: the only router
  reference test covered `seq=5`, i.e. the *prefill* kernel. The seq==1 decode
  entry (`moe_router_dots_kernel` + `moe_router_select_kernel`) had no
  reference comparison at all, which is why a wrong answer there would have
  been invisible. Now pinned by "router decode-split (seq==1) matches host
  replica" (`test_qwen_moe_kernels`), sweeping E=128/H=4096, E=128/H=2048,
  E=60 (grid not a multiple of the 128-thread block), E=16, E=8 and H=132
  (H not a multiple of 8, the scalar dot path). That test passes unmodified on
  the current neutral code, which is the evidence that the kernel was never
  the problem.
- **FOUND AND FIXED — the bit-exact MoE path multiplied every layer by alpha
  (2026-10-12).** Chasing the golden gates turned up a real defect in the
  overflow-proof SwiGLU above: it divides the activation by `alpha` once, so the
  down projection returns `out / alpha`, but the FP32 accumulator folded back
  `alpha**2`. Every layer's MoE output was therefore `alpha`=16x too large at
  Mistral's layer 32, compounding down the stack.
  **Why nothing caught it:** the Paris SelfCheck still passed in both modes (a
  memorised completion survives large numeric error), the bench still ran, and
  the new shared-expert regression test asserted `cos > 0.99` — **cosine is
  invariant under a uniform scale, so it returned exactly 1.0 for a 16x-wrong
  answer.** Mutation testing had verified that the *prescale* was present, not
  that its factor was right.
  The tell was end-to-end, not numeric: on "Repeat exactly this text: ABCDEFG"
  bit-exact mode returned `The output output output` while the auto (TC) mode
  returned `HIJKLMNOPQRSTUVWXYZ`. Host-oracle confirmation at layer 32: the GPU
  accumulator was 16.0x the reference with `corr = 0.999997`, i.e. right shape,
  wrong scale. After the one-line fix (`coeff * alpha`, not `coeff * alpha*alpha`)
  both modes agree, and the regression test now pins `|got|max / |ref|max` to
  within 5% so a scale error cannot pass again.
- **FIXED — the tensor-core routed FFN overflowed its SwiGLU activation to
  FP16 inf, costing layer 32 ~52x. Root cause, fix and instrument below.**
  The tensor-core branch of `RunMoERoutedFFN` fused SwiGLU into the NVFP4
  activation quantizer (`RunAttnSwigluPreciseQuantInternal`). That kernel does
  `const __half hv = __float2half(sig * g * u);` — it rounds the product to
  **FP16 before quantizing**, and FP16 tops out at 65504. Mistral-Small-4 layer
  32 expert 67 reaches `|silu(g)*u| = 2.5e5` from entirely ordinary FP16
  projections (max 444 and 568), so the store overflowed to `+inf`, the block
  `absmax` became `inf`, `FloatToE4m3(inf)` saturated the block scale, and every
  nibble in the tile quantized from an infinite value. The scalar path was
  immune because it keeps the product in FP32 under a power-of-two prescale.
  Measured on real weights (Paris, layer 32, routed accumulator vs a host
  oracle over the source safetensors):

  | mode | before | after |
  |---|---|---|
  | `kOff` | ratio 0.9998, corr 1.000000 | unchanged (ratio 0.9998) |
  | `kAuto` (TC) | **ratio 0.0192**, corr 0.975 | **ratio 0.9932, corr 1.000000** |

  Fix: `RunAttnSwigluPreScaledQuantInternal` keeps the product in FP32, reduces a
  global `absmax` with an atomic, prescales by the smallest power of two that
  brings it inside both the FP16 and the E4M3-block-scale span, and publishes
  `alpha` in a second slot so `moe_coeff_add_scaled_kernel` folds one reciprocal
  factor into the FP32 accumulator. Two launches instead of one, no host
  synchronization (`alpha` is device-only while `RunLsxTcGemmPreQuant` takes a
  host float). `alpha == 1` is a bit-exact no-op, so every other family is
  unchanged — pinned by a dedicated bit-identity test, and re-verified on real
  weights for Qwen3.6-35B-A3B, Gemma-4-26B-A4B, Xing4.0-29B-A4B and
  Ornith-1.5-35B-A3B, each in both numeric classes. (GLM-5.3-Flash is present
  only as an unconverted HF checkpoint, so it could not be run here.)
  **Why it survived a green suite:** Paris passes in both modes and the bench
  runs, so the suite was green over a layer whose accumulator was off by 52x.
  Nothing compared a *layer accumulator* against an oracle — only final text.
  That gap is now covered by `test_qwen_nvfp4_gemv`: one case forces a
  Mistral-shaped massive activation and asserts the recovered magnitude, one
  asserts bit-identity in the normal range, one asserts a seq==1 projection is
  invariant to a power-of-two activation prescale. Both of the first two fail
  against the old kernel.

  **Cost of the fix: reduced on 2026-10-23.** The extra prequant launches
  are hoisted (three per layer for all K slots, bit-identical), removing
  the standing cost: Mistral 41.3 -> 42.2 t/s at 2048. The scratch and the
  measurement are in
  [`open-tasks/2026-10-23-sampling-neutral-session-and-environment-oracle.md`](open-tasks/2026-10-23-sampling-neutral-session-and-environment-oracle.md).
  The rest of §T (the three golden gates) is still open.

  Two earlier diagnoses of this entry were wrong and are retracted here, since
  the record is the point: (1) "the E4M3 block scale saturates, attenuating the
  expert 94x" — the scale does saturate, but only *because* the FP16 store
  already went to inf; and (2) "Mistral never takes the `tc_fused` branch, so
  the default path is unaffected" — false, because `qwen::SetQwenTcFastMath` is
  an inline alias for `lsx_nvfp4::SetLsxTcFastMath`, the very global the
  engine-init perf policy sets (`gpu_perf.cpp`), so `kAuto` leaves it `true` and
  `kOff` forces it `false`. Checking whether the *model class* calls the setter
  says nothing about the value in force. `LSX_PERF_AUTO=0` (which pins the
  legacy bit-exact policy, `tc=false`) is the cheap one-flag bisect for "is this
  the fast-math path?", and it puts layer 32 back at ratio 0.9998.

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

## Archive

Resolved items moved out of this file on 2026-10-02, preserved with their full
investigation trail:

| Item | Outcome |
|---|---|
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