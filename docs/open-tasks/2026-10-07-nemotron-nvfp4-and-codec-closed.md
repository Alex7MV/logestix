# Nemotron-NVFP4 ingestion + E4M3FN block-scale codec - closed 2026-10-07

Verbatim snapshot of the `docs/OPEN_TASKS.md` sections L, M and N as they stood
when the work closed. Moved out of the live file because all three are RESOLVED.
Nothing here is open work - it is kept as the investigation trail, including the
hypotheses that were wrong and how they were ruled out.

Summary for the archive table:

- **L** - the NVFP4 block-scale codec now decodes/encodes the COMPLETE E4M3FN
  table (all three mantissa bits AND `e4==0` subnormals), bit-identical to
  `cuda_fp8`/`torch.float8_e4m3fn` over `0x00..0x7F`. The NVIDIA Nemotron source
  uses subnormal block scales (~3.0M bytes of its 1.89G), which the former
  `e==0 -> 0` decode silently dropped.
- **M** - NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 ingestion: stages 3-8
  (model class on a refactored base, registration, all four guard registries,
  forge flag, stage-8 tests), the 52-shard in-place conversion, and a
  real-weight SelfCheck that passes after fixing the router-load dtype (F32 vs
  BF16). FSI/CODE/DET goldens stay declared debt.
- **N** - granite-4.2-30B tool round-trip, recovered by the subnormal codec
  (was an accepted RED).

---

## L. NVFP4 block scale: logestix stores power-of-two, the checkpoints store full E4M3 (2026-10-06)

> **Status update 2026-10-06: codec DONE; rollout pending.** The decoder now
> decodes all 3 mantissa bits and the encoder writes full unsigned E4M3
> (`FloatToE4m3`/`E4m3ToFloat`, RNE), backward-compatibly. Implemented + tested
> (`test_nvfp4_scale` exhaustive decode, `test_lsx_kernels` device parity, all
> codec/kernel suites green; `fast` 88/91 with the 3 reds being the paused
> Nemotron-NVFP4 registrations). End-to-end validated on a re-converted
> Qwen3.5-9B (Paris selfcheck, CODE, DET pass). The remaining work is
> operational: re-convert the vault families so the encoder change takes effect.
> See `docs/superpowers/plans/2026-10-06-nvfp4-e4m3-rollout-report.md`.
>
> **Status update 2026-10-07: E4M3FN parity DONE (subnormals).** The decoder now
> follows the complete E4M3FN table — `e4 == 0` is the **subnormal** range
> `m*2^-9` (`0x00 -> 0`), bit-identical to `cuda_fp8`/`torch.float8_e4m3fn` over
> `0x00..0x7F`; the encoder rounds RNE on the subnormal grid too, and the
> Nemotron-NVFP4 dequant kernel now calls the canonical codec instead of a
> private decode. Measured on the real vault: 2 995 063 of 1 886 631 936 source
> scale bytes (~0.16 % of blocks) are positive subnormals (`0x01..0x07`), which
> the earlier `e4 == 0 -> 0` reading silently zeroed. See
> `NVIDIA_DOCS_NVFP4_MEMORY.md` §1.

Surfaced by ingesting `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`, the first
vault checkpoint whose weights are *already* NVFP4 (ModelOpt 0.44.0rc5,
`W4A16_NVFP4`, group 16) rather than BF16.

Everything in the two encodings matches — `U8` two E2M1 nibbles per byte with the
even element in the low nibble, one scale per 16 along K, an `F32` per-tensor
global scale. **The only difference is the block scale:**

- **source**: the COMPLETE OCP E4M3FN, i.e. 3 mantissa bits in play;
- **logestix**: `nvfp4_scale.h::FloatToE4m3P2` writes `e4 << 3` (mantissa always 0)
  and `E4m3P2ToFloat` reads only `(byte>>3)&0xF`, i.e. the scale is rounded **up to
  the next power of two** and decodes as `2^(e4-7)`.

This is a deliberate, documented deviation (`docs/NVIDIA_DOCS_NVFP4_MEMORY.md` §1:
exact pow2 identity, no saturation, one numeric class across all 17 families), and
the source format is in fact the **hardware-native** one — Blackwell
`tcgen05.mma` with `kind::mxf4nvf4` takes a **UE4M3** scale factor at
`.block16`. So logestix is the less accurate of the two, knowingly.

**Measured on the real vault** (`backbone.layers.1.mixer.experts.0.down_proj`,
311 808 blocks): the source mantissa uses **all 8 values {0..7}** and the e4 field
spans 10..15; forcing it into the pow2 window (e4 11..15, where 15 is logestix's
E4M3 NaN exclusion) **inflates the block scale by 1.33x median / 1.78x tail**
— up to 77.8 % of weight error on a block.

Consequences, both measured:

| handling of the source block scales | relative L2 vs the source dequant |
|---|---|
| byte passthrough into the logestix decoder | **29.9 %** |
| dequantize with full E4M3, then requantize via `quantize_tensor_gpu` | **10.9 / 11.2 / 11.8 %** (three expert tensors) |

A passthrough is therefore forbidden, and the converter pays ~11 % for every
already-quantized source. That is the price the logestix flavor charges a
checkpoint whose authors used the hardware format.

**Task — teach the decoder the E4M3 mantissa.** The fix is *backward compatible*:
a pow2 byte has `mantissa == 0`, so a full-E4M3 decode returns **bit-identical**
values for all 17 existing families. Nothing is re-baselined and no golden moves;
only the encode side stops discarding information.

- change `nvfp4_scale.h` / `nvfp4_codec.h` so the block scale decodes all 3
  mantissa bits (keep the pow2 *encoder* only if a caller needs a pow2-only
  flavor — no current one does);
- check the TC GEMM scale packing (`lsx_tc_gemm.cu`, the
  `repack_e8m0_ue4m3_kernel` path in `shared_experts.cu`) — the hardware already
  wants UE4M3, so this side should get *more* accurate, not less;
- **pin the zero-scale semantics**: the exact-zero test is `e4 == 0 AND
  m == 0` — never the whole byte (the original bug) and never the bare exponent
  field (which dropped the subnormals — see the 2026-10-07 note above). See the
  regression note in `NVIDIA_DOCS_NVFP4_MEMORY.md` §1;
- then re-measure: with a mantissa-aware decoder the converter can pass the source
  blocks straight through, and the ~11 % requantization loss for pre-quantized
  sources goes to zero.

Blast radius: the model-neutral `lsx_nvfp4` layer plus the TC scale packing, i.e.
everything. That is why it is filed here rather than done opportunistically — it
wants its own numeric-conformance pass (bit-exact against a host reference, and
a re-run of the `fast` label before any perf claim moves).

## M. NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 — finish the ingestion (2026-10-06)

Branch `feat/model-Nemotron3.5-nvfp4-ingest`. This is a **sibling variant** of the
already-ingested BF16 `nemotron-3.5-lightning-30b-a3b` (identical frozen geometry,
verified field-by-field; the only difference is that the source is *pre-quantized*).
Design: `docs/superpowers/specs/2026-10-06-nemotron-3.5-lightning-nvfp4-ingestion-design.md`.
The codec finding this surfaced is section **L** above.

**Status update 2026-10-07 (resolved): stages 3–8 landed and green; the
real-weight SelfCheck passes (3/3 cycles, top-1 "Paris").** The blocker was a
shared-loader dtype bug: the NVFP4 tree stores the router `gate.weight` as
**F32** (ModelOpt left it high-precision) while the BF16 sibling's tree stores
it BF16, and `LoadNemotronWeights` uploaded it with `CvKind::kBf16ToHalf`
unconditionally — so the router read F32 bytes as BF16 and every MoE layer
selected/gated experts from garbage. Fixed by picking the upload path from the
tensor's stored `dtype` (`kF32ToHalf` for F32). The single-prompt framing was
also corrected to render the checkpoint's native `<|im_start|>…<|im_end|>`
ChatML frame instead of a bare role-word frame. Bisection method and the
(non-)conversion evidence are recorded below.

Landed since the note below: the `NemotronNvfp4Model` class on a refactored
`NemotronModelBase` (shared lifecycle/generation contour), factory
registration, the four guard registries, the `--nemotron_nvfp4` forge flag +
CLI test, the `test_nemotron_moe_nvfp4_model` stage test, the
selfcheck/fsi/code/det stage tests (SKIP 4; goldens declared debt) and
`test_perf_baseline_nemotron_nvfp4`. `fast`/`convert` are green (the 3
previously-red guard tests are now green). The 52-shard source was converted
in place (`logestix/`, 18 GiB; 52 layers, 128 experts, 270 MTP tensors cut,
12 FP8 KV scales dropped, 355 pinned tensors).

**The conversion was never the problem** (the failure had two independent
causes: this loader dtype bug, and the shared-code framing). The converted
weights were verified numerically against the source, tensor class by tensor
class (host reference, `torch.float8_e4m3fn` semantics):

| class | check | result |
|---|---|---|
| routed expert up/down (NVFP4) | converted vs source dequant | rel L2 5.3% (NVFP4 grid) |
| mamba in/out_proj (FP8→NVFP4) | converted vs source FP8 dequant | rel L2 9.4% |
| attention q/k/v/o (BF16→NVFP4) | converted vs source BF16 | rel L2 9.4% |
| lm_head (NVFP4→BF16) | converted vs source dequant | rel L2 0.19% (BF16) |
| norms/conv/A_log/D/dt_bias/router/embeddings | raw byte compare | byte-identical |

The block-scale codec is bit-identical to the source (see section L, and the
2026-10-07 note). The failure localised to the shared `NemotronForwardToken`
path. A `transformers` `NemotronHForCausalLM` reference (5.15.1, weights
dequantized from the same source) initially produced, for the selfcheck prompt,
top-5 `[the, Paris, a, \n, one]` and greedy ` the city of Paris.` while the
logestix model echoed the prompt (` the capital of France…`); feeding the
reference the **converted** weights instead produced ` Paris…`, which ruled out
the conversion and pointed at the loader. Per-layer hidden-state bisection then
showed layer 0 (mamba) matching to <0.8% and layer 1 (the first MoE) diverging
by >25% — the router.

Two harness defects were found alongside the runtime bug (neither is the root
cause): the selfcheck feeds `arrow_tokenizer_encode("The capital of France is")`
which auto-prepends `BOS=1`, whereas the reference tokenizer adds none; and
`BuildPromptInputIds` uses a bare `user\n…\nassistant\n<think>\n` frame while
the checkpoint's `chat_template.jinja` uses `<|im_start|>…<|im_end|>` blocks
(the conversation renderer is already correct). A candidate runtime deviation
was also noted: `kDtMin=0.001` is applied in prefill but not decode, while the
reference applies its `dt_limit` in both phases (`mamba2_selective_state_update`
is not passed `dt_limit` and defaults to `(0, inf)`, the chunk scan uses
`time_step_limit=(0, inf)`).

Reference reproduction: `/tmp/opencode/nmv4_hf_ref3.py` (loads the source with
`transformers` `ignore_mismatched_sizes`, dequantizes the NVFP4/FP8 weights on
the GPU, runs greedy generation). Next step: bisect the logestix forward
against this reference (layer/block-level), starting with the mamba state
carry-over across the token-by-token prefill (`NemotronForwardToken`).

**Status: stages 1–2 done and green; stage 3 half-landed; stages 3–8 still open.**
Landed so far: converter + full-E4M3/FP8 dequant kernels
(`nemotron_moe_nvfp4_convert.*`, `nemotron_moe_nvfp4_dequant.{h,cu}`) with
`test_nemotron_moe_nvfp4_convert` passing; loader
(`nemotron_moe_nvfp4_load.{h,cpp}`) with `test_nemotron_moe_nvfp4_loader` passing;
variant passport `nemotron_moe_nvfp4_constants.h`; the shared rank-0
`TensorInfo::numel()` fix in `converter.cpp`; the index reader's int-or-array eos
(`eos_token_ids`, scalar `eos_token_id = front()` = 2, second stop id 11).

Remaining, in order:

1. **Stage 3 registration.** `NemotronNvfp4Model` (`kType =
   "nemotron-3.5-lightning-30b-a3b-nvfp4"`) in `model_registry.cpp`; the model
   test (`test_nemotron_moe_nvfp4_model`, `inference`) driving the parent's
   loader/kernels/expert cache.
2. **All four guard registries in the SAME commit as the files they police**
   (Model Ingestion Protocol step 8):
   `SchemeRegex()` variant token `nvfp4` (`test_code_naming_guard`);
   `kPassports` row → `nemotron_moe_nvfp4_constants.h` / ns
   `lsxcommon::nemotron_nvfp4::frozen` (own namespace — the parent's is not
   reusable); `kLifecycle` row; `Suites()` + tool-verdict rows and the
   `kGoldenDebt` rows (`test_model_suite_guard` / `test_model_independence_guard`).
3. **Stage 1 pipeline flag** `--nemotron-nvfp4` in `apps/lsxforge/src/main.cpp`
   (the converter TU is written; only the CLI entry point is missing).
4. **Stage 4b** reuses the parent's Qwen-style XML render/parse verbatim — record
   the tool-verdict row only.
5. **Stage 8 tests** not yet written: `test_nemotron_moe_nvfp4_selfcheck` and the
   `{fsi,code,det}` variants (`SKIP_RETURN_CODE 4`), plus
   `test_perf_baseline_nemotron_nvfp4` + `perf_baselines.json` entry.
6. **Real-weight run.** Convert the 20.08 GiB / 52-shard vault checkpoint
   (~17.5 GiB arrow) and pass SelfCheck on real weights before the goldens above
   mean anything. Nothing has exercised the full converted model yet.
7. **FSI/CODE/DET golden capture — attempted 2026-10-07, still debt.** HF
   transformers cannot load the ModelOpt checkpoint directly ("Unknown
   quantization type, got modelopt"), so I dequantized the source to a bf16
   `NemotronHForCausalLM` (61 GiB) and ran the capture tools against it. Two
   blockers, both empirical:
   - `accelerate` `device_map="auto"` offload corrupts this custom hybrid
     (Mamba-2 selective-scan + MoE) — the model emits repetition loops; a plain
     CPU run is pathologically slow for a 30B model here.
   - more fundamentally, the source reference loops/degenerates on all three
     synthetic prompt sets (FSI/CODE/DET) in every framing (thinking on/off/raw
     prompt), so no canonical answer can be extracted. The converted logestix
     model *does* reason coherently on the same FSI-01 prompt, so a golden
     captured from the source cannot be produced (and capturing from the
     converted tree would make the gate circular).
   The three goldens stay in `kGoldenDebt` (same as the BF16 sibling). Revisit
   on a machine with enough VRAM to run the 30B bf16 reference without offload,
   or with a working ModelOpt loader.

**Do not** byte-pass the source NVFP4 blocks to the repo decoder — measured
29.9 % relative L2 (section L). The converter dequantizes with full E4M3 and
requantizes through `quantize_tensor_gpu`; that costs ~11 % vs the source and is
the accepted price of the pow2-only block scale.

## N. Granite-4.2-30B tool round-trip regressed by the full-E4M3 encoder — **resolved 2026-10-07**

> **Resolution (2026-10-07).** Re-converting granite-4.2-30b after the E4M3FN
> subnormal alignment (section L, 2026-10-07) restores the tool round-trip:
> `test_granite_dense_selfcheck`'s real-weight tool case emits
> `get_weather({"city":"Paris"})` and Paris SelfCheck / FSI / CODE / DET all
> pass (8/8 granite targets green). The subnormal encoder changed the offline
> block scales on tiny blocks (subnormal instead of clamped to the min normal),
> which moved enough weight mass to flip this chaotic greedy decision back.
> The historical analysis below is kept for the record; the section is no
> longer an accepted RED.

Re-converting granite-4.2-30b with the full unsigned-E4M3 block-scale encoder
(section L) makes `test_granite_dense_selfcheck`'s end-to-end tool round-trip
fail: the model no longer emits a parseable tool call (it reasons in prose, or
degenerates). Paris SelfCheck / FSI / DET still pass. The regression is
**accepted, not fixed** — the encoder is intentional.

Attribution (measured on the RTX 5090 host, real weights):

| offline weight encoder | runtime act-quant | tool round-trip |
|---|---|---|
| pow2 (pre-E4M3) | E4M3 | **pass** (`get_weather` called) |
| full E4M3, RNE | E4M3 | fail |
| full E4M3, ceil | E4M3 | fail |

So the trigger is specifically the **offline weight encoding carrying the
mantissa**; the E4M3 runtime act-quant and the E4M3 decoder are both fine (the
pow2 tree above still ran E4M3 act-quant and passed). The E4M3 weights are
*better* by weight relative L2 (~9.5 % vs ~11.5 % on every sampled tensor), so
this is a chaotic greedy-decision flip, not a codec defect.

Things excluded during the investigation:

- **Not framing/tokenization.** Feeding the exact HF `apply_chat_template`
  token ids (315, native ChatML + tools) to the converted model still
  degenerates. The logestix rendered prompt text matches HF byte-for-byte; the
  logestix tokenizer segments it slightly differently (274–279 vs 298–315) but
  the ids round-trip to the same text.
- **Not the numeric mode.** Fails at both `kAuto` (TC) and `kOff` (scalar).
- **Not the runtime act-quant.** See table.
- **Not the generic-vs-native renderer.** The native `<tool_call>` XML framing
  (granite's own `chat_template.jinja`) degenerates worse than the generic
  Hermes fallback; both fail.

Consequences and the accepted position:

- `test_granite_dense_selfcheck` tool round-trip stays **RED** on real weights
  (the same disposition as the gemma FSI/CODE gates in section J). Do not
  "fix" it by loosening the assertion or by reverting the codec.
- The general BF16→NVFP4 accuracy gain from the mantissa stays in place; the
  cost is this one model's tool-calling trajectory.
- The decoder half of the transition remains backward-compatible; a `mantissa
  == 0` byte decodes bit-identically, so pre-E4M3 trees are unaffected.

If revisited, the one lead that was not exhausted: feed the **source BF16**
granite the identical tool prompt (HF) to establish the reference call, then
diff the converted logits at the decision step to see whether the E4M3 tree is
further from BF16 than the pow2 tree on the tool head (a specific tensor, e.g.
the gate/embedding, could be the flip point).
