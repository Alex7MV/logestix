# gemma-4-26b-a4b-it: FSI + CODE RED -> residual plane -> all four gates green
(2026-10-24)

Closes §J of `docs/OPEN_TASKS.md` (FSI-TEST-01/02 + CODE-TEST-02 on the MoE
member). The dense 31B sibling (§R) stays open: the same recipe does not fit
its budget, see the last section.

## 1. Two defects, not one

**(a) The NVFP4 logits kernel dropped the tied table's global scale.** Fixed
in `76a13afc`. `gemma_logits_rows_nvfp4_kernel` decoded
`weight = e2m1 * e4m3_scale` and never multiplied by the per-tensor
`global_scale`, so every gemma-4-26b logit was 1/gs = **7.01x** hot
(gs = 0.1426): raw logits read ~250 where the source reads ~35 and the tanh
soft cap (30) saturated. Greedy argmax is scale-invariant, so SelfCheck/DET
stayed green while the logit row, the temperature/top-k semantics and the
SelfCheck logit evidence were all wrong. The existing latency test passes
`gs = 1`, where the defect is invisible -- that is how it shipped. Pinned by
`test_gemma_moe_moe` "gemma NVFP4 logits apply the table global scale"
(hand-written CPU table, gs = 0.25).

That fix did **not** move the FSI/CODE tokens: ordering is invariant under a
positive scale, so the flips had a second cause.

**(b) Accumulated 4-bit body error on template near-ties.** Measured with
teacher-forced prefill logits (`final_logits`, pre-softcap) against the source
(`Gemma4ForConditionalGeneration`, bf16, `final_logit_softcapping = None`):

- the engine is *faithful*: residual stream vs source cos 0.9992 after layer 0,
  0.960-0.965 at layer 29, rms ratio 0.94-1.03; the layer-0 router picks the
  identical top-8 expert set (a mismatched set was the first hypothesis and is
  refuted). The drift is plain NVFP4 weight error (embed table rel. err 9.6%).
- error decomposition at the head: with the source's exact hidden the 4-bit
  tied table costs max 1.05 / rms 0.36 logits; with our hidden and the source's
  exact table it is max 5.58 / rms 3.46. The **body** owns the error, so FP16
  edges (the phi-4 lever) would not have fixed these.
- the decisive pairs (pre-softcap): FSI-01 step 2 after `PV=` -- source `CF`
  +12.4, converted `2` +9.9 (22-logit swing); FSI-02 the numeric substitution
  vs the symbolic denominator chain; CODE-02 step 3 -- source `(` +2.5,
  converted `+(` +2.8.

## 2. What fixes it

Offline recipe ablation (in-place NVFP4 quantization of the HF model with the
same block/e4m3 recipe, teacher-forced 40 steps; `all-q4` reproduced the
engine's flips exactly, which validated the instrument):

| recipe | flips | top-5 abs err mean |
|---|---|---|
| all 4-bit (shipped) | FSI-01 s2,s3; CODE-02 s3 | 0.76-1.70 |
| experts 4-bit, attn+MLP bf16 | none | 0.51-0.85 |
| **attn+MLP 8-bit, experts 4-bit** | **none** | 0.53-0.95 |
| all 8-bit | none | 0.36-0.56 |

The chosen recipe is the third: a second NVFP4 plane (`<name>.res` =
`w - dequant(base)`, decoded with one more GEMV with `acc=true`) on the
attention projections and the dense MLP. Experts stay 4-bit -- they are 88% of
the weights and they own the streamed read volume, so upgrading them would
have cost decode throughput for no measured gain.

## 3. Implementation

- `gemma_moe_convert.cpp`: emits `<name>.res.{packed,scales,global_scale}` for
  Q/K/V/O and the dense-MLP gate/up/down (`quantize_residual_tensor_gpu`, the
  phi-4 codec/schema, no new format); index field `"residual": "attn+mlp"`;
  the VRAM policy's `has_residual` doubles those tensors.
- `gemma_moe_load.{h,cpp}`: `attn_*_res` / `mlp_*_res` members, uploaded from
  the `.res` triple when present (a legacy 4-bit-only tree still loads: the
  weight stays null and the decode skips the accumulate pass). **`Free()`
  releases them** -- first cut leaked them, which showed up as a dense-family
  SelfCheck OOM after two loads.
- `gemma_moe_inference.cu`: one extra `RunGemmaNvfp4Gemm(..., acc=true)` per
  attn projection, and `RunGemmaDenseMlpNvfp4Chain` grew a residual form (the
  17-arg entry point stays as a wrapper) that accumulates each projection's
  plane with `acc=true` -- base dot rounds to FP16, then the residual dot adds
  and rounds once more, pinned by
  `test_gemma_moe_moe` "gemma dense MLP chain accumulates the residual planes
  with acc=true".

TDD: the converter case (`test_gemma_moe_convert`, residual records + base+res
reconstruction rel-L2 < 0.5x base, and *no* residual on the router or the tied
table), the loader case (`test_gemma_moe_loader`), and the chain case all
failed before the implementation.

## 4. Result (real weights, RTX 5090)

- Re-converted in place: 15 GiB (was 14, +1.5 GiB of attn+MLP planes);
  resident pinned set 2.29 GB/layer-set reported as 2.29e9 bytes.
- **SelfCheck PASS, FSI 3/3, CODE 3/3, DET 3/3** (all four gates green for the
  first time on this family).
- Throughput: **128.25 t/s** (perf) / **128.34 t/s** (r3) at 2048 tokens, TTFT
  127/121 ms (was 156.72 / 156.46; the first post-change sample recorded
  125.11/127.16 and the harness re-recorded on the next cooled run). The -18%
  is the doubled attn+MLP read -- attributed, expected, and re-seeded through
  the harness with the power and TTFT evidence, not hand-edited. The dense
  31B rows (47.62/47.63) are plain harness improvements on an unchanged model.

## 5. The dense 31B sibling does not fit this recipe (stays open, §R)

`gemma-4-31b-it` FSI-01 forks at the same position (source `2` 55.5 vs `CF`
42.75; the engine prefers `CF`). Measured in the engine itself (each variant
is a 65 s re-conversion plus a probe):

| dense residual placement | size | step-2 margin |
|---|---|---|
| none (shipped) | 17 GiB | `CF` +2.7 |
| `o` + `down` (offline-sim winner) | 22 GiB | `CF` +6.3 |
| `q,k,v,o` + `down` | 25 GiB | `CF` +10.4 |
| all linears | 33.7 GiB | OOM (over the card) |

Two lessons: the offline HF simulation is only qualitative -- its all-4-bit
baseline already sat on the source's side of this fork (its error is much
smaller than the engine's), so "down+o suffices" did not transfer; and on the
dense body more precision kept moving *away* from the source at this token
pair, so the required upgrade is broader than the budget allows. All the dense
residual plumbing was reverted (tree back to the 4-bit-only 17 GiB format) --
the family is unchanged from its pre-session state (FSI 2/3, everything else
green), and getting it green needs ~8 bits on every linear, i.e. a 48 GiB
card or a different block format.
