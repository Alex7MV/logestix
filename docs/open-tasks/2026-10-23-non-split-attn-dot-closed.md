# The non-split attention dot was never an FMA mystery (2026-10-23)

`docs/OPEN_TASKS.md` §I recorded that vectorising `full_attn_softmax_kernel`
"breaks parity" and that the cause had to be FMA contraction, because the row
addressing and the arithmetic were identical to the split kernel's. That
conclusion is **retracted**: the non-split vectorisation was measured with a
helper that summed only half the row.

## What actually happened

`AttnHalfDotRow`'s vector branch consumed two `__half2` per iteration
(`k2[j]`, `k2[j + 1]`, i.e. four elements) while the loop was
`for (j = 0; j < (hd >> 2); j += 2)`. The step of 2 divides the bound, so
there are `hd/8` iterations covering `4 * hd/8 = hd/2` elements: **half the
head**. For `head_dim = 128` the dot summed 64 of 128 elements.

- Introduced with the port (2026-10-05, `QwenHalfDotRow`).
- Fixed on 2026-10-23 by the inference audit, finding **X16** (commit
  `f7cc177`): the bound is `hd >> 1`, and the split path's parity test passes.
- The 2026-10-06 experiment that produced §I ran *before* X16, so the
  half-row dot is a complete explanation of the observed break. Nothing about
  contraction was ever measured.

## Why the trap held for 17 days

The split kernel had the same bug and passed the golden gates, because the
goldens exercise the **non-split** kernel: SelfCheck/FSI/CODE/DET prompts are
short enough that `splits` collapses to one chunk. So the broken split dot was
outside every oracle that existed at the time - it was found by the audit's
split-vs-non-split KV parity test, not by a model gate. Two facts that looked
like they supported "compiler" were really "the same bug in two places, one of
them unobserved".

## The change

`AttnHalfDotRow` moved above the first anonymous namespace (inside
`namespace qwen`) so both kernels share one entity - the earlier attempt failed
compilation precisely because a declaration inside the anonymous namespace is a
*different* function. `full_attn_softmax_kernel` now calls it, and the
"SPLIT KERNEL ONLY" paragraph above the helper is replaced by this record.

## Gates (all on the debug tree, rebuilt from the edited `.cu`)

| gate | result |
|---|---|
| `test_qwen_moe_36_selfcheck` (Qwen3.6-35B-A3B, pinned `logits_hash` + stress cycles) | pass, 33.7 s |
| `test_qwen_moe_36_{fsi,code,det}_selfcheck` | 3/3 pass |
| `test_qwen_dense_selfcheck` (Qwen3.5-9B) | pass |
| `test_alice_moe_selfcheck` + `{fsi,code,det}` | pass (4/4) |
| `test_ornith_moe_{fsi,code,det}_selfcheck` | 3/3 pass |
| `test_gemma_moe_det_selfcheck` | pass |
| `test_gemma_moe_{fsi,code}_selfcheck` | the known §J reds, unchanged: FSI-01/02 and CODE-02 differ only in parenthesisation |

## Measurement (AliceAI-80B-A3B, the family the change reaches)

Alice's 12 gated-attention layers call the neutral non-split kernel, so this is
`OPEN_TASKS.md` §H's "the qwen split-kernel win does not reach Alice" resolved
from the other side. Canonical protocol (AOCC tree, `--bench --max-gen 2048`,
prompt "The capital of France is"), interleaved A/B, one sample per arm per
build:

| arm | run 1 | run 2 |
|---|---|---|
| vectorised non-split dot (new) | **45.6 t/s** | **45.4 t/s** |
| scalar non-split dot (old) | 44.5 t/s | 44.6 t/s |

+2.0%, with both new samples above both old ones. The `alice-foundation-80b-a3b`
perf reference is **not** re-seeded: the recorded 42.64 is a swept
(heat-soaked) reading from the full perf run, and these are cold single runs -
comparing the two would invent a 7% gain. The A/B above is the evidence, and
the baseline's one-sided tolerance already accepts both arms.

One side effect worth recording: running `test_perf_baseline` on alice after the
change took the harness's "improvement" path and rewrote the reference to the
cold 45.42 / 45.47 t/s. That edit was reverted - a cold single run is not the
canonical recording context, and a baseline set to it would turn every later
swept run red. The next full perf sweep re-seeds both rows.

Not measured: qwen/gemma short-context gains. Their decode at 2048 tokens is on
the split path, and the SelfCheck prompts are a single step, so the change has
no production-shape effect to measure there; it is bit-identical by the gates.
