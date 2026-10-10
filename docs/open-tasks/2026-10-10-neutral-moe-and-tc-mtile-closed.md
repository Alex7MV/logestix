# Neutral routed-MoE layer, ornith de-duplication, TC M-tile fix - closed 2026-10-10

Verbatim snapshot of the parts of `docs/OPEN_TASKS.md` that this work session
closed (§R's shared TC defect and §S's ornith de-duplication), plus the two
coverage gaps found during review. Moved out of the live file because all of
them are RESOLVED. Nothing here is open work - it is kept as the investigation
trail, including the hypothesis that turned out to be **wrong** and how it was
ruled out.

Summary for the archive table:

- **TC M-tile** - `lsx_tc::TcGemmKernel` wrote only rows `[0,16)` of its 64-row
  M tile: the row-group index `wm` was computed and never used. Fixed in all
  four row computations; the TC shape sweep now covers `seq` 17/18/33/64 and
  reproduces the failure on the old kernel.
- **ornith de-duplication** - the 2182-line `ornith_moe_inference.cu` copy is
  gone; ornith calls the neutral `lsx_moe_inference` layer and keeps only its
  own generation pipeline. The expert-store loader cluster, identical in qwen,
  glm-flash and ornith, moved into the neutral layer. Net **-2776 lines**.
- **Coverage gaps** - the Ornith fused-expert split and gemma `final_logits`
  were never read back by any test; both now are, the first verified by
  mutation.

---

## The shared TC defect: wrong root cause, wrong scope, wrong impact

### What was believed

From the gemma-4-31b-it ingestion (section R, recorded 2026-10-09):

> **CRITICAL (shared layer): the TC mxf4nvf4 GEMM drops the partial M tile.**
> A prefill whose length is not a multiple of the MMA tile (measured: seq=18)
> leaves the last rows of every projection at ZERO (`lsx_tc::TcGemmKernel`,
> reached when fast-math is on and `seq > 1`). With `seq=18` the last two rows
> of the prefill hidden are 0, so the last token's logits are 0 and the first
> generated token came back as 0. `LSX_PERF_AUTO=0` (bit-exact scalar path)
> reproduces the correct `Paris.`, so the scalar kernel is fine and the fault
> is in the TC dispatch/kernel's partial-M store. This affects EVERY family
> whose `kAuto` prefill length is not tile-aligned, not just gemma (the parity
> gates hide it because they run `kOff`). **Family-local workaround landed:**
> `gemma_dense_inference.cu` runs the prefill projections with
> `SetLsxTcFastMath(false)` (restoring after; decode is GEMV and unaffected),
> which costs TTFT (56 -> 337 ms at an 18-token prompt) but is correct.
> **Proper fix (open):** guard the TC kernel's partial M sub-tile store so the
> valid rows in a short tile are written; then the workaround and the TTFT
> cost can go.

### Why the belief was wrong

The hypothesis was a *partial-M store* bug, i.e. something to do with a tile
that is only partly valid. Reading the kernel rules it out: the epilogue
bounds-checks `r0 < seq` on all four stores, and the A-staging loop covers the
full `kTileM = 64` rows. Both look correct.

The defect is a **dead variable**. The block is 256 threads = 8 warps
"arranged 4x2; each warp covers 16 rows x 32 cols" (the kernel's own comment),
so the M tile spans `wm * 16`. But:

```
470:    const int wm = warp >> 1;          // 0..3
```

`wm` was never referenced again in the entire file. Every row index was
`row_base + g`, where `g = lane >> 2` is 0..7, and the second register row is
`+8` - so a block emitted rows `[0,16)` and nothing else. Rows 16..63 of every
tile were **silently never written**, for every shape. It only *looked* like a
partial-tile bug because `seq=18` is the smallest input that exposes it.

So three of the recorded claims were wrong:

| Recorded | Actual |
|---|---|
| "partial M tile is dropped" | 48 of every 64 rows are dropped, always |
| "a prefill whose length is not a multiple of the tile" | any `seq > 16`; alignment was never the issue |
| "affects every family whose prefill length is not tile-aligned" | every family that routes `seq > 16` through this kernel at all |

### Why it survived a year of gates

`tests/test_blackwell_tc_mma.cu` swept only `seq` in `{1, 3, ...}` - the two
values below the 16-row warp group, i.e. exactly the range where the bug cannot
appear. The parity gates did not help either: they run `FastMathMode::kOff`,
which never reaches the TC path. The only evidence was gemma's SelfCheck,
which is what surfaced it.

### Reproduction and fix

The shape sweep now includes `{17,544,36} {18,544,36} {33,544,36} {64,544,36}`.
On the old kernel `seq=18` reports `max_abs_err = 1055.79` against
`max_ref = 1284.51` - i.e. entire rows unwritten - while the `seq<=3` shapes
report ~0.2.

The fix is one line of intent applied in four places (acc-in seeding, the A
fragment shared-tile index, the A scale row `sr`, and the epilogue store):

```
const int warp_row = wm * 16;
```

The `seq == 1` decode kernel is untouched: it has its own single-row layout and
an early sed hit on it during the fix, which was reverted (it is not a
`wm`-shaped kernel).

### Impact on gemma-4-31b-it: the workaround stays, but for a different reason

With the kernel fixed, gemma's prefill was put back on the TC path to recover
the TTFT. It produced `L. Paris.` instead of `Paris.` and **broke the tool
round-trip** (`calls.empty()`). The TC path is numerically approximate; gemma's
FSI/CODE/DET goldens were captured on the scalar path, so enabling it changes
the family's verified numeric class.

The family-local `SetLsxTcFastMath(false)` prefill therefore **stays**, with
the comment rewritten to say it is a choice, not a necessity. Its TTFT cost
(56 -> 337 ms at 18 tokens) is retained deliberately, scoped to this family.

### Verification

`fast` 91/91, `convert` 23/23, TC kernel tests 5/5. Real weights: gemma_dense
SelfCheck/DET/CODE, ornith SelfCheck/DET, qwen_moe_36 and gemma_moe SelfCheck
all pass. (`FSI-TEST-01` remains red for its own documented reason.)

---

## Ornith de-duplication onto the neutral layer

Recorded in section S after the rename unblocked it.

`ornith_moe_inference.cu` was 2182 lines, ~97% identical to the qwen kernel
layer, and had **already drifted**: its device kernels were still literally
named `qwen_moe_router_dots_kernel` etc., its namespace close read
`} // namespace qwen_moe` inside `namespace ornith`, and it was missing the
TC-routed expert chain the neutral layer had gained. None of that was caught,
because nothing read it back.

What was kept: `ornith_moe_inference_pipeline.{h,cu}` - ornith's generation
driver, welded to `OrnithMoePinnedWeights` exactly as qwen's is to
`QwenMoePinnedWeights`.

What moved into the neutral `lsx_moe_inference.{h,cu}`: the routed-MoE kernels,
the expert store (`MoEExpertStore`, `MoEExpertRecord`, `MoEExpertMetaDev`),
`EnsureMoEExpertResident`, and the logestix-format expert loader
(`LoadMoEExpertStore`, `AllocateMoEExpertCache`, `FreeMoEExpertCache`) - the
last of which had been duplicated verbatim in **qwen, glm-flash and ornith**.

Net **-2776 lines** (19 files, +376/-3152).

Real-weight gates: ornith SelfCheck, ornith DET, qwen_moe_36 SelfCheck,
gemma_moe SelfCheck all pass.

---

## Two coverage gaps found during review

Neither was an open task; both were found by review and closed the same session.

**Ornith fused-expert split had no value coverage.** Both converter
round-trips stopped at `fs::exists(experts/layer_0/exp_0.arrow)`, so a gate/up
swap in the transformers `gate_up_proj [E,2I,H]` split would have passed the
suite silently. The first version of the new test was *symmetric* (compare two
stampings of `w1(A)` against `w3(B)`) and was rejected after the mutation check
showed it passing **under the very swap it was meant to detect** - swapping the
plateaus swaps `w1` and `w3` in both runs, so the equality is invariant. The
shipped test is asymmetric instead: the gate half is stamped constant so every
NVFP4 block shares one amax and `w1`'s block-scale bytes are all equal, while
the up half varies per block so `w3`'s are not.

**gemma `final_logits` were only compared cycle-to-cycle.** A buffer that is
never written is still bit-identical across runs. The new test pins size,
non-degeneracy, the tanh softcap bound and the argmax-vs-last-token
relationship. Note the trap it exposed: `final_logits` is the row from the
**last** decode step, so the matching token is `gen_tokens.back()`, not
`first_gen_tok_id` - the first draft of the test asserted the latter and failed
on correct code.