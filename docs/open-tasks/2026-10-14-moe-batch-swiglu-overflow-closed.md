# Batched TC MoE swiglu overflowed FP16 and poisoned the block scale (2026-10-14)

Item **§V3** of [`../OPEN_TASKS.md`](../OPEN_TASKS.md), closed the day the fix landed. The investigation trail is kept verbatim below: the wrong hypotheses (GEMM, TMA row width, the residency mirror) and how each was ruled out are the point.

### V3. `test_qwen_moe_kernels` / `test_ornith_moe_kernels`: the batched TC chain and the legacy chain disagree (found 2026-10-14, after bad0792)

Both cases ("tc batch routed ffn matches legacy chain bitwise") compare
`RunMoERoutedFFN` (sequential, per-route, asks for residency at call time)
against `RunMoERoutedTcBatch` (composes its slot table from the residency
mirror). They disagree qualitatively: 64 of 64 elements differ, and the batched
result sits at -4547.5 where the sequential one is -1.27e6. Not an
accumulation-order reassociation.

Chasing it fixed two real defects on the way (`bad0792`): the residency mirror
was never invalidated on device-slot eviction, so it named slots that had moved
to another expert — a silent wrong-expert read in every streamed model; and both
cases warmed 32 experts into 8 slots while asserting a full-residency
precondition that fewer slots can never satisfy.

Ruled out since: the storage representation (both chains reach the same staged
weights), mirror staleness, the TC fast-math flag (the batched output is
identical with it on and off), slot capacity, and the degenerate geometry
(I=16 fails the same way a realistic I=128 does).

**Localised, then identified: `RunLsxTcGemmBatch13` writes ZEROS where the
other two implementations agree exactly.** Three implementations were run on
the same expert, same NVFP4 activations, same staged weights (K=1, one route,
eid 3, coeff 1):

* the per-slot TC GEMM (`RunMoERoutedFFN` with xq/xqs),
* the fp16 V-batch core (the same entry point with xq/xqs nulled),
* the batched TC GEMM (`RunMoERoutedTcBatch` -> `RunLsxTcGemmBatch13`).

Measured against the fp16 core as reference:

| shape | relerr, per-slot TC | relerr, batched TC |
|---|---|---|
| H=64, I=16 (N=32) | **0** | **1** |
| H=64, I=32 (N=64 = kTileN) | **0** | **1** |

relerr 1.0 is exactly the error of a zero vector: the batched GEMM writes
nothing, while the other two agree bit-for-bit and are non-zero. The N-tile
hypothesis (N must be a multiple of kTileN=64) was tested and is **wrong** -- a
full 64-wide tile fails the same way.

Earlier narrowing that still holds: the divergence is in the w13 output, before
swiglu and before the reduce (gate slot 0 differs 16/16, up half identical
zeros in both), so batching, the reduce and coefficient handling are all
excluded. It is also not eviction, slot capacity or the TC fast-math flag.

**The slot table is not the problem.** Dumped from device after the batch call:
`dev_slot=0`, `gs1 = gs2 = gs3 = 0.0833333`, `coeff = 0.6 / 0.4`, matching
`records[0][3]` exactly -- both routes correctly resolve to the same slot. Since
the batched epilogue scales every output column by that table's `gs1`/`gs3`, a
zeroed table would have produced the observed all-zero output; the table is
sound, so the cause is inside the MMA/TMA path.

**Confirmed at a second, CORRECTLY built geometry.** The fixture really does have
to be rebuilt when I changes -- `moe_intermediate_size=128` written to
config.json, `records[0][0].w1_bytes = 4096` (= 64 x 128 / 2, was 512):

| geometry | row bytes | w1_bytes | ||fp16|| | relerr, per-slot TC | relerr, batched TC |
|---|---|---|---|---|---|
| I=16 (fixture default) | 8 | 512 | non-zero | **0** | **1.00** (all zeros) |
| I=128 (rebuilt, verified) | 64 | 4096 | 9518 | **0** | **1.71** |
| I=512 (rebuilt, verified) | 256 | 16384 | 15378.7 | **0** | **1.75** |

Each row is a separately rebuilt checkpoint -- `w1_bytes` confirms the weights
really changed (512 = 64x16/2, 4096 = 64x128/2, 16384 = 64x512/2), and the
runner refuses to interpret a run whose fixture was not rebuilt.

**The TMA row-width hypothesis is refuted.** It predicted the batched path would
start working once the packed weight row reached a multiple of 128 bytes. I=512
gives exactly 256 bytes per row -- a production-shaped row -- and the batched
path is still wrong (relerr 1.75), while the per-slot path is still exact. The
failure is not a small-shape artefact.

The batched path is wrong at **every** geometry tested, including one whose row
width matches production, with two different failure signatures (zeros at I=16,
wrong values above it); the per-slot TC path matches the fp16 core exactly at all
three. Combined with the real-weight divergence this is now a family-level
defect, not a synthetic curiosity: the 2x the batched path buys is being paid for
with wrong arithmetic.

The production path IS live, and it changes real generations.** `tc_batch` is
enabled when TC fast-math is on (the default perf policy), the segment scales
exist, and neither LSX_GEMV_EXPERTS nor LSX_TC_BATCH_OFF is set -- so qwen MoE
and ornith decode through `RunMoERoutedTcBatch` and the real-weight gates pass
through it. Measured on Qwen3.6-35B-A3B with `lsxfabric --testprompt`, after a
determinism control (two runs of one configuration produce identical text, so
the harness is not the source of variation):

| configuration | output |
|---|---|
| default (batched TC on), run 1 | "Paris is a major tourist destination, attracting millions of visitors each" |
| default, run 2 | identical |
| `LSX_TC_BATCH_OFF=1` (legacy per-slot), run 1 | "Paris is located in the north-central part of France, along the Seine River..." |
| `LSX_TC_BATCH_OFF=1`, run 2 | identical |

So the two paths disagree **deterministically on real weights**, not only on the
synthetic case. That makes the red test a true alarm rather than a fixture
artifact, and it matches the claim in the pipeline comment ("bit-identical to
the legacy per-slot chain, pinned by test_qwen_moe_kernels") being false in
production.

Which path is faithful is still open, and the parity gates cut both ways:
FSI/CODE/DET are captured from the source safetensors (an external oracle) and
pass WITH the batched path on, so it is adequate but demonstrably not bit-
identical; DET only checks run-to-run stability within one configuration, so a
systematically different path passes it. Deciding needs an oracle finer than the
canonical answer -- e.g. a logits-hash comparison against the fp16 core, which
the per-slot TC path already matches exactly on synthetic shapes.

**What it costs to be safe.** Measured on Qwen3.6-35B-A3B, 256 tokens:
batched TC **166.9 t/s**, legacy per-slot **82.1 t/s** -- the unverified path is
**2.03x** faster. So "fall back to the verified path" is not free, and the
decision is a real trade, not a cleanup:

* the batched path is 2x faster, passes the external-oracle gates
  (FSI/CODE/DET, captured from the source safetensors) and deterministically
  produces different generations than the legacy path;
* the legacy path is bit-identical to the fp16 core at every geometry tested
  and 2x slower.

Nothing here says the batched path is wrong at PRODUCTION geometry -- the
synthetic failures are all at shapes no real model has. It says the invariant the
pipeline comment claims ("bit-identical, pinned by test_qwen_moe_kernels") is
false, and that the batched path has never actually been verified against a
logits-level oracle.

**The missing instrument**, stated after the above:

**Verification pass (re-checked every claim in this section from scratch).**

Confirmed, through instrumentation the tests own rather than ad-hoc probes:

* `test_qwen_moe_kernels` at I=16: `DIFF n=64/64 worst=1.26e6 @20 seq=-1.27e6
  batch=-4547.5`; at I=512: `DIFF n=64/64 worst=4.28e6 @34 seq=4.28e6
  batch=1594.8`. Every element differs, catastrophically. `test_ornith_moe_kernels`
  fails the same way.
* Real weights, identical prompt ("The capital of France is"), only the knob
  changed: batch ON -> "Paris is a major tourist destination, attracting
  millions of visitors each"; batch OFF -> "Paris is located in the north-central
  part of France, along the Seine River." Deterministic, reproduced.
* SelfCheck/FSI/DET for qwen3.6 **pass with the batch on** (34s / 56.7s / 91.2s)
  **and FSI passes with it off** (56.9s). The gates genuinely cannot see this.
* Perf, 256 tokens: 166.7 vs 81.5 t/s, 2.04x -- reproduced.
* TMA row-width hypothesis refuted (I=512, rowB=256, test-own DIFF above).

**Correction to this section's own earlier claim.** It said the legacy path is
"bit-identical to the fp16 core at every geometry tested". That is a KERNEL-level
result (the neutral `RunMoERoutedFFN` fp16 backend vs its per-slot TC backend on
the same inputs), and it stands. It does NOT make the legacy path the faithful
PRODUCTION reference, and this section should not have implied it: the obvious
production control, `LSX_PERF_AUTO=0`, resets the whole knob set
(`gpu_perf.cpp` `r = {LsxPerfKnobs{}, kLegacy}`: split-K, GEMV MiB, expert slots,
dual-stream, not just fast-math) and produces `"The capital of France is London.
The capital of France is Berlin."` -- nonsense from an uncontrolled config, not a
clean fp16 oracle. So "which side is faithful in production" remains UNRESOLVED;
what is established is only that the two TC paths disagree, deterministically,
and that the acceptance gates cannot tell them apart.

### RESOLVED 2026-10-14 — the batched swiglu stored to FP16 before quantizing

The divergence was NOT in the GEMM. Stage-by-stage, with the tests' own
instrumentation: the batched `w13` output is bit-exact to the per-slot GEMM for
every slot (`nd=0/32`), but the batched `w2` output is wrong (`nd=64/64`, relerr
1.33) — and the stage between them, the swiglu, produced an `h` of `inf`.

`moe_swiglu_quant_batch_kernel` computed `silu(gate)*up`, stored it to FP16
*before* quantizing, then took the block absmax from the rounded value. When the
product exceeds 65504 (super common for massive-activation experts:
Mistral-Small-4 layer 32 expert 67 reaches 2.5e5) it overflows to +inf, absmax
becomes inf, `FloatToE4m3(inf)` saturates the block scale, and every nibble
quantizes from an infinite value. The per-slot path already had the fix
(`RunAttnSwigluPreScaledQuantInternal`: FP32 product, per-slot power-of-two
prescale, reciprocal folded back in the accumulator); the batched path kept the
old form. Both test cases' geometry drives the product past 65504, so the case
failed somewhere no real shape made it obvious — and real Qwen/Ornith
generations diverged because they hit the same overflow.

Fix: `moe_swiglu_absmax_batch_kernel` (FP32 product, per-slot absmax) +
`moe_swiglu_quant_prescaled_batch_kernel` (per-slot power-of-two prescale, FP16
store, quantize from FP32, publish the reciprocal), and the batched reduce folds
`slot_prescale[s]` in exactly as `moe_coeff_add_scaled_kernel` does. Bit-exact
no-op when a slot already fits (alpha == 1). Both `test_qwen_moe_kernels` and
`test_ornith_moe_kernels` are green with the assertions untouched (998 and 877
assertions), real Qwen3.6 batch-on and batch-off now emit identical text, and
SelfCheck/FSI/DET still pass. Cost: one extra elementwise launch per layer,
Qwen3.6 256-token bench 166.7 -> 161.3 t/s (2.04x -> 1.98x over the legacy
81.5 t/s path, which is unchanged).

Also repaired here: commit `bad0792` had accidentally deleted
`LoadMoEExpertStore` from the "generation driver deterministic" case while
editing the sibling's slot count, so that case had been failing on an unloaded
store since.
