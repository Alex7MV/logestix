# Granite-Switch 4.1 8B FSI-01: a measured residual-plane placement (2026-11-07)

Closes `docs/OPEN_TASKS.md` §G3. The 8B sibling's FSI-01 was red on the first
conversion; it is green now, and the interesting part is *how* the fix was chosen.

## The fork

FSI-01 asks for the present-value formula. Source (BF16, CPU):
`PV=CF*((1-exp(-r*n))/(1-exp(-r)))`. First conversion:
`PV=2500000*((1-exp(-0.0835*7))/(1-exp(-0.0835)))` — the same formula evaluated
with the prompt's own numbers, all three correct. The decision is the first symbol
after `PV=`: the variable name or its value.

## Step 1 — a tie, or a bug?

Teacher-forced the golden prefix (`PV=` → ids 49569, 28) through both models and
compared the rows at that position:

| | `CF` | `250` | margin |
|---|---|---|---|
| source | +35.75 | +28.88 | **+6.88** |
| converted (4-bit only) | +30.08 | +31.34 | **−1.27** |

An 8.2-logit swing. Not noise, so the bug checklist ran before any precision work.

**Instrument note (one wasted hour).** The first probe read `final_logits` with
`max_gen_tokens=1` and concluded "the fork prefers `CF`" while free generation
emitted `250`. `final_logits` is published *after* the decode loop, so with one
generated token it belongs to the position **after** the decision. The decision is
`gen_tokens[0]`; the probe now reports that. A probe that disagrees with the gate
is a broken probe until proven otherwise.

## Step 2 — bug checklist

- tied head applies the table's `global_scale` (all answer constants correct — a
  dropped scale would not leave them right);
- `Free()` releases every plane, including the new one;
- written NVFP4 records decoded back against the BF16 checkpoint:
  **relL2 0.094–0.095** on qkv / o / gate_up / down, at layers 0, 20 and 39 —
  a healthy 4-bit body, so the codec is not the problem.

## Step 4 — placement measured, not assumed

Four trees converted from the same checkpoint, each probed at the fork
(margin = logit(`CF`) − logit(`250`), source +6.88):

| placement | Δ on disk | Δ resident | margin | FSI-01 |
|---|---|---|---|---|
| none | — | — | −1.27 | red |
| **attn** (qkv + o) | +0.9 GB | +0.94 GB | **+1.58** | **green** ← shipped |
| mlp (gate_up + down) | +3.3 GB | +3.4 GB | +2.19 | green |
| attn + mlp | +4.2 GB | +4.3 GB | +7.14 | green |

`attn` is the cheapest set that clears the gate and the one that leaves the
biggest tensors at 4 bits (the MLP is 79% of the layer's bytes), so both halves of
the rule point the same way. `attn+mlp` reproduces the source margin most closely
(+7.14 vs +6.88) for 3.3 GB more on an already-green gate.

Selection is a convert-time env (`LSX_GSW_RESIDUAL=attn|mlp|attn+mlp|none`), not a
runtime flag: the placement is baked into the artifact and recorded in the index as
`"residual": "attn"`, so an operator can tell two trees apart and the next member
can re-run the ablation without editing code.

## Step 5 — budget, stated

Device-resident 5.14 → **6.09 GB** (`6 088 178 432` bytes), tree 4.8 → **5.7 GB**,
VRAM profile min 5.51 → **6.45 GB** (the estimate was extended, so the governor is
not under-requested). The attention projections read twice the bytes per token;
they are 21% of the quantized weights. The decode-rate consequence is **not
measured** — the perf harness is Release-gated and the Release tree was not rebuilt
in the session that did this, which is recorded as a gap rather than estimated.

## Result

FSI 3/3, CODE 3/3, SelfCheck bit-exact ×3 (`b7c1aa453d83a86f`, re-measured
2026-11-07 as `930dd592ea5f5834` once the generation budget became a declared
per-family value; the budget was 256 here and is 2 canonically), adapter parity 3/3,
tokenizer parity, native tool round-trip green. DET 2/3 on DET-02, where the
**source itself** answers `CRITICAL` against the template's `CATASTROPHIC` — the
same cell the 30B records.

A negative result worth keeping: on the 4-bit-only tree the engine *passed* DET-02
by answering the template word. That was quantization error landing accidentally
on the expected answer, and it is the clearest local example of a green gate being
green for the wrong reason.

Legacy trees still load: the plane is optional in the loader (`res.d_packed ==
null` skips the accumulate pass), so the previous artifact decodes unchanged at
`ad9b0c4940bb27a9`.
