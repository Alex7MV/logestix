# Hy4-flash (`hy4-flash`) — open items

Ingested 2026-11-23/24 on branch `feat/hy4-flash-ingestion`.
Design: `docs/superpowers/specs/2026-11-23-hy4-flash-ingestion-design.md`.
Plan: `docs/superpowers/plans/2026-11-23-hy4-flash-ingestion.md`.
Report: `docs/model-reports/2026-11-24-Hy4-flash.md`.
Gate verdicts: `docs/SELFCHECK_MATRIX.md` (row `hy4-flash`).

## `H1` — RESOLVED: the iHC post term multiplied the pre-norm MoE sum

**Root cause.** `RunHy4Inference` writes the sublayer output into `scratch.agg`
before `RunIhcPost` reads it, and the two FFN branches left the value in
different buffers:

* dense layer (layer 0): `ffn_out` was normed, then cast **into** `agg` — correct;
* MoE layer (layers 1–41): `agg` still held the routed+shared sum produced by
  `RunAccumulateScaled`, i.e. the **pre-norm** value, while the normed output sat
  in `ffn_out`. The cast that refreshes `agg` was guarded by `if (!is_moe)`.

So on 41 of 42 layers the iHC post gain multiplied the un-normalised sum. That
sum's rms is ~240x below the normed output (0.0027 vs 0.66 at layer 0), so the
MLP contribution to the residual was effectively dropped while attention kept
working. The result was fluent, plausible-looking text with the wrong
distribution — which is exactly why it read as "numerics" for so long.

The sandwich norm is the tell: `post` multiplies the sublayer output *after its
norm*, in the reference literally
`post * post_mlp_layernorm(mlp(x)) + residual`.

**The measurement that found it** was not a comparison against the reference: it
was a *self-consistency* check of the engine against its own dumps. Every layer
satisfies `out == prev + post_attn ⊗ attn_out + post_ffn ⊗ mlp_out`; the engine
violated it by **27.5% at layer 1** while the reference satisfied it to 0.21%.
Reconstruction from a model's own intermediates needs no second implementation to
be trusted, and it localised the layer in one cheap run.

**Verification after the fix.**

| check | before | after |
|---|---|---|
| residual reconstruction, all 41 MoE layers | 0.2747 at layer 1 | **0.000000** |
| `l0_out` — the DENSE layer, never affected (control) | 0.999875 | 0.999875 |
| `l1_out` — the first MoE layer | 0.9647 | **0.999623** |
| … `l21_out` / `l30_out` / `l41_out` / `logits` | not comparable (the instrument bugs below) | 0.994 / 0.991 / 0.969 / 0.985 |
| first token, raw `The capital of France is` | `259` (` a`) | `15641` (` Paris`) |
| first token, reference on the artifact's weights | — | `15641` (same) |
| SelfCheck (Paris) | red | PASS x3, `923e8ab7c8027eb2` |

The engine and the reference, both fed the artifact's own NVFP4 weights, now
agree on the first token exactly (`15641`). What remains is the engine's FP16
activations against the reference's bf16-with-fp32-interiors: 3% relative L2 at
layer 1 growing to 25% at layer 41 and 17% on the logits, without moving the
argmax of that step. (Only the ONE step was compared against the reference — a
multi-token comparison would need a cached decode in the reference harness, which
it does not have; SelfCheck x3 is what covers repeatability.) That is the
tolerance a parity assertion for this family has to be written against.

**Two further defects the fix exposed.**

* `ParseHy4ToolCalls` decoded into `n_ids * 8 + 64` bytes -- 176 for a 14-token
  call whose text is 198 bytes. On overflow the decode reports "buffer too small"
  and the function returned `false`, i.e. "the model emitted no tool call", for a
  perfectly well-formed call. Now uses `runtime::DecodeGeneratedText`, which owns
  the grow-and-retry loop.
* `Scratch::Free` freed the arena and then **nulled `d_slots` without freeing
  it**. That is the 25 GB expert device-slot cache, so every `Unload()` leaked the
  whole card: a second model in the same process failed with
  `scratch allocation failed` and 965 MiB free. That is what made the SelfCheck
  binary's later cases fail in cascade, and it would bite `lsxfabric --serve` on
  any model swap.

**Two instrument defects, recorded so they are not chased again.**

* The reference harness loaded `named_parameters()` only. The router's
  `e_score_correction_bias` is an `nn.Buffer` (|max| up to 0.075 on the real
  checkpoint) and it moves which 16 of 512 experts are selected, so every MoE
  layer was routing with a zero bias. Fixed to parameters + persistent buffers,
  keyed by `state_dict()`. **This produced a false 98% divergence at layer 1's MoE
  output that pointed at quantization for a while.**
* The engine's `l%d_mlp_raw` and the reference's `blk.mlp` hook are different
  quantities on MoE layers (`post_mlp_layernorm` is `Identity` there, the norm is
  inside the MoE block), so comparing them reported a divergence that does not
  exist. Compare `mlp_out` with `mlp_out`.

**A hypothesis that was measured and refuted.** "The artifact's NVFP4 precision
is too coarse for this model" was plausible and is wrong as the primary cause:
fed the artifact's own weights, engine and reference agree at cos 0.9999–0.9990
for the first 21 layers and never below 0.969 (0.985 on the logits, same argmax),
while the quantization error is uniform (0.094 rms, cos 0.9955) across all ten
layer-0 matrices regardless of shape — the format's theoretical error, not a
converter defect. It was the pre-norm/post-norm slip that destroyed the model.

## `H2` — FSI is 2/3: the model numericises where the golden is symbolic

`tests/fixtures/fsi/hy4-flash.json` is captured; FSI-TEST-01 and FSI-TEST-03
pass, FSI-TEST-02 fails:

```
expected  Balance_Alpha=(((100000000*Rate_A)-Slip_A)*Rate_B-Slip_B)*Rate_C-Slip_C
engine    Balance_Alpha=(((100000000*1.083541)-0.0005)*83.152402-0.0005)*0.011104-0.0005
```

The prompt supplies `Rate_A=1.083541; Rate_B=83.152402; Rate_C=0.011104;
Slip_*=0.0005` and asks for the equation "using the provided variable names", so
this is instruction-following, not arithmetic: the substituted expression is
structurally and numerically the expected one. **Open: is that a property of the
model or of the `no_think` frame?** A capture at `reasoning_effort=high` would
answer it, and is unmeasured (one reference step is ~6 min at this prefix length).

## `H3` — CODE and DET are green (template goldens)

`tests/fixtures/{code,det}/hy4-flash.json` are captured and both gates pass 3/3.
`auto_path_determinism` and `cooperative_cancel` are declared and exercised.

The three capture tools (`tools/{fsi,code,det}_capture.py`) gained a
`template_tokenize` override: they used to render the frame to *text* and
re-tokenise it, which is not equivalent when the frame is carried by **special**
tokens — hy4's `<｜hy_start:opensource｜>` markers are added tokens, and the text
route re-applies `add_special_tokens` and prepends a second start marker the
served frame never has. With the override the ids come from
`apply_chat_template(tokenize=True)` and are pinned equal to
`Hy4FlashModel::BuildPromptInputIds` (verified byte-identical on a framed probe).

## `H4` — perf row: 6.02 t/s, PCIe-bound

`test_perf_baseline_hy4_flash` runs the canonical 2048-token protocol on the
Release/AOCC tree and seeds both rows (`perf` = kAuto, `r3` = kOff). The first
attempt seeded **0.0 t/s** for a run that really did decode 2048 tokens, because
`Hy4FlashModel::Infer` never filled `tokens_per_sec`/`ttft_ms` (it calls the
family driver directly instead of going through `runtime::FillInferenceResult`).
Fixed by timing the call and stamping the first emitted token through a wrapped
callback; the zero rows were dropped from the JSON rather than left as a
"baseline" nothing can ever beat.

Measured: **6.02 t/s** (`perf`/kAuto), TTFT 5.1 s, 6661 expert slots (24.7 GiB
resident, 31.7% of the 20 992 experts). The decode reads 16 experts x 41 layers x
3.8 MiB = 2.43 GiB of weight per token, so 6 t/s is ~15 GiB/s of host-to-device
traffic: the model is PCIe-bound, not compute-bound. **Open follow-up with the
clearest payoff: expert prefetch.** The selection is known one layer ahead, so
overlapping the next layer's staging with this layer's GEMVs is the lever that
does not need more VRAM; a *larger* slot cache would also cut the traffic but is
bounded by the same 32 GiB (6661 of 20 992 experts fit today).

## `H5` — the balloon the artifact is NOT carrying

Counted in the index `deferred` block (`engram=164 vision=441 dfly=87`):

* **Engram** — 126.7 GB of n-gram tables on layers 1/11/21/31. The reference DOES
  implement it (hash multipliers from a seeded RNG, `draw_hash_multipliers`), so
  it is not unknowable as stage 0 first concluded; it is cut because it does not
  fit beside the experts on this card and the engine has no lookup path. It is cut
  on both sides, so it cannot explain any divergence.
* **ViT tower** + projector, and the **5-layer DFly2 drafter** — stored-not-applied
  by policy.

## `H6` — `config.json` trap worth reporting upstream

`engram_layer_ids` is **1-based** (`[2, 12, 22, 32]`) while the engram tensors and
the model card put those layers at **1, 11, 21, 31**; the reference compensates
with `layer_idx + 1 in config.engram_layer_ids`. A converter that trusts the list
verbatim cuts the wrong layers. The converter takes the set from the frozen
passport, which follows the tensors, and fails the conversion when they disagree.
The constants test pins the off-by-one so nobody "fixes" the passport to match.

## `H7` — RESOLVED with `H1`'s third defect

The expert device-slot cache leak (see `H1`) was what made a failed case in a
shared-driver binary poison every later case in the same process. The
`scratch allocation failed` message now carries the numbers that identify it
(`free=965 MiB, reserve=1024 MiB, one slot=3888 KiB -> 0 slots`), instead of
being indistinguishable from a genuine OOM.

## Reproduction (cheapest order)

```bash
# 1. build the instruments
cmake --build --preset linux-debug -j 64 --target test_hy_moe_flash_model

# 2. the engine's per-boundary dump (raw ids of "The capital of France is")
mkdir -p /mnt/fast_storage/tmp/hy4-eng-dump
LSX_HY4_DUMP_DIR=/mnt/fast_storage/tmp/hy4-eng-dump \
  ./build/linux-debug/tests/test_hy_moe_flash_model "[.hy4diag]"

# 3. a 1-layer reference probe (9 s) or the full stack (13 min)
/mnt/fast_storage/tmp/hy4venv2/bin/python tools/hy4_flash_reference.py \
  --ids 802,8778,299,12749,341 --layers 1 --dump /tmp/hy4-ref-l0.npz

# 4. the same reference on the ARTIFACT's weights (~40 min: 20992 experts)
/mnt/fast_storage/tmp/hy4venv2/bin/python tools/hy4_flash_reference.py \
  --ids 802,8778,299,12749,341 --artifact /mnt/fast_storage/huggingface/Hy4-flash \
  --dump /tmp/hy4-ref-full-art.npz

# 5. boundary-by-boundary verdict
/mnt/fast_storage/tmp/hy4venv2/bin/python tools/hy4_flash_compare.py \
  --engine /mnt/fast_storage/tmp/hy4-eng-dump --reference /tmp/hy4-ref-full-art.npz

# 6. task-level quality probes through the engine
LSX_HY4_DIAG_PROMPT="2+2=" LSX_HY4_DIAG_MAXGEN=24 LSX_HY4_DIAG_FRAME=1 \
  ./build/linux-debug/tests/test_hy_moe_flash_model "[.hy4diag]"
```

The self-consistency check that found `H1` needs only step 2: read the dumps and
verify `l<n>_out == l<n-1>_out + post_attn ⊗ l<n>_attn_out + post_ffn ⊗
l<n>_mlp_out` with numpy. Do that first on any family whose text is fluent but
wrong.
