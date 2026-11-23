# Hy4-flash (`hy4-flash`) — open items

Ingested 2026-11-23/24 on branch `feat/hy4-flash-ingestion`.
Design: `docs/superpowers/specs/2026-11-23-hy4-flash-ingestion-design.md`.
Plan: `docs/superpowers/plans/2026-11-23-hy4-flash-ingestion.md`.
Report: `docs/model-reports/2026-11-24-Hy4-flash.md`.

## `H1` — the engine diverges from the reference on the real weights

**Status: open. This is the item that keeps the gates red, and everything below
follows from it.**

The converted artifact loads and decodes coherent English, and every host-level
gate is green (converter, tokenizer parity, loader, model lifecycle, kernels,
tool framing). But the decode does not reproduce the reference:

| probe | engine (kOff) | reference (PR #50903) |
|---|---|---|
| raw prompt `The capital of France is`, first token | `259` (` a`) | `5586` (` __`) |
| framed prompt (chat template, `no_think`), first token | not reached (SelfCheck gate) | `40` (`I`) then `3283` (`'m`) |

Neither side answers "Paris" **on the raw prompt**, and that half is now
explained: this is a chat model whose template frames the turn, so an unframed
prompt is not the probe. The unframed comparison above is only useful because
both sides saw identical ids.

### What is already ruled out

Each of these was checked against `modeling_hy_v4_flash.py` line by line and
matches (the citations are in the report's evidence section):

* iHC form — `mixes = fn @ flat * rsqrt(mean(flat²)+eps)`, `pre`/`post` per
  stream, `magnitude 2.0`, **no Sinkhorn, no comb** (`HYV4FlashHyperConnection`);
* the head collapse `hc_head` (one `hc_scale` value, `pre` only, no post term);
* sandwich norms on the SUBLAYER OUTPUT (`post_proj_norm`,
  `post_fc2_norm`, `combine_output_norm`);
* LatentMoE order and the shared expert reading the full-width input;
* the router (fp32 sigmoid, selection-only bias, renorm, ×3.998);
* the gated MLA (`sigmoid(linear_gate)` before `o_proj`) and the drop-the-sink
  softmax;
* RoPE: **interleaved** pairs, `theta` 1e7 on sparse layers and 1e4 on sliding
  ones, over the trailing 64 lanes;
* attention scale `256^-0.5` with **no** mscale;
* `kv_b` split into per-head `[nope|value]` blocks;
* the sparse layers attending densely below `index_topk`;
* rms eps `1e-5`;
* Engram cut on both sides.

### The two candidate causes, in order of likelihood

1. **A latent-space index / ordering slip** that leaves the shapes legal. The
   transposed qk projection (`qk[h][d] = kv_b_nope[h*nope+r][d] · q[h*qk_head+r]`)
   was wrong exactly this way on the first attempt (it used the nope count as the
   q stride, so every head after the first read the wrong lanes; the model still
   emitted fluent text). The same class of mistake is the prime suspect for what
   remains.
2. **FP16 vs the reference's BF16/FP32 activations.** The engine carries
   activations in FP16 while the reference computes attention scores in the
   activations' dtype and `head_weights` in fp32. That is a tolerance-class
   difference and cannot move top-1 from `' a'` to `' __'` on its own — but it
   will matter once the semantic bug is fixed, and it bounds how tight a parity
   assertion can be.

### How to close it

The reference implementation is pinned and reproducible:

```bash
python3 -m venv --system-site-packages /mnt/fast_storage/tmp/hy4venv2
/mnt/fast_storage/tmp/hy4venv2/bin/pip install --no-deps \
    "git+https://github.com/huggingface/transformers.git@refs/pull/50903/head"
/mnt/fast_storage/tmp/hy4venv2/bin/pip install "tokenizers>=0.23.1,<0.24.0"
/mnt/fast_storage/tmp/hy4venv2/bin/python tools/hy4_flash_reference.py \
    --ids <the same ids> --tokens 1
```

Two facts about that run, both learned the hard way and worth not rediscovering:

* the checkpoint's leaf names are **not** the reference's module names; the PR's
  own `conversion_mapping.py` `hy_v4_flash` list must be applied or 422 parameters
  silently keep their random initialisation and the reference answers garbage
  (`tools/hy4_flash_reference.py::RENAMES`);
* it must run in bf16 (fp32 is 310 GB and the commit limit here is 402 GB) and
  the weights must be copied in one tensor at a time (a whole `state_dict`, even
  with `assign=True`, is OOM-killed).

A per-layer bisect is the way in: run the reference with `output_hidden_states`
and compare the stream norms/means layer by layer against a debug hook in
`RunHy4Inference`. The first layer that diverges names the component; the layer
0–1 (dense FFN) vs layer 2+ (MoE) split localises attention-vs-FFN for free.

### Cost, measured

One reference step at a 5-token prefix costs **13 min** wall (155B params bf16
streamed from a 750 GB host, 160 threads). The full-prefix feeding mode is
O(n²), so a 200-token golden capture is not affordable at that shape — a
`DynamicCache` path in the reference tool is a prerequisite for the gates, not a
nicety.

## `H2` — FSI / CODE / DET are declared debt (blocked on `H1`)

The three goldens must be captured from the reference. The capture tools
(`tools/{fsi,code,det}_capture.py`) need a `hy4-flash` override that calls
`tools/hy4_flash_reference.py`'s loader; that is deliberately not wired while the
engine does not match it, because a golden captured from a model the engine
disagrees with would only convert `H1` into three red gates with no new
information.

`docs/MODELS.md` and `models/hy4-flash.json` carry the debt with this reason.

## `H3` — SelfCheck (Paris) is red

`test_hy_moe_flash_selfcheck` fails on `selfcheck_passed`, not on a crash and not
on a missing tree: the model is loaded, the frame is built by the checkpoint's own
template, and the answer is wrong (that is `H1`). `selfcheck_max_gen` is 8 and
`selfcheck_frame_prompt` is true, both taken from measurement of the source.

## `H4` — no perf row yet

`test_perf_baseline_hy4_flash` exists and the `TEST_CASE`/`add_test` pairing is in
place (`test_perf_baseline_logic` enforces it in both directions), but it SKIPs:
there is no measured row. `/mnt/fast_storage/huggingface/Hy4-flash/logestix` is
83 GB on disk (78 GB experts + 4.8 GB pinned), the decode reads ~656 experts per
token (~2.6 GB) over PCIe, and no t/s has been measured yet. Measure on the
Release/AOCC tree at the canonical 2048-token protocol and seed the row by running
the target.

## `H5` — the balloon the artifact is NOT carrying

Recorded so the next reader does not re-derive them (all counted in the index
`deferred` block: `engram=164 vision=441 dfly=87`):

* **Engram** — 126.7 GB of n-gram tables on layers 1/11/21/31. The reference DOES
  implement it (hash multipliers are drawn from a seeded RNG, `draw_hash_multipliers`),
  so it is not unknowable as stage 0 first concluded; it is cut because it does not
  fit beside the experts on this card and because the engine has no lookup path.
  Applying it is the highest-value follow-up after `H1`.
* **ViT tower** + projector, and the **5-layer DFly2 drafter** — stored-not-applied
  by policy.

## `H6` — `config.json` trap worth reporting upstream

`engram_layer_ids` is **1-based** (`[2, 12, 22, 32]`) while the engram tensors and
the model card put those layers at **1, 11, 21, 31**; the reference works around it
with `layer_idx + 1 in config.engram_layer_ids`. A converter that trusts the list
verbatim cuts the wrong layers. The converter now (a) takes the set from the frozen
passport, which follows the tensors, and (b) fails the conversion when the tensors
disagree with it. The constants test pins the off-by-one so nobody "fixes" the
passport to match the field.
