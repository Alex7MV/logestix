# Inkling-2-Mini ingestion — stage 0 triage, plan and trail

**Status: IN PROGRESS (started 2026-11-21).** Branch
`feat/inkling-2-mini-ingest`. Design:
`docs/superpowers/specs/2026-11-21-inkling-2-mini-ingest-design.md`; plan:
`docs/superpowers/plans/2026-11-21-inkling-2-mini-ingest.md`.

Source checkpoint: `/mnt/fast_storage/huggingface/Inkling-2-Mini`
(`thinkingmachines/Inkling-2-Mini`, released 2026-11-20, downloaded 2026-11-21
via modelscope with sha256 verified against HF for both safetensors; see
`/mnt/fast_storage/tmp/opencode/inkling2-mini/manifest.json`). Single 18.2 GiB
shard + a 0.59 GiB `mtp.safetensors` the index does not list.

## 0.1 Classification

**New model in the existing `inkling` family — NOT a sibling of
Inkling-2-Small.** `model_type: inkling2_loop`, architecture
`Inkling2LoopForConditionalGeneration`: a looped transformer. Layers [6,18)
run 3 passes with shared weights (48 layer applications per token from 24
layers of weights), with rank-64 MessageGate controllers and cross-pass
message queries that Small does not have. Different geometry AND new
semantics → own frozen passport, class, converter, loader, kernels
(`inkling_moe_loop_*`).

Reference semantics: **transformers 5.24.0 `models/inkling2_loop`** (wheel
extracted at `/mnt/fast_storage/tmp/inkling2-mini/tf/x`; the installed 5.15.1
has neither inkling2 nor inkling2_loop). The conversion table
(`transformers/conversion_mapping.py`, entries `inkling2_loop` /
`inkling2_loop_text`) is the authority for tensor renames, the expert
de-interleave and the cross-pass query merge.

## 0.2 Geometry (config.json + safetensors headers + hf_quant_config.json)

| field | value |
|---|---|
| hidden_size | 2304 |
| layers | 24 physical; loop [6,18) × 3 passes → 48 applications |
| heads / kv / head_dim | 24 / 8 / 128 (GQA 3); SWA identical |
| vocab / unpadded | 201024 / 200058 |
| experts / top-k / shared | 128 / 8 / 0; intermediate 1152 |
| routing | sigmoid + selection-only FP32 bias; renorm → ×8.0 |
| expert softcaps | gate 4.0, up 25.0; final logit softcap 30.0 |
| hybrid | full at {5,11,17,23}; sliding window 1024 elsewhere |
| rel bank | d_rel 16, extent 1024, NO RoPE; q/k L2-clip 22.627417, rel clip 8.0; v RMS-norm no gamma |
| log scaling | n_floor **8192** (Small: 128000), alpha 0.1, full layers |
| sconv | k=4; static sliding / dynamic full (static_w + weight_proj[32,2304]) |
| AttnRes | block 6; full layers 3 queries; attn_queries.i → layer i+1 |
| loop | gate_rank 64, is_varloop false; per-pass KV+conv cache slots (48) |
| controllers | per loop layer × {attn,k,v,mlp}: down[64,4608] BF16, forget/gain[2304,64] (FP32 in reference) |
| cross-pass queries | two [24,2304] FP32 tensors → merged (12,4,2304) |
| expert codec | NVFP4 layers 0–19 (U8 + E4M3 per 16, amax/2688), MXFP8 20–23 (F8_E4M3 + ue8m0 per 32) |
| eos / pad | 200006 / 199999; no BOS |

## 0.3 Tokenizer

tiktoken BPE, kind 2. `tokenizer.json` and `tiktoken/tokenizer.model`
**byte-identical to Inkling-2-Small** (sha256
`7f332727…` / `f170372d…`). Chat template differs only in `{% generation %}`
markers (rendered text identical). Tool format semantics therefore carry over
from the verified Small codec, re-implemented in `inkling_moe_loop_tool_format`
and re-verified against Mini's own template.

## 0.4 Ballast (strip + count)

- vision tower: 422 tensors, 0.59 GiB
- MTP: `mtp.safetensors`, 54 tensors, 0.59 GiB, 3 layers, NOT in the index

## 0.5 Feasibility (RTX 5090 32 GiB, host 750 GB)

Artifact 18.2 GiB (dense 3.0 + NVFP4 9.6 + MXFP8 5.0 + vision 0.6). Whole
model fits in VRAM (~20 GiB with KV at 2048): expert cache effectively pins
everything → compute-bound. Full BF16 source (~53 GB) fits host RAM →
**source-derived FSI/CODE/DET goldens** via HF 5.24 on CPU (operator approved
the slower capture over Small's template-derived debt).

## 0.6 Semantics to verify (trail below)

1. Loop scheduling / per-pass cache_offset — read from
   `Inkling2LoopTextModel.forward`: passes 2/3 restart from pre-loop
   hidden+block states; messages retrieved per loop layer from the PREVIOUS
   pass's block stack (skipping the first appended state) + final hidden.
2. MessageGate: `(1-tanh(forget(down)))*h + tanh(gain(down))*msg`, down input
   = concat(unweighted-RMSnorm(h), RMSnorm(msg)); applies AFTER the AttnRes
   mix, BEFORE input norm; mlp gate after mlp_input mix.
3. w13 row-interleave — conversion table confirms `Interleave(dim=1)` for
   Mini (both weights and scales), same trap as Small.
4. Cross-pass query merge: `middle_cycle_cross_pass_queries` holds q+mlp,
   `..._kv_queries` holds k+v (conversion table comment); exact row order
   verified in Task 3 against `Inkling2LoopMergeMessageQueries`.
5. NVFP4 invariant `6*max(scale)*amax/2688 == amax`; MXFP8 ue8m0 per 32 —
   verified on real shards in Task 3.
6. Attention scale head_dim**-0.5, sinks, gated output, clip-after-sconv,
   log-scaling from 8192.

## Trail

- 2026-11-21: stage 0 complete; design approved by operator; plan written;
  branch created; naming guard accepts the `loop` variant token (RED→GREEN).