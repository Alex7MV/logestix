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

## 0.6 Semantics verified (2026-11-21, against transformers 5.24.0 + real shards)

1. **Loop scheduling** — `Inkling2LoopTextModel.forward`: pass 0 runs layers
   [6,18) inline with the growing block stack; passes 1/2 restart from the
   pre-loop hidden + block snapshots; per loop layer, messages are retrieved
   by `message_input` (single-head softmax attention, unweighted RMSnorm on
   the stacked states, scale hidden**-0.5) over the PREVIOUS pass's block
   states **skipping the first appended state** (`num_pre_loop_blocks + 1:`)
   plus the previous pass's final hidden. Cache offset for pass p (1-based):
   `24 + (p-1)*12 - 6`, i.e. CacheSlot(layer, p) as in the passport.
2. **MessageGate** — `(1-tanh(forget(down)))*h + tanh(gain(down))*msg`;
   down input = concat(unweighted-RMSnorm(h), RMSnorm(msg)) cast to the down
   weight dtype (BF16), gates computed in FP32. Applies AFTER the AttnRes mix
   and BEFORE the input norm (q/k/v separately on loop passes ≥ 1); the mlp
   gate applies after the mlp_input mix, before post_attention_layernorm.
   On pass 0 the loop layers behave exactly like non-loop layers (messages
   None → single input norm).
3. **w13 row-interleave** — conversion table `inkling2_loop_experts` applies
   `Inkling2LoopDequantizeExperts()` then `Interleave(dim=1)` to w13 (weights
   AND scales via the same op chain): source rows alternate gate/up.
4. **Cross-pass query merge — VERIFIED NUMERICALLY** on the real tensors with
   the release's own `Inkling2LoopMergeMessageQueries`: merged (12,4,2304)
   order is [q, k, v, mlp] with q=queries[2i], mlp=queries[2i+1],
   k=kv_queries[2i], v=kv_queries[2i+1]; inverse round-trip byte-exact.
5. **NVFP4 invariant — VERIFIED** on layers 0/10/19, w13+w2:
   `6*max(block_scale_e)*amax_e/2688 == amax_e` exactly (rel err 0).
   MXFP8 layers 20–23: E4M3 bytes + ue8m0 scale per 32 (byte range 113–117 on
   L20 w13), decode `2**(scale-127)`.
6. **Attention** — scale head_dim**-0.5; sink = one extra logit per head from
   `sink_proj(hidden)`, zero value; output gated by sigmoid(gate_proj(hidden))
   computed in FP32; q: learned per-head scale THEN L2 clip (then log-scale on
   full layers); k: sconv FIRST then weight-free L2 clip; v: sconv then
   unweighted RMSnorm; rel states: wr_du → learned scale → clip 8.0 (→
   log-scale on full layers) → proj[d_rel, extent] → gather by distance,
   zero outside [0, extent). Log-scale factor
   `1 + 0.1*log(max((pos+1)/8192, 1))` — ACTIVE within this card's contexts.
   Static sconv normalizer sqrt(k+1), dynamic sqrt(2k+1); dynamic taps =
   static_w.T + weight_proj(hidden) per kv head.
7. **MoE** — router FP32 sigmoid, bias added for SELECTION only, weights
   renormalized (sum + 1e-20) then ×8.0; expert = softcapped SwiGLU
   (tanh(g/4)*4 · sigmoid(g) · tanh(u/25)*25) in FP32; gate_up rows after
   de-interleave: first half gate, second half up.
8. **Norms** — T5-style RMSnorm (no mean subtraction, no bias), eps 1e-6,
   weight applied in FP32 then cast back; embed_norm applied to embeddings.

## Source probe results (real weights, CPU bf16, release code)

`tools/inkling_loop_transformers.py` (transformers 5.24.0, eager attention,
64 threads, ~0.9 s/token):

- **Framed Paris probe** (native user turn, effort line "Thinking effort
  level: 0.9"): prompt ids
  `[200002,200004,34182,279,154554,150628,25,220,15,13,24,200010,200000,200004,371,151451,149879,151194,149884,200010]`;
  32-token greedy continuation opens
  `<|message_model|><|content_thinking|>The user asks: "The capital of France
  is". This is a simple factual question. The capital of France is Paris. I
  should answer` — ids in
  `/mnt/fast_storage/tmp/inkling2-mini/probes/paris.json` (+ last-step logits
  .npy). The stream opens with the model's OWN opener, as Small's does.
- **Raw knowledge probe** "The capital of France is" (ids
  `[371,151451,149879,151194,149884]`): top-1 continuation **" Paris"**
  (152717), then `.`, `\n\n` — `probes/paris_raw.json`.
- Probe decision for the engine SelfCheck: framed prompt, bit-exact mode;
  `selfcheck_max_gen` to be fixed from the engine's own decode budget at
  Task 9 (the source confirms "Paris" appears within the first 32 tokens).

## Trail

- 2026-11-21: stage 0 complete; design approved by operator; plan written;
  branch created; naming guard accepts the `loop` variant token (RED→GREEN).
- 2026-11-21: frozen passport + constants test green (Task 2); reference
  harness ran the release code on real weights — Paris confirmed both framed
  and raw; merge order + NVFP4 invariant verified numerically (Task 3).