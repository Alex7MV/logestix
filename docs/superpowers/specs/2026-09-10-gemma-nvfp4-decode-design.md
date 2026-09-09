# Gemma-4-26B-A4B decode acceleration: full NVFP4 dense + device no-handshake routing

Date: 2026-09-10
Status: approved (operator), design phase complete
Goal: raise gemma-4-26B-A4B-it decode from ~38.7 t/s toward Qwen3.6-35B-A3B-class
throughput (~103 t/s) on RTX 5090 by eliminating BF16 weight traffic and host
synchronization from the decode path.

## Background

Current gemma decode traffic per token (~5.5 GB): dense MLP 1.07 GB BF16,
attention projections ~2.2 GB BF16, tied embed/lm_head 1.48 GB BF16, routed
experts 0.71 GB NVFP4. Additionally the router handshake (busy-poll
`ev_router` + D2H `h_sel` copy) stalls the pipeline once per layer per token
(~30 sync points). Qwen3.6 by contrast keeps all dense weights NVFP4 and
composes the routed-GEMV slot table entirely on device.

Constraints / invariants:
- Format stays `logestix-gemma`; loader must keep loading existing BF16
  conversions (no forced re-conversion).
- Norms, `layer_scalar`, router weights/scales, q/k_norm stay BF16/F32
  (precision invariant).
- Determinism: all reduce orders fixed (ascending-k, ascending chunk);
  selfcheck must remain bit-exact across cycles.
- NVFP4 GEMM/GEMV goes through the model-neutral `lsx_inference_nvfp4.h`
  layer or the existing gemma routed GEMV chain (`nvfp4_gemv_math.cuh`
  contract); no parallel dispatchers.

## Phase 1 — Dense MLP → NVFP4

Converter (`gemma_moe_convert.cpp`):
- `mlp.gate_proj/up_proj/down_proj` written into `pinned/layer_N.arrow`
  as NVFP4 payloads: `data` column carries E2M1 nibble-packed weights
  (2 elements/byte), `scales`-suffixed companion tensors carry the per-16
  E8M0-style scale bytes; dtype column stays "BF16" (source dtype).
- Quantization through the existing `gpu::quantize_tensor_gpu` path
  (`PackExpertNVFP4`-equivalent), one global_scale per tensor stored as a
  separate tensor record `mlp.<proj>.global_scale` (F32 scalar).
- Index gains `"mlp_nvfp4": true`.

Loader (`gemma_moe_load.cpp`):
- `GemmaLayerWeights` gains `LsxNvfp4Weight mlp_gate/mlp_up/mlp_down`
  (packed + scales + global_scale, view-arena backed like the routed experts).
- Detection: tensor record whose packed byte length * 2 == numel (and a
  matching `.global_scale` record) loads as NVFP4; otherwise legacy BF16.
  Same `.arrow` file layout, no format bump.

Inference (`gemma_moe_inference.cu` ForwardLayer dense-MLP branch):
- Decode (seq==1): activations `d_normed` BF16→FP16 (existing conversion
  kernel), then 3× NVFP4 weight-streaming GEMV (gate/up/down) with geGLU
  applied in FP16 between up and down; FP32 accumulator; result cast to
  BF16 into the residual chain (same contract class as the routed GEMV
  chain).
- Prefill (seq>1): same GEMV cores row-tiled, or dequant-to-BF16 batch path
  — whichever the shape dispatch in `RunLsxNvfp4Gemm` selects; correctness
  gate is the selfcheck, not the specific core.
- env `LSX_GEMMA_MLP_BF16=1` restores the legacy cuBLAS BF16 branch.

## Phase 2 — Attention projections + tied embed/lm_head → NVFP4

Converter:
- `self_attn.q/k/v/o_proj` NVFP4-packed the same way into
  `pinned/layer_N.arrow` (+ per-tensor global_scale records). Full-attention
  layers (o_cols=8192, kv rows differ) handled by the same code — shapes come
  from the record.
- `embed_tokens.weight` [V, 2816] written to `pinned/global_nvfp4.arrow`
  using the existing 6-column `DenseNvfp4Schema` (packed/scales/global_scale).
  Index gains `"attn_nvfp4": true`, `"head_nvfp4": true`.

Loader:
- Attention `LsxNvfp4Weight attn_q/attn_k/attn_v/attn_o` per layer.
- `GemmaPinnedWeights` gains `embed_nvfp4` (and `head_nvfp4` aliasing the
  same packed buffer — tie_word_embeddings). The BF16 `d_embed` is no longer
  uploaded when head_nvfp4 is present (saves 1.48 GB VRAM and its load time).

Inference:
- Attention projections via NVFP4 GEMV; QK-norm, RoPE, KV cache, attention
  dot products stay BF16 (cache writes unchanged).
- Token embedding: small gather kernel dequantizes row `tok` of the NVFP4
  table on the fly (2816 elements) — replaces the `d_embed` row copy.
- LM head: `RunGemmaLogitsArgmax` rows kernel extended to stream NVFP4 rows
  (decode E2M1/E4M3 inline per 32-element chunk, same math contract as
  GemvDotRow); softcapping unchanged (applied after the dot).
- env `LSX_GEMMA_ATTN_BF16=1`, `LSX_GEMMA_HEAD_BF16=1` legacy rollbacks.

## Phase 3 — Device no-handshake routing

- At cache setup, upload a device mirror `flat_expert_meta` [total_experts]:
  `{int32 dev_slot; float gs1, gs3, gs2}`. Expert geometry is uniform
  (verified: all expert files identical size), so w1/s1/w2/... offsets are
  compile-time constants per model — no per-expert offsets in the table.
- A gather kernel composes `QwenMoeGvSlotDev tbl[K]` from `d_sel`/`d_selw`
  + the mirror (qwen `GemvGatherTable` pattern), then `RunRoutedGemvBatchSeg`
  runs without any host readback: `ev_router` busy-poll and `h_sel` D2H are
  eliminated for the MoE block.
- LRU-miss handling: fully-resident cache (3840/3840 slots on 5090) has no
  evictions, so miss-staging is unreachable in the steady state. When the
  cache is NOT fully resident (smaller GPUs / `LSX_GEMMA_EXPERT_CACHE_SLOTS`),
  the loader keeps the existing handshake path as fallback (runtime branch
  on `slots == total_experts`, computed once per generation).
- Bit-exactness: expert selection itself stays on device (the router kernel
  already computes it there); the gather is a pure permutation of table
  entries — no float reordering. Ascending-k reduce contract unchanged.

## Testing

- Converter/loader tests (`test_gemma_moe_convert`, `test_gemma_moe_loader`):
  synthetic fixture gains NVFP4 MLP/attn records + `global_nvfp4.arrow`;
  round-trip packed↔host-reference dequant; loader strictness for missing
  global_scale records; legacy-BF16 fixture still loads (compat matrix).
- Kernel tests (`test_gemma_moe_moe`, `test_qwen_nvfp4_gemv` gemma cases):
  dense-MLP GEMV chain and attention-projection GEMV parity vs host
  reference at gemma geometry (H=2816, I=2112, sliding+full shapes) —
  covering the KP>1 partition that was previously untested.
- Selfcheck (`test_gemma_moe_selfcheck`): Paris, 3 bit-exact cycles, real
  model, for each phase landing.
- Performance gate: ≥ 70 t/s (target ~2×) on `--testprompt --max-gen 64`;
  env rollbacks must restore previous behavior.

## Rollout order

Phase 1 → measure → Phase 2 → measure → Phase 3 → measure. Each phase is a
separate commit; the selfcheck is the merge gate for each.
