# Inkling-2-Small — model report

**Model:** `thinkingmachines/Inkling-2-Small` (594B total / 17B active) —
`inkling-2-small` in the logestix registry.
**Vault:** `/mnt/fast_storage/huggingface/Inkling-2-Small` (367 GB source,
366 GB converted artifact).
**Date:** 2026-11-15. **Branch:** `feat/inkling-2-small`.
**Hardware:** RTX 5090 (32 GB), 750 GB host RAM, Linux, CUDA 13.5 toolkit on a
13.4 driver.

## What the model is

A 48-layer hybrid-attention MoE with no RoPE anywhere:

| component | shape / rule |
|---|---|
| layers | 48 MoE decoders; 40 sliding-window (1024) + 8 full-attention at `(l+1)%6==0` |
| hidden | 4608; 48 Q heads / 8 KV heads at head_dim 128 (GQA group 6) |
| position | a **learned relative bank** of `d_rel` 16 profiles over 1024 backward distances; no rotation |
| attention | per-head learned **L2 clipping** (q 22.627, k weight-free, rel 8.0), a learned **sink** per head, and a sigmoid **output gate** |
| convolutions | four per layer: post-attention, post-MLP, and k/v with static taps + per-token dynamic taps (full layers only) |
| depth mixing | "AttnRes": a single-head softmax mixture over a stack of block prefix sums, block 6; three queries on full layers (separate q/k/v), one elsewhere, one `output_query` for the final norm |
| MoE | 384 routed experts, top-8, sigmoid scoring, selection-only FP32 bias, weights renormalised then ×8.0 |
| expert act | softcapped SwiGLU: `tanh(g/4)*4 * sigmoid(g) * tanh(u/25)*25`, FP32 accumulate |
| vocab | 201024 padded, 200058 scored; tiktoken BPE; logit softcap 30.0 |

## Precision: the release's own, byte for byte

The checkpoint is **mixed-precision and says so only in `hf_quant_config.json`**
(a converter that trusts `config.json` decodes a sixth of the stack as garbage):

| tensors | release | artifact | verification |
|---|---|---|---|
| experts, layers 0–39 (18 432 records) | NVFP4 (E2M1 + E4M3/16 + `amax/2688`) | same bytes | byte-identical, layers 0/20/39 |
| experts, layers 40–47 | MXFP8 (E4M3 + ue8m0/32) | same bytes; each ue8m0/32 duplicated to per-16 | byte-identical; `artifact == source` on the scale pairs |
| attention projections, router, embed/unembed | BF16 | BF16 | 996/996 pinned tensors exact |
| AttnRes queries, router bias | FP32 | FP32 | exact |

Nothing is requantized. Two consequences worth stating plainly:

- **Cost of NOT keeping the release's precision** — measured on the real weights
  with the same layer input and the same router, requantizing layers 40–47 to
  NVFP4 costs **16.0 / 16.5 / 16.9 % relative L2** on the MoE output of layers
  42 / 45 / 47. That is larger than the 9.5 % dense error that was already
  turning the model into a repetition loop.
- **Device-side widening** — dense BF16 weights are widened to FP16 on the
  device. That is not bit-exact: **0.33–0.40 %** of dense weights land in FP16
  subnormals and **0.0001–0.0005 %** underflow to zero. It is a representation
  effect, not a quantisation, and it is recorded here rather than glossed over.

## Correctness

### Gates

| gate | result |
|---|---|
| SelfCheck (Paris) | **PASS**, 3 cycles bit-identical, `logits_hash 323d30463a3d2b72` |
| FSI | see § Gates below |
| CODE | see § Gates below |
| DET | see § Gates below |
| native tool round-trip | **PASS** (`test_inkling_moe_tool_framing`, 49 framing assertions) |
| perf row @ 2048 | see § Throughput |

### The defect this ingestion is worth remembering for

The converted model answered `" the"` where the release says `" Paris"`, at
perplexity **12 881**. Every check the tree owned was green: the bytes were the
release's, the planes had the right sizes, the magnitudes were right, the
per-tensor global scale reproduced the recorded `amax` exactly, and **both**
independent references (a hand-written float64 one and the RELEASE's own
`transformers` modelling code driven on the real weights) agreed with the engine
— because all of them read the same artifact planes.

The cause: `mlp.experts.w13_weight` — and `w13_weight_scale` — is stored
**row-interleaved** (`[gate₀, up₀, gate₁, up₁, …]`), and the release de-interleaves
it (`Interleave(dim=1)` in `transformers`' `conversion_mapping.py`, entry
`"inkling2_text"`) before splitting gate from up. The converter copied the slab in
halves, so all 18 432 experts met the wrong up rows and the wrong block scales.

The measurement that broke the symmetry was the model's own output:

| measurement | split-half | de-interleaved |
|---|---|---|
| mean NLL, "The capital of France is Paris" | 9.46 → ppl **12 881** | 3.41 → ppl **30.4** |
| top-1 after "The capital of France is" | `" the"` | **`" Paris"`** |
| SelfCheck first ids | `" 0.9.9.0.0.9.0.0…"` | `<\|message_model\|><\|content_thinking\|>` |

It also retired the standing hypothesis that "the template adds no generation
prompt": given working experts, the model writes its own `<|message_model|>`
marker. A quantitative knowledge gate now guards the class
(`inkling real-weight sanity`: one prefill per position, mean NLL < log(100) and
the exact top-1).

Full trail, including the seven engine defects that preceded it and the
KV-growth over-read the new gate exposed: `docs/open-tasks/2026-11-14-inkling-2-small-ingest.md`.

## Throughput

See `tests/perf/perf_baselines.json` for the seeded row. The read per token for
the 17B-active set is dominated by the routed experts: 8 experts × 48 layers of
NVFP4 (0.5 B/elem) for layers 0–39 and E4M3 (1.0 B/elem) for 40–47.

## Known traps for the next person

1. **`w13_weight` / `w13_weight_scale` are row-interleaved** — de-interleave
   before splitting. Splitting in half is invisible to every byte check.
2. **The expert codec is mixed** (NVFP4 < 40, MXFP8 ≥ 40) and only
   `hf_quant_config.json` says so.
3. **`attn_res.attn_queries.i` belongs to layer `i+1`** (layer 0 has no
   attention mixture); `mlp_queries.i` is by layer.
4. **`mtp.safetensors` is a separate 3.4 GB file the index does not list** — 5
   MTP layers, stripped and counted, never applied.
5. **EOS is `<|content_model_end_sampling|>` (200006)**, not `<|endoftext|>`
   (199999); both stop. There is no BOS and no pad id.
6. **`transformers.models.inkling` is a different architecture** from
   `transformers.models.inkling2` (no sinks, no L2 clip, `scale = 1/head_dim`, no
   dynamic convolutions, no AttnRes). Copying from it produces plausible garbage.
