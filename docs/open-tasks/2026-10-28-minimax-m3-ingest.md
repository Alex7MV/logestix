# MiniMax-M3 ingestion — session trail (2026-10-27/28)

Branch `feat/ingest-minimax-m3`. First full ingestion of a 400B-class model on
this box: 796 GB BF16 source, 428B parameters, 60 layers, MiniMax Sparse
Attention. All eight Model Ingestion Protocol stages landed with real-weight
verification.

## Stage 0 — triage

| item | value |
|---|---|
| architecture | `MiniMaxM3SparseForCausalLM` (`model_type minimax_m3_vl`) |
| dense | 6144 hidden, 60 layers, 64 Q / 4 KV heads, head_dim 128, rotary_dim 64 (partial 0.5), theta 5e6 |
| MLP | layers 0-2 dense (I=12288); 57 MoE layers, 128 experts top-4, I=3072, shared I=3072 (ungated), routed_scale 2.0 |
| attention | MSA Lightning Indexer (4 heads x 128 dims, block 128, top-16 + 1 local block), layers >= 3 |
| norms | Gemma form (1+w), per head for q/k and for the indexer projections |
| activation | SwiGLU-OAI: `(up+1)*clamp(gate,max=7)*sigmoid(1.702*clamp(gate,max=7))` |
| router | sigmoid + `e_score_correction_bias`, top-4 by (sigmoid+bias), weights renormalized from the plain sigmoid scores (F32 weights in the checkpoint) |
| tokenizer | byte-level BPE, Llama-3/o200k pre-tokenizer, 61 added control tokens, bos `]~b]` 200019, eos `[e~[` 200020 |
| ballast | vision tower + projector + patch-merge (523 tensors, counted); no `mtp.*` tensors in the tree |
| feasibility | NVFP4 dense ~6.6 GB + FP16 embed/head 4.9 GB + routed experts 217 GB (mmap streamed) |

## Numeric decision: the head must not be 4-bit

The first conversion quantized everything, including `embed_tokens`/`lm_head`.
The engine then produced token salad while the per-layer hidden states looked
plausible. The host oracle showed the reference's first-token margin is
**0.4 logits** at a logit scale of ~38 — and a 4-bit head row (relL2 0.094,
measured by decoding the written record) perturbs every logit by ~3. With the
head and the embedding at FP16 the engine reproduced the reference's top-5
**ids and order** (`758:38.47, 44890:37.44, 705:32.52` vs `758:37.68,
44890:37.27, 705:32.59`) and answered "The capital of France is **Paris**.".
Everything else (attention, dense MLP, shared experts, routed experts) stays
NVFP4; norms FP16; router F32.

## Defects found (each pinned by a test)

1. `AddInPlaceKernel<<<1, 256>>>` for a 6144-wide residual: only 256 dims
   accumulated. Symptom: garbage text at plausible magnitudes.
2. `cudaEventElapsedTime` on an unrecorded ttft event → sticky
   `cudaErrorInvalidResourceHandle` that surfaced in the NEXT request.
3. Abort polled only after `Emit`: a pre-set abort emitted one token.
4. Tokenizer: `base_vocab_size` included the added tokens, so `literal_ids` was
   empty and every control marker was BPE'd as raw bytes; and the legacy GPT-2
   3-split pre-tokenizer split CJK runs. New `tokenizer_kind = 3`
   (`Llama3WordStream`, exact Lu/Ll/Lt case runs, merge-RANK BPE — HF ids are
   not merge-ordered) + `TokenizerSpecialPolicy.pretokenizer`; parity is now
   exact on a 14-case adversarial corpus captured from the checkpoint.

## Oracles (what proved the model is the model)

- `tools/minimax_hf_layer_oracle.py` — the checkpoint's OWN transformers code
  for one layer on the meta device, weights loaded per layer. Layer 0 agrees
  with the engine to NVFP4 error.
- `tools/minimax_reference.py` — host re-implementation over the source
  safetensors (dense graph + the routed experts actually selected, ~25 GB RAM).
  Per-layer cos/relL2 vs the engine: 0.9998/0.034 (L0) → 0.965-0.98/0.21-0.27
  (L40-59). The residual stream grows ~50x across 60 layers, which is what
  amplifies 4-bit body error; the greedy continuation matches the engine's
  text token for token.
- `LSX_MINIMAX_DUMP_LAYERS` / `LSX_MINIMAX_DUMP_FINAL` (diagnostics only) and
  `LSX_MINIMAX_DEBUG` (prompt ids + top-5) are what made the bisection
  mechanical instead of speculative.

## Verdicts (real weights, FastMathMode kOff for the parity gates)

- SelfCheck PASS ×3, `logits_hash 579a4a12cdedc279` stable.
- CODE 3/3, DET 3/3 (3 runs, byte-identical tokens), FSI 2/3 (FSI-01 red:
  the model answers the annuity-**due** form
  `PV=CF*(1-exp(-r*n))/(1-exp(-r))*exp(r)` instead of the template
  `PV=CF*((1-exp(-r*n))/(1-exp(-r)))`).
- Goldens are template-derived (the 428B BF16 source has no 32 GB execution
  path); recorded in each golden's `source` field, same status as
  DeepSeek-V4.1/Pro.

## Perf

nsys on the first working build: the serial router kernel was 80% of GPU
kernel time (1.14 ms × 57 layers/token) — rewritten to one warp per expert
(no per-expert `__syncthreads`), which took a selfcheck cycle from 2.5 min to
1 min. The remaining cost is expert staging: 228 expert stages/token
(6.9 GB) through pageable H2D (78% of CUDA API time). Rows seeded by the
harness at the canonical 2048-token protocol; see
`tests/perf/perf_baselines.json` and `docs/OPEN_TASKS.md` §R4.

Known limits, deliberate: the MSA block-selection path engages only past
`topk_blocks*block_size` = 2048 cached tokens; the canonical protocol therefore
runs on the neutral coalesced attention kernel, and past 2048 the two-pass
masked kernel is correct but not yet sparse-fast.
