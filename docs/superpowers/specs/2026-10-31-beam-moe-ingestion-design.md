# Beam-1.0-501B-A23B ingestion — Stage 0 design

**Source:** `/mnt/fast_storage/huggingface/Beam-1.0-501B-A23B`
(`reflection/Beam-1.0-501B-A23B`, BF16, 934 GB, 188 shards, 38 366 tensors).

**Reference semantics:** HF `transformers/models/beam/{configuration_beam.py,
modeling_beam.py}` (main, 2026-10-31) and `sgl-project/sglang` `models/beam.py`.
Both were read before writing any kernel; the checkpoint carries no code.

## 0.1 Classification

New independent family `beam`, type `moe`, no variant. `model_type = "beam"`,
`architectures = ["BeamForCausalLM"]`. Geometry and the depth-scaled residual /
sandwich-norm / gated-attention combination exist in no ingested family.

## 0.2 Frozen geometry (config.json)

| field | value |
|---|---|
| vocab_size | 151 936 |
| hidden_size | 5120 |
| num_hidden_layers | 52 |
| num_attention_heads | 48 |
| num_key_value_heads | 8 (GQA 6:1) |
| head_dim | 128 |
| intermediate_size (dense) | 20 480 |
| moe_intermediate_size | 2560 |
| num_experts / top_k | 256 / 8 |
| n_group / topk_group | 8 / 4 (group-limited) |
| routed_scaling_factor | 2.5 (after aggregation) |
| norm_topk_prob | true (renormalize selected sigmoid scores) |
| sliding_window | 512 (current + 511 previous) |
| layer_types | `sliding_attention` except **full** at {5,11,17,23,29,35,41,47,51} |
| mlp_layer_types | dense 0-2, sparse (MoE) 3-51 → 49 MoE layers |
| rope | sliding layers only, rotate-half full 128 dims, theta 10000 |
| full attention | **NoPE** + SSMax query scale `(1 + (pos+1)/2048)^0.1` (FP32, applied to q) |
| rms_norm_eps | 1e-6 |
| final_logit_softcapping | 30.0 (`logits/30 → tanh → ×30`, F32) |
| tie_word_embeddings | false |

Tensors (verified against the index): `embed_tokens [151936,5120]`,
`lm_head [151936,5120]`, `model.norm [5120]`, per layer
`input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`,
`post_feedforward_layernorm` (all `[5120]`), `self_attn.{q,k,v,o}` +
`self_attn.gate_proj [6144,5120]` + `self_attn.gate_bias [48]` +
`self_attn.{q_norm,k_norm} [128]`; dense layers `mlp.{gate,up,down}_proj`;
MoE layers `mlp.gate.weight [256,5120] F32`,
`mlp.gate.e_score_correction_bias [256] F32`, and
`mlp.experts.E.{gate_proj,up_proj,down_proj}.weight [2560,5120]/[5120,2560]`.
**No shared expert, no MTP, no vision.**

## 0.3 Tokenizer

Byte-level BPE, base vocab 151 643, 38 added tokens. Pre-tokenizer regex is the
Qwen2/3 single-digit form
`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
— new `BeamWordStream`, `tokenizer_kind = 6`. The Harmony markers
`<|start|> <|end|> <|message|> <|channel|> <|constrain|> <|return|> <|call|>
<rfl:parameter name="` are **`special:false`** and must be matched literally
(ids ≥ base_vocab_size). `bos_token = null` → `bos_id = -1`;
`eos_token = <|return|> = 151674`; config `eos_token_id = [151674, 151675]`;
`pad = <|endoftext|> = 151643`. `chat_template.jinja` is a modified Harmony wire
format (channels `analysis`/`final`, `# Tools` TS block, `rfl:parameter` XML
bodies) — read it in full before writing the codec.

## 0.4 Ballast

None. Every tensor is classified; an unknown name is a hard converter error.

## 0.5 Feasibility on this machine (RTX 5090 32 GB, 750 GB RAM, 5.1 TB free)

- Routed experts: 49 × 256 = 12 544 experts × 3 × 2560 × 5120 = 493 B params.
  NVFP4 ≈ 0.5625 B/param → **~277 GB on disk**, streamed through the neutral
  `ExpertStore` (mmap views + LRU). Not resident, not pinned.
- Dense + attention + router + FP16 edges: ~5 GB pinned.
- KV at the canonical 2048-token protocol: 52 layers × 2048 × 8 × 128 × 2 × 2 B
  ≈ 0.44 GB (pos-major).
- Source-oracle feasibility: BF16 934 GB > 750 GB RAM → the reference model
  **cannot run on this host**. Instruments are (a) a per-layer CPU oracle over
  the source shards (layer 0 dense + layer 3 MoE), (b) SelfCheck/determinism on
  the converted tree. FSI/CODE/DET goldens are declared debt, as for Ling/Kimi/
  DeepSeek-Pro, with the reason recorded in `docs/OPEN_TASKS.md`.

## 0.6 Semantics to pin (each is a silently-wrong-token risk)

1. **Attention gate**: `gate = gate_proj(x)` viewed as `[heads,head_dim]`
   plus `gate_bias[head]` broadcast over `head_dim`, then
   `attn_out *= sigmoid(gate)` **before** `o_proj`.
2. **q/k RMSNorm** is per-head (over `head_dim`), plain weight form, applied
   before RoPE.
3. **Sandwich norms + depth-scaled residual**: attention branch is
   `post_attention_layernorm(attn_out)`, MLP branch
   `post_feedforward_layernorm(mlp_out)`; combine is
   `branch*(1/sqrt(idx)) + residual*sqrt((idx-1)/idx)` with
   `idx = 2*(layer+1)` for attention and `idx+1` for MLP.
4. **Full layers**: no rotary; `q *= (1 + (pos+1)/2048)^0.1` in FP32.
   Sliding layers: rotary, no SSMax. Window 512 incl. current.
5. **Router**: FP32 logits, sigmoid, `+ e_score_correction_bias` for choice,
   group score = sum of top-2 in each of 8 groups, keep top-4 groups, top-8
   experts by corrected score, weights = **plain** sigmoid scores of the
   selected ids, renormalized; then × 2.5 after expert aggregation.
6. **SwiGLU** activation/product evaluated in FP32 then rounded.
7. **Softcap** on the final logits only, in FP32.
8. Attention scale `1/sqrt(128)` at QK.

## Numeric policy of the engine

Correctness-first token-by-token prefill/decode (DeepSeek-Pro/Ling precedent),
FP16 activations with FP32 residual where the reference uses BF16, FP16
embed/lm_head, FP16 norms/qk-norms/gate_bias, **F32 router + correction bias**
(the router moves the top-k selection), NVFP4 dense/attention/dense-MLP/expert
bodies. A residual plane is not placed initially; it is the documented fallback
if a parity gate is red (ingestion skill, stage 2).
