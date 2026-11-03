# Kimi-K3 ingestion — design

Date: 2026-11-03
Source: `/mnt/fast_storage/huggingface/Kimi-K3` (2.8T params, 1.5 TB, 96 shards)
Registry: `kimi-k3` (family `kimi`, variant `k3`, files `kimi_moe_k3_*`)
Format tag: `logestix-kimi-k3-nvfp4`

## 1. Stage 0 triage (measured from the checkpoint)

### 1.1 Classification

Text tower is `KimiLinearForCausalLM` (`model_type: kimi_linear`), wrapped by
`KimiK3ForConditionalGeneration` (`model_type: kimi_k3`). It is a **new variant
of the existing `kimi` family** — not a sibling of Kimi-K2.7-Code's
DeepSeek-V3 contour: the attention is Kimi Delta Attention (KDA) + gated NoPE
MLA, there is an Attention-Residual (AttnRes) depth mixer, and the MoE is a
Stable LatentMoE. It therefore gets its own frozen passport
(`lsxcommon::kimi_k3::frozen`) and its own kernels/loader; only the tokenizer
bridge (tiktoken, `tokenizer_kind 2`) and the neutral layers are shared.

### 1.2 Geometry (frozen)

| field | value |
|---|---|
| layers | 93 (`0..92`); layer 0 dense MLP, 1..92 MoE (`first_k_dense_replace=1`) |
| hidden | 7168 |
| KDA layers (0-based) | all except `3,7,11,...,91,92` (69 layers) |
| gated MLA layers | `3,7,11,...,91,92` (24 layers) |
| AttnRes | `attn_res_block_size = 12` |
| KDA | 96 q/k/v heads × 128, conv-4 depthwise (silu), `use_full_rank_gate`, `gate_lower_bound=-5`, `A_log[128]` (per channel), `dt_bias[12288]`, `o_norm[128]` |
| MLA | 96 heads; `q_lora=1536`, `kv_lora=512`, `qk_nope=128`, `qk_rope=64`, `v=128`; **NoPE** (`mla_use_nope`, no rotary module at all — the 64 "rope" dims are plain dims, `k_rot` is one head broadcast); output gate `g_proj`; scale `1/sqrt(192)`; no KV repeat (`num_key_value_heads == num_attention_heads`) |
| MoE | 896 routed experts, top-16, `moe_intermediate_size=3072`, **latent** `routed_expert_hidden_size=3584` with RMSNorm, 2 shared experts (I=6144, full hidden), sigmoid router + `e_score_correction_bias`, `moe_renormalize`, `routed_scaling_factor=1.0` |
| activation | SiTU-GLU: `beta*tanh(g/beta)*sigmoid(g) * (linear_beta*tanh(u/linear_beta))`, beta 4.0, linear_beta 25.0 |
| vocab | 163840 (163584 base + 256 reserved) |
| ids | bos 163584 `[BOS]`, stop 163586 `<|end_of_msg|>` (+ 163585 `[EOS]`), pad 163839, unk 163838 |
| norm eps | 1e-5, weight-form RMSNorm (not Gemma `1+w`) |
| ballast | `vision_tower.*` (162 tensors) + `mm_projector.*` (4), `mtp` none |

### 1.3 Source quantization

`compressed-tensors` `mxfp4-pack-quantized`:
- routed experts: E2M1 pairs packed in `U8` (`[rows, cols/2]`) + `E8M0` scale per
  32 columns (`U8 [rows, cols/32]`), **no** global scale.
- everything else (attention, shared experts, latent projections, router,
  embed/lm_head, layer-0 MLP) is BF16; KDA conv/`A_log`/`dt_bias`/`o_norm` are
  F32.

### 1.4 Semantics that must be reproduced (verified in `modeling_kimi_linear.py`,
`configuration_kimi_k3.py`, `fla/ops/kda/*`)

1. **AttnRes** (`_apply_attn_res`): sources = `[block_residual..., prefix_sum]`;
   each source is RMS-normalized (no mean subtraction), dotted with
   `norm.weight * proj.weight[0]`, softmaxed over sources, and the mixture is
   the FP32 weighted sum. At `layer % 12 == 0` the incoming `prefix_sum` is
   pushed into `block_residual` and `prefix_sum` restarts empty; the mixture for
   the attention input is skipped only when `block_residual` is empty (layer 0).
   `prefix_sum` accumulates the attention output then the MLP output. After the
   last layer the same mixer runs with `output_attn_res_{proj,norm}`, then the
   final RMSNorm.
2. **KDA** (`fused_recurrent_kda` + `fused_recurrent_kda_fwd_kernel`):
   - per step: `q,k` L2-normalized over the 128 head dims (eps 1e-6, as in the
     kernel), `q` then multiplied by `1/sqrt(128)`;
   - log-decay `g_d = lower_bound * sigmoid(exp(A_log_d) * (g_d + dt_bias[h,d]))`,
     `lower_bound=-5` (`use_lower_bound` path — `gate_lower_bound` is set);
   - `S[d,v] *= exp(g_d)`; `pred_v = sum_d S[d,v] k_d`;
     `delta_v = (v_v - pred_v) * sigmoid(beta_h)`; `S[d,v] += k_d delta_v`;
     `o_v = sum_d S[d,v] q_d` (current token included);
   - `o = rmsnorm_128(o) * sigmoid(gate)` (`FusedRMSNormGated`, sigmoid) then
     `o_proj`;
   - `A_log` has 128 entries = head_dim, so the decay is **channel-wise**
     (shared across the 96 heads); `dt_bias`/`g` are per (head, channel). The
     reference `fla` code indexes `A_log` by v-head, which is inconsistent with
     the shipped 128-entry tensor; the checkpoint shape is authoritative.
3. **Gated MLA**: `q = q_b(norm(q_a(h)))` → split 128/64; `kv = kv_a(h)` (576)
   → `[latent 512 | k_rot 64]`; `k,v = split(kv_b(norm(latent)))`; no rotation;
   scores `q·k / sqrt(192)`; `out = sigmoid(g_proj(h)) * attn_out`; `o_proj`.
   KV cache stores the 576-wide latent+rope per position per layer.
4. **LatentMoE**: router from the pre-MoE hidden (F32 dot, sigmoid, select
   top-16 by `score + bias`, weights = raw sigmoid renormalized, ×1.0);
   `x3584 = down(hidden)`; `routed = sum_k w_k * expert_k(x3584)` (each expert
   `w1`/`w3` → SiTU → `w2`); `routed = rmsnorm_3584(routed)`; `y = up(routed)`;
   `y += shared(hidden)`; `out = hidden_residual + y` handled by the AttnRes
   mixer (never a plain residual add).
5. **Tokenizer / chat**: tiktoken BPE, Han-run-first case-run pattern, XTML
   chat (`<|open|>tag attrs<|sep|>body<|close|>tag<|sep|><|end_of_msg|>`);
   `<|open|>/<|close|>/<|sep|>` are literal (tiktoken-special) tokens that
   survive decode, `<|end_of_msg|>` is the EOS.

## 2. Quantization recipe

All dense linear matrices → NVFP4 (`E2M1` + per-16 `E4M3` scale + per-tensor
`global_scale`). Routed experts: dequantize MXFP4 (E8M0/32) on the GPU, requant
into the neutral NVFP4 record with per-expert global scales (mimo/ling
precedent).

Kept high precision (never quantized): all RMSNorm weights, KDA conv/`A_log`/
`dt_bias`/`o_norm`, `b_proj` (beta) and `f_a/f_b_proj` (decay gate) in FP16,
router `gate` BF16 + `e_score_correction_bias` F32, AttnRes `proj`/`norm` FP16.

`embed_tokens` and `lm_head` → NVFP4 (1.3 GB total; the residual-plane recipe is
available if a parity gate needs it).

## 3. Memory plan (RTX 5090, 31.5 GiB usable)

| item | bytes |
|---|---|
| KDA projections (69 layers) | 17.2 GiB |
| MLA projections (24 layers) | 3.1 GiB |
| layer-0 dense MLP | 0.4 GiB |
| embed + lm_head (NVFP4) | 1.3 GiB |
| routers (BF16) | 1.2 GiB |
| norms/conv/gates/misc | 0.5 GiB |
| **pinned total** | **≈ 23.7 GiB** |
| streamed per layer: latent down/up + shared experts | 9.5 GiB (mmap Arrow views → one H2D per layer per step) |
| routed experts | 27 GiB/token through the neutral `ExpertStore` (mmap + device slot LRU) |

The streamed plane is what makes the model fit this card: pinning it needs
≈34 GiB. It is a documented knob (`LSX_KIMI_K3_PIN_DENSE=1`) for larger cards;
the default streams.

## 4. Components

```
libs/lsxcommon/src/
  kimi_moe_k3_constants.h            frozen passport (lsxcommon::kimi_k3::frozen)
  kimi_moe_k3_convert.{h,cpp}        lsxforge --kimi_k3
  kimi_moe_k3_dequant.cu             MXFP4(E8M0/32) -> BF16 partial tiles
  kimi_moe_k3_load.{h,cpp}           pinned graph + streamed plane + ExpertStore
  kimi_moe_k3_kernels.{h,cu}         SiTU, router top-16, latent MoE glue,
                                     KDA conv/recurrence (per-channel decay),
                                     AttnRes depth-mix, gated NoPE MLA decode
  kimi_moe_k3_inference_attn.{h,cu}  MLA latent KV cache + decode attention
  kimi_moe_k3_inference_pipeline.{h,cu}  93-layer driver + generation
  kimi_moe_k3_model.{h,cpp}          IModel, registry, SelfCheck, lifecycle
  kimi_moe_k3_tool_format.{h,cpp}    XTML render/parse (guard allowlist)
apps/lsxforge/                      --kimi_k3 entry
tests/                              convert/load/model/kernels/selfcheck/
                                    fsi/code/det/perf + guards
```

## 5. Kernel plan

- one stream, no per-token `cudaDeviceSynchronize`; NVFP4 only through
  `lsx_inference_nvfp4`; shared-memory sandbox (thread-0 RMW behind barriers);
  no `atomicAdd` feeding `final_logits`.
- KDA recurrence: one block per head, F32 state `[128][128]` per head per layer
  (`93 × 96 × 16 KiB = 143 MiB` device), deterministic block reductions; prefill
  and decode both use the recurrent form (token-by-token, correctness first).
- MLA decode: per-head dot over the cached latent (512) recomputed through
  `kv_b` per step (12.6 M MAC/layer/step) + a 96-head causal softmax over the
  pos-major cache.
- AttnRes mixer: one block per token, 9-source softmax, F32 accumulation.
- SiTU + SwiGLU in one kernel per expert/shared MLP.

## 6. Gates

1. Stage 0 passport + naming/passport/suite registries (same commits).
2. Converter: synthetic checkpoint E2E (`convert` label, 120 s timeout,
   `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1`).
3. Loader: synthetic converted fixture (no vault).
4. Kernels: KDA recurrence vs host reference, AttnRes vs host reference, router
   top-16 semantics, SiTU, gated NoPE attention vs host reference.
5. SelfCheck: `The capital of France is` → `Paris`, ≥3 cycles bit-exact,
   `SKIP_RETURN_CODE 4`.
6. Tokenizer parity ≥30 mixed prompts against `tiktoken.model` (Python oracle).
7. FSI/CODE/DET: **template-derived** goldens (source needs `fla-core` and a
   multi-GPU stack); declared as such in `SELFCHECK_MATRIX.md`.
8. Perf rows seeded through `lsxbenchmark --bench` (never hand-edited), both
   `perf` and `r3`, with the streamed-plane cost documented.
9. Report `docs/model-reports/2026-11-03-Kimi-K3.md` after the real-weight
   numbers exist.

## 7. Explicit deviations / risks

- `A_log[128]` per-channel decay (checkpoint shape) vs the `fla` reference's
  per-head indexing — documented above; if a gate is red on KDA numerics this
  is the first hypothesis.
- FSI/CODE/DET goldens are template-derived (no runnable source oracle).
- Decode speed is host-store bound (~27 GiB routed + 9.5 GiB streamed per
  token); the model is correctness-first here.
