# GigaChat3.5-432B-A28B-Reasoning — Ingestion Design

Date: 2026-09-20
Status: approved design, pending implementation plan
Branch: `feat/model-GigaChat3.5-432B-A28B-Reasoning`

## Overview

Ingest `GigaChat3.5-432B-A28B-Reasoning` (`GigaChat35ForCausalLM`,
`model_type=gigachat3_5`) into the logestix monorepo following the 8-stage Model
Ingestion Protocol. This is a **new model family** (`gigachat`): a **hybrid
MLA + GatedDeltaNet + MoE** network with FP8-block-128 source weights, three MTP
heads, and the new `ZeroCenteredGatedNorm`. The repo has no prior GigaChat
support and no `ZeroCenteredGatedNorm` / gated-attention precedent.

Source checkpoint (read-only):
`/mnt/llm_models/models/GigaChat3.5-432B-A28B-Reasoning`
(28 safetensors shards, FP8 E4M3 + F32 `weight_scale_inv`, BF16 for the ignored
set; 416 GB on disk; 40 main layers + 3 MTP layers, ~432B params / ~28B active).
Reference math pinned from the shipped
`modeling_gigachat3_5.py` (read-only reference, **not vendored**) and
`configuration_gigachat3_5.py`; `config.json` is authoritative where it differs
from the Python defaults (61→40 layers, `n_group` 8→1, `rms_norm_eps` 1e-20→1e-6).

Milestone scope (operator decision): **full 8-stage ingestion**, including
runnable inference and a real-weight SelfCheck. MTP is **structural only**
(converted + stored + counted, never applied at inference).

## Autonomy invariant

- `gigachat_moe_*` must not `#include` any other family's loader/inference/
  constants headers; geometry is a fresh compile-time frozen block in
  `gigachat_moe_constants.h` under `lsxcommon::gigachat::frozen`.
- Only model-neutral infrastructure is reused: `model_base`, `model_runtime`,
  `expert_store`, `DenseNvfp4Schema`/`ExpertSchema`, `lsx_inference_nvfp4`,
  `lsx_gemv`, `gpu::quantize_tensor_gpu`, tokenizer, `schema`, `path_registry`,
  `gpu_perf`, `sampling`, `resource_governor`.
- Guard registries (`test_code_naming_guard`, `test_model_independence_guard`,
  `test_model_suite_guard`, `test_ctest_labels_guard`) are extended in the SAME
  commit as the model files.

## Tensor inventory (audited 2026-09-20)

Non-layer tensors:

| Tensor | dtype | shape |
|---|---|---|
| `model.embed_tokens.weight` | BF16 | [128256, 7168] |
| `lm_head.weight` | BF16 | [128256, 7168] (untied) |
| `model.norm.weight` | BF16 | [7168] |
| `model.norm.gate_up_projection.weight` | BF16 | [16, 7168] |
| `model.norm.gate_down_projection.weight` | BF16 | [7168, 16] |

Per-layer (layers 0–39):

- Layer types: `full_attention` (MLA) at `{3,7,11,15,19,23,27,31,35,39}`;
  `linear_attention` (GatedDeltaNet) elsewhere (30 layers).
- MoE at layers 3–39 (37 layers, 256 routed experts each ⇒ **9472 experts**);
  dense MLP at layers 0–2.
- MLA layer: `q_a_proj` FP8 [1536,7168] / `q_b_proj` FP8 [12288,1536] /
  `kv_a_proj_with_mqa` FP8 [576,7168] / `kv_b_proj` FP8 [16384,512] /
  `o_proj` FP8 [7168,8192] / `gate_proj` FP8 [8192,7168] (gated attention);
  `q_a_layernorm`, `kv_a_layernorm` = ZeroCenteredGatedNorm.
- GDN layer: `in_proj_qkvz` FP8 [24576,7168] / `out_proj` FP8 [7168,8192] /
  `conv1d.weight` BF16 [16384,1,4] / `A_log` BF16 [64] /
  `dt_bias` BF16 [64] / `in_proj_ba` BF16 [128,7168] / `norm.weight` BF16 [128]
  (zero-centered sigmoid gated RMSNorm, gate from `z`).
- Every layer: `input_layernorm`, `post_attention_layernorm`,
  `post_self_attn_layernorm`, `post_feedforward_layernorm` = ZeroCenteredGatedNorm
  (`weight` + `gate_up_projection` [16,7168] + `gate_down_projection` [7168,16]).
- MoE layer: `mlp.gate.weight` BF16 [256,7168],
  `mlp.gate.e_score_correction_bias` [256],
  `mlp.shared_experts.{gate,up,down}_proj` FP8 (I=2048),
  `mlp.experts.E.{gate,up,down}_proj` FP8 (I=2048).
- Dense layer: `mlp.{gate,up}_proj` FP8 [18432,7168], `mlp.down_proj` FP8 [7168,18432].

MTP layers 40–42 (cut + counted): `eh_proj` BF16 [7168,14336],
`embed_tokens` BF16, `shared_head.head` BF16, `shared_head.norm` ZeroCenteredGatedNorm.

### FP8 tile geometry

All FP8 linears carry `weight_scale_inv` F32 of shape
`[ceil(R/128), ceil(C/128)]`. Every matrix is an exact multiple of 128 except
`kv_a_proj_with_mqa` [576,7168] with scale [5,56] (rows 576 = 4·128 + 64) — the
dequant kernel MUST clamp the partial last tile.

## Frozen geometry (compile-time)

Locked in `libs/lsxcommon/src/gigachat_moe_constants.h` with `static_assert`;
`GigaChat35Model::Load` cross-checks the parsed index against it. Escape hatch
for tests: `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY` (runtime env, repo policy).

| Constant | Value | Source |
|---|---|---|
| kType | `"gigachat3.5-432b-a28b-reasoning"` | registry id |
| kHidden | 7168 | `hidden_size` |
| kLayers | 40 | `num_hidden_layers` |
| kVocab | 128256 | `vocab_size` |
| kFullAttnLayers | `{3,7,11,15,19,23,27,31,35,39}` | `full_attention_layers` |
| kFirstKDense | 3 | `first_k_dense_replace` |
| kRmsEps | 1e-6f | `rms_norm_eps` |
| kNormGatingWeight | 2.0f | `layernorm_gating_weight` |
| kGateRank | 16 | `gate_up/down_projection` |
| kMtpLayers | `{40,41,42}` | `num_nextn_predict_layers` |

MLA (`GigaChat35Attention`):

| Constant | Value | Formula |
|---|---|---|
| kHeads | 64 | `num_attention_heads` |
| kKVHeads | 64 | `num_key_value_heads` (GQA groups = 1) |
| kQkHeadDim | 192 | `qk_nope_head_dim + qk_rope_head_dim` |
| kQkNopeHeadDim | 128 | `qk_nope_head_dim` |
| kQkRopeHeadDim | 64 | `qk_rope_head_dim` |
| kVHeadDim | 128 | `v_head_dim` |
| kQLoraRank | 1536 | `q_lora_rank` |
| kKvLoraRank | 512 | `kv_lora_rank` |
| kAlphaQ | `sqrt(7168/1536)` = 2.16025 | `use_mla_scaling_factor` |
| kAlphaKv | `sqrt(7168/512)` = 3.74166 | `use_mla_scaling_factor` |
| kRopeTheta | 100000.0 | `rope_theta` |
| kYarnFactor / kYarnOrig | 8.0 / 32768 | `rope_scaling` |
| kYarnBetaFast / kYarnBetaSlow | 32.0 / 1.0 | `rope_scaling` |
| kYarnMscale / kYarnMscaleAllDim | 1.0 / 1.0 | `rope_scaling` |
| kRopeInterleave | true | `rope_interleave` |
| kMscale | `yarn_get_mscale(8,1)=1.20794` | derived |
| kAttnScale | `192^-0.5 * kMscale^2` = 0.105299 | `Attention.scaling` |
| kGatedAttention | true | `gated_attention` |

GatedDeltaNet (`GigaChat35GatedDeltaNetWrapper`):

| Constant | Value | Formula |
|---|---|---|
| kLinearNumKeyHeads | 32 | `linear_num_key_heads` |
| kLinearNumValueHeads | 64 | `linear_num_value_heads` |
| kLinearKvRepeat | 2 | `64/32` (repeat_interleave) |
| kLinearKeyHeadDim | 128 | `linear_key_head_dim` |
| kLinearValueHeadDim | 128 | `linear_value_head_dim` |
| kLinearConvKernel | 4 | `linear_conv_kernel_dim` |
| kLinearKeyDim | 4096 | `32*128` |
| kLinearValueDim | 8192 | `64*128` |
| kLinearConvDim | 16384 | `2*4096 + 8192` |
| kLinearQkvzDim | 24576 | `2*4096 + 2*8192` |
| kLinearBaDim | 128 | `2*64` |
| kLinearSigmoidGateScale | 2.0 | `linear_sigmoid_gate_scale` |
| kLinearONormEps | 1e-6 | `linear_attn_o_norm_eps` |

MoE (`GigaChat35MoE`) and MLP:

| Constant | Value |
|---|---|
| kNumRouted | 256 |
| kTopK | 8 |
| kMoeIntermediate | 2048 |
| kSharedExperts | 1 |
| kSharedIntermediate | 2048 |
| kRouterGroups / kRouterTopkGroup | 1 / 1 (identity) |
| kRoutedScale | 2.5 |
| kNormTopkProb | true |
| kSwigluLimit | 10.0 (gate `clamp max`, up `clamp ±`) |
| kUseSharedExpertSigmoid | false |
| kDenseIntermediate | 18432 |
| kFinalLogitSoftcapping | none |

Special tokens (read from config/tokenizer, never hardcoded in kernels):

| Constant | Value |
|---|---|
| kBosId | 1 (`<s>`) |
| kEosId | 2 (`</s>`) |
| kPadId | 2 |
| kRoleSepId | 128000 (`<|role_sep|>\n`) |
| kMessageSepId | 128001 (`<|message_sep|>\n\n`) |
| kThinkId | 128014 (`<think>`) |
| kThinkEndId | 128015 (`</think>`) |

## Pinned reference formulas

### ZeroCenteredGatedNorm (all norms)

```
n   = x.float(); n = n * rsqrt(n.pow(2).mean(-1, keepdim=True) + eps)
n   = n * (1.0 + w.float())                       # zero-centered
h   = silu(n @ gate_up_projection.float().T)      # [.., 16]
h   = h @ gate_down_projection.float().T          # [.., hidden]
y   = (n * layernorm_gating_weight * sigmoid(h))  # gating weight = 2.0
return y.to(x.dtype)
```
The gate is computed from the **normalized** hidden (post-norm), not the input.

### GatedDeltaNet output norm (`gated_rmsnorm_sigmoid_zero_centered`)

```
n = o.float(); n = n * rsqrt(n.pow(2).mean(-1, keepdim=True) + o_norm_eps)
n = n * (1.0 + norm.weight.float())               # zero-centered
n = n * sigmoid(z.float()) * linear_sigmoid_gate_scale   # scale = 2.0
```

### Decoder layer (`layernorm_type = pre_post`)

```
r = x
x = input_layernorm(x)                            # ZeroCenteredGatedNorm
x = self_attn(x)                                  # MLA or GDN
x = post_self_attn_layernorm(x)
x = r + x
r = x
x = post_attention_layernorm(x)
x = mlp(x)                                        # dense or MoE
x = post_feedforward_layernorm(x)
x = r + x
```

### MLA attention

```
q_latent = q_a_layernorm(q_a_proj(x))
q = q_b_proj(alpha_q * q_latent).view(H, 192)     # split 128 nope / 64 rope
ckv = kv_a_proj_with_mqa(x)                        # [512 + 64]
k_pass = kv_a_layernorm(ckv[:512])
kv = kv_b_proj(alpha_kv * k_pass).view(H, 256)     # split 128 nope / 128 value
k_rot = ckv[512:].view(1, 64)
(q_rot, k_rot) = rope_interleave(q_rot, k_rot, yarn(pos))
q = cat(q_pass, q_rot); k = cat(k_pass, k_rot_broadcast)
attn = softmax(q @ k^T * kAttnScale, F32)          # causal; no GQA
o = attn @ v  -> flatten(8192)
o = o * sigmoid(gate_proj(x))                      # gated attention
y = o_proj(o)
```

### GatedDeltaNet forward

```
qkvz = in_proj_qkvz(x); ba = in_proj_ba(x)
# view [.., 32, 128(q)+128(k)+256(v)+256(z)]; ba [.., 32, 2(b)+2(a)]
q,k,v (reshape to [.., 4096/8192]); z (v-heads)
mixed = cat(q,k,v); mixed = silu(conv1d(mixed))    # depthwise k=4, causal
beta = sigmoid(b);  g = -exp(A_log) * softplus(a + dt_bias)
q = repeat_interleave(q, 2, dim=heads); k likewise   # 32 -> 64 heads
core = gated_delta_rule(l2norm(q), l2norm(k), v, g, beta, state)
out = gated_norm(core, z)                          # zero-centered sigmoid, x2
y = out_proj(out)
```

### MoE

```
logits = x.float() @ gate.weight.float().T         # F32
scores = sigmoid(logits)
choice = scores + e_score_correction_bias           # n_group=1 -> identity group
idx = topk(choice, 8)
w = scores.gather(idx); w /= w.sum + 1e-20          # norm_topk_prob
w *= 2.5
routed = sum_e w_e * down_e(swiglu(gate_e(x), up_e(x), L=10))
shared = down_s(swiglu(gate_s(x), up_s(x), L=10))   # all tokens
y = routed + shared
```
`swiglu`: `gate = clamp(gate, max=L)`, `up = clamp(up, -L, L)`,
`silu(gate) * up`.

## Converted layout: `logestix-gigachat-nvfp4`

Output under `<model>/logestix/` via `lsx_model_metadata`-relative paths (no
hardcoded paths in the converter). The index (`logestix/model.index.json`):

```
format: "logestix-gigachat-nvfp4"
model:  "GigaChat3.5-432B-A28B-Reasoning"
num_layers: 40
layer_types: ["linear_attention"|"full_attention"] x 40
full_attention_layers: [3,7,11,15,19,23,27,31,35,39]
geometry: { hidden, vocab, heads, qk_head_dim, q_lora, kv_lora,
            num_experts, top_k, moe_intermediate, linear_* }
vram: { min_bytes, optimal_bytes, max_bytes, context_tokens: 262144 }
files: { pinned: { global, layer_000..layer_039 },
         experts: layer_N/exp_E.arrow, shared_experts, mtp }
deferred: { mtp: "stored", mtp_layers: [40,41,42] }
```

### Globals (`pinned/global.arrow`)
- `embed.weight` BF16 [128256, 7168]
- `head.weight` BF16 [128256, 7168] (untied)
- `norm.weight` BF16 [7168] + `norm.gate_up` [16,7168] + `norm.gate_down` [7168,16]

### MLA layer (`pinned/layer_NNN.arrow`)
- 4 x ZeroCenteredGatedNorm (input / post_attention / post_self_attn /
  post_feedforward): `weight`, `gate_up`, `gate_down` BF16
- `q_a_proj.nvfp4` [1536,7168], `q_b_proj.nvfp4` [12288,1536],
  `kv_a_proj_with_mqa.nvfp4` [576,7168], `kv_b_proj.nvfp4` [16384,512],
  `o_proj.nvfp4` [7168,8192], `attn_gate_proj.nvfp4` [8192,7168]
- `q_a_layernorm` / `kv_a_layernorm` ZeroCenteredGatedNorm BF16
- MoE (all MLA layers 3–39): `mlp.gate.weight` BF16 [256,7168],
  `mlp.gate.e_score_correction_bias` F32 [256],
  `mlp.shared_experts.{gate,up,down}.nvfp4` (I=2048)

### GDN layer (`pinned/layer_NNN.arrow`)
- 4 x ZeroCenteredGatedNorm (same as above)
- `in_proj_qkvz.nvfp4` [24576,7168], `out_proj.nvfp4` [7168,8192]
- `conv1d.weight` BF16 [16384,1,4], `A_log` F32/BF16 [64],
  `dt_bias` F32/BF16 [64], `in_proj_ba` BF16 [128,7168], `norm.weight` BF16 [128]
- dense MLP (GDN layers 0–2 only): `mlp.gate_proj.nvfp4` [18432,7168],
  `up_proj.nvfp4` [18432,7168], `down_proj.nvfp4` [7168,18432]
- MoE block (GDN layers 3–39) as above

### Routed experts (`experts/layer_NNN/exp_EEE.arrow`)
Per-expert NVFP4 `gate`/`up` [2048,7168] and `down` [7168,2048] via
`ExpertSchema` (`PackedExpert` slot convention: w1=gate, w3=up, w2=down).
9472 experts total.

### MTP (`mtp/layer_NNN.arrow`, NNN in {40,41,42})
`eh_proj`, `embed_tokens`, `shared_head.head`, `shared_head.norm` — converted
and stored, never loaded resident and never applied.

Norms, gate projections, `A_log`, `dt_bias`, `conv1d`, `in_proj_ba`,
`e_score_correction_bias`, router `gate.weight`, embeddings and `lm_head` are
never quantized (MODEL INGEST step 2).

## Numeric conversion pipeline

For every FP8 linear:
1. read `weight` (F8_E4M3) + `weight_scale_inv` (F32, 128×128 tiles);
2. `DequantFp8Block128ToBf16Gpu` — new, partial-tile-clamped, gigachat-owned
   (`gigachat_moe_dequant.cu`), mirroring the GLM-5.3-Flash
   `DequantFp8TilesToBf16Gpu` pattern but handling `ceil(R/128)` grids
   (`kv_a_proj_with_mqa`);
3. `gpu::quantize_tensor_gpu(bf16, dtype="BF16")` → E2M1 nibbles + per-16 E4M3
   scales + float `global_scale` (the neutral `lsx_nvfp4` contour);
4. dense records → `DenseNvfp4Schema`; expert triplets → `ExpertSchema`.

BF16/F32 tensors pass through verbatim (safetensors dtype preserved).

## Inference data flow and state

`embed → for layer in 0..39: (MLA | GDN) + MLP → final norm → lm_head(F32) → sample`.

Per-session device state:
- **KV cache** for the 10 MLA layers (64 heads, 192-dim K incl. 64 rope /
  128-dim V; GQA=1 so no repeat).
- **GDN recurrent state** for the 30 linear layers: `[64 heads, 128, 128]` F32
  per layer + conv state `[16384, 3]` F16.
- **Expert cache**: host mmap of 9472 experts + device slot-LRU (Nemotron
  pattern: dense resident, experts streamed). Dense resident ≈ 10 GiB
  (attn/MLP/shared/embeds); experts ≈ 208 GiB NVFP4 on host.
- Single CUDA stream, no per-token `cudaDeviceSynchronize`; reductions use the
  shared-memory sandbox (thread-0 + barriers) per `nvfp4-blackwell-conventions`.

## Prompt framing and tokenizer

- Tokenizer: `BuildTokenizerArrowFiles(tokenizer.json)` (HF BPE, merges may be
  `"a b"` or `["a","b"]`). `bos_id`/`eos_id`/`pad_id` read from the config
  (1 / 2 / 2), never hardcoded in kernels.
- `BuildPromptInputIds` builds a chatml-v5 single-turn prompt:
  `<s>` + default Russian system turn + `user<|role_sep|>\n{raw}<|message_sep|>\n\n`
  + `assistant<|role_sep|>\n<think></think>` (empty think prefilled so the
  control prompt answers directly — GLM precedent). The default system text is
  encoded through `arrow_tokenizer_encode` (multi-token, not a sentinel), while
  the special tokens are resolved via `arrow_tokenizer_lookup`.
- SelfCheck uses `runtime::RunParisSelfCheck(..., {.frame_prompt = true})`;
  `"The capital of France is" → "Paris"` (top-1/short prefix). Risk recorded:
  an empty `<think></think>` prefill may not be honoured by the RL model; if it
  is not, the fallback is `substring_match` over a larger window after
  `</think>`, recorded as a deviation.

## File map and build

```
libs/lsxcommon/src/gigachat_moe_constants.h
libs/lsxcommon/src/gigachat_moe_convert.{h,cpp}
libs/lsxcommon/src/gigachat_moe_dequant.cu
libs/lsxcommon/src/gigachat_moe_load.cpp
libs/lsxcommon/src/gigachat_moe_model.{h,cpp}
libs/lsxcommon/src/gigachat_moe_device_math.h
libs/lsxcommon/src/gigachat_moe_norm_kernels.cu
libs/lsxcommon/src/gigachat_moe_kernels.cu
libs/lsxcommon/src/gigachat_moe_inference_attn.cu
libs/lsxcommon/src/gigachat_moe_inference_moe.cu
libs/lsxcommon/src/gigachat_moe_inference_pipeline.cu
libs/lsxcommon/src/gigachat_moe_expert_cache.{h,cpp}
tests/test_gigachat_moe_fixture.h
tests/test_gigachat_moe_convert.cpp
tests/test_gigachat_moe_loader.cpp
tests/test_gigachat_moe_model.cpp
tests/test_gigachat_moe_kernels.cu
tests/test_gigachat_moe_selfcheck.cpp
```

- New OBJECT library `lsxcommon_gigachat_cu` (`-fmad=false`, bit-exact math),
  linked PUBLIC into `lsxcommon`.
- New `lsxforge` flag `--gigachat`
  (`lsxcommon::gigachat::ConvertGigaChatModel`).
- Register `GigaChat35Model` in `model_registry.cpp`.
- Tokenizer bridge, passport (`passport/passport.bin` via
  `lsxnetwork::BuildPassport`) and `LsxTokenTranslator` in `IoWorker` after
  affinity pinning, before readiness.

## Guard registries (same commit)

- `test_code_naming_guard.cpp`: add `gigachat` to `fam` and `FamilyStems()`.
  No new role/subsystem/variant tokens are needed (`convert`, `load`, `model`,
  `inference_{attn,moe,pipeline}`, `kernels`, `norm_kernels`, `dequant`,
  `device_math`, `expert_cache`, `constants` all exist).
- `test_model_independence_guard.cpp`: `kPassports` row
  `{"gigachat3.5-432b-a28b-reasoning", "gigachat_moe_constants.h",
  "lsxcommon::gigachat::frozen"}`; owner marker `namespace frozen {`;
  `gigachat_moe_model.cpp` in the per-`Infer` `LsxFastMathScope` consumers list.
- `test_model_suite_guard.cpp`: `Suites()` row
  `convert → load → model → kernels → selfcheck`; add `gigachat` to its
  `FamilyStems()`.
- `test_ctest_labels_guard.cpp`: labels on every new test in the same commit.

## Testing strategy (TDD)

Strict red→green per stage; every new target gets its ctest label in the same
commit; the selfcheck target gets `SKIP_RETURN_CODE 4`.

1. **fast/host** — passport `static_assert`s; `kType` resolution and
   frozen-geometry rejection; layer-type decoding; ZeroCenteredGatedNorm /
   swiglu_limit / router reference math in `gigachat_moe_device_math.h`;
   `kAttnScale == 192^-0.5 * (0.1*ln(8)+1)^2`.
2. **convert** (label `convert`, bounded TIMEOUT ≤120 s) — synthetic E2E on
   `test_synth_checkpoint.h`: globals + one MLA + one GDN + one dense + one MoE
   layer, all 256 experts of the MoE layer, an FP8 tensor with a partial tile
   (`[576,7168]` / scale `[5,56]`); assert the `logestix-gigachat-nvfp4` index,
   NVFP4 scale geometry, untouched high-precision tensors, and the counted MTP
   record. No real-model conversion in tests.
3. **load** — synthetic converted fixture only; format-tag gate, frozen-geometry
   gate, missing/corrupt tensor diagnostics, expert LRU bookkeeping.
4. **kernels** — device vs host reference parity, bit-exact with `-fmad=false`:
   ZeroCenteredGatedNorm, GDN sigmoid-zero-centered gated norm, depthwise
   causal conv + silu, gated delta rule, interleaved yarn RoPE + MLA alpha
   scaling, gated attention, sigmoid router top-8 with `e_score_correction_bias`
   and `×2.5`, swiglu_limit expert/dense MLP, FP8 block-128 dequant (with a
   partial tile).
5. **selfcheck** — incremental 1→2-token `"Paris"` over ≥3 cycles
   (`LSX_TEST_GIGACHAT_SELFCHECK_CYCLES`), bit-exact `final_logits`, real model
   via env (`LSX_TEST_GIGACHAT_ROOT`), `SKIP()` when absent.

Perf: `tests/perf/perf_baselines.json` entry + `test_perf_baseline_gigachat`
target (label `perf`), measured after correctness at the canonical protocol.

## Error handling and strictness

- Load rejects a non-`logestix-gigachat-nvfp4` index, a geometry mismatch vs the
  frozen passport, and any missing / mis-shaped / corrupt tensor with a
  diagnostic naming the tensor.
- Conversion is strict: MTP (`model.layers.40..42.*`) is the only cut, with a
  counted record; any other unknown tensor is an error — never silently ignored.
- `SelfCheck` returns `selfcheck_passed=false` with the decoded token on
  failure; no partial-success state.

## Risks / decision points

- **Primary:** ZeroCenteredGatedNorm, GDN sigmoid-zero-centered output gating,
  gated attention and the FP8 partial-tile dequant are net-new kernels; each is
  pinned against the host reference before the layer pipeline is wired.
- Interleaved yarn RoPE + MLA `alpha_q`/`alpha_kv` scaling must match the
  reference exactly; a fast/host test pins `kAttnScale` and the RoPE table.
- Real conversion of 416 GB and 9472 experts is a long offline step (hours,
  ~230 GB output); it is not run inside tests, and the real-weight SelfCheck
  SKIPs until it is done.
- Single RTX 5090 (32 GiB): dense resident (~10 GiB) + expert streaming; t/s is
  expected to be modest and is recorded as measured, not targeted.
- Empty `<think></think>` prefill for the `"Paris"` control prompt may not be
  honoured; fallback and deviation are recorded in stage 8.
- The shipped `modeling_gigachat3_5.py` is used as a read-only reference and is
  NOT committed to the repository.

## Out of scope (follow-ups)

MTP speculative decoding; prefill chunk-scan optimization; multimodal surfaces
(none in this checkpoint); extracting the FP8-tile dequant into neutral
infrastructure and refactoring GLM-5.3-Flash onto it.
