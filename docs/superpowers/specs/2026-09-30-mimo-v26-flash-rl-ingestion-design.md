# MiMo-V2.6-Flash-RL — Ingestion Design

Date: 2026-09-30
Status: approved design, pending implementation plan
Branch: `feat/model-MiMo-V2.6-Flash-RL`

## Overview

Ingest `MiMo-V2.6-Flash-RL` (`MiMoV2ForCausalLM`, `model_type=mimo_v2`) into
the logestix monorepo following the 8-stage Model Ingestion Protocol plus the
mandatory stage 4b tool-calling bridge. This is a **new model family** (`mimo`):
a **hybrid SWA + full-attention sparse MoE** backbone with asymmetric QK/V head
dims, a learned attention sink bias on the sliding-window layers, a sigmoid
`noaux_tc` router, FP8-block-128 dense projections, and **routed experts already
stored as MXFP4** (E2M1 nibbles + F8_E8M0 per-32 scales).

Source checkpoint (read-only):
`/mnt/llm_models/models/MiMo-V2.6-Flash-RL`
(config + 64 expert-parallel safetensors shards + one MTP shard + a multimodal
vision/audio payload; 172 GB total, 309B total / ~15B activated params,
1M-token context, ChatML + `<think>` chat template).

Milestone scope (operator decision): **full 8-stage ingestion**, including
runnable inference and a real-weight SelfCheck. Multimodal surfaces
(vision / audio / speech) are **stripped**; MTP is **structural only**
(converted + stored + counted, never applied at inference). Real conversion of
172 GB is a long offline step and is never run inside tests.

Reference math is pinned from the shipped, read-only
`modeling_mimo_v2.py` / `configuration_mimo_v2.py` / `chat_template.jinja`;
`config.json` is authoritative where it differs from Python defaults.

## Autonomy invariant (first-class, operator directive)

The model MUST be autonomous relative to every other model in the repo: adding
or tuning it never changes another model's geometry or runtime behaviour. This
is the same invariant GigaChat / Nemotron / Granite satisfy and it is
machine-enforced by the four guard registries; it is extended in the SAME
commit as the model files.

| Rule | MiMo compliance |
|---|---|
| Unique frozen passport in a model-unique namespace | `libs/lsxcommon/src/mimo_moe_constants.h` in `lsxcommon::mimo::frozen` |
| Passport is a leaf (includes no other passport) | no cross-includes; enforced by `no passport includes another model's passport` |
| No cross-family geometry borrowing | `mimo_moe_*` includes no `glm_*`/`qwen_*`/`deepseek_*`/`gemma_*`/`kimi_*`/`nemotron_*`/`gigachat_*`/`ornith_*`/`granite_*` header; dequant and expert-cache are MiMo-owned TUs that mirror, not include, those precedents |
| Test suites are family-isolated | `test_mimo_moe_*` include only `test_mimo_moe_fixture.h` + the neutral `test_synth_checkpoint.h`; never another family's fixture |
| Per-`Infer` fast-math policy | `LsxFastMathScope(EffectiveFastMathMode(cfg_))` in `MiMoV26FlashModel::Infer`; row added to the guard's `consumers` list |
| Neutral infrastructure only | `lsx_inference_nvfp4`, `lsx_gemv`, `PackExpertNVFP4`, `DenseNvfp4Schema`/`ExpertSchema`, `expert_store`, `schema`, `safetensors_gpu`, `converter`, `model_base`/`model_runtime`, `tokenizer`, `path_registry`, `vram_estimate`, `gpu_perf`, `sampling`, `resource_governor` |
| Family-scoped kernel selection | no `g_mimo_*` process-global toggle is added; all NVFP4 math goes through the single neutral `lsx_nvfp4` layer |

**Shared registration surfaces (additive, one row/branch each — never a
behavioural edit to another model):**

- `libs/lsxcommon/src/model_registry.cpp` — `+ Register<MiMoV26FlashModel>()`.
- `libs/lsxcommon/CMakeLists.txt` — new `lsxcommon_mimo_cu` OBJECT library +
  source entries + link into `lsxcommon`.
- `apps/lsxforge/src/main.cpp` — `+ --mimo` flag and dispatch branch
  (`lsxcommon::mimo::ConvertMiMoModel`).
- `tests/CMakeLists.txt` — new targets; `tests/perf/perf_baselines.json` —
  new `mimo-v2.6-flash-rl` entry plus an additive `PerfModel` + `TEST_CASE`
  in the shared `tests/test_perf_baseline.cpp` harness.
- Guard files: `test_code_naming_guard.cpp` (family token `mimo` +
  `AllowedCommon` row `mimo_tool_format`), `test_model_independence_guard.cpp`
  (passport row, owner marker `namespace frozen {`, consumer row, cross-family
  scan extended to `mimo_moe_*`), `test_model_suite_guard.cpp` (suite row +
  `mimo` in `FamilyStems()` + model-stage regex), `test_ctest_labels_guard.cpp`
  (labels only).
- `AGENTS.md` Structure section; `.opencode/skills/logestix-repo-map/SKILL.md`
  per-file inventory.

There is no central format-tag registry in the repo; the format tag is
validated inside the family loader (`out.format != "logestix-mimo-nvfp4"`),
matching GigaChat/Granite.

**Never touched:** every `glm_*` / `qwen_*` / `deepseek_*` / `gemma_*` /
`kimi_*` / `nemotron_*` / `gigachat_*` / `ornith_*` / `granite_*` source,
header and test; `gpu_perf.*` stays family-neutral; the shared
`qwen_inference.*` / `lsx_kernels_*` neutral kernel layer is read-only to this
work.

**Verification (empirical):**

- Baseline before implementation: `test_code_naming_guard`,
  `test_model_independence_guard`, `test_model_suite_guard`,
  `test_ctest_labels_guard` all green.
- After: same four green, plus
  `grep -rn '#include' libs/lsxcommon/src/mimo_moe_*` shows no foreign family
  header, and `grep -rln mimo` outside `mimo*` files lists only the shared
  registration surfaces above.

## Source topology and sharding

- `model_pp0_epN_shard0.safetensors`, `N = 0..63`. Each shard `N` holds
  experts `4N..4N+3` for every one of the 47 MoE layers (1128 tensors);
  `ep0` additionally holds all backbone dense tensors and the multimodal
  payload. The converter streams all 64 shards.
- `model_mtp.safetensors` — MTP layers `model.mtp.layers.{0,1,2}` (3 layers),
  stored + counted, never applied.
- `config.json`, `tokenizer.json`, `vocab.json`, `merges.txt`,
  `tokenizer_config.json`, `chat_template.jinja`.

## Tensor inventory (audited 2026-09-30)

Non-layer tensors:

| Tensor | dtype | shape |
|---|---|---|
| `model.embed_tokens.weight` | BF16 | [152576, 4096] |
| `lm_head.weight` | BF16 | [152576, 4096] (untied) |
| `model.norm.weight` | BF16 | [4096] |

Per backbone layer (0–47), attention is **fused qkv** (`attention_projection_layout
= "fused_qkv"`):

| Tensor | GA layers (9) | SWA layers (39) |
|---|---|---|
| `self_attn.qkv_proj.weight` F8_E4M3 | [13568, 4096] | [14848, 4096] |
| `self_attn.qkv_proj.weight_scale_inv` F32 | [108, 32] | [116, 32] |
| `self_attn.o_proj.weight` BF16 | [4096, 8192] | [4096, 8192] |
| `self_attn.attention_sink_bias` BF16 | — | [64] |

QKV fused layout: `Q = 64·192 = 12288`, `K = kv_heads·192`, `V = kv_heads·128`;
GA `kv_heads = 4` → `12288 + 768 + 512 = 13568`; SWA `kv_heads = 8` →
`12288 + 1536 + 1024 = 14848`. `o_proj` input is `64·128 = 8192`.

Per layer MLP:

- Layer 0 (dense, `moe_layer_freq[0] == 0`): `mlp.gate_proj` / `mlp.up_proj`
  F8_E4M3 [16384, 4096] + `weight_scale_inv` F32 [128, 32]; `mlp.down_proj`
  F8_E4M3 [4096, 16384] + `weight_scale_inv` F32 [32, 128].
- Layers 1–47 (MoE): `mlp.gate.weight` BF16 [256, 4096],
  `mlp.gate.e_score_correction_bias` F32 [256], and per expert
  `mlp.experts.E.{gate,up}_proj.weight` U8 [2048, 2048] +
  `.weight_scale` U8 [2048, 128]; `down_proj.weight` U8 [4096, 1024] +
  `.weight_scale` U8 [4096, 64]. **12032 routed experts total** (47 · 256).

MTP layers 0–2 (stored + counted, never resident): `self_attn.qkv_proj` FP8 +
`weight_scale_inv`, `self_attn.o_proj` BF16, `self_attn.attention_sink_bias`
BF16 [64], `mlp.{gate,up,down}_proj` FP8 + `weight_scale_inv`.

Multimodal ballast (**stripped**, never converted, never counted as model
weights): `visual.blocks.*`, `audio_encoder.*`, `speech_embeddings.*`,
`audio_tokenizer/*`.

### Checkpoint quirk: GA `qkv_proj` scale row count

The GA `qkv_proj.weight` has 13568 rows (`ceil(13568/128) = 106` scale rows),
but `weight_scale_inv` is stored with **108** rows (2 spare rows). The
dequantizer indexes `scale[r/128, c/128]`, so only rows 0–105 are ever read;
the tail is ignored. SWA matches exactly (14848/128 = 116). A synthetic
converter test pins this (weight rows need not equal `scale_rows·128`).

## Frozen geometry (compile-time)

Locked in `libs/lsxcommon/src/mimo_moe_constants.h` under
`lsxcommon::mimo::frozen` with `static_assert`; `MiMoV26FlashModel::Load`
cross-checks the parsed index against it. Escape hatch for tests:
`LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY` (runtime env, repo policy).

| Constant | Value | Source |
|---|---|---|
| kType | `"mimo-v2.6-flash-rl"` | registry id |
| kHidden | 4096 | `hidden_size` |
| kLayers | 48 | `num_hidden_layers` |
| kVocab | 152576 | `vocab_size` |
| kDenseIntermediate | 16384 | `intermediate_size` |
| kMoeIntermediate | 2048 | `moe_intermediate_size` |
| kNumRouted | 256 | `n_routed_experts` |
| kTopK | 8 | `num_experts_per_tok` |
| kRmsEps | 1e-6f | `layernorm_epsilon` |
| kFullAttnLayers | {0,5,11,17,23,29,35,41,47} (rest SWA) | `hybrid_layer_pattern` |
| kMoeLayers | 1..47 | `moe_layer_freq` (index 0 dense) |
| kMtpLayers | {0,1,2} (namespaced `model.mtp.`) | `num_nextn_predict_layers` |

Attention (GA / SWA):

| Constant | GA | SWA | Formula |
|---|---|---|---|
| kHeads | 64 | 64 | `num_attention_heads` / `swa_num_attention_heads` |
| kKVHeads | 4 | 8 | `num_key_value_heads` / `swa_num_key_value_heads` |
| kQkHeadDim | 192 | 192 | `head_dim` / `swa_head_dim` |
| kVHeadDim | 128 | 128 | `v_head_dim` / `swa_v_head_dim` |
| kRopeDim | 64 | 64 | `int(head_dim · partial_rotary_factor)` |
| kRopeTheta | 10000000.0 | 10000.0 | `rope_theta` / `swa_rope_theta` |
| kSlidingWindow | — | 128 | `sliding_window` |
| kSinkBias | absent | present `[64]` | `add_swa_attention_sink_bias` |
| kQSize / kKSize / kVSize | 12288 / 768 / 512 | 12288 / 1536 / 1024 | heads·dims |
| kOSize | 8192 | 8192 | `heads · v_head_dim` |
| kAttnScale | `192^-0.5` | `192^-0.5` | `head_dim^-0.5` |
| kValueScale | 0.707f | 0.707f | `attention_value_scale` |

Router / MLP:

| Constant | Value |
|---|---|
| kScoringFunc | sigmoid |
| kTopkMethod | `noaux_tc` |
| kNGroups / kTopkGroup | 1 / 1 (identity group) |
| kNormTopkProb | true (`w /= w.sum + 1e-20`) |
| kRoutedScale | 1.0 (`routed_scaling_factor` null) |
| kNoSharedExperts | true |
| kActivation | silu (SwiGLU `silu(gate)·up`) |

Special tokens (read from `tokenizer_config.json` / `tokenizer.json`, never
hardcoded in kernels):

| Constant | Value |
|---|---|
| kEosId | 151645 (`<|im_end|>`) |
| kPadId | 151643 (`<|endoftext|>`) |
| kBosId | -1 (no BOS) |
| kImStartId | 151644 (`<|im_start|>`) |
| kThinkStart / kThinkEnd | text markers `<think>` / `</think>` (BPE, not added tokens) |

## Pinned reference formulas

### RMSNorm

```
n = x.float(); n = n * rsqrt(n.pow(2).mean(-1, keepdim=True) + eps)
return (w * n).to(x.dtype)
```

### Attention (identical for GA / SWA apart from the noted flags)

```
qkv = x @ Wqkv^T                                  # fused, FP8-dequantized / NVFP4
q = qkv[:12288].view(64, 192)
k = qkv[12288:12288+k_size].view(kv_heads, 192)
v = qkv[12288+k_size:].view(kv_heads, 128)
v = v * 0.707                                     # attention_value_scale
qr, qn = q.split([64, 128], -1); kr, kn = k.split([64, 128], -1)
(qr, kr) = hf_rotate_half_rope(qr, kr, theta)     # theta = GA 1e7 / SWA 1e4
q = cat(qr, qn); k = cat(kr, kn)
k = repeat_kv(k, 64/kv_heads); v = repeat_kv(v, 64/kv_heads)
s = q @ k^T * 192^-0.5                            # F32
s = causal_mask(s) | sliding_window_mask(s, 128)  # SWA only
if SWA:                                           # learned attention sink
    s = cat([s, sink_bias.view(1,64,1).expand(...)], -1)
s = s - s.max(-1, keepdim=True)
p = softmax(s, -1, dtype=F32)
if SWA: p = p[..., :-1]
o = p @ v   -> flatten(8192) -> o_proj
```

### Router (`noaux_tc`, `n_group = 1`)

```
logits  = x.float() @ gate.weight.float().T        # F32
scores  = sigmoid(logits)
choice  = scores + e_score_correction_bias          # group step is identity
idx     = topk(choice, 8)
w       = scores.gather(1, idx)
w       = w / (w.sum(-1, keepdim=True) + 1e-20)     # norm_topk_prob
routed  = sum_e w_e * down_e( silu(gate_e(x)) * up_e(x) )
```

### Decoder layer

```
r = x; x = input_layernorm(x); x = self_attn(x); x = r + x
r = x; x = post_attention_layernorm(x); x = mlp(x); x = r + x
```

### Final

```
h = model.norm(h); logits = lm_head(h)
```

## Converted layout: `logestix-mimo-nvfp4`

Output under the central `model-metadata/` directory through the single path
authority `lsx_model_metadata(rel)`; no hardcoded paths in the converter. The
index (`logestix/model.index.json`):

```
format: "logestix-mimo-nvfp4"
model:  "MiMo-V2.6-Flash-RL"
num_layers: 48
layer_types: ["full_attention"|"sliding_attention"] x 48
swa_layers: [1,2,3,4,6,7,...]   # 39 entries
geometry: { hidden, vocab, heads, kv_heads, kv_heads_swa, qk_head_dim,
            v_head_dim, rope_dim, num_experts, top_k, moe_intermediate,
            dense_intermediate, sliding_window }
vram: { min_bytes, optimal_bytes, max_bytes, context_tokens: 1048576 }
files: { pinned: { global, layer_000..layer_047 },
         experts: layer_N/exp_E.arrow, mtp }
deferred: { mtp: "stored", mtp_layers: [0,1,2] }
```

### Globals (`pinned/global.arrow`)
- `embed.weight` BF16 [152576, 4096]
- `head.weight` BF16 [152576, 4096] (untied)
- `norm.weight` BF16 [4096]

### Layer (`pinned/layer_NNN.arrow`)
- `input_layernorm.weight`, `post_attention_layernorm.weight` BF16 [4096]
- `qkv_proj.nvfp4` [13568|14848, 4096] (FP8 dequant → neutral NVFP4)
- `o_proj.weight` BF16 [4096, 8192] (pass-through verbatim; not quantized)
- SWA only: `attention_sink_bias` BF16 [64]
- dense layer 0: `mlp.{gate,up,down}_proj.nvfp4` (FP8 dequant → neutral NVFP4)
- MoE layers: `mlp.gate.weight` BF16 [256, 4096],
  `mlp.gate.e_score_correction_bias` F32 [256]

### Routed experts (`experts/layer_NNN/exp_EEE.arrow`)
Per-expert NVFP4 via `ExpertSchema` (`PackedExpert` slots: w1=gate, w3=up,
w2=down). 12032 experts total.

### MTP (`mtp/*`)
`model.mtp.layers.{0,1,2}` converted + stored (mirroring the backbone layer
layout), counted in the index, never loaded resident and never applied.

Norms, router `gate.weight`, `e_score_correction_bias`, `attention_sink_bias`,
embeddings and `lm_head` are never quantized (MODEL INGEST step 2).

## Numeric conversion pipeline

1. **Dense FP8** (`qkv_proj`, layer-0 `{gate,up,down}_proj`): read `weight`
   (F8_E4M3) + `weight_scale_inv` (F32, 128×128 tiles) →
   `DequantFp8Block128ToBf16Gpu` (MiMo-owned `mimo_moe_dequant.cu`, clamping the
   partial/spare tile rows and tolerating `scale_rows·128 ≥ weight_rows`) →
   neutral `gpu::quantize_tensor_gpu(bf16, "BF16")` → `DenseNvfp4Schema`.
2. **Routed experts** (MXFP4 `U8` nibbles + `U8` E8M0 per-32 scales): neutral
   `PackExpertNVFP4(ExpertData)` — the exact power-of-two repack that splits
   each 32-block into two identical per-16 halves and folds a pow2 shift into
   the stored    `global_scale` (`converter.cpp:443-475`, pinned by
   `test_deepseek_moe_convert`). **Fallback (expected to fire):**
   `ChooseE8m0ScaleShift` returns false when the per-tensor nonzero E8M0
   exponent spread exceeds 13 (`nvfp4_scale.h:92-98`) — plausible for the
   wide dynamic range of an FP4 expert matrix. On that failure MiMo's
   converter dequantizes that expert (`mimo_moe_dequant.cu`: packed E2M1
   nibbles · E8M0 per-32 → BF16) and requantizes via `quantize_tensor_gpu`.
   Both the exact-repack and the fallback paths are pinned synthetically.
3. **BF16/F32 pass-through:** embed, `lm_head`, norms, `o_proj`, router
   `gate.weight`, `e_score_correction_bias`, `attention_sink_bias` — verbatim,
   safetensors dtype preserved.

Conversion is strict: multimodal tensors and `model.mtp.*` are the only
expected non-backbone tensors; any unknown tensor is an error, never silently
ignored.

## Inference data flow and state

`embed → for layer in 0..47: (GA | SWA) attention + MLP (dense | MoE) → final
norm → lm_head (F32) → sample`.

Per-session device state:

- **KV cache**: 48 layers. GA (9): full-context `[kv_heads=4]` with K 192 /
  V 128. SWA (39): ring buffer bounded to `sliding_window = 128` with
  `kv_heads = 8`. Both K and V cached pre-value-scale (scale applied at use).
- **Expert cache**: host mmap of 12032 experts + device slot-LRU
  (`mimo_moe_expert_cache.{h,cpp}`, `expert_store` primitive; knob
  `LSX_MIMO_EXPERT_CACHE_SLOTS`). Dense backbone resident on the GPU; experts
  streamed. Dense resident ≈ 8 GiB (embed/head 2.5 GiB, `o_proj` BF16
  3.2 GiB, NVFP4 qkv + dense MLP ≈ 1.8 GiB, norms/gates/sinks); the ~12032
  experts ≈ 170 GiB NVFP4 on host.
- Single CUDA stream, no per-token `cudaDeviceSynchronize`; reductions use the
  shared-memory sandbox (thread-0 + barriers) per
  `nvfp4-blackwell-conventions`.

## Prompt framing, tokenizer and tool bridge (stages 4 + 4b)

- Tokenizer: `BuildTokenizerArrowFiles(tokenizer.json)` (HF BPE; merges may be
  `"a b"` or `["a","b"]`). `bos_id = -1`, `eos_id = 151645`, `pad_id = 151643`
  read from the config/tokenizer, never hardcoded in kernels.
- `BuildPromptInputIds` renders the verified ChatML frame:
  `<|im_start|>{role}\n{content}<|im_end|>`; assistant turns as
  `<|im_start|>assistant\n<think>{reasoning}</think>{content}` followed by the
  `<tool_call>…</tool_call>` block when the turn carries tool calls, then
  `<|im_end|>`; generation prompt `<|im_start|>assistant\n` (plus
  `<think></think>` when `enable_thinking` is false). Control markers injected
  as their own vocab ids via `arrow_tokenizer_lookup`; free text via
  `arrow_tokenizer_encode`.
- **Tool bridge** (`mimo_tool_format.{h,cpp}`, `AllowedCommon` row): system
  tools block `You are provided with the following tools:\n\n<tools>\n{json per
  line}\n</tools>`; assistant tool calls
  `<tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>`.
  `<tool_call>` / `<function=` / `<parameter=` are **not** added tokens
  (verified against `tokenizer.json` `added_tokens`) so they survive
  `arrow_tokenizer_decode`; parse from decoded text and fall back to
  `tools::ParseToolCallsText` / `tools::ParseBareToolCallJson` when absent.
  Verified against the checkpoint's own `chat_template.jinja` before coding.
- SelfCheck uses the non-thinking frame (`<think></think>` prefilled) so
  `"The capital of France is" → "Paris"` (top-1/short prefix); if the RL model
  does not honour the empty-`<think>` prefill, the deviation is recorded and a
  `substring_match` after `</think>` is the fallback.

## File map and build

```
libs/lsxcommon/src/mimo_moe_constants.h
libs/lsxcommon/src/mimo_moe_dequant.cu
libs/lsxcommon/src/mimo_moe_kernels.{h,cu}
libs/lsxcommon/src/mimo_moe_load.cpp
libs/lsxcommon/src/mimo_moe_model.cpp
libs/lsxcommon/src/mimo_moe_expert_cache.cpp
libs/lsxcommon/src/mimo_moe_inference_attn.cu
libs/lsxcommon/src/mimo_moe_inference_pipeline.cu
libs/lsxcommon/src/mimo_moe_device_math.h
libs/lsxcommon/include/lsxcommon/mimo_moe_convert.h
libs/lsxcommon/include/lsxcommon/mimo_moe_load.h
libs/lsxcommon/include/lsxcommon/mimo_moe_model.h
libs/lsxcommon/include/lsxcommon/mimo_moe_expert_cache.h
libs/lsxcommon/include/lsxcommon/mimo_moe_inference_attn.h
libs/lsxcommon/include/lsxcommon/mimo_moe_inference_pipeline.h
libs/lsxcommon/include/lsxcommon/mimo_tool_format.h
libs/lsxcommon/src/mimo_moe_convert.cpp
libs/lsxcommon/src/mimo_tool_format.cpp
tests/test_mimo_moe_fixture.h
tests/test_mimo_moe_convert.cpp
tests/test_mimo_moe_loader.cpp
tests/test_mimo_moe_model.cpp
tests/test_mimo_moe_kernels.cu
tests/test_mimo_moe_selfcheck.cpp
tests/test_mimo_moe_fsi_selfcheck.cpp
tests/test_mimo_moe_code_selfcheck.cpp
tests/test_mimo_moe_det_selfcheck.cpp
tests/test_mimo_tool_framing.cpp
```

(`perf` is a `TEST_CASE` inside the shared `tests/test_perf_baseline.cpp`,
which already carries one case per vault model; MiMo adds a `kMiMo` entry and
its `TEST_CASE`.)

- New OBJECT library `lsxcommon_mimo_cu` (`-fmad=false`, bit-exact math),
  linked PUBLIC into `lsxcommon`.
- New `lsxforge` flag `--mimo` (`lsxcommon::mimo::ConvertMiMoModel`).
- Register `MiMoV26FlashModel` in `model_registry.cpp`.
- Tokenizer bridge, passport (`passport/passport.bin` via
  `lsxnetwork::BuildPassport`) and `LsxTokenTranslator` in `IoWorker` after
  affinity pinning, before readiness.

## Guard registries (same commit)

- `test_code_naming_guard.cpp`: add `mimo` to `fam` and `FamilyStems()`; add
  `mimo_tool_format` to `AllowedCommon()` (family-prefixed, no type marker).
  No new role/subsystem/variant tokens are needed (`convert`, `dequant`,
  `load`, `model`, `inference_{attn,pipeline}`, `kernels`, `device_math`,
  `expert_cache`, `constants` all exist).
- `test_model_independence_guard.cpp`: `kPassports` row
  `{"mimo-v2.6-flash-rl", "mimo_moe_constants.h", "lsxcommon::mimo::frozen"}`;
  owner marker `namespace frozen {`; `mimo_moe_model.cpp` in the per-`Infer`
  `LsxFastMathScope` consumers list; extend the cross-family-borrowing scan to
  cover `mimo_moe_*`.
- `test_model_suite_guard.cpp`: `Suites()` row
  `convert → load → model → kernels → selfcheck → fsi → code → det` + add
  `mimo` to its `FamilyStems()` and the model-stage regex.
- `test_ctest_labels_guard.cpp`: labels on every new test in the same commit.

## Testing strategy (TDD)

Strict red→green per stage; every new target gets its ctest label in the same
commit; every selfcheck/fsi/code/det target gets `SKIP_RETURN_CODE 4`.

1. **fast/host** — passport `static_assert`s; `kType` resolution; frozen-
   geometry rejection on mismatch; layer-type decoding (9 GA / 39 SWA);
   `kAttnScale == 192^-0.5`; rope-dim = 64; router reference math (sigmoid +
   `e_score_correction_bias` + `norm_topk_prob`) and the sink-biased softmax in
   `mimo_moe_device_math.h`; `mimo_tool_format` render/parse + tokenizer
   round-trip on a synthetic tokenizer fixture.
2. **convert** (label `convert`, bounded TIMEOUT ≤120 s,
   `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1`): synthetic E2E on
   `test_synth_checkpoint.h` covering globals + one GA + one SWA + the dense
   layer + one MoE layer (all 256 experts), an FP8 tensor with a mismatched
   scale-row count (GA quirk), an MXFP4 expert repack and an MXFP4
   shift-failure fallback; assert the `logestix-mimo-nvfp4` index, NVFP4 scale
   geometry, untouched high-precision tensors and the counted MTP record. No
   real-model conversion in tests.
3. **load** — synthetic converted fixture only; format-tag gate, frozen-
   geometry gate, missing/corrupt tensor diagnostics, expert LRU bookkeeping.
4. **kernels** — device vs host reference parity, bit-exact with `-fmad=false`:
   RMSNorm, SwiGLU dense + expert MLP, sigmoid `noaux_tc` router, fused-QKV
   split, partial rotate-half RoPE (GA + SWA theta), GQA repeat, causal +
   sliding-window mask, sink-biased softmax, `v_scale`, FP8 block-128 dequant
   (with mismatched/partial tiles), MXFP4 E2M1·E8M0 → BF16.
5. **selfcheck** — incremental 1→2-token `"Paris"` over ≥3 cycles
   (`LSX_TEST_MIMO_SELFCHECK_CYCLES`), bit-exact `final_logits`, real model via
   env (`LSX_TEST_MIMO_ROOT`), `SKIP()` when absent.
6. **fsi / code / det** — the model-neutral harnesses (`fsi_selfcheck`,
   `code_selfcheck`, `det_selfcheck`) captured from the source safetensors,
   per-family goldens under `tests/fixtures/{fsi,code,det}/mimo.json`.

Perf: `tests/perf/perf_baselines.json` entry for `mimo-v2.6-flash-rl` plus a
`TEST_CASE("perf baseline: mimo-v2.6-flash-rl", "[perf]")` in the shared
`tests/test_perf_baseline.cpp` harness, measured after correctness at the
canonical protocol.

## Error handling and strictness

- Load rejects a non-`logestix-mimo-nvfp4` index, a geometry mismatch vs the
  frozen passport, and any missing / mis-shaped / corrupt tensor with a
  diagnostic naming the tensor.
- Conversion is strict: multimodal payload and MTP are the only expected
  non-backbone tensors (MTP counted, multimodal stripped); any other unknown
  tensor is an error.
- `SelfCheck` returns `selfcheck_passed=false` with the decoded token on
  failure; no partial-success state.

## Risks / decision points

- **Primary:** the attention surface is net-new for the repo — asymmetric
  QK/V dims, partial rotate-half RoPE on the first 64 dims, GA/SWA mixing,
  sliding-window ring cache and the learned sink bias are each pinned against
  a host reference before the layer pipeline is wired.
- The GA `qkv_proj` scale has more rows than `weight/128`; the dequant must
  index by `r/128` and tolerate the tail (synthetic test).
- MXFP4 → NVFP4 exact repack may fail `ChooseE8m0ScaleShift` on some experts;
  the documented dequant+requant fallback must be implemented and tested.
- Real conversion of 172 GB across 64 shards is a long offline step (hours,
  ~170 GB output); it is not run inside tests, and the real-weight SelfCheck
  SKIPs until it is done.
- Single RTX 5090 (32 GiB): dense resident (~8 GiB) + expert streaming; t/s is
  expected to be modest and is recorded as measured, not targeted.
- Empty `<think></think>` prefill for the `"Paris"` control prompt may not be
  honoured; fallback and deviation are recorded in stage 8.
- The shipped `modeling_mimo_v2.py` / `configuration_mimo_v2.py` /
  `chat_template.jinja` are used as read-only references and are NOT committed
  to the repository.

## Out of scope (follow-ups)

MTP speculative decoding; multimodal (vision / audio) ingestion; the
1M-context prefill optimization; extending the shared FP8 block-128 dequant
(GLM-Flash / GigaChat / MiMo) into one neutral infrastructure primitive; an
optional NVFP4 quantization of `o_proj` to reduce resident VRAM.
