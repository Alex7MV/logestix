# AliceAI-Foundation-80B-A3B-Base — Ingestion Design

Date: 2026-10-01
Status: approved design, pending implementation plan
Branch: `feat/alice-foundation-80b-a3b`

## Overview

Ingest `AliceAI-Foundation-80B-A3B-Base` (`AliceAIForCausalLM`,
`model_type=alice_ai`) into the logestix monorepo following the 8-stage Model
Ingestion Protocol plus the mandatory stage 4b tool-calling bridge. This is a
**new model family** (`alice`): a **hybrid KDA + gated-attention sparse MoE**
backbone with a novel **block attention-residual (AttnRes)** depth mixer, a
sigmoid router with correction bias, a shared expert, a per-channel state decay,
and a **SentencePiece-BPE** tokenizer (a first in the repo).

Source checkpoint (read-only):
`/mnt/llm_models/models/AliceAI-Foundation-80B-A3B-Base`
(config + 49 BF16 safetensors shards + SentencePiece `tokenizer.model` +
`chat_template.jinja`; 152 GB total, 80B total / ~3B activated params,
262,144-token context).

Milestone scope (operator decision): **full 8-stage ingestion**, including
runnable inference on the single RTX 5090 and a real-weight SelfCheck. The MTP
head (1 layer) is **structural only** (converted + stored + counted, never
applied at inference). Real conversion of 152 GB is a long offline step and is
never run inside tests.

Reference math is pinned from the shipped, read-only `modeling_alice_ai.py` /
`configuration_alice_ai.py` / `chat_template.jinja`; `config.json` is
authoritative where it differs from Python defaults.

## Autonomy invariant (first-class, operator directive)

The model MUST be autonomous relative to every other model in the repo: adding
or tuning it never changes another model's geometry or runtime behaviour. This
is the same invariant GigaChat / Nemotron / Granite / MiMo satisfy and it is
machine-enforced by the four guard registries; it is extended in the SAME
commit as the model files.

| Rule | Alice compliance |
|---|---|
| Unique frozen passport in a model-unique namespace | `libs/lsxcommon/src/alice_moe_constants.h` in `lsxcommon::alice::frozen` |
| Passport is a leaf (includes no other passport) | no cross-includes; enforced by `no passport includes another model's passport` |
| No cross-family geometry borrowing | `alice_moe_*` includes no other family header; the runtime is an Alice-owned TU that mirrors, not includes, the qwen/GigaChat/Nemotron precedents |
| Test suites are family-isolated | `test_alice_moe_*` include only `test_alice_moe_fixture.h` + the neutral `test_synth_checkpoint.h` |
| Per-`Infer` fast-math policy | `LsxFastMathScope(EffectiveFastMathMode(cfg_))` in `AliceFoundation80BModel::Infer`; row added to the guard's `consumers` list |
| Neutral infrastructure only | `lsx_inference_nvfp4`, `lsx_gemv`, `PackExpertNVFP4`, `DenseNvfp4Schema`/`ExpertSchema`, `expert_store`, `schema`, `safetensors_gpu`, `converter`, `model_base`/`model_runtime`, `tokenizer`, `path_registry`, `vram_estimate`, `gpu_perf`, `sampling`, `resource_governor` |
| Family-scoped kernel selection | no process-global toggle is added; all NVFP4 math goes through the single neutral `lsx_nvfp4` layer |

**Shared registration surfaces (additive, one row/branch each — never a
behavioural edit to another model):**

- `libs/lsxcommon/src/model_registry.cpp` — `+ Register<AliceFoundation80BModel>()`.
- `libs/lsxcommon/CMakeLists.txt` — new `lsxcommon_alice_cu` OBJECT library +
  source entries + link into `lsxcommon`.
- `apps/lsxforge/src/main.cpp` — `+ --alice` flag and dispatch branch
  (`lsxcommon::alice::ConvertAliceModel`).
- `tests/CMakeLists.txt` — new targets; `tests/perf/perf_baselines.json` — new
  `alice-foundation-80b-a3b` entry plus an additive `PerfModel` + `TEST_CASE`
  in the shared `tests/test_perf_baseline.cpp` harness.
- Guard files: `test_code_naming_guard.cpp` (family token `alice` +
  `AllowedCommon` row `alice_tool_format`), `test_model_independence_guard.cpp`
  (passport row, owner marker `namespace frozen {`, consumer row, cross-family
  scan extended to `alice_moe_*`), `test_model_suite_guard.cpp` (suite row +
  `alice` in `FamilyStems()` + model-stage regex), `test_ctest_labels_guard.cpp`
  (labels only).
- `AGENTS.md` Structure section; `.opencode/skills/logestix-repo-map/SKILL.md`
  per-file inventory.

There is no central format-tag registry in the repo; the format tag is
validated inside the family loader (`out.format != "logestix-alice-nvfp4"`),
matching GigaChat/Granite/MiMo.

**Never touched:** every `glm_*` / `qwen_*` / `deepseek_*` / `gemma_*` /
`kimi_*` / `nemotron_*` / `gigachat_*` / `ornith_*` / `granite_*` / `mimo_*` /
`phi_*` source, header and test; `gpu_perf.*` stays family-neutral; the shared
`lsx_kernels_*` neutral kernel layer is read-only to this work (read/extended
only additively if a genuinely neutral primitive is missing).

**Verification (empirical):**

- Baseline before implementation: `test_code_naming_guard`,
  `test_model_independence_guard`, `test_model_suite_guard`,
  `test_ctest_labels_guard` all green.
- After: same four green, plus
  `grep -rn '#include' libs/lsxcommon/src/alice_moe_*` shows no foreign family
  header, and `grep -rln alice` outside `alice*` files lists only the shared
  registration surfaces above (plus the model-neutral tokenizer edits).

## Source topology and sharding

- `model-00001..00049-of-00049.safetensors`, BF16, `__metadata__ = {"format":
  "pt"}`. `model.safetensors.index.json` maps 1307 tensors (`weight_map`).
- The 49 shards are a plain sequential split (no expert/ep parallelism): each
  layer's tensors are contiguous, so the converter streams shards in order and
  never seeks backwards.
- `config.json`, `tokenizer.model` (SentencePiece BPE, 2.5 MB),
  `tokenizer_config.json` (`LlamaTokenizer`, `add_bos_token: true`),
  `special_tokens_map.json`, `chat_template.jinja`.
- No `tokenizer.json` / `vocab.json` / `merges.txt` ship with the checkpoint.

## Tensor inventory (audited 2026-10-01)

Non-layer tensors:

| Tensor | dtype | shape |
|---|---|---|
| `model.embed_tokens.weight` | BF16 | [129024, 2048] |
| `lm_head.weight` | BF16 | [129024, 2048] (untied) |
| `model.norm.weight` | BF16 | [2048] (zero-centered) |
| `model.attnres_final.res_proj.weight` | BF16 | [1, 2048] |
| `model.attnres_final.res_norm_weight` | BF16 | [2048] |

Per **KDA layer** (36 layers: `layer_types[i] == "linear_attention"`, i.e.
`i % 4 != 3`):

| Tensor | dtype | shape |
|---|---|---|
| `input_layernorm.weight` / `post_attention_layernorm.weight` | BF16 | [2048] (ZC) |
| `linear_attn.q_proj.weight` / `.k_proj.weight` / `.v_proj.weight` | BF16 | [4096, 2048] |
| `linear_attn.q_conv1d.weight` / `.k_conv1d.weight` / `.v_conv1d.weight` | BF16 | [4096, 1, 4] |
| `linear_attn.f_a_proj.weight` | BF16 | [128, 2048] |
| `linear_attn.f_b_proj.weight` | BF16 | [4096, 128] |
| `linear_attn.g_a_proj.weight` | BF16 | [128, 2048] |
| `linear_attn.g_b_proj.weight` | BF16 | [4096, 128] |
| `linear_attn.b_proj.weight` | BF16 | [32, 2048] |
| `linear_attn.a_log_bias` | BF16 | [32] |
| `linear_attn.dt_bias` | BF16 | [4096] |
| `linear_attn.o_norm.weight` | BF16 | [128] (not zero-centered) |
| `linear_attn.o_proj.weight` | BF16 | [2048, 4096] |
| `mlp.gate.weight` | BF16 | [512, 2048] |
| `mlp.gate.e_score_correction_bias` | BF16 | [512] |
| `mlp.experts.gate_up_proj` | BF16 | [512, 1024, 2048] (fused gate+up, per expert) |
| `mlp.experts.down_proj` | BF16 | [512, 2048, 512] (per expert) |
| `mlp.shared_expert.{gate,up}_proj.weight` | BF16 | [512, 2048] |
| `mlp.shared_expert.down_proj.weight` | BF16 | [2048, 512] |
| `mlp.shared_expert_gate.weight` | BF16 | [1, 2048] |
| `mlp_res_proj.weight` / `mlp_res_norm_weight` | BF16 | [1, 2048] / [2048] |
| `attn_res_proj.weight` / `attn_res_norm_weight` (layers ≥ 1 only) | BF16 | [1, 2048] / [2048] |

Per **full-attention (gated) layer** (12 layers: `i % 4 == 3`):

| Tensor | dtype | shape |
|---|---|---|
| `input_layernorm.weight` / `post_attention_layernorm.weight` | BF16 | [2048] (ZC) |
| `self_attn.q_proj.weight` | BF16 | [8192, 2048] (2·16·256: query + output gate) |
| `self_attn.k_proj.weight` / `.v_proj.weight` | BF16 | [512, 2048] |
| `self_attn.o_proj.weight` | BF16 | [2048, 4096] |
| `self_attn.q_norm.weight` / `.k_norm.weight` | BF16 | [256] (ZC) |
| MLP + AttnRes tensors | | identical to the KDA layer |

**Layer 0 is special:** it has **no** `attn_res_proj.weight` /
`attn_res_norm_weight` (the first layer reads `embeddings` directly). All other
layers 1–47 carry them. The converter and loader must treat their absence at
layer 0 as expected, not an error.

**MTP (stored + counted, never resident):** `mtp.fc.weight` BF16 [2048, 4096]
(embedding ∥ hidden → hidden), `mtp.pre_fc_norm_embedding.weight`,
`mtp.pre_fc_norm_hidden.weight`, `mtp.norm.weight` [2048], and one full-attention
decoder block `mtp.layers.0.*` (q/k/v/o + q_norm/k_norm, MoE gate/experts/
shared expert, input/post norms). Note `mtp.layers.0` has **no AttnRes**
tensors — MTP uses a plain residual, so it is mirrored with the AttnRes fields
left empty.

### Checkpoint quirks

- `dt_bias` is **per channel** `[4096]` and `a_log_bias` is **per key head**
  `[32]` — not per key-head-dimension as in Qwen3.8-Flash. The KDA gate is
  `-exp(a_log_bias[h]) · softplus(alpha[c] + dt_bias[c])` broadcast over
  `[head_k_dim]`.
- The routed-expert weights are stored **fused and stacked**:
  `gate_up_proj[E, 2·I, H]` and `down_proj[E, H, I]`, one tensor per layer.
  There is no per-expert tensor prefix.
- `linear_num_value_heads == linear_num_key_heads == 32` → KDA repeat factor is
  1 (no query/key repeat-interleave).
- All tensors are BF16; there is no FP8/MXFP4 dequant path.

## Frozen geometry (compile-time)

Locked in `libs/lsxcommon/src/alice_moe_constants.h` under
`lsxcommon::alice::frozen` with `static_assert`; `AliceFoundation80BModel::Load`
cross-checks the parsed index against it. Escape hatch for tests:
`LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY` (runtime env, repo policy).

| Constant | Value | Source |
|---|---|---|
| kType | `"alice-foundation-80b-a3b"` | registry id |
| kHidden | 2048 | `hidden_size` |
| kLayers | 48 | `num_hidden_layers` |
| kVocab | 129024 | `vocab_size` |
| kFullAttnInterval | 4 (`i%4==3`) | `layer_types` |
| kBlockAttnRes | 4 | `block_attn_res_block_size` |
| kRmsEps | 1e-6f | `rms_norm_eps` |
| kHeads | 16 | `num_attention_heads` |
| kKVHeads | 2 | `num_key_value_heads` |
| kHeadDim | 256 | `head_dim` |
| kRotaryDim | 64 | `head_dim · partial_rotary_factor (0.25)` |
| kRopeTheta | 1000000.0 | `rope_theta` |
| kAttnScale | `256^-0.5 == 0.0625` | `head_dim^-0.5` |
| kLinearKeyHeads / kLinearValueHeads | 32 / 32 | `linear_num_*_heads` |
| kLinearKeyHeadDim / kLinearValueHeadDim | 128 / 128 | `linear_*_head_dim` |
| kLinearConvKernel | 4 | `linear_conv_kernel_dim` |
| kLinearKvRepeat | 1 | `32 / 32` |
| kNumExperts | 512 | `num_experts` |
| kTopK | 10 | `num_experts_per_tok` |
| kMoeIntermediate | 512 | `moe_intermediate_size` |
| kSharedIntermediate | 512 | `shared_expert_intermediate_size` |
| kMtpLayers | 1 (`mtp.*` prefix) | `mtp_num_hidden_layers` |

Special tokens (read from `tokenizer_config.json` / the parsed tokenizer, never
hardcoded in kernels):

| Constant | Value |
|---|---|
| kBosId | 1 (`<s>`) |
| kEosId | 2 (`</s>`) |
| kPadId | -1 (`pad_token_id` is null) |
| kUnkId | 0 (`<unk>`, byte-fallback anchor) |

## Pinned reference formulas

### Zero-centered RMSNorm (`_rms_norm`, `zero_centered=True`)

```
n = x.float(); n = n * rsqrt(n.pow(2).mean(-1, keepdim=True) + eps)
return ((1 + w) * n).to(x.dtype)          # ZC: scale = 1 + weight
```

`linear_attn.o_norm` and the AttnRes `*_res_norm_weight` are **not**
zero-centered (`scale = weight`). `input_layernorm`,
`post_attention_layernorm`, `q_norm`, `k_norm`, `model.norm` are ZC.

### KDA (Kimi Delta Attention) linear attention

```
q = Wq·x;  k = Wk·x;  v = Wv·x
G = Wg_b·silu(Wg_a·x)                     # output gate, [.,4096]
A = Wf_b·(Wf_a·x)                         # alpha, [.,4096]
B = Wb·x                                  # beta logits, [.,32]
q = silu(conv1d_q(q)[..., :L])[..., -L:]; ... k, v        # depthwise, kernel 4
q,k -> [B,L,32,128]; v -> [B,L,32,128]; A -> [B,L,32,128]

q = l2norm(q, -1) * 128^-0.5 ;  k = l2norm(k, -1)
gate = -exp(a_log_bias[h]) * softplus(A[c] + dt_bias[c])  # [B,L,32,128]
beta = sigmoid(B)                                         # [B,L,32]

S = 0                                                     # [B,32,128,128]
for t in 0..L-1:
    S  = S * exp(gate[t])[..., None]                      # per-channel decay
    p  = einsum('bhk,bhkv->bhv', k[t], S)
    d  = (v[t] - p) * beta[t][..., None]
    S += einsum('bhk,bhv->bhkv', k[t], d)
    o[t] = einsum('bhk,bhkv->bhv', q[t], S)

o = rmsnorm(o, o_norm)                                    # NOT zero-centered
o = o * sigmoid(G.view_as(o))
y = Wo·o.flatten(-2)
```

`kda_allow_negative_eigenvalues = false` → no `beta *= 2` branch. The CUDA path
may implement chunked (prefill) and fused-recurrent (decode) forms, but must
reproduce this recurrence within the bit-exact tolerance class.

### Gated full attention

```
qg = Wq·x                                 # [., 2·16·256]
q, g = chunk(qg, 2, -1)                   # q [.,16,256], g [.,16,256]
k = Wk·x -> [.,2,256];  v = Wv·x -> [.,2,256]
q = zero_centered_rmsnorm(q, q_norm)      # over head_dim
k = zero_centered_rmsnorm(k, k_norm)
(q,k) = partial_rotate_half_rope(q, k, first 64 dims, theta=1e6)
k = repeat_kv(k, 16/2);  v = repeat_kv(v, 16/2)
s = q·kᵀ * 256^-0.5                        # F32, causal
o = softmax(s)·v ... flatten(16·256=4096)
o = o * sigmoid(g.flatten(-1))
y = Wo·o
```

### AttnRes depth mixer (`_depth_softmax_mix`)

```
stacked = stack(sources, dim=0)                     # S sources, each [...,2048]
keys    = rmsnorm(stacked, norm_weight)             # NOT zero-centered
logits  = keys · res_proj.weightᵀ                   # res_proj: Linear(2048,1)
w       = softmax(logits, dim=0)
mixed   = (stacked * w).sum(dim=0)
```

`sources` = the list of completed blocks, optionally with the running `partial`
appended last.

### Decoder layer

```
if layer_idx == 0: mixed = embeddings
else:              mixed = attn_res_mix(completed_blocks, partial)
mixed = input_layernorm(mixed)                      # ZC
attn  = KDA(mixed) | gated_attn(mixed)
partial = attn if partial is None else partial + attn
mixed = mlp_res_mix(completed_blocks, partial)
out   = partial + MoE(post_attention_layernorm(mixed))   # ZC post-norm
```

### Model forward + AttnRes blocking

```
completed_blocks = [embeddings];  partial = None
for i, layer in enumerate(layers):
    if i > 0 and i % block_attn_res_block_size (4) == 0:
        completed_blocks.append(partial); partial = None
    partial, _ = layer(embeddings, completed_blocks, partial, ...)
h = model.norm(attnres_final(completed_blocks, partial))   # ZC, 13 sources
logits = lm_head(h)
```

### Router + MoE (`AliceAISigmoidTopKRouter`)

```
logits  = x.float() @ gate.weight.float().T
scores  = sigmoid(logits)
choice  = scores + e_score_correction_bias
idx     = topk(choice, 10)
w       = scores.gather(1, idx);  w = w / w.sum(-1, keepdim=True)
routed  = Σ_e w_e · down_e( silu(gate_e(x)) · up_e(x) )
shared  = shared_expert(x) · sigmoid(shared_expert_gate(x))
out     = routed + shared
```

`experts.gate_up_proj[E]` is split into `gate = [:, :512]`, `up = [:, 512:]`.

## Converted layout: `logestix-alice-nvfp4`

Output under the central `model-metadata/` directory through the single path
authority `lsx_model_metadata(rel)`; no hardcoded paths in the converter. The
index (`logestix/model.index.json`):

```
format: "logestix-alice-nvfp4"
model:  "AliceAI-Foundation-80B-A3B-Base"
num_layers: 48
layer_types: ["linear_attention" x 3, "full_attention", ...] x 12
geometry: { hidden, vocab, layers, heads, kv_heads, head_dim, rotary_dim,
            rope_theta, linear_key_heads, linear_value_heads,
            linear_key_head_dim, linear_value_head_dim, linear_conv_kernel,
            num_experts, top_k, moe_intermediate, shared_intermediate,
            block_attn_res }
vram: { min_bytes, optimal_bytes, max_bytes, context_tokens: 262144 }
files: { pinned: { global, layer_000..layer_047 }, experts: layer_N/exp_E.arrow,
         mtp }
deferred: { mtp: "stored", mtp_layers: 1 }
```

### Globals (`pinned/global.arrow`)
- `embed.weight` BF16 [129024, 2048]
- `head.weight` BF16 [129024, 2048] (untied)
- `norm.weight` BF16 [2048] (ZC)
- `attnres_final.res_proj.weight` BF16 [1, 2048]
- `attnres_final.res_norm_weight` BF16 [2048]

### Layer (`pinned/layer_NNN.arrow`)
- `input_layernorm.weight`, `post_attention_layernorm.weight` BF16 [2048]
- `mlp_res_proj.weight` [1,2048], `mlp_res_norm_weight` [2048]; layers ≥1 add
  `attn_res_proj.weight` [1,2048], `attn_res_norm_weight` [2048]
- KDA layer: NVFP4 `q_proj`, `k_proj`, `v_proj`, `o_proj`, `f_b_proj`, `g_b_proj`;
  BF16 `q/k/v_conv1d` [4096,1,4], `f_a_proj` [128,2048], `g_a_proj` [128,2048],
  `b_proj` [32,2048], `a_log_bias` [32], `dt_bias` [4096], `o_norm` [128]
- Full-attn layer: NVFP4 `q_proj` [8192,2048], `k_proj`/`v_proj` [512,2048],
  `o_proj` [2048,4096]; BF16 ZC `q_norm`/`k_norm` [256]
- MoE: `mlp.gate.weight` BF16 [512,2048], `mlp.gate.e_score_correction_bias`
  BF16 [512], `mlp.shared_expert_gate.weight` BF16 [1,2048], NVFP4
  `shared_expert.{gate,up,down}_proj`

### Routed experts (`experts/layer_NNN/exp_EEE.arrow`)
Per-expert NVFP4 via `ExpertSchema` (`PackedExpert` slots: w1=gate, w3=up,
w2=down). **24,576 experts** (48 · 512) + 512 MTP experts.

### MTP (`mtp/*`)
`mtp.*` converted + stored (mirroring a full-attention layer), counted in the
index, never loaded resident and never applied.

Norm rules: all norms, `q_norm`/`k_norm`/`o_norm`, router `gate.weight`,
`e_score_correction_bias`, AttnRes projections and norm weights, `a_log_bias`,
`dt_bias`, `b_proj`, small `*_a_proj` (out dim 128), conv1d, shared expert gate,
embedding and `lm_head` are **never quantized** (MODEL INGEST step 2).

**Precision policy (updated after real-weight validation).** The KDA and gated
attention projections (`q/k/v/o_proj`, `f_b_proj`, `g_b_proj`) are kept in
**FP16**, not NVFP4. Quantizing them amplified a ~0.6% weight error to ~18% at
the layer-0 output through the exponential KDA gate
(`exp(-exp(a_log)·softplus(α+dt))`) and the recurrence, degrading the real 80B
model to degenerate output. The **only** matrices quantized to NVFP4 are the
routed experts (`gate_up_proj`/`down_proj`) and the shared expert
(`gate/up/down_proj`) — the memory bulk. With this policy the real model answers
the control prompt correctly. This deviates from the earlier draft policy and is
the validated choice.

## Numeric conversion pipeline

1. **Dense BF16** large projections: read BF16 safetensors → neutral
   `gpu::quantize_tensor_gpu(bf16, "BF16")` → `DenseNvfp4Schema` (E2M1 nibbles +
   per-block E4M3 scales + F32 `global_scale`).
2. **Routed experts** (BF16 fused tensors): slice `gate_up_proj[E]` into
   gate/up, then neutral `PackExpertNVFP4(ExpertData)` per expert; write
   `PackedExpert` (w1=gate, w3=up, w2=down) via `ExpertSchema`.
3. **BF16 pass-through:** embed, `lm_head`, norms, `o_norm`, AttnRes weights,
   router weights/bias, `a_log_bias`, `dt_bias`, `b_proj`, `f_a/g_a_proj`,
   conv1d, shared expert gate — verbatim, safetensors dtype preserved.

Conversion is strict: `mtp.*` is the only expected non-backbone prefix; any
unknown tensor name or dtype is an error, never silently ignored.

## Tokenizer (stage 4) — native SentencePiece-BPE backend

The checkpoint ships **SentencePiece BPE** (`model_type=2`), `identity`
normalizer, `add_dummy_prefix=1`, `escape_whitespaces=1`, byte-fallback enabled
(`<0x00>`..`<0xFF>` at ids 7..262), vocab 129024. The runtime is a GPT-2
byte-level BPE and cannot represent SentencePiece semantics, so a **model-neutral
SP-BPE backend** is added to the neutral tokenizer infrastructure (no family
stem → not scanned by the naming guard):

- `BuildTokenizerArrowFilesSentencePiece(tokenizer.model, out_dir)` in
  `tokenizer_convert.cpp`: a dependency-free protobuf wire reader extracts the
  pieces `(id, bytes, score, type)` and the trainer/normalizer options; it emits
  the existing vocab arrow plus a merges arrow, and records
  `tokenizer_kind = "sp-bpe"`, `byte_fallback`, `add_dummy_prefix`,
  `escape_whitespaces` and a direct `byte_to_token_id` (each byte maps to its
  `<0xXX>` piece). Merge rules are derived from the pieces exactly as HF's
  `SpmConverter` does: for each multi-codepoint piece find the split
  `(a,b)` into existing pieces maximizing `score(a)+score(b)`, ranked by
  descending piece score.
- `ArrowTokenizerDb` gains a `tokenizer_kind` discriminator (default
  `gpt2-byte-bpe`, preserving every existing model's behaviour). `tokenizer.cpp`
  adds `sp_bpe_encode` / `sp_bpe_decode`: identity normalization, escape spaces
  to `▁`, dummy prefix, adjacent-pair merge by rank, and byte fallback for
  uncovered codepoints; decode maps byte pieces back to raw bytes and `▁` to
  space. `arrow_tokenizer_lookup` is reused unchanged for special tokens.
- Alignment invariant: `bos_id`/`eos_id`/`pad_id` are read from the checkpoint
  (`1` / `2` / `-1`), never hardcoded in kernels.

**Golden strategy:** `tools/alice_tokenize_golden.py` reconstructs the reference
encoder from the proto using `tokenizers` (and cross-checks against
`sentencepiece`/`LlamaTokenizer` when importable) and emits
`tests/fixtures/tokenizer/alice.json` (canonical prompt → exact id sequence,
plus multi-byte / byte-fallback edge cases). The host `fast` test
`test_alice_tokenizer.cpp` pins the C++ encoder against those goldens; if
`sentencepiece` is unavailable at capture time the deviation is recorded rather
than hidden.

## Tool-calling bridge (stage 4b)

Implement the **native** AliceAI chat/tool codec in
`alice_tool_format.{h,cpp}` (add `alice_tool_format` to `AllowedCommon`),
verified against `chat_template.jinja` before coding:

- Roles are `system:` / `user:` / `assistant:` / `meta:`; the frame joins
  messages with `"\n\n "` and ends a generation prompt with `"\n\n assistant:"`.
- Tools header `Тебе доступны следующие функции:` followed by one
  `function {"name":…,"description":…,"parameters":…}` line per tool.
- Assistant tool calls: `\n[TOOL_CALL_START]{name}\n{arguments-json}`.
- Reasoning is wrapped `[COT_START]…[COT_END]`; tool results render as
  `Tool {name}: {content}`.
- Control markers are injected as their own vocab ids via
  `arrow_tokenizer_lookup` (checking the parsed `added_tokens` / special piece
  types); free text is BPE-encoded. `ParseToolCalls` scans decoded text and
  falls back to `tools::ParseToolCallsText` /
  `tools::ParseBareToolCallJson` when the native markers are absent.
- Host `fast` test `test_alice_tool_framing.cpp` on a synthetic SP-BPE tokenizer
  fixture: render structure, parse round-trip, tokenizer round-trip (never
  round-trip the full rendered prompt — its header embeds an example call).
- Because the checkpoint is a **Base** model, tool-use quality is untrained;
  this bridge pins the *format*, not model behaviour, and the limitation is
  recorded.

## Inference data flow and state

`embed → for layer 0..47: input ZC-norm → AttnRes mix → (KDA | gated attention)
→ accumulate partial → AttnRes mix → post ZC-norm → MoE (routed + shared) →
partial += moe; every 4 layers fold partial into completed_blocks → final
attnres_final → ZC norm → lm_head (F32) → sample`.

Per-session device state:

- **KDA recurrent state**: 36 layers × `[32,128,128]` F32 (≈ 0.6 GiB) plus the
  3 conv states `[4096,3]` per layer.
- **Full-attention KV cache**: 12 layers, GQA 2 KV heads, head_dim 256, full
  context.
- **AttnRes scratch**: the completed-block list (≤13 tensors of [2048] per
  token) + the running partial; per-token, no cross-token state.
- **Expert cache**: host mmap of 24,576 NVFP4 experts + device slot-LRU
  (`alice_moe_expert_cache.{h,cpp}`, neutral `expert_store`; knob
  `LSX_ALICE_EXPERT_CACHE_SLOTS`). Dense backbone resident on the GPU
  (embed/head ≈ 2.1 GiB BF16, projections/shared experts/router ≈ 1–2 GiB
  NVFP4); the ≈ 24,576 experts ≈ 38 GiB NVFP4 stream from host. Estimated
  resident ≈ 4–6 GiB, leaving ≈ 25 GiB for expert slots on the 32 GiB RTX 5090.
- Single CUDA stream, no per-token `cudaDeviceSynchronize`; reductions use the
  shared-memory sandbox (thread-0 + barriers) per `nvfp4-blackwell-conventions`.

## File map and build

```
libs/lsxcommon/src/alice_moe_constants.h
libs/lsxcommon/src/alice_moe_convert.cpp
libs/lsxcommon/src/alice_moe_load.cpp
libs/lsxcommon/src/alice_moe_expert_cache.cpp
libs/lsxcommon/src/alice_moe_kernels.h
libs/lsxcommon/src/alice_moe_kernels.cu
libs/lsxcommon/src/alice_moe_device_math.h
libs/lsxcommon/src/alice_moe_inference_attn.cu
libs/lsxcommon/src/alice_moe_inference_pipeline.cu
libs/lsxcommon/src/alice_moe_model.cpp
libs/lsxcommon/src/alice_tool_format.cpp
libs/lsxcommon/include/lsxcommon/alice_moe_convert.h
libs/lsxcommon/include/lsxcommon/alice_moe_load.h
libs/lsxcommon/include/lsxcommon/alice_moe_expert_cache.h
libs/lsxcommon/include/lsxcommon/alice_moe_model.h
libs/lsxcommon/include/lsxcommon/alice_moe_inference_attn.h
libs/lsxcommon/include/lsxcommon/alice_moe_inference_pipeline.h
libs/lsxcommon/include/lsxcommon/alice_tool_format.h
tests/test_alice_moe_fixture.h
tests/test_alice_moe_convert.cpp
tests/test_alice_moe_loader.cpp
tests/test_alice_moe_model.cpp
tests/test_alice_moe_kernels.cu
tests/test_alice_moe_selfcheck.cpp
tests/test_alice_moe_fsi_selfcheck.cpp
tests/test_alice_moe_code_selfcheck.cpp
tests/test_alice_moe_det_selfcheck.cpp
tests/test_alice_tool_framing.cpp
tests/test_alice_tokenizer.cpp
tools/alice_tokenize_golden.py
```

- New OBJECT library `lsxcommon_alice_cu` (`-fmad=false`, bit-exact math),
  linked PUBLIC into `lsxcommon`.
- New `lsxforge` flag `--alice` (`lsxcommon::alice::ConvertAliceModel`).
- Register `AliceFoundation80BModel` in `model_registry.cpp`.
- Tokenizer bridge, passport (`passport/passport.bin` via
  `lsxnetwork::BuildPassport`) and `LsxTokenTranslator` in `IoWorker` after
  affinity pinning, before readiness.

## Guard registries (same commit)

- `test_code_naming_guard.cpp`: add `alice` to `fam` and `FamilyStems()`; add
  `alice_tool_format` to `AllowedCommon()` (family-prefixed, no type marker).
  No new role/subsystem/variant tokens are needed (`convert`, `load`, `model`,
  `inference_{attn,pipeline}`, `kernels`, `device_math`, `expert_cache`,
  `constants` all exist).
- `test_model_independence_guard.cpp`: `kPassports` row
  `{"alice-foundation-80b-a3b", "alice_moe_constants.h",
  "lsxcommon::alice::frozen"}`; owner marker `namespace frozen {`;
  `alice_moe_model.cpp` in the per-`Infer` `LsxFastMathScope` consumers list;
  extend the cross-family-borrowing scan to cover `alice_moe_*`.
- `test_model_suite_guard.cpp`: `Suites()` row `convert → load → model →
  kernels → selfcheck → fsi → code → det` + add `alice` to its `FamilyStems()`
  and the model-stage regex.
- `test_ctest_labels_guard.cpp`: labels on every new test in the same commit.

## Testing strategy (TDD)

Strict red→green per stage; every new target gets its ctest label in the same
commit; every selfcheck/fsi/code/det target gets `SKIP_RETURN_CODE 4`.

1. **fast/host** — passport `static_assert`s; `kType` resolution; frozen-
   geometry rejection on mismatch; layer-type decoding (36 KDA / 12 full);
   `kAttnScale == 0.0625`; rotary-dim 64; router reference math
   (sigmoid + correction bias + renormalized top-10); KDA and AttnRes reference
   math in `alice_moe_device_math.h`; SP-BPE tokenizer parity + goldens;
   `alice_tool_format` render/parse + tokenizer round-trip.
2. **convert** (label `convert`, bounded TIMEOUT ≤120 s,
   `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1`): synthetic E2E on
   `test_synth_checkpoint.h` covering globals + one KDA layer + one full-attn
   layer + layer 0 (no AttnRes) + all 512 experts + the counted MTP record;
   assert the `logestix-alice-nvfp4` index, NVFP4 scale geometry, and untouched
   high-precision tensors. No real-model conversion in tests.
3. **load** — synthetic converted fixture only; format-tag gate, frozen-
   geometry gate, layer-0 AttnRes absence, missing/corrupt tensor diagnostics,
   expert LRU bookkeeping.
4. **kernels** — device vs host reference parity, bit-exact with `-fmad=false`:
   ZC RMSNorm; AttnRes depth-softmax mix; sigmoid top-10 router; SwiGLU shared
   + expert MLP; KDA depthwise conv-4, per-channel decay, delta recurrence
   (prefill + single-token decode parity); gated full attention with partial
   rotate-half RoPE, GQA, causal mask and output gate.
5. **selfcheck** — incremental 1→2-token `"Paris"` over ≥3 cycles
   (`LSX_TEST_ALICE_SELFCHECK_CYCLES`), bit-exact `final_logits`, real model via
   env (`LSX_TEST_ALICE_ROOT`), `SKIP()` when absent.
6. **fsi / code / det** — the model-neutral harnesses (`fsi_selfcheck`,
   `code_selfcheck`, `det_selfcheck`) captured from the source safetensors,
   per-family goldens under `tests/fixtures/{fsi,code,det}/alice.json`.

Perf: `tests/perf/perf_baselines.json` entry for `alice-foundation-80b-a3b`
plus a `TEST_CASE("perf baseline: alice-foundation-80b-a3b", "[perf]")` in the
shared `tests/test_perf_baseline.cpp` harness, measured after correctness at the
canonical protocol (correctness-first token-by-token, then measured).

## Error handling and strictness

- Load rejects a non-`logestix-alice-nvfp4` index, a geometry mismatch vs the
  frozen passport, and any missing / mis-shaped / corrupt tensor with a
  diagnostic naming the tensor.
- Conversion is strict: `mtp.*` is the only expected non-backbone prefix (MTP
  counted); any other unknown tensor is an error.
- `SelfCheck` returns `selfcheck_passed=false` with the decoded token on
  failure; no partial-success state.

## Risks / decision points

- **Primary:** the AttnRes depth mixer and the KDA recurrence are net-new for
  the repo and load-bearing; both are pinned against a host reference in
  `alice_moe_device_math.h` before the layer pipeline is wired.
- The SentencePiece-BPE backend is the first in the repo. Merge derivation from
  piece scores must match HF's `SpmConverter`; byte-fallback and the `▁` dummy
  prefix are the highest-risk details. Parity is pinned by committed goldens.
- `dt_bias` is per channel `[4096]` and `a_log_bias` per key head `[32]`; the
  gate broadcast must be exact.
- Real conversion of 152 GB across 49 shards is a long offline step (hours,
  ≈ 40 GB output, tight against the ≈ 200 GB free on `/mnt/llm_models`); it is
  not run inside tests, and the real-weight SelfCheck SKIPs until it is done.
- Single RTX 5090 (32 GiB): dense resident ≈ 4–6 GiB + expert streaming; t/s is
  expected to be modest and is recorded as measured, not targeted.
- The `"The capital of France is" → "Paris"` control prompt on a **Base** model
  may not answer `"Paris"`; the deviation is recorded and the canonical
  substring/top-1 policy is applied, per the established selfcheck fallback.
- The shipped `modeling_alice_ai.py` / `configuration_alice_ai.py` /
  `chat_template.jinja` / `tokenizer.model` are used as read-only references and
  are NOT committed to the repository.

## Out of scope (follow-ups)

MTP speculative decoding; extending the native SP-BPE backend to unigram
(not needed for this checkpoint); 262k-context prefill optimization; an optional
NVFP4 quantization of `o_proj`/small projections to reduce resident VRAM.
