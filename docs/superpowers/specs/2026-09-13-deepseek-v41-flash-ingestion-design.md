# DeepSeek-V4.1-Flash — Ingestion Design (Milestone 1: text backbone e2e)

Date: 2026-09-13
Status: approved design, pending implementation plan
Branch: `feat/model-DeepSeek-V4.1-Flash`

## Overview

Ingest DeepSeek-V4.1-Flash (`DeepseekV41ForCausalLM`, `model_type=deepseek_v41`,
nested `text_config`) into the logestix monorepo following the 8-step Model
Ingestion Protocol. The operator selected approach **B — an independent,
autonomous contour**: new `deepseek_moe_v41_*` files must not include or depend
on any DeepSeek-V4-Flash code. Only model-neutral infrastructure is reused
(`model_base`, `model_runtime`, `expert_store`, `lsx_inference_nvfp4`,
`tokenizer`, `schema`, `path_registry`, `gpu_perf`, `sampling`).

This is not a variant of V4-Flash. V4.1 is a **Causal Encoder-Decoder (CED)**
sparse-attention MoE: 40 backbone layers whose compressed KV / index top-k /
candidate block masks are produced by designated *source* layers and shared
downstream in a single forward pass via a `SharedAttentionRuntime`. It adds
`compress_ratio ∈ {0,1,2}`, a two-level indexer, engram n-gram lookups, a
DSpark MTP draft, and new geometry (hidden 5120, 384 experts, q_lora 1280,
`index_n_heads` 32, FP8 32×32 blocks, `rms_eps` 1e-20).

Milestone 1 scope (operator decision): the **text backbone end-to-end**, i.e.
converter → load → forward for CED/compressor/indexer/candidate/HC/MoE/head,
targeting a working `SelfCheck` ("The capital of France is" → "Paris").
Deferred and explicitly recorded: **engram**, **DSpark MTP**, **vision+aligner**.

Source checkpoint (read-only):
`/mnt/llm_models/models/DeepSeek-V4.1-Flash` (48 safetensors shards, BF16/FP8
dense + I8-packed FP4 experts, 96 085 tensors). Reference PyTorch:
`inference/model.py` (1309 lines), `inference/kernel.py` (591 lines),
`inference/convert.py`; runtime constants `inference/config.json`. All formulas
below are pinned verbatim from that reference for bit-exact implementation.

## Autonomy invariant

- `deepseek_moe_v41_*` must not `#include` `deepseek_moe_load.cuh`,
  `deepseek_moe_inference_*.cu`, or any `deepseek_moe_*` header of V4-Flash.
- No shared constants with V4: the V4.1 geometry is a fresh compile-time frozen
  block (`deepseek_moe_v41_constants.h`).
- `tests/test_model_independence_guard.cpp` is extended with the V4.1 consumer
  (opened `LsxFastMathScope`, unique passport, no cross-includes) in the same
  commit that lands the model.

## Frozen geometry (compile-time)

Locked in `libs/lsxcommon/src/deepseek_moe_v41_constants.h` with `static_assert`;
`ValidateGeometry()` cross-checks the parsed index against these at load;
escape hatch `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY` (repo policy).

| Constant | Value | Notes |
|---|---|---|
| kType | `"deepseek-v4.1-flash"` | registry id (`detect_model_type` normalization of `"DeepSeek-V4.1-Flash"`) |
| kHidden | 5120 | `text_config.hidden_size` |
| kLayers | 40 | backbone (MTP 3 not in the loop) |
| kMoeIntermediate | 2304 | `moe_intermediate_size` |
| kSharedIntermediate | 2304 | one shared expert |
| kNumRouted | 384 | `n_routed_experts` |
| kTopK | 6 | `num_experts_per_tok` |
| kHeads | 64 | `num_attention_heads` |
| kHeadDim | 512 | |
| kRopeHeadDim | 64 | `qk_rope_head_dim` |
| kQLoraRank | 1280 | |
| kOLoraRank | 1024 | |
| kOGroups | 8 | |
| kIndexNHeads | 32 | `index_n_heads` |
| kIndexHeadDim | 128 | `index_head_dim` |
| kIndexTopk | 512 | |
| kWindow | 128 | sliding window (ring buffer) |
| kVocab | 129280 | embed/head |
| kHcMult | 4 | hyper-connections copies |
| kHcMix | (2+4)·4 = 24 | |
| kHcDim | 4·5120 = 20480 | |
| kSinkhornIters | 20 | |
| kRmsEps | 1e-20f | `norm_eps` (not 1e-6) |
| kHcEps | 1e-6f | |
| kRouteScale | 1.5f | |
| kGateTemp | 1.0f | dataclass default (not in config) |
| kSwigluLimit | 10.0f | |
| kCompressRopeTheta | 160000 | ratio>0 layers |
| kRopeTheta | 10000 | pure-SWA layers |
| kRopeFactor | 16 | YaRN |
| kOriginalSeqLen | 65536 | YaRN |
| kBetaFast/kBetaSlow | 32 / 1 | YaRN |
| kCandidateBlockSize | 8 | |
| kCandidateTopkBlocks | 2048 | |
| kEngramLayers | {1, 14} | deferred (not converted) |
| kMtpLayers | 3 | stored-not-applied |

Backbone `compress_ratios` (first 40 of 43; 40..42 are MTP zeros):
`[0,0, 2×18 (layers 2..19), 1×20 (layers 20..39)]`.

Source sets:
- `kv_source_layers = {2,8,14,20}` — own a `Compressor`, publish `compress_kv`
  and `index_k`.
- `index_source_layers = {2,8,14,20,24,28,32,36}` — run an `Indexer`, publish
  `topk_idxs`.
- `candidate_source_layer = 20` — publishes block mask; consumed by indexers at
  24/28/32/36.
- Layers 0,1: `ratio=0`, pure sliding-window attention.

## Converted layout: `logestix-v41-nvfp4`

New format tag (autonomy + anti-cross-load gate). Output directory follows the
vault convention `<model>/logestix/`. The index carries:

```
format: "logestix-v41-nvfp4"
model:  "DeepSeek-V4.1-Flash"
num_layers: 40
experts_per_layer: 384
total_experts: 15360
vram: { min_bytes, optimal_bytes, max_bytes, context_tokens: 1048576 }
files: { pinned: { global, layer_000..layer_039 }, experts: layer_N/exp_E.arrow, ... }
layers: [ { compress_ratio, is_kv_source, is_index_source, is_candidate_source } × 40 ]
deferred: { engram: "not-converted", dspark: "stored-not-applied", vision: "cut", vision_tensors: N }
```

### Globals (`pinned/global.arrow`)
- `embed.weight` BF16 [129280, 5120]
- `head.weight` BF16 [129280, 5120]
- `norm.weight` BF16 [5120]

No `hc_head_*` parameters exist in V4.1 (unlike V4): the final `hc_pre` collapses
the hc stream with the last layer's `ffn_pre`, then `norm`, then the plain BF16
`head`. Verified against the checkpoint index.

### Per-layer (`pinned/layer_N.arrow`)
- `attn_norm.weight` BF16 [5120]; `ffn_norm.weight` BF16 [5120]
- `attn.q_norm.weight` BF16 [1280]; `attn.kv_norm.weight` BF16 [512]
- `attn.attn_sink` F32 [64]
- `attn.wq_a.weight` F8_E4M3 [1280,5120] + scale E8M0 [40,160]
- `attn.wq_b.weight` F8_E4M3 [32768,1280] + scale E8M0 [1024,40]
- `attn.wkv.weight`  F8_E4M3 [512,5120] + scale E8M0 [16,160]
- `attn.wo_a.weight` F8_E4M3 [8192,4096] + scale E8M0 [256,128]
- `attn.wo_b.weight` F8_E4M3 [5120,8192] + scale E8M0 [160,256]
- HC: `hc_attn_fn`/`hc_ffn_fn` F32 [24,20480],
  `hc_attn_base`/`hc_ffn_base` F32 [24],
  `hc_attn_scale`/`hc_ffn_scale` F32 [3]
- Router: `ffn.gate.weight` BF16 [384,5120], `ffn.gate.bias` F32 [384],
  `ffn.gate.bias_vl` F32 [384] (stored; unused text-only)
- Shared expert: `ffn.shared_experts.{w1,w3}` F8_E4M3 [2304,5120] +
  E8M0 [72,160]; `w2` F8_E4M3 [5120,2304] + E8M0 [160,72]
- Compressor (only `is_kv_source`):
  `attn.compressor.wkv.weight` BF16 [512,5120],
  `attn.compressor.wgate.weight` BF16 [512,5120] (ratio>1 only),
  `attn.compressor.norm.weight` BF16 [512]
- Indexer (only `is_index_source`):
  `attn.indexer.wq_b.weight` F8_E4M3 [4096,1280] + E8M0 [128,40],
  `attn.indexer.weights_proj.weight` BF16 [32,5120],
  plus (`owns_k` only) `attn.indexer.wk.weight` BF16 [128,512], `k_norm` BF16 [128]

Scale tiles are stored raw (F8_E8M0 32×32). Decision (correctness first): the
loader dequantizes dense FP8 (MLA projections, shared experts, indexer `wq_b`)
to FP16 device caches using the 32×32 scales — the repo etalon is the V4
shared-expert `d_shared_w*_h` load-time dequant. This avoids an FP8 GEMM path in
milestone 1 (~10 GiB FP16 dense on a 32 GiB card; expert store LRU owns the
remainder). Norms/biases stay high precision, never quantized (MODEL INGEST
step 2). An in-kernel FP8 dequant is a possible later memory optimization.

### Routed experts (`experts/layer_N/exp_E.arrow`)
Source: I8 packed E2M1 nibbles + F8_E8M0 per-32 scales. Converted to the
`logestix-v1-nvfp4` convention: pow2-E4M3 per-16 scales (`e4 << 3`, split every
per-32 scale into two identical per-16 halves) + a per-tensor float
`global_scale = 2^m` chosen by the E8M0 window shift. Nibbles carried verbatim.
This is an independent re-implementation of the same transform as the V4 repack
(no shared code).

## Pinned reference formulas (bit-exact contract)

Scalars: `fp8_block_size = 32`, `fp4_block_size = 32`, `scale_fmt = "ue8m0"`.
FP8 max 448; FP4 max 6.

### RMSNorm (`model.py:281`)
```
x = x.float()
var = x.square().mean(-1, keepdim=True)
x = x * rsqrt(var + eps)          # eps = 1e-20 for V4.1
return (weight * x).to(input_dtype)
```

### RoPE + YaRN (`model.py:369`)
`freqs = 1 / base^(arange(0,dim,2)/dim)`; YaRN when `original_seq_len>0`:
`corrected_dim(r) = dim·ln(original/(r·2π))/(2·ln base)`;
`low=floor(corrected_dim(beta_fast))`, `high=ceil(corrected_dim(beta_slow))`;
`ramp=clamp((i-low)/max(high-low,1e-3),0,1)`, `smooth=1-ramp`;
`freqs = freqs/factor·(1-smooth) + freqs·smooth`. Then
`freqs = polar(1, outer(arange(seqlen), freqs))`. `apply_rotary_emb` treats
adjacent element pairs as complex, `inverse=True` conjugates (used to un-rotate
the attention output). Parameters: ratio>0 layers use
`original_seq_len=65536, base=compress_rope_theta=160000`; ratio=0 layers use
`original_seq_len=0, base=rope_theta=10000`.

### Hyper-Connections + Sinkhorn (`model.py:907`, `kernel.py:407`)
`mix_hc=(2+hc)·hc`. `hc_mixes`: flatten hc stream, one scale per token
`rsqrt = rsqrt(x.square().mean(-1) + norm_eps)`, `mixes = linear(x, hc_fn)·rsqrt`.
Split of the `mix_hc` vector: `pre[j]=sigmoid(mixes[j]·scale0 + base[j]) + hc_eps`;
`post[j]=2·sigmoid(mixes[j+hc]·scale1 + base[j+hc])`;
`comb[j,k]=mixes[j·hc+k+2hc]·scale2 + base[j·hc+k+2hc]`.
Sinkhorn: `comb = softmax(comb, -1) + hc_eps`; then `comb /= (comb.sum(-2)+hc_eps)`;
repeat `sinkhorn_iters-1` more times alternating row (`sum(-1)`) and column
(`sum(-2)`) normalization.
`hc_pre`: `sum_c(pre_mix[...,c,None] · x[...,c,:])`.
`hc_post`: `post[...,c,None]·x[...,None,:] + sum_c'(comb[...,c',c,None]·residual[...,c',None,:])`.
Block forward uses the **previous** sub-block's `pre_mix` (attention consumes the
mix produced by the previous layer's FFN; FFN consumes the mix attention
produced). `make_identity_pre_mix` = one-hot on copy 0 for the first layer.

### Attention (`model.py:613`)
```
qr = q_norm(wq_a(x)); q = wq_b(qr) → [heads, head_dim]
apply_rotary_emb(q[..., -64:], freqs)
kv = kv_norm(wkv(x)); apply_rotary_emb(kv[..., -64:], freqs)
act_quant(kv, 32, "ue8m0", E8M0, inplace=True)     # FP8 block-32 pow2, dequant back
# window ring buffer of size 128; topk_idxs = window ring indices
# if ratio>0: append compress_kv (fp4 block-16 E4M3, inplace) + index topk
o = sparse_attn(q, concat(window_kv, compress_kv), attn_sink, topk_idxs, head_dim^-0.5)
apply_rotary_emb(o[..., -64:], freqs, inverse=True)
o = o.view(groups, -1) einsum with wo_a (block-diagonal per group) → wo_b
```
`sparse_attn` (`kernel.py:311`) is online-softmax over gathered indices with
`attn_sink`: `scores_max` initialized to **-1e30**, invalid index (-1) → score
`-inf`, final `sum_exp += exp(attn_sink - scores_max)`; a row with no valid
index yields an all-zero output.

### Compressor (`model.py:429`)
`compress_ratio == 1`: `norm(wkv(x))`, weights stay BF16.
`compress_ratio > 1`: promote `wkv`/`wgate` to FP32; softmax-pool every `ratio`
consecutive tokens (`kv = Σ kv·softmax(score)`); carry a partial-group
`kv_state/score_state` across decode steps (score padded with -inf); publish only
when a group completes; `norm(kv)` before RoPE. Publish part of latent **before**
RoPE (indexer needs the unrotated form); Attention rotates the cache afterwards
with the source's ratio-relative positions (group `j` at position `j·ratio`).

### Indexer (`model.py:488`) + candidate blocks (`model.py:583`)
`owns_k` (kv_source): `k = k_norm(wk(latent))`, RoPE on `k[..., -64:]`,
`fp4_act_quant(k, 32, inplace=True)` (E8M0 pow2), store to `index_k` cache.
`q = wq_b(qr)` → [32,128], RoPE, `fp4_act_quant`.
`score = relu(Σ_h q·k)·weights_proj(x)·(128^-0.5 · 32^-0.5)`; mask positions
beyond `compress_len`; `topk = min(512, end_pos//ratio)`, re-sorted by position,
`-1` for unreachable, else `idx + offset`.
Candidate level-1 (`select_candidate_blocks`): block score = max over the block;
the block holding the newest position pinned to `+inf`; keep `topk_blocks=2048`
blocks of `block_size=8`; consumers mask scores outside the source's mask.

### Router / Gate (`model.py:792`)
`scores = linear(x.float(), weight.float()) / gate_temp` (gate_temp 1.0), then
`sqrtsoftplus = softplus(scores).sqrt()`.
`indices = topk(scores + bias)[1]` (**bias selects only, never scales**);
`weights = scores.gather(indices)`; if `norm_topk_prob` and topk>1:
`weights /= weights.sum(-1,keepdim=True) + 1e-20`; `weights *= route_scale (1.5)`.
`bias_vl` selects `bias` for image-span tokens — stored, unused text-only.

### Expert SwiGLU / MoE (`model.py:830`)
```
gate = w1(x).float(); up = w3(x).float()
up   = clamp(up,   -10, 10)
gate = clamp(gate,  -∞, 10)
y = silu(gate) * up;  y = weights*y;  out = w2(y.to(dtype))
```
MoE = top-k routed (NVFP4 GEMV via the neutral `lsx_nvfp4` layer) + one shared
expert over every token.

### act-quant (`kernel.py:41,184`)
FP8 (window KV): per-32 block, `amax = max(amax, 1e-4)`, pow2 scale
`fast_round_scale(amax/448)` (bit-manip `2^ceil(log2(amax/448)))`), quant-dequant
back to BF16 in place.
FP4 compressed KV: per-16 block, E4M3 (non-pow2) scale, `amax = max(amax, 6·2^-9)`.
FP4 indexer: per-32 block, E8M0 pow2 scale, `amax = max(amax, 6·2^-126)`.

## Deferred subsystems (explicit)

- **engram** (layers 1, 14): tables are ~98 GiB each (384M × 256 FP8) — not
  converted. `Engram.forward` contributes `h + gate·value` into the residual
  before the block. Skipping it changes the residual at layers 1 and 14. This is
  the **primary risk to the Paris selfcheck**; the plan gates on a first
  end-to-end selfcheck and escalates engram into scope (milestone 1.5) if it
  fails. Converter records `engram: not-converted`.
- **DSpark MTP** (3 layers + markov/confidence heads): converted **stored, not
  applied** (repo policy for MTP). Forward uses only the 40 backbone layers.
- **vision + aligner** (32-layer ViT, `aligner.*`, image tokens): cut at
  conversion with a counted skip; `bias_vl`/image token ids stored but unused.

## File map and build

```
libs/lsxcommon/src/deepseek_moe_v41_constants.h
libs/lsxcommon/src/deepseek_moe_v41_convert.{h,cpp}
libs/lsxcommon/src/deepseek_moe_v41_load.{h,cpp}
libs/lsxcommon/src/deepseek_moe_v41_model.{h,cpp}
libs/lsxcommon/src/deepseek_moe_v41_inference.cu
libs/lsxcommon/src/deepseek_moe_v41_device_math.h
tests/test_deepseek_moe_v41_fixture.h
tests/test_deepseek_moe_v41_convert.cpp
tests/test_deepseek_moe_v41_loader.cpp
tests/test_deepseek_moe_v41_model.cpp
tests/test_deepseek_moe_v41_kernels.cu
tests/test_deepseek_moe_v41_selfcheck.cpp
```

- New OBJECT library `lsxcommon_deepseek_v41_cu` with `-fmad=false` (bit-exact
  math), linked PUBLIC into `lsxcommon` (mirrors `lsxcommon_glm_cu`).
- New `lsxforge` flag `--deepseek_v41` (input/vault → `<model>/logestix/`),
  alongside the existing `--deepseek_nvfp4`; entry point
  `lsxcommon::deepseek_v41::ConvertDeepSeekV41Model`.
- Register `DeepSeekV41FlashModel` in `model_registry.cpp`; add the suite to
  `test_model_suite_guard.cpp` (`convert → load → model → kernels → selfcheck`)
  and to `test_code_naming_guard.cpp` allowlists in the same commit.
- Tokenizer: BPE vocab + merges are byte-identical to V4-Flash; only
  `added_tokens` differ (mm-span placeholders). Build the tokenizer arrow from
  V4.1's `tokenizer.json` via `BuildTokenizerArrowFiles`; bos/eos/pad from the
  config (0/1/2), never hardcoded.
- Passport/network: vocab identical → no new passport artifact expected; verify
  and record. `LsxTokenTranslator` integration is out of milestone 1 unless a
  test proves a gap.

## Inference data flow

`embed(BF16) → expand to hc_mult FP32 copies → for layer 0..39:`
1. `hc_mixes(stream, ...)` → (pre, post, comb); `hc_pre` with the *incoming*
   `pre_mix` (identity for layer 0) → single sublayer input.
2. `attn_norm` → attention:
   - `wq/q_norm/wq_b`; `wkv/kv_norm`; RoPE; window ring.
   - if `is_kv_source`: Compressor → publish `compress_kv` (and `index_k` if
     `owns_k`).
   - if `is_index_source`: Indexer → (candidate source publishes block mask) →
     publish `topk_idxs`.
   - `sparse_attn(window ∪ compressed, attn_sink, topk_idxs)`; un-rotate output;
     grouped `wo_a`/`wo_b`.
   - `hc_post` → back to hc stream.
3. `ffn_norm` → MoE (router top-6 + shared expert, NVFP4 GEMV) → `hc_post`.
4. carry `pre_mix = ffn_pre` to the next layer.

`final: hc_pre(stream, last pre_mix) → norm → head (BF16→F32 logits) → sample`.
`SharedAttentionRuntime` holds one slot each for `compress_kv`, `index_k`,
`topk_idxs`, `candidates`; sources write before consumers read within the same
in-order layer walk.

## Testing strategy (TDD)

Strict red→green per stage; every new test target gets its label and, for
selfcheck, `SKIP_RETURN_CODE 4` in the same commit.

1. **fast/host** — constants `static_assert`s; registry + `detect_model_type`
   normalization; `load_topology` on a synthetic index (layers/experts/vram +
   error paths); candidate/index top-k and sinkhorn on the host reference
   (`deepseek_moe_v41_device_math.h`).
2. **convert** — synthetic e2e (repo etalon:
   `test_glm_moe_flash_convert.cpp` + `test_synth_checkpoint.h`): a fake
   checkpoint with FP8 32×32 scale tiles, FP4 I8 + E8M0/32 experts, router
   bias/bias_vl, compressor/indexer tensors, HC; assert the `logestix-v41-nvfp4`
   index, scale-tile geometry, E8M0→NVFP4 pair transform, and the counted
   vision/engram skips. No real-model conversion in tests.
3. **load** — synthetic converted fixture only (suite guard forbids the vault in
   loader stages); strictness on missing/corrupt tensors, frozen-geometry gate,
   format-tag gate, expert LRU/residency bookkeeping.
4. **kernels** — device vs host reference parity, bit-exact target with
   `-fmad=false`: RMSNorm (eps 1e-20), RoPE/YaRN, HC/Sinkhorn, window+compressed
   sparse attention with `attn_sink`, Compressor (ratio 1 and 2), Indexer +
   candidate two-level top-k, router sqrtsoftplus/noaux_tc/route_scale, expert
   SwiGLU clamps, act-quant (FP8 block-32 pow2, FP4 E4M3/16, FP4 E8M0/32).
5. **selfcheck** — incremental 1→2-token "Paris" over ≥3 cycles
   (`LSX_TEST_DEEPSEEK_V41_SELFCHECK_CYCLES`) with bit-exact `final_logits`,
   real converted model via env (`LSX_TEST_DEEPSEEK_V41_ROOT`), `SKIP()` when
   absent.

## Error handling and strictness

- Load rejects a non-`logestix-v41-nvfp4` index, a geometry mismatch vs the
  frozen passport, and any missing / mis-shaped / corrupt tensor with a
  diagnostic naming the tensor.
- Conversion is strict: unknown tensors are either explicitly cut (vision,
  aligner, engram) with a counted record, or an error — never silently ignored.
- `SelfCheck` returns `selfcheck_passed=false` with the decoded token on failure;
  no partial-success state.

## Risks / decision points

- **Primary:** skipped engram perturbs layers 1/14; Paris may not pass until
  engram is implemented. Decision point after the first backbone e2e: if Paris
  fails and divergence originates at/after layer 1, pull engram into scope.
- CED source/consumer ordering is the subtlest part; kernel tests pin index and
  candidate outputs before attention integration.
- Full 552B-model conversion is a heavy operator step; all tests run on
  synthetic fixtures and the real selfcheck SKIPs without the vault.
- `rms_eps=1e-20` and `-fmad=false` are mandatory for numerical parity; a
  missing `-fmad=false` on the V4.1 object library is a silent bit-exactness
  regression.

## Out of scope (follow-ups)

engram; DSpark speculative decoding; vision/multimodal; performance tuning
(bit-exact correctness first, speed pass afterwards); passport/network changes
unless a gap is proven.
