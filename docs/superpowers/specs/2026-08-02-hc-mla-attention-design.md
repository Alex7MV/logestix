# Phase 2: Hyper-Connections + MLA Attention (DeepSeek-V4-Flash)

## Overview

`--testprompt` currently runs the model **without attention** and **without the
Hyper-Connections residual** — output is deterministic period-2 token→token
garbage.  This design adds the full per-layer block structure from the official
`inference/model.py` reference:

```
embed → h[4×4096]  (hc_mult=4 copies)
for each layer:
    residual = h
    x  = hc_pre(h, hc_attn_*)              # 4 copies → 1, Sinkhorn
    x  = attn_norm(x)
    x  = MLA(x, start_pos)                 # full causal, window=128
    h  = hc_post(x, residual, post, comb)  # 1 → 4 copies
    residual = h
    x  = hc_pre(h, hc_ffn_*)               # 4 copies → 1, Sinkhorn
    x  = ffn_norm(x)
    x  = MoE(x, input_ids)                 # shared expert only (routed = stub)
    h  = hc_post(x, residual, post, comb)  # 1 → 4 copies
final: h → hc_head (4→1) → norm → head.weight → logits
```

**Scope:** full HC (attention + FFN branches + head) + full causal MLA.
**Out of scope (this step):** CSA compressor/indexer (layers 2–42 ratios 4/128),
routed experts (kept stubbed), MTP.

## Reference Math (verified against inference/model.py + kernel.py)

### Hyper-Connections

Per token, hidden state is `hc_mult=4` copies of `dim=4096` (flattened 16384).

`hc_pre(x, hc_fn, hc_scale, hc_base)`:
```
x: [4, 4096] floats, hc_fn: [mix_hc=24, 16384], hc_scale: [3], hc_base: [24]
flat = x.flatten()                          # [16384]
rsqrt = rsqrt(mean(flat^2) + eps)           # eps = 1e-6 (norm_eps)
mixes = hc_fn @ flat * rsqrt                # [24]
(pre, post, comb) = hc_split_sinkhorn(mixes, hc_scale, hc_base)   # [4], [4], [4,4]
y = sum_j pre[j] * x[j]                     # [4096]
```

`hc_split_sinkhorn` (kernel, per token, threads=64):
```
pre[j]   = sigmoid(mixes[j]           * scale[0] + base[j])        + eps    (j<4)
post[j]  = 2 * sigmoid(mixes[j+4]     * scale[1] + base[j+4])              (j<4)
comb[j,k]= mixes[4*j + k + 8] * scale[2] + base[4*j + k + 8]               (j,k<4)
comb = softmax(comb, dim=-1) + eps                       # row softmax
comb = comb / (comb.sum(dim=-2) + eps)                  # column normalize
repeat sinkhorn_iters-1=19 times:  row-norm then col-norm (÷ (sum+eps))
```

`hc_post(x, residual, post, comb)`:
```
x: [4096], residual: [4, 4096], post: [4], comb: [4,4]
y[i] = post[i] * x + sum_j comb[i,j] * residual[j]      # [4, 4096]
```

Head `hc_head(x, hc_head_fn[4,16384], hc_head_scale[1], hc_head_base[4])`:
```
flat = x.flatten();  rsqrt = rsqrt(mean(flat^2) + eps)
mixes = hc_head_fn @ flat * rsqrt                       # [4]
pre = sigmoid(mixes * scale[0] + base) + hc_eps         # hc_eps = 1e-6
y = sum_j pre[j] * x[j]                                 # [4096]
```

### MLA (pure layers, compress_ratio == 0)

Config: `dim=4096, n_heads=64, head_dim=512, rope_head_dim=64,
q_lora_rank=1024, o_lora_rank=1024, o_groups=8, window=128, rope_theta=10000,
no YaRN for ratio==0 layers`.

```
qr = q_norm(wq_a(x))                       # [1024]; q_norm = RMSNorm(1024)
q  = wq_b(qr)                             # [32768] → [64, 512]
q  = q * rsqrt(mean(q^2, last) + eps)      # per-head RMS norm
q[..., -64:] = RoPE(q[..., -64:], freqs_cis)            # last 64 dims only

kv = kv_norm(wkv(x))                       # [512]; kv_norm = RMSNorm(512)
kv[..., -64:] = RoPE(kv[..., -64:], freqs_cis)
# QAT FP8-sim of kv[:-64] (per-64 block) — SKIPPED in FP16 impl (see Decisions)

topk_idxs = sliding-window indices (window=128), causal
o  = sparse_attn(q, kv_cache, attn_sink[64], topk_idxs, scale)   # scale = 512^-0.5
o[..., -64:] = RoPE(o[..., -64:], freqs_cis, inverse=True)       # DE-ROTATE o!

o  = o.view(8, 4096)                       # 8 groups × (8 heads × 512)
o  = einsum("gd, g r d -> g r", o, wo_a.view(8, 1024, 4096))     # [8, 1024]
x  = wo_b(o.flatten())                     # [8192] → [4096]
```

`sparse_attn` for our case (seq_len ≤ window, causal):
```
for each head h:
    s[t] = (q[h] · kv[t]) * scale          # t over causal window slots
    m    = max(s);  p[t] = exp(s[t] - m)
    sum  = sum(p) + exp(attn_sink[h] - m)  # sink added to DENOMINATOR only
    o[h] = (sum_t p[t] * kv[t]) / sum
```
Note: kv is shared across all 64 heads (`num_key_value_heads=1`), head_dim 512.

### Sliding window / KV cache

`kv_cache` size = `window_size + (max_seq_len // ratio if ratio else 0)` = 128
for pure-MLA layers.  Circular buffer indexed `start_pos % 128`.

Window topk indices (decode, `start_pos` is absolute position):
- `start_pos == 0` (prefill): `matrix = arange(seqlen)` (window not hit for
  prompt ≤ 128); masked rows `matrix > pos` → -1.
- `start_pos < win`: `[0..start_pos, -1, -1, ...]` → attends to cache slots 0..start_pos.
- `start_pos >= win-1`: circular rotation `[start_pos%win+1 ... win-1, 0 ... start_pos%win]`.

Our test never exceeds 128 tokens, so this reduces to **full causal attention**
over slots `0..start_pos` — but implement the window rotation for correctness.

### RoPE (ratio==0, YaRN disabled)

```
freqs[i] = 1 / (rope_theta ^ (2i / 64))      # i in 0..31, dim=64
freqs_cis[pos, i] = pos * freqs[i]
q/pairs (q[pos, 2i], q[pos, 2i+1]) → rotate by freqs_cis[pos, i]
```
`apply_rotary_emb` is in-place on the last 64 dims; inverse = conjugate.

## Data Structures (inference_load.cuh)

Extend `LayerWeights`:
```cpp
// Attention (F8_E4M3, dequantized lazily to FP16 scratch)
const uint8_t *d_wq_a;  const uint8_t *d_wq_a_scales;  // [1024,4096]
const uint8_t *d_wq_b;  const uint8_t *d_wq_b_scales;  // [32768,1024]
const uint8_t *d_wkv;   const uint8_t *d_wkv_scales;   // [512,4096]
const uint8_t *d_wo_a;  const uint8_t *d_wo_a_scales;  // [8192,4096] (layout per-group)
const uint8_t *d_wo_b;  const uint8_t *d_wo_b_scales;  // [4096,8192]
int64_t wq_a_n, wq_b_n, wkv_n, wo_a_n, wo_b_n;          // element counts
int64_t wq_a_s_n, ...;                                  // scale element counts
__half *d_q_norm;  __half *d_kv_norm;  __half *d_attn_norm;  // BF16[1024]/[512]/[4096]
float   *d_attn_sink;                                   // FP32[64]

// Hyper-Connections (FP32)
float *d_hc_attn_fn;   // [24, 16384]
float *d_hc_attn_base; // [24]
float *d_hc_attn_scale;// [3]
float *d_hc_ffn_fn;    // [24, 16384]
float *d_hc_ffn_base;  // [24]
float *d_hc_ffn_scale; // [3]
```

Extend `PinnedWeights`:
```cpp
float *d_hc_head_fn;    // [4, 16384]
float *d_hc_head_base;  // [4]
float *d_hc_head_scale; // [1]
```

Weight layouts (from converter-verified shard headers, layer 0):
| tensor | shape | scale shape |
|---|---|---|
| wq_a | [1024,4096] | [8,32] |
| wq_b | [32768,1024] | [256,8] |
| wkv | [512,4096] | [4,32] |
| wo_a | [8192,4096] | [64,32] |
| wo_b | [4096,8192] | [32,64] |
| attn_norm.weight | [4096] BF16 | — |
| q_norm.weight | [1024] BF16 | — |
| kv_norm.weight | [512] BF16 | — |
| attn_sink | [64] F32 | — |

Scale layout is `[rows/128, cols/128]` — one E8M0 per 128×128 tile, matching
`convert_kernel` in `safetensors_gpu.cu`.

## Loading (inference_load.cu)

In `load_pinned_weights`, per-layer `layer_N.arrow` already contains attention +
HC tensors (converter's `layers.(\d+).` regex + `pinned_category_name` writes
all non-expert tensors).  Add uploads:

1. **FP32 tensors** (hc_attn_*, hc_ffn_*, attn_sink, hc_head_* in global):
   raw `float` copy host→device (`cudaMemcpyAsync`), no conversion.
2. **BF16 tensors** (q_norm, kv_norm, attn_norm): reuse `upload_bf16_to_half`.
3. **F8_E4M3 + E8M0 scales** (wq_a/b, wkv, wo_a/b): copy raw bytes host→device
   as `uint8_t*` (same as shared-expert path).  Keep bytes + scale bytes +
   element counts; **dequant happens on the fly per GEMM** into FP16 scratch
   (user decision: no repack, no E2M1 re-quant).

Tensor name lookup in arrow: suffixes `attn.wq_a.weight`, `attn.wkv.weight`,
`attn.wo_a.weight`, `attn.q_norm.weight`, `attn.kv_norm.weight`,
`attn.attn_sink`, `attn_norm.weight`, `hc_attn_fn`, `hc_attn_base`,
`hc_attn_scale`, `hc_ffn_fn`, `hc_ffn_base`, `hc_ffn_scale`, global
`hc_head_fn`/`hc_head_base`/`hc_head_scale`.

## GPU Kernels / Compute (new file inference_attn.cu)

Scratch buffers (allocated in pipeline, sized at kHiddenDim scale):
- `d_attn_q`      [64, 512] FP16
- `d_attn_kv`     [512] FP16 (current token kv)
- `d_attn_kv_cache` [num_layers][128][512] FP16  (or per-layer persistent)
- `d_attn_o`      [64, 512] FP16
- `d_dequant2`    F8→FP16 scratch (reuse/extend existing d_dequant_scratch)
- `d_hc_state`    [4, 4096] FP16 current layer state (buffers ping-pong)
- `d_hc_x`        [4096] FP16 (single-copy attention/ffn input)
- `d_hc_post`/`d_hc_comb` [4]/[4,4] FP32 (per-token, persistent reuse)
- `d_hc_mixes`    [24] FP32, `d_hc_pre` [4] FP32

**HC kernels** (per-token, grid=1):
- `hc_pre_kernel`: computes mixes = hc_fn@flat*rsqrt, runs sinkhorn inline,
  writes `d_hc_x` = Σ pre[j]·x[j], plus post/comb to scratch.
- `hc_post_kernel`: writes `y[i] = post[i]*x + Σ_j comb[i,j]*residual[j]`.
- `hc_head_kernel`: pre = sigmoid(...)+eps, y = Σ pre[j]·x[j].

**F8→FP16 dequant** for each weight (reuse pattern from `dequant_nvfp4_to_half_kernel` /
`convert_kernel`): 128×128 tile scale → `w_fp16[row, col] = e4m3(row,col) * scale[row/128, col/128]`.

**GEMMs** (cublas FP16, following `run_shared_experts_ffn` / `run_lm_head`):
1. `wq_a[1024,4096] @ x[4096]`        → qr[1024]
2. `wq_b[32768,1024] @ qr[1024]`      → q[32768] (reshape [64,512])
3. `wkv[512,4096] @ x[4096]`          → kv[512]
4. per-head RMS(q), RoPE(q[...,-64:]), RoPE(kv[...,-64:])
5. attention kernel (below)
6. inverse RoPE(o[...,-64:])
7. `wo_a[8192,4096] @ o_flat[4096]`   → o_lora[8192] (view [8,1024])
8. `wo_b[4096,8192] @ o_lora[8192]`   → attn_out[4096]

**Attention kernel** (`mla_attn_kernel`, causal, window):
- Inputs: q[64,512] FP16, kv_cache[128,512] FP16, attn_sink[64] F32, pos,
  start_pos, window, scale.
- Per head h: dot q[h]·kv[t] over causal slots t, running max/sum
  (FlashAttention-style), denominator += exp(sink[h] - max); out = Σ p·kv / sum.
- Grid over heads; each block handles a few heads, threads over 512 dims.

**KV cache**: per-layer persistent `[128, 512]`; slot = `start_pos % 128`; memset
at start.  Causal mask from window index list.

## Pipeline Integration (inference_pipeline.cu)

Replace the current per-layer block (RMSNorm→router→shared FFN→half_add) with:

```
// per layer:
// 1. hc_pre(h, hc_attn_*) → x (single copy) + post/comb scratch
// 2. run_rmsnorm(x, attn_norm) → x_norm
// 3. MLA: q/kv GEMMs + rope + attention + de-rotate + o GEMMs → attn_out
// 4. hc_post(attn_out, h, post, comb) → h (4 copies)
// 5. hc_pre(h, hc_ffn_*) → x + post/comb scratch
// 6. run_rmsnorm(x, ffn_norm) → x_norm
// 7. run_shared_experts_ffn(x_norm) → ffn_out   (routed experts = stub, unchanged)
// 8. hc_post(ffn_out, h, post, comb) → h (4 copies) → next layer input
```

Final: `hc_head(h) → x`, `run_rmsnorm(x, final_norm)`, `run_lm_head`.

Buffers `d_hidden`/`d_hidden_next` are repurposed: `d_hidden` holds the 4-copy
HC state `[4,4096]`; a separate `d_single` [4096] holds the reduced copy.
All existing GEMM paths (`run_shared_experts_ffn`, `run_lm_head`) operate on
`d_single`/its normed variant — no signature changes needed.

Timing/debug: keep per-layer `[dbg]` RMS prints but change to debug level;
add `[dbg]` for attention (q/kv/attn_sink sanity, logits maxv).

## Decisions

- **On-the-fly F8→FP16 dequant per GEMM** (user decision) — no repack, no E2M1.
- **Skip QAT FP8-sim of kv[...,:-64]** in FP16 path: reference applies
  `act_quant(kv[:-64], 64, ...)` to match training-time FP8 rounding; our FP16
  attention stores exact values.  Slight numeric deviation, structurally
  faithful.  Revisit if output quality demands it.
- **KV cache FP16** (not FP8).
- **Routed experts stay stubbed** (only shared expert FFN).  Router output is
  computed but unused — unchanged from current behavior.
- **Persistent per-layer KV cache** for all 43 layers allocated once in pipeline
  (128×512×43×2 B ≈ 5.6 MB).

## Verification

1. Build `lsxfabric`; run `--testprompt`:
   - logits finite, maxv sane (≥ ~10);
   - no period-2 token repetition; output should start forming English
     continuation of "The capital of France is";
   - tok/s comparable to baseline (attention ~ GEMM-bound, +1–2 ms/layer).
2. Numeric checks (debug level): per-head q/kv RMS ≈ 1; attn_sink finite;
   residual (hc_state) RMS stays O(0.1–1) across layers.
3. `ctest` 7/7 still pass (no test asserts token values — only structure/metrics).
4. Commit + push to `feat/testmodel`.

## Risks

- **HC numeric stability**: 4-copy state + Sinkhorn amplification; if RMS
  diverges, clamp/verify against `eps` handling in reference (pre has `+eps`,
  comb normalized with `+eps` denominators).
- **F8 scale layout mismatch**: verify `[rows/128, cols/128]` against
  `convert_kernel` indexing before GEMM (mis-scaling → NaN).
- **wo_a grouping**: `wo_a.weight` [8192,4096] viewed as [8,1024,4096] must
  match einsum order `bsgd,grd->bsgr` (o groups are contiguous 8-head slices).
- **De-rotation of o**: forgetting inverse RoPE before o-projection silently
  degrades output.
