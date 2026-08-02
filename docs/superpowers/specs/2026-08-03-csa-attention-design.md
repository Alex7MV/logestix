# CSA (Compressed Sparse Attention) Implementation Design

**Date**: 2026-08-03  
**Branch**: feat/testmodel  
**Status**: Approved

## Problem

DeepSeek-V4-Flash uses Compressed Sparse Attention (CSA) for layers 2-41. The current
runtime only implements sliding-window MLA for all layers, skipping the compressor
and indexer tensors entirely. This causes semantic drift and degenerate output.

## Reference Architecture

From `inference/model.py` and `inference/kernel.py`:

### Layer Classification

| Layers | Type | compress_ratio | Indexer | coff |
|--------|------|---------------|---------|------|
| 0, 1, 42 | Pure MLA (sliding window) | 0 | No | 1 |
| 2, 4, 6, ..., 40 (even) | CSA with indexer | 4 | Yes | 2 |
| 3, 5, 7, ..., 41 (odd) | CSA without indexer | 128 | No | 1 |

`coff = 1 + (compress_ratio == 4)` — overlapping windows for ratio=4.

### Tensor Inventory (per CSA layer)

**Compressor** (all CSA layers):
- `compressor.ape`: F32 [ratio, coff*head_dim] — positional encoding
- `compressor.wkv.weight`: BF16 [coff*head_dim, 4096] — KV projection
- `compressor.wgate.weight`: BF16 [coff*head_dim, 4096] — gate projection
- `compressor.norm.weight`: BF16 [head_dim] — RMSNorm weight (512 elements)

**Indexer** (ratio=4 layers only):
- `indexer.wq_b.weight`: F8_E4M3 [8192, 1024] — Q projection (uses f8_gemm_1d)
- `indexer.wq_b.scale`: F8_E8M0 [64, 8] — per-128x128 tile scales
- `indexer.weights_proj.weight`: BF16 [64, 4096] — scoring projection
- `indexer.compressor.ape`: F32 [4, 256] — positional encoding (head_dim=128)
- `indexer.compressor.wkv.weight`: BF16 [256, 4096] — KV projection
- `indexer.compressor.wgate.weight`: BF16 [256, 4096] — gate projection
- `indexer.compressor.norm.weight`: BF16 [128] — RMSNorm weight

### Execution Phases (per token step)

#### Phase 1: Compressor — Gated KV Pooling

```
kv = wkv(x)          # [coff*head_dim] BF16 GEMM
score = wgate(x)     # [coff*head_dim] BF16 GEMM
score += ape[start_pos % ratio]

# Stateful accumulation
kv_state[start_pos % ratio] = kv
score_state[start_pos % ratio] = score

# Every ratio tokens: compress
if (start_pos + 1) % ratio == 0:
    compressed_kv = Σ(kv_state * softmax(score_state))  # weighted sum over ratio
    compressed_kv = norm(compressed_kv)                 # RMSNorm
    apply_rope(compressed_kv[-rd:], freqs_cis)
    act_quant(compressed_kv[:-rd], 64, inplace=True)     # FP8 simulate
    compressed_kv_cache[start_pos // ratio] = compressed_kv
```

For ratio=4 (overlap=True), the state buffers double: kv_state [2*ratio, 2*head_dim].
The overlap transform interleaves current and previous windows.

#### Phase 2: Indexer (ratio=4 only) — Top-k Compressed Position Selection

```
q_idx = wq_b(qr)                     # F8 GEMM [8192] → [64, 128] (8 heads, 128 dim)
q_idx = rotate_activation(q_idx)     # Hadamard rotation
fp4_act_quant(q_idx, inplace=True)   # FP4 simulation

# Indexer has its own compressor with head_dim=128
indexer.compressor(x, start_pos)     # maintains separate compressed_kv_cache

# Score compressed KV positions
index_score = relu(q_idx @ indexer_kv_cache^T)  # [seqlen, n_compressed]
index_score = index_score * weights_proj(x)     # [seqlen, 64] → sum over heads

# Select top-k=512 compressed positions (clamped to available)
compress_topk_idxs = index_score.topk(min(512, n_compressed))
```

For ratio=128 (no indexer): `compress_topk_idxs = get_compress_topk_idxs(ratio, ...)`
— simple positional selection (all compressed positions up to current).

#### Phase 3: Sparse Attention

```
# Combine window + compressed positions
topk_idxs = cat(window_topk_idxs, compress_topk_idxs)

# Sparse attention: gather only selected KV positions
o = sparse_attn(q, kv_cache, attn_sink, topk_idxs, softmax_scale)

# Inverse RoPE on output
apply_rope_inv(o[-rd:], freqs_cis)
```

`sparse_attn` kernel: for each head, compute attention scores over only the
`topk_idxs` selected KV positions (not the full window). Apply softmax with
sink tokens, weighted sum.

### KV Cache Layout

```
kv_cache: [window_size + max_compressed, head_dim]
          ├── window_size (128) sliding window KV
          └── max_compressed compressed KV (seq_len / ratio)
```

### Constants

```
window_size = 128 (kAttnWindow)
head_dim = 512 (kHeadDim)
rope_head_dim = 64 (kRopeHeadDim)
nope_head_dim = 448 (kHeadDim - kRopeHeadDim)
index_n_heads = 64
index_head_dim = 128
index_topk = 512
```

## Implementation Plan

### 1. LayerWeights Extension (`inference_load.cuh`)

Add fields:
```cpp
// CSA compressor (layers 2-41)
__half *d_comp_wkv = nullptr;        // [coff*head_dim, 4096] BF16→FP16
__half *d_comp_wgate = nullptr;     // [coff*head_dim, 4096] BF16→FP16
float  *d_comp_ape = nullptr;       // [ratio, coff*head_dim] F32
__half *d_comp_norm = nullptr;      // [head_dim] BF16→FP16
int comp_ratio = 0;                 // 0=MLA, 4 or 128
int comp_coff = 1;                  // 2 for ratio=4, 1 for ratio=128

// Indexer (ratio=4 only)
__half *d_idx_wq_b = nullptr;       // [8192, 1024] F8→FP16
uint8_t *d_idx_wq_b_scales = nullptr;
int idx_wq_b_rows = 0, idx_wq_b_cols = 0;
__half *d_idx_weights_proj = nullptr; // [64, 4096] BF16→FP16
// Indexer's own compressor
__half *d_idx_comp_wkv = nullptr;
__half *d_idx_comp_wgate = nullptr;
float  *d_idx_comp_ape = nullptr;
__half *d_idx_comp_norm = nullptr;
```

### 2. CSA State Buffers (pipeline)

Per-layer device buffers (allocated once):
```cpp
// Compressor state
float *d_comp_kv_state;     // [max_layers][2*4, 2*512] — for overlap
float *d_comp_score_state;  // [max_layers][2*4, 2*512]
// Compressed KV cache (per layer)
__half *d_compressed_kv;    // [max_layers][max_seq/ratio, head_dim]
// Indexer state (ratio=4 only)
float *d_idx_kv_state;
float *d_idx_score_state;
__half *d_idx_compressed_kv;
```

### 3. New Kernels (`inference_csa.cu`)

- `compressor_step_kernel` — gated pooling compression step
- `indexer_score_kernel` — score compressed KV positions
- `sparse_attn_kernel` — attention over selected positions only

### 4. Pipeline Integration (`inference_pipeline.cu`)

Replace `run_mla_attention` call for layers 2-41 with `run_csa_attention`:
```
if (lw.compress_ratio > 0) {
    run_csa_attention(cublas_h, stream, lw, d_normed,
                      d_comp_state[lay], d_compressed_kv[lay],
                      d_idx_state[lay], d_idx_compressed_kv[lay],
                      step, d_q, ..., d_attn_out, ...);
} else {
    run_mla_attention(...);  // existing path for layers 0,1,42
}
```

### 5. Weight Loading (`inference_load.cu`)

Load CSA tensors from pinned layer arrows during `load_pinned_weights`:
- Parse tensor names matching `attn.compressor.*` and `attn.indexer.*`
- Convert BF16→FP16 (reuse `upload_bf16_to_half`)
- F8_E4M3 tensors via existing `f8_gemm_1d` dequant path
- Store device pointers in `LayerWeights`

## Verification

1. HC state rms should stay bounded (<10) across all 43 layers
2. Output should produce coherent English for "The capital of France is"
3. No illegal memory access or NaN
4. tokens/s should be >0.4 (CSA adds overhead but shouldn't be dramatically slower)
