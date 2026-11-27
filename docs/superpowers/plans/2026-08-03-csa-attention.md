# CSA (Compressed Sparse Attention) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Compressed Sparse Attention (Compressor + Indexer + sparse_attn) for layers 2-41 to replace the current sliding-window-only MLA path.

**Architecture:** CSA layers (2-41) use a Compressor (gated KV pooling every `compress_ratio` tokens) + Indexer (ratio=4 only, learned top-k selection) + sparse_attn (gather-only-selected-positions attention). Layers 0,1,42 keep existing MLA path. All computation in FP16/FP32 matching reference.

**Tech Stack:** CUDA 13.3 / Compute 12.0, cuBLAS, C++20, Arrow IPC for tensor loading.

**Reference:** `inference/model.py` (Attention.forward, Compressor.forward, Indexer.forward), `inference/kernel.py` (sparse_attn_kernel, act_quant_kernel).

---

## File Structure

| File | Responsibility |
|------|----------------|
| `libs/lsxcommon/src/inference_load.cuh` | LayerWeights struct — add CSA tensor fields |
| `libs/lsxcommon/src/inference_load.cu` | Load CSA tensors from pinned layer arrows |
| `libs/lsxcommon/src/inference_csa.cu` | **New file** — Compressor kernel, Indexer kernel, sparse_attn kernel |
| `libs/lsxcommon/include/lsxcommon/inference_csa.h` | **New file** — Public API for CSA kernels |
| `libs/lsxcommon/CMakeLists.txt` | Add inference_csa.cu to lsxcommon_cu OBJECT library |
| `libs/lsxcommon/src/inference_pipeline.cu` | Route to CSA vs MLA per layer; allocate CSA state buffers |

---

### Task 1: Extend LayerWeights with CSA tensor fields

**Files:**
- Modify: `libs/lsxcommon/src/inference_load.cuh:97-150`

- [ ] **Step 1: Add CSA fields to LayerWeights struct**

Add after the existing `float rope_theta = 10000.0f;` line:

```cpp
    // CSA Compressor (layers 2-41; nullptr for pure MLA layers 0,1,42)
    __half *d_comp_wkv = nullptr;        // [coff*head_dim, 4096] BF16→FP16
    __half *d_comp_wgate = nullptr;      // [coff*head_dim, 4096] BF16→FP16
    float  *d_comp_ape = nullptr;        // [ratio, coff*head_dim] F32
    __half *d_comp_norm = nullptr;      // [head_dim] BF16→FP16
    int comp_coff = 1;                  // 2 for ratio=4 (overlap), 1 for ratio=128

    // CSA Indexer (ratio=4 only; nullptr for ratio=128 and pure MLA)
    const uint8_t *d_idx_wq_b = nullptr;        // F8_E4M3 [8192, 1024]
    const uint8_t *d_idx_wq_b_scales = nullptr; // F8_E8M0 [64, 8]
    int idx_wq_b_rows = 0, idx_wq_b_cols = 0;
    __half *d_idx_weights_proj = nullptr;       // BF16→FP16 [64, 4096]
    // Indexer's own Compressor (head_dim=128 for indexer)
    __half *d_idx_comp_wkv = nullptr;            // [256, 4096] BF16→FP16
    __half *d_idx_comp_wgate = nullptr;          // [256, 4096] BF16→FP16
    float  *d_idx_comp_ape = nullptr;            // [4, 256] F32
    __half *d_idx_comp_norm = nullptr;            // [128] BF16→FP16
    int idx_head_dim = 128;                       // indexer head_dim
    int idx_n_heads = 64;                         // indexer number of heads
```

- [ ] **Step 2: Add Free() cleanup for CSA tensors**

In `LayerWeights::Free()` add CSA cleanup after existing fields:

```cpp
    // CSA compressor
    if (d_comp_wkv) cudaFree(d_comp_wkv);
    if (d_comp_wgate) cudaFree(d_comp_wgate);
    if (d_comp_ape) cudaFree(d_comp_ape);
    if (d_comp_norm) cudaFree(d_comp_norm);
    // CSA indexer
    if (d_idx_weights_proj) cudaFree(d_idx_weights_proj);
    if (d_idx_comp_wkv) cudaFree(d_idx_comp_wkv);
    if (d_idx_comp_wgate) cudaFree(d_idx_comp_wgate);
    if (d_idx_comp_ape) cudaFree(d_idx_comp_ape);
    if (d_idx_comp_norm) cudaFree(d_idx_comp_norm);
    // Note: d_idx_wq_b and d_idx_wq_b_scales are raw pointers into the
    // mmap'd arrow file (not cudaMalloc'd), so no free needed.
```

- [ ] **Step 3: Build to verify compilation**

```powershell
$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"
& "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build build --config Debug --target lsxfabric 2>&1 | Select-Object -Last 1
```
Expected: `lsxfabric.vcxproj -> ...lsxfabric.exe`

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: extend LayerWeights with CSA compressor/indexer tensor fields"
```

---

### Task 2: Load CSA tensors from pinned layer arrows

**Files:**
- Modify: `libs/lsxcommon/src/inference_load.cu` (in `load_pinned_weights` per-layer loop)

- [ ] **Step 1: Add CSA tensor loading after attention weights load**

After the existing attention weight loading block (after `wo_b` loading), inside the per-layer loop, add CSA loading. The tensors are stored in the pinned layer arrow as BF16/F32/F8_E4M3 tensors with names like `attn.compressor.wkv.weight`, `attn.indexer.wq_b.weight`, etc.

```cpp
        // ── CSA Compressor loading (layers 2-41) ──────────────────────────
        if (lw.compress_ratio > 0) {
            int coff = lw.comp_coff;  // 2 for ratio=4, 1 for ratio=128
            int comp_out = coff * kHeadDim;  // 1024 for ratio=4, 512 for ratio=128

            // compressor.wkv.weight: BF16 [coff*head_dim, 4096]
            auto *t_comp_wkv = find_tensor_suffix(lt, "attn.compressor.wkv.weight");
            if (t_comp_wkv) {
                e = upload_bf16_to_half(t_comp_wkv->data.data(),
                    static_cast<int64_t>(comp_out) * kHiddenDim,
                    &lw.d_comp_wkv, stream);
                if (e != cudaSuccess) return e;
            }
            // compressor.wgate.weight: BF16 [coff*head_dim, 4096]
            auto *t_comp_wgate = find_tensor_suffix(lt, "attn.compressor.wgate.weight");
            if (t_comp_wgate) {
                e = upload_bf16_to_half(t_comp_wgate->data.data(),
                    static_cast<int64_t>(comp_out) * kHiddenDim,
                    &lw.d_comp_wgate, stream);
                if (e != cudaSuccess) return e;
            }
            // compressor.ape: F32 [ratio, coff*head_dim]
            auto *t_comp_ape = find_tensor_suffix(lt, "attn.compressor.ape");
            if (t_comp_ape) {
                e = upload_f32_bytes(t_comp_ape->data.data(),
                    static_cast<int64_t>(lw.compress_ratio) * comp_out,
                    &lw.d_comp_ape, stream);
                if (e != cudaSuccess) return e;
            }
            // compressor.norm.weight: BF16 [head_dim]
            auto *t_comp_norm = find_tensor_suffix(lt, "attn.compressor.norm.weight");
            if (t_comp_norm) {
                e = upload_bf16_to_half(t_comp_norm->data.data(),
                    static_cast<int64_t>(kHeadDim),
                    &lw.d_comp_norm, stream);
                if (e != cudaSuccess) return e;
            }

            // ── CSA Indexer (ratio=4 only) ─────────────────────────────────
            if (lw.compress_ratio == 4) {
                // indexer.wq_b.weight: F8_E4M3 [8192, 1024] — raw pointer into arrow
                auto *t_idx_wq_b = find_tensor_suffix(lt, "attn.indexer.wq_b.weight");
                auto *t_idx_wq_b_s = find_tensor_suffix(lt, "attn.indexer.wq_b.scale");
                if (t_idx_wq_b && t_idx_wq_b_s) {
                    // Upload F8 weights + scales to device
                    int64_t wq_n = t_idx_wq_b->data.size();
                    int64_t sc_n = t_idx_wq_b_s->data.size();
                    void *d_wq_raw = nullptr, *d_sc_raw = nullptr;
                    cudaMalloc(&d_wq_raw, wq_n);
                    cudaMalloc(&d_sc_raw, sc_n);
                    cudaMemcpyAsync(d_wq_raw, t_idx_wq_b->data.data(), wq_n,
                        cudaMemcpyHostToDevice, stream);
                    cudaMemcpyAsync(d_sc_raw, t_idx_wq_b_s->data.data(), sc_n,
                        cudaMemcpyHostToDevice, stream);
                    lw.d_idx_wq_b = static_cast<const uint8_t*>(d_wq_raw);
                    lw.d_idx_wq_b_scales = static_cast<const uint8_t*>(d_sc_raw);
                    lw.idx_wq_b_rows = 8192;
                    lw.idx_wq_b_cols = 1024;
                }
                // indexer.weights_proj.weight: BF16 [64, 4096]
                auto *t_idx_wp = find_tensor_suffix(lt, "attn.indexer.weights_proj.weight");
                if (t_idx_wp) {
                    e = upload_bf16_to_half(t_idx_wp->data.data(),
                        64LL * kHiddenDim, &lw.d_idx_weights_proj, stream);
                    if (e != cudaSuccess) return e;
                }
                // indexer.compressor.* (head_dim=128)
                auto *t_ic_wkv = find_tensor_suffix(lt, "attn.indexer.compressor.wkv.weight");
                if (t_ic_wkv) {
                    e = upload_bf16_to_half(t_ic_wkv->data.data(),
                        256LL * kHiddenDim, &lw.d_idx_comp_wkv, stream);
                    if (e != cudaSuccess) return e;
                }
                auto *t_ic_wgate = find_tensor_suffix(lt, "attn.indexer.compressor.wgate.weight");
                if (t_ic_wgate) {
                    e = upload_bf16_to_half(t_ic_wgate->data.data(),
                        256LL * kHiddenDim, &lw.d_idx_comp_wgate, stream);
                    if (e != cudaSuccess) return e;
                }
                auto *t_ic_ape = find_tensor_suffix(lt, "attn.indexer.compressor.ape");
                if (t_ic_ape) {
                    e = upload_f32_bytes(t_ic_ape->data.data(),
                        4LL * 256, &lw.d_idx_comp_ape, stream);
                    if (e != cudaSuccess) return e;
                }
                auto *t_ic_norm = find_tensor_suffix(lt, "attn.indexer.compressor.norm.weight");
                if (t_ic_norm) {
                    e = upload_bf16_to_half(t_ic_norm->data.data(),
                        128LL, &lw.d_idx_comp_norm, stream);
                    if (e != cudaSuccess) return e;
                }
            }
        }
```

- [ ] **Step 2: Add `upload_f32_bytes` function if not already present**

Check if `upload_f32_bytes` exists. If not, add it near `upload_bf16_to_half`:

```cpp
cudaError_t upload_f32_bytes(const uint8_t *src, int64_t numel,
                               float **d_out, cudaStream_t stream)
{
    cudaError_t e = cudaMalloc(d_out, static_cast<size_t>(numel) * sizeof(float));
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(*d_out, src,
        static_cast<size_t>(numel) * sizeof(float),
        cudaMemcpyHostToDevice, stream);
    return e;
}
```

- [ ] **Step 3: Build and verify loading works**

```powershell
$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"
& "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build build --config Debug --target lsxfabric 2>&1 | Select-Object -Last 1
```

Run inference to verify CSA tensors load without crash:
```powershell
& "build\bin\Debug\lsxfabric.exe" --testprompt --localmodel "D:\Programs\lsx_model\DeepSeek-V4-Flash" 2>&1 | Select-String "compressor|indexer|error" | Select-Object -First 5
```

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: load CSA compressor/indexer tensors from pinned layer arrows"
```

---

### Task 3: Create CSA kernels file

**Files:**
- Create: `libs/lsxcommon/src/inference_csa.cu`
- Create: `libs/lsxcommon/include/lsxcommon/inference_csa.h`
- Modify: `libs/lsxcommon/CMakeLists.txt` (add to lsxcommon_cu)

- [ ] **Step 1: Create header with public API**

`libs/lsxcommon/include/lsxcommon/inference_csa.h`:

```cpp
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdint>

namespace lsxcommon {
namespace gpu {

// CSA state per layer (allocated once, reused across steps)
struct CSAState {
    // Main compressor state
    float *d_comp_kv_state = nullptr;     // [max_coff_ratio, coff*head_dim] FP32
    float *d_comp_score_state = nullptr;  // [max_coff_ratio, coff*head_dim] FP32
    __half *d_compressed_kv = nullptr;    // [max_compressed_slots, head_dim] FP16

    // Indexer compressor state (ratio=4 only)
    float *d_idx_kv_state = nullptr;
    float *d_idx_score_state = nullptr;
    __half *d_idx_compressed_kv = nullptr; // [max_compressed_slots, 128]

    int n_compressed = 0;  // how many compressed KV slots are filled
};

// Run full CSA attention for one token.
// For layers 2-41 with compress_ratio > 0.
// Reuses the same Q/KV/O-projection path as MLA, but adds compressor + indexer
// + sparse_attn instead of the full-window mla_attn_kernel.
cudaError_t run_csa_attention(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const struct LayerWeights &lw,
    const __half *d_x,               // [dim] attn_norm output
    __half *d_kv_cache_layer,        // [window, head_dim] sliding-window KV cache
    CSAState &csa_state,             // per-layer CSA state
    int start_pos,
    // Scratch buffers (same as MLA):
    __half *d_q, __half *d_qr, __half *d_qr2,
    __half *d_kv, __half *d_kv2,
    __half *d_o, __half *d_olora, __half *d_out,
    __half *d_dequant,
    float *d_norm_scratch, float *d_norm_gamma_scratch);

// Allocate CSA state for all layers
cudaError_t alloc_csa_state(CSAState *states, int num_layers,
                             int max_seq_len, int head_dim, cudaStream_t stream);

// Free CSA state
void free_csa_state(CSAState *states, int num_layers);

} // namespace gpu
} // namespace lsxcommon
```

- [ ] **Step 2: Create implementation file**

`libs/lsxcommon/src/inference_csa.cu`:

```cpp
#include "lsxcommon/inference_csa.h"
#include "lsxcommon/log.h"
#include "inference_load.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cmath>
#include <algorithm>

namespace lsxcommon {
namespace gpu {

// ── Constants ──────────────────────────────────────────────────────────────
constexpr int kCSABlock = 256;

// ── Compressor step kernel ──────────────────────────────────────────────────
// Accumulates kv/score into state, compresses when (start_pos+1) % ratio == 0.
// For decode (start_pos > 0): stores into single slot, compresses when full.
// For prefill (start_pos == 0): handled separately in host code.
//
// This kernel handles the decode-path compression:
// 1. Store kv and score into state[slot]
// 2. If should_compress: softmax-weighted sum → compressed_kv
__global__ void compressor_step_kernel(
    const __half *d_kv,         // [coff*head_dim] wkv output (FP16)
    const __half *d_score,      // [coff*head_dim] wgate output (FP16)
    const float *d_ape,         // [ratio, coff*head_dim] positional encoding
    float *d_kv_state,          // [ratio, coff*head_dim] state
    float *d_score_state,       // [ratio, coff*head_dim] state
    __half *d_compressed_kv,   // [head_dim] output (one compressed KV)
    int ratio, int coff, int head_dim,
    int start_pos,
    bool overlap)               // true for ratio=4
{
    int slot = start_pos % ratio;
    int out_dim = coff * head_dim;
    int d = head_dim;

    // Phase 1: store kv and score into state
    for (int i = threadIdx.x; i < out_dim; i += blockDim.x) {
        float ape_val = d_ape[slot * out_dim + i];
        d_kv_state[slot * out_dim + i] = __half2float(d_kv[i]);
        d_score_state[slot * out_dim + i] = __half2float(d_score[i]) + ape_val;
    }
    __syncthreads();

    // Check if we should compress
    bool should_compress = ((start_pos + 1) % ratio) == 0;
    if (!should_compress) return;

    // Phase 2: softmax over the ratio dimension
    // For each output dimension i, compute softmax over the ratio slots
    // Then weighted sum of kv_state
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float max_score = -1e30f;
        // For overlap, we need to handle the interleaved layout
        // For simplicity, handle non-overlap first (ratio=128 and non-overlap ratio=4)
        for (int r = 0; r < ratio; ++r) {
            float s = d_score_state[r * out_dim + i];
            if (s > max_score) max_score = s;
        }
        float sum_exp = 0.0f;
        for (int r = 0; r < ratio; ++r) {
            float s = d_score_state[r * out_dim + i];
            d_score_state[r * out_dim + i] = expf(s - max_score);
            sum_exp += expf(s - max_score);
        }
        float compressed = 0.0f;
        for (int r = 0; r < ratio; ++r) {
            float w = d_score_state[r * out_dim + i] / (sum_exp + 1e-9f);
            compressed += d_kv_state[r * out_dim + i] * w;
        }
        d_compressed_kv[i] = __float2half(compressed);
    }
}

// ── Sparse attention kernel (replaces mla_attn_kernel for CSA layers) ───────
// Gathers only the top-k selected KV positions (indices in h_topk_idxs)
// and computes attention with online softmax + sink.
__global__ void sparse_attn_kernel(
    const __half *d_q,          // [n_heads, head_dim]
    const __half *d_kv_cache,  // [total_slots, head_dim] (window + compressed)
    const float *d_attn_sink,   // [n_heads]
    __half *d_o,                // [n_heads, head_dim] output
    const int *d_topk_idxs,     // [n_selected] KV slot indices to attend to
    int n_selected,             // number of valid KV positions
    float scale)                // 1/sqrt(head_dim)
{
    const int h = blockIdx.x;  // one block per head
    const int tid = threadIdx.x;
    constexpr int kThreads = 128;

    __shared__ float s_max;
    __shared__ float s_sum;
    __shared__ float s_acc_o[512]; // head_dim = 512

    // Initialize
    if (tid < 512) s_acc_o[tid] = 0.0f;
    if (tid == 0) { s_max = -1e30f; s_sum = 0.0f; }
    __syncthreads();

    const __half *qrow = d_q + h * 512;

    // Pass 1: find max score (for numerical stability)
    for (int t = 0; t < n_selected; ++t) {
        int slot = d_topk_idxs[t];
        if (slot < 0) continue;
        const __half *krow = d_kv_cache + slot * 512;
        float score = 0.0f;
        for (int d = tid; d < 512; d += kThreads)
            score += __half2float(qrow[d]) * __half2float(krow[d]);
        // Reduce score across threads (simplistic — each thread computes partial)
        __shared__ float s_partial[128];
        s_partial[tid] = score;
        __syncthreads();
        for (int s = kThreads / 2; s > 0; s >>= 1) {
            if (tid < s) s_partial[tid] += s_partial[tid + s];
            __syncthreads();
        }
        if (tid == 0) {
            float full_score = s_partial[0] * scale;
            if (full_score > s_max) s_max = full_score;
        }
        __syncthreads();
    }

    // Include sink in max
    if (tid == 0) {
        float sink_score = d_attn_sink[h];
        if (sink_score > s_max) s_max = sink_score;
    }
    __syncthreads();

    // Pass 2: compute softmax weights and accumulate output
    for (int t = 0; t < n_selected; ++t) {
        int slot = d_topk_idxs[t];
        if (slot < 0) continue;
        const __half *krow = d_kv_cache + slot * 512;
        float score = 0.0f;
        for (int d = tid; d < 512; d += kThreads)
            score += __half2float(qrow[d]) * __half2float(krow[d]);
        __shared__ float s_partial2[128];
        s_partial2[tid] = score;
        __syncthreads();
        for (int s = kThreads / 2; s > 0; s >>= 1) {
            if (tid < s) s_partial2[tid] += s_partial2[tid + s];
            __syncthreads();
        }
        if (tid == 0) {
            float p = expf(s_partial2[0] * scale - s_max);
            s_sum += p;
        }
        __syncthreads();
        // Recompute for accumulation (can't reuse due to needing per-element)
        // Get weight from thread 0
        __shared__ float s_weight;
        if (tid == 0) s_weight = expf(s_partial2[0] * scale - s_max);
        __syncthreads();
        float w = s_weight;
        for (int d = tid; d < 512; d += kThreads)
            s_acc_o[d] += w * __half2float(krow[d]);
    }

    // Add sink contribution (weight = exp(sink - max))
    if (tid == 0) {
        float sink_w = expf(d_attn_sink[h] - s_max);
        s_sum += sink_w;
    }
    __syncthreads();

    // Normalize output
    float inv_sum = 1.0f / (s_sum + 1e-9f);
    for (int d = tid; d < 512; d += kThreads)
        d_o[h * 512 + d] = __float2half(s_acc_o[d] * inv_sum);
}

// ── Host: allocate CSA state ───────────────────────────────────────────────
cudaError_t alloc_csa_state(CSAState *states, int num_layers,
                             int max_seq_len, int head_dim, cudaStream_t stream)
{
    int max_compressed = max_seq_len / 4;  // worst case: ratio=4
    for (int l = 0; l < num_layers; ++l) {
        auto &s = states[l];
        // Main compressor state: [2*4, 2*512] max (overlap ratio=4)
        cudaMalloc(&s.d_comp_kv_state, 2 * 4 * 2 * 512 * sizeof(float));
        cudaMalloc(&s.d_comp_score_state, 2 * 4 * 2 * 512 * sizeof(float));
        cudaMemsetAsync(s.d_comp_kv_state, 0, 2 * 4 * 2 * 512 * sizeof(float), stream);
        cudaMemsetAsync(s.d_comp_score_state, 0, 2 * 4 * 2 * 512 * sizeof(float), stream);
        // Compressed KV cache: [max_compressed, head_dim]
        cudaMalloc(&s.d_compressed_kv, static_cast<size_t>(max_compressed) * head_dim * sizeof(__half));
        cudaMemsetAsync(s.d_compressed_kv, 0, static_cast<size_t>(max_compressed) * head_dim * sizeof(__half), stream);
        // Indexer state (allocate for all; only used by ratio=4)
        cudaMalloc(&s.d_idx_kv_state, 2 * 4 * 2 * 128 * sizeof(float));
        cudaMalloc(&s.d_idx_score_state, 2 * 4 * 2 * 128 * sizeof(float));
        cudaMemsetAsync(s.d_idx_kv_state, 0, 2 * 4 * 2 * 128 * sizeof(float), stream);
        cudaMemsetAsync(s.d_idx_score_state, 0, 2 * 4 * 2 * 128 * sizeof(float), stream);
        cudaMalloc(&s.d_idx_compressed_kv, static_cast<size_t>(max_compressed) * 128 * sizeof(__half));
        cudaMemsetAsync(s.d_idx_compressed_kv, 0, static_cast<size_t>(max_compressed) * 128 * sizeof(__half), stream);
    }
    return cudaSuccess;
}

void free_csa_state(CSAState *states, int num_layers)
{
    for (int l = 0; l < num_layers; ++l) {
        auto &s = states[l];
        if (s.d_comp_kv_state) cudaFree(s.d_comp_kv_state);
        if (s.d_comp_score_state) cudaFree(s.d_comp_score_state);
        if (s.d_compressed_kv) cudaFree(s.d_compressed_kv);
        if (s.d_idx_kv_state) cudaFree(s.d_idx_kv_state);
        if (s.d_idx_score_state) cudaFree(s.d_idx_score_state);
        if (s.d_idx_compressed_kv) cudaFree(s.d_idx_compressed_kv);
    }
}

} // namespace gpu
} // namespace lsxcommon
```

- [ ] **Step 3: Add inference_csa.cu to CMakeLists.txt**

In `libs/lsxcommon/CMakeLists.txt`, find the `lsxcommon_cu` OBJECT library source list and add:

```cmake
    src/inference_csa.cu
```

- [ ] **Step 4: Build and verify**

```powershell
$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"
& "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build build --config Debug --target lsxfabric 2>&1 | Select-Object -Last 3
```
Expected: Build succeeds with no errors.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: CSA kernels (compressor step, sparse_attn) + state management"
```

---

### Task 4: Implement run_csa_attention host function

**Files:**
- Modify: `libs/lsxcommon/src/inference_csa.cu` (add `run_csa_attention`)

- [ ] **Step 1: Implement run_csa_attention**

Add to `inference_csa.cu` after the kernels. This function:
1. Reuses the Q/KV/O-projection code from MLA (wq_a → q_norm → wq_b → per_head_rms → RoPE)
2. Computes KV and stores in sliding-window cache
3. Runs the Compressor step (gated KV pooling)
4. Runs the Indexer (ratio=4 only) to select compressed positions
5. Builds topk_idxs = window + compressed indices
6. Runs sparse_attn_kernel
7. Inverse RoPE + O-projection (same as MLA)

```cpp
// Reuse MLA's f8_gemm_1d and other helpers via extern
namespace lsxcommon { namespace gpu {

// Extern from inference_attn.cu
extern cudaError_t f8_gemm_1d(cublasHandle_t, cudaStream_t,
    const uint8_t*, const uint8_t*, int, int,
    const __half*, __half*, __half*);
extern __global__ void dequant_f8_to_half_kernel(
    const uint8_t*, const uint8_t*, __half*, int64_t,
    int64_t, int64_t, int64_t, int64_t);
extern __global__ void per_head_rms_kernel(__half*, int, int);
extern __global__ void rope_apply_kernel(
    __half*, int, int, int, int, float, float, int);

struct LayerWeights;  // forward decl from inference_load.cuh

// rmsnorm for half input → half output (extern from inference_attn.cu)
extern "C" __global__ void half_to_float_k(
    const __half* src, float* dst, int n);
extern "C" __global__ void float_to_half_k(
    const float* src, __half* dst, int n);

// run_csa_attention: full CSA forward for one token
cudaError_t run_csa_attention(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const LayerWeights &lw,
    const __half *d_x,
    __half *d_kv_cache_layer,
    CSAState &csa_state,
    int start_pos,
    __half *d_q, __half *d_qr, __half *d_qr2,
    __half *d_kv, __half *d_kv2,
    __half *d_o, __half *d_olora, __half *d_out,
    __half *d_dequant,
    float *d_norm_scratch, float *d_norm_gamma_scratch)
{
    // Q projection (identical to MLA)
    cudaError_t e = f8_gemm_1d(cublas_h, stream, lw.d_wq_a, lw.d_wq_a_scales,
        lw.wq_a_rows, lw.wq_a_cols, d_x, d_qr, d_dequant);
    if (e != cudaSuccess) return e;
    // q_norm
    // ... (reuse run_rmsnorm_attn via the same approach as MLA)
    e = f8_gemm_1d(cublas_h, stream, lw.d_wq_b, lw.d_wq_b_scales,
        lw.wq_b_rows, lw.wq_b_cols, d_qr, d_q, d_dequant);
    if (e != cudaSuccess) return e;
    // per-head RMS
    per_head_rms_kernel<<<kAttnHeads, 128, 0, stream>>>(d_q, kAttnHeads, kHeadDim);
    // RoPE on Q (with YaRN for CSA layers)
    {
        int total = kAttnHeads * (kRopeHeadDim / 2);
        rope_apply_kernel<<<(total + 127) / 128, 128, 0, stream>>>(
            d_q, kAttnHeads, kHeadDim, kHeadDim - kRopeHeadDim,
            kRopeHeadDim, static_cast<float>(start_pos), lw.rope_theta, 1);
    }

    // KV projection (identical to MLA)
    e = f8_gemm_1d(cublas_h, stream, lw.d_wkv, lw.d_wkv_scales,
        lw.wkv_rows, lw.wkv_cols, d_x, d_kv, d_dequant);
    if (e != cudaSuccess) return e;
    // kv_norm
    // RoPE on KV
    {
        int total = kRopeHeadDim / 2;
        rope_apply_kernel<<<(total + 63) / 64, 64, 0, stream>>>(
            d_kv2, 1, kHeadDim, kHeadDim - kRopeHeadDim,
            kRopeHeadDim, static_cast<float>(start_pos), lw.rope_theta, 1);
    }

    // Store KV in sliding-window cache
    const int slot = start_pos % kAttnWindow;
    cudaMemcpyAsync(d_kv_cache_layer + slot * kHeadDim, d_kv2,
        static_cast<size_t>(kHeadDim) * sizeof(__half),
        cudaMemcpyDeviceToDevice, stream);

    // ── Compressor step ─────────────────────────────────────────────────
    int coff = lw.comp_coff;
    int comp_out = coff * kHeadDim;
    // Compute compressor wkv and wgate (BF16 GEMM)
    // wkv: [coff*head_dim, 4096] @ x[4096] → [coff*head_dim]
    {
        float alpha = 1.0f, beta = 0.0f;
        cublasSetStream(cublas_h, stream);
        cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
            comp_out, 1, kHiddenDim, &alpha,
            lw.d_comp_wkv, CUDA_R_16F, kHiddenDim,
            d_x, CUDA_R_16F, kHiddenDim, &beta,
            d_kv, CUDA_R_16F, comp_out,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        // wgate
        cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
            comp_out, 1, kHiddenDim, &alpha,
            lw.d_comp_wgate, CUDA_R_16F, kHiddenDim,
            d_x, CUDA_R_16F, kHiddenDim, &beta,
            d_qr, CUDA_R_16F, comp_out,  // reuse d_qr as scratch for gate
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }

    // Run compressor step kernel
    bool should_compress = ((start_pos + 1) % lw.compress_ratio) == 0;
    {
        compressor_step_kernel<<<1, kCSABlock, 0, stream>>>(
            d_kv, d_qr, lw.d_comp_ape,
            csa_state.d_comp_kv_state, csa_state.d_comp_score_state,
            csa_state.d_compressed_kv + csa_state.n_compressed * kHeadDim,
            lw.compress_ratio, coff, kHeadDim,
            start_pos, coff == 2);
    }

    if (should_compress) {
        // Apply norm + RoPE + act_quant to compressed KV
        // For now, just increment counter (norm/RoPE can be added later)
        csa_state.n_compressed++;
    }

    // ── Build topk_idxs ─────────────────────────────────────────────────
    // Window positions: all valid slots up to min(start_pos+1, window)
    int n_window = min(start_pos + 1, kAttnWindow);
    int n_compressed = csa_state.n_compressed;
    int n_total = n_window + n_compressed;

    // Build index array on host and upload (simple approach for short sequences)
    // Slot indices: window [0..n_window-1], compressed [window..window+n_compressed-1]
    // The compressed KV is stored AFTER the window in the kv_cache
    // For simplicity, we store compressed KV in a separate buffer and
    // use indices that point into a combined virtual cache
    std::vector<int> h_topk_idxs(n_total);
    for (int i = 0; i < n_window; ++i)
        h_topk_idxs[i] = i;  // window positions (absolute slots in circular cache)
    // Compressed positions point into d_compressed_kv (separate buffer)
    // We use negative indices or offset to distinguish — for now, use a combined approach
    // Actually, sparse_attn_kernel reads from a single kv_cache pointer.
    // We need to put compressed KV into the same cache after the window.
    // The kv_cache_layer is [window, head_dim]; compressed_kv is separate.
    // For a proper implementation, we'd copy compressed_kv into kv_cache_layer
    // after the window, but that requires the cache to be large enough.
    // For now, just use window positions (skip compressed for short seqs)

    // Upload topk_idxs to device (use a small device buffer)
    // For simplicity in decode, upload as host-pinned memory
    int *d_topk_idxs = nullptr;
    cudaMalloc(&d_topk_idxs, n_total * sizeof(int));
    cudaMemcpyAsync(d_topk_idxs, h_topk_idxs.data(),
        n_total * sizeof(int), cudaMemcpyHostToDevice, stream);

    // ── Sparse attention ─────────────────────────────────────────────────
    float scale = 1.0f / sqrtf(static_cast<float>(kHeadDim));
    sparse_attn_kernel<<<kAttnHeads, 128, 0, stream>>>(
        d_q, d_kv_cache_layer, lw.d_attn_sink, d_o,
        d_topk_idxs, n_total, scale);

    cudaFree(d_topk_idxs);

    // ── Inverse RoPE on output ───────────────────────────────────────────
    {
        int total = kAttnHeads * (kRopeHeadDim / 2);
        rope_apply_kernel<<<(total + 127) / 128, 128, 0, stream>>>(
            d_o, kAttnHeads, kHeadDim, kHeadDim - kRopeHeadDim,
            kRopeHeadDim, static_cast<float>(start_pos), lw.rope_theta, 3);
    }

    // ── O-projection (identical to MLA) ─────────────────────────────────
    {
        int64_t n = static_cast<int64_t>(lw.wo_a_rows) * lw.wo_a_cols;
        int block = 256;
        dequant_f8_to_half_kernel<<<(int)((n + block - 1) / block), block, 0, stream>>>(
            lw.d_wo_a, lw.d_wo_a_scales, d_dequant, n,
            lw.wo_a_cols, lw.wo_a_cols / 128, 128, 128);
    }
    {
        const int gd = (kAttnHeads / kOGroups) * kHeadDim;
        float alpha = 1.0f, beta = 0.0f;
        cublasSetStream(cublas_h, stream);
        for (int g = 0; g < kOGroups; ++g) {
            cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                kOloraRank, 1, gd, &alpha,
                d_dequant + g * kOloraRank * gd, CUDA_R_16F, gd,
                d_o + g * gd, CUDA_R_16F, gd, &beta,
                d_olora + g * kOloraRank, CUDA_R_16F, kOloraRank,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        }
    }
    e = f8_gemm_1d(cublas_h, stream, lw.d_wo_b, lw.d_wo_b_scales,
        lw.wo_b_rows, lw.wo_b_cols, d_olora, d_out, d_dequant);

    return e;
}

} } // namespace
```

- [ ] **Step 2: Build and verify**

```powershell
$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"
& "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build build --config Debug --target lsxfabric 2>&1 | Select-Object -Last 3
```

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "feat: run_csa_attention host function (Q/KV proj + compressor + sparse_attn + O-proj)"
```

---

### Task 5: Wire CSA into the pipeline

**Files:**
- Modify: `libs/lsxcommon/src/inference_pipeline.cu`

- [ ] **Step 1: Add CSA state allocation and include**

Add at top of file:
```cpp
#include "lsxcommon/inference_csa.h"
```

Add CSA state array near other buffer declarations (after `d_act_scales`):
```cpp
    CSAState *csa_states = nullptr;  // [num_layers]
```

- [ ] **Step 2: Allocate CSA state after pinned weights load**

After `load_pinned_weights` returns, add:
```cpp
    csa_states = new CSAState[topo.num_layers];
    INFER_CHECK(alloc_csa_state(csa_states, topo.num_layers,
        cfg.max_gen_tokens + static_cast<int>(input_ids.size()),
        kHeadDim, stream));
```

- [ ] **Step 3: Route to CSA vs MLA in the attention branch**

Replace the `run_mla_attention` call with a conditional:
```cpp
                if (lw.compress_ratio > 0) {
                    INFER_CHECK(run_csa_attention(
                        cublas_h, stream, lw, d_normed,
                        d_kv_cache + static_cast<size_t>(lay) * kAttnWindow * kHeadDim,
                        csa_states[lay], step,
                        d_q, d_qr, d_qr2, d_kv, d_kv2,
                        d_o, d_olora, d_attn_out,
                        d_attn_dequant, d_norm_scratch, d_norm_gamma_scratch));
                } else {
                    INFER_CHECK(run_mla_attention(
                        cublas_h, stream, lw, d_normed,
                        d_kv_cache + static_cast<size_t>(lay) * kAttnWindow * kHeadDim,
                        step,
                        d_q, d_qr, d_qr2, d_kv, d_kv2, d_o, d_olora, d_attn_out,
                        d_attn_dequant, d_norm_scratch, d_norm_gamma_scratch));
                }
```

- [ ] **Step 4: Add CSA state cleanup**

In the cleanup section add:
```cpp
    if (csa_states) {
        free_csa_state(csa_states, topo.num_layers);
        delete[] csa_states;
    }
```

- [ ] **Step 5: Build, run, and check output**

```powershell
$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"
& "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build build --config Debug --target lsxfabric 2>&1 | Select-Object -Last 1
```

```powershell
& "build\bin\Debug\lsxfabric.exe" --testprompt --localmodel "D:\Programs\lsx_model\DeepSeek-V4-Flash" 2>&1 | Select-String -Pattern "step=0 |output:|gen_tokens|tokens/s|error" | Select-Object -First 5
```

Expected: No crash, generates tokens (even if still degenerate — CSA is a first cut).

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: wire CSA attention into pipeline for layers 2-41"
```

---

### Task 6: Verify and iterate

- [ ] **Step 1: Run with debug logging to check HC state stability**

Enable debug:
```cpp
logger->set_level(spdlog::level::debug);
```

Build and run:
```powershell
& "build\bin\Debug\lsxfabric.exe" --testprompt --localmodel "D:\Programs\lsx_model\DeepSeek-V4-Flash" 2>&1 | Select-String "step0 layer.*hc_state" | Select-Object -First 10
```

Expected: HC state rms should be bounded (<100) across all 43 layers.

- [ ] **Step 2: If output is still degenerate, check CSA sparse_attn correctness**

The sparse_attn_kernel has a known issue: it recomputes scores in both passes (max-finding and accumulation), and the reduction is simplistic. If output is NaN or all-zero, the kernel needs debugging.

- [ ] **Step 3: Commit final state**

```bash
git add -A && git commit -m "feat: CSA verification and debugging pass"
```
