#include "lsxcommon/inference_csa.h"
#include "lsxcommon/log.h"
#include "inference_load.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstring>
#include <sstream>

namespace lsxcommon {
namespace gpu {

// ── Constants ──────────────────────────────────────────────────────────────
static constexpr int kCSABlock = 256;

// ── Extern from inference_attn.cu ──────────────────────────────────────────
// f8_gemm_1d is static in inference_attn.cu — we duplicate the minimal
// pieces needed (dequant kernel + gemm wrapper).
// Instead, we replicate the GEMM call directly.

// Forward decl of the dequant kernel from inference_attn.cu
extern __global__ void dequant_f8_to_half_kernel(
    const uint8_t* __restrict__ d_w,
    const uint8_t* __restrict__ d_scale,
    __half* __restrict__ d_out,
    int64_t n,
    int64_t C, int64_t SC,
    int64_t block_r, int64_t block_c);

// Forward decl of helpers from inference_attn.cu
extern __global__ void per_head_rms_kernel(__half *data, int rows, int dim);
extern __global__ void rope_apply_kernel(
    __half *data, int rows, int row_stride,
    int offset, int rope_dim, float pos,
    float theta, int yarn_enabled);

// RMSNorm for attention (half in → half out)
extern "C" cudaError_t run_rmsnorm_attn(
    cudaStream_t stream,
    const __half *d_input, const __half *d_weight,
    __half *d_output,
    float *d_scratch, float *d_gamma_scratch,
    int dim, float eps);

// ── Compressor step kernel ──────────────────────────────────────────────────
// Decode path: stores kv/score into state slot, compresses when full.
__global__ void compressor_step_kernel(
    const __half *d_kv,         // [coff*head_dim] wkv output
    const __half *d_score,     // [coff*head_dim] wgate output
    const float  *d_ape,       // [ratio, coff*head_dim]
    float *d_kv_state,         // [ratio, coff*head_dim]
    float *d_score_state,      // [ratio, coff*head_dim]
    __half *d_compressed_kv,  // [head_dim] output
    int ratio, int coff, int head_dim,
    int start_pos)
{
    const int slot = start_pos % ratio;
    const int out_dim = coff * head_dim;
    const int d = head_dim;
    const int tid = threadIdx.x;

    // Phase 1: store kv and score into state
    for (int i = tid; i < out_dim; i += blockDim.x) {
        float ape_val = d_ape[slot * out_dim + i];
        d_kv_state[slot * out_dim + i] = __half2float(d_kv[i]);
        d_score_state[slot * out_dim + i] = __half2float(d_score[i]) + ape_val;
    }
    __syncthreads();

    // Check if we should compress
    bool should_compress = ((start_pos + 1) % ratio) == 0;
    if (!should_compress) return;

    // Phase 2: softmax-weighted sum over ratio slots, output head_dim elements
    for (int i = tid; i < d; i += blockDim.x) {
        // Find max score across ratio slots for numerical stability
        float max_s = -1e30f;
        for (int r = 0; r < ratio; ++r) {
            float s = d_score_state[r * out_dim + i];
            if (s > max_s) max_s = s;
        }
        // Compute softmax weights and weighted sum
        float sum_exp = 0.0f;
        float weights[128]; // max ratio = 128
        for (int r = 0; r < ratio; ++r) {
            float s = d_score_state[r * out_dim + i];
            weights[r] = expf(s - max_s);
            sum_exp += weights[r];
        }
        float compressed = 0.0f;
        for (int r = 0; r < ratio; ++r)
            compressed += d_kv_state[r * out_dim + i] * (weights[r] / (sum_exp + 1e-9f));
        d_compressed_kv[i] = __float2half(compressed);
    }
}

// ── Sparse attention kernel ─────────────────────────────────────────────────
// One block per head. Gathers only selected KV positions.
__global__ void sparse_attn_kernel(
    const __half *d_q,          // [n_heads, head_dim]
    const __half *d_kv,         // [total_slots, head_dim] (window only for now)
    const float  *d_attn_sink,  // [n_heads]
    __half *d_o,                // [n_heads, head_dim] output
    int n_attend,               // number of positions to attend to
    float scale)                // 1/sqrt(head_dim)
{
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    constexpr int kThreads = 128;
    constexpr int kHD = 512; // head_dim

    __shared__ float s_acc[kHD];
    __shared__ float s_max;
    __shared__ float s_sum;

    if (tid < kHD) s_acc[tid] = 0.0f;
    if (tid == 0) { s_max = -1e30f; s_sum = 0.0f; }
    __syncthreads();

    const __half *qrow = d_q + h * kHD;

    // Pass 1: find max score
    __shared__ float s_part[kThreads];
    for (int t = 0; t < n_attend; ++t) {
        const __half *krow = d_kv + t * kHD;
        float partial = 0.0f;
        for (int d = tid; d < kHD; d += kThreads)
            partial += __half2float(qrow[d]) * __half2float(krow[d]);
        s_part[tid] = partial;
        __syncthreads();
        for (int s = kThreads/2; s > 0; s >>= 1) {
            if (tid < s) s_part[tid] += s_part[tid + s];
            __syncthreads();
        }
        if (tid == 0) {
            float sc = s_part[0] * scale;
            if (sc > s_max) s_max = sc;
        }
        __syncthreads();
    }
    // Include sink
    if (tid == 0) {
        float sk = d_attn_sink[h];
        if (sk > s_max) s_max = sk;
    }
    __syncthreads();

    // Pass 2: accumulate weighted output
    for (int t = 0; t < n_attend; ++t) {
        const __half *krow = d_kv + t * kHD;
        float partial = 0.0f;
        for (int d = tid; d < kHD; d += kThreads)
            partial += __half2float(qrow[d]) * __half2float(krow[d]);
        s_part[tid] = partial;
        __syncthreads();
        for (int s = kThreads/2; s > 0; s >>= 1) {
            if (tid < s) s_part[tid] += s_part[tid + s];
            __syncthreads();
        }
        __shared__ float s_weight;
        if (tid == 0) {
            s_weight = expf(s_part[0] * scale - s_max);
            s_sum += s_weight;
        }
        __syncthreads();
        float w = s_weight;
        for (int d = tid; d < kHD; d += kThreads)
            s_acc[d] += w * __half2float(krow[d]);
    }
    // Sink contribution (zero output, just adds to denominator)
    if (tid == 0) s_sum += expf(d_attn_sink[h] - s_max);
    __syncthreads();

    float inv = 1.0f / (s_sum + 1e-9f);
    for (int d = tid; d < kHD; d += kThreads)
        d_o[h * kHD + d] = __float2half(s_acc[d] * inv);
}

// ── Host: allocate CSA state ───────────────────────────────────────────────
cudaError_t alloc_csa_state(CSAState *states, int num_layers,
                             int max_seq_len, int head_dim, cudaStream_t stream)
{
    int max_comp = max_seq_len / 4 + 1;
    for (int l = 0; l < num_layers; ++l) {
        auto &s = states[l];
        int state_sz = 2 * 4 * 2 * 512; // max overlap: 2*4 slots, 2*512 dim
        cudaMalloc(&s.d_comp_kv_state, state_sz * sizeof(float));
        cudaMalloc(&s.d_comp_score_state, state_sz * sizeof(float));
        cudaMemsetAsync(s.d_comp_kv_state, 0, state_sz * sizeof(float), stream);
        cudaMemsetAsync(s.d_comp_score_state, 0, state_sz * sizeof(float), stream);
        cudaMalloc(&s.d_compressed_kv,
            static_cast<size_t>(max_comp) * head_dim * sizeof(__half));
        cudaMemsetAsync(s.d_compressed_kv, 0,
            static_cast<size_t>(max_comp) * head_dim * sizeof(__half), stream);

        int idx_state_sz = 2 * 4 * 2 * 128;
        cudaMalloc(&s.d_idx_kv_state, idx_state_sz * sizeof(float));
        cudaMalloc(&s.d_idx_score_state, idx_state_sz * sizeof(float));
        cudaMemsetAsync(s.d_idx_kv_state, 0, idx_state_sz * sizeof(float), stream);
        cudaMemsetAsync(s.d_idx_score_state, 0, idx_state_sz * sizeof(float), stream);
        cudaMalloc(&s.d_idx_compressed_kv,
            static_cast<size_t>(max_comp) * 128 * sizeof(__half));
        cudaMemsetAsync(s.d_idx_compressed_kv, 0,
            static_cast<size_t>(max_comp) * 128 * sizeof(__half), stream);
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

// ── Helper: F8 GEMM (replicated from inference_attn.cu for static access) ──
static cudaError_t f8_gemm_local(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const uint8_t* d_w, const uint8_t* d_scale,
    int rows, int cols,
    const __half* d_x, __half* d_out, __half* d_scratch)
{
    int64_t n = static_cast<int64_t>(rows) * cols;
    int block = 256;
    dequant_f8_to_half_kernel<<<(int)((n + block - 1) / block), block, 0, stream>>>(
        d_w, d_scale, d_scratch, n, cols, cols / 128, 128, 128);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    cublasSetStream(cublas_h, stream);
    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t st = cublasGemmEx(
        cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
        rows, 1, cols, &alpha,
        d_scratch, CUDA_R_16F, cols,
        d_x, CUDA_R_16F, cols, &beta,
        d_out, CUDA_R_16F, rows,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    return (st == CUBLAS_STATUS_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
}

// ── Host: run_csa_attention ────────────────────────────────────────────────
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
    // ── Q projection (identical to MLA) ──────────────────────────────────
    cudaError_t e = f8_gemm_local(cublas_h, stream, lw.d_wq_a, lw.d_wq_a_scales,
        lw.wq_a_rows, lw.wq_a_cols, d_x, d_qr, d_dequant);
    if (e != cudaSuccess) return e;
    e = run_rmsnorm_attn(stream, d_qr, lw.d_q_norm, d_qr2,
        d_norm_scratch, d_norm_gamma_scratch, kQloraRank, kRmsEps);
    if (e != cudaSuccess) return e;
    e = f8_gemm_local(cublas_h, stream, lw.d_wq_b, lw.d_wq_b_scales,
        lw.wq_b_rows, lw.wq_b_cols, d_qr2, d_q, d_dequant);
    if (e != cudaSuccess) return e;
    per_head_rms_kernel<<<kAttnHeads, 128, 0, stream>>>(d_q, kAttnHeads, kHeadDim);
    {
        int total = kAttnHeads * (kRopeHeadDim / 2);
        rope_apply_kernel<<<(total + 127) / 128, 128, 0, stream>>>(
            d_q, kAttnHeads, kHeadDim, kHeadDim - kRopeHeadDim,
            kRopeHeadDim, static_cast<float>(start_pos), lw.rope_theta, 1);
    }

    // ── KV projection (identical to MLA) ──────────────────────────────────
    e = f8_gemm_local(cublas_h, stream, lw.d_wkv, lw.d_wkv_scales,
        lw.wkv_rows, lw.wkv_cols, d_x, d_kv, d_dequant);
    if (e != cudaSuccess) return e;
    e = run_rmsnorm_attn(stream, d_kv, lw.d_kv_norm, d_kv2,
        d_norm_scratch, d_norm_gamma_scratch, kHeadDim, kRmsEps);
    if (e != cudaSuccess) return e;
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

    // ── Compressor step ──────────────────────────────────────────────────
    int coff = lw.comp_coff;
    int comp_out = coff * kHeadDim;
    // Compute compressor wkv and wgate (BF16 GEMM)
    {
        float alpha = 1.0f, beta = 0.0f;
        cublasSetStream(cublas_h, stream);
        // wkv: [coff*head_dim, hidden] @ x[hidden] → [coff*head_dim]
        cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
            comp_out, 1, kHiddenDim, &alpha,
            lw.d_comp_wkv, CUDA_R_16F, kHiddenDim,
            d_x, CUDA_R_16F, kHiddenDim, &beta,
            d_kv, CUDA_R_16F, comp_out,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        // wgate: reuse d_qr as scratch
        cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
            comp_out, 1, kHiddenDim, &alpha,
            lw.d_comp_wgate, CUDA_R_16F, kHiddenDim,
            d_x, CUDA_R_16F, kHiddenDim, &beta,
            d_qr, CUDA_R_16F, comp_out,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }

    bool should_compress = ((start_pos + 1) % lw.compress_ratio) == 0;
    {
        compressor_step_kernel<<<1, kCSABlock, 0, stream>>>(
            d_kv, d_qr, lw.d_comp_ape,
            csa_state.d_comp_kv_state, csa_state.d_comp_score_state,
            csa_state.d_compressed_kv + csa_state.n_compressed * kHeadDim,
            lw.compress_ratio, coff, kHeadDim, start_pos);
    }
    if (should_compress)
        csa_state.n_compressed++;

    // ── Sparse attention over sliding window ─────────────────────────────
    // For short sequences (< window), all positions are in the window.
    // The compressed KV is supplementary but we start with window-only.
    int n_attend = std::min(start_pos + 1, kAttnWindow);
    float scale = 1.0f / sqrtf(static_cast<float>(kHeadDim));
    sparse_attn_kernel<<<kAttnHeads, 128, 0, stream>>>(
        d_q, d_kv_cache_layer, lw.d_attn_sink, d_o,
        n_attend, scale);

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
    e = f8_gemm_local(cublas_h, stream, lw.d_wo_b, lw.d_wo_b_scales,
        lw.wo_b_rows, lw.wo_b_cols, d_olora, d_out, d_dequant);
    return e;
}

} // namespace gpu
} // namespace lsxcommon
