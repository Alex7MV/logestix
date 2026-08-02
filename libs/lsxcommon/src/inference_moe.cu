#include "lsxcommon/inference_gpu.h"
#include "lsxcommon/shared_experts.h"
#include "lsxcommon/log.h"
#include "inference_load.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cfloat>
#include <sstream>

namespace lsxcommon {
namespace gpu {

// Dequant kernel from testmodel (extern declaration)
extern __global__ void dequant_nvfp4_to_half_kernel(
    const uint8_t* __restrict__, const uint8_t* __restrict__, float,
    __half* __restrict__, int);

// Run a single expert FFN: silu(W1 @ x) * (W3 @ x) then W2 @ result
// All weights NVFP4 packed, dequantized to FP16 scratch
static cudaError_t run_single_expert_ffn(
    cublasHandle_t cublas_h,
    cudaStream_t stream,
    const uint8_t *d_w1, const uint8_t *d_w1_scales, float w1_gs,
    const uint8_t *d_w2, const uint8_t *d_w2_scales, float w2_gs,
    const uint8_t *d_w3, const uint8_t *d_w3_scales, float w3_gs,
    int hidden_dim, int intermediate_dim,
    const __half *d_input,
    __half *d_intermediate,
    __half *d_gate,
    __half *d_w1_fp16_scratch,
    __half *d_w3_fp16_scratch,
    __half *d_output,
    float weight)
{
    cublasSetStream(cublas_h, stream);
    float alpha = 1.0f, beta = 0.0f;

    int w1_el = hidden_dim * intermediate_dim;
    int block = 256;
    int grid_w1 = (w1_el + block - 1) / block;
    int grid_w3 = grid_w1;
    int w2_el = intermediate_dim * hidden_dim;
    int grid_w2 = (w2_el + block - 1) / block;

    // Dequant W1 and W3
    if (grid_w1 > 0) {
        dequant_nvfp4_to_half_kernel<<<grid_w1, block, 0, stream>>>(
            d_w1, d_w1_scales, w1_gs, d_w1_fp16_scratch, w1_el);
        dequant_nvfp4_to_half_kernel<<<grid_w3, block, 0, stream>>>(
            d_w3, d_w3_scales, w3_gs, d_w3_fp16_scratch, w1_el);
    }

    // intermediate = W1^T @ input  (W1: [intermediate_dim, hidden_dim], stored col-major)
    cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                 intermediate_dim, 1, hidden_dim, &alpha,
                 d_w1_fp16_scratch, CUDA_R_16F, hidden_dim,
                 d_input, CUDA_R_16F, hidden_dim, &beta,
                 d_intermediate, CUDA_R_16F, intermediate_dim,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // gate = W3^T @ input
    cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                 intermediate_dim, 1, hidden_dim, &alpha,
                 d_w3_fp16_scratch, CUDA_R_16F, hidden_dim,
                 d_input, CUDA_R_16F, hidden_dim, &beta,
                 d_gate, CUDA_R_16F, intermediate_dim,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // SiLU activation: silu(d_intermediate) * d_gate (element-wise)
    // silu(x) = x * sigmoid(x)
    // We need a small kernel for this
    // For now, assume gemm output is used directly (simplified)

    // Dequant W2
    if (grid_w2 > 0) {
        dequant_nvfp4_to_half_kernel<<<grid_w2, block, 0, stream>>>(
            d_w2, d_w2_scales, w2_gs, d_w1_fp16_scratch, w2_el);
    }

    // output = W2^T @ activated  (W2: [hidden_dim, intermediate_dim])
    cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                 hidden_dim, 1, intermediate_dim, &alpha,
                 d_w1_fp16_scratch, CUDA_R_16F, intermediate_dim,
                 d_intermediate, CUDA_R_16F, intermediate_dim, &beta,
                 d_output, CUDA_R_16F, hidden_dim,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    return cudaGetLastError();
}

// Run router (gate classifier) + shared experts + top-k routed experts
cudaError_t run_moe_layer(
    cublasHandle_t cublas_h,
    cudaStream_t stream,
    const __half *d_hidden,
    const __half *d_norm_weight,
    const __half *d_gate_weight,
    const LayerWeights &lw,
    int hidden_dim,
    int n_routed,
    int top_k,
    int intermediate_dim,
    // ExpertDb access (mmap pointers for routed experts)
    const std::vector<const uint8_t*> &routed_w1,
    const std::vector<const uint8_t*> &routed_w2,
    const std::vector<const uint8_t*> &routed_w3,
    const std::vector<const uint8_t*> &routed_s1,
    const std::vector<const uint8_t*> &routed_s2,
    const std::vector<const uint8_t*> &routed_s3,
    const std::vector<float> &routed_gs1,
    const std::vector<float> &routed_gs2,
    const std::vector<float> &routed_gs3,
    int64_t exp_w_bytes,
    // Scratch buffers
    __half *d_normed,
    __half *d_gate_logits,
    __half *d_intermediate_scratch,
    __half *d_gate_scratch,
    __half *d_dequant_scratch,
    __half *d_moe_output,
    // CPU buffers for routing
    float *h_gate_logits,
    int *h_top_indices,
    float *h_top_weights)
{
    // 1. RMSNorm
    {
        extern __global__ void rms_norm_kernel(const uint16_t*, const uint16_t*, uint16_t*, int, float);
        auto _x = reinterpret_cast<const uint16_t *>(d_hidden);
        auto _w = reinterpret_cast<const uint16_t *>(d_norm_weight);
        auto _o = reinterpret_cast<uint16_t *>(d_normed);
        rms_norm_kernel<<<1, 1, 0, stream>>>(_x, _w, _o, hidden_dim, 1e-5f);
        (void)0;
    }

    // 2. Router: gate_logits = normed @ gate_weight^T
    cublasSetStream(cublas_h, stream);
    float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                 n_routed, 1, hidden_dim, &alpha,
                 d_gate_weight, CUDA_R_16F, hidden_dim,
                 d_normed, CUDA_R_16F, hidden_dim, &beta,
                 d_gate_logits, CUDA_R_16F, n_routed,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // 3. Copy gate logits to CPU for softmax + top-k
    cudaMemcpyAsync(h_gate_logits, d_gate_logits,
                    static_cast<size_t>(n_routed) * sizeof(__half),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // 4. Softmax + top-k on CPU
    float max_val = -FLT_MAX;
    for (int i = 0; i < n_routed; ++i) {
        float v = __half2float(reinterpret_cast<__half*>(h_gate_logits)[i]);
        if (v > max_val) max_val = v;
    }
    float sum_exp = 0.0f;
    float probs[256];
    for (int i = 0; i < n_routed; ++i) {
        probs[i] = expf(__half2float(reinterpret_cast<__half*>(h_gate_logits)[i]) - max_val);
        sum_exp += probs[i];
    }
    float inv_sum = 1.0f / (sum_exp + 1e-10f);
    for (int i = 0; i < n_routed; ++i)
        probs[i] *= inv_sum;

    // Top-k selection
    bool selected[256] = {false};
    for (int r = 0; r < top_k; ++r) {
        float best_val = -1.0f;
        int best_idx = -1;
        for (int i = 0; i < n_routed; ++i) {
            if (!selected[i] && probs[i] > best_val) {
                best_val = probs[i];
                best_idx = i;
            }
        }
        if (best_idx >= 0) {
            selected[best_idx] = true;
            h_top_indices[r] = best_idx;
            h_top_weights[r] = best_val;
        }
    }

    // 5. Accumulate output: start with shared experts
    // Zero out moe_output
    cudaMemsetAsync(d_moe_output, 0, static_cast<size_t>(hidden_dim) * sizeof(__half), stream);

    // 6. Shared experts FFN (d_shared_w1/2/3 are NVFP4 packed)
    if (lw.d_shared_w1 && lw.shared_w1_bytes > 0) {
        // Compute intermediate dim for shared from byte count
        int shared_inter = static_cast<int>((lw.shared_w1_bytes * 2) / hidden_dim);
        // We need GPU device buffers for shared weights - they're already in VRAM
        // (loaded by inference_load.cu via cudaMemcpyAsync)
        // But d_shared_w1/2/3 are device pointers
        run_single_expert_ffn(cublas_h, stream,
            lw.d_shared_w1, lw.d_shared_w1_scales, lw.shared_w1_global_scale,
            lw.d_shared_w2, lw.d_shared_w2_scales, lw.shared_w2_global_scale,
            lw.d_shared_w3, lw.d_shared_w3_scales, lw.shared_w3_global_scale,
            hidden_dim, shared_inter,
            d_normed,
            d_intermediate_scratch, d_gate_scratch,
            d_dequant_scratch, d_dequant_scratch,
            d_moe_output, 1.0f);
    }

    // 7. Routed experts (simplified for Phase 1 — one expert at a time)
    // For each top-k expert, copy host mmap → fixed GPU slot, dequant + GEMM, weighted add
    // This requires the ping-pong DMA pattern from testmodel
    // For now, stub: skip routed experts in Phase 1

    return cudaSuccess;
}

// Router: gate @ hidden → CPU softmax + top-k → h_topk_indices
cudaError_t run_router(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const __half *d_gate, const __half *d_hidden,
    int hidden_dim, int num_routed,
    int *h_topk_indices, int topk)
{
    cublasSetStream(cublas_h, stream);
    float alpha = 1.0f, beta = 0.0f;

    __half *d_logits = nullptr;
    cudaError_t e = cudaMalloc(&d_logits, static_cast<size_t>(num_routed) * sizeof(__half));
    if (e != cudaSuccess) return e;

    cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                 num_routed, 1, hidden_dim, &alpha,
                 d_gate, CUDA_R_16F, hidden_dim,
                 d_hidden, CUDA_R_16F, hidden_dim, &beta,
                 d_logits, CUDA_R_16F, num_routed,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    float *h_logits = nullptr;
    e = cudaMallocHost(&h_logits, static_cast<size_t>(num_routed) * sizeof(float));
    if (e != cudaSuccess) { cudaFree(d_logits); return e; }

    cudaMemcpyAsync(h_logits, d_logits,
                    static_cast<size_t>(num_routed) * sizeof(__half),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Convert half → float, softmax, top-k
    float max_val = -FLT_MAX;
    for (int i = 0; i < num_routed; ++i) {
        float v = __half2float(reinterpret_cast<__half*>(h_logits)[i]);
        if (v > max_val) max_val = v;
    }
    float sum_exp = 0.0f;
    float probs[256];
    for (int i = 0; i < num_routed; ++i) {
        probs[i] = expf(__half2float(reinterpret_cast<__half*>(h_logits)[i]) - max_val);
        sum_exp += probs[i];
    }
    float inv_sum = 1.0f / (sum_exp + 1e-10f);
    for (int i = 0; i < num_routed; ++i)
        probs[i] *= inv_sum;

    bool selected[256] = {false};
    for (int r = 0; r < (topk > 256 ? 256 : topk); ++r) {
        float best_val = -1.0f;
        int best_idx = -1;
        for (int i = 0; i < num_routed; ++i) {
            if (!selected[i] && probs[i] > best_val) {
                best_val = probs[i];
                best_idx = i;
            }
        }
        if (best_idx >= 0) {
            selected[best_idx] = true;
            h_topk_indices[r] = best_idx;
        }
    }

    cudaFreeHost(h_logits);
    cudaFree(d_logits);
    return cudaSuccess;
}

// SiLU activation: silu(x) = x * sigmoid(x)
__global__ void silu_mul_kernel(__half *d_act, const __half *d_gate, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = __half2float(d_act[i]);
    float g = __half2float(d_gate[i]);
    float sig = 1.0f / (1.0f + expf(-x));
    d_act[i] = __float2half(x * sig * g);
}

// SwiGLU with swiglu_limit: gate clamped to (-inf, limit], up to [-limit, limit].
// d_gate_vec = silu(clamp(d_gate_vec, max=limit)) * clamp(d_up_vec, +/-limit)
__global__ void swiglu_clamp_kernel(
    __half *d_gate_vec, const __half *d_up_vec, int n, float limit)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = __half2float(d_gate_vec[i]);
    float u = __half2float(d_up_vec[i]);
    if (g > limit) g = limit;
    if (u >  limit) u =  limit;
    if (u < -limit) u = -limit;
    float sig = 1.0f / (1.0f + expf(-g));
    d_gate_vec[i] = __float2half(sig * g * u);
}

// Shared experts FFN: dequant NVFP4 → FP16, GEMM, SiLU activation
cudaError_t run_shared_experts_ffn(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const __half *d_hidden,
    const uint8_t *d_w1, const uint8_t *d_w2, const uint8_t *d_w3,
    const uint8_t *d_w1_scales, const uint8_t *d_w2_scales, const uint8_t *d_w3_scales,
    float w1_gs, float w2_gs, float w3_gs,
    int64_t w1_bytes, int64_t w2_bytes, int64_t w3_bytes,
    __half *d_intermediate, __half *d_gate, __half *d_dequant, __half *d_output)
{
    cublasSetStream(cublas_h, stream);
    float alpha = 1.0f, beta = 0.0f;

    int hidden_dim = kHiddenDim;
    int intermediate_dim = w1_bytes > 0
        ? static_cast<int>((w1_bytes * 2) / hidden_dim) : kIntermediateDim;
    int w1_el = hidden_dim * intermediate_dim;

    // 1. Dequant W1 → d_dequant
    if (w1_bytes > 0) {
        int64_t scales = (w1_el + kVFP4BlockSize - 1) / kVFP4BlockSize;
        cudaError_t e = dequant_shared_nvfp4(stream, d_w1, w1_bytes,
            d_w1_scales, scales, w1_gs, d_dequant, w1_el);
        if (e != cudaSuccess) return e;
    }
    {
        cudaStreamSynchronize(stream);
        __half probe[6] = {};
        cudaMemcpy(probe, d_dequant, 6 * sizeof(__half), cudaMemcpyDeviceToHost);
        std::ostringstream oss;
        oss << "[dbg] FFN W1 dequant[0..5]=";
        for (int i = 0; i < 6; ++i) oss << " " << __half2float(probe[i]);
        log::debug(oss.str());
    }

    // 2. intermediate = W1^T @ hidden → d_intermediate
    cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                 intermediate_dim, 1, hidden_dim, &alpha,
                 d_dequant, CUDA_R_16F, hidden_dim,
                 d_hidden, CUDA_R_16F, hidden_dim, &beta,
                 d_intermediate, CUDA_R_16F, intermediate_dim,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    {
        cudaStreamSynchronize(stream);
        __half probe[6] = {};
        cudaMemcpy(probe, d_intermediate, 6 * sizeof(__half), cudaMemcpyDeviceToHost);
        std::ostringstream oss;
        oss << "[dbg] FFN inter[0..5]=";
        for (int i = 0; i < 6; ++i) oss << " " << __half2float(probe[i]);
        log::debug(oss.str());
    }

    // 3. Dequant W3 → d_dequant (reuse)
    if (w3_bytes > 0) {
        int64_t scales = (w1_el + kVFP4BlockSize - 1) / kVFP4BlockSize;
        cudaError_t e = dequant_shared_nvfp4(stream, d_w3, w3_bytes,
            d_w3_scales, scales, w3_gs, d_dequant, w1_el);
        if (e != cudaSuccess) return e;
    }

    // 4. gate = W3^T @ hidden → d_gate
    cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                 intermediate_dim, 1, hidden_dim, &alpha,
                 d_dequant, CUDA_R_16F, hidden_dim,
                 d_hidden, CUDA_R_16F, hidden_dim, &beta,
                 d_gate, CUDA_R_16F, intermediate_dim,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    {
        cudaStreamSynchronize(stream);
        __half probe[6] = {};
        cudaMemcpy(probe, d_gate, 6 * sizeof(__half), cudaMemcpyDeviceToHost);
        std::ostringstream oss;
        oss << "[dbg] FFN gate[0..5]=";
        for (int i = 0; i < 6; ++i) oss << " " << __half2float(probe[i]);
        log::debug(oss.str());
    }

    // 5. SiLU + multiply: d_intermediate = silu(d_intermediate) * d_gate
    {
        int block = 256;
        int grid = (intermediate_dim + block - 1) / block;
        silu_mul_kernel<<<grid, block, 0, stream>>>(d_intermediate, d_gate, intermediate_dim);
    }

    // 6. Dequant W2 → d_dequant (reuse)
    int w2_el = intermediate_dim * hidden_dim;
    if (w2_bytes > 0) {
        int64_t scales = (w2_el + kVFP4BlockSize - 1) / kVFP4BlockSize;
        cudaError_t e = dequant_shared_nvfp4(stream, d_w2, w2_bytes,
            d_w2_scales, scales, w2_gs, d_dequant, w2_el);
        if (e != cudaSuccess) return e;
    }

    // 7. output = W2^T @ activated → d_output
    cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                 hidden_dim, 1, intermediate_dim, &alpha,
                 d_dequant, CUDA_R_16F, intermediate_dim,
                 d_intermediate, CUDA_R_16F, intermediate_dim, &beta,
                 d_output, CUDA_R_16F, hidden_dim,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    {
        cudaStreamSynchronize(stream);
        __half probe[6] = {};
        cudaMemcpy(probe, d_output, 6 * sizeof(__half), cudaMemcpyDeviceToHost);
        std::ostringstream oss;
        oss << "[dbg] FFN out[0..5]=";
        for (int i = 0; i < 6; ++i) oss << " " << __half2float(probe[i]);
        log::debug(oss.str());
    }
    return cudaSuccess;
}

// Routed-expert FFN: fp4 (E2M1 × F8_E8M0 per-32) dequant + SwiGLU GEMM.
// The packed bytes + scales are staged into device buffers by the caller.
static cudaError_t run_routed_expert_ffn(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const __half *d_hidden,
    const uint8_t *d_w1, const uint8_t *d_w1_scales, int64_t w1_el,
    const uint8_t *d_w2, const uint8_t *d_w2_scales, int64_t w2_el,
    const uint8_t *d_w3, const uint8_t *d_w3_scales, int64_t w3_el,
    __half *d_gate_vec, __half *d_up_vec, __half *d_dequant,
    __half *d_output,
    float weight, bool accumulate)
{
    cublasSetStream(cublas_h, stream);
    float alpha = 1.0f, beta = 0.0f;

    // gate = W1^T @ x  (W1: [inter, hidden], packed [inter, hidden/2])
    if (w1_el > 0) {
        int64_t packed_bytes = (w1_el + 1) / 2;
        int64_t scale_bytes  = (w1_el + 31) / 32;
        dequant_fp4_e8m0(stream, d_w1, packed_bytes, d_w1_scales, scale_bytes,
                         d_dequant, w1_el);
        cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                     kIntermediateDim, 1, kHiddenDim, &alpha,
                     d_dequant, CUDA_R_16F, kHiddenDim,
                     d_hidden, CUDA_R_16F, kHiddenDim, &beta,
                     d_gate_vec, CUDA_R_16F, kIntermediateDim,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }

    // up = W3^T @ x
    if (w3_el > 0) {
        int64_t packed_bytes = (w3_el + 1) / 2;
        int64_t scale_bytes  = (w3_el + 31) / 32;
        dequant_fp4_e8m0(stream, d_w3, packed_bytes, d_w3_scales, scale_bytes,
                         d_dequant, w3_el);
        cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                     kIntermediateDim, 1, kHiddenDim, &alpha,
                     d_dequant, CUDA_R_16F, kHiddenDim,
                     d_hidden, CUDA_R_16F, kHiddenDim, &beta,
                     d_up_vec, CUDA_R_16F, kIntermediateDim,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }

    // SwiGLU: d_gate_vec = silu(clamp(gate, max=10)) * clamp(up, +/-10)
    if (kIntermediateDim > 0) {
        int block = 256;
        int grid = (kIntermediateDim + block - 1) / block;
        swiglu_clamp_kernel<<<grid, block, 0, stream>>>(
            d_gate_vec, d_up_vec, kIntermediateDim, 10.0f);
    }

    // output += weight * (W2^T @ activated)  (W2: [hidden, inter], packed [hidden, inter/2])
    if (w2_el > 0) {
        int64_t packed_bytes = (w2_el + 1) / 2;
        int64_t scale_bytes  = (w2_el + 31) / 32;
        dequant_fp4_e8m0(stream, d_w2, packed_bytes, d_w2_scales, scale_bytes,
                         d_dequant, w2_el);
        float galpha = weight, gbeta = accumulate ? 1.0f : 0.0f;
        cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                     kHiddenDim, 1, kIntermediateDim, &galpha,
                     d_dequant, CUDA_R_16F, kIntermediateDim,
                     d_gate_vec, CUDA_R_16F, kIntermediateDim, &gbeta,
                     d_output, CUDA_R_16F, kHiddenDim,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }

    return cudaGetLastError();
}

// Routes one token: gate GEMM → sqrt(softplus) scores → top-k (bias only for
// selection; hash layers 0..2 use tid2eid) → norm_topk_prob → ×route_scale.
cudaError_t route_token(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const __half *d_gate, const __half *d_hidden,
    int hidden_dim, int num_routed,
    const int64_t *h_tid2eid, int tid2eid_stride, int token_id,
    const float *h_gate_bias,
    int topk,
    int *h_top_indices, float *h_top_weights)
{
    constexpr float kRouteScale = 1.5f;

    cublasSetStream(cublas_h, stream);
    float alpha = 1.0f, beta = 0.0f;

    __half *d_logits = nullptr;
    cudaError_t e = cudaMalloc(&d_logits, static_cast<size_t>(num_routed) * sizeof(__half));
    if (e != cudaSuccess) return e;

    cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                 num_routed, 1, hidden_dim, &alpha,
                 d_gate, CUDA_R_16F, hidden_dim,
                 d_hidden, CUDA_R_16F, hidden_dim, &beta,
                 d_logits, CUDA_R_16F, num_routed,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    float *h_logits = nullptr;
    e = cudaMallocHost(&h_logits, static_cast<size_t>(num_routed) * sizeof(float));
    if (e != cudaSuccess) { cudaFree(d_logits); return e; }

    cudaMemcpyAsync(h_logits, d_logits,
                    static_cast<size_t>(num_routed) * sizeof(__half),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // scores[i] = sqrt(softplus(logit[i]))  (unbiased)
    float scores[256];
    for (int i = 0; i < num_routed; ++i) {
        float x = __half2float(reinterpret_cast<__half*>(h_logits)[i]);
        scores[i] = sqrtf(log1pf(expf(x)));
    }

    // Selection scores: biased (bias is null for hash layers)
    float sel[256];
    for (int i = 0; i < num_routed; ++i)
        sel[i] = scores[i] + (h_gate_bias ? h_gate_bias[i] : 0.0f);

    if (h_tid2eid) {
        // Hash layer: expert indices predetermined per token id.
        if (tid2eid_stride <= 0) tid2eid_stride = topk;
        int64_t base = static_cast<int64_t>(token_id) * tid2eid_stride;
        for (int k = 0; k < topk; ++k)
            h_top_indices[k] = static_cast<int>(h_tid2eid[base + k]);
    } else {
        bool selected[256] = {false};
        for (int k = 0; k < topk; ++k) {
            float best_val = -FLT_MAX;
            int best_idx = -1;
            for (int i = 0; i < num_routed; ++i) {
                if (!selected[i] && sel[i] > best_val) {
                    best_val = sel[i];
                    best_idx = i;
                }
            }
            if (best_idx >= 0) {
                selected[best_idx] = true;
                h_top_indices[k] = best_idx;
            }
        }
    }

    // Routing weights from unbiased scores, normalized, scaled by route_scale.
    float sum_w = 0.0f;
    for (int k = 0; k < topk; ++k) {
        float w = scores[h_top_indices[k]];
        h_top_weights[k] = w;
        sum_w += w;
    }
    float inv = 1.0f / (sum_w + 1e-9f);
    for (int k = 0; k < topk; ++k)
        h_top_weights[k] *= inv * kRouteScale;

    cudaFreeHost(h_logits);
    cudaFree(d_logits);
    return cudaSuccess;
}

// Runs the top-k routed experts for one token and accumulates their weighted
// outputs into d_output (which must already hold the shared-expert result).
// Expert host pointers come from the mmap'd ExpertDb; all 6 tensors of an
// expert are staged into device scratch buffers at fixed offsets.
cudaError_t run_routed_experts(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const __half *d_normed,
    const ExpertEntry *db_entries, int layer, int experts_per_layer,
    const int *h_top_indices, const float *h_top_weights, int topk,
    __half *d_output,
    __half *d_gate_vec, __half *d_up_vec, __half *d_dequant,
    uint8_t *d_pack_stage, uint8_t *d_scale_stage,
    int64_t pack_stage_bytes, int64_t scale_stage_bytes)
{
    const int64_t w1_el = static_cast<int64_t>(kHiddenDim) * kIntermediateDim;
    const int64_t w2_el = static_cast<int64_t>(kIntermediateDim) * kHiddenDim;
    const int64_t w3_el = w1_el;

    bool accumulate = false;
    for (int k = 0; k < topk; ++k) {
        int idx = h_top_indices[k];
        if (idx < 0 || idx >= experts_per_layer) continue;
        const ExpertEntry &en = db_entries[
            static_cast<size_t>(layer) * experts_per_layer + idx];
        if (!en.host_w1 || !en.host_w2 || !en.host_w3) continue;

        const int64_t w1b = en.w1_bytes, w2b = en.w2_bytes, w3b = en.w3_bytes;
        const int64_t s1b = en.w1_scales_bytes, s2b = en.w2_scales_bytes, s3b = en.w3_scales_bytes;
        if (w1b <= 0 || w2b <= 0 || w3b <= 0 || s1b <= 0 || s2b <= 0 || s3b <= 0)
            continue;
        if (w1b + w2b + w3b > pack_stage_bytes || s1b + s2b + s3b > scale_stage_bytes)
            continue;

        uint8_t *p1 = d_pack_stage;
        uint8_t *p2 = d_pack_stage + w1b;
        uint8_t *p3 = d_pack_stage + w1b + w2b;
        uint8_t *s1 = d_scale_stage;
        uint8_t *s2 = d_scale_stage + s1b;
        uint8_t *s3 = d_scale_stage + s1b + s2b;

        cudaError_t e = cudaSuccess;
        e = cudaMemcpyAsync(p1, en.host_w1, static_cast<size_t>(w1b),
                            cudaMemcpyHostToDevice, stream);
        if (e != cudaSuccess) return e;
        e = cudaMemcpyAsync(p2, en.host_w2, static_cast<size_t>(w2b),
                            cudaMemcpyHostToDevice, stream);
        if (e != cudaSuccess) return e;
        e = cudaMemcpyAsync(p3, en.host_w3, static_cast<size_t>(w3b),
                            cudaMemcpyHostToDevice, stream);
        if (e != cudaSuccess) return e;
        e = cudaMemcpyAsync(s1, en.host_w1_scales, static_cast<size_t>(s1b),
                            cudaMemcpyHostToDevice, stream);
        if (e != cudaSuccess) return e;
        e = cudaMemcpyAsync(s2, en.host_w2_scales, static_cast<size_t>(s2b),
                            cudaMemcpyHostToDevice, stream);
        if (e != cudaSuccess) return e;
        e = cudaMemcpyAsync(s3, en.host_w3_scales, static_cast<size_t>(s3b),
                            cudaMemcpyHostToDevice, stream);
        if (e != cudaSuccess) return e;

        e = run_routed_expert_ffn(cublas_h, stream, d_normed,
            p1, s1, w1_el,
            p2, s2, w2_el,
            p3, s3, w3_el,
            d_gate_vec, d_up_vec, d_dequant, d_output,
            h_top_weights[k], accumulate);
        if (e != cudaSuccess) return e;
        accumulate = true;
    }
    return cudaSuccess;
}

} // namespace gpu
} // namespace lsxcommon
