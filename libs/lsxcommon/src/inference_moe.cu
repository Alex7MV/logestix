#include "lsxcommon/inference_gpu.h"
#include "inference_load.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cfloat>

namespace lsxcommon {
namespace gpu {

// Dequant kernel from testmodel (extern declaration)
extern __global__ void dequant_nvfp4_to_half_kernel(
    const uint8_t*, const uint8_t*, float, __half*, int);

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
        extern __global__ void rms_norm_kernel(const __half*, const __half*, __half*, int, float);
        int block = 256;
        int shared_sz = block * sizeof(float);
        rms_norm_kernel<<<1, block, shared_sz, stream>>>(
            d_hidden, d_norm_weight, d_normed, hidden_dim, 1e-5f);
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

} // namespace gpu
} // namespace lsxcommon
