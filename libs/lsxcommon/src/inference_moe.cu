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

// ── Per-128-block activation quantization (matching reference act_quant) ─────
// Computes per-128-element block scales, quantizes activations to FP8_E4M3,
// and stores both the quantized activations (as FP16) and the block scales.
//
// Mathematical invariant (matching reference kernel.py act_quant_kernel):
//   amax = max(abs(x[blk*128 : blk*128+128]))
//   act_scale = max(amax, 1e-4) / 448.0
//   x_fp8 = clamp(x / act_scale, -448, 448) → E4M3 round-trip → FP16

__global__ void act_quant_block_kernel(
    const __half* __restrict__ d_input,
    float* __restrict__        d_output,
    float* __restrict__        d_scales,
    int K)
{
    constexpr int BLK = 128;
    const int blk = blockIdx.x;
    const int base = blk * BLK;
    if (base >= K) return;
    const int n = min(BLK, K - base);

    __shared__ float s_amax;
    if (threadIdx.x == 0) s_amax = 0.0f;
    __syncthreads();

    float local_max = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = fabsf(__half2float(d_input[base + i]));
        local_max = fmaxf(local_max, v);
    }
    __shared__ float s_reduce[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, offset));
    if (lane == 0) s_reduce[warp] = local_max;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < (blockDim.x + 31) / 32) ? s_reduce[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, offset));
        if (threadIdx.x == 0) s_amax = v;
    }
    __syncthreads();

    float linear_scale = fmaxf(s_amax, 1e-4f) / 448.0f;
    int log2_s = 0;
    float vs = linear_scale;
    while (vs > 1.0f) { vs *= 0.5f; log2_s++; }
    while (vs <= 0.5f && log2_s > -149) { vs *= 2.0f; log2_s--; }
    float act_scale = exp2f(static_cast<float>(log2_s));
    if (act_scale < linear_scale) act_scale *= 2.0f;
    float inv_scale = 1.0f / act_scale;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = __half2float(d_input[base + i]) * inv_scale;
        v = fmaxf(-448.0f, fminf(448.0f, v));
        // FP8_E4M3 quantize → dequant back to float (no FP16 intermediate)
        uint32_t bits;
        std::memcpy(&bits, &v, sizeof(bits));
        uint32_t sign = (bits >> 31) & 1;
        int32_t  exp32 = (bits >> 23) & 0xFF;
        uint32_t mant3 = (bits >> 20) & 0x7;
        int e4 = exp32 - 127 + 7;
        if (e4 < 1) e4 = 0;
        if (e4 > 15) { e4 = 15; mant3 = 7; }
        if (e4 == 0) {
            d_output[base + i] = 0.0f;
        } else {
            uint32_t out_bits = (sign << 31) | (static_cast<uint32_t>(e4 + 120) << 23) | (mant3 << 20);
            float result;
            std::memcpy(&result, &out_bits, sizeof(result));
            d_output[base + i] = result;
        }
    }
    if (threadIdx.x == 0) d_scales[blk] = act_scale;
}

// Fuse per-128 act_scale into dequanted FP16 weights (per-32 weight scale already baked).
// After this, the GEMM naturally produces: Σ_k (E2M1*E8M0*act_scale) * x_quantized
//                                         = Σ_k E2M1*E8M0*act_scale * (x/act_scale)
//                                         = Σ_k E2M1*E8M0 * x  (correct)
__global__ void fuse_act_scale_into_weights_kernel(
    float* __restrict__ d_weights,
    const float* __restrict__ d_act_scales,
    int64_t N, int64_t K)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = N * K;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (; i < total; i += stride) {
        int64_t k = i % K;
        float act_scale = d_act_scales[k / 128];
        d_weights[i] *= act_scale;
    }
}

// Clamp FP16 output to prevent overflow (used after GEMM)
__global__ void clamp_half_kernel(__half* d_buf, int n, float limit)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __half2float(d_buf[i]);
    v = fmaxf(-limit, fminf(limit, v));
    d_buf[i] = __float2half(v);
}

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

// SiLU with swiglu_limit: silu(clamp(gate, max=limit)) * clamp(up, ±limit)
// Matches reference: gate = clamp(gate, max=limit); up = clamp(up, -limit, limit)
// FP32 output version — SwiGLU computed in FP32 for stability (reference: .float())
__global__ void silu_mul_kernel(__half *d_act, const __half *d_gate, int n, float limit)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = __half2float(d_act[i]);
    float g = __half2float(d_gate[i]);
    if (x > limit) x = limit;
    if (g > limit) g = limit;
    if (g < -limit) g = -limit;
    float sig = 1.0f / (1.0f + expf(-x));
    d_act[i] = __float2half(x * sig * g);
}

// SwiGLU with clamp, FP32 inputs → FP32 output (no FP16 intermediate overflow)
__global__ void swiglu_clamp_f32f_kernel(
    const float* __restrict__ d_gate,
    const float* __restrict__ d_up,
    float* __restrict__ d_out, int n, float limit)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = d_gate[i];
    float u = d_up[i];
    if (g > limit) g = limit;
    if (u >  limit) u =  limit;
    if (u < -limit) u = -limit;
    float sig = 1.0f / (1.0f + expf(-g));
    d_out[i] = sig * g * u;
}

// Apply scalar multiplier to a float buffer (in-place)
__global__ void scale_float_kernel(float* d_buf, float scalar, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    d_buf[i] *= scalar;
}

// Convert float buffer → half buffer
__global__ void float_to_half_kernel(const float* __restrict__ src,
                                      __half* __restrict__ dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2half(src[i]);
}

// Convert half buffer → float buffer
__global__ void half_to_float_kernel(const __half* __restrict__ src,
                                     float* __restrict__ dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __half2float(src[i]);
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

// Shared experts FFN: dequant F8_E4M3 x F8_E8M0 (128x128 tile) -> FP32,
// GEMM, SiLU activation. Verified against the real DeepSeek-V4-Flash
// safetensors header: shared_experts.w1.weight dtype=F8_E4M3 shape=[2048,4096],
// w1.scale dtype=F8_E8M0 shape=[16,32] -> block_r=128, block_c=128. global_scale
// is not applicable here (converter.cpp's passthrough path hardcodes 1.0 for
// already-quantized F8_E4M3 sources), which is why w1_gs/w2_gs/w3_gs are
// accepted but intentionally unused below.
// W1/W3 GEMM outputs FP32 (CUDA_R_32F) to prevent FP16 overflow.
cudaError_t run_shared_experts_ffn(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const __half *d_hidden,
    const uint8_t *d_w1, const uint8_t *d_w2, const uint8_t *d_w3,
    const uint8_t *d_w1_scales, const uint8_t *d_w2_scales, const uint8_t *d_w3_scales,
    float /*w1_gs*/, float /*w2_gs*/, float /*w3_gs*/,
    int64_t w1_bytes, int64_t w2_bytes, int64_t w3_bytes,
    float *d_intermediate_f32, float *d_gate_f32,   // FP32 W1/W3 outputs
    float *d_dequant, __half *d_output)
{
    cublasSetStream(cublas_h, stream);
    float alpha = 1.0f, beta = 0.0f;

    int hidden_dim = kHiddenDim;
    // F8_E4M3: 1 byte per element. intermediate_dim = n_elements / hidden_dim
    int intermediate_dim = w1_bytes > 0
        ? static_cast<int>(w1_bytes / hidden_dim) : kIntermediateDim;
    int w1_el = hidden_dim * intermediate_dim;

    // 1. Dequant W1 → d_dequant (F8_E4M3 × E8M0, per-128x128 tiles)
    if (w1_bytes > 0) {
        // dequant_shared_f8 does not validate/use its scale_bytes parameter
        // (kept for interface symmetry with the other dequant_* functions).
        cudaError_t e = dequant_shared_f8(stream, d_w1, w1_bytes,
            d_w1_scales, 0, d_dequant, w1_el, hidden_dim);
        if (e != cudaSuccess) return e;
    }

    // FP32 copy of activation (pure FP32 GEMMs; cuBLAS rejects F32-in x F16-in).
    // Lives in d_dequant tail after the W1 weights region.
    float *d_hidden_f32 = d_dequant + w1_el;
    {
        int b = 256;
        int g = (hidden_dim + b - 1) / b;
        half_to_float_kernel<<<g, b, 0, stream>>>(d_hidden, d_hidden_f32, hidden_dim);
    }

    // 2. intermediate = W1^T @ hidden → d_intermediate
    {
        cublasStatus_t w1s = cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                     intermediate_dim, 1, hidden_dim, &alpha,
                     d_dequant, CUDA_R_32F, hidden_dim,
                      d_hidden_f32, CUDA_R_32F, hidden_dim, &beta,
                      d_intermediate_f32, CUDA_R_32F, intermediate_dim,
                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        static int pcount = 0;
        if (pcount == 42) {
            cudaStreamSynchronize(stream);
            std::vector<float> ii(intermediate_dim);
            cudaMemcpy(ii.data(), d_intermediate_f32, intermediate_dim*sizeof(float), cudaMemcpyDeviceToHost);
            double si=0, mi=0; for (auto v:ii){si+=(double)v*v; if(std::fabs(v)>mi)mi=std::fabs(v);}
            std::vector<__half> hh(hidden_dim);
            cudaMemcpy(hh.data(), d_hidden, hidden_dim*sizeof(__half), cudaMemcpyDeviceToHost);
            std::vector<float> hhf(hidden_dim);
            for (int i=0;i<hidden_dim;++i) hhf[i]=__half2float(hh[i]);
            std::vector<float> ww(static_cast<size_t>(hidden_dim)*intermediate_dim);
            cudaMemcpy(ww.data(), d_dequant, ww.size()*sizeof(float), cudaMemcpyDeviceToHost);
            {
                FILE*f=fopen("C:/Users/aumoz/AppData/Local/Temp/opencode/l42_in.bin","wb");
                fwrite(hhf.data(), sizeof(float), hidden_dim, f); fclose(f);
                f=fopen("C:/Users/aumoz/AppData/Local/Temp/opencode/l42_w1.bin","wb");
                fwrite(ww.data(), sizeof(float), ww.size(), f); fclose(f);
                f=fopen("C:/Users/aumoz/AppData/Local/Temp/opencode/l42_inter.bin","wb");
                fwrite(ii.data(), sizeof(float), intermediate_dim, f); fclose(f);
                // raw W1 bytes + scale bytes
                size_t raw_sz = static_cast<size_t>(w1_bytes);
                std::vector<uint8_t> raw(raw_sz);
                cudaMemcpy(raw.data(), d_w1, raw_sz, cudaMemcpyDeviceToHost);
                f=fopen("C:/Users/aumoz/AppData/Local/Temp/opencode/l42_w1raw.bin","wb");
                fwrite(raw.data(), 1, raw_sz, f); fclose(f);
                const int64_t sc_sz = 16*32;
                std::vector<uint8_t> sc(sc_sz);
                cudaMemcpy(sc.data(), d_w1_scales, sc_sz, cudaMemcpyDeviceToHost);
                f=fopen("C:/Users/aumoz/AppData/Local/Temp/opencode/l42_w1sc.bin","wb");
                fwrite(sc.data(), 1, sc_sz, f); fclose(f);
            }
            double sw=0, mw=0; for (auto v:ww){sw+=(double)v*v; float a=std::fabs(v); if(a>mw)mw=a;}
            float ir = std::sqrt((float)(si / intermediate_dim));
            float hr = std::sqrt((float)(0.0));
            float wr = std::sqrt((float)(sw / ww.size()));
            log::info("[SPLIT] W1 deq rms=" + std::to_string(wr) + " max=" + std::to_string(mw)
                      + " inter_rms=" + std::to_string(ir) + " inter_max=" + std::to_string(mi)
                      + " in_rms=" + std::to_string(hr));
        }
        ++pcount;
        (void)w1s;
    }

    // 3. Dequant W3 → d_dequant (reuse, F8_E4M3 × E8M0)
    if (w3_bytes > 0) {
        cudaError_t e = dequant_shared_f8(stream, d_w3, w3_bytes,
            d_w3_scales, 0, d_dequant, w1_el, hidden_dim);
        if (e != cudaSuccess) return e;
    }

    // 4. gate = W3^T @ hidden → d_gate
    cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                 intermediate_dim, 1, hidden_dim, &alpha,
                 d_dequant, CUDA_R_32F, hidden_dim,
                  d_hidden_f32, CUDA_R_32F, hidden_dim, &beta,
                  d_gate_f32, CUDA_R_32F, intermediate_dim,
                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // 5. SiLU + multiply: d_intermediate = silu(d_intermediate) * d_gate (FP32)
    {
        int block = 256;
        int grid = (intermediate_dim + block - 1) / block;
        swiglu_clamp_f32f_kernel<<<grid, block, 0, stream>>>(
            d_intermediate_f32, d_gate_f32, d_intermediate_f32,
            intermediate_dim, 10.0f);
    }

    // 6. Dequant W2 → d_dequant (reuse)
    int w2_el = intermediate_dim * hidden_dim;
    if (w2_bytes > 0) {
        // W2: [hidden, intermediate] → cols = intermediate_dim
        cudaError_t e = dequant_shared_f8(stream, d_w2, w2_bytes,
            d_w2_scales, 0, d_dequant, w2_el, intermediate_dim);
        if (e != cudaSuccess) return e;
    }

    // 7. output = W2^T @ activated → d_output (FP16 output for hc_post)
    // Pure FP32 GEMM (FP32 weights × FP32 activation → FP32) then convert to
    // FP16, matching the routed-expert W2 pattern. cuBLAS rejects F32×F16.
    {
        int b = 256;
        // FP32 W2 output scratch located after [W2 weights][(reserved) act fp16]
        float *d_w2out_f32 = d_dequant + w2_el + intermediate_dim / 2;
        cublasStatus_t wbst = cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                     hidden_dim, 1, intermediate_dim, &alpha,
                     d_dequant, CUDA_R_32F, intermediate_dim,
                     d_intermediate_f32, CUDA_R_32F, intermediate_dim, &beta,
                     d_w2out_f32, CUDA_R_32F, hidden_dim,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        // W2 FP32 output -> FP16 d_output
        int g2 = (hidden_dim + b - 1) / b;
        float_to_half_kernel<<<g2, b, 0, stream>>>(
            d_w2out_f32, d_output, hidden_dim);
        static int pcount2 = 0;
        if (pcount2 == 42) {
            cudaStreamSynchronize(stream);
            float *d_swig = d_intermediate_f32; // swiglu already written back in place
            std::vector<float> sf(intermediate_dim);
            cudaMemcpy(sf.data(), d_swig, intermediate_dim*sizeof(float), cudaMemcpyDeviceToHost);
            double ss=0; for (auto v:sf) ss+=(double)v*v;
            std::vector<float> w2d(static_cast<size_t>(hidden_dim)*intermediate_dim);
            cudaMemcpy(w2d.data(), d_dequant, w2d.size()*sizeof(float), cudaMemcpyDeviceToHost);
            double sw2=0; for (auto v:w2d) sw2+=(double)v*v;
            double so=0;
            std::vector<float> owf(hidden_dim);
            cudaMemcpy(owf.data(), d_w2out_f32, hidden_dim*sizeof(float), cudaMemcpyDeviceToHost);
            for (auto v:owf) so+=(double)v*v;
            log::info("[SPLIT] layer42 W2 deq rms=" + std::to_string(std::sqrt((float)(sw2/w2d.size())))
                      + " swig_in_rms=" + std::to_string(std::sqrt((float)(ss/intermediate_dim)))
                      + " w2out_rms=" + std::to_string(std::sqrt((float)(so/hidden_dim))));
        }
        ++pcount2;
    }
    {
        int b = 256;
        int g = (hidden_dim + b - 1) / b;
        clamp_half_kernel<<<g, b, 0, stream>>>(d_output, hidden_dim, 65504.0f);
    }
    return cudaSuccess;
}

// Routed-expert FFN: fp4 (E2M1 x F8_E8M0 per-32) dequant + SwiGLU GEMM.
// Verified against the real DeepSeek-V4-Flash safetensors header
// ("expert_dtype": "fp4" in config.json): weight is a nibble-packed E2M1
// container (stored as raw I8 bytes, 2 elements/byte) with a flat F8_E8M0
// scale every 32 elements — global_scale does not apply (converter.cpp's
// passthrough path hardcodes 1.0 for already-quantized sources). Applies
// act_quant (FP8_E4M3 emulation) to the input activation and to the
// SwiGLU output before W2, matching the reference linear() pipeline.
cudaError_t run_routed_expert_ffn(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const __half *d_hidden,
    const uint8_t *d_w1, const uint8_t *d_w1_scales, int64_t w1_el,
    const uint8_t *d_w2, const uint8_t *d_w2_scales, int64_t w2_el,
    const uint8_t *d_w3, const uint8_t *d_w3_scales, int64_t w3_el,
    float *d_gate_f32, float *d_up_f32,    // FP32 W1/W3 outputs (no overflow)
    float *d_dequant,                    // FP32 dequant scratch (no overflow)
    float *d_output,
    float *d_act_quant,                  // FP32 activation (was __half*)
    float *d_swiglu_out,
    float *d_act_scales,
    float weight, bool accumulate)
{
    cublasSetStream(cublas_h, stream);
    float alpha = 1.0f, beta = 0.0f;
    const int block = 256;

    // ── Per-128 block activation quantization ──────────────────────────────
    // Quantizes d_hidden → d_act_quant (FP8_E4M3), stores scales → d_act_scales
    {
        int n_blocks = (kHiddenDim + 127) / 128;
        act_quant_block_kernel<<<n_blocks, block, 0, stream>>>(
            d_hidden, d_act_quant, d_act_scales, kHiddenDim);
    }

    // ── W1 GEMM: fuse act_scale into dequanted weights, then GEMM ──────────
    if (w1_el > 0) {
        int64_t packed_bytes = (w1_el + 1) / 2;
        int64_t scale_bytes  = (w1_el + 31) / 32;
        dequant_fp4_e8m0(stream, d_w1, packed_bytes, d_w1_scales, scale_bytes,
                         d_dequant, w1_el);
        // Fuse per-128 act_scale into dequanted weights (W1: [inter, hidden], K=hidden)
        {
            int64_t N = kIntermediateDim, K = kHiddenDim;
            int64_t total = N * K;
            int grid = static_cast<int>((total + block - 1) / block);
            fuse_act_scale_into_weights_kernel<<<grid, block, 0, stream>>>(
                d_dequant, d_act_scales, N, K);
        }
        cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                     kIntermediateDim, 1, kHiddenDim, &alpha,
                     d_dequant, CUDA_R_32F, kHiddenDim,
                     d_act_quant, CUDA_R_32F, kHiddenDim, &beta,
                     d_gate_f32, CUDA_R_32F, kIntermediateDim,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }

    // ── W3 GEMM: fuse same act_scale into W3 dequanted weights ─────────────
    if (w3_el > 0) {
        int64_t packed_bytes = (w3_el + 1) / 2;
        int64_t scale_bytes  = (w3_el + 31) / 32;
        dequant_fp4_e8m0(stream, d_w3, packed_bytes, d_w3_scales, scale_bytes,
                         d_dequant, w3_el);
        {
            int64_t N = kIntermediateDim, K = kHiddenDim;
            int64_t total = N * K;
            int grid = static_cast<int>((total + block - 1) / block);
            fuse_act_scale_into_weights_kernel<<<grid, block, 0, stream>>>(
                d_dequant, d_act_scales, N, K);
        }
        cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                     kIntermediateDim, 1, kHiddenDim, &alpha,
                     d_dequant, CUDA_R_32F, kHiddenDim,
                     d_act_quant, CUDA_R_32F, kHiddenDim, &beta,
                     d_up_f32, CUDA_R_32F, kIntermediateDim,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }

    // ── SwiGLU in FP32: silu(clamp(gate,max=10)) * clamp(up,±10) → FP32 ─────
    if (kIntermediateDim > 0) {
        int grid = (kIntermediateDim + block - 1) / block;
        swiglu_clamp_f32f_kernel<<<grid, block, 0, stream>>>(
            d_gate_f32, d_up_f32, d_swiglu_out, kIntermediateDim, 10.0f);
        // Apply routing weight (norm_topk_prob * 1.5) in FP32 BEFORE W2
        scale_float_kernel<<<grid, block, 0, stream>>>(
            d_swiglu_out, weight, kIntermediateDim);
    }

    // ── W2 GEMM: pure FP32 matmul of dequantized W2 against the raw
    // SwiGLU output. No act_scale fusion: the W2 activation (d_swiglu_out)
    // is used at full FP32 precision and is NOT divided by any per-128 block
    // scale, so the weights must stay unscaled. (Fusing the input's
    // d_act_scales here — as the older quantized-activation path did — scaled
    // every routed-expert output by ~act_scale≈2^-9, silently zeroing routed
    // experts and leaving the model running on attention + shared experts
    // only, which answered incorrectly.)
    if (w2_el > 0) {
        int64_t packed_bytes = (w2_el + 1) / 2;
        int64_t scale_bytes  = (w2_el + 31) / 32;
        dequant_fp4_e8m0(stream, d_w2, packed_bytes, d_w2_scales, scale_bytes,
                         d_dequant, w2_el);

        // Pure FP32 GEMM: FP32 weights × FP32 activation → FP32 output
        // Use d_swiglu_out directly as activation input (already FP32)
        float gbeta = accumulate ? 1.0f : 0.0f;
        cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                     kHiddenDim, 1, kIntermediateDim, &alpha,
                     d_dequant, CUDA_R_32F, kIntermediateDim,
                     d_swiglu_out, CUDA_R_32F, kIntermediateDim, &gbeta,
                     d_output, CUDA_R_32F, kHiddenDim,
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
    float *d_output,
    float *d_gate_f32, float *d_up_f32,   // FP32 W1/W3 outputs
    float *d_dequant,
    float *d_act_quant,                  // FP32 activation
    float *d_swiglu_out,
    float *d_act_scales,
    uint8_t *d_pack_stage, uint8_t *d_scale_stage,
    int64_t pack_stage_bytes, int64_t scale_stage_bytes)
{
    const int64_t w1_el = static_cast<int64_t>(kHiddenDim) * kIntermediateDim;
    const int64_t w2_el = static_cast<int64_t>(kIntermediateDim) * kHiddenDim;
    const int64_t w3_el = w1_el;

    // d_output already holds the shared-expert FFN result (or was freshly
    // zeroed by the caller when there are no shared experts — see
    // inference_pipeline.cu's d_ffn_accum setup) *before* this function is
    // called, so every routed expert's W2 GEMM must accumulate (beta=1),
    // never overwrite (beta=0). This used to start `false`, which silently
    // discarded the shared-expert contribution on every layer/token that
    // selected >=1 routed expert.
    const bool accumulate = true;
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
            d_gate_f32, d_up_f32, d_dequant, d_output,
            d_act_quant, d_swiglu_out, d_act_scales,
            h_top_weights[k], accumulate);
        if (e != cudaSuccess) return e;
    }
    return cudaSuccess;
}

} // namespace gpu
} // namespace lsxcommon
