#include "lsxcommon/inference_gpu.h"
#include "lsxcommon/shared_experts.h"
#include "lsxcommon/log.h"
#include "deepseek_inference_load.cuh"
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
        if (w1s != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;
    }

    // 3. Dequant W3 → d_dequant (reuse, F8_E4M3 × E8M0)
    if (w3_bytes > 0) {
        cudaError_t e = dequant_shared_f8(stream, d_w3, w3_bytes,
            d_w3_scales, 0, d_dequant, w1_el, hidden_dim);
        if (e != cudaSuccess) return e;
    }

    // 4. gate = W3^T @ hidden → d_gate
    {
        cublasStatus_t w3s = cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                     intermediate_dim, 1, hidden_dim, &alpha,
                     d_dequant, CUDA_R_32F, hidden_dim,
                      d_hidden_f32, CUDA_R_32F, hidden_dim, &beta,
                      d_gate_f32, CUDA_R_32F, intermediate_dim,
                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (w3s != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;
    }

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
        if (wbst != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;
        int g2 = (hidden_dim + b - 1) / b;
        float_to_half_kernel<<<g2, b, 0, stream>>>(
            d_w2out_f32, d_output, hidden_dim);
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
        cudaError_t de = dequant_fp4_e8m0(stream, d_w1, packed_bytes, d_w1_scales,
                                          scale_bytes, d_dequant, w1_el);
        if (de != cudaSuccess) return de;
        // Fuse per-128 act_scale into dequanted weights (W1: [inter, hidden], K=hidden)
        {
            int64_t N = kIntermediateDim, K = kHiddenDim;
            int64_t total = N * K;
            int grid = static_cast<int>((total + block - 1) / block);
            fuse_act_scale_into_weights_kernel<<<grid, block, 0, stream>>>(
                d_dequant, d_act_scales, N, K);
        }
        cublasStatus_t w1s = cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                     kIntermediateDim, 1, kHiddenDim, &alpha,
                     d_dequant, CUDA_R_32F, kHiddenDim,
                     d_act_quant, CUDA_R_32F, kHiddenDim, &beta,
                     d_gate_f32, CUDA_R_32F, kIntermediateDim,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (w1s != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;
    }

    // ── W3 GEMM: fuse same act_scale into W3 dequanted weights ─────────────
    if (w3_el > 0) {
        int64_t packed_bytes = (w3_el + 1) / 2;
        int64_t scale_bytes  = (w3_el + 31) / 32;
        cudaError_t de = dequant_fp4_e8m0(stream, d_w3, packed_bytes, d_w3_scales,
                                          scale_bytes, d_dequant, w3_el);
        if (de != cudaSuccess) return de;
        {
            int64_t N = kIntermediateDim, K = kHiddenDim;
            int64_t total = N * K;
            int grid = static_cast<int>((total + block - 1) / block);
            fuse_act_scale_into_weights_kernel<<<grid, block, 0, stream>>>(
                d_dequant, d_act_scales, N, K);
        }
        cublasStatus_t w3s = cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                     kIntermediateDim, 1, kHiddenDim, &alpha,
                     d_dequant, CUDA_R_32F, kHiddenDim,
                     d_act_quant, CUDA_R_32F, kHiddenDim, &beta,
                     d_up_f32, CUDA_R_32F, kIntermediateDim,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (w3s != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;
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
        cudaError_t de = dequant_fp4_e8m0(stream, d_w2, packed_bytes, d_w2_scales,
                                          scale_bytes, d_dequant, w2_el);
        if (de != cudaSuccess) return de;

        // Pure FP32 GEMM: FP32 weights × FP32 activation → FP32 output
        // Use d_swiglu_out directly as activation input (already FP32)
        float gbeta = accumulate ? 1.0f : 0.0f;
        cublasStatus_t w2s = cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                     kHiddenDim, 1, kIntermediateDim, &alpha,
                     d_dequant, CUDA_R_32F, kIntermediateDim,
                     d_swiglu_out, CUDA_R_32F, kIntermediateDim, &gbeta,
                     d_output, CUDA_R_32F, kHiddenDim,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (w2s != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;
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
    constexpr int kMaxRouted = 256;  // fixed-size scores[]/sel[]/selected[] below

    // Defensive bounds: `scores`/`sel`/`selected` are fixed-size stack arrays
    // sized for the model's known geometry (kNumRouted==256). A `num_routed`
    // larger than that would overflow them; a `topk` larger than the actual
    // candidate pool would leave `h_top_indices` slots unfilled further down.
    if (num_routed <= 0 || num_routed > kMaxRouted)
        return cudaErrorInvalidValue;
    if (topk > num_routed) topk = num_routed;
    if (topk <= 0) return cudaErrorInvalidValue;

    cublasSetStream(cublas_h, stream);
    float alpha = 1.0f, beta = 0.0f;

    __half *d_logits = nullptr;
    cudaError_t e = cudaMalloc(&d_logits, static_cast<size_t>(num_routed) * sizeof(__half));
    if (e != cudaSuccess) return e;

    cublasStatus_t rstat = cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                 num_routed, 1, hidden_dim, &alpha,
                 d_gate, CUDA_R_16F, hidden_dim,
                 d_hidden, CUDA_R_16F, hidden_dim, &beta,
                 d_logits, CUDA_R_16F, num_routed,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (rstat != CUBLAS_STATUS_SUCCESS) { cudaFree(d_logits); return cudaErrorUnknown; }

    float *h_logits = nullptr;
    e = cudaMallocHost(&h_logits, static_cast<size_t>(num_routed) * sizeof(float));
    if (e != cudaSuccess) { cudaFree(d_logits); return e; }

    cudaMemcpyAsync(h_logits, d_logits,
                    static_cast<size_t>(num_routed) * sizeof(__half),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // scores[i] = sqrt(softplus(logit[i]))  (unbiased)
    float scores[kMaxRouted];
    for (int i = 0; i < num_routed; ++i) {
        float x = __half2float(reinterpret_cast<__half*>(h_logits)[i]);
        scores[i] = sqrtf(log1pf(expf(x)));
    }

    // Selection scores: biased (bias is null for hash layers)
    float sel[kMaxRouted];
    for (int i = 0; i < num_routed; ++i)
        sel[i] = scores[i] + (h_gate_bias ? h_gate_bias[i] : 0.0f);

    // Every slot starts unfilled (-1) so a routing round that can't find
    // `topk` distinct candidates (num_routed < topk — already clamped above
    // so this can't happen — or every remaining candidate score is NaN, which
    // always compares false against `best_val` and so is never selected)
    // leaves a detectable sentinel instead of silently reusing whatever
    // stale expert index/weight this long-lived buffer held from a
    // previous layer/token's routing decision.
    for (int k = 0; k < topk; ++k) h_top_indices[k] = -1;

    if (h_tid2eid) {
        // Hash layer: expert indices predetermined per token id.
        if (tid2eid_stride <= 0) tid2eid_stride = topk;
        int64_t base = static_cast<int64_t>(token_id) * tid2eid_stride;
        for (int k = 0; k < topk; ++k)
            h_top_indices[k] = static_cast<int>(h_tid2eid[base + k]);
    } else {
        bool selected[kMaxRouted] = {false};
        bool any_unfilled = false;
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
            } else {
                any_unfilled = true;
            }
        }
        if (any_unfilled) {
            // All remaining candidates were NaN (or otherwise unselectable).
            // Fall back to a deterministic, in-range index rather than
            // leaving -1 sentinels for the weight loop below to choke on.
            log::warn("[route_token] top-k selection could not fill all "
                      "slots (NaN/invalid router logits?) — falling back "
                      "to expert 0 for unfilled slots");
            for (int k = 0; k < topk; ++k)
                if (h_top_indices[k] < 0) h_top_indices[k] = 0;
        }
    }

    // Routing weights from unbiased scores, normalized, scaled by route_scale.
    float sum_w = 0.0f;
    for (int k = 0; k < topk; ++k) {
        int idx = h_top_indices[k];
        float w = (idx >= 0 && idx < num_routed) ? scores[idx] : 0.0f;
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
    // deepseek_inference_pipeline.cu's d_ffn_accum setup) *before* this function is
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
