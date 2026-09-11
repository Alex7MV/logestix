#include "lsxcommon/inference_gpu.h"
#include "lsxcommon/shared_experts.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/log.h"
#include "deepseek_moe_load.cuh"
#include "lsx_kernels_act.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <atomic>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cfloat>
#include <sstream>

namespace lsxcommon {

// ── Pending prefetch-copy registry (munmap safety) ──────────────────────────
// issue_routed_staging enqueues H2D copies whose SOURCE is an mmap-backed
// expert entry. The driver may execute them after the API call returns,
// while the host keeps running — and ensure_expert_resident may munmap that
// very mapping for a later layer. Each issued expert registers (source
// base, completion event); expert_store drains (host-syncs) before unmapping.
namespace {
struct PendingCopy { const void* base; cudaEvent_t ev; };
std::vector<PendingCopy> g_pending_copies;
} // namespace

void lsx_pending_copy_drain(const void* host_ptr)
{
    for (size_t i = 0; i < g_pending_copies.size();) {
        const PendingCopy &pc = g_pending_copies[i];
        // The whole expert entry lives inside one arrow mmap: a 64 MB window
        // from the w1 base covers packed weights + all scale arrays.
        const uintptr_t lo = reinterpret_cast<uintptr_t>(pc.base);
        const uintptr_t hi = lo + (size_t{64} << 20);
        const uintptr_t q = reinterpret_cast<uintptr_t>(host_ptr);
        if (q >= lo && q < hi) {
            cudaEventSynchronize(pc.ev);
            g_pending_copies.erase(g_pending_copies.begin() + i);
        } else {
            ++i;
        }
    }
}

namespace gpu {

// Routed experts are NVFP4-native (logestix-v1-nvfp4): the neutral
// lsx_nvfp4 layer drives kernel selection via its process-global policy,
// asserted per-call by DeepSeekV4FlashModel::Infer's LsxFastMathScope.
// The legacy family-scoped TC toggle (g_deepseek_fast_math) is retired.

// Dequant kernel from benchmark (extern declaration)

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
    __half* __restrict__       d_output_h,
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
            d_output_h[base + i] = __float2half(0.0f);
        } else {
            uint32_t out_bits = (sign << 31) | (static_cast<uint32_t>(e4 + 120) << 23) | (mant3 << 20);
            float result;
            std::memcpy(&result, &out_bits, sizeof(result));
            d_output[base + i] = result;
            // E4M3-grid values are exactly representable in FP16 — the half
            // copy feeds the FP16 tensor-core GEMMs without precision loss.
            d_output_h[base + i] = __float2half(result);
        }
    }
    if (threadIdx.x == 0) d_scales[blk] = act_scale;
}

// Clamp FP32 → FP16 with overflow protection (feeds FP16 GEMM inputs).
__global__ void clamp_f32_to_f16_moe_kernel(const float* __restrict__ src,
                                             __half* __restrict__ dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = src[i];
    v = fmaxf(-65504.0f, fminf(65504.0f, v));
    dst[i] = __float2half(v);
}

// Fold the per-128 activation quantization scales (powers of two) into the
// FP16 quantized activation: act'[k] = act[k] * act_scales[k/128]. A power-of-
// two multiply is exact in FP16, so Σ w·act' == Σ (w·act_scale)·act with zero
// added rounding — the W1/W3 tensor-core GEMMs stay numerically equivalent to
// the old FP32 fused-scale path.
__global__ void scale_act_blocks_kernel(__half* __restrict__ act,
                                         const float* __restrict__ act_scales,
                                         int K)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= K) return;
    act[i] = __float2half(__half2float(act[i]) * act_scales[i >> 7]);
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

// TC epilogue: fold a __half expert output into the FP32 routed accumulator.
__global__ void acc_half_to_f32_kernel(float* __restrict__ dst,
                                       const __half* __restrict__ src,
                                       int n, bool accumulate)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = __half2float(src[i]);
    dst[i] = accumulate ? dst[i] + v : v;
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

// SwiGLU with clamp, FP32 inputs → FP32 output (no FP16 intermediate overflow).
// NOTE: deliberately NOT __restrict__ — the shared-experts call site passes
// d_out aliased to d_gate (element-wise, same-index in-place update).
// SwiGLU (sig form): unified kernels::SwigluSigKernel (lsx_kernels_act.cuh).

// Apply scalar multiplier to a float buffer (in-place)
__global__ void scale_float_kernel(float* d_buf, float scalar, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    d_buf[i] *= scalar;
}

// float_to_half_kernel / half_to_float_kernel are defined once in
// deepseek_moe_load.cuh (included above) and shared across this
// pipeline's .cu files.

// SwiGLU with swiglu_limit: gate clamped to (-inf, limit], up to [-limit, limit].
// d_gate_vec = silu(clamp(d_gate_vec, max=limit)) * clamp(d_up_vec, +/-limit)
// SwiGLU FP16 (sig form): unified kernels::SwigluSigKernel.

// Shared experts FFN: dequant F8_E4M3 x F8_E8M0 (128x128 tile) -> FP16,
// FP16 tensor-core GEMMs (FP32 accumulate/output), SiLU activation. Verified
// against the real DeepSeek-V4-Flash safetensors header: shared_experts.w1.weight
// dtype=F8_E4M3 shape=[2048,4096], w1.scale dtype=F8_E8M0 shape=[16,32] ->
// block_r=128, block_c=128. global_scale is not applicable here (converter.cpp's
// passthrough path hardcodes 1.0 for already-quantized F8_E4M3 sources), which
// is why w1_gs/w2_gs/w3_gs are accepted but intentionally unused below.
//
// Blackwell perf: weights are dequanted straight to FP16 (E4M3 x E8M0 products
// are exactly representable in FP16 for in-range scales) and every GEMM runs
// as CUDA_R_16F inputs with CUBLAS_COMPUTE_32F — half the weight-materialization
// traffic of the old FP32 path and full 5th-gen Tensor Core utilization.
// W1/W3 GEMM outputs stay FP32 to prevent FP16 overflow.
cudaError_t run_shared_experts_ffn(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const __half *d_hidden,
    const uint8_t *d_w1, const uint8_t *d_w2, const uint8_t *d_w3,
    const uint8_t *d_w1_scales, const uint8_t *d_w2_scales, const uint8_t *d_w3_scales,
    float /*w1_gs*/, float /*w2_gs*/, float /*w3_gs*/,
    int64_t w1_bytes, int64_t w2_bytes, int64_t w3_bytes,
    float *d_intermediate_f32, float *d_gate_f32,   // FP32 W1/W3 outputs
    float *d_dequant, __half *d_output,
    const __half *d_w1_h_cache = nullptr,
    const __half *d_w3_h_cache = nullptr,
    const float *d_w2_f32_cache = nullptr)
{
    cublasSetStream(cublas_h, stream);
    float alpha = 1.0f, beta = 0.0f;

    int hidden_dim = kHiddenDim;
    // F8_E4M3: 1 byte per element. intermediate_dim = n_elements / hidden_dim
    int intermediate_dim = w1_bytes > 0
        ? static_cast<int>(w1_bytes / hidden_dim) : kIntermediateDim;
    int w1_el = hidden_dim * intermediate_dim;
    int w2_el = intermediate_dim * hidden_dim;

    // FP16 scratch carved from the (float-typed) d_dequant arena slice:
    // [dequanted W1/W3 weights: w1_el halves]. The W2 FP32 pass reuses the
    // full d_dequant region (stream-ordered after the W3 GEMM that read the
    // fp16 weights), and its FP32 output scratch sits past the W2 weights.
    __half* const d_w_half = reinterpret_cast<__half*>(d_dequant);
    float*  const d_w2out_f32 = d_dequant + w2_el + intermediate_dim / 2;
    // Load-time dequant caches (bit-exact: produced by the same kernels at
    // load; when present the per-call dequant launches are skipped).
    const __half* const w1_ptr = d_w1_h_cache ? d_w1_h_cache : d_w_half;
    const __half* const w3_ptr = d_w3_h_cache ? d_w3_h_cache : d_w_half;
    const float*  const w2_ptr = d_w2_f32_cache ? d_w2_f32_cache : d_dequant;
    // 1. Dequant W1 → d_w_half (F8_E4M3 × E8M0, per-128x128 tiles)
    if (w1_bytes > 0 && !d_w1_h_cache) {
        // dequant_shared_f8_half does not validate/use its scale_bytes parameter
        // (kept for interface symmetry with the other dequant_* functions).
        cudaError_t e = dequant_shared_f8_half(stream, d_w1, w1_bytes,
            d_w1_scales, 0, d_w_half, w1_el, hidden_dim);
        if (e != cudaSuccess) return e;
    }

    // 2. intermediate = W1^T @ hidden → d_intermediate (FP16 TC GEMM, FP32 out)
    {
        cublasStatus_t w1s = cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                     intermediate_dim, 1, hidden_dim, &alpha,
                     w1_ptr, CUDA_R_16F, hidden_dim,
                      d_hidden, CUDA_R_16F, hidden_dim, &beta,
                      d_intermediate_f32, CUDA_R_32F, intermediate_dim,
                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (w1s != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;
    }

    // 3. Dequant W3 → d_w_half (reuse)
    if (w3_bytes > 0 && !d_w3_h_cache) {
        cudaError_t e = dequant_shared_f8_half(stream, d_w3, w3_bytes,
            d_w3_scales, 0, d_w_half, w1_el, hidden_dim);
        if (e != cudaSuccess) return e;
    }

    // 4. gate = W3^T @ hidden → d_gate
    {
        cublasStatus_t w3s = cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                     intermediate_dim, 1, hidden_dim, &alpha,
                     w3_ptr, CUDA_R_16F, hidden_dim,
                      d_hidden, CUDA_R_16F, hidden_dim, &beta,
                      d_gate_f32, CUDA_R_32F, intermediate_dim,
                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (w3s != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;
    }

    // 5. SiLU + multiply: d_intermediate = silu(d_intermediate) * d_gate (FP32).
    //    W2 stays on the FP32 path: its activation terms are large and
    //    heavily cancelling, so FP16-rounding them shifts outputs beyond the
    //    numeric contract (same rationale as the routed-expert W2).
    {
        int block = 256;
        int grid = (intermediate_dim + block - 1) / block;
        ::lsxcommon::kernels::SwigluSigKernel<float, ::lsxcommon::kernels::ExpFlavor::Precise, true><<<grid, block, 0, stream>>>(
            d_intermediate_f32, d_gate_f32, d_intermediate_f32,
            intermediate_dim, 10.0f);
    }

    // 6. Dequant W2 → d_dequant (reuse, FP32 — overwrites the W1/W3 fp16
    //    region; stream-ordered after the W3 GEMM that read it). Skipped
    //    when the load-time FP32 cache is wired in.
    if (w2_bytes > 0 && !d_w2_f32_cache) {
        // W2: [hidden, intermediate] → cols = intermediate_dim
        cudaError_t e = dequant_shared_f8(stream, d_w2, w2_bytes,
            d_w2_scales, 0, d_dequant, w2_el, intermediate_dim);
        if (e != cudaSuccess) return e;
    }

    // 7. output = W2^T @ activated → FP32 scratch → clamped FP16 output
    {
        cublasStatus_t wbst = cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                     hidden_dim, 1, intermediate_dim, &alpha,
                     w2_ptr, CUDA_R_32F, intermediate_dim,
                     d_intermediate_f32, CUDA_R_32F, intermediate_dim, &beta,
                     d_w2out_f32, CUDA_R_32F, hidden_dim,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (wbst != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;
        int b = 256;
        int g2 = (hidden_dim + b - 1) / b;
        clamp_f32_to_f16_moe_kernel<<<g2, b, 0, stream>>>(
            d_w2out_f32, d_output, hidden_dim);
    }
    // Surface any launch error from the kernels above here instead of letting
    // it surface at an unrelated later checked call.
    return cudaGetLastError();
}


// v1-nvfp4 routed-expert FFN: weights stay NVFP4-packed end to end. The
// neutral lsx_nvfp4 layer streams them straight through the seq==1
// weight-streaming GEMV (no dequantization stage, no cuBLAS). gate/up share
// one pair-GEMV launch over the contiguous [w1|w3] rows of the cache slot /
// staging area; w2 follows the GLM-Flash 32 t/s pattern:
// pair-GEMV -> silu_mul -> w2 GEMV -> weighted FP32 accumulate.
//   d_w13/d_s13: [w1|w3] rows ([I,H] each) + scales [s1|s3] (per-16)
//   d_w2/d_s2:   [H,I] packed + scales; gs* decode with the vault's
//                per-tensor global scales (repack-folded 2^m).
// Tolerance class: FP16 SwiGLU intermediates + k-order-free GEMV (same
// numeric contract as every other model on the lsx_nvfp4 contour).
// Routing weight + fixed down-scale folded BEFORE the w2 GEMV: the raw w2
// output can exceed the __half y contract (measured: unweighted synthetic
// and real-magnitude routed activations reach ~1e6 in FP32), so the GEMV
// writes y2 pre-divided by kY2Downscale and the FP32 accumulator undoes it
// (FP16 is a floating format — the relative error contract is unchanged).
constexpr float kY2Downscale = 64.0f;
__global__ void scale_half_kernel(__half* __restrict__ buf, float w, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    buf[i] = __float2half(__half2float(buf[i]) * w);
}
// dst[i] += w * y[i] — FP32 routed accumulator, __half GEMV output.
__global__ void coeff_add_f32_kernel(const __half* __restrict__ y, float w,
                                     float* __restrict__ dst, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] += w * __half2float(y[i]);
}

cudaError_t run_routed_expert_ffn(
    cudaStream_t stream,
    const __half *d_hidden,
    const uint8_t *d_w13, const uint8_t *d_s13, float gs1, float gs3,
    const uint8_t *d_w2, const uint8_t *d_s2, float gs2,
    __half *d_yg,        // [2*I] halves: [gate | up] pair-GEMV output
    __half *d_y2,        // [H] halves: w2 GEMV output
    float *d_output,
    float weight)
{
    constexpr int I = kIntermediateDim;
    constexpr int H = kHiddenDim;


    if (lsx_nvfp4::RunLsxNvfp4GemmGateUpPair(stream, d_hidden, d_w13, d_s13,
            gs1, gs3, d_yg, H, I, false) != cudaSuccess)
        return cudaGetLastError();
    // SwiGLU FP16: silu(clamp(gate, max=limit)) * clamp(up, +/-limit) in
    // place — the result lands in d_yg[0..I), ready for w2.
    silu_mul_kernel<<<(I + 255) / 256, 256, 0, stream>>>(
        d_yg, d_yg + I, I, 10.0f);
    // Fold the routing weight BEFORE w2 (legacy scale_float_kernel order)
    // plus the y-contract down-scale; undo both in the FP32 accumulator.
    scale_half_kernel<<<(I + 255) / 256, 256, 0, stream>>>(
        d_yg, weight / kY2Downscale, I);
    if (lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_yg, d_w2, d_s2, gs2, d_y2, 1,
            I, H, false) != cudaSuccess)
        return cudaGetLastError();
    if (std::getenv("LSX_DS_FFN_PROBE")) {
        cudaStreamSynchronize(stream);
        __half hq[4];
        cudaMemcpy(hq, d_y2, sizeof(hq), cudaMemcpyDeviceToHost);
        fprintf(stderr, "P2 y2: %f %f %f\n", __half2float(hq[0]),
                __half2float(hq[1]), __half2float(hq[2]));
    }
    coeff_add_f32_kernel<<<(H + 255) / 256, 256, 0, stream>>>(
        d_y2, kY2Downscale, d_output, H);
    return cudaGetLastError();
}

// Routes one token: gate GEMM → sqrt(softplus) scores → top-k (bias only for
// selection; hash layers 0..2 use tid2eid) → norm_topk_prob → ×route_scale.
// `d_logits_scratch` / `h_logits_scratch` are caller-owned arena carves sized
// [num_routed] — route_token performs ZERO CUDA allocations (spec item 12).
cudaError_t route_token_begin(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const __half *d_gate, const __half *d_hidden,
    int hidden_dim, int num_routed,
    __half *d_logits_scratch, float *h_logits_scratch,
    cudaEvent_t ev)
{
    if (num_routed <= 0 || num_routed > kNumRouted)
        return cudaErrorInvalidValue;

    cublasSetStream(cublas_h, stream);
    float alpha = 1.0f, beta = 0.0f;

    cublasStatus_t rstat = cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                 num_routed, 1, hidden_dim, &alpha,
                 d_gate, CUDA_R_16F, hidden_dim,
                 d_hidden, CUDA_R_16F, hidden_dim, &beta,
                 d_logits_scratch, CUDA_R_16F, num_routed,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (rstat != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;

    // Async readback; the fence is the caller-owned event so the pipeline can
    // enqueue independent GPU work (shared-expert FFN) before waiting.
    if (!ev) return cudaErrorInvalidValue;
    cudaError_t e = cudaMemcpyAsync(h_logits_scratch, d_logits_scratch,
                                    static_cast<size_t>(num_routed) * sizeof(__half),
                                    cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess) return e;
    return cudaEventRecord(ev, stream);
}

namespace {

// Host-side scoring over the readback logits: identical math to the original
// fused route_token (sqrt(softplus) scores → biased top-k selection →
// normalized weights ×route_scale). Bit-exact by construction.
cudaError_t route_scores_host(
    int num_routed, int token_id,
    const int64_t *h_tid2eid, int tid2eid_stride,
    const float *h_gate_bias,
    int topk,
    float *h_logits_scratch,
    int *h_top_indices, float *h_top_weights)
{
    constexpr float kRouteScale = 1.5f;

    // Defensive bounds: `scores`/`sel`/`selected` are fixed-size stack arrays
    // sized for the model's known geometry (kNumRouted, declared in
    // deepseek_moe_load.cuh). A `num_routed`
    // larger than that would overflow them; a `topk` larger than the actual
    // candidate pool would leave `h_top_indices` slots unfilled further down.
    if (num_routed <= 0 || num_routed > kNumRouted)
        return cudaErrorInvalidValue;
    if (topk > num_routed) topk = num_routed;
    if (topk <= 0) return cudaErrorInvalidValue;

    // scores[i] = sqrt(softplus(logit[i]))  (unbiased)
    float scores[kNumRouted];
    for (int i = 0; i < num_routed; ++i) {
        float x = __half2float(reinterpret_cast<__half*>(h_logits_scratch)[i]);
        scores[i] = sqrtf(log1pf(expf(x)));
    }

    // Selection scores: biased (bias is null for hash layers)
    float sel[kNumRouted];
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
        bool selected[kNumRouted] = {false};
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

    return cudaSuccess;
}

} // namespace

cudaError_t route_token_finish(
    int num_routed, int token_id,
    const int64_t *h_tid2eid, int tid2eid_stride,
    const float *h_gate_bias,
    int topk,
    float *h_logits_scratch,
    int *h_top_indices, float *h_top_weights,
    cudaEvent_t ev)
{
    if (!ev) return cudaErrorInvalidValue;
    // Event-scoped fence: waits only until the router GEMM + D2H retire, not
    // for whatever else the pipeline enqueued behind them on this stream.
    cudaError_t e = cudaEventSynchronize(ev);
    if (e != cudaSuccess) return e;
    return route_scores_host(num_routed, token_id, h_tid2eid, tid2eid_stride,
                             h_gate_bias, topk, h_logits_scratch,
                             h_top_indices, h_top_weights);
}

cudaError_t route_token(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const __half *d_gate, const __half *d_hidden,
    int hidden_dim, int num_routed,
    const int64_t *h_tid2eid, int tid2eid_stride, int token_id,
    const float *h_gate_bias,
    int topk,
    __half *d_logits_scratch, float *h_logits_scratch,
    int *h_top_indices, float *h_top_weights)
{
    cudaEvent_t ev = nullptr;
    cudaError_t e = cudaEventCreate(&ev);
    if (e != cudaSuccess) return e;
    e = route_token_begin(cublas_h, stream, d_gate, d_hidden, hidden_dim,
                          num_routed, d_logits_scratch, h_logits_scratch, ev);
    if (e == cudaSuccess)
        e = route_token_finish(num_routed, token_id, h_tid2eid, tid2eid_stride,
                               h_gate_bias, topk, h_logits_scratch,
                               h_top_indices, h_top_weights, ev);
    cudaEventDestroy(ev);
    return e;
}

// Runs the top-k routed experts for one token and accumulates their weighted
// outputs into d_output (which must already hold the shared-expert result).
// Expert host pointers come from the mmap'd ExpertDb; all 6 tensors of an
// expert are staged into device scratch buffers at fixed offsets — unless the
// expert already lives in the resident device cache (@p cache), in which case
// the H2D copies are skipped entirely and the cached slot is addressed
// directly (single-stream ordering makes slot reuse safe).
cudaError_t run_routed_experts_ex(
    cudaStream_t stream, cudaStream_t copy_stream, cudaEvent_t *stage_ev,
    cudaEvent_t pre_ev,
    const __half *d_normed,
    const ExpertEntry *db_entries, int layer, int experts_per_layer,
    const int *h_top_indices, const float *h_top_weights, int topk,
    float *d_output,
    __half *d_yg, __half *d_y2,
    uint8_t *d_pack_stage, uint8_t *d_scale_stage,
    int64_t pack_stage_bytes, int64_t scale_stage_bytes,
    const ExpertDeviceCache *cache)
{
    const int64_t w1_el = static_cast<int64_t>(kHiddenDim) * kIntermediateDim;
    const int64_t w2_el = static_cast<int64_t>(kIntermediateDim) * kHiddenDim;
    const int64_t w3_el = w1_el;

    // d_output already holds the shared-expert FFN result (or was freshly
    // zeroed by the caller when there are no shared experts), so every
    // routed expert's weighted output must accumulate (coeff add), never
    // overwrite.
    // Prefetch fence: when the pipeline issued the miss copies early (before
    // the shared-expert FFN), one event fences ALL of them.
    if (pre_ev) {
        cudaError_t pe = cudaStreamWaitEvent(stream, pre_ev, 0);
        if (pe != cudaSuccess) return pe;
    }
    for (int k = 0; k < topk; ++k) {
        int idx = h_top_indices[k];
        if (idx < 0 || idx >= experts_per_layer) continue;
        const ExpertEntry &en = db_entries[
            static_cast<size_t>(layer) * experts_per_layer + idx];


        const int64_t w1b = en.w1_bytes, w2b = en.w2_bytes, w3b = en.w3_bytes;
        const int64_t s1b = en.w1_scales_bytes, s2b = en.w2_scales_bytes, s3b = en.w3_scales_bytes;

        // ── Resolve source pointers: resident cache hit or H2D staging ──────
        // Cache slot / staging layout: packed [w1|w3|w2], scales [s1|s3|s2]
        // — w1/w3 adjacent so the pair-GEMV reads one contiguous range.
        uint8_t *p13, *p2, *s13, *s2;
        int cache_slot = -1;
        bool use_cache = false;
        if (cache && cache->slots > 0
            && w1b + w2b + w3b <= cache->pack_stride
            && s1b + s2b + s3b <= cache->scale_stride) {
            const int flat = static_cast<int>(layer) * experts_per_layer + idx;
            if (flat >= 0 && flat < cache->total_experts)
                cache_slot = cache->h_slot_of[flat];
            if (cache_slot < 0) {
                // Miss: pick a victim slot — per-layer round-robin partition
                // when configured, else the global LRU tail — and stage the
                // expert into it in the pair layout (w1, w3, w2 / s1, s3, s2).
                if (cache->layer_slots > 0) {
                    const int base = layer * cache->layer_slots;
                    cache_slot = base
                        + (cache->h_rotor[layer] % cache->layer_slots);
                    ++cache->h_rotor[layer];
                } else {
                    cache_slot = cache->h_lru[cache->slots - 1];
                }
                const int victim = cache->h_slot_owner[cache_slot];
                if (victim >= 0 && victim < cache->total_experts)
                    cache->h_slot_of[victim] = -1;
                uint8_t *pk = cache->d_pack
                    + static_cast<int64_t>(cache_slot) * cache->pack_stride;
                uint8_t *sc = cache->d_scale
                    + static_cast<int64_t>(cache_slot) * cache->scale_stride;
                cudaError_t e = cudaSuccess;
                e = cudaMemcpyAsync(pk, en.host_w1, static_cast<size_t>(w1b),
                                    cudaMemcpyHostToDevice, copy_stream ? copy_stream : stream);
                e = cudaMemcpyAsync(pk + w1b, en.host_w3, static_cast<size_t>(w3b),
                                    cudaMemcpyHostToDevice, copy_stream ? copy_stream : stream);
                if (e != cudaSuccess) return e;
                e = cudaMemcpyAsync(pk + w1b + w3b, en.host_w2, static_cast<size_t>(w2b),
                                    cudaMemcpyHostToDevice, copy_stream ? copy_stream : stream);
                if (e != cudaSuccess) return e;
                e = cudaMemcpyAsync(sc, en.host_w1_scales, static_cast<size_t>(s1b),
                                    cudaMemcpyHostToDevice, copy_stream ? copy_stream : stream);
                if (e != cudaSuccess) return e;
                e = cudaMemcpyAsync(sc + s1b, en.host_w3_scales, static_cast<size_t>(s3b),
                                    cudaMemcpyHostToDevice, copy_stream ? copy_stream : stream);
                if (e != cudaSuccess) return e;
                e = cudaMemcpyAsync(sc + s1b + s3b, en.host_w2_scales, static_cast<size_t>(s2b),
                                    cudaMemcpyHostToDevice, copy_stream ? copy_stream : stream);
                if (e != cudaSuccess) return e;
                cache->h_slot_of[flat] = cache_slot;
                cache->h_slot_owner[cache_slot] = flat;
            }
            // Move the slot to the MRU front (global-LRU mode only; the
            // partitioned mode uses the per-layer rotor instead).
            if (cache->layer_slots <= 0) {
                for (int i = 0; i < cache->slots; ++i) {
                    if (cache->h_lru[i] == cache_slot) {
                        for (int j = i; j > 0; --j)
                            cache->h_lru[j] = cache->h_lru[j - 1];
                        cache->h_lru[0] = cache_slot;
                        break;
                    }
                }
            }
            p13 = cache->d_pack + static_cast<int64_t>(cache_slot) * cache->pack_stride;
            p2 = p13 + w1b + w3b;
            s13 = cache->d_scale + static_cast<int64_t>(cache_slot) * cache->scale_stride;
            s2 = s13 + s1b + s3b;
            use_cache = true;
            if (std::getenv("LSX_DS_VERIFY_READ")) {
                cudaStreamSynchronize(stream);
                static std::vector<uint8_t> hr;
                hr.resize(static_cast<size_t>(w1b + w3b + w2b));
                cudaMemcpy(hr.data(), p13, hr.size(), cudaMemcpyDeviceToHost);
                int bad = memcmp(hr.data(), en.host_w1, w1b) != 0
                    || memcmp(hr.data() + w1b, en.host_w3, w3b) != 0
                    || memcmp(hr.data() + w1b + w3b, en.host_w2, w2b) != 0;
                static std::vector<uint8_t> hs;
                hs.resize(static_cast<size_t>(s1b + s3b + s2b));
                cudaMemcpy(hs.data(), s13, hs.size(), cudaMemcpyDeviceToHost);
                int bads = memcmp(hs.data(), en.host_w1_scales, s1b) != 0
                    || memcmp(hs.data() + s1b, en.host_w3_scales, s3b) != 0
                    || memcmp(hs.data() + s1b + s3b, en.host_w2_scales, s2b) != 0;
                fprintf(stderr, "VREAD flat=%d slot=%d pk=%d sc=%d\n",
                        flat, cache_slot, bad ? 0 : 1, bads ? 0 : 1);
            }
        }
        if (!use_cache) {
            if (w1b + w2b + w3b > pack_stage_bytes
                || s1b + s2b + s3b > scale_stage_bytes)
                continue;
            p13 = d_pack_stage;
            p2 = d_pack_stage + w1b + w3b;
            s13 = d_scale_stage;
            s2 = d_scale_stage + s1b + s3b;

cudaError_t e = cudaSuccess;
            if (en.pinned_blob) {
                // Pinned-blob fast path: the blob is already laid out
                // [w1|w3|w2][s1|s3|s2] — two large contiguous DMAs instead
                // of six fragments.
                e = cudaMemcpyAsync(p13, en.pinned_blob,
                    static_cast<size_t>(en.blob_pack_bytes),
                    cudaMemcpyHostToDevice,
                    copy_stream ? copy_stream : stream);
                if (e != cudaSuccess) return e;
                e = cudaMemcpyAsync(s13,
                    en.pinned_blob + en.blob_pack_bytes,
                    static_cast<size_t>(en.blob_scale_bytes),
                    cudaMemcpyHostToDevice,
                    copy_stream ? copy_stream : stream);
                if (e != cudaSuccess) return e;
            } else {
            e = cudaMemcpyAsync(p13, en.host_w1, static_cast<size_t>(w1b),
                                cudaMemcpyHostToDevice, copy_stream ? copy_stream : stream);
            if (e != cudaSuccess) return e;
            e = cudaMemcpyAsync(p13 + w1b, en.host_w3, static_cast<size_t>(w3b),
                                cudaMemcpyHostToDevice, copy_stream ? copy_stream : stream);
            if (e != cudaSuccess) return e;
            e = cudaMemcpyAsync(p2, en.host_w2, static_cast<size_t>(w2b),
                                cudaMemcpyHostToDevice, copy_stream ? copy_stream : stream);
            if (e != cudaSuccess) return e;
            e = cudaMemcpyAsync(s13, en.host_w1_scales, static_cast<size_t>(s1b),
                                cudaMemcpyHostToDevice, copy_stream ? copy_stream : stream);
            if (e != cudaSuccess) return e;
            e = cudaMemcpyAsync(s13 + s1b, en.host_w3_scales, static_cast<size_t>(s3b),
                                cudaMemcpyHostToDevice, copy_stream ? copy_stream : stream);
            if (e != cudaSuccess) return e;
            e = cudaMemcpyAsync(s2, en.host_w2_scales, static_cast<size_t>(s2b),
                                cudaMemcpyHostToDevice, copy_stream ? copy_stream : stream);
            if (e != cudaSuccess) return e;
            }
        }
        // Fence the staging copies for THIS expert; the compute stream
        // waits here so the w13 GEMV never reads a half-copied slot. Copies
        // of the remaining experts keep flowing on copy_stream in parallel.
        if (copy_stream) {
            cudaError_t ee = cudaEventRecord(stage_ev[k], copy_stream);
            if (ee != cudaSuccess) return ee;
            ee = cudaStreamWaitEvent(stream, stage_ev[k], 0);
            if (ee != cudaSuccess) return ee;
        }
        cudaError_t e = run_routed_expert_ffn(stream, d_normed,
            p13, s13, en.w1_global_scale, en.w3_global_scale,
            p2, s2, en.w2_global_scale,
            d_yg, d_y2, d_output, h_top_weights[k]);
        if (e != cudaSuccess) return e;
    }
    return cudaSuccess;
}

// Legacy entry point (no resident cache) — kept stable for existing callers
// and tests; the production pipeline uses run_routed_experts_ex with a cache.
cudaError_t run_routed_experts(
    cudaStream_t stream,
    const __half *d_normed,
    const ExpertEntry *db_entries, int layer, int experts_per_layer,
    const int *h_top_indices, const float *h_top_weights, int topk,
    float *d_output,
    __half *d_yg, __half *d_y2,
    uint8_t *d_pack_stage, uint8_t *d_scale_stage,
    int64_t pack_stage_bytes, int64_t scale_stage_bytes)
{
    return run_routed_experts_ex(stream, nullptr, nullptr, nullptr, d_normed,
        db_entries, layer, experts_per_layer,
        h_top_indices, h_top_weights, topk,
        d_output, d_yg, d_y2,
        d_pack_stage, d_scale_stage, pack_stage_bytes, scale_stage_bytes,
        nullptr);
}

// Prefetch the top-k experts' H2D staging onto @p copy_stream as early as
// possible (immediately after the router readback), so the transfers overlap
// the shared-expert FFN the compute stream is about to run. Fills resident
// cache slots in the pair layout; misses only — hits are free. Records
// @p ev on @p copy_stream when any copy was issued (the consumer waits on it).
cudaError_t issue_routed_staging(
    cudaStream_t copy_stream, cudaEvent_t ev, ExpertDeviceCache *cache,
    const ExpertEntry *db_entries, int layer, int experts_per_layer,
    const int *h_top_indices, int topk)
{
    if (!copy_stream || !cache || cache->slots <= 0) return cudaSuccess;
    // Bisect switch: pure-plumbing run (no fills, no copies).
    if (std::getenv("LSX_DS_PF_NOOP")) return cudaSuccess;
    bool issued = false;
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
        if (w1b + w2b + w3b > cache->pack_stride
            || s1b + s2b + s3b > cache->scale_stride)
            continue;
        const int flat = static_cast<int>(layer) * experts_per_layer + idx;
        if (flat < 0 || flat >= cache->total_experts) continue;
        if (cache->h_slot_of[flat] >= 0) continue;   // resident hit
        int slot = cache->h_lru[cache->slots - 1];
        const int victim = cache->h_slot_owner[slot];
        if (victim >= 0 && victim < cache->total_experts)
            cache->h_slot_of[victim] = -1;
        uint8_t *pk = cache->d_pack + static_cast<int64_t>(slot) * cache->pack_stride;
        uint8_t *sc = cache->d_scale + static_cast<int64_t>(slot) * cache->scale_stride;
        cudaError_t e = cudaSuccess;
if (en.pinned_blob) {
            // Pinned-blob fast path (two contiguous DMAs).
            e = cudaMemcpyAsync(pk, en.pinned_blob,
                static_cast<size_t>(en.blob_pack_bytes),
                cudaMemcpyHostToDevice, copy_stream);
            if (e != cudaSuccess) return e;
            e = cudaMemcpyAsync(sc, en.pinned_blob + en.blob_pack_bytes,
                static_cast<size_t>(en.blob_scale_bytes),
                cudaMemcpyHostToDevice, copy_stream);
            if (e != cudaSuccess) return e;
        } else {
        e = cudaMemcpyAsync(pk, en.host_w1, static_cast<size_t>(w1b),
                            cudaMemcpyHostToDevice, copy_stream);
        if (e != cudaSuccess) return e;
        e = cudaMemcpyAsync(pk + w1b, en.host_w3, static_cast<size_t>(w3b),
                            cudaMemcpyHostToDevice, copy_stream);
        if (e != cudaSuccess) return e;
        e = cudaMemcpyAsync(pk + w1b + w3b, en.host_w2, static_cast<size_t>(w2b),
                            cudaMemcpyHostToDevice, copy_stream);
        if (e != cudaSuccess) return e;
        e = cudaMemcpyAsync(sc, en.host_w1_scales, static_cast<size_t>(s1b),
                            cudaMemcpyHostToDevice, copy_stream);
        if (e != cudaSuccess) return e;
        e = cudaMemcpyAsync(sc + s1b, en.host_w3_scales, static_cast<size_t>(s3b),
                            cudaMemcpyHostToDevice, copy_stream);
        if (e != cudaSuccess) return e;
        e = cudaMemcpyAsync(sc + s1b + s3b, en.host_w2_scales, static_cast<size_t>(s2b),
                            cudaMemcpyHostToDevice, copy_stream);
        if (e != cudaSuccess) return e;
        }
        cache->h_slot_of[flat] = slot;
        cache->h_slot_owner[slot] = flat;
        if (std::getenv("LSX_DS_VERIFY_COPY")) {
            cudaStreamSynchronize(copy_stream);
            static std::vector<uint8_t> hb;
            hb.resize(static_cast<size_t>(w1b + w3b + w2b));
            cudaMemcpy(hb.data(), pk, hb.size(), cudaMemcpyDeviceToHost);
            int bad = memcmp(hb.data(), en.host_w1, w1b) != 0
                || memcmp(hb.data() + w1b, en.host_w3, w3b) != 0
                || memcmp(hb.data() + w1b + w3b, en.host_w2, w2b) != 0;
            fprintf(stderr, "VCOPY flat=%d slot=%d match=%d\n",
                    flat, slot, bad ? 0 : 1);
        }
        cache->h_slot_of[flat] = slot;
        cache->h_slot_owner[slot] = flat;
        // MRU front.
        for (int i = 0; i < cache->slots; ++i) {
            if (cache->h_lru[i] == slot) {
                for (int j = i; j > 0; --j)
                    cache->h_lru[j] = cache->h_lru[j - 1];
                cache->h_lru[0] = slot;
                break;
            }
        }
        // MRU front.
        for (int i = 0; i < cache->slots; ++i) {
            if (cache->h_lru[i] == slot) {
                for (int j = i; j > 0; --j)
                    cache->h_lru[j] = cache->h_lru[j - 1];
                cache->h_lru[0] = slot;
                break;
            }
        }
        issued = true;
    }
    if (issued) {
        cudaError_t e = cudaEventRecord(ev, copy_stream);
        if (e != cudaSuccess) return e;
        // Track this batch's sources against the completion event.
        for (int k = 0; k < topk; ++k) {
            int idx = h_top_indices[k];
            if (idx < 0 || idx >= experts_per_layer) continue;
            const ExpertEntry &en = db_entries[
                static_cast<size_t>(layer) * experts_per_layer + idx];
            if (!en.host_w1) continue;
            g_pending_copies.push_back({en.host_w1, ev});
        }
    }
    return cudaSuccess;
}

} // namespace gpu
} // namespace lsxcommon
