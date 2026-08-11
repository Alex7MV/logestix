// GLM-5.2 MoE router kernels: RMSNorm(raw activations) -> gate projection ->
// warp-per-token softmax + top_k=8 selection, all on the caller's explicit
// async stream. Selection is a warp-level total-order arg-extreme scan built
// on __shfl_down_sync; the per-token routing record is written as 8-byte
// aligned 64-bit grid stores (see GlmRoutedSlot in glm_inference.h).
#include "lsxcommon/glm_inference.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cfloat>
#include <cstring>

namespace lsxcommon {
namespace glm {

namespace {

constexpr int kGlmRmsBlock = 256;

// ── NVFP4 expert-weight decode (Cycle 4, layers 3-77) ──────────────────────
// One E4M3 power-of-two block scale covers kGlmNvfp4Block (128) consecutive
// elements of the packed E2M1 nibble stream; a single decoded E8M0 global
// modifier scales the whole tensor. Dequant invariant:
//   out[i] = E2M1(nibble_i) * 2^(((scales[i>>7]>>3)&0xF) - 7) * global_mod
// Advances 4 elements per thread (2 packed bytes), which always stay inside
// one 128-block except across the ragged tail.

__constant__ float kGlmE2M1Table[16] = {
      0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
     -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__device__ __forceinline__ float GlmE4M3P2ToFloat(uint8_t e4m3)
{
    if (e4m3 == 0) return 0.0f;
    const uint32_t e4 = (e4m3 >> 3) & 0x0Fu;
    return __uint_as_float((e4 + 120u) << 23);
}

constexpr int kGlmDequantThreads = 256;
constexpr int kGlmDequantPerThread = 4;

__launch_bounds__(kGlmDequantThreads)
__global__ void GlmDequantNvfp4Kernel(
    const uint8_t *__restrict__ packed,
    const uint8_t *__restrict__ scales,
    float                        global_modifier,
    __half *__restrict__         out,
    int64_t                      n)
{
    const int64_t i0 = (static_cast<int64_t>(blockIdx.x) * blockDim.x +
                        static_cast<int64_t>(threadIdx.x)) * kGlmDequantPerThread;
    if (i0 >= n) return;
    if (i0 + kGlmDequantPerThread <= n) {
        const uint8_t b0 = packed[i0 >> 1];
        const uint8_t b1 = packed[(i0 >> 1) + 1];
        const uint8_t nib0 = b0 & 0x0Fu;
        const uint8_t nib1 = b0 >> 4;
        const uint8_t nib2 = b1 & 0x0Fu;
        const uint8_t nib3 = b1 >> 4;
        const float sc = GlmE4M3P2ToFloat(scales[i0 >> 7]) * global_modifier;
        out[i0 + 0] = __float2half(kGlmE2M1Table[nib0] * sc);
        out[i0 + 1] = __float2half(kGlmE2M1Table[nib1] * sc);
        out[i0 + 2] = __float2half(kGlmE2M1Table[nib2] * sc);
        out[i0 + 3] = __float2half(kGlmE2M1Table[nib3] * sc);
        return;
    }
    for (int64_t i = i0; i < n; ++i) {
        const uint8_t b   = packed[i >> 1];
        const uint8_t nib = (i & 1) ? static_cast<uint8_t>(b >> 4)
                                    : static_cast<uint8_t>(b & 0x0Fu);
        const float sc = GlmE4M3P2ToFloat(scales[i >> 7]) * global_modifier;
        out[i] = __float2half(kGlmE2M1Table[nib] * sc);
    }
}

// SwiGLU projection epilogue: w1(gate) branch through SiLU, w3(up) branch as
// the plain multiplier (GLM-5.2, no clamp). FP32 math, FP16 in/out.
__global__ void GlmSwigluKernel(__half *__restrict__ gate,
                                const __half *__restrict__ up, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = __half2float(gate[i]);
    float u = __half2float(up[i]);
    float s = g / (1.0f + expf(-g));
    gate[i] = __float2half(s * u);
}

// Expert W2 epilogue: out[t][h] += coeff * tmp[h]. FP32 accumulate, FP16 out.
__global__ void GlmExpertAccumKernel(__half *__restrict__ out,
                                     const __half *__restrict__ tmp,
                                     float coeff, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2half(__half2float(out[i]) + coeff * __half2float(tmp[i]));
}

// RMSNorm over a [rows, dim] FP16 matrix, one block per row. Mirrors the
// verified GlmRmsNormKernel in glm_inference.cu (shfl_xor warp reduce +
// shared-memory cross-warp reduce).
__global__ void GlmMoERmsNormKernel(const __half *__restrict__ x,
                                    const __half *__restrict__ w,
                                    __half *__restrict__ y,
                                    int rows, int dim, float eps)
{
    constexpr int kBlock = kGlmRmsBlock;
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float s_red[kBlock / 32];
    float acc = 0.0f;
    for (int i = tid; i < dim; i += kBlock) {
        float v = __half2float(x[row * dim + i]);
        acc += v * v;
    }
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffffu, acc, off);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) s_red[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        float v = (tid < (kBlock >> 5)) ? s_red[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_xor_sync(0xffffffffu, v, off);
        if (lane == 0) s_red[0] = v;
    }
    __syncthreads();
    const float rinv = rsqrtf(s_red[0] / static_cast<float>(dim) + eps);
    const __half *xr = x + row * dim;
    __half *yr = y + row * dim;
    for (int i = tid; i < dim; i += kBlock) {
        float v = __half2float(xr[i]);
        yr[i] = __float2half(v * rinv * __half2float(w[i]));
    }
}

// Warp-per-token top_k selection kernel.
//
//   d_logits:  [seq, kGlmNumRouted] FP32, row-major token-major. Row t lives
//              at t*kGlmNumRouted; each lane owns 8 consecutive experts
//              (256/32).
//   d_gate_bias: [kGlmNumRouted] FP32 per-expert selection bias or nullptr.
//   d_routing: [seq, kGlmMoETopK] GlmRoutedSlot grid, 8-byte aligned.
//
// Pipeline per token (one warp):
//   1. Load 8 logits/lane, reduce global max over the UNBIASED logits via
//      __shfl_down_sync, then plain softmax weights.
//   2. For each of the kGlmMoETopK slots, find the global arg-extreme of the
//      selection keys ((logit + bias), total order: larger key, smaller eid
//      on tie) via __shfl_down_sync; clear the winning lane's bit so the next
//      iteration picks the next-highest distinct expert.
//   3. Lanes 0..kGlmMoETopK-1 store the winners with a single 64-bit write
//      per slot; every slot address is an 8-byte boundary (alignas(8) record
//      with 8-byte stride).
__global__ void GlmMoETopkKernel(const float *__restrict__ d_logits,
                                 const float *__restrict__ d_gate_bias,
                                 GlmRoutedSlot *__restrict__ d_routing,
                                 int seq)
{
    constexpr int kLanes   = 32;
    constexpr int kPerLane = kGlmNumRouted / kLanes; // 8 experts per lane
    constexpr unsigned kFullMask = 0xffffffffu;

    const int lane   = threadIdx.x & 31;
    const int warp   = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    const int t      = static_cast<int>(blockIdx.x) * nwarps + warp;
    if (t >= seq) return;

    const float *lm  = d_logits + static_cast<size_t>(t) * kGlmNumRouted;
    const int    e0  = lane * kPerLane;

    // Softmax max + denominator over the unbiased logits.
    float lmax = -FLT_MAX;
#pragma unroll
    for (int j = 0; j < kPerLane; ++j)
        lmax = fmaxf(lmax, lm[e0 + j]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        lmax = fmaxf(lmax, __shfl_down_sync(kFullMask, lmax, off));
    // Only lane 0 holds the true global max after the down-reduction; the
    // softmax normalization must be identical across every lane, so broadcast.
    lmax = __shfl_sync(kFullMask, lmax, 0);

    float prob[kPerLane];
    float psum = 0.0f;
#pragma unroll
    for (int j = 0; j < kPerLane; ++j) {
        const float p = expf(lm[e0 + j] - lmax);
        prob[j] = p;
        psum += p;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        psum += __shfl_down_sync(kFullMask, psum, off);
    // Same as lmax: broadcast the global sum so every lane normalizes with the
    // identical denominator (partial-sum lanes would otherwise scale winners
    // differently, corrupting the coefficient equality across equal-slot ties).
    psum = __shfl_sync(kFullMask, psum, 0);
    const float inv = 1.0f / psum;
#pragma unroll
    for (int j = 0; j < kPerLane; ++j) prob[j] *= inv;

    // Selection keys: logit + optional per-expert correction bias.
    float sel[kPerLane];
#pragma unroll
    for (int j = 0; j < kPerLane; ++j)
        sel[j] = d_gate_bias ? lm[e0 + j] + d_gate_bias[e0 + j] : lm[e0 + j];

    GlmRoutedSlot top[kGlmMoETopK];
    uint8_t live = 0xffu; // per-lane candidate bitmap (8 experts / lane)
#pragma unroll
    for (int r = 0; r < kGlmMoETopK; ++r) {
        float bv = -FLT_MAX;
        int   bi = -1;
#pragma unroll
        for (int j = 0; j < kPerLane; ++j) {
            if (!(live & (1u << j))) continue;
            const float v = sel[j];
            const int   ei = e0 + j;
            if (v > bv || (v == bv && ei < bi)) { bv = v; bi = ei; }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_down_sync(kFullMask, bv, off);
            const int   oi = __shfl_down_sync(kFullMask, bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        // Global winner (total order makes it unique), broadcast from lane 0.
        const int gi = __shfl_sync(kFullMask, bi, 0);
        const int owner = gi >> 3; // lane owning expert gi
        const float garea = (lane == owner) ? prob[gi & 7] : 0.0f;
        const float gcoeff = __shfl_sync(kFullMask, garea, owner);
        top[r].eid   = gi;
        top[r].coeff = gcoeff;
        if (lane == owner) live &= static_cast<uint8_t>(~(1u << (gi & 7)));
    }

    // 8-byte-boundary aligned 64-bit grid writes, one slot per active lane.
    GlmRoutedSlot *row = d_routing + static_cast<size_t>(t) * kGlmMoETopK;
    if (lane < kGlmMoETopK) {
        static_assert(sizeof(GlmRoutedSlot) == 8u,
                      "routing slot must pack into 8 bytes");
        long long bits;
        std::memcpy(&bits, &top[lane], sizeof(bits));
        reinterpret_cast<long long *>(row)[lane] = bits;
    }
}

} // anonymous namespace

cudaError_t RunGlmMoERouter(cublasHandle_t cublas_h, cudaStream_t stream,
                            const GlmLayerWeights &lw, int seq,
                            const __half *d_hidden, const float *d_gate_bias,
                            __half *d_normed, float *d_logits,
                            GlmRoutedSlot *d_routing)
{
    if (!lw.d_post_attn_norm || !lw.d_router_gate ||
        !d_hidden || !d_normed || !d_logits || !d_routing || seq <= 0)
        return cudaErrorInvalidValue;

    // 1. RMSNorm(raw activations) -> d_normed, rows=seq on the async stream.
    GlmMoERmsNormKernel<<<seq, kGlmRmsBlock, 0, stream>>>(
        d_hidden, lw.d_post_attn_norm, d_normed, seq, kGlmHidden,
        kGlmRmsEps);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;

    // 2. logits[t][e] = sum_k normed[t][k] * gate[e][k].
    // Gate row-major [kGlmNumRouted, kGlmHidden] is a column-major
    // [kGlmHidden, kGlmNumRouted] matrix with lda = kGlmHidden, so
    // C = gate^T @ normed with m=256, n=seq, k=6144; the FP32 C column-major
    // layout C[e + t*256] doubles as the token-major [seq, 256] row for the
    // top-k kernel (same index arithmetic).
    cublasSetStream(cublas_h, stream);
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    const cublasStatus_t st = cublasGemmEx(
        cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
        kGlmNumRouted, seq, kGlmHidden, &alpha,
        lw.d_router_gate, CUDA_R_16F, kGlmHidden,
        d_normed,        CUDA_R_16F, kGlmHidden,
        &beta, d_logits, CUDA_R_32F, kGlmNumRouted,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) return cudaErrorInvalidValue;

    // 3. Warp-per-token softmax + top_k selection, 4 tokens per 128-thread
    // block; Blackwell occupancy stays 48 warps/SM-class agnostic (small grid).
    constexpr int kBlock = 128;
    const int tokens_per_block = kBlock / 32;
    const int grid = (seq + tokens_per_block - 1) / tokens_per_block;
    GlmMoETopkKernel<<<grid, kBlock, 0, stream>>>(
        d_logits, d_gate_bias, d_routing, seq);
    return cudaGetLastError();
}

cudaError_t RunGlmExpertsStack(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const GlmRoutedSlot *d_routing, int seq,
    const __half *d_act,
    const GlmExpertDeviceBlock *h_experts, int n_experts,
    __half *d_out, __half *d_scratch)
{
    constexpr int64_t kMatEl = kGlmExpertMatEl;
    constexpr int64_t kExpectPacked = kMatEl / 2;
    constexpr int64_t kExpectScale  = kMatEl / kGlmNvfp4Block;
    constexpr int    kBlock  = 256;

    if (!d_routing || !d_act || !h_experts || !d_out || !d_scratch ||
        seq <= 0 || n_experts <= 0 || !cublas_h)
        return cudaErrorInvalidValue;

    const int slots = seq * kGlmMoETopK;
    if (slots < 0) return cudaErrorInvalidValue;

    // Stage the tiny routing grid back to host so per-slot expert launches
    // can be picked from the host descriptor registry. Single localized
    // stream barrier on the caller's stream; never a device-wide sync.
    GlmRoutedSlot *h_routing = nullptr;
    cudaError_t e = cudaMallocHost(&h_routing,
                                   static_cast<size_t>(slots) *
                                       sizeof(GlmRoutedSlot));
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(h_routing, d_routing,
                        static_cast<size_t>(slots) * sizeof(GlmRoutedSlot),
                        cudaMemcpyDeviceToHost, stream);
    if (e == cudaSuccess) e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess) { cudaFreeHost(h_routing); return e; }

    // Scratch layout (halves): dequant w-deck / tmp / inter / gate.
    __half *d_deq   = d_scratch;                                      // kMatEl
    __half *d_tmp   = d_deq + kMatEl;                                 // kGlmHidden
    __half *d_inter = d_tmp + kGlmHidden;                             // kGlmMoeIntermediate
    __half *d_gate  = d_inter + kGlmMoeIntermediate;                  // kGlmMoeIntermediate

    e = cudaMemsetAsync(d_out, 0,
                        static_cast<size_t>(seq) * kGlmHidden * sizeof(__half),
                        stream);
    if (e != cudaSuccess) { cudaFreeHost(h_routing); return e; }

    const int64_t dequant_per_block =
        static_cast<int64_t>(kGlmDequantThreads) * kGlmDequantPerThread;
    const int deq_grid =
        static_cast<int>((kMatEl + dequant_per_block - 1) / dequant_per_block);

    cublasSetStream(cublas_h, stream);
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    for (int s = 0; s < slots; ++s) {
        const GlmRoutedSlot &rs = h_routing[s];
        if (rs.eid < 0 || rs.eid >= n_experts) continue;
        const GlmExpertDeviceBlock &eb = h_experts[rs.eid];
        if (!eb.d_w1_packed || !eb.d_w1_scales ||
            !eb.d_w2_packed || !eb.d_w2_scales ||
            !eb.d_w3_packed || !eb.d_w3_scales ||
            eb.w1_packed_bytes != kExpectPacked ||
            eb.w2_packed_bytes != kExpectPacked ||
            eb.w3_packed_bytes != kExpectPacked ||
            eb.w1_scale_bytes  != kExpectScale  ||
            eb.w2_scale_bytes  != kExpectScale  ||
            eb.w3_scale_bytes  != kExpectScale)
            continue; // inactive / foreign format: deterministic zero contribution

        const int   t    = s / kGlmMoETopK;
        const __half *x_t = d_act + static_cast<size_t>(t) * kGlmHidden;
        __half *out_t = d_out + static_cast<size_t>(t) * kGlmHidden;

        // inter = w1^T @ x  (w1 row-major [inter, hidden] dequantized to FP16)
        GlmDequantNvfp4Kernel<<<deq_grid, kGlmDequantThreads, 0, stream>>>(
            eb.d_w1_packed, eb.d_w1_scales, eb.w1_global_scale, d_deq, kMatEl);
        e = cudaGetLastError();
        cublasStatus_t cs = CUBLAS_STATUS_SUCCESS;
        if (e == cudaSuccess)
            cs = cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                kGlmMoeIntermediate, 1, kGlmHidden, &alpha,
                d_deq, CUDA_R_16F, kGlmHidden,
                x_t,   CUDA_R_16F, kGlmHidden,
                &beta, d_inter, CUDA_R_16F, kGlmMoeIntermediate,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (cs != CUBLAS_STATUS_SUCCESS) e = cudaErrorInvalidValue;
        if (e != cudaSuccess) break;

        // gate = w3^T @ x
        GlmDequantNvfp4Kernel<<<deq_grid, kGlmDequantThreads, 0, stream>>>(
            eb.d_w3_packed, eb.d_w3_scales, eb.w3_global_scale, d_deq, kMatEl);
        e = cudaGetLastError();
        if (e == cudaSuccess)
            cs = cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                kGlmMoeIntermediate, 1, kGlmHidden, &alpha,
                d_deq, CUDA_R_16F, kGlmHidden,
                x_t,   CUDA_R_16F, kGlmHidden,
                &beta, d_gate, CUDA_R_16F, kGlmMoeIntermediate,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (cs != CUBLAS_STATUS_SUCCESS) e = cudaErrorInvalidValue;
        if (e != cudaSuccess) break;

        // swiglu: inter = silu(inter) * gate, in place (FP32 math)
        const int swig_blocks = (kGlmMoeIntermediate + kBlock - 1) / kBlock;
        GlmSwigluKernel<<<swig_blocks, kBlock, 0, stream>>>(
            d_inter, d_gate, kGlmMoeIntermediate);
        e = cudaGetLastError();
        if (e != cudaSuccess) break;

        // down = w2^T @ swig  (w2 row-major [hidden, inter])
        GlmDequantNvfp4Kernel<<<deq_grid, kGlmDequantThreads, 0, stream>>>(
            eb.d_w2_packed, eb.d_w2_scales, eb.w2_global_scale, d_deq, kMatEl);
        e = cudaGetLastError();
        if (e == cudaSuccess)
            cs = cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                kGlmHidden, 1, kGlmMoeIntermediate, &alpha,
                d_deq,   CUDA_R_16F, kGlmMoeIntermediate,
                d_inter, CUDA_R_16F, kGlmMoeIntermediate,
                &beta, d_tmp, CUDA_R_16F, kGlmHidden,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (cs != CUBLAS_STATUS_SUCCESS) e = cudaErrorInvalidValue;
        if (e != cudaSuccess) break;

        // out_t[h] += coeff * tmp[h]
        const int acc_blocks = (kGlmHidden + kBlock - 1) / kBlock;
        GlmExpertAccumKernel<<<acc_blocks, kBlock, 0, stream>>>(
            out_t, d_tmp, rs.coeff, kGlmHidden);
        e = cudaGetLastError();
        if (e != cudaSuccess) break;
    }

    cudaFreeHost(h_routing);
    return e;
}

} // namespace glm
} // namespace lsxcommon