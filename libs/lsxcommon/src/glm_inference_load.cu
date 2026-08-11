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

} // namespace glm
} // namespace lsxcommon