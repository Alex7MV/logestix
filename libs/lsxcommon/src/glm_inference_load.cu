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

#include "glm_device_math.h"

namespace lsxcommon {
namespace glm {

namespace {

using device::GlmExpfGlibc;
using device::GlmRsqrtfGlibc;

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
    float s = g / (1.0f + GlmExpfGlibc(-g));
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

// RMSNorm over a [rows, dim] FP16 matrix, one block per row. Bit-exact mirror
// of the host RmsNormFP32 reference: SINGLE sequential fp32 scan over
// 0..dim-1 for the square sum, GlmRsqrtfGlibc(acc/dim + eps) output scale.
// Matches
// GlmRmsNormKernel in glm_inference.cu; both rely on the -fmad=false target
// flag to keep the accumulation non-contracted (the -O0 host baseline cannot
// FMA either).
__global__ void GlmMoERmsNormKernel(const __half *__restrict__ x,
                                    const __half *__restrict__ w,
                                    __half *__restrict__ y,
                                    int rows, int dim, float eps)
{
    constexpr int kBlock = kGlmRmsBlock;
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float s_rms;
    if (tid == 0) {
        const __half *xr = x + row * dim;
        float acc = 0.0f;
        for (int i = 0; i < dim; ++i) {
            const float v = __half2float(xr[i]);
            acc += v * v;
        }
        s_rms = GlmRsqrtfGlibc(acc / static_cast<float>(dim) + eps);
    }
    __syncthreads();
    const float rinv = s_rms;
    const __half *xr = x + row * dim;
    __half *yr = y + row * dim;
    for (int i = tid; i < dim; i += kBlock) {
        float v = __half2float(xr[i]);
        yr[i] = __float2half(v * rinv * __half2float(w[i]));
    }
}

// Router logits: logit[t][e] = (float)Σ_q F(gate[e][q])·F(normed[t][q]) in
// FP64 over ascending q — bit-exact mirror of CpuRouter. gate row-major
// [kGlmNumRouted, kGlmHidden]; out [seq, kGlmNumRouted] token-major
// (column-major [kGlmNumRouted, seq] at out[e + t*256], the same layout the
// previous cuBLAS call produced). Plain non-contracted double mul+add
// (-fmad=false), like the -O0 host baseline.
__global__ void GlmMoELogitsFp64Kernel(const __half *__restrict__ gate,
                                       const __half *__restrict__ x,
                                       float *__restrict__ out,
                                       int m, int k, int n)
{
    const int64_t total = static_cast<int64_t>(m) * n;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
    if (idx >= total) return;
    const int t = static_cast<int>(idx / m);
    const int e = static_cast<int>(idx - static_cast<int64_t>(t) * m);
    const __half *gr = gate + static_cast<int64_t>(e) * k;
    const __half *xr = x + static_cast<int64_t>(t) * k;
    double acc = 0.0;
    for (int q = 0; q < k; ++q)
        acc += static_cast<double>(__half2float(gr[q])) *
               static_cast<double>(__half2float(xr[q]));
    out[idx] = static_cast<float>(acc);
}

// Per-token sequential router selection: bit-exact mirror of CpuRouter.
// One thread per token replays the host order exactly:
//   1. lmax  = sequential max over the UNBIASED logits (ascending e),
//   2. psum  = sequential Σ GlmExpfGlibc(logit[e] - lmax) (ascending e),
//   3. inv   = 1/psum, prob[e] = GlmExpfGlibc(logit[e] - lmax) * inv,
//   4. top-k: for each rank, a sequential scan over 0..255 picking the largest
//      sel[e] = logit[e] + bias[e] among the unused experts (ties -> smallest
//      eid), then coeff = prob[eid]. Identical loop/order to the host.
__global__ void GlmMoETopkKernel(const float *__restrict__ d_logits,
                                 const float *__restrict__ d_gate_bias,
                                 GlmRoutedSlot *__restrict__ d_routing,
                                 int seq)
{
    const int t = blockIdx.x;
    if (t >= seq) return;
    if (threadIdx.x != 0) return;

    const float *lm = d_logits + static_cast<size_t>(t) * kGlmNumRouted;

    float lmax = -1e30f;
    for (int e = 0; e < kGlmNumRouted; ++e) lmax = fmaxf(lmax, lm[e]);
    float psum = 0.0f;
    for (int e = 0; e < kGlmNumRouted; ++e) psum += GlmExpfGlibc(lm[e] - lmax);
    const float inv = 1.0f / psum;

    bool used[kGlmNumRouted];
    for (int e = 0; e < kGlmNumRouted; ++e) used[e] = false;

    GlmRoutedSlot *row = d_routing + static_cast<size_t>(t) * kGlmMoETopK;
    for (int r = 0; r < kGlmMoETopK; ++r) {
        float bv = -1e30f;
        int   bi = -1;
        for (int e = 0; e < kGlmNumRouted; ++e) {
            if (used[e]) continue;
            const float v = d_gate_bias ? lm[e] + d_gate_bias[e] : lm[e];
            if (v > bv || (v == bv && e < bi)) { bv = v; bi = e; }
        }
        row[r].eid   = bi;
        row[r].coeff = GlmExpfGlibc(lm[bi] - lmax) * inv;
        used[bi] = true;
    }
}

// FP64-accumulate FP16 GEMM (same contract as GlmGemmFp64Kernel in
// glm_inference.cu): out[t][i] = H((float)Σ_q F(w[i][q])·F(x[t][q])), plain
// non-contracted double accumulation, w row-major [m, k], x row-major [k, n],
// out column-major [m, n].
__global__ void GlmGemmFp64Kernel(const __half *__restrict__ w,
                                  const __half *__restrict__ x,
                                  __half *__restrict__ out,
                                  int m, int k, int n)
{
    const int64_t total = static_cast<int64_t>(m) * n;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
    if (idx >= total) return;
    const int t = static_cast<int>(idx / m);
    const int i = static_cast<int>(idx - static_cast<int64_t>(t) * m);
    const __half *wr = w + static_cast<int64_t>(i) * k;
    const __half *xr = x + static_cast<int64_t>(t) * k;
    double acc = 0.0;
    for (int q = 0; q < k; ++q)
        acc += static_cast<double>(__half2float(wr[q])) *
               static_cast<double>(__half2float(xr[q]));
    out[idx] = __float2half(static_cast<float>(acc));
}

// ── IndexShare attention kernels (Cycle 5, layers 3-77) ─────────────────────
// The central memory-mapped shared indexer mask arrays (state->q_key/key/
// score/slot) are recomputed ONLY on the structural modulo check
// (layer_idx % 4 == 0); every intermediate layer bypasses these projection
// kernels entirely and reuses the cached slots. Every dot product below is a
// straight ascending-loop FP32 fmaf accumulation with FP16 rounding only at
// the store points, so the device math is exactly reproducible on the host.

constexpr int   kIdxAppendBlock = 256;
constexpr float kGlmIndexAttnScale = 0.088388347648318440f; // 1/sqrt(128)

__global__ void GlmIdxInitKernel(GlmIndexShareState *__restrict__ s)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        s->n_used = 0;
        s->turn_id = -1;
        s->refresh_layer = -1;
    }
}

// Single-block turn-append: on the first call of a new turn, copies the seq
// hidden vectors into the pre-allocated KV staging buffer and bumps n_used.
// n_used is read through the atomic return so every thread sees the same
// slot base; the whole registration stays kernel-ordered (no host sync).
__global__ void GlmIdxAppendKernel(GlmIndexShareState *__restrict__ s,
                                   const __half *__restrict__ x,
                                   int seq, int turn)
{
    const bool append = (turn != s->turn_id);
    __shared__ int s_base;
    if (threadIdx.x == 0)
        s_base = append ? atomicAdd(&s->n_used, seq) : s->n_used;
    __syncthreads();
    if (!append) return;
    const int base = s_base;
    const int total = seq * kGlmHidden;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        const int t = idx / kGlmHidden;
        const int h = idx - t * kGlmHidden;
        s->kv[base + t][h] = x[idx];
    }
    if (threadIdx.x == 0) s->turn_id = turn;
}

// Refresh: score[τ][h] = dot(kv[τ], weights_proj[h]) and the winning head
// slot[τ] = argmax_h score (strict >, so ties resolve to the smallest h).
// One warp per slot: lane h owns the h-th head dot (ascending-k fmaf).
__global__ void GlmIdxScoreKernel(
    GlmIndexShareState *__restrict__ s,
    const __half *__restrict__ weights_proj)
{
    const int slot = blockIdx.x;
    if (slot >= s->n_used) return;
    const int h = threadIdx.x;
    const __half *v = s->kv[slot];
    const __half *wp = weights_proj + h * kGlmHidden;
    float acc = 0.0f;
    for (int k = 0; k < kGlmHidden; ++k)
        acc = fmaf(__half2float(v[k]), __half2float(wp[k]), acc);
    __shared__ float s_sc[kGlmIndexNHeads];
    s_sc[h] = acc;
    __syncthreads();
    if (h == 0) {
        int   best = 0;
        float bv   = s_sc[0];
        for (int j = 1; j < kGlmIndexNHeads; ++j)
            if (s_sc[j] > bv) { bv = s_sc[j]; best = j; }
        s->slot[slot] = best;
    }
    __syncthreads();
    s->score[slot][h] = s_sc[h];
}

// Refresh: key[τ][r] = RMSNorm(dot(kv[τ], wk[r])) * k_norm[r] + k_bias[r].
// One thread per dim r computes the ascending-k dot; the RMS reduction is a
// single-thread ascending scan over the 128 shared kraw values to keep the
// float sequence identical to the host reference.
__global__ void GlmIdxKeyKernel(GlmIndexShareState *__restrict__ s,
                                const __half *__restrict__ wk,
                                const __half *__restrict__ k_norm,
                                const float *__restrict__ k_bias)
{
    const int slot = blockIdx.x;
    if (slot >= s->n_used) return;
    const int r = threadIdx.x;
    const __half *v = s->kv[slot];
    const __half *wkr = wk + r * kGlmHidden;
    float acc = 0.0f;
    for (int k = 0; k < kGlmHidden; ++k)
        acc = fmaf(__half2float(v[k]), __half2float(wkr[k]), acc);
    __shared__ float s_kraw[kGlmIndexHeadDim];
    __shared__ float s_rms;
    s_kraw[r] = acc;
    __syncthreads();
    if (r == 0) {
        float sq = 0.0f;
        for (int j = 0; j < kGlmIndexHeadDim; ++j)
            sq = fmaf(s_kraw[j], s_kraw[j], sq);
        s_rms = sqrtf(sq / static_cast<float>(kGlmIndexHeadDim) + kGlmRmsEps);
    }
    __syncthreads();
    const float kk =
        (acc / s_rms) * __half2float(k_norm[r]) + k_bias[r];
    s->key[slot][r] = __float2half(kk);
}

// Refresh: q_key[τ][r] = dot(kv[τ][0:kGlmQLoraRank], wq_b[slot[τ]][r]).
// The winning head index comes from GlmIdxScoreKernel (stream-ordered). The
// cached winner-head query is what intermediate layers reuse verbatim.
__global__ void GlmIdxQueryKernel(GlmIndexShareState *__restrict__ s,
                                  const __half *__restrict__ wq_b)
{
    const int slot = blockIdx.x;
    if (slot >= s->n_used) return;
    const int r = threadIdx.x;
    const int h = s->slot[slot];
    const __half *v = s->kv[slot];
    const __half *wqr = wq_b + (h * kGlmIndexHeadDim + r) * kGlmQLoraRank;
    float acc = 0.0f;
    for (int k = 0; k < kGlmQLoraRank; ++k)
        acc = fmaf(__half2float(v[k]), __half2float(wqr[k]), acc);
    s->q_key[slot][r] = __float2half(acc);
}

__global__ void GlmIdxStampKernel(GlmIndexShareState *__restrict__ s,
                                  int layer_idx)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) s->refresh_layer = layer_idx;
}

// Attention over the occupied KV slots, one block per token. Per-token order
// is fully deterministic: one ascending-dot per slot, softmax scans in slot
// order, and the residual blend accumulates across slots in order. The
// per-block query/key/sim cache is staged in CUDA shared memory; the sparse
// value pointers are pulled straight from the KV staging buffer.
__global__ void GlmIdxAttnKernel(GlmIndexShareState *__restrict__ s,
                                 const __half *__restrict__ x,
                                 __half *__restrict__ out, int seq)
{
    __shared__ float s_sim[kGlmIndexMaxSlots];
    __shared__ float s_p[kGlmIndexMaxSlots];
    __shared__ float s_m;
    __shared__ float s_sum;

    const int t    = blockIdx.x;
    const int tid  = threadIdx.x;
    const int n    = s->n_used;
    const int base = n - seq + t;
    const int nout = kGlmHidden;

    if (n <= 0) {
        for (int h = tid; h < nout; h += blockDim.x)
            out[t * nout + h] = x[t * nout + h];
        return;
    }

    const __half *qq = s->q_key[base];
    for (int sl = tid; sl < n; sl += blockDim.x) {
        const __half *ks = s->key[sl];
        float acc = 0.0f;
        for (int d = 0; d < kGlmIndexHeadDim; ++d)
            acc = fmaf(__half2float(qq[d]), __half2float(ks[d]), acc);
        s_sim[sl] = acc * kGlmIndexAttnScale;
    }
    __syncthreads();

    if (tid == 0) {
        float m = -FLT_MAX;
        for (int sl = 0; sl < n; ++sl) m = fmaxf(m, s_sim[sl]);
        s_m = m;
    }
    __syncthreads();
    for (int sl = tid; sl < n; sl += blockDim.x)
        s_p[sl] = GlmExpfGlibc(s_sim[sl] - s_m);
    __syncthreads();
    if (tid == 0) {
        float sum = 0.0f;
        for (int sl = 0; sl < n; ++sl) sum += s_p[sl];
        s_sum = sum;
    }
    __syncthreads();
    const float inv = 1.0f / s_sum;
    for (int sl = tid; sl < n; sl += blockDim.x) s_p[sl] *= inv;
    __syncthreads();

    for (int h = tid; h < nout; h += blockDim.x) {
        float acc = __half2float(x[t * nout + h]);
        for (int sl = 0; sl < n; ++sl)
            acc = fmaf(s_p[sl], __half2float(s->kv[sl][h]), acc);
        out[t * nout + h] = __float2half(acc);
    }
}

} // anonymous namespace

cudaError_t RunGlmMoERouter(cublasHandle_t cublas_h, cudaStream_t stream,
                            const GlmLayerWeights &lw, int seq,
                            const __half *d_hidden, const float *d_gate_bias,
                            __half *d_normed, float *d_logits,
                            GlmRoutedSlot *d_routing)
{
    (void)cublas_h;
    if (!lw.d_post_attn_norm || !lw.d_router_gate ||
        !d_hidden || !d_normed || !d_logits || !d_routing || seq <= 0)
        return cudaErrorInvalidValue;

    // 1. RMSNorm(raw activations) -> d_normed, rows=seq on the async stream.
    GlmMoERmsNormKernel<<<seq, kGlmRmsBlock, 0, stream>>>(
        d_hidden, lw.d_post_attn_norm, d_normed, seq, kGlmHidden,
        kGlmRmsEps);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;

    // 2. logits[t][e] = (float)Σ_k normed[t][k]·gate[e][k] in FP64.
    const int logits_total = kGlmNumRouted * seq;
    const int logits_blocks = (logits_total + kGlmRmsBlock - 1) / kGlmRmsBlock;
    GlmMoELogitsFp64Kernel<<<logits_blocks, kGlmRmsBlock, 0, stream>>>(
        lw.d_router_gate, d_normed, d_logits, kGlmNumRouted, kGlmHidden, seq);
    e = cudaGetLastError();
    if (e != cudaSuccess) return e;

    // 3. Per-token sequential softmax + top_k selection (host-order mirror).
    GlmMoETopkKernel<<<seq, 1, 0, stream>>>(
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

    const int inter_blocks =
        (kGlmMoeIntermediate + kBlock - 1) / kBlock;
    const int hidden_blocks = (kGlmHidden + kBlock - 1) / kBlock;

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
        if (e != cudaSuccess) break;
        GlmGemmFp64Kernel<<<inter_blocks, kBlock, 0, stream>>>(
            d_deq, x_t, d_inter, kGlmMoeIntermediate, kGlmHidden, 1);
        e = cudaGetLastError();
        if (e != cudaSuccess) break;

        // gate = w3^T @ x
        GlmDequantNvfp4Kernel<<<deq_grid, kGlmDequantThreads, 0, stream>>>(
            eb.d_w3_packed, eb.d_w3_scales, eb.w3_global_scale, d_deq, kMatEl);
        e = cudaGetLastError();
        if (e != cudaSuccess) break;
        GlmGemmFp64Kernel<<<inter_blocks, kBlock, 0, stream>>>(
            d_deq, x_t, d_gate, kGlmMoeIntermediate, kGlmHidden, 1);
        e = cudaGetLastError();
        if (e != cudaSuccess) break;

        // swiglu: inter = silu(inter) * gate, in place (FP32 math)
        GlmSwigluKernel<<<inter_blocks, kBlock, 0, stream>>>(
            d_inter, d_gate, kGlmMoeIntermediate);
        e = cudaGetLastError();
        if (e != cudaSuccess) break;

        // down = w2^T @ swig  (w2 row-major [hidden, inter])
        GlmDequantNvfp4Kernel<<<deq_grid, kGlmDequantThreads, 0, stream>>>(
            eb.d_w2_packed, eb.d_w2_scales, eb.w2_global_scale, d_deq, kMatEl);
        e = cudaGetLastError();
        if (e != cudaSuccess) break;
        GlmGemmFp64Kernel<<<hidden_blocks, kBlock, 0, stream>>>(
            d_deq, d_inter, d_tmp, kGlmHidden, kGlmMoeIntermediate, 1);
        e = cudaGetLastError();
        if (e != cudaSuccess) break;

        // out_t[h] += coeff * tmp[h]
        GlmExpertAccumKernel<<<hidden_blocks, kBlock, 0, stream>>>(
            out_t, d_tmp, rs.coeff, kGlmHidden);
        e = cudaGetLastError();
        if (e != cudaSuccess) break;
    }

    cudaFreeHost(h_routing);
    return e;
}

// ── Cycle 5: IndexShare attention pass (layers 3-77) ────────────────────────
cudaError_t GlmIndexShareReset(GlmIndexShareState *d_state, cudaStream_t stream)
{
    if (!d_state) return cudaErrorInvalidValue;
    GlmIdxInitKernel<<<1, 1, 0, stream>>>(d_state);
    return cudaGetLastError();
}

cudaError_t RunGlmIndexShareAttention(
    cudaStream_t stream, int layer_idx, int turn,
    const GlmLayerWeights &lw, int seq,
    const __half *d_hidden, __half *d_out, GlmIndexShareState *d_state)
{
    if (!d_state || !d_hidden || !d_out || seq <= 0 ||
        layer_idx < 0 || layer_idx > 78)
        return cudaErrorInvalidValue;

    // Rigid structural modulo check: the group head (layer_idx % 4 == 0) is
    // the ONLY place the central shared indexer mask arrays are calculated,
    // block-aligned and refreshed. A group head that carries no indexer
    // weights (null pointers, e.g. heterogeneous GLM grid) degrades to reuse.
    const bool refresh = (layer_idx % 4 == 0) &&
                         lw.d_idx_wq_b && lw.d_idx_wk &&
                         lw.d_idx_weights_proj && lw.d_idx_k_norm;

    // 1. Once-per-turn KV registration into the pre-allocated staging buffer
    //    (kernel-ordered, never a host-side synchronization fence).
    GlmIdxAppendKernel<<<1, kIdxAppendBlock, 0, stream>>>(
        d_state, d_hidden, seq, turn);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;

    // 2. Refresh branch: recalculate + block-align + refresh the shared masks.
    //    Intermediate layers (layer_idx % 4 != 0) skip every projection
    //    kernel below and pull the cached slot pointers from d_state.
    if (refresh) {
        GlmIdxScoreKernel<<<kGlmIndexMaxSlots, kGlmIndexNHeads, 0, stream>>>(
            d_state, lw.d_idx_weights_proj);
        e = cudaGetLastError();
        if (e != cudaSuccess) return e;
        GlmIdxKeyKernel<<<kGlmIndexMaxSlots, kGlmIndexHeadDim, 0, stream>>>(
            d_state, lw.d_idx_wk, lw.d_idx_k_norm, lw.d_idx_k_norm_bias);
        e = cudaGetLastError();
        if (e != cudaSuccess) return e;
        GlmIdxQueryKernel<<<kGlmIndexMaxSlots, kGlmIndexHeadDim, 0, stream>>>(
            d_state, lw.d_idx_wq_b);
        e = cudaGetLastError();
        if (e != cudaSuccess) return e;
        GlmIdxStampKernel<<<1, 1, 0, stream>>>(d_state, layer_idx);
        e = cudaGetLastError();
        if (e != cudaSuccess) return e;
    }

    // 3. Sparse attention over the cached slots (shared-memory key/query
    //    stage, KV values in the staging buffer), residual-added output.
    GlmIdxAttnKernel<<<seq, kIdxAppendBlock, 0, stream>>>(
        d_state, d_hidden, d_out, seq);
    return cudaGetLastError();
}

} // namespace glm
} // namespace lsxcommon