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
#include "glm_expert_kernels.cuh"
#include "nvfp4_codec.h"
#include "lsxcommon/log.h"

namespace lsxcommon {
namespace glm {

namespace {

using device::GlmExpfGlibc;
using device::GlmRsqrtfGlibc;

constexpr int kGlmRmsBlock = 256;

// ── NVFP4 expert-weight decode (Cycle 4, layers 3-77) ──────────────────────
// One E4M3 power-of-two block scale covers kGlmNvfp4Block (16) consecutive
// elements of the packed E2M1 nibble stream; a single decoded E8M0 global
// modifier scales the whole tensor. Dequant invariant:
//   out[i] = E2M1(nibble_i) * 2^(((scales[i>>4]>>3)&0xF) - 7) * global_mod
// Advances 4 elements per thread (2 packed bytes), which always stay inside
// one 16-block except across the ragged tail.

// E2M1 table + E4M3 scale now live in nvfp4_codec.h (gpu::nvfp4); alias them
// into this anonymous namespace so GlmDequantNvfp4Kernel binds to the single
// text source. Bit-exact on valid power-of-two block scales (note: the shared
// E4M3ScaleToFloat guards on e4==0, equal to the previous E4M3ScaleToFloat
// e4m3==0 guard for mantissa-zero scales).
using gpu::nvfp4::kE2M1Table;
using gpu::nvfp4::E4M3ScaleToFloat;

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
        // i0 is a multiple of 4 → the two packed bytes land on a 2-byte
        // boundary (single ushort load) and the four output halves on an
        // 8-byte boundary (single uint2 store). Element values and order are
        // unchanged — bit-exact.
        const unsigned short pair =
            *reinterpret_cast<const unsigned short *>(packed + (i0 >> 1));
        const uint8_t b0 = static_cast<uint8_t>(pair & 0xFFu);
        const uint8_t b1 = static_cast<uint8_t>(pair >> 8);
        const uint8_t nib0 = b0 & 0x0Fu;
        const uint8_t nib1 = b0 >> 4;
        const uint8_t nib2 = b1 & 0x0Fu;
        const uint8_t nib3 = b1 >> 4;
        const float sc = E4M3ScaleToFloat(scales[i0 >> 4]) * global_modifier;
        uint2 ov;
        const __half h0 = __float2half(kE2M1Table[nib0] * sc);
        const __half h1 = __float2half(kE2M1Table[nib1] * sc);
        const __half h2 = __float2half(kE2M1Table[nib2] * sc);
        const __half h3 = __float2half(kE2M1Table[nib3] * sc);
        ov.x = static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&h0)) |
               (static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&h1)) << 16);
        ov.y = static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&h2)) |
               (static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&h3)) << 16);
        *reinterpret_cast<uint2 *>(out + i0) = ov;
        return;
    }
    for (int64_t i = i0; i < n; ++i) {
        const uint8_t b   = packed[i >> 1];
        const uint8_t nib = (i & 1) ? static_cast<uint8_t>(b >> 4)
                                    : static_cast<uint8_t>(b & 0x0Fu);
        const float sc = E4M3ScaleToFloat(scales[i >> 4]) * global_modifier;
        out[i] = __float2half(kE2M1Table[nib] * sc);
    }
}

// SwiGLU / RMSNorm / FP64-GEMM kernels: single textual source in
// glm_expert_kernels.cuh (-fmad=false). MoE router launches bind to them
// below (GlmRealSwigluKernel / GlmRmsNormKernel / GlmGemmFp64Kernel).

// Expert W2 epilogue: out[t][h] += coeff * tmp[h]. FP32 accumulate, FP16 out.
__global__ void GlmExpertAccumKernel(__half *__restrict__ out,
                                     const __half *__restrict__ tmp,
                                     float coeff, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2half(__half2float(out[i]) + coeff * __half2float(tmp[i]));
}

// GlmRmsNormKernel (canonical, in glm_expert_kernels.cuh) already provides the
// [rows, dim] one-block-per-row RMSNorm for the MoE router launches below.

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
    // Thread-per-token: the per-token selection loop is unchanged (bit-exact
    // sequential order), but prefill now uses seq threads in one wave instead
    // of seq nearly-empty single-thread blocks.
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= seq) return;

    const float *lm = d_logits + static_cast<size_t>(t) * kGlmNumRouted;

    float lmax = -1e30f;
    for (int e = 0; e < kGlmNumRouted; ++e) lmax = fmaxf(lmax, lm[e]);
    float psum = 0.0f;
    for (int e = 0; e < kGlmNumRouted; ++e) psum += GlmExpfGlibc(lm[e] - lmax);
    const float inv = 1.0f / psum;

    bool used[kGlmNumRouted];
    for (int e = 0; e < kGlmNumRouted; ++e) used[e] = false;

    GlmRoutedSlot *row = d_routing + static_cast<size_t>(t) * kGlmMoETopK;
    // Pre-poison every slot so an all-NaN early exit leaves eid = -1 records
    // (skipped downstream) instead of uninitialized memory.
    for (int r = 0; r < kGlmMoETopK; ++r) {
        row[r].eid = -1;
        row[r].coeff = 0.0f;
    }
    for (int r = 0; r < kGlmMoETopK; ++r) {
        float bv = -1e30f;
        int   bi = -1;
        for (int e = 0; e < kGlmNumRouted; ++e) {
            if (used[e]) continue;
            const float v = d_gate_bias ? lm[e] + d_gate_bias[e] : lm[e];
            if (v > bv || (v == bv && e < bi)) { bv = v; bi = e; }
        }
        if (bi < 0) return;  // all logits/bias sums NaN: no valid selection
        // Single 64-bit store of the 8-byte-aligned slot (coeff | eid), as the
        // header contract documents: the token's kGlmMoETopK slots form one
        // 8-byte-aligned record. GlmRoutedSlot lays eid at offset 0 and coeff
        // at offset 4, so on little-endian the coeff bits must occupy the HIGH
        // word of the packed u64 and eid the LOW word.
        const float coeff = GlmExpfGlibc(lm[bi] - lmax) * inv;
        const uint64_t packed =
            (static_cast<uint64_t>(__float_as_uint(coeff)) << 32) |
            static_cast<uint32_t>(bi);
        *reinterpret_cast<uint64_t *>(&row[r]) = packed;
        used[bi] = true;
    }
}

// GlmGemmFp64Kernel: canonical definition in glm_expert_kernels.cuh
// (bit-exact FP64-accum FP16 GEMM); IndexShare launches below bind to it.

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
// Defensive capacity guard: a turn whose slot base would exceed the fixed
// kGlmIndexMaxSlots staging is consumed without writing (the KV staging and
// the attention shared arrays are fixed-size), so no out-of-bounds access is
// ever issued even if a caller bypasses the host-side capacity validation.
__global__ void GlmIdxAppendKernel(GlmIndexShareState *__restrict__ s,
                                   const __half *__restrict__ x,
                                   int seq, int turn)
{
    const bool append = (turn != s->turn_id);
    __shared__ int  s_base;
    __shared__ bool s_ok;
    if (threadIdx.x == 0) {
        if (append) {
            const int base = atomicAdd(&s->n_used, seq);
            s_base = base;
            s_ok = (base + seq <= kGlmIndexMaxSlots);
            if (!s_ok) s->turn_id = turn;   // consume the turn even on overflow
        } else {
            s_base = s->n_used;
            s_ok = false;
        }
    }
    __syncthreads();
    if (!s_ok) return;
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
        (acc / s_rms) * __half2float(k_norm[r]) + (k_bias ? k_bias[r] : 0.0f);
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
    // Clamp to the fixed staging capacity: if a caller bypassed host-side
    // validation and n_used grew beyond kGlmIndexMaxSlots, the shared arrays
    // and KV staging below must not be indexed past the allocation.
    const int n    = (s->n_used < kGlmIndexMaxSlots) ? s->n_used
                                                     : kGlmIndexMaxSlots;
    const int base = n - seq + t;
    const int nout = kGlmHidden;

    if (n <= 0) {
        for (int h = tid; h < nout; h += blockDim.x)
            out[t * nout + h] = x[t * nout + h];
        return;
    }
    // A turn that overflows capacity is consumed without appending, so
    // n_used can be smaller than seq for late tokens; base would go negative.
    // Identity-copy those rows instead of indexing q_key out of bounds.
    if (base < 0) {
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
        float m = -1e30f;   // sentinel, same as the host reference
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
    GlmRmsNormKernel<<<seq, kGlmRmsBlock, 0, stream>>>(
    d_hidden, lw.d_post_attn_norm, d_normed, kGlmHidden,
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
    GlmMoETopkKernel<<<(seq + kGlmRmsBlock - 1) / kGlmRmsBlock, kGlmRmsBlock, 0, stream>>>(
        d_logits, d_gate_bias, d_routing, seq);
    return cudaGetLastError();
}

// ── Real GLM-5.2 router (GlmMoeDsaTopkRouter) ────────────────────────────────
// Per-token sequential selection mirroring GlmMoeDsaTopkRouter.forward exactly
// (n_group = 1 / topk_group = 1, so the group-mask step is a no-op):
//   1. score[e] = sigmoid(logits[e])            (per-expert, NOT softmax)
//   2. sel[e]   = score[e] + e_score_correction_bias[e]  (bias ONLY for choice)
//   3. top-8 by sel[e] over the full 256 (ties -> smallest eid)
//   4. coeff = score[eid]; then norm_topk_prob: coeff /= (Σ coeff + 1e-20),
//      then routed_scaling_factor: coeff *= 2.5
__global__ void GlmRealTopkKernel(const float *__restrict__ d_logits,
                                  const float *__restrict__ d_gate_bias,
                                  GlmRoutedSlot *__restrict__ d_routing,
                                  int seq)
{
    // Thread-per-token (see GlmMoETopkKernel): identical per-token order.
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= seq) return;

    const float *lm = d_logits + static_cast<size_t>(t) * kGlmNumRouted;

    float score[kGlmNumRouted];
    for (int e = 0; e < kGlmNumRouted; ++e)
        score[e] = 1.0f / (1.0f + device::GlmExpfGlibc(-lm[e]));

    bool used[kGlmNumRouted];
    for (int e = 0; e < kGlmNumRouted; ++e) used[e] = false;

    GlmRoutedSlot *row = d_routing + static_cast<size_t>(t) * kGlmMoETopK;
    // Pre-poison every slot so a mid-rank break (all-NaN scores) cannot leave
    // uninitialized records that pass downstream bounds checks.
    for (int r = 0; r < kGlmMoETopK; ++r) {
        row[r].eid = -1;
        row[r].coeff = 0.0f;
    }
    float sum = 0.0f;
    int written = 0;
    for (int r = 0; r < kGlmMoETopK; ++r) {
        float bv = -1e30f;
        int   bi = -1;
        for (int e = 0; e < kGlmNumRouted; ++e) {
            if (used[e]) continue;
            const float v = d_gate_bias ? score[e] + d_gate_bias[e] : score[e];
            if (v > bv || (v == bv && e < bi)) { bv = v; bi = e; }
        }
        // Single 64-bit store (eid | coeff), same record layout as GlmMoETopkKernel.
        if (bi < 0) break;  // all scores NaN/Inf: no valid selection remains
        const float coeff = score[bi];
        const uint64_t packed =
            (static_cast<uint64_t>(__float_as_uint(coeff)) << 32) |
            static_cast<uint32_t>(bi);
        *reinterpret_cast<uint64_t *>(&row[r]) = packed;
        sum += coeff;
        used[bi] = true;
        ++written;
    }
    const float denom = sum + 1e-20f;
    for (int r = 0; r < written; ++r)
        row[r].coeff = (row[r].coeff / denom) * 2.5f;
}

// Real GLM-5.2 router for the generation pass: identical entry contract to
// RunGlmMoERouter (post-attn RMSNorm -> logits -> d_routing grid), but the
// top-k selection follows GlmMoeDsaTopkRouter (sigmoid scores + bias-only
// selection + norm_topk_prob + routed_scaling_factor=2.5).
cudaError_t RunGlmRealRouter(cublasHandle_t cublas_h, cudaStream_t stream,
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
    GlmRmsNormKernel<<<seq, kGlmRmsBlock, 0, stream>>>(
    d_hidden, lw.d_post_attn_norm, d_normed, kGlmHidden,
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

    // 3. Per-token sigmoid selection + norm_topk + scaling.
    GlmRealTopkKernel<<<(seq + kGlmRmsBlock - 1) / kGlmRmsBlock, kGlmRmsBlock, 0, stream>>>(
        d_logits, d_gate_bias, d_routing, seq);
    return cudaGetLastError();
}

// The per-routed-slot expert decode chain shared by RunGlmExpertsStack and
// RunGlmExpertsStackOnDemand: w1^T @ x -> inter, w3^T @ x -> gate, silu(inter)
// * gate -> swig, w2^T @ swig -> tmp, out += coeff * tmp. `eb` is a single
// NVFP4 block descriptor; inactive / foreign-format blocks contribute nothing
// (deterministic zero contribution). All kernels run on <stream>.
cudaError_t RunGlmExpertSlot(cublasHandle_t cublas_h, cudaStream_t stream,
                             const GlmExpertDeviceBlock &eb,
                             const __half *x_t, __half *out_t,
                             __half *d_scratch, float coeff)
{
    (void)cublas_h;
    constexpr int64_t kMatEl = kGlmExpertMatEl;
    constexpr int64_t kExpectPacked = kMatEl / 2;
    constexpr int64_t kExpectScale  = kMatEl / kGlmNvfp4Block;
    constexpr int    kBlock  = 256;

    if (!eb.d_w1_packed || !eb.d_w1_scales ||
        !eb.d_w2_packed || !eb.d_w2_scales ||
        !eb.d_w3_packed || !eb.d_w3_scales ||
        eb.w1_packed_bytes != kExpectPacked ||
        eb.w2_packed_bytes != kExpectPacked ||
        eb.w3_packed_bytes != kExpectPacked ||
        eb.w1_scale_bytes  != kExpectScale  ||
        eb.w2_scale_bytes  != kExpectScale  ||
        eb.w3_scale_bytes  != kExpectScale)
        return cudaSuccess; // inactive / foreign format: deterministic zero contribution

    // Scratch layout (halves): dequant w-deck / tmp / inter / gate.
    __half *d_deq   = d_scratch;                                      // kMatEl
    __half *d_tmp   = d_deq + kMatEl;                                 // kGlmHidden
    __half *d_inter = d_tmp + kGlmHidden;                             // kGlmMoeIntermediate
    __half *d_gate  = d_inter + kGlmMoeIntermediate;                  // kGlmMoeIntermediate

    const int64_t dequant_per_block =
        static_cast<int64_t>(kGlmDequantThreads) * kGlmDequantPerThread;
    const int deq_grid =
        static_cast<int>((kMatEl + dequant_per_block - 1) / dequant_per_block);
    const int inter_blocks =
        (kGlmMoeIntermediate + kBlock - 1) / kBlock;
    const int hidden_blocks = (kGlmHidden + kBlock - 1) / kBlock;

    // inter = w1^T @ x  (w1 row-major [inter, hidden] dequantized to FP16)
    GlmDequantNvfp4Kernel<<<deq_grid, kGlmDequantThreads, 0, stream>>>(
        eb.d_w1_packed, eb.d_w1_scales, eb.w1_global_scale, d_deq, kMatEl);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        log::error("[glm_expert] dequant w1 fault: " +
                   std::string(cudaGetErrorString(e)));
        return e;
    }
    GlmGemmFp64Kernel<<<inter_blocks, kBlock, 0, stream>>>(
        d_deq, x_t, d_inter, kGlmMoeIntermediate, kGlmHidden, 1);
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        log::error("[glm_expert] gemm w1 fault: " +
                   std::string(cudaGetErrorString(e)));
        return e;
    }

    // gate = w3^T @ x
    GlmDequantNvfp4Kernel<<<deq_grid, kGlmDequantThreads, 0, stream>>>(
        eb.d_w3_packed, eb.d_w3_scales, eb.w3_global_scale, d_deq, kMatEl);
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        log::error("[glm_expert] dequant w3 fault: " +
                   std::string(cudaGetErrorString(e)));
        return e;
    }
    GlmGemmFp64Kernel<<<inter_blocks, kBlock, 0, stream>>>(
        d_deq, x_t, d_gate, kGlmMoeIntermediate, kGlmHidden, 1);
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        log::error("[glm_expert] gemm w3 fault: " +
                   std::string(cudaGetErrorString(e)));
        return e;
    }

    // swiglu: inter = silu(inter) * gate, in place (FP32 math)
    GlmRealSwigluKernel<<<inter_blocks, kBlock, 0, stream>>>(
        d_inter, d_gate, kGlmMoeIntermediate);
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        log::error("[glm_expert] swiglu fault: " +
                   std::string(cudaGetErrorString(e)));
        return e;
    }

    // down = w2^T @ swig  (w2 row-major [hidden, inter])
    GlmDequantNvfp4Kernel<<<deq_grid, kGlmDequantThreads, 0, stream>>>(
        eb.d_w2_packed, eb.d_w2_scales, eb.w2_global_scale, d_deq, kMatEl);
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        log::error("[glm_expert] dequant w2 fault: " +
                   std::string(cudaGetErrorString(e)));
        return e;
    }
    GlmGemmFp64Kernel<<<hidden_blocks, kBlock, 0, stream>>>(
        d_deq, d_inter, d_tmp, kGlmHidden, kGlmMoeIntermediate, 1);
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        log::error("[glm_expert] gemm w2 fault: " +
                   std::string(cudaGetErrorString(e)));
        return e;
    }

    // out_t[h] += coeff * tmp[h]
    GlmExpertAccumKernel<<<hidden_blocks, kBlock, 0, stream>>>(
        out_t, d_tmp, coeff, kGlmHidden);
    return cudaGetLastError();
}

cudaError_t RunGlmExpertsStack(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const GlmRoutedSlot *d_routing, int seq,
    const __half *d_act,
    const GlmExpertDeviceBlock *h_experts, int n_experts,
    __half *d_out, __half *d_scratch,
    GlmRoutedSlot *h_routing_staging /* = nullptr */)
{
    if (!d_routing || !d_act || !h_experts || !d_out || !d_scratch ||
        seq <= 0 || n_experts <= 0 || !cublas_h)
        return cudaErrorInvalidValue;

    const int slots = seq * kGlmMoETopK;
    if (slots < 0) return cudaErrorInvalidValue;

    // Stage the tiny routing grid back to host so per-slot expert launches
    // can be picked from the host descriptor registry. Single localized
    // stream barrier on the caller's stream; never a device-wide sync.
    // The caller may pre-allocate the staging buffer once per generation
    // (RunGlmGeneration) to avoid a cudaMallocHost/FreeHost pair per call;
    // otherwise it is allocated here.
    const bool own_staging = (h_routing_staging == nullptr);
    GlmRoutedSlot *h_routing = h_routing_staging;
    cudaError_t e = cudaSuccess;
    if (own_staging) {
        e = cudaMallocHost(&h_routing,
                           static_cast<size_t>(slots) *
                               sizeof(GlmRoutedSlot));
        if (e != cudaSuccess) return e;
    }
    e = cudaMemcpyAsync(h_routing, d_routing,
                        static_cast<size_t>(slots) * sizeof(GlmRoutedSlot),
                        cudaMemcpyDeviceToHost, stream);
    if (e == cudaSuccess) e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess) {
        if (own_staging) cudaFreeHost(h_routing);
        return e;
    }

    e = cudaMemsetAsync(d_out, 0,
                        static_cast<size_t>(seq) * kGlmHidden * sizeof(__half),
                        stream);
    if (e != cudaSuccess) {
        if (own_staging) cudaFreeHost(h_routing);
        return e;
    }

    for (int s = 0; s < slots; ++s) {
        const GlmRoutedSlot &rs = h_routing[s];
        if (rs.eid < 0 || rs.eid >= n_experts) continue;
        const int   t    = s / kGlmMoETopK;
        const __half *x_t = d_act + static_cast<size_t>(t) * kGlmHidden;
        __half *out_t = d_out + static_cast<size_t>(t) * kGlmHidden;
        e = RunGlmExpertSlot(cublas_h, stream, h_experts[rs.eid],
                             x_t, out_t, d_scratch, rs.coeff);
        if (e != cudaSuccess) break;
    }

    if (own_staging) cudaFreeHost(h_routing);
    return e;
}

// VRAM-budgeted twin of RunGlmExpertsStack: identical routing-grid staging
// barrier, per-slot decode chain and deterministic inactive-expert skip, but
// the routed experts are pulled from a host memory-mapped index (GlmExpertDb)
// instead of a resident device descriptor table. For every occupied slot the
// expert's packed E2M1 payloads + E4M3 scales are streamed into the caller's
// reusable device staging buffers (one expert at a time) and the identical
// RunGlmExpertSlot chain runs against them. A full 385 GiB expert model never
// needs more than the pinned weights + ~2 × one-expert staging in VRAM.
cudaError_t RunGlmExpertsStackOnDemand(
    cublasHandle_t cublas_h, cudaStream_t stream,
    cudaStream_t stream_copy,
    const GlmRoutedSlot *d_routing, int seq,
    const __half *d_act,
    GlmExpertDb &db, int layer, int n_experts,
    __half *d_out, __half *d_scratch,
    uint8_t *d_pack_stage, uint8_t *d_scale_stage,
    uint8_t *d_pack_stage_b, uint8_t *d_scale_stage_b,
    cudaEvent_t ev_h2d[2], cudaEvent_t ev_gemm[2],
    GlmRoutedSlot *h_routing_staging /* = nullptr */,
    GlmExpertDeviceCache *xcache /* = nullptr */,
    cudaEvent_t ev_cache_gemm /* = nullptr */)
{
    constexpr int64_t kMatEl = kGlmExpertMatEl;
    constexpr int64_t kPackPerMat = kMatEl / 2;
    constexpr int64_t kScalePerMat = kMatEl / kGlmNvfp4Block;

    if (!d_routing || !d_act || !d_out || !d_scratch ||
        !d_pack_stage || !d_scale_stage ||
        seq <= 0 || n_experts <= 0 || layer < 0 || !cublas_h)
        return cudaErrorInvalidValue;

    const bool use_cache =
        xcache && xcache->slots > 0 && xcache->d_pack && xcache->d_scale &&
        xcache->h_slot_of && xcache->h_slot_owner && xcache->h_lru &&
        xcache->pack_stride >= 3 * kPackPerMat &&
        xcache->scale_stride >= 3 * kScalePerMat;

    const int slots = seq * kGlmMoETopK;
    if (slots < 0) return cudaErrorInvalidValue;

    // Same single localized stream barrier as RunGlmExpertsStack: stage the
    // tiny routing grid back to host so per-slot expert launches can be picked
    // from the host descriptor registry.
    const bool own_staging = (h_routing_staging == nullptr);
    GlmRoutedSlot *h_routing = h_routing_staging;
    cudaError_t e = cudaSuccess;
    if (own_staging) {
        e = cudaMallocHost(&h_routing,
                           static_cast<size_t>(slots) *
                               sizeof(GlmRoutedSlot));
        if (e != cudaSuccess) return e;
    }
    e = cudaMemcpyAsync(h_routing, d_routing,
                        static_cast<size_t>(slots) * sizeof(GlmRoutedSlot),
                        cudaMemcpyDeviceToHost, stream);
    if (e == cudaSuccess) e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess) {
        if (own_staging) cudaFreeHost(h_routing);
        return e;
    }

    e = cudaMemsetAsync(d_out, 0,
                        static_cast<size_t>(seq) * kGlmHidden * sizeof(__half),
                        stream);
    if (e != cudaSuccess) {
        if (own_staging) cudaFreeHost(h_routing);
        return e;
    }

    const bool two = stream_copy != nullptr && d_pack_stage_b && d_scale_stage_b &&
                     ev_h2d && ev_gemm;
    uint8_t *sets_p[2] = { d_pack_stage, d_pack_stage_b };
    uint8_t *sets_s[2] = { d_scale_stage, d_scale_stage_b };

    auto stage_slot = [&](int si, cudaStream_t cst) -> cudaError_t {
        const GlmRoutedSlot &rs = h_routing[si];
        if (rs.eid < 0 || rs.eid >= n_experts) return cudaSuccess;
        const size_t flat = static_cast<size_t>(layer) *
                                static_cast<size_t>(db.experts_per_layer) +
                            static_cast<size_t>(rs.eid);
        if (flat >= db.entries.size()) return cudaSuccess;
        const GlmExpertEntry &en = db.entries[flat];
        if (!en.host_w1 || !en.host_w2 || !en.host_w3 ||
            !en.host_w1_scales || !en.host_w2_scales || !en.host_w3_scales ||
            en.w1_bytes != kPackPerMat || en.w2_bytes != kPackPerMat ||
            en.w3_bytes != kPackPerMat ||
            en.w1_scales_bytes != kScalePerMat ||
            en.w2_scales_bytes != kScalePerMat ||
            en.w3_scales_bytes != kScalePerMat)
            return cudaSuccess;
        const int si_p = (si & 1);

        // Single-stream fallback always uses set 0; two-stream uses set parity.
        const int idx = two ? si_p : 0;
        uint8_t *p1 = sets_p[idx], *p2 = p1 + kPackPerMat, *p3 = p2 + kPackPerMat;
        uint8_t *s1 = sets_s[idx], *s2 = s1 + kScalePerMat, *s3 = s2 + kScalePerMat;
        const void *src[6] = { en.host_w1, en.host_w2, en.host_w3,
                               en.host_w1_scales, en.host_w2_scales, en.host_w3_scales };
        void *dst[6] = { p1, p2, p3, s1, s2, s3 };
        const size_t nb[6] = { (size_t)kPackPerMat, (size_t)kPackPerMat, (size_t)kPackPerMat,
                               (size_t)kScalePerMat, (size_t)kScalePerMat, (size_t)kScalePerMat };
        for (int f = 0; f < 6; ++f) {
            cudaError_t ce = cudaMemcpyAsync(dst[f], src[f], nb[f],
                                             cudaMemcpyHostToDevice, cst);
            if (ce != cudaSuccess) return ce;
        }
        return cudaSuccess;
    };

    // ── Device-cache helpers (active when use_cache) ────────────────────────
    // Eligibility gate shared by both cache paths: routing record in range,
    // entry resident and format-exact. Returns the flat db index or -1.
    auto cache_flat = [&](const GlmRoutedSlot &rs) -> int {
        if (rs.eid < 0 || rs.eid >= n_experts) return -1;
        const size_t flat = static_cast<size_t>(layer) *
                                static_cast<size_t>(db.experts_per_layer) +
                            static_cast<size_t>(rs.eid);
        if (flat >= db.entries.size()) return -1;
        const GlmExpertEntry &en = db.entries[flat];
        if (!en.host_w1 || !en.host_w2 || !en.host_w3 ||
            !en.host_w1_scales || !en.host_w2_scales || !en.host_w3_scales ||
            en.w1_bytes != kPackPerMat || en.w2_bytes != kPackPerMat ||
            en.w3_bytes != kPackPerMat ||
            en.w1_scales_bytes != kScalePerMat ||
            en.w2_scales_bytes != kScalePerMat ||
            en.w3_scales_bytes != kScalePerMat)
            return -1;
        return static_cast<int>(flat);
    };
    // Six H2D copies of one expert set into its own cache slot region.
    auto copy_into_cache_slot = [&](int slot, const GlmExpertEntry &en,
                                    cudaStream_t cst) -> cudaError_t {
        uint8_t *base_p = xcache->d_pack +
                          static_cast<size_t>(slot) * xcache->pack_stride;
        uint8_t *base_s = xcache->d_scale +
                          static_cast<size_t>(slot) * xcache->scale_stride;
        const void *src[6] = { en.host_w1, en.host_w2, en.host_w3,
                               en.host_w1_scales, en.host_w2_scales,
                               en.host_w3_scales };
        void *dst[6] = { base_p, base_p + kPackPerMat, base_p + 2 * kPackPerMat,
                         base_s, base_s + kScalePerMat, base_s + 2 * kScalePerMat };
        const size_t nb[6] = { (size_t)kPackPerMat, (size_t)kPackPerMat,
                               (size_t)kPackPerMat, (size_t)kScalePerMat,
                               (size_t)kScalePerMat, (size_t)kScalePerMat };
        for (int f = 0; f < 6; ++f) {
            cudaError_t ce = cudaMemcpyAsync(dst[f], src[f], nb[f],
                                             cudaMemcpyHostToDevice, cst);
            if (ce != cudaSuccess) return ce;
        }
        return cudaSuccess;
    };
    // NVFP4 descriptor block pointing at a cached slot region.
    auto eb_from_cache_slot = [&](int slot, const GlmExpertEntry &en,
                                  GlmExpertDeviceBlock &eb) {
        uint8_t *base_p = xcache->d_pack +
                          static_cast<size_t>(slot) * xcache->pack_stride;
        uint8_t *base_s = xcache->d_scale +
                          static_cast<size_t>(slot) * xcache->scale_stride;
        eb.d_w1_packed = base_p;                        eb.d_w1_scales = base_s;
        eb.w1_global_scale = en.w1_global_scale;
        eb.w1_packed_bytes = kPackPerMat;               eb.w1_scale_bytes = kScalePerMat;
        eb.d_w2_packed = base_p + kPackPerMat;          eb.d_w2_scales = base_s + kScalePerMat;
        eb.w2_global_scale = en.w2_global_scale;
        eb.w2_packed_bytes = kPackPerMat;               eb.w2_scale_bytes = kScalePerMat;
        eb.d_w3_packed = base_p + 2 * kPackPerMat;      eb.d_w3_scales = base_s + 2 * kScalePerMat;
        eb.w3_global_scale = en.w3_global_scale;
        eb.w3_packed_bytes = kPackPerMat;               eb.w3_scale_bytes = kScalePerMat;
    };

    cudaError_t last = cudaSuccess;

    if (!two) {
        // Single-stream cached path: hits run straight from the slot, misses
        // copy on the calling stream (plain stream ordering suffices).
        if (use_cache) {
            for (int s = 0; s < slots; ++s) {
                const GlmRoutedSlot &rs = h_routing[s];
                // Parse-first ordering (mirrors the legacy loop): the entry's
                // host pointers/byte counts are populated lazily by
                // ensure_glm_expert_resident, so it MUST run before any
                // db.entries field is inspected.
                if (rs.eid < 0 || rs.eid >= n_experts) continue;
                ensure_glm_expert_resident(db, layer, rs.eid, stream);
                const int flat = cache_flat(rs);
                if (flat < 0) continue;
                const GlmExpertEntry &en =
                    db.entries[static_cast<size_t>(flat)];
                int cslot = CacheLookup(*xcache, flat);
                if (cslot < 0) {
                    cslot = CacheInsert(*xcache, flat);
                    last = copy_into_cache_slot(cslot, en, stream);
                    if (last != cudaSuccess) break;
                }
                GlmExpertDeviceBlock eb;
                eb_from_cache_slot(cslot, en, eb);
                const int   t    = s / kGlmMoETopK;
                const __half *x_t = d_act + static_cast<size_t>(t) * kGlmHidden;
                __half *out_t = d_out + static_cast<size_t>(t) * kGlmHidden;
                last = RunGlmExpertSlot(cublas_h, stream, eb,
                                        x_t, out_t, d_scratch,
                                        rs.coeff);
                if (last != cudaSuccess) break;
            }
            if (own_staging) cudaFreeHost(h_routing);
            return last;
        }
        // Single-stream fallback (byte-identical to the pre-two-stream loop):
        // stage+run each active slot on the calling stream.
        for (int s = 0; s < slots; ++s) {
            const GlmRoutedSlot &rs = h_routing[s];
            if (rs.eid < 0 || rs.eid >= n_experts) continue;
            ensure_glm_expert_resident(db, layer, rs.eid, stream);
            const size_t flat = static_cast<size_t>(layer) *
                                    static_cast<size_t>(db.experts_per_layer) +
                                static_cast<size_t>(rs.eid);
            if (flat >= db.entries.size()) continue;
            const GlmExpertEntry &en = db.entries[flat];
            if (!en.host_w1 || !en.host_w2 || !en.host_w3 ||
                !en.host_w1_scales || !en.host_w2_scales || !en.host_w3_scales ||
                en.w1_bytes != kPackPerMat || en.w2_bytes != kPackPerMat ||
                en.w3_bytes != kPackPerMat ||
                en.w1_scales_bytes != kScalePerMat ||
                en.w2_scales_bytes != kScalePerMat ||
                en.w3_scales_bytes != kScalePerMat)
                continue; // missing / foreign format: deterministic zero contribution

            last = stage_slot(s, stream);
            if (last != cudaSuccess) break;

            GlmExpertDeviceBlock eb;
            eb.d_w1_packed = sets_p[0]; eb.d_w1_scales = sets_s[0];
            eb.w1_global_scale = en.w1_global_scale;
            eb.w1_packed_bytes = kPackPerMat; eb.w1_scale_bytes = kScalePerMat;
            eb.d_w2_packed = sets_p[0] + kPackPerMat; eb.d_w2_scales = sets_s[0] + kScalePerMat;
            eb.w2_global_scale = en.w2_global_scale;
            eb.w2_packed_bytes = kPackPerMat; eb.w2_scale_bytes = kScalePerMat;
            eb.d_w3_packed = sets_p[0] + 2 * kPackPerMat; eb.d_w3_scales = sets_s[0] + 2 * kScalePerMat;
            eb.w3_global_scale = en.w3_global_scale;
            eb.w3_packed_bytes = kPackPerMat; eb.w3_scale_bytes = kScalePerMat;

            const int   t    = s / kGlmMoETopK;
            const __half *x_t = d_act + static_cast<size_t>(t) * kGlmHidden;
            __half *out_t = d_out + static_cast<size_t>(t) * kGlmHidden;
            last = RunGlmExpertSlot(cublas_h, stream, eb,
                                    x_t, out_t, d_scratch, rs.coeff);
            if (last != cudaSuccess) break;
        }
        if (own_staging) cudaFreeHost(h_routing);
        return last;
    }

    // Two-stream pipeline. Slot 0 staged on the compute stream so its GEMM
    // needs no cross-stream wait. For s>0 the copy stream staged slot s during
    // iteration s-1 and recorded ev_h2d[s&1]; the compute stream waits it before
    // the GEMM. The copy producer for slot s+1 waits ev_gemm[(s+1)&1] (the GEMM
    // that last read that set) before overwriting.
    //
    // Preflight residency: parse every routed expert BEFORE any staging.
    // Otherwise stage_slot silently skips non-resident entries (host_w1 ==
    // nullptr) and the GEMM runs on stale staging buffers; and an LRU eviction
    // mid-loop could munmap a host mmap whose H2D copy (slot 0 runs on the
    // compute stream) is still in flight. With all targets resident up front,
    // no eviction can occur between staging and GEMM. The first preflight
    // EnsureResident that evicts synchronizes stream_copy, draining any prior
    // in-flight copies owned by this db.
    for (int s = 0; s < slots; ++s)
        ensure_glm_expert_resident(db, layer, h_routing[s].eid, stream_copy);

    // Two-stream cached pipeline: each active slot resolves through the
    // device-slot LRU. Hits run straight from their slot (no H2D, no wait);
    // misses claim a slot and copy on stream_copy, recording ev_h2d[s&1]
    // which the compute stream waits before that slot's GEMM. ev_cache_gemm
    // (recorded after every cached GEMM) is waited by stream_copy before an
    // insert overwrites an evicted victim slot — the victim's last reader is
    // always some earlier GEMM on the compute stream, and events on one
    // stream are ordered, so waiting for the latest one drains all of them.
    if (use_cache) {
        bool gemm_recorded = false;
        for (int s = 0; s < slots; ++s) {
            const int flat = cache_flat(h_routing[s]);
            if (flat < 0) continue;
            const GlmExpertEntry &en = db.entries[static_cast<size_t>(flat)];
            int cslot = CacheLookup(*xcache, flat);
            const bool hit = cslot >= 0;
            if (!hit) {
                cslot = CacheInsert(*xcache, flat);
                if (gemm_recorded)
                    cudaStreamWaitEvent(stream_copy, ev_cache_gemm, 0);
                last = copy_into_cache_slot(cslot, en, stream_copy);
                if (last != cudaSuccess) break;
                cudaEventRecord(ev_h2d[s & 1], stream_copy);
                cudaStreamWaitEvent(stream, ev_h2d[s & 1], 0);
            }
            GlmExpertDeviceBlock eb;
            eb_from_cache_slot(cslot, en, eb);
            const int   t    = s / kGlmMoETopK;
            const __half *x_t = d_act + static_cast<size_t>(t) * kGlmHidden;
            __half *out_t = d_out + static_cast<size_t>(t) * kGlmHidden;
            last = RunGlmExpertSlot(cublas_h, stream, eb,
                                    x_t, out_t, d_scratch,
                                    h_routing[s].coeff);
            if (last != cudaSuccess) break;
            cudaEventRecord(ev_cache_gemm, stream);
            gemm_recorded = true;
        }
        if (own_staging) cudaFreeHost(h_routing);
        return last;
    }

    cudaError_t ef = stage_slot(0, stream);
    if (ef != cudaSuccess) { if (own_staging) cudaFreeHost(h_routing); return ef; }

    bool set_seen[2] = {false, false};
    for (int s = 0; s < slots; ++s) {
        const GlmRoutedSlot &rs = h_routing[s];
        if (rs.eid < 0 || rs.eid >= n_experts) continue;
        ensure_glm_expert_resident(db, layer, rs.eid, stream_copy);
        const int this_set = (s & 1);

        if (s > 0)
            cudaStreamWaitEvent(stream, ev_h2d[this_set], 0);

        if (s + 1 < slots) {
            const int nxt = (s + 1) & 1;
            // First write of a set has no prior GEMM to wait on (the set was
            // never read); skip the wait until the set has been used once.
            if (set_seen[nxt])
                cudaStreamWaitEvent(stream_copy, ev_gemm[nxt], 0);
            cudaError_t pe = stage_slot(s + 1, stream_copy);
            if (pe != cudaSuccess) { last = pe; break; }
            cudaEventRecord(ev_h2d[nxt], stream_copy);
        }

        const size_t flat = static_cast<size_t>(layer) *
                                static_cast<size_t>(db.experts_per_layer) +
                            static_cast<size_t>(rs.eid);
        if (flat >= db.entries.size()) continue;
        const GlmExpertEntry &en = db.entries[flat];
        if (!en.host_w1 || !en.host_w2 || !en.host_w3 ||
            !en.host_w1_scales || !en.host_w2_scales || !en.host_w3_scales ||
            en.w1_bytes != kPackPerMat || en.w2_bytes != kPackPerMat ||
            en.w3_bytes != kPackPerMat ||
            en.w1_scales_bytes != kScalePerMat ||
            en.w2_scales_bytes != kScalePerMat ||
            en.w3_scales_bytes != kScalePerMat)
            continue;

        GlmExpertDeviceBlock eb;
        eb.d_w1_packed = sets_p[this_set];  eb.d_w1_scales = sets_s[this_set];
        eb.w1_global_scale = en.w1_global_scale;
        eb.w1_packed_bytes = kPackPerMat; eb.w1_scale_bytes = kScalePerMat;
        eb.d_w2_packed = sets_p[this_set] + kPackPerMat; eb.d_w2_scales = sets_s[this_set] + kScalePerMat;
        eb.w2_global_scale = en.w2_global_scale;
        eb.w2_packed_bytes = kPackPerMat; eb.w2_scale_bytes = kScalePerMat;
        eb.d_w3_packed = sets_p[this_set] + 2 * kPackPerMat; eb.d_w3_scales = sets_s[this_set] + 2 * kScalePerMat;
        eb.w3_global_scale = en.w3_global_scale;
        eb.w3_packed_bytes = kPackPerMat; eb.w3_scale_bytes = kScalePerMat;

        const int t = s / kGlmMoETopK;
        const __half *x_t = d_act + static_cast<size_t>(t) * kGlmHidden;
        __half *out_t = d_out + static_cast<size_t>(t) * kGlmHidden;
        last = RunGlmExpertSlot(cublas_h, stream, eb, x_t, out_t,
                                d_scratch, rs.coeff);
        if (last != cudaSuccess) break;
        cudaEventRecord(ev_gemm[this_set], stream);
        set_seen[this_set] = true;
    }

    if (own_staging) cudaFreeHost(h_routing);
    return last;
}

// ── Cycle 9: MoE virtual-slice kernels (layers 3-77) ───────────────────────
// out[i] = src[i]: identity pass-through for a MoE layer that carries no
// router weights (heterogeneous grid fallback).
__global__ void GlmMoECopyKernel(__half *__restrict__ dst,
                                 const __half *__restrict__ src, int64_t n)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// out[i] = fp16(moe[i] + base[i]): MoE MLP residual. base is the layer input
// (pre-norm activations), moe the expert stack result — the MoE branch always
// residual-adds the un-normed layer input, mirroring the dense stack.
__global__ void GlmMoEResidualKernel(__half *__restrict__ out,
                                     const __half *__restrict__ moe,
                                     const __half *__restrict__ base,
                                     int64_t n)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
    if (i < n)
        out[i] = __float2half(__half2float(moe[i]) + __half2float(base[i]));
}

// Folds the per-(token, expert) virtual activation slices into the token
// output: out[t][h] = H(Σ_r routing[t][r].coeff · slices[(t·n_experts +
// routing[t][r].eid)·hidden + h]). Only the kGlmMoETopK selected slices are
// read; a slot whose eid falls outside [0, n_experts) contributes nothing
// (the slot writer already skipped it). FP32 accumulate, FP16 store.
__global__ void GlmMoESliceAccumKernel(
    const __half *__restrict__ slices,
    const GlmRoutedSlot *__restrict__ routing,
    __half *__restrict__ out, int seq, int n_experts, int hidden)
{
    const int64_t total = static_cast<int64_t>(seq) * hidden;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
    if (idx >= total) return;
    const int t = static_cast<int>(idx / hidden);
    const int h = static_cast<int>(idx - static_cast<int64_t>(t) * hidden);
    const GlmRoutedSlot *row = routing + static_cast<size_t>(t) * kGlmMoETopK;
    float acc = 0.0f;
    for (int r = 0; r < kGlmMoETopK; ++r) {
        const GlmRoutedSlot &rs = row[r];
        if (rs.eid < 0 || rs.eid >= n_experts) continue;
        const __half *slice =
            slices + (static_cast<int64_t>(t) * n_experts + rs.eid) * hidden;
        acc += rs.coeff * __half2float(slice[h]);
    }
    out[idx] = __float2half(acc);
}

// ── Cycle 9: MoE arena + virtual-slice stack (layers 3-77) ─────────────────
cudaError_t GlmMoEArena::Allocate(int seq_, int hidden_, int n_experts_)
{
    Free();
    constexpr int64_t kRouted = kGlmNumRouted;
    if (seq_ <= 0 || hidden_ <= 0 || n_experts_ <= 0)
        return cudaErrorInvalidValue;
    const int64_t n_seq_r = static_cast<int64_t>(seq_) * kRouted;
    const int64_t route_n = static_cast<int64_t>(seq_) * kGlmMoETopK;
    const int64_t hid_n   = static_cast<int64_t>(seq_) * hidden_;
    const int64_t slice_n = n_seq_r * hidden_;
    const int64_t scratch_n = kGlmExpertMatEl + hidden_ +
                              2 * static_cast<int64_t>(kGlmMoeIntermediate);

    int64_t off = 0;
    auto page = [&](int64_t bytes, int64_t align) -> int64_t {
        off = (off + align - 1) / align * align;
        const int64_t at = off;
        off += bytes;
        return at;
    };
    const int64_t o_norm    = page(hid_n * sizeof(__half), 16);
    const int64_t o_logits  = page(n_seq_r * sizeof(float), 16);
    const int64_t o_route   = page(route_n * sizeof(GlmRoutedSlot), 16);
    const int64_t o_slices  = page(slice_n * sizeof(__half), 16);
    const int64_t o_moe     = page(hid_n * sizeof(__half), 16);
    const int64_t o_scratch = page(scratch_n * sizeof(__half), 16);
    const int64_t o_bias    = page(kRouted * sizeof(float), 16);
    const int64_t o_guard   = page(1024, 16);   // sentinel tail for overrun checks
    bytes = static_cast<size_t>(off);

    uint8_t *base = nullptr;
    cudaError_t e = cudaMalloc(reinterpret_cast<void **>(&base), bytes);
    if (e != cudaSuccess) {
        bytes = 0;
        return e;
    }
    d_base      = base;
    d_normed    = reinterpret_cast<__half *>(base + o_norm);
    d_logits    = reinterpret_cast<float *>(base + o_logits);
    d_routing   = reinterpret_cast<GlmRoutedSlot *>(base + o_route);
    d_slices    = reinterpret_cast<__half *>(base + o_slices);
    d_moe       = reinterpret_cast<__half *>(base + o_moe);
    d_scratch   = reinterpret_cast<__half *>(base + o_scratch);
    d_gate_bias = reinterpret_cast<float *>(base + o_bias);

    e = cudaMallocHost(&h_routing_staging,
                       static_cast<size_t>(route_n) *
                           sizeof(GlmRoutedSlot));
    if (e != cudaSuccess) {
        Free();
        return e;
    }
    seq = seq_;
    hidden = hidden_;
    n_experts = n_experts_;
    return cudaSuccess;
}

// TDD Cycle 10: extension of the arena to the FULL 79-layer loop footprint.
// One contiguous device allocation carries the IndexShare K-V cache slots
// (GlmIndexShareState), the dense/MoE + attention activation workspaces and the
// Layer-78 MTP execution spaces (eh_proj output, enorm/hmtp, h78, shared-head
// normalized hidden and the [vocab] logits row). A single combined pinned host
// block holds both the routing-grid staging and the argmax slot. The virtual
// per-expert slice grid (d_slices) is intentionally omitted: the generation
// loop never materializes it (it runs the cheaper RunGlmExpertsStack path), so
// a full-capacity prefill cannot balloon this arena by seq*256*hidden halves.
cudaError_t GlmMoEArena::AllocateGeneration(int seq_, int hidden_,
                                            int n_experts_, int vocab_)
{
    Free();
    constexpr int64_t kRouted = kGlmNumRouted;
    if (seq_ <= 0 || hidden_ <= 0 || n_experts_ <= 0 || vocab_ <= 0)
        return cudaErrorInvalidValue;
    const int64_t route_n = static_cast<int64_t>(seq_) * kGlmMoETopK;
    const int64_t logit_n = static_cast<int64_t>(seq_) * kRouted;
    const int64_t hid_n   = static_cast<int64_t>(seq_) * hidden_;
    const int64_t inter_n = static_cast<int64_t>(seq_) * kGlmMoeIntermediate;
    const int64_t scratch_n = kGlmExpertMatEl + hidden_ +
                              2 * static_cast<int64_t>(kGlmMoeIntermediate);

    int64_t off = 0;
    auto page = [&](int64_t bytes_, int64_t align) -> int64_t {
        off = (off + align - 1) / align * align;
        const int64_t at = off;
        off += bytes_;
        return at;
    };
    const int64_t o_state   = page(sizeof(GlmIndexShareState), 16);
    const int64_t o_norm    = page(hid_n * sizeof(__half), 16);
    const int64_t o_logits  = page(logit_n * sizeof(float), 16);
    const int64_t o_route   = page(route_n * sizeof(GlmRoutedSlot), 16);
    const int64_t o_moe     = page(hid_n * sizeof(__half), 16);
    const int64_t o_scratch = page(scratch_n * sizeof(__half), 16);
    const int64_t o_bias    = page(kRouted * sizeof(float), 16);
    const int64_t o_attn    = page(hid_n * sizeof(__half), 16);
    const int64_t o_gate    = page(inter_n * sizeof(__half), 16);
    const int64_t o_up      = page(inter_n * sizeof(__half), 16);
    const int64_t o_cur     = page(hid_n * sizeof(__half), 16);
    const int64_t o_alt     = page(hid_n * sizeof(__half), 16);
    const int64_t o_hidrow  = page(hidden_ * sizeof(__half), 16);
    const int64_t o_mtp     = page(hidden_ * sizeof(__half), 16);
    const int64_t o_hmtp    = page(hidden_ * sizeof(__half), 16);
    const int64_t o_h78     = page(hidden_ * sizeof(__half), 16);
    const int64_t o_hn      = page(hidden_ * sizeof(__half), 16);
    const int64_t o_logits_head = page(int64_t(vocab_) * sizeof(float), 16);
    const int64_t o_argmax  = page(sizeof(GlmArgmaxSlot), 16);
    const int64_t o_ids     = page(int64_t(seq_) * sizeof(int32_t), 16);
    const int64_t o_guard   = page(1024, 16);   // sentinel tail for overrun checks
    bytes = static_cast<size_t>(off);

    uint8_t *base = nullptr;
    cudaError_t e = cudaMalloc(reinterpret_cast<void **>(&base), bytes);
    if (e != cudaSuccess) {
        bytes = 0;
        return e;
    }
    d_base      = base;
    d_state     = reinterpret_cast<GlmIndexShareState *>(base + o_state);
    d_normed    = reinterpret_cast<__half *>(base + o_norm);
    d_logits    = reinterpret_cast<float *>(base + o_logits);
    d_routing   = reinterpret_cast<GlmRoutedSlot *>(base + o_route);
    d_moe       = reinterpret_cast<__half *>(base + o_moe);
    d_scratch   = reinterpret_cast<__half *>(base + o_scratch);
    d_gate_bias = reinterpret_cast<float *>(base + o_bias);
    d_attn      = reinterpret_cast<__half *>(base + o_attn);
    d_gate      = reinterpret_cast<__half *>(base + o_gate);
    d_up        = reinterpret_cast<__half *>(base + o_up);
    d_cur       = reinterpret_cast<__half *>(base + o_cur);
    d_alt       = reinterpret_cast<__half *>(base + o_alt);
    d_hidrow    = reinterpret_cast<__half *>(base + o_hidrow);
    d_mtp       = reinterpret_cast<__half *>(base + o_mtp);
    d_hmtp      = reinterpret_cast<__half *>(base + o_hmtp);
    d_h78       = reinterpret_cast<__half *>(base + o_h78);
    d_hn        = reinterpret_cast<__half *>(base + o_hn);
    d_logits_head = reinterpret_cast<float *>(base + o_logits_head);
    d_argmax    = reinterpret_cast<GlmArgmaxSlot *>(base + o_argmax);
    d_ids       = reinterpret_cast<int32_t *>(base + o_ids);

    // Single combined pinned host staging block.
    const int64_t h_route_bytes = route_n * sizeof(GlmRoutedSlot);
    const int64_t h_argmax_bytes = sizeof(GlmArgmaxSlot);
    uint8_t *hbase = nullptr;
    e = cudaMallocHost(reinterpret_cast<void **>(&hbase),
                       static_cast<size_t>(h_route_bytes + h_argmax_bytes));
    if (e != cudaSuccess) {
        Free();
        return e;
    }
    h_routing_staging = reinterpret_cast<GlmRoutedSlot *>(hbase);
    h_argmax = reinterpret_cast<GlmArgmaxSlot *>(hbase + h_route_bytes);

    seq = seq_;
    hidden = hidden_;
    n_experts = n_experts_;
    vocab = vocab_;
    return cudaSuccess;
}

void GlmMoEArena::Free()
{
    if (h_routing_staging) cudaFreeHost(h_routing_staging);
    if (d_base) cudaFree(d_base);
    h_routing_staging = nullptr;
    d_base = nullptr;
    d_normed = nullptr; d_logits = nullptr; d_routing = nullptr;
    d_slices = nullptr; d_moe = nullptr; d_scratch = nullptr;
    d_gate_bias = nullptr;
    d_state = nullptr;
    d_attn = nullptr; d_gate = nullptr; d_up = nullptr;
    d_cur = nullptr; d_alt = nullptr;
    d_hidrow = nullptr; d_mtp = nullptr; d_hmtp = nullptr;
    d_h78 = nullptr; d_hn = nullptr; d_logits_head = nullptr;
    d_argmax = nullptr; d_ids = nullptr; h_argmax = nullptr;
    seq = 0; hidden = 0; n_experts = 0; vocab = 0; bytes = 0;
}

cudaError_t RunGlmMoESlices(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const GlmRoutedSlot *d_routing, int seq,
    const __half *d_act,
    const GlmExpertDeviceBlock *h_experts, int n_experts,
    __half *d_slices, __half *d_out, __half *d_scratch,
    GlmRoutedSlot *h_routing_staging)
{
    constexpr int64_t kMatEl = kGlmExpertMatEl;
    constexpr int64_t kExpectPacked = kMatEl / 2;
    constexpr int64_t kExpectScale  = kMatEl / kGlmNvfp4Block;
    constexpr int    kBlock  = 256;

    if (!d_routing || !d_act || !h_experts || !d_slices || !d_out ||
        !d_scratch || seq <= 0 || n_experts <= 0 || !cublas_h)
        return cudaErrorInvalidValue;

    const int slots = seq * kGlmMoETopK;
    if (slots < 0) return cudaErrorInvalidValue;

    const bool own_staging = (h_routing_staging == nullptr);
    GlmRoutedSlot *h_routing = h_routing_staging;
    cudaError_t e = cudaSuccess;
    if (own_staging) {
        e = cudaMallocHost(&h_routing,
                           static_cast<size_t>(slots) * sizeof(GlmRoutedSlot));
        if (e != cudaSuccess) return e;
    }
    e = cudaMemcpyAsync(h_routing, d_routing,
                        static_cast<size_t>(slots) * sizeof(GlmRoutedSlot),
                        cudaMemcpyDeviceToHost, stream);
    if (e == cudaSuccess) e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess) {
        if (own_staging) cudaFreeHost(h_routing);
        return e;
    }

    // Scratch layout (halves): dequant w-deck / (reserved tmp) / inter / gate.
    __half *d_deq   = d_scratch;                                // kMatEl
    __half *d_inter = d_deq + kMatEl;                           // kGlmMoeIntermediate
    __half *d_gate  = d_inter + kGlmMoeIntermediate;            // kGlmMoeIntermediate

    const int64_t dequant_per_block =
        static_cast<int64_t>(kGlmDequantThreads) * kGlmDequantPerThread;
    const int deq_grid =
        static_cast<int>((kMatEl + dequant_per_block - 1) / dequant_per_block);
    const int inter_blocks = (kGlmMoeIntermediate + kBlock - 1) / kBlock;
    const int hidden_blocks = (kGlmHidden + kBlock - 1) / kBlock;

    // Zero the entire virtual-activation arena BEFORE stamping this layer's
    // expert slices. An inactive/foreign selected expert leaves its slice at 0
    // so the fold below contributes nothing — otherwise the arena would carry
    // STALE slices from a previously active layer and re-apply them here,
    // silently corrupting the identity residual of every inactive layer.
    const int64_t slice_n =
        static_cast<int64_t>(seq) * n_experts * kGlmHidden;
    e = cudaMemsetAsync(d_slices, 0,
                        static_cast<size_t>(slice_n) * sizeof(__half), stream);
    if (e != cudaSuccess) return e;

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
            eb.w1_scale_bytes  != kExpectScale ||
            eb.w2_scale_bytes  != kExpectScale ||
            eb.w3_scale_bytes  != kExpectScale)
            continue; // inactive / foreign format: deterministic zero contribution

        const int t = s / kGlmMoETopK;
        const __half *x_t = d_act + static_cast<size_t>(t) * kGlmHidden;
        __half *slice =
            d_slices + (static_cast<size_t>(t) * n_experts + rs.eid) * kGlmHidden;

        // inter = w1^T @ x
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

        // swiglu: inter = silu(inter) * gate
        GlmRealSwigluKernel<<<inter_blocks, kBlock, 0, stream>>>(
            d_inter, d_gate, kGlmMoeIntermediate);
        e = cudaGetLastError();
        if (e != cudaSuccess) break;

        // down = w2^T @ swig, written DIRECTLY into the (t, eid) slice.
        GlmDequantNvfp4Kernel<<<deq_grid, kGlmDequantThreads, 0, stream>>>(
            eb.d_w2_packed, eb.d_w2_scales, eb.w2_global_scale, d_deq, kMatEl);
        e = cudaGetLastError();
        if (e != cudaSuccess) break;
        GlmGemmFp64Kernel<<<hidden_blocks, kBlock, 0, stream>>>(
            d_deq, d_inter, slice, kGlmHidden, kGlmMoeIntermediate, 1);
        e = cudaGetLastError();
        if (e != cudaSuccess) break;
    }

    if (own_staging) cudaFreeHost(h_routing);
    if (e != cudaSuccess) return e;

    // Folds the co-scaled selected slices into d_out on the same async stream.
    const int64_t out_n = static_cast<int64_t>(seq) * kGlmHidden;
    GlmMoESliceAccumKernel<<<static_cast<int>((out_n + kBlock - 1) / kBlock),
                             kBlock, 0, stream>>>(
        d_slices, d_routing, d_out, seq, n_experts, kGlmHidden);
    return cudaGetLastError();
}

cudaError_t RunGlmMoEStack(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const GlmPinnedWeights &pw, const GlmExpertTables &experts,
    int first_layer, int last_layer, int seq,
    __half *d_act, __half *d_alt, __half *d_result,
    GlmMoEArena &arena)
{
    if (!cublas_h || !stream || seq <= 0 || first_layer < 0 ||
        last_layer <= first_layer ||
        pw.layers.size() < static_cast<size_t>(last_layer) ||
        experts.tables.size() < static_cast<size_t>(last_layer) ||
        !d_act || !d_alt || !d_result ||
        arena.d_slices == nullptr || arena.seq < seq ||
        arena.hidden != kGlmHidden || arena.n_experts < kGlmNumRouted)
        return cudaErrorInvalidValue;

    __half *cur = d_act;
    const int out_blocks = static_cast<int>(
        (static_cast<int64_t>(seq) * kGlmHidden + kGlmRmsBlock - 1) /
        kGlmRmsBlock);
    for (int l = first_layer; l < last_layer; ++l) {
        const GlmLayerWeights &lw = pw.layers[l];
        __half *nxt = (l == last_layer - 1)
                          ? d_result
                          : (cur == d_act ? d_alt : d_act);
        cudaError_t e;
        if (!lw.d_post_attn_norm || !lw.d_router_gate) {
            // Heterogeneous grid: a layer without router weights is identity.
            GlmMoECopyKernel<<<out_blocks, kGlmRmsBlock, 0, stream>>>(
                nxt, cur, static_cast<int64_t>(seq) * kGlmHidden);
            e = cudaGetLastError();
            if (e != cudaSuccess) return e;
            cur = nxt;
            continue;
        }
        const float *d_bias = nullptr;
        if (lw.h_gate_bias && lw.gate_bias_n == kGlmNumRouted) {
            e = cudaMemcpyAsync(arena.d_gate_bias, lw.h_gate_bias,
                                static_cast<size_t>(kGlmNumRouted) *
                                    sizeof(float),
                                cudaMemcpyHostToDevice, stream);
            if (e != cudaSuccess) return e;
            d_bias = arena.d_gate_bias;
        }
        e = RunGlmMoERouter(cublas_h, stream, lw, seq, cur, d_bias,
                            arena.d_normed, arena.d_logits, arena.d_routing);
        if (e != cudaSuccess) return e;
        e = RunGlmMoESlices(cublas_h, stream, arena.d_routing, seq,
                            arena.d_normed, experts.tables[l].data(),
                            arena.n_experts, arena.d_slices, arena.d_moe,
                            arena.d_scratch, arena.h_routing_staging);
        if (e != cudaSuccess) return e;
        // residual: nxt = moe + cur  (cur = un-normed layer input)
        GlmMoEResidualKernel<<<out_blocks, kGlmRmsBlock, 0, stream>>>(
            nxt, arena.d_moe, cur, static_cast<int64_t>(seq) * kGlmHidden);
        e = cudaGetLastError();
        if (e != cudaSuccess) return e;
        cur = nxt;
    }
    return cudaSuccess;
}

// ── Cycle 5: IndexShare attention pass (layers 3-77) ────────────────────────
cudaError_t GlmIndexShareReset(GlmIndexShareState *d_state, cudaStream_t stream)
{
    if (!d_state) return cudaErrorInvalidValue;
    // Explicitly zero the shared state before the init kernel: relying on the
    // fresh cudaMalloc'ed state (or on a previous reuse without Free) left the
    // key/score/slot arrays with stale content on re-entry.
    cudaError_t e = cudaMemsetAsync(d_state, 0, sizeof(GlmIndexShareState),
                                    stream);
    if (e != cudaSuccess) return e;
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
