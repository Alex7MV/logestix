// Granite-4.2-30B dense decoder runtime (Model Ingestion Protocol step 5).
//
// One stream, no per-token cudaDeviceSynchronize. All dense projections go
// through the model-neutral NVFP4 layer (lsx_nvfp4::RunLsxNvfp4Gemm); norms,
// SwiGLU and residual adds use the shared model-neutral lsx_kernels_*.cuh
// primitives. The only family-local kernels are full-dimension RoPE and the
// causal GQA attention (both deterministic, block-reduction based).

#include "lsxcommon/granite_dense_inference.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <vector>

#include "lsxcommon/granite_dense_constants.h"
#include "lsxcommon/log.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/shared_experts.h"

#include "lsx_kernels_act.cuh"
#include "lsx_kernels_common.cuh"
#include "lsx_kernels_math.cuh"
#include "lsx_kernels_norm.cuh"

namespace lsxcommon {
namespace granite {

using lsx_nvfp4::RunLsxNvfp4Gemm;

namespace {

constexpr int kThreads = 128;

// ── Kernels ─────────────────────────────────────────────────────────────────

// Full-dimension RoPE (HF rotate_half pairing): dims i and i+hd/2 form a pair.
// q: [seq, nh, hd], k: [seq, nkv, hd]; positions are absolute (kv_base + t).
__global__ void granite_rope_kernel(__half *__restrict__ q,
                                    __half *__restrict__ k,
                                    const int32_t *__restrict__ pos, int seq,
                                    int nh, int nkv, int hd, double theta)
{
    const int half = hd / 2;
    const int64_t total =
        static_cast<int64_t>(seq) * (nh + nkv) * half;
    const int64_t idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int i = static_cast<int>(idx % half);
    const int64_t rest = idx / half;
    const int head = static_cast<int>(rest % (nh + nkv));
    const int t = static_cast<int>(rest / (nh + nkv));

    const float freq =
        powf(static_cast<float>(theta),
             -2.0f * static_cast<float>(i) / static_cast<float>(hd));
    const float ang = static_cast<float>(pos[t]) * freq;
    const float c = cosf(ang);
    const float s = sinf(ang);

    __half *row = (head < nh)
                      ? q + (static_cast<int64_t>(t) * nh + head) * hd
                      : k + (static_cast<int64_t>(t) * nkv + (head - nh)) * hd;
    const float x1 = __half2float(row[i]);
    const float x2 = __half2float(row[i + half]);
    row[i] = __float2half(__fsub_rn(__fmul_rn(x1, c), __fmul_rn(x2, s)));
    row[i + half] =
        __float2half(__fadd_rn(__fmul_rn(x1, s), __fmul_rn(x2, c)));
}

// Causal GQA attention with a persistent KV cache base. One block per
// (query row, head). Chunked online softmax; block reductions through shared
// memory with __syncthreads() (no atomics, deterministic).
__global__ void granite_attn_kernel(const __half *__restrict__ q,
                                    const __half *__restrict__ k,
                                    const __half *__restrict__ v,
                                    __half *__restrict__ out, int seq, int nh,
                                    int nkv, int hd, float scale, int kv_base)
{
    const int t = blockIdx.x / nh;
    const int h = blockIdx.x % nh;
    if (t >= seq) return;
    const int kvh = h / (nh / nkv);
    const int tid = threadIdx.x;

    extern __shared__ float smem[];
    float *q_s = smem;              // [hd]
    float *o_s = smem + hd;         // [hd]
    float *sc = smem + 2 * hd;      // [blockDim]
    float *rd = sc + blockDim.x;    // [blockDim]

    const __half *qrow = q + (static_cast<int64_t>(t) * nh + h) * hd;
    for (int d = tid; d < hd; d += blockDim.x)
        q_s[d] = __half2float(qrow[d]);
    for (int d = tid; d < hd; d += blockDim.x) o_s[d] = 0.f;
    __syncthreads();

    const int bound = kv_base + t;  // inclusive last attended key
    float m = -CUDART_INF_F;
    float l = 0.f;
    for (int start = 0; start <= bound; start += blockDim.x) {
        const int cnt = min(static_cast<int>(blockDim.x), bound + 1 - start);
        float score = -CUDART_INF_F;
        if (tid < cnt) {
            const __half *krow =
                k + (static_cast<int64_t>(start + tid) * nkv + kvh) * hd;
            float dot = 0.f;
            for (int d = 0; d < hd; ++d)
                dot = __fmaf_rn(q_s[d], __half2float(krow[d]), dot);
            score = __fmul_rn(dot, scale);
        }
        sc[tid] = score;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s && sc[tid] < sc[tid + s]) sc[tid] = sc[tid + s];
            __syncthreads();
        }
        const float m_new = sc[0];
        __syncthreads();
        const float e = (tid < cnt) ? __expf(__fsub_rn(score, m_new)) : 0.f;
        sc[tid] = e;
        rd[tid] = e;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) rd[tid] = __fadd_rn(rd[tid], rd[tid + s]);
            __syncthreads();
        }
        const float l_new = rd[0];
        __syncthreads();
        const float rescale =
            (m == -CUDART_INF_F) ? 0.f : __expf(__fsub_rn(m, m_new));
        for (int d = tid; d < hd; d += blockDim.x) {
            float acc = __fmul_rn(o_s[d], rescale);
            for (int j = 0; j < cnt; ++j) {
                const __half *vrow =
                    v + (static_cast<int64_t>(start + j) * nkv + kvh) * hd;
                acc = __fmaf_rn(sc[j], __half2float(vrow[d]), acc);
            }
            o_s[d] = acc;
        }
        m = m_new;
        l = __fmaf_rn(l, rescale, l_new);
        __syncthreads();
    }

    const float inv = (l > 0.f) ? __frcp_rn(l) : 0.f;
    __half *orow = out + (static_cast<int64_t>(t) * nh + h) * hd;
    for (int d = tid; d < hd; d += blockDim.x)
        orow[d] = __float2half(__fmul_rn(o_s[d], inv));
}

// ── Split-KV (flash-decoding) attention ─────────────────────────────────────
// The kernel above launches seq*nh blocks; at decode (seq==1) that is only nh
// blocks, so ~80% of the SM array is idle and the kernel sits ~60x above its
// KV-read memory bound (nsys 2026-09-22: 26.5% of decode wall, ctx 2048).
// This variant partitions the causal key range [0, kv_base+t] into `splits`
// contiguous chunks, computes an online-softmax partial (m, l, o) per
// (row, head, split), and a second kernel merges the partials in fixed split
// order. Fixed boundaries + fixed merge order => run-to-run deterministic.
// Splits never straddle: n0 = total*sv/splits, n1 = total*(sv+1)/splits.
// `splits` is chosen at dispatch so each split owns ~kAttnSplitKeys keys (a
// fixed 8 left the SM array at 12% occupancy on long contexts); the partial
// slab is sized for kAttnSplitMax.
constexpr int kAttnSplitMax = 32;
constexpr int kAttnSplitKeys = 64;

__global__ void granite_attn_split_kernel(const __half *__restrict__ q,
                                          const __half *__restrict__ k,
                                          const __half *__restrict__ v,
                                          float *__restrict__ pm,
                                          float *__restrict__ pl,
                                          float *__restrict__ po, int seq, int nh,
                                          int nkv, int hd, float scale,
                                          int kv_base, int splits)
{
    const int t = blockIdx.x / nh;
    const int h = blockIdx.x % nh;
    const int sv = blockIdx.y;
    if (t >= seq) return;
    const int kvh = h / (nh / nkv);
    const int tid = threadIdx.x;

    extern __shared__ float smem[];
    float *q_s = smem;              // [hd]
    float *o_s = smem + hd;         // [hd]
    float *sc = smem + 2 * hd;      // [blockDim]
    float *rd = sc + blockDim.x;    // [blockDim]

    const int bound = kv_base + t;  // inclusive last attended key
    const int total = bound + 1;
    const int n0 = static_cast<int>(static_cast<int64_t>(total) * sv / splits);
    const int n1 =
        static_cast<int>(static_cast<int64_t>(total) * (sv + 1) / splits);
    const int64_t slot = (static_cast<int64_t>(sv) * seq + t) * nh + h;

    const __half *qrow = q + (static_cast<int64_t>(t) * nh + h) * hd;
    for (int d = tid; d < hd; d += blockDim.x)
        q_s[d] = __half2float(qrow[d]);
    for (int d = tid; d < hd; d += blockDim.x) o_s[d] = 0.f;
    __syncthreads();

    float m = -CUDART_INF_F;
    float l = 0.f;
    for (int start = n0; start < n1; start += blockDim.x) {
        const int cnt = min(static_cast<int>(blockDim.x), n1 - start);
        float score = -CUDART_INF_F;
        if (tid < cnt) {
            const __half *krow =
                k + (static_cast<int64_t>(start + tid) * nkv + kvh) * hd;
            // half2 loads + two accumulator chains: the single 128-deep FMA
            // chain was the QK critical path (2x shorter chain, half the
            // load instructions).
            const __half2 *k2 = reinterpret_cast<const __half2 *>(krow);
            float d0 = 0.f, d1 = 0.f;
            for (int d2 = 0; d2 < (hd >> 1); ++d2) {
                const float2 kf = __half22float2(k2[d2]);
                d0 = __fmaf_rn(q_s[2 * d2], kf.x, d0);
                d1 = __fmaf_rn(q_s[2 * d2 + 1], kf.y, d1);
            }
            score = __fmul_rn(__fadd_rn(d0, d1), scale);
        }
        sc[tid] = score;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s && sc[tid] < sc[tid + s]) sc[tid] = sc[tid + s];
            __syncthreads();
        }
        const float m_new = sc[0];
        __syncthreads();
        const float e = (tid < cnt) ? __expf(__fsub_rn(score, m_new)) : 0.f;
        sc[tid] = e;
        rd[tid] = e;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) rd[tid] = __fadd_rn(rd[tid], rd[tid + s]);
            __syncthreads();
        }
        const float l_new = rd[0];
        __syncthreads();
        const float rescale =
            (m == -CUDART_INF_F) ? 0.f : __expf(__fsub_rn(m, m_new));
        for (int d = tid; d < hd; d += blockDim.x) {
            float acc = __fmul_rn(o_s[d], rescale);
            for (int j = 0; j < cnt; ++j) {
                const __half *vrow =
                    v + (static_cast<int64_t>(start + j) * nkv + kvh) * hd;
                acc = __fmaf_rn(sc[j], __half2float(vrow[d]), acc);
            }
            o_s[d] = acc;
        }
        m = m_new;
        l = __fmaf_rn(l, rescale, l_new);
        __syncthreads();
    }

    if (tid == 0) {
        pm[slot] = m;
        pl[slot] = l;
    }
    for (int d = tid; d < hd; d += blockDim.x) po[slot * hd + d] = o_s[d];
}

// Merges the `splits` online-softmax partials per (row, head) in fixed split
// order. m is a max (order-free); l/o accumulate in split index order.
__global__ void granite_attn_combine_kernel(const float *__restrict__ pm,
                                            const float *__restrict__ pl,
                                            const float *__restrict__ po,
                                            __half *__restrict__ out, int seq,
                                            int nh, int hd, int splits)
{
    const int t = blockIdx.x / nh;
    const int h = blockIdx.x % nh;
    if (t >= seq) return;
    const int tid = threadIdx.x;
    const int64_t base = static_cast<int64_t>(t) * nh + h;

    float m = -CUDART_INF_F;
    for (int s = 0; s < splits; ++s) {
        const float ms = pm[static_cast<int64_t>(s) * seq * nh + base];
        m = fmaxf(m, ms);
    }
    float l = 0.f;
    for (int s = 0; s < splits; ++s) {
        const int64_t slot = static_cast<int64_t>(s) * seq * nh + base;
        l += pl[slot] * __expf(__fsub_rn(pm[slot], m));
    }
    __half *orow = out + base * hd;
    for (int d = tid; d < hd; d += blockDim.x) {
        float a = 0.f;
        for (int s = 0; s < splits; ++s) {
            const int64_t slot = static_cast<int64_t>(s) * seq * nh + base;
            a += po[slot * hd + d] * __expf(__fsub_rn(pm[slot], m));
        }
        orow[d] = __float2half((l > 0.f) ? __fdiv_rn(a, l) : 0.f);
    }
}

// SwiGLU over the fused [seq, 2I] gate_up buffer (rows [0,I)=gate, [I,2I)=up).
__global__ void granite_swiglu_fused_kernel(const __half *__restrict__ gu,
                                            __half *__restrict__ out, int I,
                                            int64_t n)
{
    const int64_t i =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t t = i / I;
    const int j = static_cast<int>(i - t * I);
    const float g = __half2float(gu[t * 2 * I + j]);
    const float u = __half2float(gu[t * 2 * I + I + j]);
    const float s = __fdiv_rn(g, __fadd_rn(1.0f, __expf(-g)));
    out[i] = __float2half(__fmul_rn(s, u));
}

// Splits the fused [seq, NH*HD + 2*NKV*HD] q/k/v projection output into the
// contiguous per-tensor buffers RoPE/attention consume.
__global__ void granite_split_qkv_kernel(const __half *__restrict__ qkv,
                                         __half *__restrict__ q,
                                         __half *__restrict__ k,
                                         __half *__restrict__ v, int seq,
                                         int qn, int kn)
{
    const int row = qn + 2 * kn;
    const int64_t i =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<int64_t>(seq) * row) return;
    const int t = static_cast<int>(i / row);
    const int j = static_cast<int>(i - static_cast<int64_t>(t) * row);
    if (j < qn)
        q[static_cast<int64_t>(t) * qn + j] = qkv[i];
    else if (j < qn + kn)
        k[static_cast<int64_t>(t) * kn + (j - qn)] = qkv[i];
    else
        v[static_cast<int64_t>(t) * kn + (j - qn - kn)] = qkv[i];
}

// ── Launchers ───────────────────────────────────────────────────────────────

// Decode RMSNorm is one row per launch (seq==1), so the shared block-tree
// kernel is pure latency: nsys (2026-09-22) put it at 16.7 us/call and 14.3%
// of the whole decode wall, and an isolated sweep measured 12.2 us for
// RmsNormTreeKernel<128> (its per-thread strided sum plus a 7-level
// shared-memory tree with a __syncthreads per level). This kernel loads
// half2, reduces with warp shuffles and a single cross-warp round (2
// __syncthreads total) and re-reads the L2-resident row for the scale pass:
// measured 4.0 us (128/256/512 threads all equal -- the remainder is launch
// latency). Numeric class: FP32 accumulate, fixed warp-tree order (deterministic).
__global__ void granite_rmsnorm_kernel(const __half *__restrict__ x,
                                       const __half *__restrict__ w,
                                       __half *__restrict__ out, int hidden,
                                       float eps)
{
    const int row = blockIdx.x;
    const int tpb = blockDim.x;
    const int n2 = hidden >> 1;
    const __half *xr = x + static_cast<int64_t>(row) * hidden;
    __half *orow = out + static_cast<int64_t>(row) * hidden;
    const __half2 *x2 = reinterpret_cast<const __half2 *>(xr);
    const __half2 *w2 = reinterpret_cast<const __half2 *>(w);
    __half2 *o2 = reinterpret_cast<__half2 *>(orow);

    float sum = 0.f;
    for (int j = threadIdx.x; j < n2; j += tpb) {
        const float2 f = __half22float2(x2[j]);
        sum = __fmaf_rn(f.x, f.x, __fmaf_rn(f.y, f.y, sum));
    }
    for (int o = 16; o > 0; o >>= 1)
        sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffu, sum, o));
    __shared__ float ws[32];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (lane == 0) ws[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < (tpb >> 5)) ? ws[lane] : 0.f;
        for (int o = 16; o > 0; o >>= 1)
            v = __fadd_rn(v, __shfl_down_sync(0xffffffffu, v, o));
        if (lane == 0) ws[0] = v;
    }
    __syncthreads();
    const float inv =
        rsqrtf(__fadd_rn(__fdiv_rn(ws[0], static_cast<float>(hidden)), eps));
    for (int j = threadIdx.x; j < n2; j += tpb) {
        const float2 f = __half22float2(x2[j]);
        const float2 g = __half22float2(w2[j]);
        o2[j] = __floats2half2_rn(__fmul_rn(__fmul_rn(f.x, inv), g.x),
                                  __fmul_rn(__fmul_rn(f.y, inv), g.y));
    }
}

cudaError_t GraniteRmsNorm(cudaStream_t stream, __half *out, const __half *x,
                           const __half *w, int rows, int hidden, float eps)
{
    // The vectorized kernel needs an even hidden (half2) and 4-byte-aligned
    // rows; fall back to the shared tree kernel otherwise.
    const bool vec_ok = (hidden % 2) == 0 && hidden >= 64;
    if (vec_ok) {
        constexpr int kNormThreads = 256;
        granite_rmsnorm_kernel<<<rows, kNormThreads, 0, stream>>>(x, w, out,
                                                                  hidden, eps);
        return cudaGetLastError();
    }
    const size_t shmem =
        kThreads * sizeof(float) + static_cast<size_t>(hidden) * sizeof(__half);
    if (shmem > 48 * 1024) return cudaErrorInvalidValue;
    kernels::RmsNormTreeKernel<__half, kernels::NormWeight::Direct>
        <<<rows, kThreads, shmem, stream>>>(x, w, out, rows, hidden, eps);
    return cudaGetLastError();
}

cudaError_t GraniteRope(cudaStream_t stream, __half *q, __half *k,
                        const int32_t *pos, int seq, int nh, int nkv, int hd,
                        double theta)
{
    const int64_t total = static_cast<int64_t>(seq) * (nh + nkv) * (hd / 2);
    const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
    granite_rope_kernel<<<blocks, kThreads, 0, stream>>>(
        q, k, pos, seq, nh, nkv, hd, theta);
    return cudaGetLastError();
}

// Split dispatch floors: the split path only pays off when the key range is
// long enough to fill the SM array (min total = kv_base+1) and seq is small
// enough that the caller pre-allocated the [kAttnSplitMaxSeq, splits] partial
// slab (long-seq prefill already has seq*nh blocks of its own).
constexpr int kAttnSplitMaxSeq = 8;
constexpr int kAttnSplitMinKv = 256;

cudaError_t GraniteAttentionSplit(cudaStream_t stream, const __half *q,
                                  const __half *k, const __half *v, __half *out,
                                  float *partials, int seq, int nh, int nkv,
                                  int hd, float scale, int kv_base, int splits)
{
    if (splits < 1) splits = 1;
    if (splits > kAttnSplitMax) splits = kAttnSplitMax;
    const size_t shmem =
        (2 * static_cast<size_t>(hd) + 2 * kThreads) * sizeof(float);
    const int64_t slots = static_cast<int64_t>(splits) * seq * nh;
    float *pm = partials;
    float *pl = pm + slots;
    float *po = pl + slots;
    const dim3 grid(seq * nh, splits);
    granite_attn_split_kernel<<<grid, kThreads, shmem, stream>>>(
        q, k, v, pm, pl, po, seq, nh, nkv, hd, scale, kv_base, splits);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    granite_attn_combine_kernel<<<seq * nh, kThreads, 0, stream>>>(
        pm, pl, po, out, seq, nh, hd, splits);
    return cudaGetLastError();
}

cudaError_t GraniteAttention(cudaStream_t stream, const __half *q,
                             const __half *k, const __half *v, __half *out,
                             float *partials, int seq, int nh, int nkv, int hd,
                             float scale, int kv_base)
{
    if (partials != nullptr && seq <= kAttnSplitMaxSeq &&
        kv_base + 1 >= kAttnSplitMinKv) {
        // One split per ~kAttnSplitKeys keys, capped by the partial slab.
        int splits = (kv_base + 1 + kAttnSplitKeys - 1) / kAttnSplitKeys;
        return GraniteAttentionSplit(stream, q, k, v, out, partials, seq, nh,
                                     nkv, hd, scale, kv_base, splits);
    }
    const size_t shmem =
        (2 * static_cast<size_t>(hd) + 2 * kThreads) * sizeof(float);
    granite_attn_kernel<<<seq * nh, kThreads, shmem, stream>>>(
        q, k, v, out, seq, nh, nkv, hd, scale, kv_base);
    return cudaGetLastError();
}

cudaError_t GraniteSplitQKV(cudaStream_t stream, const __half *qkv, __half *q,
                            __half *k, __half *v, int seq, int qn, int kn)
{
    const int64_t total = static_cast<int64_t>(seq) * (qn + 2 * kn);
    const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
    granite_split_qkv_kernel<<<blocks, kThreads, 0, stream>>>(qkv, q, k, v, seq,
                                                              qn, kn);
    return cudaGetLastError();
}

cudaError_t GraniteSwiGLU(cudaStream_t stream, const __half *gu, __half *out,
                          int I, int64_t n)
{
    const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
    granite_swiglu_fused_kernel<<<blocks, kThreads, 0, stream>>>(gu, out, I, n);
    return cudaGetLastError();
}

// ── Workspace ───────────────────────────────────────────────────────────────

struct Workspace {
    __half *x = nullptr;
    __half *norm = nullptr;
    __half *q = nullptr;
    __half *k = nullptr;
    __half *v = nullptr;
    __half *qkv = nullptr;
    __half *attn = nullptr;
    __half *proj = nullptr;
    __half *gate_up = nullptr;
    __half *act = nullptr;
    __half *down = nullptr;
    __half *logits = nullptr;
    __half *embed_h = nullptr;
    int32_t *ids = nullptr;
    int32_t *pos = nullptr;
    int32_t *tok = nullptr;
    float *maxv = nullptr;
    float *attn_part = nullptr;   // split-KV softmax partials [splits,seq,nh,hd+2]
    int max_seq = 0;

    cudaError_t Alloc(const GraniteTopology &top, int max_seq_);
    void Free();
};

void Workspace::Free()
{
    cudaFree(x); cudaFree(norm); cudaFree(q); cudaFree(k); cudaFree(v);
    cudaFree(attn); cudaFree(proj); cudaFree(gate_up); cudaFree(act);
    cudaFree(qkv);
    cudaFree(down); cudaFree(logits); cudaFree(embed_h);
    cudaFree(ids); cudaFree(pos); cudaFree(tok); cudaFree(maxv);
    cudaFree(attn_part);
    *this = Workspace{};
}

cudaError_t Workspace::Alloc(const GraniteTopology &top, int max_seq_)
{
    max_seq = max_seq_;
    const int H = top.hidden_size;
    const int I = top.intermediate_size;
    const int V = top.vocab_size;
    const int NH = top.num_attention_heads;
    const int NKV = top.num_key_value_heads;
    const int HD = top.head_dim;
    const int64_t S = max_seq;

    auto alloc = [](void **p, int64_t bytes) -> cudaError_t {
        return cudaMalloc(p, static_cast<size_t>(bytes));
    };
    cudaError_t e = cudaSuccess;
    if ((e = alloc(reinterpret_cast<void **>(&x), S * H * 2))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&norm), S * H * 2))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&q), S * NH * HD * 2))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&k), S * NKV * HD * 2))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&v), S * NKV * HD * 2))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&attn), S * NH * HD * 2))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&qkv),
                   S * (NH + 2 * NKV) * HD * 2))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&proj), S * H * 2))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&gate_up), S * 2 * I * 2))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&act), S * I * 2))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&down), S * H * 2))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&logits),
                   static_cast<int64_t>(V) * 2))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&embed_h),
                   static_cast<int64_t>(V) * H * 2))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&ids), S * sizeof(int32_t)))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&pos), S * sizeof(int32_t)))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&tok), sizeof(int32_t)))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&maxv), sizeof(float)))) return e;
    // Split-KV partials only ever hold the short-seq (decode) regime; long-seq
    // prefill uses the unsplit kernel (seq*nh blocks are plentiful there).
    const int64_t sp_seq = std::min<int64_t>(S, kAttnSplitMaxSeq);
    if ((e = alloc(reinterpret_cast<void **>(&attn_part),
                   static_cast<int64_t>(kAttnSplitMax) * sp_seq * NH * (HD + 2) *
                       sizeof(float))))
        return e;
    return cudaSuccess;
}

} // namespace

// ── KV cache ────────────────────────────────────────────────────────────────

void GraniteKvCache::Free()
{
    cudaFree(d_k);
    cudaFree(d_v);
    d_k = nullptr;
    d_v = nullptr;
    last_tok = -1;
    kv_rows = 0;
    max_total = 0;
    cache_bytes = 0;
}

cudaError_t GraniteKvCache::Allocate(const GranitePinnedWeights &pw,
                                     int max_total_, cudaStream_t stream)
{
    Free();
    const int64_t row = static_cast<int64_t>(pw.top.num_key_value_heads) *
                        pw.top.head_dim;
    const int64_t bytes = static_cast<int64_t>(pw.top.num_layers) * max_total_ *
                          row * sizeof(__half);
    cudaError_t e = cudaMalloc(reinterpret_cast<void **>(&d_k),
                               static_cast<size_t>(bytes));
    if (e != cudaSuccess) return e;
    e = cudaMalloc(reinterpret_cast<void **>(&d_v), static_cast<size_t>(bytes));
    if (e != cudaSuccess) { Free(); return e; }
    e = cudaMemsetAsync(d_k, 0, static_cast<size_t>(bytes), stream);
    if (e != cudaSuccess) { Free(); return e; }
    e = cudaMemsetAsync(d_v, 0, static_cast<size_t>(bytes), stream);
    if (e != cudaSuccess) { Free(); return e; }
    max_total = max_total_;
    cache_bytes = bytes * 2;
    return cudaSuccess;
}

cudaError_t GraniteKvCache::EnsureCapacity(const GranitePinnedWeights &pw,
                                           int needed, cudaStream_t stream)
{
    if (d_k && needed <= max_total) return cudaSuccess;
    const int new_total = std::max(needed, max_total * 2);
    // Preserve written rows by copying into the fresh allocation.
    const int64_t row = static_cast<int64_t>(pw.top.num_key_value_heads) *
                        pw.top.head_dim;
    const int rows = kv_rows;
    __half *old_k = d_k;
    __half *old_v = d_v;
    const int old_total = max_total;
    d_k = nullptr;
    d_v = nullptr;
    max_total = 0;
    cache_bytes = 0;
    cudaError_t e = Allocate(pw, new_total, stream);
    if (e != cudaSuccess) { cudaFree(old_k); cudaFree(old_v); return e; }
    if (rows > 0 && old_k) {
        const size_t layer_bytes = static_cast<size_t>(old_total) * row *
                                   sizeof(__half);
        const size_t copy_bytes = static_cast<size_t>(rows) * row * sizeof(__half);
        for (int l = 0; l < pw.top.num_layers; ++l) {
            e = cudaMemcpyAsync(d_k + static_cast<int64_t>(l) * new_total * row,
                                old_k + static_cast<int64_t>(l) * old_total * row,
                                copy_bytes, cudaMemcpyDeviceToDevice, stream);
            if (e != cudaSuccess) break;
            e = cudaMemcpyAsync(d_v + static_cast<int64_t>(l) * new_total * row,
                                old_v + static_cast<int64_t>(l) * old_total * row,
                                copy_bytes, cudaMemcpyDeviceToDevice, stream);
            if (e != cudaSuccess) break;
        }
        (void)layer_bytes;
    }
    cudaFree(old_k);
    cudaFree(old_v);
    return e;
}

// ── Layer / generation ──────────────────────────────────────────────────────

namespace {

cudaError_t RunLayer(cudaStream_t stream, const GranitePinnedWeights &pw,
                     const GraniteLayerWeights &lw, int seq, int kv_base,
                     const int32_t *d_pos, Workspace &ws, __half *cache_k,
                     __half *cache_v)
{
    const auto &top = pw.top;
    const int H = top.hidden_size;
    const int I = top.intermediate_size;
    const int NH = top.num_attention_heads;
    const int NKV = top.num_key_value_heads;
    const int HD = top.head_dim;

    cudaError_t e = GraniteRmsNorm(stream, ws.norm, ws.x, lw.d_input_norm, seq,
                                   H, static_cast<float>(top.rms_norm_eps));
    if (e != cudaSuccess) return e;

    if (lw.qkv.d_packed) {
        // One GEMV over the fused [NH*HD + 2*NKV*HD, K] weight, then split.
        e = RunLsxNvfp4Gemm(stream, ws.norm, lw.qkv.d_packed, lw.qkv.d_scales,
                            lw.qkv.global_scale, ws.qkv, seq, H,
                            (NH + 2 * NKV) * HD);
        if (e != cudaSuccess) return e;
        e = GraniteSplitQKV(stream, ws.qkv, ws.q, ws.k, ws.v, seq, NH * HD,
                            NKV * HD);
        if (e != cudaSuccess) return e;
    } else {
        e = RunLsxNvfp4Gemm(stream, ws.norm, lw.q.d_packed, lw.q.d_scales,
                            lw.q.global_scale, ws.q, seq, H, NH * HD);
        if (e != cudaSuccess) return e;
        e = RunLsxNvfp4Gemm(stream, ws.norm, lw.k.d_packed, lw.k.d_scales,
                            lw.k.global_scale, ws.k, seq, H, NKV * HD);
        if (e != cudaSuccess) return e;
        e = RunLsxNvfp4Gemm(stream, ws.norm, lw.v.d_packed, lw.v.d_scales,
                            lw.v.global_scale, ws.v, seq, H, NKV * HD);
        if (e != cudaSuccess) return e;
    }

    e = GraniteRope(stream, ws.q, ws.k, d_pos, seq, NH, NKV, HD,
                    top.rope_theta);
    if (e != cudaSuccess) return e;

    // Append this chunk's k/v rows into the persistent cache at kv_base.
    const int64_t kv_row = static_cast<int64_t>(NKV) * HD;
    const size_t layer_stride = static_cast<size_t>(ws.max_seq) * kv_row;
    // cache layer offset is handled by the caller passing the layer slice,
    // but the capacity may exceed ws.max_seq; the caller passes the true
    // cache stride via cache_k/cache_v pointers already offset per layer and
    // we compute the destination from kv_base using the cache's own stride.
    e = cudaMemcpyAsync(cache_k + static_cast<int64_t>(kv_base) * kv_row,
                        ws.k, static_cast<size_t>(seq) * kv_row * sizeof(__half),
                        cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(cache_v + static_cast<int64_t>(kv_base) * kv_row,
                        ws.v, static_cast<size_t>(seq) * kv_row * sizeof(__half),
                        cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess) return e;
    (void)layer_stride;

    e = GraniteAttention(stream, ws.q, cache_k, cache_v, ws.attn, ws.attn_part,
                         seq, NH, NKV, HD,
                         static_cast<float>(top.attention_multiplier), kv_base);
    if (e != cudaSuccess) return e;

    e = RunLsxNvfp4Gemm(stream, ws.attn, lw.o.d_packed, lw.o.d_scales,
                        lw.o.global_scale, ws.proj, seq, NH * HD, H);
    if (e != cudaSuccess) return e;
    kernels::AddInPlaceKernel<__half>
        <<<(seq * H + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            ws.proj, ws.x, static_cast<int64_t>(seq) * H);
    e = cudaGetLastError();
    if (e != cudaSuccess) return e;

    e = GraniteRmsNorm(stream, ws.norm, ws.x, lw.d_post_norm, seq, H,
                       static_cast<float>(top.rms_norm_eps));
    if (e != cudaSuccess) return e;

    e = RunLsxNvfp4Gemm(stream, ws.norm, lw.mlp_gate_up.d_packed,
                        lw.mlp_gate_up.d_scales, lw.mlp_gate_up.global_scale,
                        ws.gate_up, seq, H, 2 * I);
    if (e != cudaSuccess) return e;
    e = GraniteSwiGLU(stream, ws.gate_up, ws.act, I,
                      static_cast<int64_t>(seq) * I);
    if (e != cudaSuccess) return e;
    e = RunLsxNvfp4Gemm(stream, ws.act, lw.mlp_down.d_packed,
                        lw.mlp_down.d_scales, lw.mlp_down.global_scale, ws.down,
                        seq, I, H);
    if (e != cudaSuccess) return e;
    kernels::AddInPlaceKernel<__half>
        <<<(seq * H + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            ws.down, ws.x, static_cast<int64_t>(seq) * H);
    return cudaGetLastError();
}

cudaError_t SampleNext(cudaStream_t stream, Workspace &ws,
                       const GranitePinnedWeights &pw, int seq,
                       lsxcommon::sample::ISampler &sampler, int32_t &tok,
                       float &maxv)
{
    const int H = pw.top.hidden_size;
    const int V = pw.top.vocab_size;
    cudaError_t e = GraniteRmsNorm(
        stream, ws.norm, ws.x, pw.d_final_norm, seq, H,
        static_cast<float>(pw.top.rms_norm_eps));
    if (e != cudaSuccess) return e;
    const __half *last = ws.norm + static_cast<int64_t>(seq - 1) * H;
    e = RunLsxNvfp4Gemm(stream, last, pw.head.d_packed, pw.head.d_scales,
                        pw.head.global_scale, ws.logits, 1, H, V);
    if (e != cudaSuccess) return e;
    e = sampler.Sample(stream, ws.logits, V,
                       lsxcommon::sample::LogitType::kF16, ws.tok, ws.maxv);
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(&tok, ws.tok, sizeof(int32_t),
                        cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(&maxv, ws.maxv, sizeof(float), cudaMemcpyDeviceToHost,
                        stream);
    return e;
}

} // namespace

// ── Testable kernel wrappers ────────────────────────────────────────────────

cudaError_t RunGraniteRopeKernel(cudaStream_t stream, __half *q, __half *k,
                                 const int32_t *pos, int seq, int nh, int nkv,
                                 int hd, double theta)
{
    return GraniteRope(stream, q, k, pos, seq, nh, nkv, hd, theta);
}

cudaError_t RunGraniteAttentionKernel(cudaStream_t stream, const __half *q,
                                      const __half *k, const __half *v,
                                      __half *out, int seq, int nh, int nkv,
                                      int hd, float scale, int kv_base)
{
    return GraniteAttention(stream, q, k, v, out, nullptr, seq, nh, nkv, hd,
                            scale, kv_base);
}

cudaError_t RunGraniteAttentionSplitKernel(cudaStream_t stream, const __half *q,
                                           const __half *k, const __half *v,
                                           __half *out, int seq, int nh, int nkv,
                                           int hd, float scale, int kv_base)
{
    const int64_t slots = static_cast<int64_t>(kAttnSplitMax) * seq * nh;
    const size_t bytes = static_cast<size_t>(slots) * (hd + 2) * sizeof(float);
    float *partials = nullptr;
    if (cudaMalloc(reinterpret_cast<void **>(&partials), bytes) != cudaSuccess)
        return cudaGetLastError();
    // Forced-split: exercise the maximum split count (many short/empty
    // boundary ranges) regardless of shape.
    const cudaError_t e =
        GraniteAttentionSplit(stream, q, k, v, out, partials, seq, nh, nkv, hd,
                              scale, kv_base, kAttnSplitMax);
    cudaFree(partials);
    return e;
}

cudaError_t RunGraniteRmsNormKernel(cudaStream_t stream, __half *out,
                                    const __half *x, const __half *w, int rows,
                                    int hidden, float eps)
{
    return GraniteRmsNorm(stream, out, x, w, rows, hidden, eps);
}

// ── Generation ──────────────────────────────────────────────────────────────

cudaError_t RunGraniteGeneration(cudaStream_t stream,
                                 const GranitePinnedWeights &pw,
                                 const std::vector<int32_t> &prompt, int max_new,
                                 int eos_id, GraniteGenerationResult &result,
                                 lsxcommon::sample::ISampler &sampler,
                                 GraniteKvCache *kv,
                                 const lsxcommon::GenerationStream *gs)
{
    const auto t_start = std::chrono::steady_clock::now();
    result = GraniteGenerationResult{};
    const auto &top = pw.top;
    const int H = top.hidden_size;
    const int V = top.vocab_size;
    const int P = static_cast<int>(prompt.size());
    if (P <= 0) {
        result.error_msg = "empty prompt";
        return cudaErrorInvalidValue;
    }
    if (top.num_layers <= 0 || pw.layers.size() <
                                   static_cast<size_t>(top.num_layers)) {
        result.error_msg = "model has no loaded layers";
        return cudaErrorInvalidValue;
    }
    if (!pw.embed.d_packed || !pw.head.d_packed || !pw.d_final_norm) {
        result.error_msg = "embed/head/final-norm missing";
        return cudaErrorInvalidValue;
    }

    Workspace ws;
    int work_seq = P;
    if (!kv) work_seq = P + std::max(max_new, 1);
    cudaError_t e = ws.Alloc(top, work_seq);
    if (e != cudaSuccess) { ws.Free(); result.error_msg = "workspace alloc"; return e; }

    // Dequant embed to FP16 for the gather.
    e = lsxcommon::gpu::dequant_shared_nvfp4(
        stream, pw.embed.d_packed, pw.embed.packed_bytes, pw.embed.d_scales,
        pw.embed.scale_bytes, pw.embed.global_scale, ws.embed_h,
        static_cast<int64_t>(V) * H);
    if (e != cudaSuccess) { ws.Free(); result.error_msg = "dequant embed"; return e; }

    GraniteKvCache local_kv;
    GraniteKvCache *cache = kv;
    if (!cache) {
        e = local_kv.Allocate(pw, P + std::max(max_new, 1), stream);
        if (e != cudaSuccess) { ws.Free(); result.error_msg = "kv alloc"; return e; }
        cache = &local_kv;
    } else {
        e = cache->EnsureCapacity(pw, P + std::max(max_new, 1), stream);
        if (e != cudaSuccess) { ws.Free(); result.error_msg = "kv grow"; return e; }
    }

    const int64_t kv_row = static_cast<int64_t>(top.num_key_value_heads) *
                           top.head_dim;

    // ── Prefill ──
    {
        const int seq = P;
        std::vector<int32_t> pos(static_cast<size_t>(seq));
        for (int i = 0; i < seq; ++i) pos[static_cast<size_t>(i)] = i;
        e = cudaMemcpyAsync(ws.ids, prompt.data(),
                            static_cast<size_t>(seq) * sizeof(int32_t),
                            cudaMemcpyHostToDevice, stream);
        if (e == cudaSuccess)
            e = cudaMemcpyAsync(ws.pos, pos.data(),
                                static_cast<size_t>(seq) * sizeof(int32_t),
                                cudaMemcpyHostToDevice, stream);
        if (e == cudaSuccess)
            kernels::EmbedGatherKernel<__half>
                <<<(seq * H + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                    ws.embed_h, ws.ids, ws.x, seq, H);
        if (e == cudaSuccess) e = cudaGetLastError();
        if (e != cudaSuccess) { ws.Free(); result.error_msg = "prefill embed"; return e; }

        for (int l = 0; l < top.num_layers && e == cudaSuccess; ++l) {
            __half *ck = cache->d_k + static_cast<int64_t>(l) *
                                          cache->max_total * kv_row;
            __half *cv = cache->d_v + static_cast<int64_t>(l) *
                                          cache->max_total * kv_row;
            e = RunLayer(stream, pw, pw.layers[static_cast<size_t>(l)], seq, 0,
                         ws.pos, ws, ck, cv);
        }
        if (e == cudaSuccess) {
            int32_t tok = -1;
            float maxv = 0.f;
            e = SampleNext(stream, ws, pw, seq, sampler, tok, maxv);
            if (e == cudaSuccess) {
                e = cudaStreamSynchronize(stream);
                if (e == cudaSuccess) {
                    cache->kv_rows = seq;
                    cache->last_tok = tok;
                    result.first_token = tok;
                    result.tokens.push_back(tok);
                    if (gs) gs->Emit(static_cast<std::uint32_t>(tok), 0u);
                    result.resident_bytes = pw.resident_bytes;
                }
            }
        }
        if (e != cudaSuccess) { ws.Free(); result.error_msg = "prefill"; return e; }
    }

    const double t_ttft = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t_start)
                              .count();
    result.ttft_ms = t_ttft;

    if (result.first_token == eos_id) result.eos_terminated = true;

    // ── Decode ──
    for (int step = 1; step < max_new && !result.eos_terminated; ++step) {
        const int seq = 1;
        const int kv_base = cache->kv_rows;
        const int32_t next = cache->last_tok;
        const int32_t p0 = kv_base;
        e = cudaMemcpyAsync(ws.ids, &next, sizeof(int32_t),
                            cudaMemcpyHostToDevice, stream);
        if (e == cudaSuccess)
            e = cudaMemcpyAsync(ws.pos, &p0, sizeof(int32_t),
                                cudaMemcpyHostToDevice, stream);
        if (e == cudaSuccess)
            kernels::EmbedGatherKernel<__half>
                <<<(H + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                    ws.embed_h, ws.ids, ws.x, seq, H);
        if (e == cudaSuccess) e = cudaGetLastError();
        if (e != cudaSuccess) { ws.Free(); result.error_msg = "decode embed"; return e; }

        for (int l = 0; l < top.num_layers && e == cudaSuccess; ++l) {
            __half *ck = cache->d_k + static_cast<int64_t>(l) *
                                          cache->max_total * kv_row;
            __half *cv = cache->d_v + static_cast<int64_t>(l) *
                                          cache->max_total * kv_row;
            e = RunLayer(stream, pw, pw.layers[static_cast<size_t>(l)], seq, kv_base,
                         ws.pos, ws, ck, cv);
        }
        if (e != cudaSuccess) { ws.Free(); result.error_msg = "decode layer"; return e; }

        int32_t tok = -1;
        float maxv = 0.f;
        e = SampleNext(stream, ws, pw, seq, sampler, tok, maxv);
        if (e == cudaSuccess) e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess) { ws.Free(); result.error_msg = "decode sample"; return e; }

        cache->last_tok = tok;
        cache->kv_rows += 1;
        result.tokens.push_back(tok);
        if (gs)
            gs->Emit(static_cast<std::uint32_t>(tok),
                     static_cast<std::uint32_t>(result.tokens.size() - 1));
        if (tok == eos_id) result.eos_terminated = true;
        if (gs && gs->Aborted()) break;
    }

    // Capture the final logits row (determinism evidence for SelfCheck). The
    // GEMM emits FP16 logits; promote to FP32 on the host.
    {
        std::vector<__half> h(static_cast<size_t>(V));
        e = cudaMemcpyAsync(h.data(), ws.logits,
                            static_cast<size_t>(V) * sizeof(__half),
                            cudaMemcpyDeviceToHost, stream);
        if (e == cudaSuccess) e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess) { ws.Free(); result.error_msg = "logits copy"; return e; }
        result.final_logits.resize(static_cast<size_t>(V));
        for (int i = 0; i < V; ++i)
            result.final_logits[static_cast<size_t>(i)] =
                __half2float(h[static_cast<size_t>(i)]);
    }

    const bool has_last = !result.tokens.empty();
    result.ok = has_last;
    if (!has_last && result.error_msg.empty())
        result.error_msg = "no tokens generated";
    result.total_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t_start)
                          .count();
    result.resident_bytes = pw.resident_bytes;

    ws.Free();
    if (!kv) local_kv.Free();
    return cudaSuccess;
}

} // namespace granite
} // namespace lsxcommon
