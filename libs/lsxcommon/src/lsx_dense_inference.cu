// Model-neutral dense-decoder kernels. Bodies are the Granite-4.2-30B runtime
// kernels moved verbatim; see tools/dedup_provenance.py for the proof.

#include "lsxcommon/lsx_dense_inference.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include "lsx_kernels_common.cuh"
#include "lsx_kernels_math.cuh"
#include "lsx_kernels_norm.cuh"

namespace lsxcommon {
namespace dense {

namespace {

constexpr int kThreads = 128;

// ── Kernels ─────────────────────────────────────────────────────────────────

// Full-dimension RoPE (HF rotate_half pairing): dims i and i+hd/2 form a pair.
// q: [seq, nh, hd], k: [seq, nkv, hd]; positions are absolute (kv_base + t).
__global__ void dense_rope_kernel(__half *__restrict__ q,
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
__global__ void dense_attn_kernel(const __half *__restrict__ q,
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

__global__ void dense_attn_split_kernel(const __half *__restrict__ q,
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
__global__ void dense_attn_combine_kernel(const float *__restrict__ pm,
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
__global__ void dense_swiglu_fused_kernel(const __half *__restrict__ gu,
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
__global__ void dense_split_qkv_kernel(const __half *__restrict__ qkv,
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
__global__ void dense_rmsnorm_kernel(const __half *__restrict__ x,
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

} // namespace

cudaError_t RunDenseRmsNormKernel(cudaStream_t stream, __half *out, const __half *x,
                           const __half *w, int rows, int hidden, float eps)
{
    // The vectorized kernel needs an even hidden (half2) and 4-byte-aligned
    // rows; fall back to the shared tree kernel otherwise.
    const bool vec_ok = (hidden % 2) == 0 && hidden >= 64;
    if (vec_ok) {
        constexpr int kNormThreads = 256;
        dense_rmsnorm_kernel<<<rows, kNormThreads, 0, stream>>>(x, w, out,
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

cudaError_t RunDenseRopeKernel(cudaStream_t stream, __half *q, __half *k,
                        const int32_t *pos, int seq, int nh, int nkv, int hd,
                        double theta)
{
    const int64_t total = static_cast<int64_t>(seq) * (nh + nkv) * (hd / 2);
    const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
    dense_rope_kernel<<<blocks, kThreads, 0, stream>>>(
        q, k, pos, seq, nh, nkv, hd, theta);
    return cudaGetLastError();
}

// Split dispatch floors: the split path only pays off when the key range is
// long enough to fill the SM array (min total = kv_base+1) and seq is small
// enough that the caller pre-allocated the [kAttnSplitMaxSeq, splits] partial
// slab (long-seq prefill already has seq*nh blocks of its own).
constexpr int kAttnSplitMaxSeq = 8;
constexpr int kAttnSplitMinKv = 256;

cudaError_t RunDenseAttentionSplitKernel(cudaStream_t stream, const __half *q,
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
    dense_attn_split_kernel<<<grid, kThreads, shmem, stream>>>(
        q, k, v, pm, pl, po, seq, nh, nkv, hd, scale, kv_base, splits);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    dense_attn_combine_kernel<<<seq * nh, kThreads, 0, stream>>>(
        pm, pl, po, out, seq, nh, hd, splits);
    return cudaGetLastError();
}

cudaError_t RunDenseAttentionKernel(cudaStream_t stream, const __half *q,
                             const __half *k, const __half *v, __half *out,
                             float *partials, int seq, int nh, int nkv, int hd,
                             float scale, int kv_base)
{
    if (partials != nullptr && seq <= kAttnSplitMaxSeq &&
        kv_base + 1 >= kAttnSplitMinKv) {
        // One split per ~kAttnSplitKeys keys, capped by the partial slab.
        int splits = (kv_base + 1 + kAttnSplitKeys - 1) / kAttnSplitKeys;
        return RunDenseAttentionSplitKernel(stream, q, k, v, out, partials, seq, nh,
                                     nkv, hd, scale, kv_base, splits);
    }
    const size_t shmem =
        (2 * static_cast<size_t>(hd) + 2 * kThreads) * sizeof(float);
    dense_attn_kernel<<<seq * nh, kThreads, shmem, stream>>>(
        q, k, v, out, seq, nh, nkv, hd, scale, kv_base);
    return cudaGetLastError();
}

cudaError_t RunDenseSplitQKVKernel(cudaStream_t stream, const __half *qkv, __half *q,
                            __half *k, __half *v, int seq, int qn, int kn)
{
    const int64_t total = static_cast<int64_t>(seq) * (qn + 2 * kn);
    const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
    dense_split_qkv_kernel<<<blocks, kThreads, 0, stream>>>(qkv, q, k, v, seq,
                                                              qn, kn);
    return cudaGetLastError();
}

cudaError_t RunDenseSwiGLUKernel(cudaStream_t stream, const __half *gu, __half *out,
                          int I, int64_t n)
{
    const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
    dense_swiglu_fused_kernel<<<blocks, kThreads, 0, stream>>>(gu, out, I, n);
    return cudaGetLastError();
}
} // namespace dense
} // namespace lsxcommon
