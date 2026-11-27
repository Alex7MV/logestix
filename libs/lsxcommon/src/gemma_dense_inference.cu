// Gemma-4-31B-it (dense) decoder runtime — Model Ingestion Protocol step 5.
//
// One stream, no per-token cudaDeviceSynchronize. Every dense projection goes
// through the model-neutral NVFP4 layer (lsx_nvfp4::RunLsxNvfp4Gemm); the
// family-local kernels are the four per-layer norms, the HF-faithful RoPE,
// the causal GQA attention with the sliding band, GeGLU, the NVFP4 embed
// gather and the softcapped logits/argmax.
//
// Reference semantics (transformers/models/gemma4, verified numerically
// against Gemma4TextRotaryEmbedding + Gemma4TextDecoderLayer):
//
//   x = x * layer_scalar
//   x = x + post_ffn(mlp(pre_ffn(x)))                      # GeGLU
//   x = x + post_attn(o_proj(attn))                        # sandwich norm
//   x = input_norm(x) -> q/k/v -> q_norm/k_norm/v_norm -> RoPE -> attention
//
// with three facts that are easy to get wrong and each has a test:
//   * the attention score scale is 1.0 (q/k are per-head RMS-normalized), NOT
//     1/sqrt(head_dim);
//   * V on a full-attention layer is the RAW k_proj output, and v_norm is
//     applied to it (not the k-normed, rotated key);
//   * the global layers use "proportional" RoPE: the encoding spans the whole
//     head dim and pairs (i, i + head_dim/2), with only the first
//     partial_rotary_factor*head_dim/2 frequency pairs live.

#include "lsxcommon/gemma_dense_inference.h"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "lsxcommon/gemma_dense_constants.h"
#include "lsxcommon/log.h"
#include "lsxcommon/lsx_inference_nvfp4.h"

#include "lsx_kernels_common.cuh"
#include "lsx_kernels_math.cuh"
#include "lsx_kernels_norm.cuh"
#include "lsx_kernels_quant.cuh"

namespace lsxcommon {
namespace gemma_dense {

using lsx_nvfp4::RunLsxNvfp4Gemm;

namespace {

constexpr int kThreads = 128;
constexpr int kNormThreads = 256;

#define GD_FAIL_LINE(e)                                                     \
    do {                                                                    \
        log::error("[gemma_dense] failure at line " +                       \
                   std::to_string(__LINE__) + ": " +                        \
                   cudaGetErrorString(e));                                   \
    } while (0)

// ── RMSNorm ─────────────────────────────────────────────────────────────────

// Fast RMSNorm for the decode path: half2 loads, a shuffle reduction and ONE
// cross-warp round (2 __syncthreads) instead of the generic tree kernel's
// per-level barriers. The tree kernel measured 11.4 us/call at hidden=5376
// (241 norms/token, ~12.5% of decode kernel time); this is the form the repo
// already ships for granite. Numeric class: FP32 accumulate, fixed warp-tree
// order (deterministic), same as the tree.
__global__ void gemma_dense_rmsnorm_kernel(const __half *__restrict__ x,
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
    __half2 *o2 = reinterpret_cast<__half2 *>(orow);
    for (int j = threadIdx.x; j < n2; j += tpb) {
        const float2 f = __half22float2(x2[j]);
        const float2 g = __half22float2(w2[j]);
        o2[j] = __floats2half2_rn(__fmul_rn(__fmul_rn(f.x, inv), g.x),
                                  __fmul_rn(__fmul_rn(f.y, inv), g.y));
    }
}

__global__ void gemma_dense_rmsnorm_plain_kernel(
    const __half *__restrict__ x, __half *__restrict__ out, int hidden,
    float eps)
{
    const int row = blockIdx.x;
    const int tpb = blockDim.x;
    const int n2 = hidden >> 1;
    const __half *xr = x + static_cast<int64_t>(row) * hidden;
    __half *orow = out + static_cast<int64_t>(row) * hidden;
    const __half2 *x2 = reinterpret_cast<const __half2 *>(xr);
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
    __half2 *o2 = reinterpret_cast<__half2 *>(orow);
    for (int j = threadIdx.x; j < n2; j += tpb) {
        const float2 f = __half22float2(x2[j]);
        o2[j] = __floats2half2_rn(__fmul_rn(f.x, inv), __fmul_rn(f.y, inv));
    }
}

cudaError_t RmsNorm(cudaStream_t stream, const __half *x, const __half *w,
                    __half *out, int rows, int hidden, float eps)
{
    if ((hidden % 2) == 0 && hidden >= 64) {
        if (w) {
            gemma_dense_rmsnorm_kernel<<<rows, kNormThreads, 0, stream>>>(
                x, w, out, hidden, eps);
        } else {
            gemma_dense_rmsnorm_plain_kernel<<<rows, kNormThreads, 0, stream>>>(
                x, out, hidden, eps);
        }
        return cudaGetLastError();
    }
    constexpr int kBlock = kNormThreads;
    const size_t shmem =
        kBlock * sizeof(float) +
        std::min(hidden * static_cast<int>(sizeof(__half)), 48 * 1024);
    if (w) {
        kernels::RmsNormTreeKernel<__half, kernels::NormWeight::Direct>
            <<<rows, kBlock, shmem, stream>>>(x, w, out, rows, hidden, eps);
    } else {
        kernels::RmsNormTreeKernel<__half, kernels::NormWeight::None>
            <<<rows, kBlock, shmem, stream>>>(x, nullptr, out, rows, hidden,
                                              eps);
    }
    return cudaGetLastError();
}

// ── Per-head RMSNorm ────────────────────────────────────────────────────────
// Grid (rows, heads): ONE block per (token, head), reducing over head_dim.
// This is the q_norm / k_norm (weighted) and v_norm (plain) form; the shared
// row kernel is wrong here because a row spans several heads.
__global__ void gemma_dense_head_norm_kernel(const __half *__restrict__ x,
                                             const __half *__restrict__ w,
                                             __half *__restrict__ out,
                                             int rows, int heads, int head_dim,
                                             float eps)
{
    // NOTE: no dynamic shared memory here. A kernel must not mix `extern
    // __shared__` with a static `__shared__` array (undefined behaviour that
    // showed up as NaN on a real checkpoint but not on synthetic shapes).
    // One block per (token, head): the flattened row index spans rows*heads.
    const int row = blockIdx.y * heads + blockIdx.x;
    if (row >= rows * heads) return;
    const __half *xr = x + static_cast<int64_t>(row) * head_dim;
    __half *orow = out + static_cast<int64_t>(row) * head_dim;
    const int tpb = blockDim.x;

    float sum = 0.f;
    for (int d = threadIdx.x; d < head_dim; d += tpb) {
        const float v = __half2float(xr[d]);
        sum = __fmaf_rn(v, v, sum);
    }
    for (int off = 16; off > 0; off >>= 1)
        sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffu, sum, off));
    __shared__ float ws[32];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (lane == 0) ws[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < (tpb >> 5)) ? ws[lane] : 0.f;
        for (int off = 16; off > 0; off >>= 1)
            v = __fadd_rn(v, __shfl_down_sync(0xffffffffu, v, off));
        if (lane == 0) ws[0] = v;
    }
    __syncthreads();
    const float inv =
        rsqrtf(__fadd_rn(__fdiv_rn(ws[0], static_cast<float>(head_dim)), eps));
    for (int d = threadIdx.x; d < head_dim; d += tpb) {
        const float v =
            __fmul_rn(__half2float(xr[d]), inv);
        orow[d] = w ? __float2half(__fmul_rn(v, __half2float(w[d]))) : __float2half(v);
    }
}

// ── RoPE ────────────────────────────────────────────────────────────────────
// rotate_half over the WHOLE head dim: dims (i, i + hd/2) form a pair for
// i < rot_angles with frequency theta^(-2i/hd); every other pair carries
// cos = 1 / sin = 0 and is a copy. q and k are rotated in place; values are
// never rotated.
__global__ void gemma_dense_rope_kernel(const int32_t *__restrict__ pos,
                                        __half *__restrict__ q,
                                        __half *__restrict__ k, int seq, int nh,
                                        int nkv, int hd, int rot_angles,
                                        double theta)
{
    const int half = hd / 2;
    const int64_t total =
        static_cast<int64_t>(seq) * (nh + nkv) * static_cast<int64_t>(half);
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
         idx < total; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int i = static_cast<int>(idx % half);
        const int64_t rest = idx / half;
        const int head = static_cast<int>(rest % (nh + nkv));
        const int t = static_cast<int>(rest / (nh + nkv));
        const float c = 1.0f, s = 0.0f;
        float cc = c, ss = s;
        if (i < rot_angles) {
            const float freq = static_cast<float>(
                std::pow(theta, -2.0 * static_cast<double>(i) / hd));
            const float ang = static_cast<float>(pos[t]) * freq;
            cc = cosf(ang);
            ss = sinf(ang);
        }
        __half *row = (head < nh)
                          ? q + (static_cast<int64_t>(t) * nh + head) * hd
                          : k + (static_cast<int64_t>(t) * nkv +
                                 (head - nh)) * hd;
        const float a = __half2float(row[i]);
        const float b = __half2float(row[i + half]);
        row[i] = __float2half(__fsub_rn(__fmul_rn(a, cc), __fmul_rn(b, ss)));
        row[i + half] =
            __float2half(__fadd_rn(__fmul_rn(b, cc), __fmul_rn(a, ss)));
    }
}

// ── Attention ───────────────────────────────────────────────────────────────
// One block per (query row, head). Shared-memory reductions behind
// __syncthreads() (no atomics, deterministic); the sliding band predicate is
// evaluated per key slot so an out-of-band row never enters any reduction.
// KV addresses come exclusively from KvCacheGeometry::Offset.
__global__ void gemma_dense_attn_kernel(const __half *__restrict__ q,
                                        const __half *__restrict__ kc,
                                        const __half *__restrict__ vc,
                                        __half *__restrict__ out, int seq, int nh,
                                        int nkv, int hd, int window, float scale,
                                        int pos_base, KvCacheGeometry kgeom,
                                        KvCacheGeometry vgeom)
{
    const int t = blockIdx.x / nh;
    const int h = blockIdx.x % nh;
    if (t >= seq) return;
    const int kvh = h / (nh / nkv);
    const int tid = threadIdx.x;
    const int tpb = blockDim.x;

    extern __shared__ float smem[];
    float *q_s = smem;            // [hd]
    float *o_s = smem + hd;       // [hd]
    float *sc = smem + 2 * hd;    // [tpb]
    float *rd = sc + tpb;         // [tpb]

    const __half *qrow = q + (static_cast<int64_t>(t) * nh + h) * hd;
    for (int d = tid; d < hd; d += tpb) q_s[d] = __half2float(qrow[d]);
    for (int d = tid; d < hd; d += tpb) o_s[d] = 0.f;
    __syncthreads();

    const int qpos = pos_base + t;
    const int lo = window > 0 ? max(0, qpos - window + 1) : 0;
    float m = -CUDART_INF_F;
    float l = 0.f;
    for (int start = lo; start <= qpos; start += tpb) {
        const int cnt = min(tpb, qpos + 1 - start);
        float score = -CUDART_INF_F;
        if (tid < cnt) {
            const __half *krow =
                kc + kgeom.Offset(0, kvh, start + tid);
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
        for (int s = tpb / 2; s > 0; s >>= 1) {
            if (tid < s && sc[tid] < sc[tid + s]) sc[tid] = sc[tid + s];
            __syncthreads();
        }
        const float m_new = sc[0];
        __syncthreads();
        const float e = (tid < cnt)
                            ? device::ExpfGlibc(__fsub_rn(score, m_new))
                            : 0.f;
        sc[tid] = e;
        rd[tid] = e;
        __syncthreads();
        for (int s = tpb / 2; s > 0; s >>= 1) {
            if (tid < s) rd[tid] = __fadd_rn(rd[tid], rd[tid + s]);
            __syncthreads();
        }
        const float l_new = rd[0];
        __syncthreads();
        const float rescale =
            (m == -CUDART_INF_F) ? 0.f : device::ExpfGlibc(__fsub_rn(m, m_new));
        for (int d = tid; d < hd; d += tpb) {
            float acc = __fmul_rn(o_s[d], rescale);
            for (int j = 0; j < cnt; ++j) {
                const __half *vrow = vc + vgeom.Offset(0, kvh, start + j);
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
    for (int d = tid; d < hd; d += tpb)
        orow[d] = __float2half(__fmul_rn(o_s[d], inv));
}

// Split-KV (flash-decoding) variant: the key range is partitioned into
// `splits` contiguous chunks, each producing an online-softmax partial that a
// second kernel merges in fixed split order. Fixed boundaries + fixed merge
// order => run-to-run deterministic; differs from the unsplit kernel only by
// FP reduction grouping (same tolerance class).
constexpr int kAttnSplitKeys = 64;

__global__ void gemma_dense_attn_split_kernel(
    const __half *__restrict__ q, const __half *__restrict__ kc,
    const __half *__restrict__ vc, float *__restrict__ pm,
    float *__restrict__ pl, float *__restrict__ po, int seq, int nh, int nkv,
    int hd, int window, float scale, int pos_base, int splits,
    KvCacheGeometry kgeom, KvCacheGeometry vgeom)
{
    const int t = blockIdx.x / nh;
    const int h = blockIdx.x % nh;
    const int sv = blockIdx.y;
    if (t >= seq) return;
    const int kvh = h / (nh / nkv);
    const int tid = threadIdx.x;
    const int tpb = blockDim.x;

    extern __shared__ float smem[];
    float *q_s = smem;
    float *o_s = smem + hd;
    float *sc = smem + 2 * hd;
    float *rd = sc + tpb;

    const int qpos = pos_base + t;
    const int lo = window > 0 ? max(0, qpos - window + 1) : 0;
    const int total = qpos + 1 - lo;
    // Splits never straddle: even fixed boundaries over the same key range.
    const int n0 = static_cast<int>(static_cast<int64_t>(total) * sv / splits);
    const int n1 =
        static_cast<int>(static_cast<int64_t>(total) * (sv + 1) / splits);
    const int64_t slot = (static_cast<int64_t>(sv) * seq + t) * nh + h;

    const __half *qrow = q + (static_cast<int64_t>(t) * nh + h) * hd;
    for (int d = tid; d < hd; d += tpb) q_s[d] = __half2float(qrow[d]);
    for (int d = tid; d < hd; d += tpb) o_s[d] = 0.f;
    __syncthreads();

    float m = -CUDART_INF_F;
    float l = 0.f;
    for (int start = n0; start < n1; start += tpb) {
        const int cnt = min(tpb, n1 - start);
        float score = -CUDART_INF_F;
        if (tid < cnt) {
            const __half *krow = kc + kgeom.Offset(0, kvh, lo + start + tid);
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
        for (int s = tpb / 2; s > 0; s >>= 1) {
            if (tid < s && sc[tid] < sc[tid + s]) sc[tid] = sc[tid + s];
            __syncthreads();
        }
        const float m_new = sc[0];
        __syncthreads();
        const float e = (tid < cnt)
                            ? device::ExpfGlibc(__fsub_rn(score, m_new))
                            : 0.f;
        sc[tid] = e;
        rd[tid] = e;
        __syncthreads();
        for (int s = tpb / 2; s > 0; s >>= 1) {
            if (tid < s) rd[tid] = __fadd_rn(rd[tid], rd[tid + s]);
            __syncthreads();
        }
        const float l_new = rd[0];
        __syncthreads();
        const float rescale =
            (m == -CUDART_INF_F) ? 0.f : device::ExpfGlibc(__fsub_rn(m, m_new));
        for (int d = tid; d < hd; d += tpb) {
            float acc = __fmul_rn(o_s[d], rescale);
            for (int j = 0; j < cnt; ++j) {
                const __half *vrow =
                    vc + vgeom.Offset(0, kvh, lo + start + j);
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
    for (int d = tid; d < hd; d += tpb) po[slot * hd + d] = o_s[d];
}

__global__ void gemma_dense_attn_combine_kernel(
    const float *__restrict__ pm, const float *__restrict__ pl,
    const float *__restrict__ po, __half *__restrict__ out, int seq, int nh,
    int hd, int splits)
{
    const int t = blockIdx.x / nh;
    const int h = blockIdx.x % nh;
    if (t >= seq) return;
    const int tid = threadIdx.x;
    const int64_t base = static_cast<int64_t>(t) * nh + h;

    float m = -CUDART_INF_F;
    for (int s = 0; s < splits; ++s) {
        const float ms =
            pm[static_cast<int64_t>(s) * seq * nh + base];
        m = fmaxf(m, ms);
    }
    float l = 0.f;
    for (int s = 0; s < splits; ++s) {
        const int64_t slot = static_cast<int64_t>(s) * seq * nh + base;
        l = __fadd_rn(l, __fmul_rn(pl[slot], device::ExpfGlibc(__fsub_rn(pm[slot], m))));
    }
    __half *orow = out + base * hd;
    for (int d = tid; d < hd; d += blockDim.x) {
        float a = 0.f;
        for (int s = 0; s < splits; ++s) {
            const int64_t slot = static_cast<int64_t>(s) * seq * nh + base;
            a = __fadd_rn(a, __fmul_rn(po[slot * hd + d],
                                       device::ExpfGlibc(__fsub_rn(pm[slot], m))));
        }
        orow[d] = __float2half((l > 0.f) ? __fdiv_rn(a, l) : 0.f);
    }
}

// ── GeGLU ───────────────────────────────────────────────────────────────────
// ACT2FN["gelu_pytorch_tanh"]: 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715 x^3))).
// gate and up are the two SEPARATE [seq, I] projection outputs (the neutral
// GEMM writes contiguous [seq, N] blocks, so they cannot be fused into one
// per-token [2I] row here).
__global__ void gemma_dense_geglu_kernel(const __half *__restrict__ gate,
                                         const __half *__restrict__ up,
                                         __half *__restrict__ out, int64_t n)
{
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
    if (i >= n) return;
    const float g = __half2float(gate[i]);
    const float u = __half2float(up[i]);
    const float cube = g * g * g;
    const float inner = __fmaf_rn(0.044715f, cube, g);
    const float gelu = __fmul_rn(
        0.5f * g,
        __fadd_rn(1.0f, tanhf(kSqrt2OverPi * inner)));
    out[i] = __float2half(__fmul_rn(gelu, u));
}

// ── Embed gather (NVFP4 tied table) ─────────────────────────────────────────
// One block per (token, hidden-chunk); the table is never materialised in FP16
// (a 262144 x 5376 dequantised table would be 2.8 GB).
__global__ void gemma_dense_embed_nvfp4_kernel(
    const uint8_t *__restrict__ packed, const uint8_t *__restrict__ scales,
    float global_scale, const int32_t *__restrict__ ids, __half *__restrict__ out,
    int hidden, float embed_scale)
{
    const int tok = blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= hidden) return;
    const int64_t flat =
        static_cast<int64_t>(ids[tok]) * hidden + d;
    const uint8_t byte = packed[flat >> 1];
    const int nib = (flat & 1) ? (byte >> 4) : (byte & 0xF);
    const float v = gpu::nvfp4::E2M1Value(nib) *
                    kernels::Nvfp4DecodeScale(scales[flat >> 4], global_scale);
    out[static_cast<int64_t>(tok) * hidden + d] = __float2half(v * embed_scale);
}

// ── Softcapped logits + deterministic argmax ─────────────────────────────────
// Phase 1: one warp per vocabulary row, strict ascending-column FP32
// accumulation, then the softcap. Phase 2: block winners over the row in
// ascending block order, phase 3: a single block reduces those winners in
// ascending block order, lowest index on a tie. No atomicAdd feeds
// final_logits (the shared-memory sandbox rule), and run-to-run the result is
// bit-identical.
constexpr int kLogitsArgMaxBlocks = 4096;  // phase-2 blocks over the vocab row

// Applies the soft cap over the half GEMV output into the FP32 logits row.
__global__ void gemma_dense_softcap_kernel(const __half *__restrict__ in,
                                           float *__restrict__ out, int vocab,
                                           float cap)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= vocab) return;
    out[i] = tanhf(__half2float(in[i]) / cap) * cap;
}

// Phase 2: one block per contiguous slice of the vocabulary row. The slice is
// scanned in ascending index order (thread t takes t, t+blockDim, ...) and the
// block winner is the first strict maximum.
__global__ void gemma_dense_argmax_blocks_kernel(
    const float *__restrict__ logits, float *__restrict__ blk_best,
    std::int32_t *__restrict__ blk_idx, int vocab)
{
    __shared__ float sb[kThreads];
    __shared__ int si[kThreads];
    const int base = blockIdx.x * blockDim.x;
    float best = -CUDART_INF_F;
    int best_i = -1;
    for (int v = base + threadIdx.x; v < vocab; v += blockDim.x) {
        const float val = logits[v];
        if (val > best) {
            best = val;
            best_i = v;
        }
    }
    sb[threadIdx.x] = best;
    si[threadIdx.x] = best_i;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            const int other = si[threadIdx.x + s];
            // Ascending halves: the lower index wins a tie, so only replace on
            // a STRICTLY greater value.
            if (other >= 0 &&
                (si[threadIdx.x] < 0 || sb[threadIdx.x + s] > sb[threadIdx.x])) {
                sb[threadIdx.x] = sb[threadIdx.x + s];
                si[threadIdx.x] = other;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        blk_best[blockIdx.x] = sb[0];
        blk_idx[blockIdx.x] = si[0];
    }
}

// Phase 3: one block reduces the block winners in ascending block order.
__global__ void gemma_dense_argmax_reduce_kernel(
    const float *__restrict__ blk_best, const std::int32_t *__restrict__ blk_idx,
    std::int32_t *__restrict__ d_argmax, float *__restrict__ d_maxv,
    int n_blocks)
{
    __shared__ float sb[kThreads];
    __shared__ int si[kThreads];
    float best = -CUDART_INF_F;
    int best_i = -1;
    for (int b = threadIdx.x; b < n_blocks; b += blockDim.x) {
        if (blk_best[b] > best) {
            best = blk_best[b];
            best_i = blk_idx[b];
        }
    }
    sb[threadIdx.x] = best;
    si[threadIdx.x] = best_i;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            const int other = si[threadIdx.x + s];
            if (other >= 0 &&
                (si[threadIdx.x] < 0 || sb[threadIdx.x + s] > sb[threadIdx.x])) {
                sb[threadIdx.x] = sb[threadIdx.x + s];
                si[threadIdx.x] = other;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *d_argmax = si[0];
        *d_maxv = sb[0];
    }
}

// Split dispatch floors: the split path only pays off when the attended span
// is long enough to fill the SM array, and the caller has pre-allocated the
// partial slab for the short-seq regime.
constexpr int kAttnSplitMaxSeq = 8;
constexpr int kAttnSplitMinKv = 256;

// ── Residual scale ──────────────────────────────────────────────────────────
// hidden *= layer_scalar (a per-layer constant, applied last as upstream does).
__global__ void gemma_dense_scale_kernel(__half *__restrict__ x, float scalar,
                                         int64_t n)
{
    const int64_t i =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = __float2half(__fmul_rn(__half2float(x[i]), scalar));
}

// ── Workspace ───────────────────────────────────────────────────────────────

} // namespace

// Public generation-context lifecycle: must live OUTSIDE any anonymous
// namespace or its internal linkage leaves the model TU's call unresolved.
GemmaDenseGenContext *CreateGemmaDenseGenContext()
{
    return new GemmaDenseGenContext();
}

void DestroyGemmaDenseGenContext(GemmaDenseGenContext *ctx) { delete ctx; }

// ── Workspace ───────────────────────────────────────────────────────────────

void GemmaDenseGenContext::Workspace::Free()
{
    auto F = [](void **p) {
        if (*p) {
            cudaFree(*p);
            *p = nullptr;
        }
    };
    F((void **)&x);
    F((void **)&normed);
    F((void **)&q);
    F((void **)&k);
    F((void **)&v);
    F((void **)&attn);
    F((void **)&proj);
    F((void **)&gate_up);
    F((void **)&act);
    F((void **)&ffn);
    F((void **)&logits_h);
    F((void **)&logits);
    F((void **)&ids);
    F((void **)&pos);
    F((void **)&argmax);
    F((void **)&maxv);
    F((void **)&attn_part);
    seq_cap = q_rows_cap = kv_rows_cap = 0;
}

cudaError_t GemmaDenseGenContext::Workspace::Alloc(std::int64_t H,
                                                   std::int64_t I,
                                                   std::int64_t V, size_t S,
                                                   size_t q_rows_max,
                                                   size_t kv_rows_max,
                                                   size_t nh, size_t hd)
{
    Free();
    cudaError_t e = cudaSuccess;
    auto A = [&](void **p, size_t bytes) {
        return cudaMalloc(p, bytes);
    };
    if ((e = A((void **)&x, S * H * 2)) != cudaSuccess) return e;
    if ((e = A((void **)&normed, S * H * 2)) != cudaSuccess) return e;
    if ((e = A((void **)&q, S * q_rows_max * 2)) != cudaSuccess) return e;
    if ((e = A((void **)&k, S * kv_rows_max * 2)) != cudaSuccess) return e;
    if ((e = A((void **)&v, S * kv_rows_max * 2)) != cudaSuccess) return e;
    if ((e = A((void **)&attn, S * q_rows_max * 2)) != cudaSuccess) return e;
    if ((e = A((void **)&proj, S * H * 2)) != cudaSuccess) return e;
    if ((e = A((void **)&gate_up, S * 2 * I * 2)) != cudaSuccess) return e;
    if ((e = A((void **)&act, S * I * 2)) != cudaSuccess) return e;
    if ((e = A((void **)&ffn, S * H * 2)) != cudaSuccess) return e;
    if ((e = A((void **)&logits_h, static_cast<size_t>(V) * 2)) != cudaSuccess)
        return e;
    // logits carry the argmax tail: vocab floats + per-block winners
    // (value + index), which the phase-2/3 kernels own.
    const size_t argmax_tail =
        static_cast<size_t>((V + kThreads - 1) / kThreads) * 2;
    if ((e = A((void **)&logits, (static_cast<size_t>(V) + argmax_tail) * 4)) !=
        cudaSuccess)
        return e;
    if ((e = A((void **)&ids, S * sizeof(std::int32_t))) != cudaSuccess) return e;
    if ((e = A((void **)&pos, S * sizeof(std::int32_t))) != cudaSuccess) return e;
    if ((e = A((void **)&argmax, 4)) != cudaSuccess) return e;
    if ((e = A((void **)&maxv, 4)) != cudaSuccess) return e;
    const size_t sp_seq = std::min<size_t>(S, kAttnSplitMaxSeq);
    if ((e = A((void **)&attn_part,
               static_cast<size_t>(kGemmaDenseAttnSplitMax) * sp_seq * nh *
                   (hd + 2) * sizeof(float))) != cudaSuccess)
        return e;
    seq_cap = S;
    q_rows_cap = q_rows_max;
    kv_rows_cap = kv_rows_max;
    return cudaSuccess;
}

// ── Public launchers ────────────────────────────────────────────────────────

cudaError_t RunGemmaDenseRMSNorm(cudaStream_t stream, const __half *x,
                                 const __half *w, __half *out, int rows,
                                 int hidden, float eps)
{
    if (!x || !out || rows <= 0 || hidden <= 0) return cudaErrorInvalidValue;
    return RmsNorm(stream, x, w, out, rows, hidden, eps);
}

cudaError_t RunGemmaDenseHeadNorm(cudaStream_t stream, const __half *x,
                                  const __half *w, __half *out, int rows,
                                  int heads, int head_dim, float eps)
{
    if (!x || !out || rows <= 0 || heads <= 0 || head_dim <= 0)
        return cudaErrorInvalidValue;
    const dim3 grid(static_cast<unsigned>(heads), static_cast<unsigned>(rows));
    gemma_dense_head_norm_kernel<<<grid, kNormThreads, 0, stream>>>(
        x, w, out, rows, heads, head_dim, eps);
    return cudaGetLastError();
}

cudaError_t RunGemmaDenseRope(cudaStream_t stream, const int32_t *d_pos,
                              __half *q, __half *k, int seq, int nh, int nkv,
                              int hd, int rot_angles, double theta)
{
    if (!d_pos || !q || !k || seq <= 0 || nh <= 0 || nkv <= 0 || hd <= 0 ||
        hd % 2 != 0 || rot_angles <= 0 || rot_angles > hd / 2)
        return cudaErrorInvalidValue;
    const int64_t total =
        static_cast<int64_t>(seq) * (nh + nkv) * (hd / 2);
    const int blocks =
        static_cast<int>((total + kThreads * 4 - 1) / (kThreads * 4));
    gemma_dense_rope_kernel<<<blocks > 0 ? blocks : 1, kThreads, 0, stream>>>(
        d_pos, q, k, seq, nh, nkv, hd, rot_angles, theta);
    return cudaGetLastError();
}

namespace {

cudaError_t AttentionSplit(cudaStream_t stream, const __half *q,
                            const __half *k, const __half *v, __half *out,
                            float *partials, int seq, int nh, int nkv, int hd,
                            int window, float scale, int pos_base, int splits,
                            KvCacheGeometry kgeom, KvCacheGeometry vgeom)
{
    if (splits < 1) splits = 1;
    if (splits > kGemmaDenseAttnSplitMax) splits = kGemmaDenseAttnSplitMax;
    const size_t shmem =
        (2 * static_cast<size_t>(hd) + 2 * static_cast<size_t>(kThreads)) *
        sizeof(float);
    const int64_t slots = static_cast<int64_t>(splits) * seq * nh;
    float *pm = partials;
    float *pl = pm + slots;
    float *po = pl + slots;
    const dim3 grid(static_cast<unsigned>(seq * nh),
                    static_cast<unsigned>(splits));
    gemma_dense_attn_split_kernel<<<grid, kThreads, shmem, stream>>>(
        q, k, v, pm, pl, po, seq, nh, nkv, hd, window, scale, pos_base, splits,
        kgeom, vgeom);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { GD_FAIL_LINE(e); return e; }
    gemma_dense_attn_combine_kernel<<<seq * nh, kThreads, 0, stream>>>(
        pm, pl, po, out, seq, nh, hd, splits);
    return cudaGetLastError();
}

} // namespace

cudaError_t RunGemmaDenseAttention(cudaStream_t stream, const __half *q,
                                   const __half *k, const __half *v, __half *out,
                                   int seq, int nh, int nkv, int hd, int window,
                                   float scale, int pos_base,
                                   KvCacheGeometry kgeom, KvCacheGeometry vgeom)
{
    if (!q || !k || !v || !out || seq <= 0 || nh <= 0 || nkv <= 0 || hd <= 0 ||
        nh % nkv != 0 || hd % 2 != 0)
        return cudaErrorInvalidValue;
    const size_t shmem =
        (2 * static_cast<size_t>(hd) + 2 * static_cast<size_t>(kThreads)) *
        sizeof(float);
    gemma_dense_attn_kernel<<<seq * nh, kThreads, shmem, stream>>>(
        q, k, v, out, seq, nh, nkv, hd, window, scale, pos_base, kgeom, vgeom);
    return cudaGetLastError();
}

cudaError_t RunGemmaDenseAttentionSplit(cudaStream_t stream, const __half *q,
                                        const __half *k, const __half *v,
                                        __half *out, float *partials, int seq,
                                        int nh, int nkv, int hd, int window,
                                        float scale, int pos_base, int splits,
                                        KvCacheGeometry kgeom,
                                        KvCacheGeometry vgeom)
{
    if (!partials) return cudaErrorInvalidValue;
    return AttentionSplit(stream, q, k, v, out, partials, seq, nh, nkv, hd,
                          window, scale, pos_base, splits, kgeom, vgeom);
}

cudaError_t RunGemmaDenseGeGLU(cudaStream_t stream, const __half *gate,
                               const __half *up, __half *out, std::int64_t n)
{
    if (!gate || !up || !out || n <= 0) return cudaErrorInvalidValue;
    const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
    gemma_dense_geglu_kernel<<<blocks, kThreads, 0, stream>>>(gate, up, out, n);
    return cudaGetLastError();
}

cudaError_t RunGemmaDenseEmbedGatherNvfp4(cudaStream_t stream,
                                          const std::uint8_t *packed,
                                          const std::uint8_t *scales,
                                          float global_scale,
                                          const std::int32_t *ids, __half *out,
                                          int seq, int vocab, int hidden,
                                          float embed_scale)
{
    if (!packed || !scales || !ids || !out || seq <= 0 || vocab <= 0 ||
        hidden <= 0)
        return cudaErrorInvalidValue;
    const dim3 grid(static_cast<unsigned>((hidden + 255) / 256),
                    static_cast<unsigned>(seq));
    gemma_dense_embed_nvfp4_kernel<<<grid, 256, 0, stream>>>(
        packed, scales, global_scale, ids, out, hidden, embed_scale);
    return cudaGetLastError();
}

cudaError_t RunGemmaDenseLogitsArgmax(cudaStream_t stream,
                                      const std::uint8_t *packed,
                                      const std::uint8_t *scales,
                                      float global_scale, const __half *x,
                                      int vocab, int hidden, float cap,
                                      std::int32_t *d_argmax, float *d_maxlogit,
                                      float *d_logits, __half *d_logits_h)
{
    if (!packed || !scales || !x || !d_argmax || !d_maxlogit || !d_logits ||
        !d_logits_h || vocab <= 0 || hidden <= 0 || cap <= 0.f)
        return cudaErrorInvalidValue;
    // Warp-per-row GEMV through the neutral layer (the hand-rolled row kernel
    // measured 308 GB/s; the neutral warp-row path ~670 GB/s), then the cap.
    cudaError_t e = RunLsxNvfp4Gemm(stream, x, packed, scales, global_scale,
                                    d_logits_h, 1, hidden, vocab);
    if (e != cudaSuccess) { GD_FAIL_LINE(e); return e; }
    gemma_dense_softcap_kernel<<<(vocab + kThreads - 1) / kThreads, kThreads, 0,
                                 stream>>>(d_logits_h, d_logits, vocab, cap);
    e = cudaGetLastError();
    if (e != cudaSuccess) { GD_FAIL_LINE(e); return e; }

    const int blocks = (vocab + kThreads - 1) / kThreads;
    float *blk_best = d_logits + vocab;          // caller-sized tail
    std::int32_t *blk_idx = reinterpret_cast<std::int32_t *>(blk_best + blocks);
    if (blocks > kLogitsArgMaxBlocks) return cudaErrorInvalidValue;
    gemma_dense_argmax_blocks_kernel<<<blocks, kThreads, 0, stream>>>(
        d_logits, blk_best, blk_idx, vocab);
    e = cudaGetLastError();
    if (e != cudaSuccess) { GD_FAIL_LINE(e); return e; }
    gemma_dense_argmax_reduce_kernel<<<1, kThreads, 0, stream>>>(
        blk_best, blk_idx, d_argmax, d_maxlogit, blocks);
    return cudaGetLastError();
}

// ── Layer / generation ───────────────────────────────────────────────────────

namespace {

// One layer, `seq` tokens, writing k/v rows at [pos_base, pos_base + seq).
//
//   n   = rms(x, input_norm)
//   q,k,v = proj(n); q_norm/k_norm; (full layers: v = raw k, then v_norm);
//           RoPE on q and k only
//   a   = attention(scale 1.0, sliding band on the local layers)
//   x  += rms(a @ o_proj, post_attn)
//   n2  = rms(x, pre_ffn); x += rms(mlp(n2), post_ffn); x *= layer_scalar
cudaError_t ForwardLayer(cudaStream_t stream,
                         const GemmaDensePinnedWeights &pw,
                         const GemmaDenseLayerWeights &lw, int layer_idx,
                         GemmaDenseGenContext::Workspace &ws, int seq,
                         int pos_base, __half *dk, __half *dv,
                         KvCacheGeometry kgeo, KvCacheGeometry vgeo)
{
    const auto &top = pw.top;
    const std::int64_t H = top.hidden_size;
    const std::int64_t I = top.intermediate_size;
    const bool full = top.IsFull(layer_idx);
    const int nh = top.num_attention_heads;
    const int hd = top.HeadDim(layer_idx);
    const int nkv = top.KvHeads(layer_idx);
    const int q_rows = top.QRows(layer_idx);
    const int kv_rows = top.KvRows(layer_idx);
    const float eps = top.rms_norm_eps;
    // RoPE: a sliding layer rotates the whole head; a global layer rotates the
    // first partial_rotary_factor of it, paired across the FULL head dim.
    const int rot_angles =
        full ? static_cast<int>(top.partial_rotary_factor * hd / 2) : hd / 2;
    const double theta = top.RopeTheta(layer_idx);
    const int window = top.Window(layer_idx);

    // layer_scalar is a constant weight: read the host mirror taken at Load
    // rather than a device readback + full stream sync per layer per forward.
    float layer_scalar = 1.0f;
    if (layer_idx >= 0 &&
        static_cast<size_t>(layer_idx) < pw.h_layer_scalar.size())
        layer_scalar = pw.h_layer_scalar[static_cast<size_t>(layer_idx)];

    cudaError_t e = RunGemmaDenseRMSNorm(stream, ws.x, lw.d_input_norm,
                                         ws.normed, seq,
                                         static_cast<int>(H),
                                         static_cast<float>(eps));
    if (e != cudaSuccess) { GD_FAIL_LINE(e); return e; }

    // q / k / v projections (all NVFP4 through the neutral layer).
    e = RunLsxNvfp4Gemm(stream, ws.normed, lw.attn_q.d_packed,
                        lw.attn_q.d_scales, lw.attn_q.global_scale, ws.q, seq,
                        static_cast<int>(H), q_rows);
    if (e != cudaSuccess) { GD_FAIL_LINE(e); return e; }
    e = RunLsxNvfp4Gemm(stream, ws.normed, lw.attn_k.d_packed,
                        lw.attn_k.d_scales, lw.attn_k.global_scale, ws.k, seq,
                        static_cast<int>(H), kv_rows);
    if (e != cudaSuccess) { GD_FAIL_LINE(e); return e; }
    if (lw.HasV()) {
        e = RunLsxNvfp4Gemm(stream, ws.normed, lw.attn_v.d_packed,
                            lw.attn_v.d_scales, lw.attn_v.global_scale, ws.v,
                            seq, static_cast<int>(H), kv_rows);
        if (e != cudaSuccess) { GD_FAIL_LINE(e); return e; }
    } else {
        // attention_k_eq_v on the global layers: V IS the raw k_proj output
        // (upstream takes `value_states = key_states` BEFORE k_norm/RoPE).
        e = cudaMemcpyAsync(ws.v, ws.k,
                            static_cast<size_t>(seq) * kv_rows * sizeof(__half),
                            cudaMemcpyDeviceToDevice, stream);
        if (e != cudaSuccess) { GD_FAIL_LINE(e); return e; }
    }

    // q_norm / k_norm (weighted) and v_norm (plain). Both act PER HEAD.
    e = RunGemmaDenseHeadNorm(stream, ws.q, lw.d_q_norm, ws.q, seq, nh, hd,
                              static_cast<float>(eps));
    if (e != cudaSuccess) { GD_FAIL_LINE(e); return e; }
    e = RunGemmaDenseHeadNorm(stream, ws.k, lw.d_k_norm, ws.k, seq, nkv, hd,
                              static_cast<float>(eps));
    if (e != cudaSuccess) { GD_FAIL_LINE(e); return e; }
    e = RunGemmaDenseHeadNorm(stream, ws.v, nullptr, ws.v, seq, nkv, hd,
                              static_cast<float>(eps));
    if (e != cudaSuccess) { GD_FAIL_LINE(e); return e; }

    // RoPE on q and k only; values keep their (unrotated) v_norm form.
    {
        std::vector<std::int32_t> pos(static_cast<size_t>(seq));
        for (int i = 0; i < seq; ++i)
            pos[static_cast<size_t>(i)] = pos_base + i;
        e = cudaMemcpyAsync(ws.pos, pos.data(),
                            sizeof(std::int32_t) * static_cast<size_t>(seq),
                            cudaMemcpyHostToDevice, stream);
        if (e != cudaSuccess) { GD_FAIL_LINE(e); return e; }
        e = RunGemmaDenseRope(stream, ws.pos, ws.q, ws.k, seq, nh, nkv, hd,
                              rot_angles, theta);
        if (e != cudaSuccess) { GD_FAIL_LINE(e); return e; }
    }

    // Append this chunk's k/v rows to the layer cache. The cache is POS-MAJOR
    // (KvCacheLayout::kPosMajor), so one token's row is kv_heads*head_dim
    // contiguous halves: copying only head_stride (== head_dim) would leave
    // every KV head but the first reading uninitialized memory.
    const std::size_t row_bytes = static_cast<std::size_t>(kgeo.kv_heads) *
                                  static_cast<std::size_t>(kgeo.head_dim) *
                                  sizeof(__half);
    e = cudaMemcpyAsync(dk + kgeo.Offset(0, 0, pos_base), ws.k,
                        row_bytes * static_cast<size_t>(seq),
                        cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess) { GD_FAIL_LINE(e); return e; }
    e = cudaMemcpyAsync(dv + vgeo.Offset(0, 0, pos_base), ws.v,
                        row_bytes * static_cast<size_t>(seq),
                        cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess) { GD_FAIL_LINE(e); return e; }

    // Attention. Decode (seq == 1) takes the split-KV path for long context;
    // short contexts use the unsplit kernel.
    if (seq == 1) {
        int splits = 1;
        const int span = window > 0 ? std::min(window, pos_base + 1)
                                   : pos_base + 1;
        if (span >= kAttnSplitMinKv)
            splits = (span + kAttnSplitKeys - 1) / kAttnSplitKeys;
        if (splits <= 1) {
            e = RunGemmaDenseAttention(stream, ws.q, dk, dv, ws.attn, seq, nh,
                                       nkv, hd, window, frozen::kAttentionScale,
                                       pos_base, kgeo, vgeo);
        } else {
            e = AttentionSplit(stream, ws.q, dk, dv, ws.attn, ws.attn_part, seq,
                               nh, nkv, hd, window, frozen::kAttentionScale,
                               pos_base, splits, kgeo, vgeo);
        }
    } else {
        e = RunGemmaDenseAttention(stream, ws.q, dk, dv, ws.attn, seq, nh, nkv,
                                   hd, window, frozen::kAttentionScale,
                                   pos_base, kgeo, vgeo);
    }
    if (e != cudaSuccess) { GD_FAIL_LINE(e); return e; }

    // o_proj + the sandwich post-norm + residual.
    e = RunLsxNvfp4Gemm(stream, ws.attn, lw.attn_o.d_packed, lw.attn_o.d_scales,
                        lw.attn_o.global_scale, ws.proj, seq, q_rows,
                        static_cast<int>(H));
    if (e != cudaSuccess) { GD_FAIL_LINE(e); return e; }
    e = RunGemmaDenseRMSNorm(stream, ws.proj, lw.d_post_norm, ws.normed, seq,
                             static_cast<int>(H), static_cast<float>(eps));
    if (e != cudaSuccess) { GD_FAIL_LINE(e); return e; }
    {
        const std::int64_t numel = static_cast<std::int64_t>(seq) * H;
        kernels::AddInPlaceKernel<__half>
            <<<static_cast<unsigned>((numel + kThreads - 1) / kThreads),
               kThreads, 0, stream>>>(ws.normed, ws.x, numel);
        e = cudaGetLastError();
        if (e != cudaSuccess) { GD_FAIL_LINE(e); return e; }
    }

    // Dense MLP: GeGLU, then the post-FFN norm + residual.
    e = RunGemmaDenseRMSNorm(stream, ws.x, lw.d_pre_ffn, ws.normed, seq,
                             static_cast<int>(H), static_cast<float>(eps));
    if (e != cudaSuccess) { GD_FAIL_LINE(e); return e; }
    // gate and up project the SAME row; they are kept in two [seq, I] buffers
    // because the neutral GEMM writes contiguous rows (the GeGLU kernel takes
    // the two projections separately).
    e = RunLsxNvfp4Gemm(stream, ws.normed, lw.mlp_gate.d_packed,
                        lw.mlp_gate.d_scales, lw.mlp_gate.global_scale,
                        ws.gate_up, seq, static_cast<int>(H),
                        static_cast<int>(I));
    if (e != cudaSuccess) { GD_FAIL_LINE(e); return e; }
    e = RunLsxNvfp4Gemm(stream, ws.normed, lw.mlp_up.d_packed,
                        lw.mlp_up.d_scales, lw.mlp_up.global_scale, ws.act,
                        seq, static_cast<int>(H), static_cast<int>(I));
    if (e != cudaSuccess) { GD_FAIL_LINE(e); return e; }
    e = RunGemmaDenseGeGLU(stream, ws.gate_up, ws.act, ws.gate_up,
                           static_cast<std::int64_t>(seq) * I);
    if (e != cudaSuccess) { GD_FAIL_LINE(e); return e; }
    // The down projection consumes the GeGLU result (written back into
    // ws.gate_up), NOT the raw up projection.
    e = RunLsxNvfp4Gemm(stream, ws.gate_up, lw.mlp_down.d_packed,
                        lw.mlp_down.d_scales, lw.mlp_down.global_scale,
                        ws.ffn, seq, static_cast<int>(I), static_cast<int>(H));
    if (e != cudaSuccess) { GD_FAIL_LINE(e); return e; }
    e = RunGemmaDenseRMSNorm(stream, ws.ffn, lw.d_post_ffn, ws.proj, seq,
                             static_cast<int>(H), static_cast<float>(eps));
    if (e != cudaSuccess) { GD_FAIL_LINE(e); return e; }
    {
        const std::int64_t numel = static_cast<std::int64_t>(seq) * H;
        kernels::AddInPlaceKernel<__half>
            <<<static_cast<unsigned>((numel + kThreads - 1) / kThreads),
               kThreads, 0, stream>>>(ws.proj, ws.x, numel);
        if (e == cudaSuccess) e = cudaGetLastError();
        if (e != cudaSuccess) { GD_FAIL_LINE(e); return e; }
        // hidden *= layer_scalar (upstream's nn.Buffer, applied last).
        gemma_dense_scale_kernel<<<static_cast<unsigned>(
                                       (numel + kThreads - 1) / kThreads),
                                   kThreads, 0, stream>>>(ws.x, layer_scalar,
                                                          numel);
        return cudaGetLastError();
    }
}

// Final norm + tied softcapped head + argmax; returns the sampled token.
cudaError_t SampleNext(cudaStream_t stream,
                       GemmaDenseGenContext::Workspace &ws,
                       const GemmaDensePinnedWeights &pw, int seq, float softcap)
{
    const std::int64_t H = pw.top.hidden_size;
    const std::int64_t V = pw.top.vocab_size;
    cudaError_t e = RunGemmaDenseRMSNorm(stream, ws.x, pw.d_final_norm,
                                         ws.normed, seq, static_cast<int>(H),
                                         pw.top.rms_norm_eps);
    if (e != cudaSuccess) { GD_FAIL_LINE(e); return e; }
    const __half *last = ws.normed + static_cast<std::int64_t>(seq - 1) * H;
    e = RunGemmaDenseLogitsArgmax(stream, pw.embed.d_packed,
                                  pw.embed.d_scales, pw.embed.global_scale,
                                  last, static_cast<int>(V),
                                  static_cast<int>(H), softcap, ws.argmax,
                                  ws.maxv, ws.logits, ws.logits_h);
    return e;
}

} // namespace

// ── Generation ──────────────────────────────────────────────────────────────

cudaError_t RunGemmaDenseGeneration(
    cudaStream_t stream, const GemmaDensePinnedWeights &pw,
    const std::vector<std::int32_t> &input_ids, int max_gen_tokens,
    const std::vector<int> &eos_ids, GemmaDenseGenerationResult &out,
    GemmaDenseKvCache &kv, GemmaDenseGenContext *ctx,
    lsxcommon::sample::ISampler &sampler,
    const lsxcommon::GenerationStream *gs)
{
    const auto t_start = std::chrono::steady_clock::now();
    out = GemmaDenseGenerationResult{};
    const auto &top = pw.top;
    const int H = top.hidden_size;
    const int V = top.vocab_size;
    const int P = static_cast<int>(input_ids.size());
    if (P <= 0) {
        out.error_msg = "empty prompt";
        return cudaErrorInvalidValue;
    }
    if (top.num_layers <= 0 ||
        pw.layers.size() < static_cast<size_t>(top.num_layers)) {
        out.error_msg = "model has no loaded layers";
        return cudaErrorInvalidValue;
    }
    if (!pw.embed.d_packed || !pw.d_final_norm) {
        out.error_msg = "embed/head/final-norm missing";
        return cudaErrorInvalidValue;
    }
    if (!ctx) {
        out.error_msg = "no generation context";
        return cudaErrorInvalidValue;
    }
    if (!kv.allocated()) {
        out.error_msg = "kv cache not allocated";
        return cudaErrorInvalidValue;
    }

    size_t q_rows_max = 0, kv_rows_max = 0, hd_max = 0;
    for (int l = 0; l < top.num_layers; ++l) {
        q_rows_max = std::max<size_t>(q_rows_max,
                                      static_cast<size_t>(top.QRows(l)));
        kv_rows_max = std::max<size_t>(kv_rows_max,
                                       static_cast<size_t>(top.KvRows(l)));
        hd_max = std::max<size_t>(hd_max, static_cast<size_t>(top.HeadDim(l)));
    }

    auto &ws = ctx->ws;
    const size_t need = static_cast<size_t>(P) +
                        static_cast<size_t>(std::max(max_gen_tokens, 1)) + 8;
    if (!ctx->ws_ready || ws.seq_cap < need || ws.q_rows_cap < q_rows_max ||
        ws.kv_rows_cap < kv_rows_max) {
        const cudaError_t e = ws.Alloc(H, top.intermediate_size, V,
                                       std::max<size_t>(need, 64), q_rows_max,
                                       kv_rows_max,
                                       static_cast<size_t>(top.num_attention_heads),
                                       hd_max);
        if (e != cudaSuccess) {
            out.error_msg = "workspace alloc failed";
            return e;
        }
        ctx->ws_ready = true;
    }

    const float softcap = top.final_logit_softcapping;

    auto run_layers = [&](int seq, int pos_base) -> cudaError_t {
        for (int l = 0; l < top.num_layers; ++l) {
            const auto &lw = pw.layers[static_cast<size_t>(l)];
            const cudaError_t e = ForwardLayer(
                stream, pw, lw, l, ws, seq, pos_base,
                kv.dk[static_cast<size_t>(l)], kv.dv[static_cast<size_t>(l)],
                kv.geo[static_cast<size_t>(l)], kv.geo[static_cast<size_t>(l)]);
            if (e != cudaSuccess) { GD_FAIL_LINE(e); return e; }
        }
        return cudaSuccess;
    };

    const auto is_eos = [&](std::int32_t t) {
        for (const int e : eos_ids)
            if (t == e) return true;
        return false;
    };

    // ── Prefill ──
    {
        const int seq = P;
        // Prefill stays on the bit-exact scalar path by choice, not by
        // necessity: the shared TC GEMM's partial-M bug that used to force
        // this is fixed (lsx_tc_gemm.cu now applies the warp row group), but
        // running gemma's prefill through TC changes its numeric class --
        // the first token and the tool round-trip both move, and this
        // family's FSI/CODE/DET goldens were captured on the scalar path.
        const bool tc_saved = lsx_nvfp4::LsxTcFastMath();
        lsx_nvfp4::SetLsxTcFastMath(false);
        cudaError_t e = cudaMemcpyAsync(ws.ids, input_ids.data(),
                                        sizeof(std::int32_t) *
                                            static_cast<size_t>(seq),
                                        cudaMemcpyHostToDevice, stream);
        if (e == cudaSuccess)
            e = RunGemmaDenseEmbedGatherNvfp4(stream, pw.embed.d_packed,
                                              pw.embed.d_scales,
                                              pw.embed.global_scale, ws.ids,
                                              ws.x, seq, V, H,
                                              frozen::kEmbedScale);
        if (e == cudaSuccess) e = run_layers(seq, 0);
        std::int32_t tok = -1;
        if (e == cudaSuccess) e = SampleNext(stream, ws, pw, seq, softcap);
        if (e == cudaSuccess && !sampler.IsGreedy())
            e = sampler.Sample(stream, ws.logits, static_cast<int>(V),
                               lsxcommon::sample::LogitType::kF32, ws.argmax,
                               ws.maxv);
        if (e == cudaSuccess)
            e = cudaMemcpyAsync(&tok, ws.argmax, sizeof(std::int32_t),
                                cudaMemcpyDeviceToHost, stream);
        if (e == cudaSuccess) e = cudaStreamSynchronize(stream);
        lsx_nvfp4::SetLsxTcFastMath(tc_saved);
        if (e != cudaSuccess) {
            out.error_msg = "prefill failed";
            return e;
        }
        if (tok < 0 || tok >= V) {
            out.error_msg = "sampled token out of range";
            return cudaErrorInvalidValue;
        }
        out.tokens.push_back(tok);
        if (gs) gs->Emit(static_cast<std::uint32_t>(tok), 0u);
    }

    out.ttft_ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - t_start)
                      .count();
    if (is_eos(out.tokens.front())) out.eos_terminated = true;

    // ── Decode ──
    int pos_base = P;
    for (int step = 1;
         step < max_gen_tokens && !out.eos_terminated; ++step) {
        if (gs && gs->Aborted()) break;
        const std::int32_t next = out.tokens.back();
        cudaError_t e = cudaMemcpyAsync(ws.ids, &next, sizeof(std::int32_t),
                                        cudaMemcpyHostToDevice, stream);
        if (e == cudaSuccess)
            e = RunGemmaDenseEmbedGatherNvfp4(stream, pw.embed.d_packed,
                                              pw.embed.d_scales,
                                              pw.embed.global_scale, ws.ids,
                                              ws.x, 1, V, H,
                                              frozen::kEmbedScale);
        if (e == cudaSuccess) e = run_layers(1, pos_base);
        std::int32_t tok = -1;
        if (e == cudaSuccess) e = SampleNext(stream, ws, pw, 1, softcap);
        if (e == cudaSuccess && !sampler.IsGreedy())
            e = sampler.Sample(stream, ws.logits, static_cast<int>(V),
                               lsxcommon::sample::LogitType::kF32, ws.argmax,
                               ws.maxv);
        if (e == cudaSuccess)
            e = cudaMemcpyAsync(&tok, ws.argmax, sizeof(std::int32_t),
                                cudaMemcpyDeviceToHost, stream);
        if (e == cudaSuccess) e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess) {
            out.error_msg = "decode failed";
            return e;
        }
        if (tok < 0 || tok >= V) {
            out.error_msg = "sampled token out of range";
            return cudaErrorInvalidValue;
        }
        ++pos_base;
        out.tokens.push_back(tok);
        if (gs)
            gs->Emit(static_cast<std::uint32_t>(tok),
                     static_cast<std::uint32_t>(out.tokens.size() - 1));
        if (is_eos(tok)) out.eos_terminated = true;
    }

    // Final logits row (determinism evidence for SelfCheck).
    {
        std::vector<float> h(static_cast<size_t>(V));
        cudaError_t e = cudaMemcpyAsync(h.data(), ws.logits,
                                        static_cast<size_t>(V) * sizeof(float),
                                        cudaMemcpyDeviceToHost, stream);
        if (e == cudaSuccess) e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess) {
            out.error_msg = "final logits copy failed";
            return e;
        }
        out.final_logits = std::move(h);
    }

    out.ok = !out.tokens.empty();
    if (!out.ok && out.error_msg.empty())
        out.error_msg = "no tokens generated";
    out.total_ms = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - t_start)
                       .count();
    out.resident_bytes = pw.resident_bytes;
    return cudaSuccess;
}

} // namespace gemma_dense
} // namespace lsxcommon
