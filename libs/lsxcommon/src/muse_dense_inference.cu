// Muse-Glimmer-30B CUDA runtime — Model Ingestion Protocol steps 5/6.
//
// One stream, no per-token cudaDeviceSynchronize. Every dense projection goes
// through the model-neutral NVFP4 layer (lsx_nvfp4::RunLsxNvfp4Gemm); the FP16
// head through the neutral gemv layer. The family-local kernels are the
// centered norms, the weightless per-head QK norm + Q scale, the sliding-band
// / NoPE causal GQA attention, the sigmoid attention output gate, SwiGLU
// (silu), the FP16 embed gather and the softcapped FP16-head logits/argmax.
//
// Reference semantics (transformers/models/muse_glimmer, verified numerically):
//   x = x + post_attn(o_proj(sigmoid(gate_proj(n)) * attn(qk_norm(rope?(…)))))
//   x = x + post_ffn(down(silu(gate(pre(x))) * up(pre(x))))
// with four facts that are easy to get wrong and each has a test:
//   * the four per-layer norms are CENTERED (1+w); the final norm is plain w;
//   * the attention gate is elementwise sigmoid over q_rows, BEFORE o_proj;
//   * full-attention layers are NoPE (no RoPE) and attend the whole prefix;
//   * logits = cap * tanh(raw * output_multiplier / cap).

#include "lsxcommon/muse_dense_inference.h"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "lsxcommon/muse_dense_constants.h"
#include "lsxcommon/log.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/lsx_gemv.h"

#include "lsx_kernels_common.cuh"
#include "lsx_kernels_math.cuh"
#include "lsx_kernels_norm.cuh"
#include "lsx_kernels_act.cuh"

namespace lsxcommon {
namespace muse_dense {

using lsx_nvfp4::RunLsxNvfp4Gemm;

namespace {

constexpr int kThreads = 128;
constexpr int kNormThreads = 256;

#define MD_FAIL_LINE(e)                                                     \
    do {                                                                    \
        log::error("[muse_dense] failure at line " +                        \
                   std::to_string(__LINE__) + ": " +                        \
                   cudaGetErrorString(e));                                   \
    } while (0)

// ── Per-head weightless QK norm + Q scale ───────────────────────────────────
// One block per (token, head), reducing over head_dim. q heads get q_scale,
// k heads get 1.0. The shared-memory reduction is thread-0-only behind
// __syncthreads() (the sandbox rule); no atomics.
__global__ void muse_dense_qk_norm_kernel(__half *__restrict__ q,
                                          __half *__restrict__ k, int seq,
                                          int nh, int nkv, int hd, float eps,
                                          float q_scale)
{
    const int row = blockIdx.x;             // spans seq*(nh+nkv)
    const int total_q = seq * nh;
    const bool is_q = row < total_q;
    const int local = is_q ? row : row - total_q;
    const int heads = is_q ? nh : nkv;
    if (local >= seq * heads) return;
    __half *base = is_q ? q : k;
    __half *xr = base + static_cast<int64_t>(local) * hd;
    const float scale = is_q ? q_scale : 1.0f;
    const int tpb = blockDim.x;

    float sum = 0.f;
    for (int d = threadIdx.x; d < hd; d += tpb) {
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
        rsqrtf(__fadd_rn(__fdiv_rn(ws[0], static_cast<float>(hd)), eps));
    for (int d = threadIdx.x; d < hd; d += tpb) {
        const float v = __fmul_rn(__fmul_rn(__half2float(xr[d]), inv), scale);
        xr[d] = __float2half(v);
    }
}

// ── RoPE (whole head dim) ───────────────────────────────────────────────────
__global__ void muse_dense_rope_kernel(const int32_t *__restrict__ pos,
                                       __half *__restrict__ q,
                                       __half *__restrict__ k, int seq, int nh,
                                       int nkv, int hd, double theta)
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
        const float freq = static_cast<float>(
            std::pow(theta, -2.0 * static_cast<double>(i) / hd));
        const float ang = static_cast<float>(pos[t]) * freq;
        const float cc = cosf(ang);
        const float ss = sinf(ang);
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
// evaluated per key slot. KV addresses come exclusively from
// KvCacheGeometry::Offset. Structurally the gemma dense attention kernel with
// the standard 1/sqrt(hd) scale and the NoPE/sliding window passed per layer.
__global__ void muse_dense_attn_kernel(const __half *__restrict__ q,
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
    float *q_s = smem;
    float *o_s = smem + hd;
    float *sc = smem + 2 * hd;
    float *rd = sc + tpb;

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
            const __half *krow = kc + kgeom.Offset(0, kvh, start + tid);
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

// Split-KV (flash-decoding) variant: fixed contiguous chunks + fixed merge
// order => run-to-run deterministic.
constexpr int kAttnSplitKeys = 64;

__global__ void muse_dense_attn_split_kernel(
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
                const __half *vrow = vc + vgeom.Offset(0, kvh, lo + start + j);
                acc = __fmaf_rn(sc[j], __half2float(vrow[d]), acc);
            }
            o_s[d] = acc;
        }
        m = m_new;
        l = __fmaf_rn(l, rescale, l_new);
        __syncthreads();
    }

    pm[slot] = m;
    pl[slot] = l;
    for (int d = tid; d < hd; d += tpb) po[slot * hd + d] = o_s[d];
}

__global__ void muse_dense_attn_combine_kernel(
    const float *__restrict__ pm, const float *__restrict__ pl,
    const float *__restrict__ po, __half *__restrict__ out, int seq, int nh,
    int hd, int splits)
{
    const int base = blockIdx.x;            // (t*nh + h)
    const int d = threadIdx.x;
    if (d >= hd) return;
    float m = -CUDART_INF_F;
    for (int s = 0; s < splits; ++s) {
        const int64_t slot = static_cast<int64_t>(s) * seq * nh + base;
        m = max(m, pm[slot]);
    }
    float l = 0.f, a = 0.f;
    for (int s = 0; s < splits; ++s) {
        const int64_t slot = static_cast<int64_t>(s) * seq * nh + base;
        const float w = device::ExpfGlibc(__fsub_rn(pm[slot], m));
        l = __fadd_rn(l, __fmul_rn(pl[slot], w));
        a = __fadd_rn(a, __fmul_rn(po[slot * hd + d], w));
    }
    const int t = base / nh;
    const int h = base % nh;
    __half *orow = out + (static_cast<int64_t>(t) * nh + h) * hd;
    orow[d] = __float2half((l > 0.f) ? __fdiv_rn(a, l) : 0.f);
}

// ── Attention output gate ───────────────────────────────────────────────────
__global__ void muse_dense_attn_gate_kernel(const __half *__restrict__ attn,
                                            const __half *__restrict__ gate,
                                            __half *__restrict__ out, int64_t n)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
    if (i >= n) return;
    const float a = __half2float(attn[i]);
    const float g = __half2float(gate[i]);
    const float sig = __fdiv_rn(1.0f, __fadd_rn(1.0f, device::ExpfGlibc(-g)));
    out[i] = __float2half(__fmul_rn(a, sig));
}

// ── Softcapped logits + deterministic argmax ────────────────────────────────
constexpr int kLogitsArgMaxBlocks = 4096;

__global__ void muse_dense_softcap_kernel(const float *__restrict__ in,
                                          float *__restrict__ out, int vocab,
                                          float multiplier, float cap)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= vocab) return;
    out[i] = __fmul_rn(tanhf(__fmul_rn(in[i], multiplier) / cap), cap);
}

__global__ void muse_dense_argmax_blocks_kernel(
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
        if (val > best) { best = val; best_i = v; }
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
        blk_best[blockIdx.x] = sb[0];
        blk_idx[blockIdx.x] = si[0];
    }
}

__global__ void muse_dense_argmax_reduce_kernel(
    const float *__restrict__ blk_best, const std::int32_t *__restrict__ blk_idx,
    std::int32_t *__restrict__ d_argmax, float *__restrict__ d_maxv,
    int n_blocks)
{
    __shared__ float sb[kThreads];
    __shared__ int si[kThreads];
    float best = -CUDART_INF_F;
    int best_i = -1;
    for (int b = threadIdx.x; b < n_blocks; b += blockDim.x) {
        if (blk_best[b] > best) { best = blk_best[b]; best_i = blk_idx[b]; }
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
    if (threadIdx.x == 0) { *d_argmax = si[0]; *d_maxv = sb[0]; }
}

constexpr int kAttnSplitMinKv = 256;

} // namespace

// ── Public norm wrappers ────────────────────────────────────────────────────

cudaError_t RunMuseDenseCenteredNorm(cudaStream_t stream, const __half *x,
                                     const __half *w, __half *out, int rows,
                                     int hidden, float eps)
{
    if (!x || !w || !out || rows <= 0 || hidden <= 0)
        return cudaErrorInvalidValue;
    const int threads = std::min(hidden, kNormThreads);
    const size_t shmem = threads * sizeof(float) +
                         std::min(hidden * static_cast<int>(sizeof(__half)),
                                  48 * 1024);
    kernels::RmsNormTreeKernel<__half, kernels::NormWeight::Offset1PlusW>
        <<<rows, threads, shmem, stream>>>(x, w, out, rows, hidden, eps);
    return cudaGetLastError();
}

cudaError_t RunMuseDensePlainNorm(cudaStream_t stream, const __half *x,
                                  const __half *w, __half *out, int rows,
                                  int hidden, float eps)
{
    if (!x || !out || rows <= 0 || hidden <= 0) return cudaErrorInvalidValue;
    const int threads = std::min(hidden, kNormThreads);
    const size_t shmem = threads * sizeof(float) +
                         std::min(hidden * static_cast<int>(sizeof(__half)),
                                  48 * 1024);
    if (w) {
        kernels::RmsNormTreeKernel<__half, kernels::NormWeight::Direct>
            <<<rows, threads, shmem, stream>>>(x, w, out, rows, hidden, eps);
    } else {
        kernels::RmsNormTreeKernel<__half, kernels::NormWeight::None>
            <<<rows, threads, shmem, stream>>>(x, nullptr, out, rows, hidden,
                                               eps);
    }
    return cudaGetLastError();
}

cudaError_t RunMuseDenseQkNormScale(cudaStream_t stream, __half *q, __half *k,
                                    int seq, int nh, int nkv, int hd, float eps,
                                    float q_scale)
{
    if (!q || !k || seq <= 0 || nh <= 0 || nkv <= 0 || hd <= 0)
        return cudaErrorInvalidValue;
    const int rows = seq * (nh + nkv);
    muse_dense_qk_norm_kernel<<<rows, kNormThreads, 0, stream>>>(
        q, k, seq, nh, nkv, hd, eps, q_scale);
    return cudaGetLastError();
}

cudaError_t RunMuseDenseRope(cudaStream_t stream, const int32_t *d_pos,
                             __half *q, __half *k, int seq, int nh, int nkv,
                             int hd, double theta)
{
    if (!d_pos || !q || !k || seq <= 0 || nh <= 0 || nkv <= 0 || hd <= 0 ||
        hd % 2 != 0)
        return cudaErrorInvalidValue;
    const int64_t total = static_cast<int64_t>(seq) * (nh + nkv) * (hd / 2);
    const int blocks =
        static_cast<int>((total + kThreads * 4 - 1) / (kThreads * 4));
    muse_dense_rope_kernel<<<blocks > 0 ? blocks : 1, kThreads, 0, stream>>>(
        d_pos, q, k, seq, nh, nkv, hd, theta);
    return cudaGetLastError();
}

cudaError_t RunMuseDenseAttention(cudaStream_t stream, const __half *q,
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
    muse_dense_attn_kernel<<<seq * nh, kThreads, shmem, stream>>>(
        q, k, v, out, seq, nh, nkv, hd, window, scale, pos_base, kgeom, vgeom);
    return cudaGetLastError();
}

cudaError_t RunMuseDenseAttentionSplit(cudaStream_t stream, const __half *q,
                                       const __half *k, const __half *v,
                                       __half *out, float *partials, int seq,
                                       int nh, int nkv, int hd, int window,
                                       float scale, int pos_base, int splits,
                                       KvCacheGeometry kgeom,
                                       KvCacheGeometry vgeom)
{
    if (!q || !k || !v || !out || !partials || seq <= 0 || nh <= 0 ||
        nkv <= 0 || hd <= 0 || nh % nkv != 0)
        return cudaErrorInvalidValue;
    if (splits < 1) splits = 1;
    if (splits > kMuseDenseAttnSplitMax) splits = kMuseDenseAttnSplitMax;
    const size_t shmem =
        (2 * static_cast<size_t>(hd) + 2 * static_cast<size_t>(kThreads)) *
        sizeof(float);
    const int64_t slots = static_cast<int64_t>(splits) * seq * nh;
    float *pm = partials;
    float *pl = pm + slots;
    float *po = pl + slots;
    const dim3 grid(static_cast<unsigned>(seq * nh),
                    static_cast<unsigned>(splits));
    muse_dense_attn_split_kernel<<<grid, kThreads, shmem, stream>>>(
        q, k, v, pm, pl, po, seq, nh, nkv, hd, window, scale, pos_base, splits,
        kgeom, vgeom);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { MD_FAIL_LINE(e); return e; }
    muse_dense_attn_combine_kernel<<<seq * nh, hd, 0, stream>>>(
        pm, pl, po, out, seq, nh, hd, splits);
    return cudaGetLastError();
}

cudaError_t RunMuseDenseAttnGate(cudaStream_t stream, const __half *attn,
                                 const __half *gate, __half *out, std::int64_t n)
{
    if (!attn || !gate || !out || n <= 0) return cudaErrorInvalidValue;
    const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
    muse_dense_attn_gate_kernel<<<blocks, kThreads, 0, stream>>>(attn, gate, out,
                                                                 n);
    return cudaGetLastError();
}

cudaError_t RunMuseDenseSwiGLU(cudaStream_t stream, const __half *gate,
                               const __half *up, __half *out, std::int64_t n)
{
    if (!gate || !up || !out || n <= 0) return cudaErrorInvalidValue;
    const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
    // silu(g) * u = (g / (1 + e^-g)) * u — the neutral division-form SwiGLU,
    // no clamp (Muse has no activation softcap).
    kernels::SwigluKernel<__half, kernels::ExpFlavor::Glibc, /*Clamp=*/false>
        <<<blocks, kThreads, 0, stream>>>(gate, up, out, n, 0.f);
    return cudaGetLastError();
}

cudaError_t RunMuseDenseEmbedGatherFp16(cudaStream_t stream, const __half *embed,
                                        const std::int32_t *ids, __half *out,
                                        int seq, int hidden)
{
    if (!embed || !ids || !out || seq <= 0 || hidden <= 0)
        return cudaErrorInvalidValue;
    const int threads = 256;
    const int blocks =
        static_cast<int>((static_cast<int64_t>(seq) * hidden + threads - 1) /
                         threads);
    kernels::EmbedGatherKernel<__half><<<blocks, threads, 0, stream>>>(
        embed, ids, out, seq, hidden);
    return cudaGetLastError();
}

cudaError_t RunMuseDenseLogitsArgmax(cudaStream_t stream, const __half *lm_head,
                                     const __half *x, int vocab, int hidden,
                                     float multiplier, float cap,
                                     std::int32_t *d_argmax, float *d_maxlogit,
                                     float *d_logits)
{
    if (!lm_head || !x || !d_argmax || !d_maxlogit || !d_logits ||
        vocab <= 0 || hidden <= 0 || cap <= 0.f)
        return cudaErrorInvalidValue;
    // FP16 head GEMV through the neutral device-independent layer: raw logits
    // into the FP32 row (alpha 1, beta 0), then the multiplier + softcap.
    cudaError_t e = gemv::RunLxGemmVec(stream, lm_head, x, 1, vocab, hidden,
                                       gemv::LxDtype::kF16, gemv::LxDtype::kF16,
                                       1.0f, d_logits, gemv::LxDtype::kF32,
                                       0.0f);
    if (e != cudaSuccess) { MD_FAIL_LINE(e); return e; }
    muse_dense_softcap_kernel<<<(vocab + kThreads - 1) / kThreads, kThreads, 0,
                                stream>>>(d_logits, d_logits, vocab, multiplier,
                                          cap);
    e = cudaGetLastError();
    if (e != cudaSuccess) { MD_FAIL_LINE(e); return e; }

    const int blocks = (vocab + kThreads - 1) / kThreads;
    float *blk_best = d_logits + vocab;
    std::int32_t *blk_idx = reinterpret_cast<std::int32_t *>(blk_best + blocks);
    if (blocks > kLogitsArgMaxBlocks) return cudaErrorInvalidValue;
    muse_dense_argmax_blocks_kernel<<<blocks, kThreads, 0, stream>>>(
        d_logits, blk_best, blk_idx, vocab);
    e = cudaGetLastError();
    if (e != cudaSuccess) { MD_FAIL_LINE(e); return e; }
    muse_dense_argmax_reduce_kernel<<<1, kThreads, 0, stream>>>(
        blk_best, blk_idx, d_argmax, d_maxlogit, blocks);
    return cudaGetLastError();
}

// ── Workspace ───────────────────────────────────────────────────────────────

void MuseDenseGenContext::Workspace::Free()
{
    auto F = [](void **p) {
        if (*p) { cudaFree(*p); *p = nullptr; }
    };
    F((void **)&x);
    F((void **)&normed);
    F((void **)&q);
    F((void **)&k);
    F((void **)&v);
    F((void **)&gate);
    F((void **)&attn);
    F((void **)&proj);
    F((void **)&mlp_gate);
    F((void **)&mlp_up);
    F((void **)&ffn);
    F((void **)&logits);
    F((void **)&ids);
    F((void **)&pos);
    F((void **)&argmax);
    F((void **)&maxv);
    F((void **)&attn_part);
    seq_cap = q_rows_cap = kv_rows_cap = 0;
}

cudaError_t MuseDenseGenContext::Workspace::Alloc(std::int64_t H, std::int64_t I,
                                                  std::int64_t V, size_t S,
                                                  size_t q_rows, size_t kv_rows,
                                                  size_t nh, size_t hd)
{
    Free();
    cudaError_t e = cudaSuccess;
    auto A = [&](void **p, size_t bytes) { return cudaMalloc(p, bytes); };
    if ((e = A((void **)&x, S * H * 2)) != cudaSuccess) return e;
    if ((e = A((void **)&normed, S * H * 2)) != cudaSuccess) return e;
    if ((e = A((void **)&q, S * q_rows * 2)) != cudaSuccess) return e;
    if ((e = A((void **)&k, S * kv_rows * 2)) != cudaSuccess) return e;
    if ((e = A((void **)&v, S * kv_rows * 2)) != cudaSuccess) return e;
    if ((e = A((void **)&gate, S * q_rows * 2)) != cudaSuccess) return e;
    if ((e = A((void **)&attn, S * q_rows * 2)) != cudaSuccess) return e;
    if ((e = A((void **)&proj, S * H * 2)) != cudaSuccess) return e;
    if ((e = A((void **)&mlp_gate, S * I * 2)) != cudaSuccess) return e;
    if ((e = A((void **)&mlp_up, S * I * 2)) != cudaSuccess) return e;
    if ((e = A((void **)&ffn, S * H * 2)) != cudaSuccess) return e;
    const size_t argmax_tail =
        static_cast<size_t>((V + kThreads - 1) / kThreads) * 2;
    if ((e = A((void **)&logits, (static_cast<size_t>(V) + argmax_tail) * 4)) !=
        cudaSuccess)
        return e;
    if ((e = A((void **)&ids, S * sizeof(std::int32_t))) != cudaSuccess) return e;
    if ((e = A((void **)&pos, S * sizeof(std::int32_t))) != cudaSuccess) return e;
    if ((e = A((void **)&argmax, 4)) != cudaSuccess) return e;
    if ((e = A((void **)&maxv, 4)) != cudaSuccess) return e;
    const size_t sp_seq = std::min<size_t>(S, 8);
    if ((e = A((void **)&attn_part,
               static_cast<size_t>(kMuseDenseAttnSplitMax) * sp_seq * nh *
                   (hd + 2) * sizeof(float))) != cudaSuccess)
        return e;
    seq_cap = S;
    q_rows_cap = q_rows;
    kv_rows_cap = kv_rows;
    return cudaSuccess;
}

MuseDenseGenContext *CreateMuseDenseGenContext()
{
    return new MuseDenseGenContext();
}

void DestroyMuseDenseGenContext(MuseDenseGenContext *ctx) { delete ctx; }

// ── Layer / generation ──────────────────────────────────────────────────────

namespace {

// One layer, `seq` tokens, writing k/v rows at [pos_base, pos_base + seq).
//
//   n    = centered_norm(x, input_norm)                  # eps rms_norm_eps
//   q,k,v = proj(n); weightless qk_norm; Q *= 3.87
//   sliding layers: RoPE(q,k); full (NoPE) layers: NO RoPE
//   a    = attention(scale 1/sqrt(hd), sliding band / full prefix)
//   a   *= sigmoid(gate_proj(n))                          # BEFORE o_proj
//   x   += centered_norm(o_proj(a), post_attn)            # eps post_norm_eps
//   n2   = centered_norm(x, pre_ffn)                      # eps rms_norm_eps
//   x   += centered_norm(down(silu(gate(n2))*up(n2)), post_ffn)  # post eps
cudaError_t ForwardLayer(cudaStream_t stream,
                         const MuseDensePinnedWeights &pw,
                         const MuseDenseLayerWeights &lw, int layer_idx,
                         MuseDenseGenContext::Workspace &ws, int seq,
                         int pos_base, __half *dk, __half *dv,
                         KvCacheGeometry kgeo, KvCacheGeometry vgeo)
{
    const auto &top = pw.top;
    const std::int64_t H = top.hidden_size;
    const std::int64_t I = top.intermediate_size;
    const bool full = top.IsFull(layer_idx);
    const int nh = top.num_attention_heads;
    const int nkv = top.num_key_value_heads;
    const int hd = top.head_dim;
    const int q_rows = top.QRows();
    const int kv_rows = top.KvRows();
    const float pre_eps = top.rms_norm_eps;
    const float post_eps = top.post_norm_eps;
    const double theta = top.RopeTheta(layer_idx);
    const int window = top.Window(layer_idx);

    cudaError_t e = RunMuseDenseCenteredNorm(stream, ws.x, lw.d_input_norm,
                                             ws.normed, seq, static_cast<int>(H),
                                             pre_eps);
    if (e != cudaSuccess) { MD_FAIL_LINE(e); return e; }

    // q / k / v / gate projections (all NVFP4 through the neutral layer). The
    // gate projects the SAME normed input as q/k/v.
    e = RunLsxNvfp4Gemm(stream, ws.normed, lw.attn_q.d_packed,
                        lw.attn_q.d_scales, lw.attn_q.global_scale, ws.q, seq,
                        static_cast<int>(H), q_rows);
    if (e != cudaSuccess) { MD_FAIL_LINE(e); return e; }
    e = RunLsxNvfp4Gemm(stream, ws.normed, lw.attn_k.d_packed,
                        lw.attn_k.d_scales, lw.attn_k.global_scale, ws.k, seq,
                        static_cast<int>(H), kv_rows);
    if (e != cudaSuccess) { MD_FAIL_LINE(e); return e; }
    e = RunLsxNvfp4Gemm(stream, ws.normed, lw.attn_v.d_packed,
                        lw.attn_v.d_scales, lw.attn_v.global_scale, ws.v, seq,
                        static_cast<int>(H), kv_rows);
    if (e != cudaSuccess) { MD_FAIL_LINE(e); return e; }
    e = RunLsxNvfp4Gemm(stream, ws.normed, lw.attn_gate.d_packed,
                        lw.attn_gate.d_scales, lw.attn_gate.global_scale,
                        ws.gate, seq, static_cast<int>(H), q_rows);
    if (e != cudaSuccess) { MD_FAIL_LINE(e); return e; }

    // Weightless per-head QK norm; Q gets the 3.87 scale.
    e = RunMuseDenseQkNormScale(stream, ws.q, ws.k, seq, nh, nkv, hd, pre_eps,
                                top.qk_scale_factor);
    if (e != cudaSuccess) { MD_FAIL_LINE(e); return e; }

    // RoPE on q/k ONLY on the sliding layers; the full (NoPE) layers skip it.
    if (!full && theta > 0.0) {
        std::vector<std::int32_t> pos(static_cast<size_t>(seq));
        for (int i = 0; i < seq; ++i)
            pos[static_cast<size_t>(i)] = pos_base + i;
        e = cudaMemcpyAsync(ws.pos, pos.data(),
                            sizeof(std::int32_t) * static_cast<size_t>(seq),
                            cudaMemcpyHostToDevice, stream);
        if (e != cudaSuccess) { MD_FAIL_LINE(e); return e; }
        e = RunMuseDenseRope(stream, ws.pos, ws.q, ws.k, seq, nh, nkv, hd,
                             theta);
        if (e != cudaSuccess) { MD_FAIL_LINE(e); return e; }
    }

    // Append this chunk's k/v rows to the pos-major cache (one token's row is
    // kv_heads*head_dim contiguous halves).
    const std::size_t row_bytes = static_cast<std::size_t>(kgeo.kv_heads) *
                                  static_cast<std::size_t>(kgeo.head_dim) *
                                  sizeof(__half);
    e = cudaMemcpyAsync(dk + kgeo.Offset(0, 0, pos_base), ws.k,
                        row_bytes * static_cast<size_t>(seq),
                        cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess) { MD_FAIL_LINE(e); return e; }
    e = cudaMemcpyAsync(dv + vgeo.Offset(0, 0, pos_base), ws.v,
                        row_bytes * static_cast<size_t>(seq),
                        cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess) { MD_FAIL_LINE(e); return e; }

    // Attention (decode seq==1 takes split-KV for long context).
    if (seq == 1) {
        int splits = 1;
        const int span = window > 0 ? std::min(window, pos_base + 1)
                                    : pos_base + 1;
        if (span >= kAttnSplitMinKv)
            splits = (span + kAttnSplitKeys - 1) / kAttnSplitKeys;
        if (splits <= 1) {
            e = RunMuseDenseAttention(stream, ws.q, dk, dv, ws.attn, seq, nh,
                                      nkv, hd, window, frozen::kAttentionScale,
                                      pos_base, kgeo, vgeo);
        } else {
            e = RunMuseDenseAttentionSplit(stream, ws.q, dk, dv, ws.attn,
                                           ws.attn_part, seq, nh, nkv, hd,
                                           window, frozen::kAttentionScale,
                                           pos_base, splits, kgeo, vgeo);
        }
    } else {
        e = RunMuseDenseAttention(stream, ws.q, dk, dv, ws.attn, seq, nh, nkv,
                                  hd, window, frozen::kAttentionScale, pos_base,
                                  kgeo, vgeo);
    }
    if (e != cudaSuccess) { MD_FAIL_LINE(e); return e; }

    // Attention OUTPUT gate: attn *= sigmoid(gate), elementwise, BEFORE o_proj.
    e = RunMuseDenseAttnGate(stream, ws.attn, ws.gate, ws.attn,
                             static_cast<std::int64_t>(seq) * q_rows);
    if (e != cudaSuccess) { MD_FAIL_LINE(e); return e; }

    // o_proj + centered post-attn norm + residual.
    e = RunLsxNvfp4Gemm(stream, ws.attn, lw.attn_o.d_packed, lw.attn_o.d_scales,
                        lw.attn_o.global_scale, ws.proj, seq, q_rows,
                        static_cast<int>(H));
    if (e != cudaSuccess) { MD_FAIL_LINE(e); return e; }
    e = RunMuseDenseCenteredNorm(stream, ws.proj, lw.d_post_attn_norm,
                                 ws.normed, seq, static_cast<int>(H), post_eps);
    if (e != cudaSuccess) { MD_FAIL_LINE(e); return e; }
    {
        const std::int64_t numel = static_cast<std::int64_t>(seq) * H;
        kernels::AddInPlaceKernel<__half>
            <<<static_cast<unsigned>((numel + kThreads - 1) / kThreads),
               kThreads, 0, stream>>>(ws.normed, ws.x, numel);
        e = cudaGetLastError();
        if (e != cudaSuccess) { MD_FAIL_LINE(e); return e; }
    }

    // SwiGLU MLP: pre-ffn norm, gate/up, silu(gate)*up, down, post-ffn norm.
    e = RunMuseDenseCenteredNorm(stream, ws.x, lw.d_pre_ffn, ws.normed, seq,
                                 static_cast<int>(H), pre_eps);
    if (e != cudaSuccess) { MD_FAIL_LINE(e); return e; }
    e = RunLsxNvfp4Gemm(stream, ws.normed, lw.mlp_gate.d_packed,
                        lw.mlp_gate.d_scales, lw.mlp_gate.global_scale,
                        ws.mlp_gate, seq, static_cast<int>(H),
                        static_cast<int>(I));
    if (e != cudaSuccess) { MD_FAIL_LINE(e); return e; }
    e = RunLsxNvfp4Gemm(stream, ws.normed, lw.mlp_up.d_packed,
                        lw.mlp_up.d_scales, lw.mlp_up.global_scale, ws.mlp_up,
                        seq, static_cast<int>(H), static_cast<int>(I));
    if (e != cudaSuccess) { MD_FAIL_LINE(e); return e; }
    e = RunMuseDenseSwiGLU(stream, ws.mlp_gate, ws.mlp_up, ws.mlp_gate,
                           static_cast<std::int64_t>(seq) * I);
    if (e != cudaSuccess) { MD_FAIL_LINE(e); return e; }
    e = RunLsxNvfp4Gemm(stream, ws.mlp_gate, lw.mlp_down.d_packed,
                        lw.mlp_down.d_scales, lw.mlp_down.global_scale, ws.ffn,
                        seq, static_cast<int>(I), static_cast<int>(H));
    if (e != cudaSuccess) { MD_FAIL_LINE(e); return e; }
    e = RunMuseDenseCenteredNorm(stream, ws.ffn, lw.d_post_ffn, ws.proj, seq,
                                 static_cast<int>(H), post_eps);
    if (e != cudaSuccess) { MD_FAIL_LINE(e); return e; }
    {
        const std::int64_t numel = static_cast<std::int64_t>(seq) * H;
        kernels::AddInPlaceKernel<__half>
            <<<static_cast<unsigned>((numel + kThreads - 1) / kThreads),
               kThreads, 0, stream>>>(ws.proj, ws.x, numel);
        e = cudaGetLastError();
        if (e != cudaSuccess) { MD_FAIL_LINE(e); return e; }
    }
    return cudaSuccess;
}

// Final plain norm + FP16 softcapped head + argmax.
cudaError_t SampleNext(cudaStream_t stream,
                       MuseDenseGenContext::Workspace &ws,
                       const MuseDensePinnedWeights &pw, int seq)
{
    const std::int64_t H = pw.top.hidden_size;
    const std::int64_t V = pw.top.vocab_size;
    cudaError_t e = RunMuseDensePlainNorm(stream, ws.x, pw.d_final_norm,
                                          ws.normed, seq, static_cast<int>(H),
                                          pw.top.rms_norm_eps);
    if (e != cudaSuccess) { MD_FAIL_LINE(e); return e; }
    const __half *last = ws.normed + static_cast<std::int64_t>(seq - 1) * H;
    return RunMuseDenseLogitsArgmax(stream, pw.d_lm_head, last,
                                    static_cast<int>(V), static_cast<int>(H),
                                    pw.top.output_multiplier,
                                    pw.top.final_logit_softcapping, ws.argmax,
                                    ws.maxv, ws.logits);
}

} // namespace

cudaError_t RunMuseDenseGeneration(
    cudaStream_t stream, const MuseDensePinnedWeights &pw,
    const std::vector<std::int32_t> &input_ids, int max_gen_tokens,
    const std::vector<int> &eos_ids, MuseDenseGenerationResult &out,
    MuseDenseKvCache &kv, MuseDenseGenContext *ctx,
    lsxcommon::sample::ISampler &sampler, const lsxcommon::GenerationStream *gs)
{
    const auto t_start = std::chrono::steady_clock::now();
    out = MuseDenseGenerationResult{};
    const auto &top = pw.top;
    const int H = top.hidden_size;
    const int V = top.vocab_size;
    const int P = static_cast<int>(input_ids.size());
    if (P <= 0) { out.error_msg = "empty prompt"; return cudaErrorInvalidValue; }
    if (top.num_layers <= 0 ||
        pw.layers.size() < static_cast<size_t>(top.num_layers)) {
        out.error_msg = "model has no loaded layers";
        return cudaErrorInvalidValue;
    }
    if (!pw.d_embed || !pw.d_lm_head || !pw.d_final_norm) {
        out.error_msg = "embed/head/final-norm missing";
        return cudaErrorInvalidValue;
    }
    if (!kv.allocated()) {
        out.error_msg = "kv cache not allocated";
        return cudaErrorInvalidValue;
    }
    MuseDenseGenContext local_ctx;
    if (!ctx) ctx = &local_ctx;

    const size_t q_rows = static_cast<size_t>(top.QRows());
    const size_t kv_rows = static_cast<size_t>(top.KvRows());
    auto &ws = ctx->ws;
    const size_t need = static_cast<size_t>(P) +
                        static_cast<size_t>(std::max(max_gen_tokens, 1)) + 8;
    if (!ctx->ws_ready || ws.seq_cap < need || ws.q_rows_cap < q_rows ||
        ws.kv_rows_cap < kv_rows) {
        const cudaError_t e = ws.Alloc(H, top.intermediate_size, V,
                                       std::max<size_t>(need, 64), q_rows,
                                       kv_rows,
                                       static_cast<size_t>(top.num_attention_heads),
                                       static_cast<size_t>(top.head_dim));
        if (e != cudaSuccess) { out.error_msg = "workspace alloc failed"; return e; }
        ctx->ws_ready = true;
    }

    auto run_layers = [&](int seq, int pos_base) -> cudaError_t {
        for (int l = 0; l < top.num_layers; ++l) {
            const auto &lw = pw.layers[static_cast<size_t>(l)];
            const cudaError_t e = ForwardLayer(
                stream, pw, lw, l, ws, seq, pos_base,
                kv.dk[static_cast<size_t>(l)], kv.dv[static_cast<size_t>(l)],
                kv.geo[static_cast<size_t>(l)], kv.geo[static_cast<size_t>(l)]);
            if (e != cudaSuccess) { MD_FAIL_LINE(e); return e; }
        }
        return cudaSuccess;
    };
    const auto is_eos = [&](std::int32_t t) {
        for (const int e : eos_ids) if (t == e) return true;
        return false;
    };
    // Embed gather + weightless embed norm (NO sqrt(hidden) scale).
    auto embed = [&](int seq) -> cudaError_t {
        cudaError_t e = RunMuseDenseEmbedGatherFp16(stream, pw.d_embed, ws.ids,
                                                    ws.x, seq, H);
        if (e != cudaSuccess) return e;
        return RunMuseDensePlainNorm(stream, ws.x, nullptr, ws.x, seq, H,
                                     top.rms_norm_eps);
    };

    // ── Prefill ── (scalar path: the goldens are captured on it)
    {
        const int seq = P;
        const bool tc_saved = lsx_nvfp4::LsxTcFastMath();
        lsx_nvfp4::SetLsxTcFastMath(false);
        cudaError_t e = cudaMemcpyAsync(ws.ids, input_ids.data(),
                                        sizeof(std::int32_t) *
                                            static_cast<size_t>(seq),
                                        cudaMemcpyHostToDevice, stream);
        if (e == cudaSuccess) e = embed(seq);
        if (e == cudaSuccess) e = run_layers(seq, 0);
        std::int32_t tok = -1;
        if (e == cudaSuccess) e = SampleNext(stream, ws, pw, seq);
        if (e == cudaSuccess && !sampler.IsGreedy())
            e = sampler.Sample(stream, ws.logits, V,
                               lsxcommon::sample::LogitType::kF32, ws.argmax,
                               ws.maxv);
        if (e == cudaSuccess)
            e = cudaMemcpyAsync(&tok, ws.argmax, sizeof(std::int32_t),
                                cudaMemcpyDeviceToHost, stream);
        if (e == cudaSuccess) e = cudaStreamSynchronize(stream);
        lsx_nvfp4::SetLsxTcFastMath(tc_saved);
        if (e != cudaSuccess) { out.error_msg = "prefill failed"; return e; }
        if (tok < 0 || tok >= V) {
            out.error_msg = "sampled token out of range";
            return cudaErrorInvalidValue;
        }
        out.tokens.push_back(tok);
        if (gs) gs->Emit(static_cast<std::uint32_t>(tok), 0u);
    }

    out.ttft_ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - t_start).count();
    if (is_eos(out.tokens.front())) out.eos_terminated = true;

    // ── Decode ──
    int pos_base = P;
    for (int step = 1; step < max_gen_tokens && !out.eos_terminated; ++step) {
        if (gs && gs->Aborted()) break;
        const std::int32_t next = out.tokens.back();
        cudaError_t e = cudaMemcpyAsync(ws.ids, &next, sizeof(std::int32_t),
                                        cudaMemcpyHostToDevice, stream);
        if (e == cudaSuccess) e = embed(1);
        if (e == cudaSuccess) e = run_layers(1, pos_base);
        std::int32_t tok = -1;
        if (e == cudaSuccess) e = SampleNext(stream, ws, pw, 1);
        if (e == cudaSuccess && !sampler.IsGreedy())
            e = sampler.Sample(stream, ws.logits, V,
                               lsxcommon::sample::LogitType::kF32, ws.argmax,
                               ws.maxv);
        if (e == cudaSuccess)
            e = cudaMemcpyAsync(&tok, ws.argmax, sizeof(std::int32_t),
                                cudaMemcpyDeviceToHost, stream);
        if (e == cudaSuccess) e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess) { out.error_msg = "decode failed"; return e; }
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
        if (e != cudaSuccess) { out.error_msg = "final logits copy failed"; return e; }
        out.final_logits = std::move(h);
    }

    out.ok = !out.tokens.empty();
    if (!out.ok && out.error_msg.empty()) out.error_msg = "no tokens generated";
    out.total_ms = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - t_start).count();
    out.resident_bytes = pw.resident_bytes;
    return cudaSuccess;
}

} // namespace muse_dense
} // namespace lsxcommon