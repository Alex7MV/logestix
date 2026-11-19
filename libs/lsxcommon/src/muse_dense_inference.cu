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

// ── Generation driver (body lands with Task 7) ──────────────────────────────

cudaError_t RunMuseDenseGeneration(
    cudaStream_t stream, const MuseDensePinnedWeights &pw,
    const std::vector<std::int32_t> &input_ids, int max_gen_tokens,
    const std::vector<int> &eos_ids, MuseDenseGenerationResult &out,
    MuseDenseKvCache &kv, lsxcommon::sample::ISampler &sampler,
    const lsxcommon::GenerationStream *gs)
{
    (void)stream;
    (void)pw;
    (void)input_ids;
    (void)max_gen_tokens;
    (void)eos_ids;
    (void)kv;
    (void)sampler;
    (void)gs;
    out.ok = false;
    out.error_msg = "muse-glimmer generation driver not yet implemented";
    return cudaErrorNotYetImplemented;
}

} // namespace muse_dense
} // namespace lsxcommon