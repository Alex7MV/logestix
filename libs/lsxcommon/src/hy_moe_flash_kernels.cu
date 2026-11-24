// hy_moe_flash_kernels.cu — Hy4-flash family kernels (see the header for the
// contract of each one). No atomics feed final_logits: every reduction has a
// fixed iteration order so the bit-exact class is reproducible.
#include "lsxcommon/hy_moe_flash_kernels.h"

#include <algorithm>
#include <cmath>

#include "lsxcommon/hy_moe_flash_device_math.h"
#include "nvfp4_gemv_math.cuh"

namespace lsxcommon {
namespace hy4 {
namespace gpu {

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float WarpSum(float v)
{
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
    return v;
}

// BLOCK-wide sum (<= 1024 threads). WarpSum alone is warp-scoped: a 128-wide
// RMSNorm block that used it normalized by the first warp's partial sum only.
__device__ __forceinline__ float BlockSum(float v, float *scratch)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v = WarpSum(v);
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    const int warps = (blockDim.x + 31) / 32;
    float total = (threadIdx.x < warps) ? scratch[threadIdx.x] : 0.0f;
    if (warp == 0) total = WarpSum(total);
    // The result is broadcast through shared memory, and the trailing barrier
    // lets the same scratch be reused by the very next reduction.
    if (threadIdx.x == 0) scratch[0] = total;
    __syncthreads();
    return scratch[0];
}

// ── iHC ────────────────────────────────────────────────────────────────────

// Block 0: the stream rsqrt. Block r>0: one mix row.
__global__ void ihc_mix_kernel(const float *__restrict__ x,
                               const float *__restrict__ fn, int rows, int total,
                               float eps, float *__restrict__ mix,
                               float *__restrict__ rsqrt_out)
{
    const int lane = threadIdx.x & 31;
    // Every block recomputes the stream norm: a grid-wide dependency on block 0
    // would need a cooperative launch (or a second kernel), and the norm is 384
    // lane-strided elements. Block 0 stores it for the caller.
    double ss = 0.0;
    for (int k = lane; k < total; k += 32)
        ss += static_cast<double>(x[k]) * x[k];
    float nv = static_cast<float>(ss);
    nv = WarpSum(nv);
    nv = rsqrtf(nv / static_cast<float>(total) + eps);
    if (blockIdx.x == 0) {
        if (lane == 0) rsqrt_out[0] = nv;
        return;
    }
    const int row = blockIdx.x - 1;
    if (row >= rows) return;
    const float *fr = fn + static_cast<size_t>(row) * total;
    // Fixed k order (lane-strided), fp64 partial sums for a stable reduction.
    double d = 0.0;
    for (int k = lane; k < total; k += 32)
        d += static_cast<double>(fr[k]) * x[k];
    float v = static_cast<float>(d);
    v = WarpSum(v);
    if (lane == 0) mix[row] = v * nv;
}

__global__ void ihc_pre_kernel(const float *__restrict__ mix,
                               const float *__restrict__ base,
                               const float *__restrict__ scale, int streams,
                               int hidden, float eps, float magnitude,
                               const float *__restrict__ x,
                               float *__restrict__ agg, float *__restrict__ pre,
                               float *__restrict__ post)
{
    // pre/post are cheap and idempotent: every block recomputes them in
    // registers, block 0 stores them.
    float p[8];
    float po[8];
    for (int j = 0; j < streams; ++j) {
        p[j] = Sigmoid(mix[j] * scale[0] + base[j]) + eps;
        po[j] = magnitude * Sigmoid(mix[streams + j] * scale[1] +
                                    base[streams + j]) + eps;
    }
    if (blockIdx.x == 0 && threadIdx.x < streams) {
        pre[threadIdx.x] = p[threadIdx.x];
        if (post) post[threadIdx.x] = po[threadIdx.x];
    }
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < hidden;
         i += gridDim.x * blockDim.x) {
        float a = 0.0f;
        for (int j = 0; j < streams; ++j) a += p[j] * x[j * hidden + i];
        agg[i] = a;
    }
}

__global__ void ihc_post_kernel(const float *__restrict__ agg,
                                const float *__restrict__ post, int streams,
                                int hidden, float *__restrict__ streams_buf)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < hidden;
         i += gridDim.x * blockDim.x) {
        for (int j = 0; j < streams; ++j)
            streams_buf[j * hidden + i] =
                post[j] * agg[i] + streams_buf[j * hidden + i];
    }
}

// ── casts ──────────────────────────────────────────────────────────────────
__global__ void cast_f2h_kernel(const float *__restrict__ src,
                                __half *__restrict__ dst, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half_rn(src[i]);
}

__global__ void cast_h2f_kernel(const __half *__restrict__ src,
                                float *__restrict__ dst, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

// ── RoPE ───────────────────────────────────────────────────────────────────
__global__ void rope_half_kernel(__half *__restrict__ x, int rows, int row_len,
                                 int rope_dim, const float *__restrict__ cos_row,
                                 const float *__restrict__ sin_row, bool inverse)
{
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int half = rope_dim / 2;
    for (int j = threadIdx.x; j < half; j += blockDim.x) {
        const int base = row * row_len + (row_len - rope_dim);
        const float c = cos_row[j];
        const float s = inverse ? -sin_row[j] : sin_row[j];
        const float x0 = __half2float(x[base + 2 * j]);
        const float x1 = __half2float(x[base + 2 * j + 1]);
        x[base + 2 * j] = __float2half_rn(x0 * c - x1 * s);
        x[base + 2 * j + 1] = __float2half_rn(x0 * s + x1 * c);
    }
}

// ── router ─────────────────────────────────────────────────────────────────
__global__ void router_logits_kernel(const __half *__restrict__ x,
                                     const __half *__restrict__ gate,
                                     int n_routed, int hidden,
                                     float *__restrict__ scores)
{
    const int e = blockIdx.x;
    if (e >= n_routed) return;
    const __half *row = gate + static_cast<size_t>(e) * hidden;
    float acc = 0.0f;
    for (int k = threadIdx.x; k < hidden; k += blockDim.x)
        acc += __half2float(row[k]) * __half2float(x[k]);
    __shared__ float red[kThreads / 32];
    const float total = BlockSum(acc, red);
    if (threadIdx.x == 0) scores[e] = Sigmoid(total);
}

__global__ void router_topk_kernel(const float *__restrict__ scores,
                                   const float *__restrict__ bias, int n_routed,
                                   int topk, float route_scale,
                                   int32_t *__restrict__ out_idx,
                                   float *__restrict__ out_w)
{
    extern __shared__ unsigned char smem[];
    float *choice = reinterpret_cast<float *>(smem);
    unsigned char *used = reinterpret_cast<unsigned char *>(choice + n_routed);
    for (int j = threadIdx.x; j < n_routed; j += blockDim.x) {
        choice[j] = scores[j] + bias[j];
        used[j] = 0;
    }
    __syncthreads();
    float picked[32];
    const int k = topk < 32 ? topk : 32;
    for (int t = 0; t < k; ++t) {
        // Deterministic selection: thread 0 scans in index order.
        if (threadIdx.x == 0) {
            int best = -1;
            float bv = -1e30f;
            for (int j = 0; j < n_routed; ++j) {
                if (used[j]) continue;
                if (best < 0 || choice[j] > bv) {
                    bv = choice[j];
                    best = j;
                }
            }
            used[best] = 1;
            out_idx[t] = best;
            picked[t] = scores[best];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int t = 0; t < k; ++t) sum += picked[t];
        const float denom = sum + 1e-20f;
        for (int t = 0; t < k; ++t)
            out_w[t] = picked[t] / denom * route_scale;
    }
}

// ── expert glue ────────────────────────────────────────────────────────────
__global__ void swiglu_clamp_kernel(const __half *__restrict__ gate,
                                    const __half *__restrict__ up,
                                    __half *__restrict__ out, int n, float limit)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = __float2half_rn(
            SwigluClamp(__half2float(gate[i]), __half2float(up[i]), limit));
}

__global__ void accumulate_scaled_kernel(const __half *__restrict__ src,
                                         float *__restrict__ dst, int n,
                                         float scale)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += scale * __half2float(src[i]);
}

__global__ void gated_scale_kernel(__half *__restrict__ x,
                                   const __half *__restrict__ gate, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] = __float2half_rn(__half2float(x[i]) *
                               Sigmoid(__half2float(gate[i])));
}

// ── MLA ────────────────────────────────────────────────────────────────────
// One block per head; one thread per output column (kv_lora wide).
//   qk[h][d] = sum_r W[h*nope + r][d] * q[h*(nope+rope) + r]
__global__ void mla_qk_project_kernel(const __half *__restrict__ q,
                                      const uint8_t *__restrict__ w,
                                      const uint8_t *__restrict__ sw, float gs,
                                      int heads, int nope, int kv_lora,
                                      int q_stride, float *__restrict__ qk)
{
    const int h = blockIdx.x;
    if (h >= heads) return;
    const int d = threadIdx.x;
    if (d >= kv_lora) return;
    // q is [heads][qk_head] with the nope lanes first: head h's nope block is at
    // h*q_stride, and q_stride is the FULL qk_head width (nope + rope).
    const __half *qh = q + static_cast<size_t>(h) * q_stride;
    float acc = 0.0f;
    for (int r = 0; r < nope; ++r) {
        const size_t row = static_cast<size_t>(h) * nope + r;
        const uint8_t byte = w[row * (kv_lora / 2) + (d >> 1)];
        const uint8_t sb = sw[row * (kv_lora / 16) + (d >> 4)];
        acc += __half2float(qh[r]) * Nvfp4Element(byte, d & 1, sb, gs);
    }
    qk[static_cast<size_t>(h) * kv_lora + d] = acc;
}

// One block per head: scores -> sink softmax -> weighted latent accumulation.
// `scores` is [heads][npos] scratch, `acc` is [heads][kv_lora].
__global__ void mla_sink_softmax_kernel(const float *__restrict__ qk,
                                        const __half *__restrict__ q_all,
                                        int q_stride, int q_rope_off,
                                        const __half *__restrict__ latent,
                                        int64_t latent_stride,
                                        const int32_t *__restrict__ pos, int npos,
                                        int heads, int kv_lora, int rope,
                                        const float *__restrict__ sink,
                                        float scale, float *__restrict__ scores,
                                        float *__restrict__ acc)
{
    const int h = blockIdx.x;
    if (h >= heads) return;
    const float *qkh = qk + static_cast<size_t>(h) * kv_lora;
    // The rope lanes of head h live at q_all + h*q_stride + q_rope_off, not at a
    // contiguous offset: the query buffer is [heads][qk_head] with nope first.
    const __half *qrh = q_all + static_cast<size_t>(h) * q_stride + q_rope_off;
    float *sh = scores + static_cast<size_t>(h) * npos;
    __shared__ float s_warp[32];
    __shared__ float s_den;

    // Score pass: a block-wide reduction per position, so the result does not
    // depend on the block width (bit-exact across launch shapes).
    for (int t = 0; t < npos; ++t) {
        const int32_t p = pos[t];
        if (p < 0) {
            if (threadIdx.x == 0) sh[t] = -INFINITY;
            __syncthreads();
            continue;
        }
        const __half *lat = latent + static_cast<int64_t>(p) * latent_stride;
        float dot = 0.0f;
        for (int d = threadIdx.x; d < kv_lora; d += blockDim.x)
            dot += qkh[d] * __half2float(lat[d]);
        for (int d = threadIdx.x; d < rope; d += blockDim.x)
            dot += __half2float(qrh[d]) * __half2float(lat[kv_lora + d]);
        dot = BlockSum(dot, s_warp);
        if (threadIdx.x == 0) sh[t] = dot * scale;
        __syncthreads();
    }

    // Sink softmax: the sink is one extra logit that is dropped afterwards, so
    // its probability mass leaves the output below 1.
    if (threadIdx.x == 0) {
        float m = sink[h];
        for (int t = 0; t < npos; ++t) m = fmaxf(m, sh[t]);
        float den = __expf(sink[h] - m);
        for (int t = 0; t < npos; ++t) den += __expf(sh[t] - m);
        s_den = den;
        for (int t = 0; t < npos; ++t) sh[t] = __expf(sh[t] - m) / den;
    }
    __syncthreads();

    float *ah = acc + static_cast<size_t>(h) * kv_lora;
    for (int d = threadIdx.x; d < kv_lora; d += blockDim.x) {
        float a = 0.0f;
        for (int t = 0; t < npos; ++t) {
            const int32_t p = pos[t];
            if (p < 0) continue;
            const __half *lat = latent + static_cast<int64_t>(p) * latent_stride;
            a += sh[t] * __half2float(lat[d]);
        }
        ah[d] = a;
    }
}

// o[h][i] = sum_d W[h*v_head + i][d] * acc[h][d]
__global__ void mla_value_project_kernel(const float *__restrict__ acc,
                                         const uint8_t *__restrict__ w,
                                         const uint8_t *__restrict__ sw,
                                         float gs, int heads, int v_head,
                                         int kv_lora, __half *__restrict__ out)
{
    const int h = blockIdx.x;
    if (h >= heads) return;
    const int i = threadIdx.x;
    if (i >= v_head) return;
    const size_t row = static_cast<size_t>(h) * v_head + i;
    const float *a = acc + static_cast<size_t>(h) * kv_lora;
    float acc_f = 0.0f;
    for (int d = 0; d < kv_lora; ++d) {
        const uint8_t byte = w[row * (kv_lora / 2) + (d >> 1)];
        const uint8_t sb = sw[row * (kv_lora / 16) + (d >> 4)];
        acc_f += a[d] * Nvfp4Element(byte, d & 1, sb, gs);
    }
    out[row] = __float2half_rn(acc_f);
}

__global__ void window_positions_kernel(int32_t *__restrict__ pos, int window,
                                        int p)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= window) return;
    const int start = p - window + 1;
    const int v = start + t;
    pos[t] = (v < 0) ? -1 : v;
}

// ── IndexPool ──────────────────────────────────────────────────────────────
// Weighted RMSNorm over `dim` lanes, in place, fp32 accumulate.
__global__ void rms_norm_half_kernel(__half *__restrict__ x,
                                     const __half *__restrict__ w, int dim,
                                     float eps)
{
    __shared__ float s_warp[32];
    __shared__ float s_rs;
    float ss = 0.0f;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        const float v = __half2float(x[d]);
        ss += v * v;
    }
    ss = BlockSum(ss, s_warp);
    if (threadIdx.x == 0) s_rs = rsqrtf(ss / static_cast<float>(dim) + eps);
    __syncthreads();
    const float rs = s_rs;
    for (int d = threadIdx.x; d < dim; d += blockDim.x)
        x[d] = __float2half_rn(__half2float(x[d]) * rs * __half2float(w[d]));
}

// Unweighted RMSNorm used by the pooled key (`key_norm` is a scale vector, so
// the weighted form above is used for both; this one is the F32 variant for the
// pooled key whose scale vector is F32).
__global__ void rms_norm_pool_kernel(float *__restrict__ x,
                                     const float *__restrict__ w, int dim,
                                     float eps)
{
    __shared__ float s_warp[32];
    __shared__ float s_rs;
    float ss = 0.0f;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) ss += x[d] * x[d];
    ss = BlockSum(ss, s_warp);
    if (threadIdx.x == 0) s_rs = rsqrtf(ss / static_cast<float>(dim) + eps);
    __syncthreads();
    const float rs = s_rs;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) x[d] = x[d] * rs * w[d];
}

// One COMPLETE group of `ratio` token keys -> one pooled key.
//   u_i = k_i + ape_i ;  v_i = W_kv u_i ;  s_i = sigmoid(W_gate u_i)
//   p   = key_norm( sum_i s_i * v_i )
__global__ void index_pool_group_kernel(const __half *__restrict__ keys,
                                        const float *__restrict__ ape,
                                        const float *__restrict__ w_kv,
                                        const float *__restrict__ w_gate,
                                        const float *__restrict__ norm,
                                        int ratio, int dim, __half *__restrict__ out)
{
    extern __shared__ float u[];   // [ratio][dim]
    const int d = threadIdx.x;
    for (int i = 0; i < ratio; ++i)
        for (int e = d; e < dim; e += blockDim.x)
            u[i * dim + e] = __half2float(keys[i * dim + e]) + ape[i * dim + e];
    __syncthreads();
    float acc = 0.0f;
    for (int e = d; e < dim; e += blockDim.x) {
        float v = 0.0f;
        for (int i = 0; i < ratio; ++i) {
            float kv = 0.0f;
            float gt = 0.0f;
            for (int k = 0; k < dim; ++k) {
                const float uk = u[i * dim + k];
                kv += w_kv[e * dim + k] * uk;
                gt += w_gate[e * dim + k] * uk;
            }
            v += Sigmoid(gt) * kv;
        }
        acc = v;
        __shared__ float tmp[128];
        tmp[d] = acc;
        __syncthreads();
        float ss = 0.0f;
        for (int k = 0; k < dim; ++k) ss += tmp[k] * tmp[k];
        out[e] = __float2half_rn(acc * rsqrtf(ss / static_cast<float>(dim) + 1e-5f) *
                                 norm[e]);
    }
}

// score[g] = head_scale * sum_h w[h] * relu(qk[h] . p[g])
__global__ void index_group_scores_kernel(const float *__restrict__ qk,
                                          const float *__restrict__ w_heads,
                                          const __half *__restrict__ pooled,
                                          int n_groups, int heads, int dim,
                                          float head_scale,
                                          float *__restrict__ scores)
{
    const int g = blockIdx.x;
    if (g >= n_groups) return;
    const __half *p = pooled + static_cast<size_t>(g) * dim;
    float acc = 0.0f;
    __shared__ float s_dot[kThreads / 32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int h = 0; h < heads; ++h) {
        const float *qh = qk + static_cast<size_t>(h) * dim;
        float dot = 0.0f;
        for (int d = threadIdx.x; d < dim; d += blockDim.x)
            dot += qh[d] * __half2float(p[d]);
        dot = WarpSum(dot);
        if (lane == 0) s_dot[warp] = dot;
        __syncthreads();
        if (threadIdx.x == 0) {
            float tot = 0.0f;
            for (int w = 0; w < kThreads / 32; ++w) tot += s_dot[w];
            acc += w_heads[h] * fmaxf(tot, 0.0f);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) scores[g] = acc * head_scale;
}

// Keeps the `topk` best groups (ties -> lower index) and expands them to token
// positions in ascending order; the tail is padded with -1.
__global__ void index_select_groups_kernel(const float *__restrict__ scores,
                                           int n_complete, int ratio, int topk,
                                           int tail_start, int n_tail,
                                           int32_t *__restrict__ out, int n_pad,
                                           unsigned char *__restrict__ used)
{
    if (threadIdx.x != 0) return;
    const int keep = topk < n_complete ? topk : n_complete;
    for (int i = 0; i < n_complete; ++i) used[i] = 0;
    for (int k = 0; k < keep; ++k) {
        int best = -1;
        float bv = -1e30f;
        for (int i = 0; i < n_complete; ++i) {
            if (used[i]) continue;
            if (best < 0 || scores[i] > bv) {
                bv = scores[i];
                best = i;
            }
        }
        used[best] = 1;
    }
    // Selected complete groups expand in ascending order, then the CURRENT
    // partial group's already-decoded tokens (they are the newest positions, so
    // the list stays sorted and the caller can feed it straight to attention).
    int c = 0;
    for (int i = 0; i < n_complete; ++i) {
        if (!used[i]) continue;
        for (int r = 0; r < ratio && c < n_pad; ++r) out[c++] = i * ratio + r;
    }
    for (int t = 0; t < n_tail && c < n_pad; ++t) out[c++] = tail_start + t;
    for (int i = c; i < n_pad; ++i) out[i] = -1;
}

// ── vocabulary ─────────────────────────────────────────────────────────────
__global__ void embed_lookup_kernel(const __half *__restrict__ embed, int hidden,
                                    const int32_t *__restrict__ tok, int n,
                                    __half *__restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * hidden) return;
    const int t = i / hidden;
    const int d = i % hidden;
    out[i] = embed[static_cast<size_t>(tok[t]) * hidden + d];
}

__global__ void head_logits_kernel(const __half *__restrict__ head,
                                   const __half *__restrict__ x, int vocab,
                                   int hidden, float *__restrict__ logits)
{
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= vocab) return;
    const __half *row = head + static_cast<size_t>(warp) * hidden;
    float acc = 0.0f;
    for (int k = lane; k < hidden; k += 32)
        acc += __half2float(row[k]) * __half2float(x[k]);
    acc = WarpSum(acc);
    if (lane == 0) logits[warp] = acc;
}

__global__ void argmax_kernel(const float *__restrict__ x, int n,
                              int32_t *__restrict__ out_idx,
                              float *__restrict__ out_val)
{
    __shared__ float v[kThreads / 32];
    __shared__ int i32[kThreads / 32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    float bv = -INFINITY;
    int bi = -1;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float xv = x[i];
        if (bi < 0 || xv > bv) {
            bv = xv;
            bi = i;
        }
    }
    // deterministic block reduction in ascending index order
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_xor_sync(0xFFFFFFFFu, bv, off);
        const int oi = __shfl_xor_sync(0xFFFFFFFFu, bi, off);
        if (oi >= 0 && (bi < 0 || ov > bv || (ov == bv && oi < bi))) {
            bv = ov;
            bi = oi;
        }
    }
    if (lane == 0) {
        v[warp] = bv;
        i32[warp] = bi;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float rv = -INFINITY;
        int ri = -1;
        for (int w = 0; w < kThreads / 32; ++w) {
            if (i32[w] >= 0 && (ri < 0 || v[w] > rv || (v[w] == rv && i32[w] < ri))) {
                rv = v[w];
                ri = i32[w];
            }
        }
        out_idx[0] = ri;
        out_val[0] = rv;
    }
}

} // namespace

// ── Launchers ──────────────────────────────────────────────────────────────

cudaError_t RunIhcMix(const float *x, const float *fn, const float *base,
                      const float *scale, int streams, int hidden, float eps,
                      float magnitude, float *agg, float *pre, float *post,
                      float *mix, float *rsqrt, cudaStream_t stream)
{
    const int total = streams * hidden;
    ihc_mix_kernel<<<2 * streams + 1, 32, 0, stream>>>(x, fn, 2 * streams,
                                                       total, eps, mix, rsqrt);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    ihc_pre_kernel<<<16, kThreads, 0, stream>>>(mix, base, scale, streams,
                                                hidden, eps, magnitude, x, agg,
                                                pre, post);
    return cudaGetLastError();
}

cudaError_t RunIhcPost(const float *agg, const float *post, int streams,
                       int hidden, float *streams_buf, cudaStream_t stream)
{
    ihc_post_kernel<<<16, kThreads, 0, stream>>>(agg, post, streams, hidden,
                                                 streams_buf);
    return cudaGetLastError();
}

cudaError_t RunIhcHead(const float *x, const float *fn, const float *base,
                       const float *scale, int streams, int hidden, float eps,
                       float *agg, float *pre, float *mix, float *rsqrt,
                       cudaStream_t stream)
{
    const int total = streams * hidden;
    ihc_mix_kernel<<<streams + 1, 32, 0, stream>>>(x, fn, streams, total, eps,
                                                   mix, rsqrt);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    // The head collapse has no post term: a null post buffer keeps the same
    // pre-mixdown code path.
    ihc_pre_kernel<<<16, kThreads, 0, stream>>>(mix, base, scale, streams,
                                                hidden, eps, 0.0f, x, agg, pre,
                                                nullptr);
    return cudaGetLastError();
}

cudaError_t RunCastF32ToHalf(const float *src, __half *dst, int n,
                             cudaStream_t stream)
{
    if (n <= 0) return cudaSuccess;
    cast_f2h_kernel<<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(src,
                                                                           dst, n);
    return cudaGetLastError();
}

cudaError_t RunCastHalfToF32(const __half *src, float *dst, int n,
                             cudaStream_t stream)
{
    if (n <= 0) return cudaSuccess;
    cast_h2f_kernel<<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(src,
                                                                           dst, n);
    return cudaGetLastError();
}

cudaError_t RunRopeHalf(__half *x, int rows, int row_len, int rope_dim,
                        const float *cos_row, const float *sin_row, bool inverse,
                        cudaStream_t stream)
{
    if (rows <= 0 || rope_dim <= 0) return cudaSuccess;
    rope_half_kernel<<<rows, std::min(rope_dim / 2, 256), 0, stream>>>(
        x, rows, row_len, rope_dim, cos_row, sin_row, inverse);
    return cudaGetLastError();
}

cudaError_t RunRouterLogits(const __half *x, const __half *gate, int n_routed,
                            int hidden, float *scores, cudaStream_t stream)
{
    if (n_routed <= 0) return cudaSuccess;
    router_logits_kernel<<<n_routed, kThreads, 0, stream>>>(x, gate, n_routed,
                                                            hidden, scores);
    return cudaGetLastError();
}

cudaError_t RunRouterTopK(const float *scores, const float *bias, int n_routed,
                          int topk, float route_scale, int32_t *out_idx,
                          float *out_w, cudaStream_t stream)
{
    if (n_routed <= 0 || topk <= 0) return cudaSuccess;
    const size_t sh = static_cast<size_t>(n_routed) * sizeof(float) + n_routed;
    router_topk_kernel<<<1, kThreads, sh, stream>>>(scores, bias, n_routed, topk,
                                                    route_scale, out_idx, out_w);
    return cudaGetLastError();
}

cudaError_t RunSwigluClamp(const __half *gate, const __half *up, __half *out,
                           int n, float limit, cudaStream_t stream)
{
    if (n <= 0) return cudaSuccess;
    swiglu_clamp_kernel<<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        gate, up, out, n, limit);
    return cudaGetLastError();
}

cudaError_t RunAccumulateScaled(const __half *src, float *dst, int n,
                                float scale, cudaStream_t stream)
{
    if (n <= 0) return cudaSuccess;
    accumulate_scaled_kernel<<<(n + kThreads - 1) / kThreads, kThreads, 0,
                               stream>>>(src, dst, n, scale);
    return cudaGetLastError();
}

cudaError_t RunGatedScale(__half *x, const __half *gate, int n,
                          cudaStream_t stream)
{
    if (n <= 0) return cudaSuccess;
    gated_scale_kernel<<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        x, gate, n);
    return cudaGetLastError();
}

cudaError_t RunMlaQkProject(const __half *q, const lsx_nvfp4::LsxNvfp4Weight &w,
                            int heads, int nope, int q_stride, float *qk,
                            cudaStream_t stream)
{
    if (heads <= 0 || nope <= 0) return cudaSuccess;
    if (!w.d_packed || !w.d_scales) return cudaErrorInvalidValue;
    mla_qk_project_kernel<<<heads, w.cols, 0, stream>>>(
        q, w.d_packed, w.d_scales, w.global_scale, heads, nope, w.cols, q_stride,
        qk);
    return cudaGetLastError();
}

cudaError_t RunMlaSinkSoftmax(const float *qk, const __half *q_all,
                              int q_stride, int q_rope_off,
                              const __half *latent, int64_t latent_stride,
                              const int32_t *pos, int npos, int heads,
                              int kv_lora, int rope, const float *sink,
                              float scale, float *scores, float *acc,
                              cudaStream_t stream)
{
    if (heads <= 0 || npos <= 0) return cudaSuccess;
    mla_sink_softmax_kernel<<<heads, kThreads, 0, stream>>>(
        qk, q_all, q_stride, q_rope_off, latent, latent_stride, pos, npos, heads,
        kv_lora, rope, sink, scale, scores, acc);
    return cudaGetLastError();
}

cudaError_t RunMlaValueProject(const float *acc,
                               const lsx_nvfp4::LsxNvfp4Weight &w, int heads,
                               int v_head, int kv_lora, __half *out,
                               cudaStream_t stream)
{
    if (heads <= 0 || v_head <= 0) return cudaSuccess;
    if (!w.d_packed || !w.d_scales) return cudaErrorInvalidValue;
    mla_value_project_kernel<<<heads, v_head, 0, stream>>>(
        acc, w.d_packed, w.d_scales, w.global_scale, heads, v_head, kv_lora,
        out);
    return cudaGetLastError();
}

cudaError_t RunWindowPositions(int32_t *pos, int window, int p,
                               cudaStream_t stream)
{
    if (window <= 0) return cudaSuccess;
    window_positions_kernel<<<(window + kThreads - 1) / kThreads, kThreads, 0,
                              stream>>>(pos, window, p);
    return cudaGetLastError();
}

cudaError_t RunEmbedLookup(const __half *embed, int vocab, int hidden,
                           const int32_t *tok, int n, __half *out,
                           cudaStream_t stream)
{
    (void)vocab;
    if (n <= 0 || hidden <= 0) return cudaSuccess;
    const int total = n * hidden;
    embed_lookup_kernel<<<(total + kThreads - 1) / kThreads, kThreads, 0,
                          stream>>>(embed, hidden, tok, n, out);
    return cudaGetLastError();
}

cudaError_t RunHeadLogits(const __half *head, const __half *x, int vocab,
                          int hidden, float *logits, cudaStream_t stream)
{
    if (vocab <= 0) return cudaSuccess;
    const int warps_per_block = kThreads / 32;
    const int blocks = (vocab + warps_per_block - 1) / warps_per_block;
    head_logits_kernel<<<blocks, kThreads, 0, stream>>>(head, x, vocab, hidden,
                                                        logits);
    return cudaGetLastError();
}

cudaError_t RunArgmax(const float *x, int n, int32_t *out_idx, float *out_val,
                      cudaStream_t stream)
{
    if (n <= 0) return cudaSuccess;
    argmax_kernel<<<1, kThreads, 0, stream>>>(x, n, out_idx, out_val);
    return cudaGetLastError();
}


cudaError_t RunRmsNormHalf(__half *x, const __half *w, int dim, float eps,
                           cudaStream_t stream)
{
    if (dim <= 0) return cudaSuccess;
    rms_norm_half_kernel<<<1, 128, 0, stream>>>(x, w, dim, eps);
    return cudaGetLastError();
}

cudaError_t RunIndexPoolGroup(const __half *keys, const float *ape,
                              const float *w_kv, const float *w_gate,
                              const float *norm, int ratio, int dim,
                              __half *out, cudaStream_t stream)
{
    if (ratio <= 0 || dim <= 0 || dim > 128) return cudaErrorInvalidValue;
    const size_t sh = static_cast<size_t>(ratio) * dim * sizeof(float);
    index_pool_group_kernel<<<1, 128, sh, stream>>>(keys, ape, w_kv, w_gate,
                                                    norm, ratio, dim, out);
    return cudaGetLastError();
}

cudaError_t RunIndexGroupScores(const float *qk, const float *w_heads,
                                const __half *pooled, int n_groups, int heads,
                                int dim, float head_scale, float *scores,
                                cudaStream_t stream)
{
    if (n_groups <= 0) return cudaSuccess;
    index_group_scores_kernel<<<n_groups, kThreads, 0, stream>>>(
        qk, w_heads, pooled, n_groups, heads, dim, head_scale, scores);
    return cudaGetLastError();
}

cudaError_t RunIndexSelectGroups(const float *scores, int n_complete, int ratio,
                                 int topk, int tail_start, int n_tail,
                                 int32_t *out, int n_pad, unsigned char *used,
                                 cudaStream_t stream)
{
    if (n_complete <= 0 && n_tail <= 0) return cudaSuccess;
    index_select_groups_kernel<<<1, 32, 0, stream>>>(
        scores, n_complete, ratio, topk, tail_start, n_tail, out, n_pad, used);
    return cudaGetLastError();
}

} // namespace gpu
} // namespace hy4
} // namespace lsxcommon
