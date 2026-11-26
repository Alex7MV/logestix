// GigaChat4-37B-A4B DSA lightning indexer (decode step). See
// gigachat_moe_37b_inference_indexer.h for the transcribed semantics.
#include "lsxcommon/gigachat_moe_37b_inference_indexer.h"

#include "lsx_kernels_common.cuh"

#include "lsxcommon/lsx_gemv.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstddef>

namespace lsxcommon::gigachat {
namespace {

constexpr int kBlock = 256;
int Grid(int n, int threads = kBlock) { return (n + threads - 1) / threads; }

__device__ __forceinline__ float Bf16ToF32(const __nv_bfloat16 *p, int i)
{
    return __bfloat162float(p[i]);
}

// Non-interleaved rotate-half on the first `rope` lanes: the reference builds
// rotate_half(x) = cat(-x2, x1) over the half split, so lane i pairs with lane
// i + rope/2 and both use the same cos/sin frequency i.
__global__ void indexer_rope_kernel(float *__restrict__ q, float *__restrict__ k,
                                    int heads, int head_dim, int rope,
                                    const float *__restrict__ cos_t,
                                    const float *__restrict__ sin_t)
{
    const int half = rope / 2;
    const int total = heads * half;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x)
    {
        const int h = idx / half;
        const int j = idx % half;
        const float c = cos_t[j];
        const float s = sin_t[j];
        float *qh = q + static_cast<size_t>(h) * head_dim;
        const float a = qh[j];
        const float b = qh[j + half];
        qh[j] = a * c - b * s;
        qh[j + half] = b * c + a * s;
    }
    if (blockIdx.x == 0)
    {
        for (int j = threadIdx.x; j < half; j += blockDim.x)
        {
            const float c = cos_t[j];
            const float s = sin_t[j];
            const float a = k[j];
            const float b = k[j + half];
            k[j] = a * c - b * s;
            k[j + half] = b * c + a * s;
        }
    }
}

// Plain RMSNorm: xn = x * rsqrt(mean(x^2) + eps) * w. One block, deterministic
// warp reduction, no atomics.
__global__ void indexer_knorm_kernel(float *__restrict__ k,
                                     const __nv_bfloat16 *__restrict__ w,
                                     int n, float eps)
{
    __shared__ float red[32];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    float ss = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) ss += k[i] * k[i];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        ss += __shfl_down_sync(0xffffffffu, ss, off);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    float tot = (tid < static_cast<int>(blockDim.x / 32)) ? red[tid] : 0.0f;
    if (tid < 32)
    {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            tot += __shfl_down_sync(0xffffffffu, tot, off);
        if (tid == 0) red[0] = tot;
    }
    __syncthreads();
    const float inv =
        rsqrtf(red[0] / static_cast<float>(n) + eps);
    for (int i = tid; i < n; i += blockDim.x)
        k[i] = k[i] * inv * __bfloat162float(w[i]);
}

// score_t = scale * sum_h weights_h * relu(q_h . k_t). One thread per position,
// ascending `+=` per head so the accumulation order is fixed by the geometry.
__global__ void indexer_scores_kernel(const float *__restrict__ q,
                                      const float *__restrict__ k_cache,
                                      const float *__restrict__ weights,
                                      int heads, int head_dim, float scale,
                                      int n, float *__restrict__ scores)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const float *kt = k_cache + static_cast<size_t>(t) * head_dim;
    float acc = 0.0f;
    for (int h = 0; h < heads; ++h)
    {
        const float *qh = q + static_cast<size_t>(h) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) dot += qh[d] * kt[d];
        acc += weights[h] * fmaxf(dot, 0.0f);
    }
    scores[t] = acc * scale;
}

__device__ __forceinline__ unsigned OrderedKey(float v)
{
    const unsigned u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}

__global__ void topk_select_kernel(const float *__restrict__ scores, int n,
                                   int k, int32_t *__restrict__ out_idx,
                                   int *__restrict__ d_count)
{
    __shared__ unsigned s_lo, s_hi, s_mid, s_cnt;
    __shared__ unsigned s_part[32];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps = (blockDim.x + 31) / 32;

    unsigned lo = 0xFFFFFFFFu, hi = 0x00000000u;
    for (int i = tid; i < n; i += blockDim.x)
    {
        const unsigned u = OrderedKey(scores[i]);
        lo = min(lo, u);
        hi = max(hi, u);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
    {
        lo = min(lo, __shfl_down_sync(0xffffffffu, lo, off));
        hi = max(hi, __shfl_down_sync(0xffffffffu, hi, off));
    }
    __shared__ unsigned s_mn[32], s_mx[32];
    if (lane == 0)
    {
        s_mn[warp] = lo;
        s_mx[warp] = hi;
    }
    __syncthreads();
    if (tid == 0)
    {
        unsigned a = 0xFFFFFFFFu, b = 0x00000000u;
        for (int i = 0; i < warps; ++i)
        {
            a = min(a, s_mn[i]);
            b = max(b, s_mx[i]);
        }
        s_lo = a;
        s_hi = b;
    }
    __syncthreads();
    lo = s_lo;
    hi = s_hi;

    // Binary search the k-th largest value (monotone uint order).
    while (lo < hi)
    {
        if (tid == 0) s_mid = lo + (hi - lo + 1) / 2;
        __syncthreads();
        const unsigned mid = s_mid;
        unsigned c = 0;
        for (int i = tid; i < n; i += blockDim.x)
            if (OrderedKey(scores[i]) >= mid) ++c;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            c += __shfl_down_sync(0xffffffffu, c, off);
        if (lane == 0) s_part[warp] = c;
        __syncthreads();
        if (tid == 0)
        {
            unsigned tot = 0;
            for (int i = 0; i < warps; ++i) tot += s_part[i];
            s_cnt = tot;
        }
        __syncthreads();
        if (tid == 0)
        {
            if (s_cnt >= static_cast<unsigned>(k)) lo = mid;
            else hi = mid - 1;
            s_lo = lo;
            s_hi = hi;
        }
        __syncthreads();
        lo = s_lo;
        hi = s_hi;
    }
    const unsigned thr = lo;

    // One ascending scan (thread 0): every strictly-greater entry, then exactly
    // `k - gt` entries from the tie group in ascending index order. The total is
    // exactly k, so a tie can never crowd out a strictly-greater entry.
    if (tid == 0)
    {
        int gt = 0;
        for (int i = 0; i < n; ++i)
            if (OrderedKey(scores[i]) > thr) ++gt;
        const int need_eq = k - gt;
        int written = 0, eq = 0;
        for (int i = 0; i < n; ++i)
        {
            const unsigned u = OrderedKey(scores[i]);
            if (u > thr)
                out_idx[written++] = i;
            else if (u == thr && eq < need_eq)
            {
                out_idx[written++] = i;
                ++eq;
            }
        }
        *d_count = written;
    }
}

__global__ void topk_identity_kernel(int n, int32_t *__restrict__ out_idx,
                                     int *__restrict__ d_count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out_idx[i] = i;
    if (i == 0) *d_count = n;
}

__global__ void scale_f32_to_half_kernel(const float *__restrict__ x,
                                        __half *__restrict__ y, int n,
                                        float scale)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i] * scale);
}

__global__ void weights_scale_kernel(float *__restrict__ w, int n, float scale)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) w[i] *= scale;
}

__global__ void write_k_cache_kernel(const float *__restrict__ k, int head_dim,
                                     float *__restrict__ cache, int pos)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < head_dim)
        cache[static_cast<size_t>(pos) * head_dim + i] = k[i];
}

} // namespace

void GcDsaIndexerWeights::Free()
{
    wq_b.Free();
    cudaFree(const_cast<uint8_t *>(d_wk));
    cudaFree(const_cast<uint8_t *>(d_weights_proj));
    cudaFree(const_cast<uint8_t *>(d_k_norm));
    d_wk = nullptr;
    d_weights_proj = nullptr;
    d_k_norm = nullptr;
}

bool GcDsaIndexerScratch::Alloc(const GcDsaIndexerDims &d, int capacity)
{
    Free();
    if (d.hidden <= 0 || d.q_lora <= 0 || d.heads <= 0 || d.head_dim <= 0 ||
        capacity <= 0)
        return false;
    int64_t total = 0;
    auto need = [&](int64_t b) { total = (total + 255) / 256 * 256 + b; };
    need(static_cast<int64_t>(d.q_lora) * 2);              // q_h
    need(static_cast<int64_t>(d.hidden) * 2);              // x_h
    need(static_cast<int64_t>(d.head_dim) * 2);            // k_h
    need(static_cast<int64_t>(d.heads) * d.head_dim * 4);  // q_f
    need(static_cast<int64_t>(d.head_dim) * 4);            // k_f
    need(static_cast<int64_t>(d.heads) * 4);               // w_f
    need(static_cast<int64_t>(capacity) * 4);              // scores
    void *base = nullptr;
    if (cudaMalloc(&base, static_cast<size_t>(total)) != cudaSuccess)
        return false;
    d_base = base;
    bytes = total;
    int64_t off = 0;
    auto carve = [&](int64_t b) {
        off = (off + 255) / 256 * 256;
        void *p = static_cast<uint8_t *>(base) + off;
        off += b;
        return p;
    };
    q_h = carve(static_cast<int64_t>(d.q_lora) * 2);
    x_h = carve(static_cast<int64_t>(d.hidden) * 2);
    k_h = carve(static_cast<int64_t>(d.head_dim) * 2);
    q_f = carve(static_cast<int64_t>(d.heads) * d.head_dim * 4);
    k_f = carve(static_cast<int64_t>(d.head_dim) * 4);
    w_f = carve(static_cast<int64_t>(d.heads) * 4);
    scores = carve(static_cast<int64_t>(capacity) * 4);
    return true;
}

void GcDsaIndexerScratch::Free()
{
    if (d_base)
    {
        cudaFree(d_base);
        d_base = nullptr;
    }
    bytes = 0;
    q_h = k_h = x_h = q_f = k_f = w_f = k_normed = scores = nullptr;
}

cudaError_t RunGigaChatDsaIndexerRope(float *q, float *k, int heads, int head_dim,
                                    int rope, const float *cos_t,
                                    const float *sin_t, int pos,
                                    cudaStream_t stream)
{
    if (!q || !k || !cos_t || !sin_t || heads <= 0 || head_dim <= 0 ||
        rope <= 0 || rope % 2 != 0 || rope > head_dim)
        return cudaErrorInvalidValue;
    const int half = rope / 2;
    const float *c = cos_t + static_cast<size_t>(pos) * half;
    const float *s = sin_t + static_cast<size_t>(pos) * half;
    indexer_rope_kernel<<<Grid(heads * half), kBlock, 0, stream>>>(
        q, k, heads, head_dim, rope, c, s);
    return cudaGetLastError();
}

cudaError_t RunGigaChatDsaIndexerKNorm(float *k, const uint8_t *w_bf16, int n,
                                     float eps, cudaStream_t stream)
{
    if (!k || !w_bf16 || n <= 0) return cudaErrorInvalidValue;
    indexer_knorm_kernel<<<1, 128, 0, stream>>>(
        k, reinterpret_cast<const __nv_bfloat16 *>(w_bf16), n, eps);
    return cudaGetLastError();
}

cudaError_t RunGigaChatDsaIndexerScores(const float *q, const float *k_cache,
                                      const float *weights, int heads,
                                      int head_dim, float scale, int n,
                                      float *scores, cudaStream_t stream)
{
    if (!q || !k_cache || !weights || !scores || heads <= 0 || head_dim <= 0 ||
        n < 0)
        return cudaErrorInvalidValue;
    if (n == 0) return cudaSuccess;
    indexer_scores_kernel<<<Grid(n), kBlock, 0, stream>>>(
        q, k_cache, weights, heads, head_dim, scale, n, scores);
    return cudaGetLastError();
}

cudaError_t RunGigaChatDsaTopkSelect(const float *scores, int n, int k,
                                   int32_t *out_idx, int *d_count,
                                   cudaStream_t stream)
{
    if (!scores || !out_idx || !d_count || n <= 0 || k <= 0)
        return cudaErrorInvalidValue;
    if (k >= n)
    {
        // Identity: every past position is selected.
        topk_identity_kernel<<<Grid(n), kBlock, 0, stream>>>(n, out_idx,
                                                             d_count);
        return cudaGetLastError();
    }
    topk_select_kernel<<<1, kBlock, 0, stream>>>(scores, n, k, out_idx,
                                                 d_count);
    return cudaGetLastError();
}

cudaError_t RunGigaChatDsaIndexerStep(const float *q_latent_scaled,
                                    const uint8_t *hidden_bf16,
                                    const GcDsaIndexerWeights &w,
                                    const GcDsaIndexerDims &d,
                                    const float *cos_t, const float *sin_t,
                                    int pos, int capacity, float *k_cache,
                                    GcDsaIndexerScratch &sc, int32_t *out_idx,
                                    int *out_count, cudaStream_t stream)
{
    if (!q_latent_scaled || !hidden_bf16 || !k_cache || !out_idx || !out_count ||
        !sc.d_base)
        return cudaErrorInvalidValue;
    if (pos < 0 || pos >= capacity) return cudaErrorInvalidValue;
    if (w.wq_b.rows != d.heads * d.head_dim || w.wq_b.cols != d.q_lora)
        return cudaErrorInvalidValue;

    __half *q_h = static_cast<__half *>(sc.q_h);
    __half *x_h = static_cast<__half *>(sc.x_h);
    float *q_f = static_cast<float *>(sc.q_f);
    float *k_f = static_cast<float *>(sc.k_f);
    float *w_f = static_cast<float *>(sc.w_f);
    float *scores = static_cast<float *>(sc.scores);
    const __nv_bfloat16 *x_bf =
        reinterpret_cast<const __nv_bfloat16 *>(hidden_bf16);
    const __nv_bfloat16 *wk = reinterpret_cast<const __nv_bfloat16 *>(w.d_wk);
    const __nv_bfloat16 *wp =
        reinterpret_cast<const __nv_bfloat16 *>(w.d_weights_proj);

#define GC4_TRY(expr)                     \
    do {                                  \
        const cudaError_t _e = (expr);     \
        if (_e != cudaSuccess) return _e; \
    } while (false)

    // q = wq_b(alpha_q * q_latent) through the neutral NVFP4 GEMV. The caller
    // passes the raw latent and the scale (the reference scales it before
    // q_b_proj, and the indexer consumes the same scaled latent).
    scale_f32_to_half_kernel<<<Grid(d.q_lora, 128), 128, 0, stream>>>(
        q_latent_scaled, q_h, d.q_lora, d.latent_scale);
    GC4_TRY(cudaGetLastError());
    GC4_TRY(lsx_nvfp4::RunLsxNvfp4Gemm(stream, q_h, w.wq_b.d_packed,
                                       w.wq_b.d_scales, w.wq_b.global_scale,
                                       q_h, 1, w.wq_b.cols, w.wq_b.rows));
    lsxcommon::kernels::F16ToF32Kernel<<<Grid(d.heads * d.head_dim, 128), 128,
                                         0, stream>>>(
        q_h, q_f, d.heads * d.head_dim);
    GC4_TRY(cudaGetLastError());

    // k = k_norm(wk(x)); weights = weights_proj(x) * heads^-0.5.
    GC4_TRY(gemv::RunLxGemmVec(stream, w.d_wk, x_bf, 1, d.head_dim, d.hidden,
                               gemv::LxDtype::kBF16, gemv::LxDtype::kBF16, 1.0f,
                               k_f, gemv::LxDtype::kF32, 0.0f));
    GC4_TRY(RunGigaChatDsaIndexerKNorm(k_f, w.d_k_norm, d.head_dim,
                                      d.onorm_eps, stream));
    GC4_TRY(gemv::RunLxGemmVec(stream, w.d_weights_proj, x_bf, 1, d.heads,
                               d.hidden, gemv::LxDtype::kBF16,
                               gemv::LxDtype::kBF16, 1.0f, w_f,
                               gemv::LxDtype::kF32, 0.0f));
    weights_scale_kernel<<<Grid(d.heads, 128), 128, 0, stream>>>(
        w_f, d.heads, 1.0f / sqrtf(static_cast<float>(d.heads)));
    GC4_TRY(cudaGetLastError());

    // Non-interleaved rope on the first `rope` lanes.
    GC4_TRY(RunGigaChatDsaIndexerRope(q_f, k_f, d.heads, d.head_dim, d.rope,
                                    cos_t, sin_t, pos, stream));

    // Append this position's key (post-rope) to the layer cache.
    write_k_cache_kernel<<<Grid(d.head_dim, 128), 128, 0, stream>>>(
        k_f, d.head_dim, k_cache, pos);
    GC4_TRY(cudaGetLastError());

    const int n = pos + 1;
    GC4_TRY(RunGigaChatDsaIndexerScores(q_f, k_cache, w_f, d.heads, d.head_dim,
                                       d.score_scale, n, scores, stream));
    GC4_TRY(RunGigaChatDsaTopkSelect(scores, n, d.topk, out_idx, out_count,
                                   stream));
#undef GC4_TRY
    return cudaSuccess;
}

} // namespace lsxcommon::gigachat
