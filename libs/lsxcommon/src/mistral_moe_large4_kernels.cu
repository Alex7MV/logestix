// Mistral-Large-4-1T-A52B family kernels (Model Ingestion Protocol stage 5).
// Correctness-first: deterministic reductions, no atomics feeding logits, and
// the shared-memory sandbox rule (thread-0 RMW behind __syncthreads()).

#include "lsxcommon/mistral_moe_large4_kernels.h"

#include "lsxcommon/log.h"
#include "nvfp4_codec.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace lsxcommon {
namespace mistral_large4 {
namespace {

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 v)
{
    return __bfloat162float(v);
}

// ── RoPE ──────────────────────────────────────────────────────────────────
__global__ void rope_interleaved_heads_kernel(float *vec, int heads, int nope,
                                              int rope_dim, const float *cos_t,
                                              const float *sin_t, int pos)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int pairs_per_head = rope_dim / 2;
    const int total = heads * pairs_per_head;
    if (idx >= total) return;
    const int h = idx / pairs_per_head;
    const int j = idx % pairs_per_head;
    const int base = h * (nope + rope_dim) + nope + 2 * j;
    const float c = cos_t[static_cast<int64_t>(pos) * pairs_per_head + j];
    const float s = sin_t[static_cast<int64_t>(pos) * pairs_per_head + j];
    const float x0 = vec[base];
    const float x1 = vec[base + 1];
    vec[base] = x0 * c - x1 * s;
    vec[base + 1] = x1 * c + x0 * s;
}

__global__ void rope_interleaved_vec_kernel(float *vec, int rope_dim,
                                            const float *cos_t,
                                            const float *sin_t, int pos)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= rope_dim / 2) return;
    const float c = cos_t[static_cast<int64_t>(pos) * (rope_dim / 2) + j];
    const float s = sin_t[static_cast<int64_t>(pos) * (rope_dim / 2) + j];
    const float x0 = vec[2 * j];
    const float x1 = vec[2 * j + 1];
    vec[2 * j] = x0 * c - x1 * s;
    vec[2 * j + 1] = x1 * c + x0 * s;
}

// ── ssmax ─────────────────────────────────────────────────────────────────
__global__ void ssmax_scale_kernel(float *q, int heads, int head_dim,
                                   const __nv_bfloat16 *s,
                                   const __nv_bfloat16 *b, int pos)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = heads * head_dim;
    if (idx >= total) return;
    const int h = idx / head_dim;
    // Mirrors the reference: the scale and bias are BF16, log(pos+1) is
    // rounded to BF16, and the product is formed in the activation dtype.
    const float log_pos = logf(static_cast<float>(pos + 1));
    const float scale = __bfloat162float(__float2bfloat16(
        __bfloat162float(s[h]) * __bfloat162float(__float2bfloat16(log_pos)) +
        __bfloat162float(b[h])));
    q[idx] *= scale;
}

__global__ void rms_norm_kernel(const float *x,
                                const __nv_bfloat16 *w, int n, float eps,
                                __nv_bfloat16 *out)
{
    __shared__ float s_ss;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        const float v = x[i];
        acc += v * v;
    }
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    __shared__ float s_part[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) s_part[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0)
    {
        float t = 0.0f;
        const int warps = (blockDim.x + 31) / 32;
        for (int i = 0; i < warps; ++i) t += s_part[i];
        s_ss = t;
    }
    __syncthreads();
    const float inv = rsqrtf(s_ss / static_cast<float>(n) + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        out[i] = __float2bfloat16(x[i] * inv * bf16_to_f32(w[i]));
}

// ── LayerNorm (weight + bias) ─────────────────────────────────────────────
__global__ void layernorm_kernel(const __nv_bfloat16 *x,
                                 const __nv_bfloat16 *w,
                                 const __nv_bfloat16 *b, int n, float eps,
                                 float *out)
{
    __shared__ float s_sum;
    __shared__ float s_sumsq;
    if (threadIdx.x == 0)
    {
        s_sum = 0.0f;
        s_sumsq = 0.0f;
    }
    __syncthreads();
    float sum = 0.0f, sumsq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        const float v = bf16_to_f32(x[i]);
        sum += v;
        sumsq += v * v;
    }
    // Deterministic block reduction: warp shuffles then shared, thread 0 only.
    for (int off = 16; off > 0; off >>= 1)
    {
        sum += __shfl_down_sync(0xffffffffu, sum, off);
        sumsq += __shfl_down_sync(0xffffffffu, sumsq, off);
    }
    __shared__ float s_warp[8];
    __shared__ float s_warp2[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0)
    {
        s_warp[warp] = sum;
        s_warp2[warp] = sumsq;
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        float t = 0.0f, t2 = 0.0f;
        const int warps = (blockDim.x + 31) / 32;
        for (int i = 0; i < warps; ++i)
        {
            t += s_warp[i];
            t2 += s_warp2[i];
        }
        s_sum = t;
        s_sumsq = t2;
    }
    __syncthreads();
    const float mean = s_sum / static_cast<float>(n);
    const float var = s_sumsq / static_cast<float>(n) - mean * mean;
    const float inv = rsqrtf(var + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        out[i] = (bf16_to_f32(x[i]) - mean) * inv * bf16_to_f32(w[i]) +
                 bf16_to_f32(b[i]);
}

// ── Router ────────────────────────────────────────────────────────────────
__global__ void router_kernel(const __nv_bfloat16 *__restrict__ w,
                              const float *__restrict__ x, int E, int K,
                              int H, moe::MoERoutedSlot *slots)
{
    extern __shared__ float smem[];
    float *logits = smem;  // [E]
    for (int e = threadIdx.x; e < E; e += blockDim.x)
    {
        const __nv_bfloat16 *row = w + static_cast<int64_t>(e) * H;
        float acc = 0.0f;
        for (int k = 0; k < H; ++k) acc += bf16_to_f32(row[k]) * x[k];
        logits[e] = acc;
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        // softmax (max-subtracted), then top-K by value with ascending-index
        // tie-break, renormalized.
        float mx = logits[0];
        for (int e = 1; e < E; ++e) mx = fmaxf(mx, logits[e]);
        float den = 0.0f;
        for (int e = 0; e < E; ++e)
        {
            logits[e] = expf(logits[e] - mx);
            den += logits[e];
        }
        const float inv = 1.0f / den;
        for (int e = 0; e < E; ++e) logits[e] *= inv;

        int chosen[moe::kMaxTopK];
        float chosen_v[moe::kMaxTopK];
        for (int i = 0; i < K; ++i)
        {
            int best = -1;
            float bestv = -1.0f;
            for (int e = 0; e < E; ++e)
            {
                bool used = false;
                for (int j = 0; j < i; ++j)
                    if (chosen[j] == e)
                    {
                        used = true;
                        break;
                    }
                if (used) continue;
                if (best < 0 || logits[e] > bestv)
                {
                    best = e;
                    bestv = logits[e];
                }
            }
            chosen[i] = best;
            chosen_v[i] = bestv;
        }
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) sum += chosen_v[i];
        const float rn = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (int i = 0; i < K; ++i)
        {
            slots[i].eid = chosen[i];
            slots[i].coeff = chosen_v[i] * rn;
        }
    }
}

// ── Indexer ───────────────────────────────────────────────────────────────
__global__ void indexer_scores_kernel(const float *__restrict__ q,
                                      const __nv_bfloat16 *__restrict__ k,
                                      const float *__restrict__ weights,
                                      int heads, int head_dim, float scale,
                                      int n, float *__restrict__ scores)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const __nv_bfloat16 *kt = k + static_cast<int64_t>(t) * head_dim;
    float acc = 0.0f;
    for (int h = 0; h < heads; ++h)
    {
        const float *qh = q + static_cast<int64_t>(h) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) dot += qh[d] * bf16_to_f32(kt[d]);
        const float r = fmaxf(dot * scale, 0.0f);
        acc += weights[h] * r;
    }
    scores[t] = acc;
}

__device__ __forceinline__ unsigned ordered_key(float v)
{
    const unsigned u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}

__global__ void topk_identity_kernel(int n, int32_t *__restrict__ out_idx,
                                     int *__restrict__ d_count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out_idx[i] = i;
    if (i == 0) *d_count = n;
}

// Exact k-th largest over `n` floats (monotone uint order) by block-wide
// binary search, then an ascending compacted index list.
__global__ void topk_select_kernel(const float *__restrict__ scores, int n,
                                   int k, int32_t *__restrict__ out_idx,
                                   int *__restrict__ d_count)
{
    __shared__ unsigned s_lo, s_hi, s_mid, s_cnt;
    const int tid = threadIdx.x;
    if (tid == 0)
    {
        s_lo = 0xFFFFFFFFu;
        s_hi = 0x00000000u;
    }
    __syncthreads();
    unsigned lo = 0xFFFFFFFFu, hi = 0x00000000u;
    for (int i = tid; i < n; i += blockDim.x)
    {
        const unsigned u = ordered_key(scores[i]);
        lo = min(lo, u);
        hi = max(hi, u);
    }
    // Block min/max through shared, thread-0 only (sandbox rule).
    __shared__ unsigned s_mn[32], s_mx[32];
    const int lane = tid & 31, warp = tid >> 5;
    for (int off = 16; off > 0; off >>= 1)
    {
        lo = min(lo, __shfl_down_sync(0xffffffffu, lo, off));
        hi = max(hi, __shfl_down_sync(0xffffffffu, hi, off));
    }
    if (lane == 0)
    {
        s_mn[warp] = lo;
        s_mx[warp] = hi;
    }
    __syncthreads();
    if (tid == 0)
    {
        unsigned a = 0xFFFFFFFFu, b = 0x00000000u;
        const int warps = (blockDim.x + 31) / 32;
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

    // Binary search the k-th largest value.
    while (lo < hi)
    {
        if (tid == 0) s_mid = lo + (hi - lo + 1) / 2;
        __syncthreads();
        const unsigned mid = s_mid;
        unsigned c = 0;
        for (int i = tid; i < n; i += blockDim.x)
            if (ordered_key(scores[i]) >= mid) ++c;
        for (int off = 16; off > 0; off >>= 1)
            c += __shfl_down_sync(0xffffffffu, c, off);
        __shared__ unsigned s_part[32];
        if (lane == 0) s_part[warp] = c;
        __syncthreads();
        if (tid == 0)
        {
            unsigned tot = 0;
            const int warps = (blockDim.x + 31) / 32;
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

    // One ascending scan (thread 0): every strictly-greater entry plus exactly
    // `k - gt` entries from the tie group, in ascending index order. The total
    // is exactly k, so a tie can never crowd out a strictly-greater entry.
    if (tid == 0)
    {
        int gt = 0;
        for (int i = 0; i < n; ++i)
            if (ordered_key(scores[i]) > thr) ++gt;
        const int need_eq = k - gt;
        int written = 0, eq = 0;
        for (int i = 0; i < n; ++i)
        {
            const unsigned u = ordered_key(scores[i]);
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

// ── MLA absorption ────────────────────────────────────────────────────────
__global__ void absorb_query_kernel(const __nv_bfloat16 *__restrict__ wkv_b,
                                    const float *__restrict__ q, int heads,
                                    int qk_head, int kn, int v_head,
                                    int kv_lora, float *__restrict__ q_abs)
{
    const int h = blockIdx.x;
    const int kcol = threadIdx.x;
    if (kcol >= kv_lora) return;
    float acc = 0.0f;
    const int row0 = h * (kn + v_head);
    for (int d = 0; d < kn; ++d)
        acc += q[h * qk_head + d] *
               bf16_to_f32(wkv_b[static_cast<int64_t>(row0 + d) * kv_lora +
                                kcol]);
    q_abs[h * kv_lora + kcol] = acc;
}

__global__ void value_project_kernel(const __nv_bfloat16 *__restrict__ wkv_b,
                                     const float *__restrict__ ctx, int heads,
                                     int kn, int v_head, int kv_lora,
                                     float *__restrict__ out)
{
    const int h = blockIdx.x;
    const int vcol = threadIdx.x;
    if (vcol >= v_head) return;
    const int row = h * (kn + v_head) + kn + vcol;
    float acc = 0.0f;
    const __nv_bfloat16 *w = wkv_b + static_cast<int64_t>(row) * kv_lora;
    const float *c = ctx + static_cast<int64_t>(h) * kv_lora;
    for (int k = 0; k < kv_lora; ++k) acc += bf16_to_f32(w[k]) * c[k];
    out[h * v_head + vcol] = acc;
}

// ── elementwise ───────────────────────────────────────────────────────────
__global__ void bf16_to_f32_kernel(const __nv_bfloat16 *src, float *dst, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __bfloat162float(src[i]);
}
__global__ void acc_add_scaled_f16_kernel(float *acc, const __half *src,
                                          const float *scale, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) acc[i] += (*scale) * __half2float(src[i]);
}
__global__ void logit_softcap_kernel(float *logits, int n, float cap)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) logits[i] = tanhf(logits[i] / cap) * cap;
}
__global__ void bf16_to_f16_kernel(const __nv_bfloat16 *src, __half *dst,
                                   int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(__bfloat162float(src[i]));
}
__global__ void f16_to_f32_kernel(const __half *src, float *dst, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}
__global__ void f32_to_f16_kernel(const float *src, __half *dst, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}
__global__ void f32_to_bf16_kernel(const float *src, __nv_bfloat16 *dst, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2bfloat16(src[i]);
}
__global__ void add_inplace_kernel(float *a, const float *b, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

constexpr int kBlock = 256;
inline int Grid(int n) { return (n + kBlock - 1) / kBlock; }

} // namespace

bool BuildLarge4RopeTables(int capacity, const std::vector<float> &alpha,
                           int rope_dim, double theta, double factor,
                           double orig, double beta_fast, double beta_slow,
                           std::vector<float> &cos_t, std::vector<float> &sin_t)
{
    if (capacity <= 0 || rope_dim <= 0 || rope_dim % 2 != 0) return false;
    const int half = rope_dim / 2;
    if (static_cast<int>(alpha.size()) != half) return false;

    const auto correction_dim = [&](double num_rotations) {
        return (rope_dim * std::log(orig / (num_rotations * 2.0 * 3.14159265358979323846))) /
               (2.0 * std::log(theta));
    };
    double low = std::floor(correction_dim(beta_fast));
    double high = std::ceil(correction_dim(beta_slow));
    low = std::max(low, 0.0);
    high = std::min(high, static_cast<double>(rope_dim - 1));
    if (low == high) high += 0.001;

    std::vector<double> inv_freq(static_cast<size_t>(half));
    for (int j = 0; j < half; ++j)
    {
        const double exponent =
            static_cast<double>(2 * j) / static_cast<double>(rope_dim);
        const double base_freq = std::pow(theta, exponent);
        const double extrap = 1.0 / base_freq;
        const double interp = 1.0 / (factor * base_freq);
        double f = (static_cast<double>(j) - low) / (high - low);
        f = std::min(1.0, std::max(0.0, f));
        const double ramp = 1.0 - f;
        inv_freq[static_cast<size_t>(j)] =
            interp * (1.0 - ramp) + extrap * ramp;
    }

    cos_t.assign(static_cast<size_t>(capacity) * half, 0.0f);
    sin_t.assign(static_cast<size_t>(capacity) * half, 0.0f);
    for (int p = 0; p < capacity; ++p)
        for (int j = 0; j < half; ++j)
        {
            const double freq =
                inv_freq[static_cast<size_t>(j)] *
                std::exp(static_cast<double>(alpha[static_cast<size_t>(j)]));
            const double ang = static_cast<double>(p) * freq;
            cos_t[static_cast<size_t>(p) * half + j] =
                static_cast<float>(std::cos(ang));
            sin_t[static_cast<size_t>(p) * half + j] =
                static_cast<float>(std::sin(ang));
        }
    return true;
}

cudaError_t UploadLarge4RopeTables(cudaStream_t stream,
                                   const std::vector<float> &cos_t,
                                   const std::vector<float> &sin_t,
                                   float **d_cos, float **d_sin)
{
    if (cos_t.size() != sin_t.size() || cos_t.empty())
        return cudaErrorInvalidValue;
    if (cudaMalloc(d_cos, cos_t.size() * sizeof(float)) != cudaSuccess)
        return cudaErrorMemoryAllocation;
    if (cudaMalloc(d_sin, sin_t.size() * sizeof(float)) != cudaSuccess)
    {
        cudaFree(*d_cos);
        *d_cos = nullptr;
        return cudaErrorMemoryAllocation;
    }
    if (cudaMemcpyAsync(*d_cos, cos_t.data(), cos_t.size() * sizeof(float),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(*d_sin, sin_t.data(), sin_t.size() * sizeof(float),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess)
        return cudaErrorUnknown;
    return cudaSuccess;
}

cudaError_t LaunchLarge4RopeInterleaved(float *vec, int heads, int head_dim,
                                        int nope, int rope_dim,
                                        const float *d_cos, const float *d_sin,
                                        int pos, cudaStream_t stream)
{
    if (rope_dim <= 0) return cudaSuccess;
    const int total = heads * (rope_dim / 2);
    (void)head_dim;
    rope_interleaved_heads_kernel<<<Grid(total), kBlock, 0, stream>>>(
        vec, heads, nope, rope_dim, d_cos, d_sin, pos);
    return cudaGetLastError();
}

cudaError_t LaunchLarge4RopeInterleavedVec(float *vec, int rope_dim,
                                           const float *d_cos,
                                           const float *d_sin, int pos,
                                           cudaStream_t stream)
{
    if (rope_dim <= 0) return cudaSuccess;
    rope_interleaved_vec_kernel<<<Grid(rope_dim / 2), kBlock, 0, stream>>>(
        vec, rope_dim, d_cos, d_sin, pos);
    return cudaGetLastError();
}

cudaError_t LaunchLarge4SsmaxScale(float *q, int heads, int head_dim,
                                   const __nv_bfloat16 *s,
                                   const __nv_bfloat16 *b, int pos,
                                   cudaStream_t stream)
{
    ssmax_scale_kernel<<<Grid(heads * head_dim), kBlock, 0, stream>>>(
        q, heads, head_dim, s, b, pos);
    return cudaGetLastError();
}

cudaError_t LaunchLarge4RmsNorm(const float *x, const __nv_bfloat16 *w, int n,
                                float eps, __nv_bfloat16 *out,
                                cudaStream_t stream)
{
    rms_norm_kernel<<<1, kBlock, 0, stream>>>(x, w, n, eps, out);
    return cudaGetLastError();
}

cudaError_t LaunchLarge4LayerNorm(const __nv_bfloat16 *x,
                                  const __nv_bfloat16 *w,
                                  const __nv_bfloat16 *b, int n, float eps,
                                  float *out, cudaStream_t stream)
{
    layernorm_kernel<<<1, kBlock, 0, stream>>>(x, w, b, n, eps, out);
    return cudaGetLastError();
}

cudaError_t LaunchLarge4Router(const __nv_bfloat16 *router_w, const float *x,
                               int experts, int top_k, int hidden,
                               moe::MoERoutedSlot *d_slots,
                               cudaStream_t stream)
{
    if (top_k > moe::kMaxTopK) return cudaErrorInvalidValue;
    router_kernel<<<1, kBlock, experts * sizeof(float), stream>>>(
        router_w, x, experts, top_k, hidden, d_slots);
    return cudaGetLastError();
}

cudaError_t LaunchLarge4IndexerScores(const float *q, const __nv_bfloat16 *k,
                                      const float *weights, int heads,
                                      int head_dim, float scale, int n,
                                      float *scores, cudaStream_t stream)
{
    indexer_scores_kernel<<<Grid(n), kBlock, 0, stream>>>(
        q, k, weights, heads, head_dim, scale, n, scores);
    return cudaGetLastError();
}

cudaError_t LaunchLarge4TopkSelect(const float *scores, int n, int k,
                                   int32_t *out_idx, int *d_count,
                                   cudaStream_t stream)
{
    if (n <= 0) return cudaErrorInvalidValue;
    if (k >= n)
    {
        // Identity: every past position selected.
        topk_identity_kernel<<<Grid(n), kBlock, 0, stream>>>(n, out_idx,
                                                             d_count);
        return cudaGetLastError();
    }
    topk_select_kernel<<<1, kBlock, 0, stream>>>(scores, n, k, out_idx,
                                                 d_count);
    return cudaGetLastError();
}

cudaError_t LaunchLarge4AbsorbQuery(const __nv_bfloat16 *wkv_b,
                                    const float *q, int heads, int qk_head,
                                    int kn, int v_head, int kv_lora,
                                    float *q_abs, cudaStream_t stream)
{
    absorb_query_kernel<<<heads, kv_lora, 0, stream>>>(
        wkv_b, q, heads, qk_head, kn, v_head, kv_lora, q_abs);
    return cudaGetLastError();
}

cudaError_t LaunchLarge4ValueProject(const __nv_bfloat16 *wkv_b,
                                     const float *ctx, int heads, int kn,
                                     int v_head, int kv_lora, float *out,
                                     cudaStream_t stream)
{
    value_project_kernel<<<heads, v_head, 0, stream>>>(
        wkv_b, ctx, heads, kn, v_head, kv_lora, out);
    return cudaGetLastError();
}

cudaError_t LaunchLarge4Bf16ToF32(const __nv_bfloat16 *src, float *dst,
                                  int n, cudaStream_t stream)
{
    bf16_to_f32_kernel<<<Grid(n), kBlock, 0, stream>>>(src, dst, n);
    return cudaGetLastError();
}
cudaError_t LaunchLarge4AccAddScaledF16ToF32(float *acc, const __half *src,
                                             const float *d_scale, int n,
                                             cudaStream_t stream)
{
    acc_add_scaled_f16_kernel<<<Grid(n), kBlock, 0, stream>>>(acc, src, d_scale,
                                                              n);
    return cudaGetLastError();
}
cudaError_t LaunchLarge4LogitSoftcap(float *logits, int n, float cap,
                                     cudaStream_t stream)
{
    logit_softcap_kernel<<<Grid(n), kBlock, 0, stream>>>(logits, n, cap);
    return cudaGetLastError();
}
cudaError_t LaunchLarge4Bf16ToF16(const __nv_bfloat16 *src, __half *dst,
                                  int n, cudaStream_t stream)
{
    bf16_to_f16_kernel<<<Grid(n), kBlock, 0, stream>>>(src, dst, n);
    return cudaGetLastError();
}
cudaError_t LaunchLarge4F16ToF32(const __half *src, float *dst, int n,
                                 cudaStream_t stream)
{
    f16_to_f32_kernel<<<Grid(n), kBlock, 0, stream>>>(src, dst, n);
    return cudaGetLastError();
}
cudaError_t LaunchLarge4F32ToF16(const float *src, __half *dst, int n,
                                 cudaStream_t stream)
{
    f32_to_f16_kernel<<<Grid(n), kBlock, 0, stream>>>(src, dst, n);
    return cudaGetLastError();
}
cudaError_t LaunchLarge4F32ToBf16(const float *src, __nv_bfloat16 *dst, int n,
                                  cudaStream_t stream)
{
    f32_to_bf16_kernel<<<Grid(n), kBlock, 0, stream>>>(src, dst, n);
    return cudaGetLastError();
}
cudaError_t LaunchLarge4AddInPlace(float *a, const float *b, int n,
                                   cudaStream_t stream)
{
    add_inplace_kernel<<<Grid(n), kBlock, 0, stream>>>(a, b, n);
    return cudaGetLastError();
}

} // namespace mistral_large4
} // namespace lsxcommon
