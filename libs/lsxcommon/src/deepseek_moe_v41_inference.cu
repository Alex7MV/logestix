#include "deepseek_moe_v41_inference.h"
#include "lsxcommon/lsx_gemv.h"

#include <cuda_fp16.h>

#include "lsxcommon/lsx_inference_nvfp4.h"

namespace lsxcommon {
namespace deepseek_v41 {

namespace {

__device__ __forceinline__ float HcSigmoid(float v)
{
    return 1.0f / (1.0f + expf(-v));
}

__global__ void rmsnorm_half_kernel(const __half *__restrict__ x,
                                    const __half *__restrict__ gamma,
                                    __half *__restrict__ out, int n, float eps)
{
    __shared__ float s_red[32];
    const int tid = threadIdx.x;
    double ss = 0.0;
    for (int i = tid; i < n; i += blockDim.x)
        ss += static_cast<double>(__half2float(x[i])) *
              static_cast<double>(__half2float(x[i]));
    float v = static_cast<float>(ss);
    const int lane = tid & 31, warp = tid >> 5;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
    if (lane == 0) s_red[warp] = v;
    __syncthreads();
    const int nwarp = blockDim.x >> 5;
    if (warp == 0) {
        float w = (lane < nwarp) ? s_red[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            w += __shfl_xor_sync(0xFFFFFFFFu, w, off);
        if (lane == 0) s_red[0] = w;
    }
    __syncthreads();
    const float inv = 1.0f / sqrtf(s_red[0] / static_cast<float>(n) + eps);
    for (int i = tid; i < n; i += blockDim.x)
        out[i] = __float2half_rn(__half2float(gamma[i]) *
                                 __half2float(x[i]) * inv);
}

__global__ void rope_half_kernel(__half *__restrict__ x, int rows, int row_len,
                                 int rope_dim, const float *__restrict__ cos_row,
                                 const float *__restrict__ sin_row,
                                 bool inverse)
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

// One block per token. `mix_hc` warps each reduce one hc_fn row; warp 0 also
// produces the stream rsqrt; thread 0 splits and Sinkhorns the comb matrix.
__global__ void hc_mixes_kernel(const float *__restrict__ x,
                                const float *__restrict__ hc_fn,
                                const float *__restrict__ hc_scale,
                                const float *__restrict__ hc_base, int hc,
                                int dim, int iters, float eps, float *pre,
                                float *post, float *comb)
{
    const int mix_hc = (2 + hc) * hc;
    const int total = hc * dim;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    __shared__ float s_rsqrt;
    __shared__ float s_mix[64];

    if (warp == 0) {  // stream rsqrt
        double ss = 0.0;
        for (int k = lane; k < total; k += 32)
            ss += static_cast<double>(x[k]) * x[k];
        float v = static_cast<float>(ss);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
        if (lane == 0)
            s_rsqrt = 1.0f / sqrtf(v / static_cast<float>(total) + eps);
    }
    if (warp < mix_hc) {
        float d = 0.0f;
        for (int k = lane; k < total; k += 32)
            d += hc_fn[warp * total + k] * x[k];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            d += __shfl_xor_sync(0xFFFFFFFFu, d, off);
        if (lane == 0) s_mix[warp] = d;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;

    for (int m = 0; m < mix_hc; ++m) s_mix[m] = s_mix[m] * s_rsqrt;
    const float *mx = s_mix;
    for (int j = 0; j < hc; ++j) {
        pre[j] = HcSigmoid(mx[j] * hc_scale[0] + hc_base[j]) + eps;
        post[j] = 2.0f *
                  HcSigmoid(mx[j + hc] * hc_scale[1] + hc_base[j + hc]);
    }
    for (int j = 0; j < hc; ++j)
        for (int k = 0; k < hc; ++k)
            comb[j * hc + k] =
                mx[j * hc + k + hc * 2] * hc_scale[2] +
                hc_base[j * hc + k + hc * 2];
    // comb = comb.softmax(-1) + eps
    for (int j = 0; j < hc; ++j) {
        float m = comb[j * hc];
        for (int k = 1; k < hc; ++k) m = fmaxf(m, comb[j * hc + k]);
        float s = 0.0f;
        for (int k = 0; k < hc; ++k) {
            comb[j * hc + k] = expf(comb[j * hc + k] - m);
            s += comb[j * hc + k];
        }
        for (int k = 0; k < hc; ++k)
            comb[j * hc + k] = comb[j * hc + k] / s + eps;
    }
    // comb = comb / (comb.sum(-2) + eps)
    for (int k = 0; k < hc; ++k) {
        float s = 0.0f;
        for (int j = 0; j < hc; ++j) s += comb[j * hc + k];
        for (int j = 0; j < hc; ++j)
            comb[j * hc + k] = comb[j * hc + k] / (s + eps);
    }
    for (int it = 0; it < iters - 1; ++it) {
        for (int j = 0; j < hc; ++j) {  // row normalize
            float s = 0.0f;
            for (int k = 0; k < hc; ++k) s += comb[j * hc + k];
            for (int k = 0; k < hc; ++k)
                comb[j * hc + k] = comb[j * hc + k] / (s + eps);
        }
        for (int k = 0; k < hc; ++k) {  // column normalize
            float s = 0.0f;
            for (int j = 0; j < hc; ++j) s += comb[j * hc + k];
            for (int j = 0; j < hc; ++j)
                comb[j * hc + k] = comb[j * hc + k] / (s + eps);
        }
    }
}

__global__ void hc_pre_kernel(const float *__restrict__ x,
                              const float *__restrict__ pre_mix,
                              float *__restrict__ out, int hc, int dim)
{
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;
    float acc = 0.0f;
    for (int c = 0; c < hc; ++c) acc += pre_mix[c] * x[c * dim + d];
    out[d] = acc;
}

__global__ void hc_post_kernel(const float *__restrict__ x,
                               const float *__restrict__ residual,
                               const float *__restrict__ post,
                               const float *__restrict__ comb,
                               float *__restrict__ out, int hc, int dim)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hc * dim) return;
    const int c = idx / dim;
    const int d = idx % dim;
    float acc = post[c] * x[d];
    for (int cp = 0; cp < hc; ++cp) acc += comb[cp * hc + c] * residual[cp * dim + d];
    out[idx] = acc;
}

__global__ void sparse_attn_half_kernel(const __half *__restrict__ q,
                                        const __half *__restrict__ kv,
                                        const float *__restrict__ sink,
                                        const int *__restrict__ idxs,
                                        __half *__restrict__ o, int heads,
                                        int d, int topk, float scale)
{
    const int row = blockIdx.x;
    const int head = blockIdx.y;
    const int lane = threadIdx.x;      // one warp per (row, head)
    const int per = d / 32;
    float acc[32];
    for (int k = 0; k < per; ++k) acc[k] = 0.0f;
    float m = -1e30f, s = 0.0f;

    const __half *qh = q + (static_cast<size_t>(row) * heads + head) * d;
    const int *ridx = idxs + static_cast<size_t>(row) * topk;
    for (int t = 0; t < topk; ++t) {
        const int idx = ridx[t];
        float dot = 0.0f;
        const __half *kh = nullptr;
        if (idx >= 0) {
            kh = kv + static_cast<size_t>(idx) * d;
            for (int k = 0; k < per; ++k)
                dot += __half2float(qh[lane + k * 32]) *
                       __half2float(kh[lane + k * 32]);
        } else {
            dot = -INFINITY;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            dot += __shfl_xor_sync(0xFFFFFFFFu, dot, off);
        dot *= scale;
        const float m_new = fmaxf(m, dot);
        const float corr = expf(m - m_new);
        s *= corr;
        if (idx >= 0) {
            const float p = expf(dot - m_new);
            s += p;
            for (int k = 0; k < per; ++k)
                acc[k] = acc[k] * corr + p * __half2float(kh[lane + k * 32]);
        } else {
            for (int k = 0; k < per; ++k) acc[k] *= corr;
        }
        m = m_new;
    }
    const float sinkv = sink ? sink[head] : -INFINITY;
    s += expf(sinkv - m);
    const float inv = (s > 0.0f) ? (1.0f / s) : 0.0f;
    __half *oh = o + (static_cast<size_t>(row) * heads + head) * d;
    for (int k = 0; k < per; ++k)
        oh[lane + k * 32] = __float2half_rn(acc[k] * inv);
}

__global__ void compressor_pool_kernel(const float *__restrict__ kv,
                                       const float *__restrict__ score,
                                       int ratio, int d,
                                       float *__restrict__ out)
{
    const int g = blockIdx.x;
    const int j = threadIdx.x;
    if (j >= d) return;
    const float *kvg = kv + static_cast<size_t>(g) * ratio * d;
    const float *scg = score + static_cast<size_t>(g) * ratio * d;
    float m = scg[j];
    for (int r = 1; r < ratio; ++r) m = fmaxf(m, scg[r * d + j]);
    float s = 0.0f, acc = 0.0f;
    for (int r = 0; r < ratio; ++r) {
        const float e = expf(scg[r * d + j] - m);
        s += e;
        acc += e * kvg[r * d + j];
    }
    out[g * d + j] = acc / s;
}

__global__ void indexer_scores_kernel(const __half *__restrict__ q, int heads,
                                      int dim, const __half *__restrict__ k,
                                      int n_pos, const float *__restrict__ w,
                                      float *__restrict__ out)
{
    const int pos = blockIdx.x;
    const int lane = threadIdx.x;   // one warp per position
    const __half *kp = k + static_cast<size_t>(pos) * dim;
    float acc = 0.0f;
    for (int h = 0; h < heads; ++h) {
        const __half *qh = q + static_cast<size_t>(h) * dim;
        float d = 0.0f;
        for (int j = lane; j < dim; j += 32)
            d += __half2float(qh[j]) * __half2float(kp[j]);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            d += __shfl_xor_sync(0xFFFFFFFFu, d, off);
        acc += fmaxf(d, 0.0f) * w[h];
    }
    if (lane == 0) out[pos] = acc;
}

__global__ void expert_swiglu_kernel(const float *__restrict__ gate,
                                     const float *__restrict__ up,
                                     float *__restrict__ out, int n, float limit)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float u = up[i];
    float g = gate[i];
    if (limit > 0.0f) {
        u = fminf(fmaxf(u, -limit), limit);
        g = fminf(g, limit);
    }
    out[i] = (g / (1.0f + expf(-g))) * u;
}

__global__ void embed_lookup_kernel(const __half *__restrict__ embed,
                                    int hidden, const int *__restrict__ ids,
                                    __half *__restrict__ out)
{
    const int tok = blockIdx.x;
    const int id = ids[tok];
    const __half *src = embed + static_cast<size_t>(id) * hidden;
    __half *dst = out + static_cast<size_t>(tok) * hidden;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) dst[j] = src[j];
}

__global__ void hc_expand_kernel(const __half *__restrict__ x,
                                 float *__restrict__ out, int dim, int hc)
{
    const int c = blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;
    out[c * dim + d] = __half2float(x[d]);
}

__global__ void cast_f32_to_half_kernel(const float *__restrict__ src,
                                        __half *__restrict__ dst, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half_rn(src[i]);
}

__global__ void cast_half_to_f32_kernel(const __half *__restrict__ src,
                                        float *__restrict__ dst, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

__global__ void fill_f32_kernel(float *__restrict__ dst, int n, float value)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = value;
}

__global__ void scale_f32_inplace_kernel(float *__restrict__ x, int n,
                                         float s)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

__global__ void accum_half_into_f32_kernel(const __half *__restrict__ src,
                                           float *__restrict__ accum, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) accum[i] += __half2float(src[i]);
}

// torch F.softplus default: threshold 20 returns the input unchanged.
__device__ __forceinline__ float Softplus(float v)
{
    return (v > 20.0f) ? v : log1pf(expf(v));
}

__global__ void router_topk_kernel(const float *__restrict__ scores,
                                   const float *__restrict__ bias, int n,
                                   int topk, float temp, float route_scale,
                                   bool norm, int *__restrict__ out_idx,
                                   float *__restrict__ out_w)
{
    extern __shared__ float sc[];   // n softplus-sqrt scores
    __shared__ int s_used[1024];
    __shared__ int s_idx[32];
    __shared__ float s_w[32];
    const int t = threadIdx.x;
    if (t < n) {
        const float v = scores[t] / temp;
        sc[t] = sqrtf(Softplus(v));
        s_used[t] = 0;
    }
    __syncthreads();
    if (t != 0) return;
    for (int k = 0; k < topk; ++k) {
        int best = -1;
        float bestval = 0.0f;
        for (int i = 0; i < n; ++i) {
            if (s_used[i]) continue;
            const float v = sc[i] + (bias ? bias[i] : 0.0f);
            if (best < 0 || v > bestval) { bestval = v; best = i; }
        }
        s_used[best] = 1;
        s_idx[k] = best;
        s_w[k] = sc[best];
    }
    float sum = 0.0f;
    for (int k = 0; k < topk; ++k) sum += s_w[k];
    for (int k = 0; k < topk; ++k) {
        float w = s_w[k];
        if (norm && topk > 1) w /= (sum + 1e-20f);
        out_idx[k] = s_idx[k];
        out_w[k] = w * route_scale;
    }
}

} // namespace

cudaError_t LaunchRmsNormHalf(const uint16_t *x, const uint16_t *gamma,
                              uint16_t *out, int n, float eps,
                              cudaStream_t stream)
{
    if (n <= 0) return cudaSuccess;
    rmsnorm_half_kernel<<<1, 256, 0, stream>>>(
        reinterpret_cast<const __half *>(x),
        reinterpret_cast<const __half *>(gamma),
        reinterpret_cast<__half *>(out), n, eps);
    return cudaGetLastError();
}

cudaError_t LaunchRopeHalf(uint16_t *x, int rows, int row_len, int rope_dim,
                           const float *cos_row, const float *sin_row,
                           bool inverse, cudaStream_t stream)
{
    if (rows <= 0 || rope_dim <= 0) return cudaSuccess;
    const int half = rope_dim / 2;
    rope_half_kernel<<<rows, std::min(half, 256), 0, stream>>>(
        reinterpret_cast<__half *>(x), rows, row_len, rope_dim, cos_row,
        sin_row, inverse);
    return cudaGetLastError();
}

cudaError_t LaunchHcMixes(const float *x, const float *hc_fn,
                          const float *hc_scale, const float *hc_base, int hc,
                          int dim, int sinkhorn_iters, float eps, float *pre,
                          float *post, float *comb, cudaStream_t stream)
{
    const int mix_hc = (2 + hc) * hc;
    const int threads = 32 * mix_hc;
    hc_mixes_kernel<<<1, threads, 0, stream>>>(x, hc_fn, hc_scale, hc_base, hc,
                                               dim, sinkhorn_iters, eps, pre,
                                               post, comb);
    return cudaGetLastError();
}

cudaError_t LaunchHcPre(const float *x, const float *pre_mix, float *out,
                        int hc, int dim, cudaStream_t stream)
{
    const int block = 256;
    hc_pre_kernel<<<(dim + block - 1) / block, block, 0, stream>>>(
        x, pre_mix, out, hc, dim);
    return cudaGetLastError();
}

cudaError_t LaunchHcPost(const float *x, const float *residual,
                         const float *post, const float *comb, float *out,
                         int hc, int dim, cudaStream_t stream)
{
    const int block = 256;
    const int total = hc * dim;
    hc_post_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
        x, residual, post, comb, out, hc, dim);
    return cudaGetLastError();
}

cudaError_t LaunchSparseAttnHalf(const uint16_t *q, int q_rows, int heads,
                                 int d, const uint16_t *kv, int kv_rows,
                                 const float *sink, const int32_t *idxs,
                                 int topk, uint16_t *out, float softmax_scale,
                                 cudaStream_t stream)
{
    (void)kv_rows;
    if (q_rows <= 0 || heads <= 0 || d <= 0 || topk <= 0)
        return cudaSuccess;
    if (d % 32 != 0) return cudaErrorInvalidValue;
    const dim3 grid(q_rows, heads);
    sparse_attn_half_kernel<<<grid, 32, 0, stream>>>(
        reinterpret_cast<const __half *>(q),
        reinterpret_cast<const __half *>(kv),
        sink, reinterpret_cast<const int *>(idxs),
        reinterpret_cast<__half *>(out), heads, d, topk, softmax_scale);
    return cudaGetLastError();
}

cudaError_t LaunchCompressorPool(const float *kv, const float *score,
                                 int n_groups, int ratio, int d,
                                 int first_token, float *out,
                                 cudaStream_t stream)
{
    if (n_groups <= 0 || ratio <= 0 || d <= 0) return cudaSuccess;
    compressor_pool_kernel<<<n_groups, d, 0, stream>>>(
        kv + static_cast<size_t>(first_token) * d,
        score + static_cast<size_t>(first_token) * d, ratio, d, out);
    return cudaGetLastError();
}

cudaError_t LaunchIndexerScores(const uint16_t *q, int heads, int dim,
                                const uint16_t *k, int n_pos, const float *w,
                                float *out, cudaStream_t stream)
{
    if (n_pos <= 0 || heads <= 0 || dim <= 0) return cudaSuccess;
    indexer_scores_kernel<<<n_pos, 32, 0, stream>>>(
        reinterpret_cast<const __half *>(q), heads, dim,
        reinterpret_cast<const __half *>(k), n_pos, w, out);
    return cudaGetLastError();
}

cudaError_t LaunchExpertSwiglu(const float *gate, const float *up, float *out,
                               int n, float limit, cudaStream_t stream)
{
    if (n <= 0) return cudaSuccess;
    const int block = 256;
    expert_swiglu_kernel<<<(n + block - 1) / block, block, 0, stream>>>(
        gate, up, out, n, limit);
    return cudaGetLastError();
}

cudaError_t LaunchGemmFp16(cudaStream_t stream,
                           const uint16_t *A, const uint16_t *B, uint16_t *C,
                           int M, int N, int K)
{
    if (M <= 0 || N <= 0 || K <= 0) return cudaSuccess;
    // Row-major C[M,N] = A[M,K]*B[K,N].
    if (N == 1) {
        // Shared-expert down projection: C[M] = A[M,K]*B[K]. One row-major
        // GEMV over the M weight rows -- replacing the per-row loop removed
        // ~200k kernel launches per forward pass (99.6% of GPU time).
        return gemv::RunLxGemmVec(
            stream, A, B, 1, M, K, gemv::LxDtype::kF16, gemv::LxDtype::kF16,
            1.0f, C, gemv::LxDtype::kF16, 0.0f);
    }
    // General N: B is [K,N], so each output row is a transposed-W GEMV over
    // B. Device-independent (numeric-contract 1.1.1).
    for (int m = 0; m < M; ++m) {
        const cudaError_t e = gemv::RunLxGemmVecWT(
            stream, B, A + static_cast<int64_t>(m) * K, 1, N, K,
            gemv::LxDtype::kF16, gemv::LxDtype::kF16, 1.0f,
            C + static_cast<int64_t>(m) * N, gemv::LxDtype::kF16, 0.0f);
        if (e != cudaSuccess) return e;
    }
    return cudaSuccess;
}

cudaError_t LaunchHeadLogits(cudaStream_t stream,
                             const uint16_t *w, const uint16_t *x,
                             float *logits, int vocab, int hidden)
{
    if (vocab <= 0 || hidden <= 0) return cudaSuccess;
    return gemv::RunLxGemmVec(stream, w, x, 1, vocab, hidden,
                              gemv::LxDtype::kF16, gemv::LxDtype::kF16, 1.0f,
                              logits, gemv::LxDtype::kF32, 0.0f);
}

cudaError_t LaunchEmbedLookup(const uint16_t *embed, int vocab, int hidden,
                              const int32_t *ids, int n, uint16_t *out,
                              cudaStream_t stream)
{
    (void)vocab;
    if (n <= 0 || hidden <= 0) return cudaSuccess;
    embed_lookup_kernel<<<n, 256, 0, stream>>>(
        reinterpret_cast<const __half *>(embed), hidden,
        reinterpret_cast<const int *>(ids), reinterpret_cast<__half *>(out));
    return cudaGetLastError();
}

cudaError_t LaunchHcExpand(const uint16_t *x, int dim, int hc, float *out,
                           cudaStream_t stream)
{
    if (dim <= 0 || hc <= 0) return cudaSuccess;
    const int block = 256;
    const dim3 grid((dim + block - 1) / block, hc);
    hc_expand_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half *>(x), out, dim, hc);
    return cudaGetLastError();
}

cudaError_t LaunchCastF32ToHalf(const float *src, uint16_t *dst, int n,
                                cudaStream_t stream)
{
    if (n <= 0) return cudaSuccess;
    const int block = 256;
    cast_f32_to_half_kernel<<<(n + block - 1) / block, block, 0, stream>>>(
        src, reinterpret_cast<__half *>(dst), n);
    return cudaGetLastError();
}

cudaError_t LaunchGemvFp16ToF32(cudaStream_t stream,
                                const uint16_t *w, const uint16_t *x, float *y,
                                int M, int K)
{
    if (M <= 0 || K <= 0) return cudaSuccess;
    return gemv::RunLxGemmVec(stream, w, x, 1, M, K, gemv::LxDtype::kF16,
                              gemv::LxDtype::kF16, 1.0f, y,
                              gemv::LxDtype::kF32, 0.0f);
}

cudaError_t SharedExpertForward(cudaStream_t stream,
                                const uint16_t *x, int hidden,
                                const uint16_t *w1, const uint16_t *w3,
                                const uint16_t *w2, int inter, float limit,
                                uint16_t *out)
{
    float *gate = nullptr, *up = nullptr, *mid = nullptr;
    uint16_t *mid_h = nullptr;
    if (cudaMalloc(&gate, static_cast<size_t>(inter) * 4) != cudaSuccess ||
        cudaMalloc(&up, static_cast<size_t>(inter) * 4) != cudaSuccess ||
        cudaMalloc(&mid, static_cast<size_t>(inter) * 4) != cudaSuccess ||
        cudaMalloc(&mid_h, static_cast<size_t>(inter) * 2) != cudaSuccess) {
        if (gate) cudaFree(gate);
        if (up) cudaFree(up);
        if (mid) cudaFree(mid);
        if (mid_h) cudaFree(mid_h);
        return cudaErrorInvalidValue;
    }
    cudaError_t e = LaunchGemvFp16ToF32(stream, w1, x, gate, inter,
                                        hidden);
    if (e == cudaSuccess)
        e = LaunchGemvFp16ToF32(stream, w3, x, up, inter, hidden);
    if (e == cudaSuccess)
        e = LaunchExpertSwiglu(gate, up, mid, inter, limit, stream);
    if (e == cudaSuccess)
        e = LaunchCastF32ToHalf(mid, mid_h, inter, stream);
    if (e == cudaSuccess)
        e = LaunchGemmFp16(stream, w2, mid_h, out, hidden, 1, inter);
    cudaFree(gate);
    cudaFree(up);
    cudaFree(mid);
    cudaFree(mid_h);
    return e;
}

cudaError_t LaunchFillF32(float *dst, int n, float value, cudaStream_t stream)
{
    if (n <= 0) return cudaSuccess;
    const int block = 256;
    fill_f32_kernel<<<(n + block - 1) / block, block, 0, stream>>>(dst, n,
                                                                   value);
    return cudaGetLastError();
}

cudaError_t LaunchCastHalfToF32(const uint16_t *src, float *dst, int n,
                                cudaStream_t stream)
{
    if (n <= 0) return cudaSuccess;
    const int block = 256;
    cast_half_to_f32_kernel<<<(n + block - 1) / block, block, 0, stream>>>(
        reinterpret_cast<const __half *>(src), dst, n);
    return cudaGetLastError();
}

cudaError_t LaunchAccumulateHalfToF32(const uint16_t *src, float *acc, int n,
                                      cudaStream_t stream)
{
    if (n <= 0) return cudaSuccess;
    const int block = 256;
    accum_half_into_f32_kernel<<<(n + block - 1) / block, block, 0, stream>>>(
        reinterpret_cast<const __half *>(src), acc, n);
    return cudaGetLastError();
}

cudaError_t LaunchRouterTopK(cudaStream_t stream,
                             const uint16_t *x, const uint16_t *gate_w,
                             const float *bias, int n_routed, int hidden,
                             int topk, float temp, float route_scale, bool norm,
                             int32_t *out_idx, float *out_w)
{
    if (n_routed <= 0 || hidden <= 0 || topk <= 0) return cudaSuccess;
    if (n_routed > 1024) return cudaErrorInvalidValue;
    if (topk > n_routed) topk = n_routed;
    if (topk > 32) topk = 32;
    float *scores = nullptr;
    if (cudaMalloc(&scores, sizeof(float) * n_routed) != cudaSuccess)
        return cudaGetLastError();
    cudaError_t e = LaunchGemvFp16ToF32(stream, gate_w, x, scores,
                                        n_routed, hidden);
    if (e == cudaSuccess) {
        router_topk_kernel<<<1, n_routed, sizeof(float) * n_routed, stream>>>(
            scores, bias, n_routed, topk, temp, route_scale, norm,
            reinterpret_cast<int *>(out_idx), out_w);
        e = cudaGetLastError();
    }
    cudaFree(scores);
    return e;
}

std::size_t ExpertFfnScratchBytes(int inter, int hidden)
{
    size_t bytes = 0;
    auto bump = [&](size_t n) { bytes += (n + 255) / 256 * 256; };
    bump(static_cast<size_t>(inter) * 2);      // gate_h
    bump(static_cast<size_t>(inter) * 2);      // up_h
    bump(static_cast<size_t>(inter) * 4);      // gate_f
    bump(static_cast<size_t>(inter) * 4);      // up_f
    bump(static_cast<size_t>(inter) * 4);      // mid_f
    bump(static_cast<size_t>(inter) * 2);      // mid_h
    bump(static_cast<size_t>(hidden) * 2);     // out_h
    return bytes;
}

cudaError_t LaunchExpertFfnScratch(cudaStream_t stream,
                                   const uint16_t *x, int hidden,
                                   const DeviceExpertWeight &w1,
                                   const DeviceExpertWeight &w3,
                                   const DeviceExpertWeight &w2, int inter,
                                   float swiglu_limit, float weight,
                                   float *accum_f32, void *scratch,
                                   std::size_t scratch_bytes)
{
    if (hidden <= 0 || inter <= 0) return cudaSuccess;
    if (!w1.packed || !w1.scales || !w3.packed || !w3.scales || !w2.packed ||
        !w2.scales)
        return cudaErrorInvalidValue;
    if (!scratch || scratch_bytes < ExpertFfnScratchBytes(inter, hidden))
        return cudaErrorInvalidValue;

    uint8_t *p = static_cast<uint8_t *>(scratch);
    auto take = [&](size_t n) {
        uint8_t *r = p;
        p += (n + 255) / 256 * 256;
        return r;
    };
    __half *gate_h = reinterpret_cast<__half *>(take(inter * 2));
    __half *up_h = reinterpret_cast<__half *>(take(inter * 2));
    float *gate_f = reinterpret_cast<float *>(take(inter * 4));
    float *up_f = reinterpret_cast<float *>(take(inter * 4));
    float *mid_f = reinterpret_cast<float *>(take(inter * 4));
    __half *mid_h = reinterpret_cast<__half *>(take(inter * 2));
    __half *out_h = reinterpret_cast<__half *>(take(hidden * 2));

    const __half *xh = reinterpret_cast<const __half *>(x);
    cudaError_t e = lsx_nvfp4::RunLsxNvfp4Gemm(
        stream, xh, w1.packed, w1.scales, w1.global_scale, gate_h, 1, hidden,
        inter);
    if (e == cudaSuccess)
        e = lsx_nvfp4::RunLsxNvfp4Gemm(stream, xh, w3.packed, w3.scales,
                                       w3.global_scale, up_h, 1, hidden, inter);
    if (e == cudaSuccess) {
        cast_half_to_f32_kernel<<<(inter + 255) / 256, 256, 0, stream>>>(
            gate_h, gate_f, inter);
        cast_half_to_f32_kernel<<<(inter + 255) / 256, 256, 0, stream>>>(
            up_h, up_f, inter);
        e = cudaGetLastError();
    }
    if (e == cudaSuccess)
        e = LaunchExpertSwiglu(gate_f, up_f, mid_f, inter, swiglu_limit, stream);
    // The routing weight scales the SwiGLU activation before w2 (model.py:850).
    if (e == cudaSuccess && weight != 1.0f) {
        scale_f32_inplace_kernel<<<(inter + 255) / 256, 256, 0, stream>>>(
            mid_f, inter, weight);
        e = cudaGetLastError();
    }
    if (e == cudaSuccess)
        e = LaunchCastF32ToHalf(mid_f, reinterpret_cast<uint16_t *>(mid_h),
                                inter, stream);
    if (e == cudaSuccess)
        e = lsx_nvfp4::RunLsxNvfp4Gemm(stream, mid_h, w2.packed, w2.scales,
                                       w2.global_scale, out_h, 1, inter,
                                       hidden);
    if (e == cudaSuccess) {
        accum_half_into_f32_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(
            out_h, accum_f32, hidden);
        e = cudaGetLastError();
    }
    return e;
}

cudaError_t LaunchExpertFfn(cudaStream_t stream,
                            const uint16_t *x, int hidden,
                            const DeviceExpertWeight &w1,
                            const DeviceExpertWeight &w3,
                            const DeviceExpertWeight &w2, int inter,
                            float swiglu_limit, float weight,
                            float *accum_f32)
{
    void *base = nullptr;
    const size_t bytes = ExpertFfnScratchBytes(inter, hidden);
    if (cudaMalloc(&base, bytes) != cudaSuccess) return cudaGetLastError();
    const cudaError_t e = LaunchExpertFfnScratch(
        stream, x, hidden, w1, w3, w2, inter, swiglu_limit, weight,
        accum_f32, base, bytes);
    cudaFree(base);
    return e;
}

} // namespace deepseek_v41
} // namespace lsxcommon
