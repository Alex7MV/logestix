#include "deepseek_moe_v41_inference.h"

#include <cuda_fp16.h>

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

} // namespace deepseek_v41
} // namespace lsxcommon
