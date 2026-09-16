// Nemotron-3.5 full-attention kernels (Model Ingestion Protocol step 5):
// full-dimension RoPE (HF rotate_half) + single-token GQA decode attention.
#include "nemotron_moe_kernels.h"

#include <cuda_runtime.h>

namespace lsxcommon::nemotron_moe {

namespace {

__global__ void rope_kernel(float *__restrict__ x, int n_heads, int head_dim,
                            int pos, float theta)
{
    const int half = head_dim / 2;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_heads * half) return;
    // One thread handles a (head, half-index) pair across the active heads.
    const int h = idx / half;
    const int j = idx % half;
    const size_t base = static_cast<size_t>(h) * head_dim;
    const float freq = powf(theta, -2.0f * static_cast<float>(j) /
                                       static_cast<float>(head_dim));
    const float angle = static_cast<float>(pos) * freq;
    const float c = cosf(angle);
    const float s = sinf(angle);
    const float x1 = x[base + j];
    const float x2 = x[base + half + j];
    x[base + j] = x1 * c - x2 * s;
    x[base + half + j] = x1 * s + x2 * c;
}

__global__ void attn_decode_kernel(const float *__restrict__ q,
                                   const float *__restrict__ k_cache,
                                   const float *__restrict__ v_cache, int heads,
                                   int kv_heads, int head_dim, int seq_len,
                                   int cache_stride, float scale,
                                   float *__restrict__ out)
{
    __shared__ float red[512];
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    if (d >= head_dim) return;  // one thread per head_dim lane
    const int rep = heads / kv_heads;
    const int hk = h / rep;
    const float qd = q[h * head_dim + d];

    float m = -INFINITY;
    float l = 0.0f;
    float o = 0.0f;
    for (int t = 0; t < seq_len; ++t) {
        const size_t kv_base =
            (static_cast<size_t>(hk) * cache_stride + t) * head_dim;
        red[d] = qd * k_cache[kv_base + d];
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (d < s) red[d] += red[d + s];
            __syncthreads();
        }
        const float score = red[0] * scale;
        const float m_new = fmaxf(m, score);
        const float alpha = expf(m - m_new);
        const float p = expf(score - m_new);
        l = l * alpha + p;
        o = o * alpha + p * v_cache[kv_base + d];
        m = m_new;
        __syncthreads();
    }
    out[h * head_dim + d] = o / l;
}

} // namespace

void RunNemotronRope(float *x, int n_heads, int head_dim, int pos, float theta,
                     cudaStream_t stream)
{
    const int half = (head_dim + 1) / 2;
    const int threads = 128;
    const int total = n_heads * half;
    const int blocks = (total + threads - 1) / threads;
    rope_kernel<<<blocks, threads, 0, stream>>>(x, n_heads, head_dim, pos,
                                                theta);
}

void RunNemotronAttnDecode(const float *q, const float *k_cache,
                           const float *v_cache, int heads, int kv_heads,
                           int head_dim, int seq_len, int cache_stride,
                           float scale, float *out, cudaStream_t stream)
{
    // One block per query head; head_dim threads (<= 512).
    attn_decode_kernel<<<heads, head_dim, 0, stream>>>(
        q, k_cache, v_cache, heads, kv_heads, head_dim, seq_len, cache_stride,
        scale, out);
}

} // namespace lsxcommon::nemotron_moe
