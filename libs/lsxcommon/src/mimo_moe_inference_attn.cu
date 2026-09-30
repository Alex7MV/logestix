// MiMo-V2.6-Flash-RL attention prefill kernel. One block per (query, head):
// threads cooperatively load q/k, apply partial rotate-half RoPE, reduce the
// QK dot, softmax (with an optional learned sink column), then mix V. Single
// stream; shared-memory reduction sandbox; no atomicAdd into the output.
#include "lsxcommon/mimo_moe_inference_attn.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>

namespace lsxcommon {
namespace mimo {
namespace {

__device__ __forceinline__ void RopeHalf(float *v, int pos, int rope_dim,
                                         double theta)
{
    const int half = rope_dim >> 1;
    for (int d = threadIdx.x; d < half; d += blockDim.x)
    {
        const double freq = pow(theta, -2.0 * d / rope_dim);
        const double ang = pos * freq;
        float c, s;
        sincosf(static_cast<float>(ang), &s, &c);
        const float x1 = v[d], x2 = v[d + half];
        v[d] = x1 * c - x2 * s;
        v[d + half] = x2 * c + x1 * s;
    }
}

__global__ void mimo_attn_prefill_kernel(const __half *__restrict__ qkv,
                                         const __half *__restrict__ sink,
                                         int seq, int heads, int kv_heads,
                                         int head_dim, int v_head_dim,
                                         int rope_dim, int window, bool is_swa,
                                         float value_scale, float scale,
                                         double theta, int q_size, int k_size,
                                         __half *__restrict__ out)
{
    const int i = blockIdx.x;   // query position
    const int h = blockIdx.y;   // query head
    if (i >= seq || h >= heads) return;

    const int rows = q_size + 2 * k_size;
    const int group = heads / kv_heads;
    const int kh = h / group;

    extern __shared__ float sm[];
    float *q_s = sm;                 // head_dim
    float *k_s = q_s + head_dim;     // head_dim
    float *scratch = k_s + head_dim; // blockDim.x
    float *score = scratch + blockDim.x;   // seq
    float *prob = score + seq;             // seq

    const __half *qrow = qkv + static_cast<int64_t>(i) * rows;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        q_s[d] = __half2float(qrow[h * head_dim + d]);
    __syncthreads();
    RopeHalf(q_s, i, rope_dim, theta);
    __syncthreads();

    const int j0 = is_swa ? max(0, i - window + 1) : 0;
    const int nk = i - j0 + 1;
    for (int j = j0; j <= i; ++j)
    {
        const __half *krow = qkv + static_cast<int64_t>(j) * rows;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
            k_s[d] = __half2float(krow[q_size + kh * head_dim + d]);
        __syncthreads();
        RopeHalf(k_s, j, rope_dim, theta);
        __syncthreads();
        float acc = 0.0f;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
            acc = __fadd_rn(acc, __fmul_rn(q_s[d], k_s[d]));
        scratch[threadIdx.x] = acc;
        __syncthreads();
        for (int s = blockDim.x >> 1; s > 0; s >>= 1)
        {
            if (threadIdx.x < s)
                scratch[threadIdx.x] =
                    __fadd_rn(scratch[threadIdx.x], scratch[threadIdx.x + s]);
            __syncthreads();
        }
        if (threadIdx.x == 0) score[j - j0] = __fmul_rn(scratch[0], scale);
        __syncthreads();
    }

    const bool has_sink = (sink != nullptr);
    float sink_v = 0.0f;
    if (has_sink && threadIdx.x == 0) sink_v = __half2float(sink[h]);
    if (threadIdx.x == 0)
    {
        float mx = has_sink ? sink_v : -3.4e38f;
        for (int t = 0; t < nk; ++t) mx = fmaxf(mx, score[t]);
        float sum = 0.0f;
        for (int t = 0; t < nk; ++t)
        {
            const float e = __expf(score[t] - mx);
            prob[t] = e;
            sum += e;
        }
        if (has_sink) sum += __expf(sink_v - mx);
        const float inv = 1.0f / sum;
        for (int t = 0; t < nk; ++t) prob[t] *= inv;
    }
    __syncthreads();

    float *o = reinterpret_cast<float *>(scratch);  // reuse scratch per-thread
    for (int d = threadIdx.x; d < v_head_dim; d += blockDim.x)
    {
        float acc = 0.0f;
        for (int t = 0; t < nk; ++t)
        {
            const __half *vrow = qkv + static_cast<int64_t>(j0 + t) * rows;
            const float vv = __half2float(vrow[q_size + k_size +
                                               kh * v_head_dim + d]);
            acc = __fadd_rn(acc, __fmul_rn(prob[t], vv));
        }
        o[d] = __fmul_rn(acc, value_scale);
    }
    __syncthreads();
    __half *orow = out + (static_cast<int64_t>(i) * heads + h) * v_head_dim;
    for (int d = threadIdx.x; d < v_head_dim; d += blockDim.x)
        orow[d] = __float2half(o[d]);
}

} // namespace

cudaError_t RunMimoAttentionPrefill(cudaStream_t stream, const __half *qkv,
                                    const __half *sink, int seq,
                                    const MimoAttnShape &p, __half *out)
{
    if (seq <= 0) return cudaSuccess;
    const int threads = 256;
    const int rows = p.q_size + 2 * p.k_size;
    const size_t smem = (2 * p.head_dim + threads + 2 * seq) * sizeof(float);
    dim3 grid(seq, p.heads);
    mimo_attn_prefill_kernel<<<grid, threads, smem, stream>>>(
        qkv, sink, seq, p.heads, p.kv_heads, p.head_dim, p.v_head_dim,
        p.rope_dim, p.window, p.is_swa, p.value_scale, p.scale, p.rope_theta,
        p.q_size, p.k_size, out);
    (void)rows;
    return cudaGetLastError();
}

// One block per query head; threads cooperate on head_dim. The new k/v are
// appended to the cache first, then the query attends over the cached window.
__global__ void mimo_attn_decode_kernel(const __half *__restrict__ qkv_token,
                                        int pos,
                                        const __half *__restrict__ sink,
                                        int heads, int kv_heads, int head_dim,
                                        int v_head_dim, int rope_dim,
                                        int window, bool is_swa,
                                        float value_scale, float scale,
                                        double theta, int q_size, int k_size,
                                        __half *__restrict__ k_cache,
                                        __half *__restrict__ v_cache,
                                        __half *__restrict__ out)
{
    const int h = blockIdx.x;
    if (h >= heads) return;
    const int group = heads / kv_heads;
    const int kh = h / group;

    extern __shared__ float sm[];
    float *q_s = sm;                 // head_dim
    float *k_s = q_s + head_dim;     // head_dim
    float *v_s = k_s + head_dim;     // v_head_dim
    float *scratch = v_s + v_head_dim;  // blockDim.x

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        q_s[d] = __half2float(qkv_token[h * head_dim + d]);
    __syncthreads();
    RopeHalf(q_s, pos, rope_dim, theta);
    __syncthreads();

    // Append roped k and value-scaled v into the cache at `pos`.
    __half *kc = k_cache + (static_cast<int64_t>(pos) * kv_heads + kh) * head_dim;
    __half *vc = v_cache + (static_cast<int64_t>(pos) * kv_heads + kh) * v_head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        k_s[d] = __half2float(qkv_token[q_size + kh * head_dim + d]);
    __syncthreads();
    RopeHalf(k_s, pos, rope_dim, theta);
    __syncthreads();
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        kc[d] = __float2half(k_s[d]);
    for (int d = threadIdx.x; d < v_head_dim; d += blockDim.x)
        vc[d] = __float2half(__fmul_rn(
            __half2float(qkv_token[q_size + k_size + kh * v_head_dim + d]),
            value_scale));
    __syncthreads();

    const int j0 = is_swa ? max(0, pos - window + 1) : 0;
    const int nk = pos - j0 + 1;
    __shared__ float s_m, s_l, s_score;
    if (threadIdx.x == 0) { s_m = -INFINITY; s_l = 0.0f; }
    __syncthreads();
    // Online softmax over cached keys (shared m/l broadcast to the block).
    float *acc = v_s;
    for (int d = threadIdx.x; d < v_head_dim; d += blockDim.x) acc[d] = 0.0f;
    __syncthreads();
    for (int j = j0; j <= pos; ++j)
    {
        const __half *kj =
            k_cache + (static_cast<int64_t>(j) * kv_heads + kh) * head_dim;
        float dot = 0.0f;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
            dot = __fadd_rn(dot, __fmul_rn(q_s[d], __half2float(kj[d])));
        scratch[threadIdx.x] = dot;
        __syncthreads();
        for (int s = blockDim.x >> 1; s > 0; s >>= 1)
        {
            if (threadIdx.x < s)
                scratch[threadIdx.x] =
                    __fadd_rn(scratch[threadIdx.x], scratch[threadIdx.x + s]);
            __syncthreads();
        }
        if (threadIdx.x == 0) s_score = __fmul_rn(scratch[0], scale);
        __syncthreads();
        const float new_m = fmaxf(s_m, s_score);
        const float corr = isinf(s_m) ? 0.0f : __expf(s_m - new_m);
        const float pj = __expf(s_score - new_m);
        const __half *vj =
            v_cache + (static_cast<int64_t>(j) * kv_heads + kh) * v_head_dim;
        for (int d = threadIdx.x; d < v_head_dim; d += blockDim.x)
            acc[d] = __fadd_rn(__fmul_rn(acc[d], corr),
                               __fmul_rn(pj, __half2float(vj[d])));
        __syncthreads();
        if (threadIdx.x == 0) { s_l = __fadd_rn(__fmul_rn(s_l, corr), pj); s_m = new_m; }
        __syncthreads();
    }
    if (sink != nullptr)
    {
        if (threadIdx.x == 0)
        {
            const float sv = __half2float(sink[h]);
            const float new_m = fmaxf(s_m, sv);
            const float corr = isinf(s_m) ? 0.0f : __expf(s_m - new_m);
            s_l = __fadd_rn(__fmul_rn(s_l, corr), __expf(sv - new_m));
            s_m = new_m;
            s_score = corr;
        }
        __syncthreads();
        const float corr = s_score;
        for (int d = threadIdx.x; d < v_head_dim; d += blockDim.x)
            acc[d] = __fmul_rn(acc[d], corr);
        __syncthreads();
    }
    const float inv = 1.0f / s_l;
    __half *orow = out + h * v_head_dim;
    for (int d = threadIdx.x; d < v_head_dim; d += blockDim.x)
        orow[d] = __float2half(__fmul_rn(acc[d], inv));
}

cudaError_t RunMimoAttentionDecode(cudaStream_t stream, const __half *qkv_token,
                                   int pos, const __half *sink,
                                   const MimoAttnShape &p, __half *k_cache,
                                   __half *v_cache, __half *out)
{
    if (pos < 0) return cudaErrorInvalidValue;
    const int threads = 256;
    const int rows = p.q_size + 2 * p.k_size;
    (void)rows;
    const size_t smem =
        (2 * p.head_dim + p.v_head_dim + threads) * sizeof(float);
    mimo_attn_decode_kernel<<<p.heads, threads, smem, stream>>>(
        qkv_token, pos, sink, p.heads, p.kv_heads, p.head_dim, p.v_head_dim,
        p.rope_dim, p.window, p.is_swa, p.value_scale, p.scale, p.rope_theta,
        p.q_size, p.k_size, k_cache, v_cache, out);
    return cudaGetLastError();
}

} // namespace mimo
} // namespace lsxcommon
