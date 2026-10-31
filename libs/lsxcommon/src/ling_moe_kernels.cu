// Ling-3.1-flash kernels: KDA, gated MLA, group-limited router, epilogues.
#include "lsxcommon/ling_moe_kernels.h"

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>

namespace lsxcommon::ling {
namespace {

__device__ __forceinline__ float sigmoid_f(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float silu_f(float x)
{
    return x * sigmoid_f(x);
}

__device__ __forceinline__ float bf16_bits_to_float(uint16_t b)
{
    const uint32_t u = static_cast<uint32_t>(b) << 16;
    float v;
    memcpy(&v, &u, sizeof(v));
    return v;
}

__global__ void rmsnorm_kernel(const __half *__restrict__ x,
                               const __half *__restrict__ w,
                               __half *__restrict__ out, int hidden,
                               float eps)
{
    const int r = blockIdx.x;
    const __half *row = x + static_cast<int64_t>(r) * hidden;
    __shared__ float red[256];
    float acc = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        const float v = __half2float(row[i]);
        acc += v * v;
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = rsqrtf(red[0] / static_cast<float>(hidden) + eps);
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        const float v = __half2float(row[i]) * inv * __half2float(w[i]);
        out[static_cast<int64_t>(r) * hidden + i] = __float2half(v);
    }
}

__global__ void conv_decode_kernel(__half *__restrict__ hist,
                                   const __half *__restrict__ cur,
                                   const __half *__restrict__ weight,
                                   __half *__restrict__ out, int channels,
                                   int kernel, bool apply_silu)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const int taps = kernel - 1;
    float acc = 0.0f;
    for (int i = 0; i < taps; ++i)
        acc += __half2float(hist[static_cast<int64_t>(i) * channels + c]) *
               __half2float(weight[static_cast<int64_t>(c) * kernel + i]);
    acc += __half2float(cur[c]) *
           __half2float(weight[static_cast<int64_t>(c) * kernel + taps]);
    out[c] = __float2half(apply_silu ? silu_f(acc) : acc);
    for (int i = 0; i + 1 < taps; ++i)
        hist[static_cast<int64_t>(i) * channels + c] =
            hist[static_cast<int64_t>(i + 1) * channels + c];
    hist[static_cast<int64_t>(taps - 1) * channels + c] = cur[c];
}

__global__ void l2norm_qk_kernel(__half *__restrict__ q, __half *__restrict__ k,
                                 int dim, float q_scale, int heads)
{
    const int blk = blockIdx.x;
    __half *row = blk < heads
                      ? q + static_cast<int64_t>(blk) * dim
                      : k + static_cast<int64_t>(blk - heads) * dim;
    __shared__ float red[256];
    float acc = 0.0f;
    if (threadIdx.x < dim) {
        const float v = __half2float(row[threadIdx.x]);
        acc = v * v;
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x < dim) {
        const float norm = sqrtf(red[0] + 1e-6f);
        const float sc = blk < heads ? q_scale : 1.0f;
        row[threadIdx.x] =
            __float2half(__half2float(row[threadIdx.x]) / norm * sc);
    }
}

__global__ void kda_gating_kernel(const __half *__restrict__ b,
                                  const __half *__restrict__ f,
                                  const float *__restrict__ a_log,
                                  const float *__restrict__ dt_bias,
                                  float *__restrict__ beta,
                                  float *__restrict__ decay, int heads,
                                  int dim, float lower_bound)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= heads * dim) return;
    const int h = idx / dim;
    if (idx % dim == 0) beta[h] = sigmoid_f(__half2float(b[h]));
    const float dt = __half2float(f[idx]) + dt_bias[idx];
    decay[idx] = expf(lower_bound * sigmoid_f(expf(a_log[h]) * dt));
}

__global__ void kda_step_kernel(const __half *__restrict__ q,
                                const __half *__restrict__ k,
                                const __half *__restrict__ v,
                                const float *__restrict__ beta,
                                const float *__restrict__ decay,
                                float *__restrict__ state,
                                __half *__restrict__ out, int heads, int dim)
{
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    if (d >= dim) return;
    extern __shared__ float sh[];
    float *col = sh;  // [dim][dim] for this head
    const float *src = state + static_cast<int64_t>(h) * dim * dim;
    for (int l = threadIdx.x; l < dim * dim; l += blockDim.x)
        col[l] = src[l];
    __syncthreads();
    for (int l = 0; l < dim; ++l)
        col[static_cast<int64_t>(l) * dim + d] *= decay[h * dim + l];
    __syncthreads();
    float kv = 0.0f;
    for (int l = 0; l < dim; ++l)
        kv += col[static_cast<int64_t>(l) * dim + d] *
              __half2float(k[h * dim + l]);
    const float delta = beta[h] * (__half2float(v[h * dim + d]) - kv);
    for (int l = 0; l < dim; ++l)
        col[static_cast<int64_t>(l) * dim + d] +=
            __half2float(k[h * dim + l]) * delta;
    __syncthreads();
    float acc = 0.0f;
    for (int l = 0; l < dim; ++l)
        acc += col[static_cast<int64_t>(l) * dim + d] *
               __half2float(q[h * dim + l]);
    out[h * dim + d] = __float2half(acc);
    float *dst = state + static_cast<int64_t>(h) * dim * dim;
    for (int l = threadIdx.x; l < dim * dim; l += blockDim.x)
        dst[l] = col[l];
}

__global__ void rmsnorm_gated_kernel(const __half *__restrict__ core,
                                     const __half *__restrict__ gate,
                                     const __half *__restrict__ w,
                                     __half *__restrict__ out, int dim,
                                     float eps)
{
    const int h = blockIdx.x;
    __shared__ float red[256];
    const __half *row = core + static_cast<int64_t>(h) * dim;
    float acc = 0.0f;
    if (threadIdx.x < dim) {
        const float v = __half2float(row[threadIdx.x]);
        acc = v * v;
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x < dim) {
        const int64_t at = static_cast<int64_t>(h) * dim + threadIdx.x;
        const float v = __half2float(row[threadIdx.x]) *
                        rsqrtf(red[0] / static_cast<float>(dim) + eps);
        const float g = sigmoid_f(__half2float(gate[at]));
        out[at] = __float2half(v * __half2float(w[threadIdx.x]) * g);
    }
}

__global__ void rope_q_kernel(__half *__restrict__ q, int head_dim,
                              int rope_dim, int pos, double theta)
{
    const int h = blockIdx.x;
    const int half = rope_dim / 2;
    const int i = threadIdx.x;
    if (i >= half) return;
    const double freq = pow(theta, -2.0 * static_cast<double>(i) /
                                         static_cast<double>(rope_dim));
    const double ang = static_cast<double>(pos) * freq;
    const float c = static_cast<float>(cos(ang));
    const float s = static_cast<float>(sin(ang));
    const int base = h * head_dim + head_dim - rope_dim + 2 * i;
    const float x0 = __half2float(q[base]);
    const float x1 = __half2float(q[base + 1]);
    q[base] = __float2half(x0 * c - x1 * s);
    q[base + 1] = __float2half(x1 * c + x0 * s);
}

__global__ void rope_k_kernel(__half *__restrict__ k, int rope_dim, int pos,
                              double theta)
{
    const int half = rope_dim / 2;
    const int i = threadIdx.x;
    if (i >= half) return;
    const double freq = pow(theta, -2.0 * static_cast<double>(i) /
                                         static_cast<double>(rope_dim));
    const double ang = static_cast<double>(pos) * freq;
    const float c = static_cast<float>(cos(ang));
    const float s = static_cast<float>(sin(ang));
    const int base = 2 * i;
    const float x0 = __half2float(k[base]);
    const float x1 = __half2float(k[base + 1]);
    k[base] = __float2half(x0 * c - x1 * s);
    k[base + 1] = __float2half(x1 * c + x0 * s);
}

__global__ void router_topk_kernel(const uint8_t *__restrict__ gate_bf16,
                                   const float *__restrict__ bias,
                                   const __half *__restrict__ x, int experts,
                                   int top_k, int n_group, int topk_group,
                                   int hidden, float routed_scale,
                                   moe::MoERoutedSlot *__restrict__ slots)
{
    extern __shared__ float sh[];
    float *s_scores = sh;              // [experts] raw sigmoid
    float *s_choice = sh + experts;    // [experts] sigmoid + bias
    float *s_gscore = s_choice + experts;  // [n_group]
    for (int e = threadIdx.x; e < experts; e += blockDim.x) {
        float dot = 0.0f;
        for (int h = 0; h < hidden; ++h) {
            const uint16_t gb =
                static_cast<uint16_t>(gate_bf16[static_cast<int64_t>(e) *
                                                    hidden * 2 +
                                                h * 2]) |
                (static_cast<uint16_t>(gate_bf16[static_cast<int64_t>(e) *
                                                     hidden * 2 +
                                                 h * 2 + 1])
                 << 8);
            dot += bf16_bits_to_float(gb) * __half2float(x[h]);
        }
        const float s = sigmoid_f(dot);
        s_scores[e] = s;
        s_choice[e] = s + bias[e];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const int gsize = experts / n_group;
        for (int g = 0; g < n_group; ++g) {
            float b0 = -1e30f, b1 = -1e30f;
            for (int j = 0; j < gsize; ++j) {
                const float v = s_choice[g * gsize + j];
                if (v > b0) {
                    b1 = b0;
                    b0 = v;
                } else if (v > b1) {
                    b1 = v;
                }
            }
            s_gscore[g] = b0 + b1;
        }
        bool gsel[32];
        for (int g = 0; g < n_group; ++g) gsel[g] = false;
        for (int k = 0; k < topk_group; ++k) {
            int best = -1;
            float bestv = -1e30f;
            for (int g = 0; g < n_group; ++g) {
                if (gsel[g]) continue;
                if (s_gscore[g] > bestv) {
                    bestv = s_gscore[g];
                    best = g;
                }
            }
            if (best >= 0) gsel[best] = true;
        }
        float chosen_w[16];
        int chosen_e[16];
        for (int k = 0; k < top_k; ++k) {
            int best = -1;
            float bestv = -1e30f;
            for (int e = 0; e < experts; ++e) {
                if (!gsel[e / gsize]) continue;
                bool taken = false;
                for (int j = 0; j < k; ++j)
                    if (chosen_e[j] == e) taken = true;
                if (taken) continue;
                if (s_choice[e] > bestv) {
                    bestv = s_choice[e];
                    best = e;
                }
            }
            chosen_e[k] = best;
            chosen_w[k] = best >= 0 ? s_scores[best] : 0.0f;
        }
        float sum = 0.0f;
        for (int k = 0; k < top_k; ++k) sum += chosen_w[k];
        for (int k = 0; k < top_k; ++k) {
            slots[k].eid = chosen_e[k];
            slots[k].coeff = sum > 0.0f ? chosen_w[k] / sum * routed_scale
                                       : 0.0f;
        }
    }
}

__global__ void swiglu_kernel(const __half *__restrict__ gate,
                              const __half *__restrict__ up,
                              __half *__restrict__ out, int n, float limit)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = __half2float(gate[i]);
    const float u = __half2float(up[i]);
    float v = silu_f(g);
    if (limit > 0.0f) {
        v = fminf(v, limit);
        v *= fminf(fmaxf(u, -limit), limit);
    } else {
        v *= u;
    }
    out[i] = __float2half(v);
}

__global__ void mla_attn_kernel(const __half *__restrict__ q,
                                const __half *__restrict__ kv,
                                float *__restrict__ scores,
                                const KvCacheGeometry geom, int layer_slot,
                                int64_t count, int head_dim, int v_head,
                                float scale, __half *__restrict__ out,
                                int capacity)
{
    const int h = blockIdx.x;
    const int64_t base = geom.Offset(layer_slot, h, 0, 0);
    const __half *qh = q + static_cast<int64_t>(h) * head_dim;
    float *sc = scores + static_cast<int64_t>(h) * capacity;
    __shared__ float red[256];
    float local_max = -1e30f;
    for (int64_t p = threadIdx.x; p < count; p += blockDim.x) {
        const __half *krow = kv + base + static_cast<int64_t>(p) * geom.pos_stride;
        float dot = 0.0f;
        for (int i = 0; i < head_dim; ++i)
            dot += __half2float(qh[i]) * __half2float(krow[i]);
        const float s = dot * scale;
        sc[p] = s;
        local_max = fmaxf(local_max, s);
    }
    red[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    const float m = red[0];
    float local_sum = 0.0f;
    for (int64_t p = threadIdx.x; p < count; p += blockDim.x) {
        const float e = expf(sc[p] - m);
        sc[p] = e;
        local_sum += e;
    }
    red[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = 1.0f / red[0];
    for (int64_t p = threadIdx.x; p < count; p += blockDim.x) sc[p] *= inv;
    __syncthreads();
    if (threadIdx.x >= v_head) return;
    const int d = threadIdx.x;
    float acc = 0.0f;
    for (int64_t p = 0; p < count; ++p) {
        const __half *vrow =
            kv + base + static_cast<int64_t>(p) * geom.pos_stride + head_dim;
        acc += sc[p] * __half2float(vrow[d]);
    }
    out[static_cast<int64_t>(h) * v_head + d] = __float2half(acc);
}

__global__ void mla_out_gate_kernel(__half *__restrict__ out,
                                    const __half *__restrict__ gate, int v_head)
{
    const int h = blockIdx.x;
    const float g = sigmoid_f(__half2float(gate[h]));
    for (int d = threadIdx.x; d < v_head; d += blockDim.x)
        out[static_cast<int64_t>(h) * v_head + d] =
            __float2half(__half2float(out[static_cast<int64_t>(h) * v_head + d]) * g);
}

__global__ void embed_kernel(const __half *__restrict__ table, int32_t id,
                             __half *__restrict__ out, int hidden)
{
    for (int i = threadIdx.x; i < hidden; i += blockDim.x)
        out[i] = table[static_cast<int64_t>(id) * hidden + i];
}

__global__ void add_routed_shared_kernel(const __half *__restrict__ resid,
                                         const __half *__restrict__ shared,
                                         const float *__restrict__ routed,
                                         __half *__restrict__ out, int hidden)
{
    for (int i = threadIdx.x; i < hidden; i += blockDim.x)
        out[i] = __float2half(__half2float(resid[i]) +
                              __half2float(shared[i]) + routed[i]);
}

} // namespace

cudaError_t RunLingRmsNorm(cudaStream_t stream, const __half *x,
                           const __half *w, __half *out, int rows, int hidden,
                           float eps)
{
    rmsnorm_kernel<<<rows, 256, 0, stream>>>(x, w, out, hidden, eps);
    return cudaGetLastError();
}

cudaError_t RunLingConvDecodeSilu(cudaStream_t stream, __half *hist,
                                  const __half *cur, const __half *weight,
                                  __half *out, int channels, int kernel,
                                  bool apply_silu)
{
    const int blocks = (channels + 255) / 256;
    conv_decode_kernel<<<blocks, 256, 0, stream>>>(
        hist, cur, weight, out, channels, kernel, apply_silu);
    return cudaGetLastError();
}

cudaError_t RunLingL2NormQK(cudaStream_t stream, __half *q, __half *k,
                            int heads, int dim, float q_scale)
{
    if (dim > 256) return cudaErrorInvalidValue;
    l2norm_qk_kernel<<<2 * heads, 256, 0, stream>>>(q, k, dim, q_scale, heads);
    return cudaGetLastError();
}

cudaError_t RunLingKdaGating(cudaStream_t stream, const __half *b,
                             const __half *f, const float *a_log,
                             const float *dt_bias, float *beta, float *decay,
                             int heads, int dim, float lower_bound)
{
    const int n = heads * dim;
    kda_gating_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        b, f, a_log, dt_bias, beta, decay, heads, dim, lower_bound);
    return cudaGetLastError();
}

cudaError_t RunLingKdaStep(cudaStream_t stream, const __half *q,
                           const __half *k, const __half *v,
                           const float *beta, const float *decay,
                           float *state, __half *out, int heads, int dim)
{
    const size_t shmem = static_cast<size_t>(dim) * dim * sizeof(float);
    auto kernel = kda_step_kernel;
    if (shmem > 48 * 1024) {
        const cudaError_t e = cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shmem));
        if (e != cudaSuccess) return e;
    }
    kernel<<<heads, dim, shmem, stream>>>(q, k, v, beta, decay, state, out,
                                          heads, dim);
    return cudaGetLastError();
}

cudaError_t RunLingRmsNormGated(cudaStream_t stream, const __half *core,
                                const __half *gate, const __half *w,
                                __half *out, int heads, int dim, float eps)
{
    if (dim > 256) return cudaErrorInvalidValue;
    rmsnorm_gated_kernel<<<heads, 256, 0, stream>>>(core, gate, w, out, dim,
                                                    eps);
    return cudaGetLastError();
}

cudaError_t RunLingRopeInterleaved(cudaStream_t stream, __half *q,
                                   __half *k_rot, int heads, int head_dim,
                                   int rope_dim, int pos, double theta)
{
    const int half = rope_dim / 2;
    if (half > 256) return cudaErrorInvalidValue;
    rope_q_kernel<<<dim3(heads, 1), 256, 0, stream>>>(q, head_dim, rope_dim,
                                                      pos, theta);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    rope_k_kernel<<<1, 256, 0, stream>>>(k_rot, rope_dim, pos, theta);
    return cudaGetLastError();
}

cudaError_t RunLingRouterTopK(cudaStream_t stream, const uint8_t *gate_bf16,
                              const float *bias, const __half *x, int experts,
                              int top_k, int n_group, int topk_group,
                              int hidden, float routed_scale,
                              moe::MoERoutedSlot *slots)
{
    if (top_k > 16 || n_group > 32) return cudaErrorInvalidValue;
    const size_t shmem =
        static_cast<size_t>(2 * experts + n_group) * sizeof(float);
    router_topk_kernel<<<1, 256, shmem, stream>>>(
        gate_bf16, bias, x, experts, top_k, n_group, topk_group, hidden,
        routed_scale, slots);
    return cudaGetLastError();
}

cudaError_t RunLingSwiGLU(cudaStream_t stream, const __half *gate,
                          const __half *up, __half *out, int n, float limit)
{
    swiglu_kernel<<<(n + 255) / 256, 256, 0, stream>>>(gate, up, out, n, limit);
    return cudaGetLastError();
}

cudaError_t RunLingMlaAttention(cudaStream_t stream, const __half *q,
                                const __half *kv, float *scores,
                                const KvCacheGeometry &geom, int layer_slot,
                                int64_t count, int heads, int head_dim,
                                int v_head, float scale, __half *out)
{
    mla_attn_kernel<<<heads, 256, 0, stream>>>(
        q, kv, scores, geom, layer_slot, count, head_dim, v_head, scale, out,
        static_cast<int>(geom.capacity));
    return cudaGetLastError();
}

cudaError_t RunLingMlaOutGate(cudaStream_t stream, __half *out,
                              const __half *gate, int heads, int v_head)
{
    mla_out_gate_kernel<<<heads, 128, 0, stream>>>(out, gate, v_head);
    return cudaGetLastError();
}

cudaError_t RunLingEmbedToken(cudaStream_t stream, const __half *embed,
                              int32_t token_id, __half *out, int hidden)
{
    embed_kernel<<<1, 256, 0, stream>>>(embed, token_id, out, hidden);
    return cudaGetLastError();
}

cudaError_t RunLingAddRoutedShared(cudaStream_t stream, const __half *resid,
                                   const __half *shared, const float *routed,
                                   __half *out, int hidden)
{
    add_routed_shared_kernel<<<1, 256, 0, stream>>>(resid, shared, routed, out,
                                                    hidden);
    return cudaGetLastError();
}

} // namespace lsxcommon::ling
