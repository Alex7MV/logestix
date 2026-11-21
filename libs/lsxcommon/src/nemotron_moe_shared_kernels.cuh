// nemotron_moe_shared_kernels.cuh — kernels shared by Nemotron-3.5-Lightning
// and Nemotron-3-Ultra (dedup phase 7). The five bodies were byte-identical
// between the two family TUs (proven by normalized diff); they move here
// verbatim. Both family TUs are -fmad=false, and this header is included by
// both, so the arithmetic stays under the same flag.
//
// ssm_step references frozen::kDtMin — each family's passport defines it
// (both 0.001f); the include order in the family .cu keeps the family's own
// constants header in scope, so the kernel reads the family's value.
//
// Templated on a dummy Tag (default void) for weak linkage: a plain
// __global__ in a header multiply defines its host stub when two TUs link
// into one binary (the RmsNormTreeKernel pattern). Call sites use <>.
//
// NOT shared: gated_rmsnorm_kernel / router_kernel (the bodies differ —
// ultra strides the block past 1024 threads and sizes shared arrays to its
// own kNumRouted=512 vs lightning's 128).
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace lsxcommon {
namespace nemotron_shared {

namespace detail {
__device__ __forceinline__ float silu_f(float x)
{
    return x / (1.0f + expf(-x));
}
} // namespace detail

template <typename Tag = void>
__global__ void conv_step_kernel(const float *__restrict__ xBC,
                                 const float *__restrict__ w,
                                 const float *__restrict__ bias,
                                 const float *__restrict__ state_in,
                                 float *__restrict__ state_out,
                                 float *__restrict__ out, int conv_dim, int K)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    const int S = K - 1;
    float acc = bias ? bias[c] : 0.0f;
    for (int k = 0; k < S; ++k)
        acc += w[c * K + k] * state_in[c * S + k];
    acc += w[c * K + (K - 1)] * xBC[c];
    for (int k = 0; k < S - 1; ++k)
        state_out[c * S + k] = state_in[c * S + k + 1];
    state_out[c * S + (S - 1)] = xBC[c];
    out[c] = detail::silu_f(acc);
}

template <typename Tag = void>
__global__ void ssm_step_kernel(const float *__restrict__ state_in,
                                const float *__restrict__ xs,
                                const float *__restrict__ B,
                                const float *__restrict__ C,
                                const float *__restrict__ D,
                                const float *__restrict__ dt,
                                const float *__restrict__ dt_bias,
                                const float *__restrict__ A_log, int heads,
                                int head_dim, int groups, int state_size,
                                int clamp_dt, float dt_min,
                                float *__restrict__ state_out,
                                float *__restrict__ out)
{
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    if (h >= heads || d >= head_dim) return;

    float dth = dt[h] + dt_bias[h];
    // softplus, numerically stable
    dth = dth > 20.0f ? dth : log1pf(expf(dth));
    // Reference split: the chunk scan (prefill) clamps dt, the single-token
    // selective update (decode) does not.
    if (clamp_dt && dth < dt_min) dth = dt_min;
    const float A = -expf(A_log[h]);
    const float dA = expf(dth * A);
    const int hpg = heads / groups;
    const int g = h / hpg;
    const float xsd = xs[h * head_dim + d];
    const float *Bg = B + g * state_size;
    const float *Cg = C + g * state_size;
    const size_t base =
        (static_cast<size_t>(h) * head_dim + d) * state_size;
    float acc = 0.0f;
    for (int s = 0; s < state_size; ++s) {
        const float st = state_in[base + s] * dA + dth * Bg[s] * xsd;
        state_out[base + s] = st;
        acc += st * Cg[s];
    }
    out[h * head_dim + d] = acc + xsd * D[h];
}

template <typename Tag = void>
__global__ void relu2_half_kernel(__half *__restrict__ x, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = __half2float(x[i]);
    const float r = v > 0.0f ? v : 0.0f;
    x[i] = __float2half(r * r);
}

template <typename Tag = void>
__global__ void moe_weighted_sum_kernel(const __half *__restrict__ y,
                                        const float *__restrict__ w, int count,
                                        int n, float *__restrict__ acc,
                                        int add)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    float s = add ? acc[j] : 0.0f;
    for (int e = 0; e < count; ++e)
        s += w[e] * __half2float(y[static_cast<size_t>(e) * n + j]);
    acc[j] = s;
}

template <typename Tag = void>
__global__ void add_half_to_f32_kernel(const __half *__restrict__ x, int n,
                                       float *__restrict__ acc)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    acc[j] += __half2float(x[j]);
}


// ── Pipeline kernels (byte-identical in both family pipelines) ──────────

template <typename Tag = void>
__global__ void embed_gather_kernel(const __half *__restrict__ e, int H,
                                    int token, __half *__restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H) return;
    out[i] = e[static_cast<size_t>(token) * H + i];
}

template <typename Tag = void>
__global__ void rmsnorm_half_kernel(const __half *__restrict__ x,
                                    const __half *__restrict__ w, int n,
                                    float eps, __half *__restrict__ out)
{
    __shared__ float red[256];
    const int t = threadIdx.x;
    float s = 0.0f;
    for (int i = t; i < n; i += blockDim.x) {
        const float v = __half2float(x[i]);
        s += v * v;
    }
    red[t] = s;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (t < st) red[t] += red[t + st];
        __syncthreads();
    }
    const float inv = rsqrtf(red[0] / static_cast<float>(n) + eps);
    for (int i = t; i < n; i += blockDim.x)
        out[i] = __float2half(__half2float(x[i]) * inv *
                              __half2float(w[i]));
}

template <typename Tag = void>
__global__ void residual_add_kernel(__half *__restrict__ h,
                                    const __half *__restrict__ sub, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    h[i] = __float2half(__half2float(h[i]) + __half2float(sub[i]));
}

template <typename Tag = void>
__global__ void kv_store_kernel(float *__restrict__ cache,
                                const float *__restrict__ src, int seq_pos,
                                int max_seq, int head_dim)
{
    // cache [kv_heads, max_seq, head_dim]; copy one row per head.
    const int row = blockIdx.x;  // kv head index
    const int d = threadIdx.x;
    if (d >= head_dim) return;
    cache[(static_cast<size_t>(row) * max_seq + seq_pos) * head_dim + d] =
        src[static_cast<size_t>(row) * head_dim + d];
}

} // namespace nemotron_shared
} // namespace lsxcommon