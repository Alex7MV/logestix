// Nemotron-3.5 Mamba-2 mixer kernels (Model Ingestion Protocol step 5).
// Shared-memory sandbox: reductions use a shared array + thread-0 sum behind
// barriers; no atomics feed logits. Math is pinned by
// nemotron_moe_device_math.h (host reference).
#include "nemotron_moe_kernels.h"

#include "nemotron_moe_constants.h"

#include <cuda_runtime.h>

namespace lsxcommon::nemotron_moe {

namespace {

__device__ __forceinline__ float silu_f(float x)
{
    return x / (1.0f + expf(-x));
}

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
    out[c] = silu_f(acc);
}

__global__ void ssm_step_kernel(const float *__restrict__ state_in,
                                const float *__restrict__ xs,
                                const float *__restrict__ B,
                                const float *__restrict__ C,
                                const float *__restrict__ D,
                                const float *__restrict__ dt,
                                const float *__restrict__ dt_bias,
                                const float *__restrict__ A_log, int heads,
                                int head_dim, int groups, int state_size,
                                int clamp_dt, float *__restrict__ state_out,
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
    if (clamp_dt && dth < frozen::kDtMin) dth = frozen::kDtMin;
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

__global__ void gated_rmsnorm_kernel(const float *__restrict__ x,
                                     const float *__restrict__ gate,
                                     const float *__restrict__ weight, int n,
                                     int group_size, float eps,
                                     float *__restrict__ out)
{
    extern __shared__ float sq[];  // group_size floats
    const int g = blockIdx.x;
    const int i = threadIdx.x;
    const int base = g * group_size;
    float h = 0.0f;
    if (i < group_size) {
        h = x[base + i] * silu_f(gate[base + i]);
        sq[i] = h * h;
    }
    __syncthreads();
    if (i == 0) {
        float s = 0.0f;
        for (int k = 0; k < group_size; ++k) s += sq[k];
        sq[0] = s;
    }
    __syncthreads();
    if (i < group_size) {
        const float inv = rsqrtf(sq[0] / static_cast<float>(group_size) + eps);
        out[base + i] = weight[base + i] * h * inv;
    }
}

__global__ void router_kernel(const float *__restrict__ logits,
                              const float *__restrict__ bias, int E, int k,
                              float scale, float *__restrict__ weights,
                              int *__restrict__ indices)
{
    // Parallel top-k: each iteration block-reduces the max unselected choice
    // (lowest index wins ties, matching ref::RouterTopK). E <= 128 (frozen
    // passport). One block; the reduction is O(k log2 E) with all threads
    // active instead of a serial O(k*E^2) scan on thread 0.
    __shared__ float s[128];
    __shared__ float ch[128];
    __shared__ int sel[128];
    __shared__ float rv[128];
    __shared__ int ri[128];
    const int e = threadIdx.x;
    if (e < E) {
        const float sg = 1.0f / (1.0f + expf(-logits[e]));
        s[e] = sg;
        ch[e] = sg + bias[e];
        sel[e] = 0;
    }
    __syncthreads();
    for (int t = 0; t < k; ++t) {
        rv[e] = (e < E && !sel[e]) ? ch[e] : -INFINITY;
        ri[e] = e;
        __syncthreads();
        for (int sft = blockDim.x / 2; sft > 0; sft >>= 1) {
            if (e < sft) {
                if (rv[e + sft] > rv[e] ||
                    (rv[e + sft] == rv[e] && ri[e + sft] < ri[e])) {
                    rv[e] = rv[e + sft];
                    ri[e] = ri[e + sft];
                }
            }
            __syncthreads();
        }
        if (e == 0) {
            indices[t] = ri[0];
            sel[ri[0]] = 1;
        }
        __syncthreads();
    }
    if (e == 0) {
        float sum = 0.0f;
        for (int t = 0; t < k; ++t) {
            weights[t] = s[indices[t]];
            sum += weights[t];
        }
        const float denom = sum + 1e-20f;
        for (int t = 0; t < k; ++t) weights[t] = weights[t] / denom * scale;
    }
}

__global__ void relu2_kernel(const float *__restrict__ x, int n,
                             float *__restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float r = x[i] > 0.0f ? x[i] : 0.0f;
    out[i] = r * r;
}

__global__ void relu2_half_kernel(__half *__restrict__ x, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = __half2float(x[i]);
    const float r = v > 0.0f ? v : 0.0f;
    x[i] = __float2half(r * r);
}

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

__global__ void add_half_to_f32_kernel(const __half *__restrict__ x, int n,
                                       float *__restrict__ acc)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    acc[j] += __half2float(x[j]);
}

} // namespace

void RunNemotronRouter(const float *logits, const float *e_score_bias, int E,
                       int k, float scale, float *weights, int *indices,
                       cudaStream_t stream)
{
    int threads = 1;
    while (threads < E) threads <<= 1;  // power-of-two reduction width
    router_kernel<<<1, threads, 0, stream>>>(
        logits, e_score_bias, E, k, scale, weights, indices);
}

void RunNemotronRelu2(const float *x, int n, float *out, cudaStream_t stream)
{
    const int threads = 256;
    relu2_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(x, n,
                                                                     out);
}

void RunNemotronRelu2Half(__half *x, int n, cudaStream_t stream)
{
    const int threads = 256;
    relu2_half_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(x,
                                                                           n);
}

void RunNemotronMoeWeightedSum(const __half *y, const float *weights, int count,
                               int n, float *acc, bool add,
                               cudaStream_t stream)
{
    const int threads = 256;
    moe_weighted_sum_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(
        y, weights, count, n, acc, add ? 1 : 0);
}

void RunNemotronAddHalfToF32(const __half *x, int n, float *acc,
                             cudaStream_t stream)
{
    const int threads = 256;
    add_half_to_f32_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(
        x, n, acc);
}

void RunNemotronConvStep(const float *xBC, const float *w, const float *bias,
                         const float *state_in, float *state_out, float *out,
                         int conv_dim, int K, cudaStream_t stream)
{
    const int threads = 256;
    const int blocks = (conv_dim + threads - 1) / threads;
    conv_step_kernel<<<blocks, threads, 0, stream>>>(
        xBC, w, bias, state_in, state_out, out, conv_dim, K);
}

void RunNemotronSelectiveStateStep(const float *state_in, const float *xs,
                                   const float *B, const float *C,
                                   const float *D, const float *dt,
                                   const float *dt_bias, const float *A_log,
                                   int heads, int head_dim, int groups,
                                   int state_size, int clamp_dt, float *state_out,
                                   float *out, cudaStream_t stream)
{
    ssm_step_kernel<<<heads, head_dim, 0, stream>>>(
        state_in, xs, B, C, D, dt, dt_bias, A_log, heads, head_dim, groups,
        state_size, clamp_dt, state_out, out);
}

void RunNemotronGatedRmsNorm(const float *x, const float *gate,
                             const float *weight, int n, int group_size,
                             float eps, float *out, cudaStream_t stream)
{
    const int groups = n / group_size;
    gated_rmsnorm_kernel<<<groups, group_size, group_size * sizeof(float),
                           stream>>>(x, gate, weight, n, group_size, eps, out);
}

} // namespace lsxcommon::nemotron_moe
