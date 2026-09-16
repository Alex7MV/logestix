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
                                float *__restrict__ state_out,
                                float *__restrict__ out)
{
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    if (h >= heads || d >= head_dim) return;

    float dth = dt[h] + dt_bias[h];
    // softplus, numerically stable
    dth = dth > 20.0f ? dth : log1pf(expf(dth));
    if (dth < frozen::kDtMin) dth = frozen::kDtMin;
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
    extern __shared__ float sh[];  // [E] sigmoid scores, [E] choice + bias
    float *s = sh;
    float *ch = sh + E;
    const int e = threadIdx.x;
    if (e < E) {
        const float sg = 1.0f / (1.0f + expf(-logits[e]));
        s[e] = sg;
        ch[e] = sg + bias[e];
    }
    __syncthreads();
    if (e == 0) {
        for (int t = 0; t < k; ++t) {
            int best = -1;
            for (int j = 0; j < E; ++j) {
                bool taken = false;
                for (int p = 0; p < t; ++p)
                    if (indices[p] == j) taken = true;
                if (taken) continue;
                if (best < 0 || ch[j] > ch[best]) best = j;
            }
            indices[t] = best;
        }
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

} // namespace

void RunNemotronRouter(const float *logits, const float *e_score_bias, int E,
                       int k, float scale, float *weights, int *indices,
                       cudaStream_t stream)
{
    router_kernel<<<1, E, 2 * E * sizeof(float), stream>>>(
        logits, e_score_bias, E, k, scale, weights, indices);
}

void RunNemotronRelu2(const float *x, int n, float *out, cudaStream_t stream)
{
    const int threads = 256;
    relu2_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(x, n,
                                                                     out);
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
                                   int state_size, float *state_out, float *out,
                                   cudaStream_t stream)
{
    ssm_step_kernel<<<heads, head_dim, 0, stream>>>(
        state_in, xs, B, C, D, dt, dt_bias, A_log, heads, head_dim, groups,
        state_size, state_out, out);
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
