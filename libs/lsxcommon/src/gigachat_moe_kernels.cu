// GigaChat3.5 elementwise gating kernels (Plan 2a): swiglu_limit MLP
// activation, gated-attention sigmoid, and the sigmoid top-k router.
#include "gigachat_moe_kernels.h"

#include "gigachat_moe_device_math.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace lsxcommon::gigachat {
namespace {

constexpr int kMaxTopK = 16;

__global__ void swiglu_limit_kernel(const float *__restrict__ g,
                                    const float *__restrict__ u, float limit,
                                    int n, float *__restrict__ mid)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) mid[i] = dev::SwigluLimitElement(g[i], u[i], limit);
}

__global__ void sigmoid_gate_kernel(float *__restrict__ y,
                                    const float *__restrict__ gate, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = dev::SigmoidGateElement(y[i], gate[i]);
}

// One block, deterministic thread-0 sequential top-k (lowest index wins ties).
__global__ void router_topk_kernel(const float *__restrict__ x,
                                   const __nv_bfloat16 *__restrict__ w,
                                   const float *__restrict__ bias, int E, int H,
                                   int top_k, float scale,
                                   int *__restrict__ out_idx,
                                   float *__restrict__ out_w)
{
    extern __shared__ float smem[];  // scores[E] + choice[E]
    float *scores = smem;
    float *choice = smem + E;

    for (int e = threadIdx.x; e < E; e += blockDim.x)
    {
        const __nv_bfloat16 *wr = w + static_cast<size_t>(e) * H;
        float a = 0.0f;
        for (int i = 0; i < H; ++i) a += __bfloat162float(wr[i]) * x[i];
        const float s = dev::Sigmoid(a);
        scores[e] = s;
        choice[e] = s + (bias ? bias[e] : 0.0f);
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        int idx[kMaxTopK];
        for (int k = 0; k < top_k; ++k)
        {
            int best = -1;
            float bv = -3.4e38f;
            for (int e = 0; e < E; ++e)
            {
                bool used = false;
                for (int j = 0; j < k; ++j)
                    if (idx[j] == e)
                    {
                        used = true;
                        break;
                    }
                if (!used && choice[e] > bv)
                {
                    bv = choice[e];
                    best = e;
                }
            }
            idx[k] = best;
        }
        float sum = 0.0f;
        for (int k = 0; k < top_k; ++k) sum += scores[idx[k]];
        const float denom = sum + 1e-20f;
        for (int k = 0; k < top_k; ++k)
        {
            out_idx[k] = idx[k];
            out_w[k] = scores[idx[k]] / denom * scale;
        }
    }
}

int GridFor(int n, int threads) { return (n + threads - 1) / threads; }

} // namespace

cudaError_t RunSwigluLimit(const float *gate, const float *up, float limit,
                           int n, float *mid, cudaStream_t stream)
{
    if (!gate || !up || !mid || n < 0) return cudaErrorInvalidValue;
    if (n == 0) return cudaSuccess;
    const int threads = 256;
    swiglu_limit_kernel<<<GridFor(n, threads), threads, 0, stream>>>(
        gate, up, limit, n, mid);
    return cudaGetLastError();
}

cudaError_t RunSigmoidGate(float *y, const float *gate, int n,
                           cudaStream_t stream)
{
    if (!y || !gate || n < 0) return cudaErrorInvalidValue;
    if (n == 0) return cudaSuccess;
    const int threads = 256;
    sigmoid_gate_kernel<<<GridFor(n, threads), threads, 0, stream>>>(y, gate, n);
    return cudaGetLastError();
}

cudaError_t RunGigaChatRouterTopK(const float *x, const uint8_t *w,
                                  const float *bias, int num_experts, int hidden,
                                  int top_k, float routed_scaling_factor,
                                  int *out_idx, float *out_weights,
                                  cudaStream_t stream)
{
    if (!x || !w || !out_idx || !out_weights || num_experts <= 0 || hidden <= 0 ||
        top_k <= 0 || top_k > kMaxTopK)
        return cudaErrorInvalidValue;
    int threads = num_experts < 256 ? num_experts : 256;
    if (threads < 1) threads = 1;
    const size_t smem = static_cast<size_t>(2) * num_experts * sizeof(float);
    router_topk_kernel<<<1, threads, smem, stream>>>(
        x, reinterpret_cast<const __nv_bfloat16 *>(w), bias, num_experts, hidden,
        top_k, routed_scaling_factor, out_idx, out_weights);
    return cudaGetLastError();
}

} // namespace lsxcommon::gigachat
