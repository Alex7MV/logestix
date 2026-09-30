// MiMo-V2.6-Flash-RL non-GEMM kernels (RMSNorm / router / SwiGLU). Shared
// primitives from lsx_kernels_*.cuh; Glibc exp flavor for host parity.
#include "lsxcommon/mimo_moe_kernels.h"

#include "lsx_kernels_act.cuh"
#include "lsx_kernels_common.cuh"
#include "lsx_kernels_norm.cuh"

namespace lsxcommon {
namespace mimo {

cudaError_t RunMimoRmsNorm(cudaStream_t stream, const __half *x,
                           const __half *w, int rows, int hidden, float eps,
                           __half *out)
{
    if (rows <= 0 || hidden <= 0) return cudaSuccess;
    const int threads = 256;
    const size_t smem = static_cast<size_t>(threads) * sizeof(float) +
                        static_cast<size_t>(hidden) * sizeof(__half);
    kernels::RmsNormTreeKernel<__half, kernels::NormWeight::Direct>
        <<<rows, threads, smem, stream>>>(x, w, out, rows, hidden, eps);
    return cudaGetLastError();
}

__global__ void mimo_router_kernel(const __half *__restrict__ x,
                                   const __nv_bfloat16 *__restrict__ gate_w,
                                   const float *__restrict__ bias, int hidden,
                                   int num_experts, int top_k,
                                   int *__restrict__ idx,
                                   float *__restrict__ w)
{
    extern __shared__ float sh[];  // num_experts logits
    const int row = blockIdx.x;
    const __half *xr = x + static_cast<int64_t>(row) * hidden;
    for (int e = threadIdx.x; e < num_experts; e += blockDim.x)
    {
        const __nv_bfloat16 *wr = gate_w + static_cast<int64_t>(e) * hidden;
        float dot = 0.0f;
        for (int k = 0; k < hidden; ++k)
            dot = __fadd_rn(dot, __fmul_rn(__half2float(xr[k]),
                                           __bfloat162float(wr[k])));
        sh[e] = dot;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;

    const int max_e = 256;
    float scores[max_e];
    for (int i = 0; i < num_experts; ++i)
    {
        const float s = 1.0f / (1.0f + kernels::ExpF<kernels::ExpFlavor::Glibc>(-sh[i]));
        scores[i] = s;
        sh[i] = s + bias[i];  // choice = sigmoid(logit) + bias
    }
    int sel[8];
    for (int t = 0; t < top_k; ++t)
    {
        int best = -1;
        float bv = -3.4e38f;
        for (int i = 0; i < num_experts; ++i)
        {
            bool used = false;
            for (int u = 0; u < t; ++u) used = used || (sel[u] == i);
            if (used) continue;
            if (sh[i] > bv) { bv = sh[i]; best = i; }
        }
        sel[t] = best;
    }
    float sum = 0.0f;
    for (int t = 0; t < top_k; ++t) sum += scores[sel[t]];
    for (int t = 0; t < top_k; ++t)
    {
        idx[row * top_k + t] = sel[t];
        w[row * top_k + t] = scores[sel[t]] / (sum + 1e-20f);
    }
}

cudaError_t RunMimoRouter(cudaStream_t stream, const __half *x,
                          const __nv_bfloat16 *gate_w, const float *bias,
                          int rows, int hidden, int num_experts, int top_k,
                          int *idx_out, float *w_out)
{
    if (rows <= 0) return cudaSuccess;
    if (num_experts <= 0 || num_experts > 256 || top_k <= 0 || top_k > 8)
        return cudaErrorInvalidValue;
    const int threads = num_experts < 256 ? num_experts : 256;
    const size_t smem = static_cast<size_t>(num_experts) * sizeof(float);
    mimo_router_kernel<<<rows, threads, smem, stream>>>(
        x, gate_w, bias, hidden, num_experts, top_k, idx_out, w_out);
    return cudaGetLastError();
}

cudaError_t RunMimoSwiGLU(cudaStream_t stream, const __half *gate,
                          const __half *up, int64_t n, __half *out)
{
    if (n <= 0) return cudaSuccess;
    const int threads = 256;
    const int64_t blocks = (n + threads - 1) / threads;
    kernels::SwigluKernel<__half, kernels::ExpFlavor::Glibc, false>
        <<<static_cast<int>(blocks), threads, 0, stream>>>(gate, up, out, n,
                                                           0.0f);
    return cudaGetLastError();
}

} // namespace mimo
} // namespace lsxcommon
