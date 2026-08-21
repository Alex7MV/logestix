// glm_expert_kernels.cuh — single textual source for the bit-exact GLM-5.2
// kernels that were previously duplicated across glm_inference.cu and
// glm_inference_load.cu.
//
// CUDA/NVCC object-libraries do NOT link __global__ device symbols across
// separate .cu translation units: each TU compiles its own device copy. These
// kernels are therefore defined in an includable header so both .cu TUs
// compile the SAME source text (DRY), producing per-TU device copies identical
// to the bytes they previously emitted inline. All rely on the -fmad=false
// target flag of lsxcommon_glm_cu to stay non-FMA-contracted and bit-exact
// against the -O0 host reference.

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "glm_device_math.h"

namespace lsxcommon {
namespace glm {

// The kernels are enclosed in a per-TU anonymous namespace so each .cu TU that
// includes this header keeps internal-linkage host stubs (no LNK2005 across
// the two translation units of lsxcommon_glm_cu), while still compiling the
// SAME device source text in both.
namespace {

using device::GlmExpfGlibc;
using device::GlmRsqrtfGlibc;

// Silu SwiGLU, no clamp (FP32 math, FP16 in/out).
__global__ void GlmRealSwigluKernel(__half *__restrict__ gate,
                                    const __half *__restrict__ up, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = __half2float(gate[i]);
    const float u = __half2float(up[i]);
    gate[i] = __float2half((g / (1.0f + GlmExpfGlibc(-g))) * u);
}

// RMSNorm over a [rows, dim] FP16 matrix, one block per row. Single sequential
// fp32 scan over 0..dim-1 for the square sum, GlmRsqrtfGlibc(acc/dim + eps).
// -fmad=false keeps the accumulation non-contracted (matches -O0 host).
__global__ void GlmRmsNormKernel(const __half *__restrict__ x,
                                 const __half *__restrict__ w,
                                 __half *__restrict__ y,
                                 int dim, float eps)
{
    constexpr int kBlock = 256;
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float s_rms;
    if (tid == 0) {
        const __half *xr = x + row * dim;
        float acc = 0.0f;
        for (int i = 0; i < dim; ++i) {
            const float v = __half2float(xr[i]);
            acc += v * v;
        }
        s_rms = GlmRsqrtfGlibc(acc / static_cast<float>(dim) + eps);
    }
    __syncthreads();
    const float rinv = s_rms;
    for (int i = tid; i < dim; i += kBlock) {
        float v = __half2float(x[row * dim + i]);
        y[row * dim + i] = __float2half(v * rinv * __half2float(w[i]));
    }
}

// FP64-accumulate FP16 GEMM: out[t][i] = H((float)Σ_q F(w[i][q])·F(x[t][q]))
// over ascending q — bit-exact mirror of the host MatMulFp16. w row-major
// [m, k], x row-major [k, n], out column-major [m, n]. -fmad=false (no DFMA).
// Loads are vectorized (uint4 = 8 halves); GLM geometry keeps k a multiple of
// 8 so every row base stays 16-byte aligned, and the ascending-q accumulation
// order is preserved exactly (vector loads only widen the memory transactions).
__global__ void GlmGemmFp64Kernel(const __half *__restrict__ w,
                                  const __half *__restrict__ x,
                                  __half *__restrict__ out,
                                  int m, int k, int n)
{
    const int64_t total = static_cast<int64_t>(m) * n;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
    if (idx >= total) return;
    const int t = static_cast<int>(idx / m);
    const int i = static_cast<int>(idx - static_cast<int64_t>(t) * m);
    const __half *wr = w + static_cast<int64_t>(i) * k;
    const __half *xr = x + static_cast<int64_t>(t) * k;
    double acc = 0.0;
    int q = 0;
    for (; q + 8 <= k; q += 8) {
        const uint4 wv = *reinterpret_cast<const uint4 *>(wr + q);
        const uint4 xv = *reinterpret_cast<const uint4 *>(xr + q);
        const __half *wh = reinterpret_cast<const __half *>(&wv);
        const __half *xh = reinterpret_cast<const __half *>(&xv);
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            acc += static_cast<double>(__half2float(wh[j])) *
                   static_cast<double>(__half2float(xh[j]));
    }
    for (; q < k; ++q)
        acc += static_cast<double>(__half2float(wr[q])) *
               static_cast<double>(__half2float(xr[q]));
    out[idx] = __float2half(static_cast<float>(acc));
}

} // namespace (anonymous)

} // namespace glm
} // namespace lsxcommon