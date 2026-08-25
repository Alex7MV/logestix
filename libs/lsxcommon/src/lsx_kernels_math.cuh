// lsx_kernels_math.cuh — unified elementwise math kernels (Phase A).
// Replaces per-model add/convert/embed copies in qwen_inference.cu,
// qwen_moe_inference.cu, glm_inference.cu, kimi_inference.cu and the
// deepseek pipeline. Arithmetic in FP32 via explicit intrinsics
// (fmad-flag independent); output rounding per CastOut.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "lsx_kernels_common.cuh"

namespace lsxcommon {
namespace kernels {

// dst[i] = dst[i] + src[i]   (former qwen_add_kernel / qwen_moe_add_kernel /
//                             GlmAddResidualKernel)
template <typename T>
__global__ void AddInPlaceKernel(const T* __restrict__ src, T* __restrict__ dst,
                                 int64_t n)
{
    const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    dst[i] = detail::CastOut<T>(
        __fadd_rn(detail::CastIn(dst[i]), detail::CastIn(src[i])));
}

// out[i] = a[i] + b[i]       (former GlmResidualKernel / kimi_residual_add_kernel)
template <typename T>
__global__ void AddOutKernel(const T* __restrict__ a, const T* __restrict__ b,
                             T* __restrict__ out, int64_t n)
{
    const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    out[i] = detail::CastOut<T>(
        __fadd_rn(detail::CastIn(a[i]), detail::CastIn(b[i])));
}

// out[t, h] = table[ids[t], h]   (former qwen embed_kernel / GlmEmbedKernel /
//                                 kimi_embed_kernel)
template <typename T>
__global__ void EmbedGatherKernel(const T* __restrict__ table,
                                  const int32_t* __restrict__ ids,
                                  T* __restrict__ out, int seq, int hidden)
{
    const int64_t total = static_cast<int64_t>(seq) * hidden;
    const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i >= total) return;
    const int t = static_cast<int>(i / hidden);
    const int h = static_cast<int>(i - static_cast<int64_t>(t) * hidden);
    out[i] = table[static_cast<int64_t>(ids[t]) * hidden + h];
}

// dst[i] = clamp(src[i], +-clamp_abs) cast to T
// (former f2h converts with clamp_abs = INFINITY; deepseek clamps use 65504)
template <typename T>
__global__ void F32toTKernel(const float* __restrict__ src, T* __restrict__ dst,
                             int64_t n, float clamp_abs = INFINITY)
{
    const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    const float v = fmaxf(-clamp_abs, fminf(clamp_abs, src[i]));
    dst[i] = detail::CastOut<T>(v);
}

} // namespace kernels
} // namespace lsxcommon
