// lsx_kernels_norm.cuh — unified RMSNorm (parallel tree-reduction family).
// Covers qwen_inference.cu rmsnorm_offset_kernel and both gemma kernels
// (direct-weight / no-weight). Bit-exact models keep their own kernels:
//   - glm_expert_kernels.cuh GlmRmsNormKernel: thread-0 SEQUENTIAL scan +
//     GlmRsqrtfGlibc (byte-exact host mirror) — not replicable by a tree.
//   - kimi_rmsnorm_rows*: same sequential structure, BF16.
//   - deepseek rms_norm_float_kernel: F32 rows, shared via load.cuh pair.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "lsx_kernels_common.cuh"

namespace lsxcommon {
namespace kernels {

enum class NormWeight { None, Direct, Offset1PlusW };

template <typename T, NormWeight W>
__global__ void RmsNormTreeKernel(const T* __restrict__ x,
                                  const T* __restrict__ w, T* __restrict__ out,
                                  int rows, int hidden, float eps)
{
    const int row = blockIdx.x;
    if (row >= rows) return;
    const T* xr = x + static_cast<int64_t>(row) * hidden;
    T* orow = out + static_cast<int64_t>(row) * hidden;
    extern __shared__ float squares[];
    float sum = 0.0f;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        const float v = detail::CastIn(xr[j]);
        sum = __fadd_rn(sum, __fmul_rn(v, v));
    }
    squares[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            squares[threadIdx.x] = __fadd_rn(squares[threadIdx.x],
                                             squares[threadIdx.x + s]);
        __syncthreads();
    }
    const float inv =
        rsqrtf(__fadd_rn(__fdiv_rn(squares[0], static_cast<float>(hidden)), eps));
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float v = detail::CastIn(xr[j]);
        if (W == NormWeight::None) {
            orow[j] = detail::CastOut<T>(__fmul_rn(v, inv));
        } else if (W == NormWeight::Direct) {
            orow[j] = detail::CastOut<T>(
                __fmul_rn(__fmul_rn(v, inv), detail::CastIn(w[j])));
        } else {
            orow[j] = detail::CastOut<T>(
                __fmul_rn(__fmul_rn(v, inv),
                          __fadd_rn(1.0f, detail::CastIn(w[j]))));
        }
    }
}

} // namespace kernels
} // namespace lsxcommon
