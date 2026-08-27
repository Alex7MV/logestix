// lsx_kernels_common.cuh — shared numeric primitives for the unified kernel
// library (Phase A). Included by every lsx_kernels_*.cuh.
//
// CastOut preserves each former call site's rounding:
//   __half        -> __float2half      (RNE, as all FP16 sites used)
//   __nv_bfloat16 -> __float2bfloat16_rn
//   float         -> identity
//
// ExpFlavor selects the exponential implementation; bit-exact models
// (GLM) instantiate ExpFlavor::Glibc which delegates to the glibc-replica
// device math. All shared arithmetic uses explicit *_rn intrinsics so
// results are independent of the -fmad object-lib flag (see CMakeLists).
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "glm_moe_device_math.h"

namespace lsxcommon {
namespace kernels {

namespace detail {

template <typename T> __device__ __forceinline__ T CastOut(float v);
template <> __device__ __forceinline__ __half CastOut<__half>(float v)
{
    return __float2half(v);
}
template <> __device__ __forceinline__ __nv_bfloat16 CastOut<__nv_bfloat16>(float v)
{
    return __float2bfloat16_rn(v);
}
template <> __device__ __forceinline__ float CastOut<float>(float v)
{
    return v;
}

template <typename T> __device__ __forceinline__ float CastIn(T v);
template <> __device__ __forceinline__ float CastIn<__half>(__half v)
{
    return __half2float(v);
}
template <> __device__ __forceinline__ float CastIn<__nv_bfloat16>(__nv_bfloat16 v)
{
    return __bfloat162float(v);
}
template <> __device__ __forceinline__ float CastIn<float>(float v)
{
    return v;
}

} // namespace detail

enum class ExpFlavor { Fast, Precise, Glibc };

template <ExpFlavor F> __device__ __forceinline__ float ExpF(float x);
template <> __device__ __forceinline__ float ExpF<ExpFlavor::Fast>(float x)
{
    return __expf(x);
}
template <> __device__ __forceinline__ float ExpF<ExpFlavor::Precise>(float x)
{
    return expf(x);
}
template <> __device__ __forceinline__ float ExpF<ExpFlavor::Glibc>(float x)
{
    return glm::device::GlmExpfGlibc(x);
}

} // namespace kernels
} // namespace lsxcommon
