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

#include "lsx_device_math.h"

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
    return device::ExpfGlibc(x);
}

// ── Elementwise casts (dedup phase 3) ────────────────────────────────────
// One grid-stride int64_t implementation per direction replaces the former
// per-family one-shot/grid-stride copies (f32_to_f16_kernel etc.). The casts
// are elementwise with no RMW: every index is written exactly once with the
// same value regardless of grid shape, so any launch configuration that
// covered n before still covers it. Rounding is the one the copies used:
// __float2half (RNE), __float2bfloat16_rn, and the bf16->f16 composition.
// Equivalence pinned by tests/test_kernels_common.cu on n in
// {0,1,255,1024,100000} with a capped grid.
//
// Templated on the src/dst types (defaults are the canonical pair) so each
// including TU gets a weak-linkage instantiation — a plain __global__ in a
// header multiply defines its host stub when two TUs link into one binary
// (the RmsNormTreeKernel pattern). Call sites use the defaults:
// kernels::F32ToF16Kernel<<<...>>>(src, dst, n).
template <typename SrcT = float, typename DstT = __half>
__global__ void F32ToF16Kernel(const SrcT *__restrict__ src,
                               DstT *__restrict__ dst, int64_t n)
{
    for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) +
                     threadIdx.x;
         i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x)
        dst[i] = __float2half(src[i]);
}

template <typename SrcT = __half, typename DstT = float>
__global__ void F16ToF32Kernel(const SrcT *__restrict__ src,
                               DstT *__restrict__ dst, int64_t n)
{
    for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) +
                     threadIdx.x;
         i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x)
        dst[i] = __half2float(src[i]);
}

template <typename SrcT = float, typename DstT = __nv_bfloat16>
__global__ void F32ToBf16Kernel(const SrcT *__restrict__ src,
                                DstT *__restrict__ dst, int64_t n)
{
    for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) +
                     threadIdx.x;
         i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x)
        dst[i] = __float2bfloat16_rn(src[i]);
}

template <typename SrcT = __nv_bfloat16, typename DstT = __half>
__global__ void Bf16ToF16Kernel(const SrcT *__restrict__ src,
                                DstT *__restrict__ dst, int64_t n)
{
    for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) +
                     threadIdx.x;
         i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x)
        dst[i] = __float2half(__bfloat162float(src[i]));
}

template <typename SrcT = __nv_bfloat16, typename DstT = float>
__global__ void Bf16ToF32Kernel(const SrcT *__restrict__ src,
                                DstT *__restrict__ dst, int64_t n)
{
    for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) +
                     threadIdx.x;
         i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x)
        dst[i] = __bfloat162float(src[i]);
}

template <typename SrcT = __half, typename DstT = __nv_bfloat16>
__global__ void F16ToBf16Kernel(const SrcT *__restrict__ src,
                                DstT *__restrict__ dst, int64_t n)
{
    for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) +
                     threadIdx.x;
         i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x)
        dst[i] = __float2bfloat16_rn(__half2float(src[i]));
}

} // namespace kernels
} // namespace lsxcommon
