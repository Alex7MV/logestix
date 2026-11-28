// lsx_gemv.cu — model-neutral, device-independent GEMV for decode shapes
// (see include/lsxcommon/lsx_gemv.h). One warp per output row; the K loop is
// warp-strided and reduced with a fixed shuffle tree in fp32, so the numeric
// result is identical on every GPU in the supported set and independent of any
// workspace/heuristic (numeric-contract §1.1.1).

#include "lsxcommon/lsx_gemv.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace lsxcommon {
namespace gemv {

namespace {

__device__ __forceinline__ float ToF32(const __half v)
{
    return __half2float(v);
}
__device__ __forceinline__ float ToF32(const __nv_bfloat16 v)
{
    return __bfloat162float(v);
}
__device__ __forceinline__ float ToF32(const float v)
{
    return v;
}

template <int kOutKind>
__device__ __forceinline__ void StoreOut(void *y, int64_t i, float v)
{
    if constexpr (kOutKind == 0)
        static_cast<__half *>(y)[i] = __float2half_rn(v);
    else if constexpr (kOutKind == 1)
        static_cast<__nv_bfloat16 *>(y)[i] = __float2bfloat16_rn(v);
    else
        static_cast<float *>(y)[i] = v;
}

template <typename WT, typename XT, int kOutKind>
__global__ void GemvKernel(const WT *__restrict__ w, const XT *__restrict__ x,
                           int seq, int N, int K, float alpha,
                           void *__restrict__ y, float beta)
{
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int row = warp;
    const int s = blockIdx.y;
    if (row >= N || s >= seq) return;

    const WT *wr = w + static_cast<int64_t>(row) * K;
    const XT *xr = x + static_cast<int64_t>(s) * K;
    float acc = 0.0f;
    for (int k = lane; k < K; k += 32)
        acc += ToF32(wr[k]) * ToF32(xr[k]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
    if (lane == 0) {
        const int64_t o = static_cast<int64_t>(s) * N + row;
        float v = alpha * acc;
        if (beta != 0.0f) {
            if constexpr (kOutKind == 0)
                v += beta * __half2float(static_cast<__half *>(y)[o]);
            else if constexpr (kOutKind == 1)
                v += beta * __bfloat162float(
                                static_cast<__nv_bfloat16 *>(y)[o]);
            else
                v += beta * static_cast<float *>(y)[o];
        }
        StoreOut<kOutKind>(y, o, v);
    }
}

template <typename WT, typename XT, int kOutKind>
cudaError_t Launch(cudaStream_t stream, const void *w, const void *x, int seq,
                   int N, int K, float alpha, void *y, float beta)
{
    constexpr int kThreads = 256;  // 8 warps/block
    const int rows_per_block = kThreads / 32;
    const int grid_x = (N + rows_per_block - 1) / rows_per_block;
    dim3 grid(grid_x, seq);
    GemvKernel<WT, XT, kOutKind><<<grid, kThreads, 0, stream>>>(
        static_cast<const WT *>(w), static_cast<const XT *>(x), seq, N, K,
        alpha, y, beta);
    return cudaGetLastError();
}

int OutKind(LxDtype dt)
{
    return dt == LxDtype::kF16 ? 0 : (dt == LxDtype::kBF16 ? 1 : 2);
}

// [K, N]-major weights: y[s][n] = sum_k x[s][k] * w[k*N + n].
template <typename WT, typename XT, int kOutKind>
__global__ void GemvTWKernel(const WT *__restrict__ w, const XT *__restrict__ x,
                             int seq, int N, int K, float alpha,
                             void *__restrict__ y, float beta)
{
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int n = warp;
    const int s = blockIdx.y;
    if (n >= N || s >= seq) return;
    const XT *xr = x + static_cast<int64_t>(s) * K;
    float acc = 0.0f;
    for (int k = lane; k < K; k += 32)
        acc += ToF32(xr[k]) * ToF32(w[static_cast<int64_t>(k) * N + n]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
    if (lane == 0) {
        const int64_t o = static_cast<int64_t>(s) * N + n;
        float v = alpha * acc;
        if (beta != 0.0f) {
            if constexpr (kOutKind == 0)
                v += beta * __half2float(static_cast<__half *>(y)[o]);
            else if constexpr (kOutKind == 1)
                v += beta * __bfloat162float(
                                static_cast<__nv_bfloat16 *>(y)[o]);
            else
                v += beta * static_cast<float *>(y)[o];
        }
        StoreOut<kOutKind>(y, o, v);
    }
}

template <typename WT, typename XT, int kOutKind>
cudaError_t LaunchTW(cudaStream_t stream, const void *w, const void *x,
                     int seq, int N, int K, float alpha, void *y, float beta)
{
    constexpr int kThreads = 256;
    const int grid_x = (N + kThreads / 32 - 1) / (kThreads / 32);
    dim3 grid(grid_x, seq);
    GemvTWKernel<WT, XT, kOutKind><<<grid, kThreads, 0, stream>>>(
        static_cast<const WT *>(w), static_cast<const XT *>(x), seq, N, K,
        alpha, y, beta);
    return cudaGetLastError();
}

} // namespace

cudaError_t RunLxGemmVec(cudaStream_t stream, const void *w, const void *x,
                         int seq, int N, int K, LxDtype wdt, LxDtype xdt,
                         float alpha, void *y, LxDtype ydt, float beta)
{
    if (!w || !x || !y || seq <= 0 || N <= 0 || K <= 0)
        return cudaErrorInvalidValue;
    // Supported operand pairs (decode paths): f16 x f16, bf16 x bf16, and the
    // MIXED bf16 x f16 a `bf16` artifact needs (its projections are stored in
    // BF16, its activations stay FP16). The mixed pair is not a numeric mode:
    // ToF32() widens both operands exactly and the product is taken in fp32, so
    // it yields the same bits as f16 x f16 on the same values. Any other pair is
    // rejected below rather than reinterpreted. Output kind is dispatched at
    // compile time.
    if (wdt == LxDtype::kF16 && xdt == LxDtype::kF16) {
        switch (OutKind(ydt)) {
            case 0:
                return Launch<__half, __half, 0>(stream, w, x, seq, N, K,
                                                 alpha, y, beta);
            case 1:
                return Launch<__half, __half, 1>(stream, w, x, seq, N, K,
                                                 alpha, y, beta);
            default:
                return Launch<__half, __half, 2>(stream, w, x, seq, N, K,
                                                 alpha, y, beta);
        }
    }
    if (wdt == LxDtype::kBF16 && xdt == LxDtype::kBF16) {
        switch (OutKind(ydt)) {
            case 0:
                return Launch<__nv_bfloat16, __nv_bfloat16, 0>(
                    stream, w, x, seq, N, K, alpha, y, beta);
            case 1:
                return Launch<__nv_bfloat16, __nv_bfloat16, 1>(
                    stream, w, x, seq, N, K, alpha, y, beta);
            default:
                return Launch<__nv_bfloat16, __nv_bfloat16, 2>(
                    stream, w, x, seq, N, K, alpha, y, beta);
        }
    }
    if (wdt == LxDtype::kBF16 && xdt == LxDtype::kF16) {
        switch (OutKind(ydt)) {
            case 0:
                return Launch<__nv_bfloat16, __half, 0>(
                    stream, w, x, seq, N, K, alpha, y, beta);
            case 1:
                return Launch<__nv_bfloat16, __half, 1>(
                    stream, w, x, seq, N, K, alpha, y, beta);
            default:
                return Launch<__nv_bfloat16, __half, 2>(
                    stream, w, x, seq, N, K, alpha, y, beta);
        }
    }
    if (wdt == LxDtype::kF32 && xdt == LxDtype::kF32) {
        switch (OutKind(ydt)) {
            case 0:
                return Launch<float, float, 0>(stream, w, x, seq, N, K, alpha,
                                               y, beta);
            case 1:
                return Launch<float, float, 1>(stream, w, x, seq, N, K, alpha,
                                               y, beta);
            default:
                return Launch<float, float, 2>(stream, w, x, seq, N, K, alpha,
                                               y, beta);
        }
    }
    return cudaErrorNotSupported;
}

cudaError_t RunLxGemmVecWT(cudaStream_t stream, const void *w, const void *x,
                           int seq, int N, int K, LxDtype wdt, LxDtype xdt,
                           float alpha, void *y, LxDtype ydt, float beta)
{
    if (!w || !x || !y || seq <= 0 || N <= 0 || K <= 0)
        return cudaErrorInvalidValue;
    if (wdt == LxDtype::kF16 && xdt == LxDtype::kF16) {
        switch (OutKind(ydt)) {
            case 0:
                return LaunchTW<__half, __half, 0>(stream, w, x, seq, N, K,
                                                   alpha, y, beta);
            case 1:
                return LaunchTW<__half, __half, 1>(stream, w, x, seq, N, K,
                                                   alpha, y, beta);
            default:
                return LaunchTW<__half, __half, 2>(stream, w, x, seq, N, K,
                                                   alpha, y, beta);
        }
    }
    if (wdt == LxDtype::kBF16 && xdt == LxDtype::kBF16) {
        switch (OutKind(ydt)) {
            case 0:
                return LaunchTW<__nv_bfloat16, __nv_bfloat16, 0>(
                    stream, w, x, seq, N, K, alpha, y, beta);
            case 1:
                return LaunchTW<__nv_bfloat16, __nv_bfloat16, 1>(
                    stream, w, x, seq, N, K, alpha, y, beta);
            default:
                return LaunchTW<__nv_bfloat16, __nv_bfloat16, 2>(
                    stream, w, x, seq, N, K, alpha, y, beta);
        }
    }
    // Same mixed pair as RunLxGemmVec above (bf16 artifact, fp16 activations).
    if (wdt == LxDtype::kBF16 && xdt == LxDtype::kF16) {
        switch (OutKind(ydt)) {
            case 0:
                return LaunchTW<__nv_bfloat16, __half, 0>(
                    stream, w, x, seq, N, K, alpha, y, beta);
            case 1:
                return LaunchTW<__nv_bfloat16, __half, 1>(
                    stream, w, x, seq, N, K, alpha, y, beta);
            default:
                return LaunchTW<__nv_bfloat16, __half, 2>(
                    stream, w, x, seq, N, K, alpha, y, beta);
        }
    }
    return cudaErrorNotSupported;
}

} // namespace gemv
} // namespace lsxcommon
