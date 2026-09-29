// GigaChat3.5 ZeroCenteredGatedNorm device kernel (Plan 2a).
//
// out[r][i] = xn[i] * (1 + w[i]) * 2.0 * sigmoid( (Wdown @ silu(Wup @ xn))[i] )
// where xn = RMS-normalized x[r]. Single CUDA stream, deterministic row
// reduction (thread 0, fixed order) -> shared memory, no atomics.
#include "gigachat_moe_kernels.h"

#include "gigachat_moe_device_math.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace lsxcommon::gigachat {
namespace {

__device__ __forceinline__ float WarpSum(float v)
{
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

// Parallel ZeroCenteredGatedNorm. Fixed geometry: one block per row, blockDim
// a multiple of 32 (<=1024), the RMS reduction a warp/block tree and the
// `w_up` dot split warp-per-row, so the result is deterministic and
// device-independent (numeric-contract 1.1). The accumulation order differs
// from the former thread-0 sequential scan; verified by the selfcheck/parity.
__global__ void zero_centered_gated_norm_kernel(
    const float *__restrict__ x, int dim, int rank,
    const __nv_bfloat16 *__restrict__ w,
    const __nv_bfloat16 *__restrict__ w_up,
    const __nv_bfloat16 *__restrict__ w_down, float eps,
    float *__restrict__ out)
{
    const int r = blockIdx.x;
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = (nt + 31) >> 5;
    const float *xr = x + static_cast<size_t>(r) * dim;
    float *orow = out + static_cast<size_t>(r) * dim;
    extern __shared__ float smem[];  // [dim] xn + [kGateRank] gate
    __shared__ float red[32];

    float ss = 0.0f;
    for (int i = tid; i < dim; i += nt) ss += xr[i] * xr[i];
    ss = WarpSum(ss);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    if (warp == 0)
    {
        ss = (lane < nwarps) ? red[lane] : 0.0f;
        ss = WarpSum(ss);
        if (lane == 0) red[0] = ss;
    }
    __syncthreads();
    const float inv = rsqrtf(red[0] / static_cast<float>(dim) + eps);

    for (int i = tid; i < dim; i += nt) smem[i] = xr[i] * inv;
    __syncthreads();

    for (int k = warp; k < rank; k += nwarps)
    {
        const __nv_bfloat16 *up = w_up + static_cast<size_t>(k) * dim;
        float a = 0.0f;
        for (int i = lane; i < dim; i += 32)
            a += __bfloat162float(up[i]) * smem[i];
        a = WarpSum(a);
        if (lane == 0) smem[dim + k] = dev::Silu(a);
    }
    __syncthreads();

    for (int i = tid; i < dim; i += nt)
    {
        float g = 0.0f;
        for (int k = 0; k < rank; ++k)
            g += __bfloat162float(w_down[static_cast<size_t>(i) * rank + k]) *
                 smem[dim + k];
        orow[i] = dev::ZeroCenteredGatedNormElement(
            smem[i], __bfloat162float(w[i]), g);
    }
}

__global__ void gdn_output_norm_kernel(
    const float *__restrict__ o, const float *__restrict__ z,
    const __nv_bfloat16 *__restrict__ w, int hd, float eps, float scale,
    float *__restrict__ out)
{
    const int h = blockIdx.x;
    const float *oh = o + static_cast<size_t>(h) * hd;
    const float *zh = z + static_cast<size_t>(h) * hd;
    float *out_h = out + static_cast<size_t>(h) * hd;
    extern __shared__ float smem[];  // [hd] normalized

    if (threadIdx.x == 0)
    {
        float ss = 0.0f;
        for (int d = 0; d < hd; ++d) ss += oh[d] * oh[d];
        const float inv = rsqrtf(ss / static_cast<float>(hd) + eps);
        for (int d = 0; d < hd; ++d) smem[d] = oh[d] * inv;
    }
    __syncthreads();

    for (int d = threadIdx.x; d < hd; d += blockDim.x)
        out_h[d] = dev::GdnOutputNormElement(smem[d],
                                             __bfloat162float(w[d]), zh[d],
                                             eps, scale);
}

} // namespace

cudaError_t RunGdnOutputNorm(const float *o, const float *z, const uint8_t *w,
                             int heads, int head_dim, float eps, float scale,
                             float *out, cudaStream_t stream)
{
    if (!o || !z || !w || !out || heads <= 0 || head_dim <= 0)
        return cudaErrorInvalidValue;
    int threads = head_dim < 256 ? head_dim : 256;
    const size_t smem = static_cast<size_t>(head_dim) * sizeof(float);
    gdn_output_norm_kernel<<<heads, threads, smem, stream>>>(
        o, z, reinterpret_cast<const __nv_bfloat16 *>(w), head_dim, eps, scale,
        out);
    return cudaGetLastError();
}

cudaError_t RunZeroCenteredGatedNorm(const float *x, int rows, int dim,
                                     int rank, const uint8_t *w,
                                     const uint8_t *w_up,
                                     const uint8_t *w_down, float eps,
                                     float *out, cudaStream_t stream)
{
    if (!x || !w || !w_up || !w_down || !out || rows <= 0 || dim <= 0 ||
        rank <= 0)
        return cudaErrorInvalidValue;
    int threads = dim < 256 ? dim : 256;
    if (threads < rank) threads = rank;
    // Full warps only (the block reduction uses __shfl with a full mask).
    threads = (threads + 31) & ~31;
    if (threads > 1024) threads = 1024;
    const size_t smem = (static_cast<size_t>(dim) + rank) * sizeof(float);
    zero_centered_gated_norm_kernel<<<rows, threads, smem, stream>>>(
        x, dim, rank, reinterpret_cast<const __nv_bfloat16 *>(w),
        reinterpret_cast<const __nv_bfloat16 *>(w_up),
        reinterpret_cast<const __nv_bfloat16 *>(w_down), eps, out);
    return cudaGetLastError();
}

} // namespace lsxcommon::gigachat
