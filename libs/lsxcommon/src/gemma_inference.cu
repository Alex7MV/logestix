#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>

#include "lsxcommon/gemma4_inference.h"
#include "lsxcommon/gemma4_inference_load.h"

namespace lsxcommon {
namespace gemma {

namespace {

__device__ __forceinline__ unsigned int bf16_bits_wide(unsigned short bits)
{
    return static_cast<unsigned int>(bits) << 16;
}

__global__ void gemma_bf16_to_half_inplace_kernel(unsigned short *buf,
                                                  long long n)
{
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int wide = bf16_bits_wide(buf[i]);
    float f = __uint_as_float(wide);
    __half h = __float2half(f);
    buf[i] = __half_as_ushort(h);
}

} // namespace

cudaError_t GemmaBf16ToHalfInPlace(uint16_t *d_buf, int64_t numel,
                                   cudaStream_t stream)
{
    if (!d_buf || numel <= 0) return cudaErrorInvalidValue;
    if (reinterpret_cast<uintptr_t>(d_buf) % 16 != 0)
        return cudaErrorInvalidValue;
    constexpr int kBlock = 256;
    long long grid = (numel + kBlock - 1) / kBlock;
    if (grid > 2147483647LL) grid = 2147483647LL;
    gemma_bf16_to_half_inplace_kernel<<<static_cast<int>(grid), kBlock, 0,
                                        stream>>>(reinterpret_cast<unsigned short *>(d_buf),
                                                  numel);
    return cudaGetLastError();
}

// ── RMSNorm (offset-weight form) ────────────────────────────────────────────

__global__ void gemma_rmsnorm_kernel(const __half *__restrict__ x,
                                     const __half *__restrict__ w,
                                     __half *out, int rows, int hidden,
                                     float eps)
{
    const int row = blockIdx.x;
    if (row >= rows) return;
    const __half *xr = x + static_cast<int64_t>(row) * hidden;
    __half *orow = out + static_cast<int64_t>(row) * hidden;
    extern __shared__ float squares[];
    float sum = 0.0f;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float v = __half2float(xr[j]);
        sum += v * v;
    }
    squares[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) squares[threadIdx.x] += squares[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = rsqrtf(squares[0] / static_cast<float>(hidden) + eps);
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float wf = __half2float(w[j]);
        orow[j] = __float2half(__half2float(xr[j]) * inv * (1.0f + wf));
    }
}

cudaError_t RunGemmaRMSNorm(const __half *d_x, const __half *d_w, __half *d_out,
                            int rows, int hidden, float eps,
                            cudaStream_t stream)
{
    if (!d_x || !d_w || !d_out || rows <= 0 || hidden <= 0)
        return cudaErrorInvalidValue;
    constexpr int kBlock = 256;
    gemma_rmsnorm_kernel<<<rows, kBlock, kBlock * sizeof(float), stream>>>(
        d_x, d_w, d_out, rows, hidden, eps);
    return cudaGetLastError();
}

// ── RoPE ────────────────────────────────────────────────────────────────────

__global__ void gemma_rope_cos_sin_kernel(const int32_t *__restrict__ pos,
                                          __half *__restrict__ cos_buf,
                                          __half *__restrict__ sin_buf,
                                          int seq, int rotary_dim, double theta)
{
    const int64_t total = static_cast<int64_t>(seq) * (rotary_dim / 2);
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
         idx < total; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int t = static_cast<int>(idx / (rotary_dim / 2));
        const int d = static_cast<int>(idx % (rotary_dim / 2));
        double inv =
            1.0 / pow(theta, static_cast<double>(2 * d) / rotary_dim);
        double ang = static_cast<double>(pos[t]) * inv;
        cos_buf[idx] = __float2half(static_cast<float>(cos(ang)));
        sin_buf[idx] = __float2half(static_cast<float>(sin(ang)));
    }
}

// HF rotate_half pairing: pair (j, j+rot/2); cos/sin layout [seq, rot/2].
__global__ void gemma_partial_rope_kernel(const __half *__restrict__ x,
                                          const __half *__restrict__ cos_buf,
                                          const __half *__restrict__ sin_buf,
                                          __half *__restrict__ out, int seq,
                                          int heads, int head_dim,
                                          int rotary_dim)
{
    const int half = rotary_dim / 2;

    // Rotation.
    const int64_t total =
        static_cast<int64_t>(seq) * heads * static_cast<int64_t>(half);
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
         idx < total; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int j = static_cast<int>(idx % half);
        const int h = static_cast<int>((idx / half) % heads);
        const int t = static_cast<int>(idx / (static_cast<int64_t>(half) * heads));
        const float c = __half2float(cos_buf[t * half + j]);
        const float s = __half2float(sin_buf[t * half + j]);
        const int64_t row = (static_cast<int64_t>(t) * heads + h) * head_dim;
        const float x1 = __half2float(x[row + j]);
        const float x2 = __half2float(x[row + j + half]);
        out[row + j] = __float2half(x1 * c - x2 * s);
        out[row + j + half] = __float2half(x2 * c + x1 * s);
    }

    // Pass-through tail.
    const int64_t npt = static_cast<int64_t>(seq) * heads *
                        (head_dim - rotary_dim);
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
         idx < npt; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int pk = static_cast<int>(idx % (head_dim - rotary_dim));
        const int rh =
            static_cast<int>(idx / (head_dim - rotary_dim));
        const int64_t base =
            static_cast<int64_t>(rh) * head_dim + rotary_dim;
        out[base + pk] = x[base + pk];
    }
}

cudaError_t RunGemmaRopeCosSin(const int32_t *d_pos, __half *d_cos,
                               __half *d_sin, int seq, int rotary_dim,
                               double theta, cudaStream_t stream)
{
    if (!d_pos || !d_cos || !d_sin || seq <= 0 || rotary_dim <= 0 ||
        rotary_dim % 2 != 0)
        return cudaErrorInvalidValue;
    constexpr int kBlock = 256;
    const int64_t total =
        static_cast<int64_t>(seq) * (rotary_dim / 2);
    int grid = static_cast<int>((total + kBlock - 1) / kBlock);
    gemma_rope_cos_sin_kernel<<<grid, kBlock, 0, stream>>>(d_pos, d_cos, d_sin,
                                                           seq, rotary_dim,
                                                           theta);
    return cudaGetLastError();
}

cudaError_t RunGemmaPartialRope(const __half *d_x, const __half *d_cos,
                                const __half *d_sin, __half *d_out, int seq,
                                int heads, int head_dim, int rotary_dim,
                                cudaStream_t stream)
{
    if (!d_x || !d_cos || !d_sin || !d_out || seq <= 0 || heads <= 0 ||
        head_dim <= 0 || rotary_dim <= 0 || rotary_dim > head_dim ||
        rotary_dim % 2 != 0)
        return cudaErrorInvalidValue;
    constexpr int kBlock = 256;
    gemma_partial_rope_kernel<<<1024, kBlock, 0, stream>>>(
        d_x, d_cos, d_sin, d_out, seq, heads, head_dim, rotary_dim);
    return cudaGetLastError();
}

} // namespace gemma
} // namespace lsxcommon
