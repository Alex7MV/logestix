#include "deepseek_moe_v41_dequant.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace lsxcommon {
namespace deepseek_v41 {

namespace {

// Independent E4M3 decode (sign, exp4 bias 7, mant4/8); the E4M3xE8M0 product
// is what the loader caches. Kept in bit math so the host reference and the
// device kernel share one unambiguous definition.
__device__ __forceinline__ float E4m3ToFloat(uint8_t b)
{
    const int sign = (b >> 7) & 0x1;
    const int exp = (b >> 3) & 0xF;
    const int mant = b & 0x7;
    float v;
    if (exp == 0)
        v = static_cast<float>(mant) * (1.0f / 8.0f) * (1.0f / 64.0f);
    else
        v = (1.0f + static_cast<float>(mant) * (1.0f / 8.0f)) *
            exp2f(static_cast<float>(exp - 7));
    return sign ? -v : v;
}

__device__ __forceinline__ float E8m0ToFloat(uint8_t b)
{
    return exp2f(static_cast<float>(static_cast<int>(b) - 127));
}

__global__ void bf16_to_half_kernel(const uint16_t *__restrict__ src,
                                    __half *__restrict__ dst, int64_t n)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t bits = static_cast<uint32_t>(src[i]) << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    dst[i] = __float2half_rn(f);
}

__global__ void fp8_block_dequant_kernel(const uint8_t *__restrict__ w,
                                         const uint8_t *__restrict__ s,
                                         __half *__restrict__ out, int rows,
                                         int cols)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= cols || row >= rows) return;
    const int scols = (cols + 31) / 32;
    const float scale = E8m0ToFloat(s[static_cast<size_t>(row / 32) * scols +
                                      (col / 32)]);
    const size_t idx = static_cast<size_t>(row) * cols + col;
    out[idx] = __float2half_rn(E4m3ToFloat(w[idx]) * scale);
}

// ── FP4 activation quantize-dequant ────────────────────────────────────────

__device__ __forceinline__ float E2m1Rne(float v)
{
    const float tab[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float a = fabsf(v);
    int best = 0;
    float bd = 1e30f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float d = fabsf(a - tab[i]);
        if (d < bd) { bd = d; best = i; }
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (i != best && fabsf(fabsf(a - tab[i]) - bd) < 1e-6f &&
            (i % 2 == 0) && (best % 2 == 1))
            best = i;
    }
    return copysignf(tab[best], v);
}

// 2^ceil(log2(t)) via IEEE bit inspection (kernel.py:fast_round_scale).
__device__ __forceinline__ float FastRoundScale(float t)
{
    if (t <= 0.0f) return ldexpf(1.0f, -127);
    const uint32_t bits = __float_as_uint(t);
    const int exp = static_cast<int>((bits >> 23) & 0xFF) - 127 +
                    (((bits & 0x7FFFFF) != 0) ? 1 : 0);
    return ldexpf(1.0f, exp);
}

__device__ __forceinline__ float E4m3Round(float v)
{
    const __nv_fp8_storage_t s =
        __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    return __half2float(__nv_cvt_fp8_to_halfraw(s, __NV_E4M3));
}

__global__ void fp4_act_quant_kernel(__half *__restrict__ x, int n, int block,
                                     int e8m0)
{
    __shared__ float s_amax[32];
    const int row = blockIdx.y;
    const int base = blockIdx.x * block;
    const int j = threadIdx.x;
    __half *xr = x + static_cast<size_t>(row) * n + base;

    float a = fabsf(__half2float(xr[j]));
    s_amax[j] = a;
    __syncthreads();
    if (j == 0) {
        float m = 0.0f;
        for (int k = 0; k < block; ++k) m = fmaxf(m, s_amax[k]);
        s_amax[0] = (e8m0)
            ? FastRoundScale(fmaxf(m, 6.0f * ldexpf(1.0f, -126)) * (1.0f / 6.0f))
            : E4m3Round(fmaxf(m, 6.0f * ldexpf(1.0f, -9)) * (1.0f / 6.0f));
    }
    __syncthreads();
    const float s = s_amax[0];
    const float q = fminf(fmaxf(__half2float(xr[j]) / s, -6.0f), 6.0f);
    xr[j] = __float2half_rn(E2m1Rne(q) * s);
}

__global__ void fp8_act_quant_kernel(__half *__restrict__ x, int n, int block)
{
    __shared__ float s_amax[32];
    const int row = blockIdx.y;
    const int base = blockIdx.x * block;
    const int j = threadIdx.x;
    __half *xr = x + static_cast<size_t>(row) * n + base;

    s_amax[j] = fabsf(__half2float(xr[j]));
    __syncthreads();
    if (j == 0) {
        float m = 1e-4f;
        for (int k = 0; k < block; ++k) m = fmaxf(m, s_amax[k]);
        s_amax[0] = FastRoundScale(m / 448.0f);
    }
    __syncthreads();
    const float s = s_amax[0];
    const float q = fminf(fmaxf(__half2float(xr[j]) / s, -448.0f), 448.0f);
    xr[j] = __float2half_rn(E4m3Round(q) * s);
}

} // namespace

cudaError_t LaunchBf16ToHalf(const uint8_t *src, uint16_t *dst, int64_t n,
                             cudaStream_t stream)
{
    if (n <= 0) return cudaSuccess;
    const int threads = 256;
    const int64_t blocks = (n + threads - 1) / threads;
    bf16_to_half_kernel<<<static_cast<unsigned>(blocks), threads, 0, stream>>>(
        reinterpret_cast<const uint16_t *>(src),
        reinterpret_cast<__half *>(dst), n);
    return cudaGetLastError();
}

cudaError_t LaunchFp8BlockDequantToHalf(const uint8_t *w,
                                        const uint8_t *scales, uint16_t *out,
                                        int rows, int cols,
                                        cudaStream_t stream)
{
    if (rows <= 0 || cols <= 0) return cudaSuccess;
    if (rows % 32 != 0 || cols % 32 != 0) return cudaErrorInvalidValue;
    const dim3 block(32, 8);
    const dim3 grid((cols + block.x - 1) / block.x,
                    (rows + block.y - 1) / block.y);
    fp8_block_dequant_kernel<<<grid, block, 0, stream>>>(
        w, scales, reinterpret_cast<__half *>(out), rows, cols);
    return cudaGetLastError();
}

cudaError_t LaunchFp4ActQuantInplace(uint16_t *x, int rows, int n, int block,
                                     bool e8m0_scale, cudaStream_t stream)
{
    if (rows <= 0 || n <= 0) return cudaSuccess;
    if (n % block != 0 || block > 32)
        return cudaErrorInvalidValue;
    const dim3 grid(n / block, rows);
    fp4_act_quant_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<__half *>(x), n, block, e8m0_scale ? 1 : 0);
    return cudaGetLastError();
}

cudaError_t LaunchFp8ActQuantInplace(uint16_t *x, int rows, int n, int block,
                                     cudaStream_t stream)
{
    if (rows <= 0 || n <= 0) return cudaSuccess;
    if (n % block != 0 || block > 32) return cudaErrorInvalidValue;
    const dim3 grid(n / block, rows);
    fp8_act_quant_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<__half *>(x), n, block);
    return cudaGetLastError();
}

} // namespace deepseek_v41
} // namespace lsxcommon
