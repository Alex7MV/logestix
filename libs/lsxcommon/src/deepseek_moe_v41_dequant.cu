#include "deepseek_moe_v41_dequant.h"

#include <cuda_fp16.h>

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

} // namespace deepseek_v41
} // namespace lsxcommon
