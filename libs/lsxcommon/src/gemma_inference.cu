#include <cuda_bf16.h>
#include <cuda_fp16.h>

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

} // namespace gemma
} // namespace lsxcommon
