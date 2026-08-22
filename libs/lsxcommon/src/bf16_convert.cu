#include "lsxcommon/fast_load.h"

#include <cuda_fp16.h>

namespace lsxcommon {
namespace fastload {
namespace {

// In-place BF16→FP16 conversion, bit-exact mirror of the legacy host scalar
// path: bf16 bits<<16 reinterpreted as FP32, rounded to nearest-even FP16.
__global__ void bf16_to_half_in_place_kernel(uint16_t *__restrict__ buf,
                                             int64_t n)
{
    constexpr int64_t kVec = 8;
    const int64_t nv = n / kVec;
    const int64_t stride =
        static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t v = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
         v < nv; v += stride) {
        uint4 raw = *reinterpret_cast<const uint4 *>(buf + v * kVec);
        uint16_t w[8] = {
            static_cast<uint16_t>(raw.x & 0xFFFFu),
            static_cast<uint16_t>(raw.x >> 16),
            static_cast<uint16_t>(raw.y & 0xFFFFu),
            static_cast<uint16_t>(raw.y >> 16),
            static_cast<uint16_t>(raw.z & 0xFFFFu),
            static_cast<uint16_t>(raw.z >> 16),
            static_cast<uint16_t>(raw.w & 0xFFFFu),
            static_cast<uint16_t>(raw.w >> 16),
        };
        __half h[8];
#pragma unroll
        for (int k = 0; k < 8; ++k)
            h[k] = __float2half(__uint_as_float(
                static_cast<uint32_t>(w[k]) << 16));
        uint16_t o[8];
#pragma unroll
        for (int k = 0; k < 8; ++k)
            o[k] = *reinterpret_cast<uint16_t *>(&h[k]);
        uint4 ov = make_uint4(
            o[0] | (static_cast<uint32_t>(o[1]) << 16),
            o[2] | (static_cast<uint32_t>(o[3]) << 16),
            o[4] | (static_cast<uint32_t>(o[5]) << 16),
            o[6] | (static_cast<uint32_t>(o[7]) << 16));
        *reinterpret_cast<uint4 *>(buf + v * kVec) = ov;
    }
    const int64_t tail0 = nv * kVec;
    for (int64_t i = tail0 + static_cast<int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
         i < n; i += stride) {
        __half hv =
            __float2half(__uint_as_float(static_cast<uint32_t>(buf[i]) << 16));
        buf[i] = *reinterpret_cast<uint16_t *>(&hv);
    }
}

} // namespace

cudaError_t Bf16ToHalfInPlace(uint16_t *d_buf, int64_t numel,
                              cudaStream_t stream)
{
    if (!d_buf || numel <= 0) return cudaErrorInvalidValue;
    constexpr int kBlock = 256;
    const int64_t nv = numel / 8;
    int64_t blocks64 = (nv + kBlock - 1) / kBlock;
    if (blocks64 < 1) blocks64 = 1;
    if (blocks64 > 65535) blocks64 = 65535;
    // Drain any stale thread-local CUDA error left by unrelated earlier
    // calls, so cudaGetLastError() below reflects THIS launch only.
    (void)cudaGetLastError();
    bf16_to_half_in_place_kernel<<<static_cast<int>(blocks64), kBlock, 0,
                                   stream>>>(d_buf, numel);
    return cudaGetLastError();
}

} // namespace fastload
} // namespace lsxcommon
