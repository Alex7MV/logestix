// deepseek_moe_v41_fp8_codec.h — device-side FP8/E8M0 decode for
// DeepSeek-V4.1-Flash dense weights. Single source of the exact semantics the
// loader's fp8_block_dequant_kernel materialized (sign, exp4 bias 7, mant4/8;
// E8M0 = 2^(b-127) with b=0 -> 2^-127, NOT the clamped-flavor 0). The on-the-fly
// FP8 GEMV reproduces the loader's FP16 rounding so the two paths are
// bit-identical.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace lsxcommon {
namespace deepseek_v41 {

__device__ __forceinline__ float Fp8E4m3ToFloat(uint8_t b)
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

__device__ __forceinline__ float Fp8E8m0ToFloat(uint8_t b)
{
    return exp2f(static_cast<float>(static_cast<int>(b) - 127));
}

// Bit-identical, exp2f-free fast variants for the decode GEMV's inner loop.
// E4M3: (1+m/8)*2^(e-7) is realizable exactly as a float by assembling the
// exponent field e+120; e=15,m=7 therefore also yields 480 (the formula's
// value, not a NaN). Subnormal e=0 is m*2^-9. E8M0: 2^(b-127) is the exponent
// field b; b=0 reads 0.0 here vs 2^-127 in the formula, but the GEMV rounds the
// product to FP16, where both underflow to 0 -- so the post-round result is
// unchanged.
__device__ __forceinline__ float Fp8E4m3ToFloatFast(uint8_t b)
{
    const int s = (b >> 7) & 0x1;
    const int e = (b >> 3) & 0xF;
    const int m = b & 0x7;
    float v;
    if (e == 0)
        v = static_cast<float>(m) * 0.001953125f;  // m * 2^-9
    else
        v = (1.0f + static_cast<float>(m) * 0.125f) *
            __uint_as_float(static_cast<uint32_t>(e + 120) << 23);
    return s ? -v : v;
}

__device__ __forceinline__ float Fp8E8m0ToFloatFast(uint8_t b)
{
    return __uint_as_float(static_cast<uint32_t>(b) << 23);
}

} // namespace deepseek_v41
} // namespace lsxcommon
