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

} // namespace deepseek_v41
} // namespace lsxcommon
