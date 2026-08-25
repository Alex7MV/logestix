// nvfp4_scale.h — canonical power-of-two E4M3 block-scale codecs (NVFP4).
// Single textual source shared by the offline converter (schema.cpp), the
// runtime act-quant kernel (qwen_tc_gemm.cu) and the TC test host reference.
// Semantics: exponent rounds UP (ceil) so the block maximum never saturates;
// exact powers of two map onto themselves. Header-only, CUDA-free, safe to
// include from non-CUDA TUs.
#pragma once

#include <cstdint>
#include <cstring>

#ifdef __CUDACC__
#define LSX_NVFP4_SCALE_HD __host__ __device__
#else
#define LSX_NVFP4_SCALE_HD
#endif

namespace lsxcommon {
namespace nvfp4 {

// Encode s >= 0 as the smallest power of two >= s in the pow2-E4M3 payload
// layout (byte = e4 << 3, value 2^(e4-7)). s == 0 -> 0 (reserved zero);
// float subnormals clamp to the minimum normal scale 2^-6 (byte 0x08);
// e4 clamps to [1, 14] (max scale 2^7, avoiding the E4M3 NaN slot).
LSX_NVFP4_SCALE_HD inline uint8_t FloatToE4m3P2(float s)
{
    if (s == 0.0f) return 0;
    uint32_t bits;
    std::memcpy(&bits, &s, sizeof(bits));
    const int exp8 = static_cast<int>((bits >> 23) & 0xFF);
    if (exp8 == 0) return 0x08;
    const uint32_t mant = bits & 0x7FFFFFu;
    int p2_exp = exp8 - 127;
    if (mant != 0) ++p2_exp;
    int e4 = p2_exp + 7;
    if (e4 < 1) e4 = 1;
    if (e4 > 14) e4 = 14;
    return static_cast<uint8_t>(static_cast<uint8_t>(e4) << 3);
}

// Exact inverse of FloatToE4m3P2: byte -> 2^(e4-7); 0 -> 0.0f.
LSX_NVFP4_SCALE_HD inline float E4m3P2ToFloat(uint8_t e4m3)
{
    if (e4m3 == 0) return 0.0f;
    const int exp_unb = static_cast<int>((e4m3 >> 3) & 0x0F) - 7;
    const uint32_t bits = static_cast<uint32_t>(exp_unb + 127) << 23;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

} // namespace nvfp4
} // namespace lsxcommon
