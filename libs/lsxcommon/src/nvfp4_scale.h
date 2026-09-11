// nvfp4_scale.h — canonical power-of-two E4M3 block-scale codecs (NVFP4).
// Single textual source shared by the offline converter (schema.cpp), the
// runtime act-quant kernel (lsx_tc_gemm.cu) and the TC test host reference.
// Semantics: exponent rounds UP (ceil) so the block maximum never saturates;
// exact powers of two map onto themselves. Header-only, CUDA-free, safe to
// include from non-CUDA TUs. Decode must stay in sync with
// nvfp4_codec.h E4M3ScaleToFloat (CUDA-TU counterpart).
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

// Exact inverse of FloatToE4m3P2: byte -> 2^(e4-7). The ZERO test is on the
// e4 FIELD, not the whole byte: bytes 0x80..0x87 carry a zero exponent field
// and decode to exactly 0 (per E4M3 semantics they would be -0/negative
// subnormals — never legitimate scale factors). Decoding them as 2^-7 via a
// whole-byte zero check silently corrupts GEMMs whose stored scale patterns
// contain such bytes (pinned by tests/test_qwen_nvfp4_tile_gemm.cu).
LSX_NVFP4_SCALE_HD inline float E4m3P2ToFloat(uint8_t e4m3)
{
    if (((e4m3 >> 3) & 0x0F) == 0) return 0.0f;
    const int exp_unb = static_cast<int>((e4m3 >> 3) & 0x0F) - 7;
    const uint32_t bits = static_cast<uint32_t>(exp_unb + 127) << 23;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// Map one MXFP4 E8M0 exponent byte (value 2^(b-127); 0 -> 0, 0xFF -> FLT_MAX
// per the clamped decode flavor in lsx_kernels_quant.cuh) onto the TWO
// per-16 pow2-E4M3 scale bytes of the same 32-element block (DeepSeek vault
// NVFP4 repack). Exact for the representable range b in [121..134]
// (2^-6 .. 2^7); outside it saturates to the nearest representable scale —
// the finite-clamp flavor the routed-expert FFN already relies on. Byte
// 0x00 (zero scale) maps to 0x00 (decode: 0.0f).
LSX_NVFP4_SCALE_HD inline void E8m0ToE4m3Pow2Pair(uint8_t e8m0, uint8_t out[2])
{
    if (e8m0 == 0) {
        out[0] = 0;
        out[1] = 0;
        return;
    }
    int e4 = static_cast<int>(e8m0) - 120;  // 2^(b-127) == 2^(e4-7)
    if (e8m0 == 0xFF) e4 = 14;              // FLT_MAX flavor: clamp finite
    if (e4 < 1) e4 = 1;
    if (e4 > 14) e4 = 14;                   // avoid the E4M3 NaN slot (15)
    const uint8_t mapped = static_cast<uint8_t>(static_cast<uint8_t>(e4) << 3);
    out[0] = mapped;
    out[1] = mapped;
}

} // namespace nvfp4
} // namespace lsxcommon
