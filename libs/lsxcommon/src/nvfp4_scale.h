// nvfp4_scale.h — canonical full unsigned-E4M3 (UE4M3) block-scale codecs.
// Single textual source shared by the offline converter (schema.cpp), the
// runtime act-quant kernels (lsx_tc_gemm.cu + the family inference TUs) and
// the test host references. Semantics: value = 2^(e-7)*(1+m/8), RNE encode,
// e4 field 0 -> 0 (no subnormals), 0x7F reserved. Header-only, CUDA-free,
// safe to include from non-CUDA TUs. Decode must stay in sync with
// nvfp4_codec.h E4M3ScaleToFloat (the CUDA-TU delegate).
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

// Encode s >= 0 as the nearest unsigned E4M3 (UE4M3) byte, ties-to-even.
// Layout value = 2^(e-7)*(1+m/8), e=(b>>3)&0xF, m=b&0x7. s==0 (or negative/
// NaN) -> 0x00; s < 2^-6 -> 0x08 (min normal; no subnormals); s > 448 ->
// 0x7E (max finite). The reserved NaN slot 0x7F is never emitted. Pure
// integer arithmetic so host and device are bit-identical by construction.
LSX_NVFP4_SCALE_HD inline uint8_t FloatToE4m3(float s)
{
    if (!(s > 0.0f)) return 0x00;          // 0, -0, negative, NaN
    uint32_t bits;
    std::memcpy(&bits, &s, sizeof(bits));
    const uint32_t exp8 = (bits >> 23) & 0xFFu;
    if (exp8 == 0xFFu) return 0x7E;        // Inf -> max finite
    if (exp8 == 0u) return 0x08;           // float subnormal -> min normal
    const int e = static_cast<int>(exp8) - 127;  // floor exponent
    if (e < -6) return 0x08;
    if (e > 8) return 0x7E;
    const uint32_t mant = bits & 0x7FFFFFu;
    const uint32_t q = mant >> 20;         // top 3 bits = E4M3 mantissa
    const uint32_t rem = mant & 0xFFFFFu;
    int me = static_cast<int>(q);
    const uint32_t half = 0x80000u;        // 2^19 = half a mantissa step
    if (rem > half || (rem == half && (me & 1))) ++me;  // RNE
    int E = e + 7;
    if (me == 8) { me = 0; ++E; }          // carry into the exponent
    if (E < 1) return 0x08;
    if (E > 15) return 0x7E;
    if (E == 15 && me > 6) return 0x7E;    // cannot reach the 0x7F NaN slot
    return static_cast<uint8_t>((E << 3) | me);
}

// Decode one block-scale byte: e4 field 0 -> 0.0f (field-zero contract,
// no subnormals); else 2^(e-7)*(1+m/8), exact in fp32. 0x7F (reserved,
// never emitted) -> NaN. mantissa==0 bytes reproduce the old pow2 decode
// bit-for-bit, so every existing converted tree is unchanged.
LSX_NVFP4_SCALE_HD inline float E4m3ToFloat(uint8_t b)
{
    const uint32_t e = (b >> 3) & 0xFu;
    if (e == 0u) return 0.0f;
    const uint32_t m = b & 0x7u;
    if (e == 15u && m == 7u) {
        const uint32_t nan_bits = 0x7FC00000u;
        float nan_v;
        std::memcpy(&nan_v, &nan_bits, sizeof(nan_v));
        return nan_v;
    }
    const uint32_t bits = ((e + 120u) << 23) | (m << 20);
    float v;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

// Map one MXFP4 E8M0 exponent byte (value 2^(b-127); 0 -> 0 per the clamped
// decode flavor in lsx_kernels_quant.cuh) onto the TWO per-16 pow2-E4M3
// scale bytes of the same 32-element block (DeepSeek vault NVFP4 repack).
//
// @p gs_shift_m is the per-tensor global-scale exponent: the repacked tensor
// carries global_scale = 2^m, and the NVFP4 decode multiplies by it, so the
// exactness contract is
//     E4m3ToFloat(out[0]) * 2^m == 2^(b-127)      (i.e. e4 = b-120-m)
// whenever e4 lands in [1, 14]; outside it saturates to the nearest bound
// (lossy — a correct per-tensor shift choice from ChooseE8m0ScaleShift
// keeps every real byte exact). Byte 0x00 (zero scale) maps to 0x00.
LSX_NVFP4_SCALE_HD inline void E8m0ToE4m3Pow2Pair(uint8_t e8m0, int gs_shift_m,
                                                  uint8_t out[2])
{
    if (e8m0 == 0) {
        out[0] = 0;
        out[1] = 0;
        return;
    }
    int e4 = static_cast<int>(e8m0) - 120 - gs_shift_m;
    if (e4 < 1) e4 = 1;
    if (e4 > 14) e4 = 14;                   // avoid the E4M3 NaN slot (15)
    const uint8_t mapped = static_cast<uint8_t>(static_cast<uint8_t>(e4) << 3);
    out[0] = mapped;
    out[1] = mapped;
}

// Choose the per-tensor global-scale exponent @p m for a tensor whose E8M0
// scale bytes span [@p b_min, @p b_max]: the shift maps the smallest byte
// onto e4 = 1 (min representable scale 2^-6) and stays exact as long as the
// byte spread is <= 13 (e4 range width). Returns false when the spread
// exceeds the representable window (caller decides: error out or clamp).
// Bytes above 0x7F carry no sign meaning in E8M0; the window math is pure
// exponent arithmetic, valid for the full E8M0 domain.
inline bool ChooseE8m0ScaleShift(uint8_t b_min, uint8_t b_max, int &m)
{
    if (b_min == 0 || b_max == 0) return false;  // zero scale: not a range
    if (b_max - b_min > 13) return false;
    m = static_cast<int>(b_min) - 121;
    return true;
}

} // namespace nvfp4
} // namespace lsxcommon
