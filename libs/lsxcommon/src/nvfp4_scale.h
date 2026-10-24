// nvfp4_scale.h — canonical unsigned-E4M3 (UE4M3) block-scale codecs.
// Single textual source shared by the offline converter (schema.cpp), the
// runtime act-quant kernels (lsx_tc_gemm.cu + the family inference TUs) and
// the test host references. Header-only, CUDA-free, safe to include from
// non-CUDA TUs. Decode must stay in sync with nvfp4_codec.h
// E4M3ScaleToFloat (the CUDA-TU delegate).
//
// Semantics: the COMPLETE OCP/NVIDIA E4M3 (E4M3FN) value table over the valid
// scale domain (bit 7 clear) —
//   * e4 = (b>>3) & 0xF, m = b & 0x7;
//   * e4 >= 1 : value = 2^(e-7)*(1+m/8);
//   * e4 == 0 : SUBNORMAL value = m * 2^-9 (0x00 -> 0) — the block scales of
//     the pre-quantized ModelOpt W4A16_NVFP4 checkpoints (e.g.
//     NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) legitimately use these
//     bytes, so decoding them as 0 loses real blocks;
//   * encode is round-to-nearest-even, INCLUDING the subnormal grid, so
//     decode(encode(x)) is the nearest representable value for every x;
//   * 0x7F (e4==15, m==7) is the reserved NaN slot and is never emitted.
// Over 0x00..0x7F this is bit-identical to cuda_fp8 / torch.float8_e4m3fn, so
// a source block-scale byte decodes exactly as NVIDIA's decoder reads it.
// Bit 7 is the reserved sign bit of the unsigned hardware scale type
// (`float_ue4m3_t`); the encoder never sets it and the decoder masks it.
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

// Encode s >= 0 as the nearest E4M3FN byte, ties-to-even. Layout e=(b>>3)&0xF,
// m=b&0x7: e>=1 -> 2^(e-7)*(1+m/8); e==0 -> subnormal m*2^-9. s==0 (or
// negative/NaN) -> 0x00; s in the subnormal range rounds on the 2^-9 grid
// (m==8 carries up to the min normal 0x08); s > 448 -> 0x7E (max finite). The
// reserved NaN slot 0x7F is never emitted. Host and device are bit-identical:
// the normal path is pure integer arithmetic and the subnormal path is a
// single IEEE multiply plus an integer RNE.
LSX_NVFP4_SCALE_HD inline uint8_t FloatToE4m3(float s)
{
    if (!(s > 0.0f)) return 0x00;          // 0, -0, negative, NaN
    uint32_t bits;
    std::memcpy(&bits, &s, sizeof(bits));
    const uint32_t exp8 = (bits >> 23) & 0xFFu;
    if (exp8 == 0xFFu) return 0x7E;        // Inf -> max finite
    if (exp8 < 121u) {                     // s < 2^-6, or a float subnormal
        // E4M3FN subnormal range (e4 field 0): nearest value on the 2^-9 grid,
        // ties-to-even; m == 8 carries into the min normal 0x08 = 2^-6. Below
        // half the smallest subnormal (2^-10) the value rounds to 0x00.
        const float t = s * 512.0f;        // s / 2^-9
        int m = static_cast<int>(t);       // t >= 0, truncation is floor
        const float frac = t - static_cast<float>(m);
        if (frac > 0.5f || (frac == 0.5f && (m & 1))) ++m;
        if (m <= 0) return 0x00;
        if (m >= 8) return 0x08;
        return static_cast<uint8_t>(m);
    }
    const int e = static_cast<int>(exp8) - 127;  // floor exponent
    if (e > 8) return 0x7E;
    const uint32_t mant = bits & 0x7FFFFFu;
    const uint32_t q = mant >> 20;         // top 3 bits = E4M3 mantissa
    const uint32_t rem = mant & 0xFFFFFu;
    int me = static_cast<int>(q);
    const uint32_t half = 0x80000u;        // 2^19 = half a mantissa step
    if (rem > half || (rem == half && (me & 1))) ++me;  // RNE
    int E = e + 7;
    if (me == 8) { me = 0; ++E; }          // carry into the exponent
    // E >= 1 always: this branch is only reached for exp8 >= 121 (s >= 2^-6),
    // so e >= -6 and E = e + 7 >= 1. The old `if (E < 1) return 0x08;` clamp
    // became unreachable when the subnormal range moved below the normal path.
    if (E > 15) return 0x7E;
    if (E == 15 && me > 6) return 0x7E;    // cannot reach the 0x7F NaN slot
    return static_cast<uint8_t>((E << 3) | me);
}

// Decode one block-scale byte: e4 == 0 -> the E4M3FN SUBNORMAL m * 2^-9
// (0x00 -> 0.0f); else 2^(e-7)*(1+m/8), exact in fp32. 0x7F (reserved, never
// emitted) -> NaN. Bit 7 is the reserved sign bit of the unsigned UE4M3
// hardware scale type and is masked off, so a byte with bit 7 set decodes by
// its low 7 bits. Over 0x00..0x7F this is bit-identical to
// cuda_fp8 / torch.float8_e4m3fn, so a pre-quantized ModelOpt W4A16_NVFP4
// block scale decodes exactly as NVIDIA reads it (subnormals included).
//
// Shape matters, not just semantics: this is the inner-loop scale decode of the
// seq==1 streaming GEMV, so every instruction lands on a DRAM-latency-bound
// loop. The early-return form (NaN test, then the subnormal return, then the
// normal path) measured 131.5-132.1 t/s on phi-4 at 2048 tokens; the
// one-return form below, with both rare cases as selects on an already computed
// register, measures 132.5-133.1 (+0.8%) with byte-identical results. The
// pre-codec pow2-only decode, kept as the cost reference, read 133.1-133.6.
// Pinned by tests/test_nvfp4_scale.cpp (exhaustive host) and the exhaustive
// device-parity case in tests/test_lsx_kernels.cu.
LSX_NVFP4_SCALE_HD inline float E4m3ToFloat(uint8_t b)
{
    const uint32_t e = (b >> 3) & 0xFu;
    const uint32_t m = b & 0x7u;
    const uint32_t bits = ((e + 120u) << 23) | (m << 20);
    // 0x7F (e=15, m=7) is the reserved NaN slot: its normal-path bit pattern
    // is 0x43F00000, so one compare on the folded register covers it without
    // an extra compare on (e, m).
    const uint32_t nan_bits = 0x7FC00000u;
    float v, nan_v;
    std::memcpy(&v, &bits, sizeof(v));
    std::memcpy(&nan_v, &nan_bits, sizeof(nan_v));
    // e == 0 decodes to the subnormal m * 2^-9. Folding it out of the
    // normal-path value costs one fused multiply-add and no int->float
    // convert: v = 2^-7 * (1 + m/8) there, so 2*v - 2^-6 == m * 2^-9 exactly
    // (both operands are dyadic and the subtraction is Sterbenz-exact).
    const float sub = 2.0f * v - 0.015625f;  // == m * 2^-9
    return (bits == 0x43F00000u) ? nan_v : ((e == 0u) ? sub : v);
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
