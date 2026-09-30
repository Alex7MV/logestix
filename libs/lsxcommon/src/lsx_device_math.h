// Model-neutral device math helpers.
//
// These reproduce, bit-exactly, the libm functions a host FP32 reference
// resolves to on the target build host (gcc/glibc):
//   * std::exp (gcc, glibc)  -> glibc's __expf  (e_expf.c, table-based)
//   * std::sqrtf/rsqrtf      -> glibc rsqrtf = 1.0f/sqrtf(x) (s_rsqrt_template.c)
// The CPU reference is compiled at -O0 without FMA ISA and without
// -ffast-math, so every arithmetic step below must stay a plain (non-FMA)
// IEEE operation. The target CUDA build enforces this with -fmad=false.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace lsxcommon {
namespace device {

// ── glibc __expf (sysdeps/ieee754/flt-32/e_expf.c) ─────────────────────────
// Single-precision exp that reproduces glibc's implementation bit-for-bit:
// exp(x) ~= 2^(k/N) * 2^(r/N), N = 1 << EXP2F_TABLE_BITS = 32, with the
// argument reduced by z = x*N/ln2 = k + r, r in [-1/2, 1/2], and the scaled
// minimax polynomial from e_exp2f_data.c. All computation is done in double
// with IEEE (non-contracted) operations.
__device__ __forceinline__ float ExpfGlibc(float x)
{
    constexpr int kExp2fTableBits = 5;
    constexpr int kN = 1 << kExp2fTableBits;
    // tab[i] = uint(2^(i/N)) - (i << 52-BITS)   (e_exp2f_data.c)
    constexpr uint64_t kTab[kN] = {
        0x3ff0000000000000ull, 0x3fefd9b0d3158574ull, 0x3fefb5586cf9890full,
        0x3fef9301d0125b51ull, 0x3fef72b83c7d517bull, 0x3fef54873168b9aaull,
        0x3fef387a6e756238ull, 0x3fef1e9df51fdee1ull, 0x3fef06fe0a31b715ull,
        0x3feef1a7373aa9cbull, 0x3feedea64c123422ull, 0x3feece086061892dull,
        0x3feebfdad5362a27ull, 0x3feeb42b569d4f82ull, 0x3feeab07dd485429ull,
        0x3feea47eb03a5585ull, 0x3feea09e667f3bcdull, 0x3fee9f75e8ec5f74ull,
        0x3feea11473eb0187ull, 0x3feea589994cce13ull, 0x3feeace5422aa0dbull,
        0x3feeb737b0cdc5e5ull, 0x3feec49182a3f090ull, 0x3feed503b23e255dull,
        0x3feee89f995ad3adull, 0x3feeff76f2fb5e47ull, 0x3fef199bdd85529cull,
        0x3fef3720dcef9069ull, 0x3fef5818dcfba487ull, 0x3fef7c97337b9b5full,
        0x3fefa4afa2a490daull, 0x3fefd0765b6e4540ull,
    };
    // shift = 0x1.8p+52 (1.5 * 2^52), used for the round-to-integer trick.
    constexpr double kShift = 0x1.8p+52;
    // invln2_scaled = 0x1.71547652b82fep+0 * N  ->  *32 is exact (pow of two)
    constexpr double kInvLn2N = 0x1.71547652b82fep+5;
    // poly_scaled = {P0/N^3, P1/N^2, P2/N} (exact pow-of-two scalings)
    constexpr double kC0 = 0x1.c6af84b912394p-20;
    constexpr double kC1 = 0x1.ebfce50fac4f3p-13;
    constexpr double kC2 = 0x1.62e42ff0c52d6p-6;

    const uint32_t ux = __float_as_uint(x);
    const uint32_t abstop = (ux >> 20) & 0x7ffu;
    if (abstop >= 0x42bu) {  // |x| >= 88 or x is NaN
        if (ux == 0xff800000u) return 0.0f;                 // exp(-inf) = 0
        if (abstop >= 0x7ffu) return x + x;                 // inf -> inf, NaN -> NaN
        if (x > 0x1.62e42ep6f) return __uint_as_float(0x7f800000u);  // +overflow -> +inf
        if (x < -0x1.9fe368p6f) return 0.0f;                // underflow -> 0
    }

    // z = x*N/ln2 = k + r, r in [-1/2, 1/2]; kd = round-to-nearest-even(z).
    const double xd = static_cast<double>(x);
    const double z = kInvLn2N * xd;
    const double kd = static_cast<double>(static_cast<uint64_t>(z + kShift)) - kShift;
    // glibc's converttoint returns a signed long that is then widened to
    // uint64_t; the negative-to-unsigned wrap must be well-defined, so go
    // through int64_t first.
    const uint64_t ki = static_cast<uint64_t>(static_cast<int64_t>(kd));

    const double r = z - kd;
    // exp(x) = 2^(k/N) * 2^(r/N) ~= s * (C0*r^3 + C1*r^2 + C2*r + 1)
    uint64_t t = kTab[ki % kN];
    t += ki << (52 - kExp2fTableBits);
    const double s = __longlong_as_double(static_cast<long long>(t));
    const double zz = kC0 * r + kC1;
    const double r2 = r * r;
    double y = kC2 * r + 1.0;
    y = zz * r2 + y;
    y = y * s;
    return static_cast<float>(y);
}

// ── glibc rsqrtf (math/s_rsqrt_template.c): rsqrtf(x) = 1.0f/sqrtf(x) ──────
// sqrtf (correctly rounded) followed by a correctly rounded fp32 division.
// Must NOT be contracted into an FMA/approximate rsqrt sequence; the build's
// -fmad=false guarantees the plain IEEE sequence.
__device__ __forceinline__ float RsqrtfGlibc(float x)
{
    return 1.0f / sqrtf(x);
}

}  // namespace device
}  // namespace lsxcommon
