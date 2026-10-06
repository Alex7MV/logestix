// test_nvfp4_scale.cpp — canonical E4M3FN (UE4M3) NVFP4 block-scale codec.
// Contract over the valid scale domain 0x00..0x7F, bit-identical to
// cuda_fp8 / torch.float8_e4m3fn:
//   e=(b>>3)&0xF, m=b&0x7; e>=1 -> 2^(e-7)*(1+m/8);
//   e==0 -> SUBNORMAL m*2^-9 (0x00 -> 0); 0x7F reserved (NaN) never emitted;
//   bit 7 (reserved sign bit of the unsigned hardware scale) is masked.
#include "nvfp4_scale.h"
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

// Independent reference (std::ldexp), not the codec under test.
float RefDecode(uint8_t b)
{
    const int e = (b >> 3) & 0xF;
    const int m = b & 0x7;
    if (e == 0)
        return std::ldexp(static_cast<float>(m), -9);  // subnormal m * 2^-9
    if (e == 15 && m == 7) return std::nanf("");
    return std::ldexp(1.0f + static_cast<float>(m) / 8.0f, e - 7);
}

bool SameBits(float a, float b)
{
    uint32_t x, y;
    std::memcpy(&x, &a, sizeof(x));
    std::memcpy(&y, &b, sizeof(y));
    return x == y;
}

} // namespace

TEST_CASE("full-E4M3 decode: exhaustive over all 128 positive bytes",
          "[nvfp4][scale][e4m3]")
{
    for (int b = 0; b < 0x80; ++b) {
        const float got =
            lsxcommon::nvfp4::E4m3ToFloat(static_cast<uint8_t>(b));
        const float want = RefDecode(static_cast<uint8_t>(b));
        INFO("byte=0x" << std::hex << b);
        if (std::isnan(want)) {
            CHECK(std::isnan(got));
        } else {
            CHECK(SameBits(got, want));
        }
    }
}

TEST_CASE("full-E4M3 encode: RNE points and clamps",
          "[nvfp4][scale][e4m3]")
{
    using lsxcommon::nvfp4::FloatToE4m3;
    CHECK(FloatToE4m3(0.0f) == 0x00);
    CHECK(FloatToE4m3(-0.0f) == 0x00);
    CHECK(FloatToE4m3(std::nanf("")) == 0x00);
    CHECK(FloatToE4m3(1e-45f) == 0x00);          // below half the min subnormal
    CHECK(FloatToE4m3(0.0009765625f) == 0x00);   // 2^-10: tie -> even -> 0x00
    CHECK(FloatToE4m3(0.001953125f) == 0x01);    // 2^-9 = min subnormal
    CHECK(FloatToE4m3(0.0029296875f) == 0x02);   // 3*2^-10: tie -> even -> 0x02
    CHECK(FloatToE4m3(0.013671875f) == 0x07);    // 7*2^-9 = max subnormal
    CHECK(FloatToE4m3(0.0146484375f) == 0x08);   // 7.5*2^-9: carry to min normal
    CHECK(FloatToE4m3(0.0078125f) == 0x04);      // 2^-7 = 4*2^-9 subnormal
    CHECK(FloatToE4m3(0.015625f) == 0x08);       // 2^-6
    CHECK(FloatToE4m3(1.0f) == 0x38);
    CHECK(FloatToE4m3(3.0f) == 0x44);            // was 0x48 under pow2
    CHECK(FloatToE4m3(0.1f) == 0x1D);            // 2^-4*1.625
    CHECK(FloatToE4m3(448.0f) == 0x7E);
    CHECK(FloatToE4m3(449.0f) == 0x7E);
    CHECK(FloatToE4m3(1e30f) == 0x7E);
    CHECK(FloatToE4m3(256.0f) == 0x78);          // exact, was clamped to 0x70
}

TEST_CASE("full-E4M3 encode: ties-to-even", "[nvfp4][scale][e4m3]")
{
    using lsxcommon::nvfp4::FloatToE4m3;
    // 1.0625 = 1 + 1/16 -> midpoint between m=0 and m=1 -> even -> m=0.
    CHECK(FloatToE4m3(1.0625f) == 0x38);
    // 1.1875 = 1 + 3/16 -> midpoint between m=1 and m=2 -> even -> m=2.
    CHECK(FloatToE4m3(1.1875f) == 0x3A);
}

TEST_CASE("full-E4M3: pow2 bytes are bit-identical to the old codec",
          "[nvfp4][scale][e4m3]")
{
    using lsxcommon::nvfp4::FloatToE4m3;
    // Every exact power of two keeps its old byte and decode.
    for (int e = 1; e <= 14; ++e) {
        const float v = std::ldexp(1.0f, e - 7);
        const uint8_t b = lsxcommon::nvfp4::FloatToE4m3(v);
        CHECK(b == static_cast<uint8_t>(e << 3));
        CHECK(SameBits(lsxcommon::nvfp4::E4m3ToFloat(b), v));
    }
    // Subnormal contract: 0x00..0x07 decode to m*2^-9 (not 0), matching the
    // source ModelOpt W4A16_NVFP4 block scales.
    for (int b = 0; b <= 7; ++b) {
        const float want = std::ldexp(static_cast<float>(b), -9);
        CHECK(SameBits(lsxcommon::nvfp4::E4m3ToFloat(static_cast<uint8_t>(b)),
                       want));
    }
    // Bit 7 is the reserved sign bit of the unsigned hardware scale type and
    // is masked: 0x80|b decodes exactly like b.
    for (int b = 0; b <= 7; ++b)
        CHECK(lsxcommon::nvfp4::E4m3ToFloat(static_cast<uint8_t>(0x80 | b)) ==
              lsxcommon::nvfp4::E4m3ToFloat(static_cast<uint8_t>(b)));
    CHECK(FloatToE4m3(3.0f) != 0x00);            // sanity: encoder live
}
