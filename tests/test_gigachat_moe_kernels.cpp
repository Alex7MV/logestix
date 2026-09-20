// TDD — GigaChat3.5 numeric kernels. The FP8 block-128 dequant parity test
// lives here now; Plan 2 appends the device-kernel parity tests.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "lsxcommon/gigachat_moe_convert.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace lsxcommon::gigachat;

namespace {

float Fp8E4M3ToFloat(uint8_t b)
{
    const int s = (b >> 7) & 1, e = (b >> 3) & 0xF, m = b & 0x7;
    if (e == 0)
        return (s ? -1.f : 1.f) * std::ldexp(static_cast<float>(m) / 8.0f, -6);
    if (e == 15 && m == 7) return 0.0f;  // NaN encoding: unused in this fixture
    return (s ? -1.f : 1.f) * std::ldexp(1.0f + static_cast<float>(m) / 8.0f, e - 7);
}

float Bf16ToFloat(uint16_t b)
{
    const int s = (b >> 15) & 1, e = (b >> 7) & 0xFF, m = b & 0x7F;
    float v;
    if (e == 0)
        v = (static_cast<float>(m) / 128.0f) * std::ldexp(1.0f, -126);
    else
        v = (1.0f + static_cast<float>(m) / 128.0f) * std::ldexp(1.0f, e - 127);
    return s ? -v : v;
}

} // namespace

TEST_CASE("gigachat fp8 block-128 dequant parity, partial tile",
          "[gigachat][inference]")
{
    // 130 rows x 130 cols -> scale grid ceil(130/128)=2 x 2 (partial last tile).
    const int R = 130, C = 130;
    std::vector<uint8_t> fp8(static_cast<size_t>(R) * C);
    for (int i = 0; i < R * C; ++i)
        fp8[i] = static_cast<uint8_t>((i % 15) | 0x08);
    std::vector<uint8_t> scale(2 * 2 * 4);
    const float s0 = 0.5f, s1 = 2.0f, s2 = 1.0f, s3 = 0.25f;
    std::memcpy(scale.data() + 0, &s0, 4);
    std::memcpy(scale.data() + 4, &s1, 4);
    std::memcpy(scale.data() + 8, &s2, 4);
    std::memcpy(scale.data() + 12, &s3, 4);

    std::vector<uint8_t> bf16;
    auto st = DequantFp8Block128ToBf16Gpu(fp8, {R, C}, scale, {2, 2}, bf16);
    REQUIRE(st.ok());
    REQUIRE(bf16.size() == static_cast<size_t>(R) * C * 2);

    auto bf16_at = [&](int r, int c) {
        const uint16_t b = static_cast<uint16_t>(bf16[(r * C + c) * 2]) |
                           (static_cast<uint16_t>(bf16[(r * C + c) * 2 + 1]) << 8);
        return Bf16ToFloat(b);
    };

    // Tile scales apply per 128x128 block; the last row/col use tile [1][1].
    const float expect00 = Fp8E4M3ToFloat(fp8[0]) * s0;
    CHECK(bf16_at(0, 0) == Catch::Approx(expect00).margin(1e-3));
    const float expect_last = Fp8E4M3ToFloat(fp8[129 * C + 129]) * s3;
    CHECK(bf16_at(129, 129) == Catch::Approx(expect_last).margin(1e-3));
    const float expect_edge = Fp8E4M3ToFloat(fp8[128 * C]) * s2;
    CHECK(bf16_at(128, 0) == Catch::Approx(expect_edge).margin(1e-3));
}
