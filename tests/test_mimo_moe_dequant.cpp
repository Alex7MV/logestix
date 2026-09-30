// TDD — MiMo converter dequant parity: FP8 block-128 (incl. the GA spare
// scale row quirk) and MXFP4 -> BF16 against a host scalar reference.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "lsxcommon/mimo_moe_convert.h"

namespace {

float HostE4m3(uint8_t b)
{
    const int s = (b >> 7) & 1;
    const int e = (b >> 3) & 0xF;
    const int m = b & 0x7;
    float v;
    if (e == 0) v = std::ldexp(static_cast<float>(m) / 8.0f, -6);
    else v = std::ldexp(1.0f + static_cast<float>(m) / 8.0f, e - 7);
    return s ? -v : v;
}

float HostE2m1(int nib)
{
    static const float kMag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const float v = kMag[nib & 7];
    return (nib & 8) ? -v : v;
}

float HostE8m0(uint8_t b) { return b == 0 ? 0.0f : std::ldexp(1.0f, b - 127); }

float Bf16At(const std::vector<uint8_t> &v, size_t i)
{
    uint16_t bits = static_cast<uint16_t>(v[2 * i] | (v[2 * i + 1] << 8));
    uint32_t f = static_cast<uint32_t>(bits) << 16;
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

} // namespace

TEST_CASE("mimo FP8 block-128 dequant parity (with spare scale rows)",
          "[mimo][dequant]")
{
    const int rows = 5, cols = 16;
    const int sr = 2, sc = 1;  // sr*128 > rows: spare scale row must be ignored
    std::vector<uint8_t> fp8(rows * cols);
    for (size_t i = 0; i < fp8.size(); ++i)
        fp8[i] = static_cast<uint8_t>((0x20 + i) & 0x7F);
    std::vector<uint8_t> scale(sr * sc * 4);
    float s0 = 1.5f, s1 = 2.0f;
    std::memcpy(scale.data(), &s0, 4);
    std::memcpy(scale.data() + 4, &s1, 4);

    std::vector<uint8_t> out;
    REQUIRE(lsxcommon::mimo::DequantFp8Block128ToBf16Gpu(
                fp8, {rows, cols}, scale, {sr, sc}, out).ok());
    REQUIRE(out.size() == static_cast<size_t>(rows * cols) * 2);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
        {
            const int tile = (r / 128) * sc + (c / 128);
            const float exp = HostE4m3(fp8[r * cols + c]) * (tile == 0 ? s0 : s1);
            REQUIRE(std::fabs(Bf16At(out, r * cols + c) - exp) < 2e-2f);
        }
}

TEST_CASE("mimo MXFP4 dequant parity", "[mimo][dequant]")
{
    const int rows = 3, pc = 16;   // packed cols; logical cols = 32
    const int lc = pc * 2;
    std::vector<uint8_t> packed(rows * pc);
    for (size_t i = 0; i < packed.size(); ++i)
        packed[i] = static_cast<uint8_t>(0x21 + i);
    std::vector<uint8_t> e8m0(rows, 126);  // one 32-elem block per row

    std::vector<uint8_t> out;
    REQUIRE(lsxcommon::mimo::DequantMxfp4ToBf16Gpu(
                packed, {rows, pc}, e8m0, {rows, 1}, out).ok());
    REQUIRE(out.size() == static_cast<size_t>(rows * lc) * 2);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < lc; ++c)
        {
            const int li = r * lc + c;
            const uint8_t byte = packed[li >> 1];
            const int nib = (li & 1) ? (byte >> 4) : (byte & 0xF);
            const float exp = HostE2m1(nib) * HostE8m0(126);
            REQUIRE(std::fabs(Bf16At(out, li) - exp) < 2e-2f);
        }
}
