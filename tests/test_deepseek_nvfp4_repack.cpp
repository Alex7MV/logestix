// tests/test_deepseek_nvfp4_repack.cpp — host unit tests for the E8M0 ->
// pow2-E4M3 per-16 scale pair primitive (DeepSeek vault NVFP4 repack,
// docs/superpowers/plans/2026-09-11-deepseek-nvfp4-migration.md Task 1).
//
// Real-vault evidence (layer_0/exp_0): scale bytes span [119..122] with
// global_scale = 1.0 — true E8M0 values 2^-8..2^-5. A pow2-E4M3 payload
// interpretation is rejected by the data itself (e4 = b>>3 hits the NaN
// slot 15 for 99.9% of bytes), so the source format is E8M0 per-32 and the
// repack must fold a per-tensor global_scale shift 2^m to stay exact.
#include <catch2/catch_test_macros.hpp>

#include "nvfp4_scale.h"

#include <cstdint>
#include <cmath>

namespace {

// Host decode of the target pow2-E4M3 payload — matches the runtime
// gpu::nvfp4::E4M3ScaleToFloat (nvfp4_codec.h) / E4m3P2ToFloat semantics:
// decode depends on the e4 field only, e4 == 0 decodes to 0.
float Pow2E4m3Decode(uint8_t b)
{
    const uint32_t e4 = (b >> 3) & 0x0Fu;
    if (e4 == 0) return 0.0f;
    const uint32_t bits = (e4 + 120u) << 23;
    float v;
    __builtin_memcpy(&v, &bits, sizeof(v));
    return v;
}

// Host decode of the source E8M0 scale byte — clamped flavor
// (lsx_kernels_quant.cuh E8m0ClampedToFloat): 0 -> 0, else 2^(b-127).
float E8m0Decode(uint8_t b)
{
    if (b == 0) return 0.0f;
    const uint32_t bits = static_cast<uint32_t>(b) << 23;
    float v;
    __builtin_memcpy(&v, &bits, sizeof(v));
    return v;
}

// ldexp-based exact power-of-two scale for the contract check.
float GsFromShift(int m) { return std::ldexp(1.0f, m); }

} // namespace

TEST_CASE("e8m0 scale byte -> pow2 e4m3 per-16 pair (exact with shift)",
          "[fast][deepseek]")
{
    using lsxcommon::nvfp4::E8m0ToE4m3Pow2Pair;
    using lsxcommon::nvfp4::ChooseE8m0ScaleShift;

    // ── Exhaustive contract over the real-vault window and neighbors ──────
    // For every byte b and every shift m where e4 = b-120-m is representable
    // ([1..14]), the pair decodes to EXACTLY the E8M0 value scaled by 2^m.
    for (int m = -40; m <= 40; ++m) {
        for (int b = 0; b < 256; ++b) {
            if (b == 0) continue;  // zero byte covered below
            const uint8_t src = static_cast<uint8_t>(b);
            uint8_t pair[2] = {0xEE, 0xEE};
            E8m0ToE4m3Pow2Pair(src, m, pair);

            // Both halves of the 32-element block carry the same scale.
            REQUIRE(pair[0] == pair[1]);

            const int e4 = b - 120 - m;
            if (e4 >= 1 && e4 <= 14) {
                // Exactness: decoded * 2^m must equal the E8M0 value
                // bit-for-bit (both are pure powers of two).
                const float got = Pow2E4m3Decode(pair[0]) * GsFromShift(m);
                const float want = E8m0Decode(src);
                uint32_t g, w;
                __builtin_memcpy(&g, &got, sizeof(g));
                __builtin_memcpy(&w, &want, sizeof(w));
                INFO("b=" << b << " m=" << m);
                CHECK(g == w);
            } else {
                // Saturation bounds outside the representable window.
                const float got = Pow2E4m3Decode(pair[0]);
                INFO("b=" << b << " m=" << m << " e4=" << e4);
                if (e4 < 1) CHECK(pair[0] == 0x08);   // min scale 2^-6
                else        CHECK(pair[0] == 0x70);   // max scale 2^7
                CHECK(got != 0.0f);
            }
        }
    }

    SECTION("real-vault window [119..122], shift m=-2 maps everything exact") {
        int m = 0;
        REQUIRE(ChooseE8m0ScaleShift(119, 122, m));
        CHECK(m == -2);
        for (uint8_t b = 119; b <= 122; ++b) {
            uint8_t p[2];
            E8m0ToE4m3Pow2Pair(b, m, p);
            const float got = Pow2E4m3Decode(p[0]) * GsFromShift(m);
            const float want = E8m0Decode(b);
            uint32_t g, w;
            __builtin_memcpy(&g, &got, sizeof(g));
            __builtin_memcpy(&w, &want, sizeof(w));
            CHECK(g == w);
        }
    }

    SECTION("zero scale byte stays zero") {
        uint8_t p[2] = {0xEE, 0xEE};
        E8m0ToE4m3Pow2Pair(0x00, -2, p);
        CHECK(p[0] == 0x00);
        CHECK(p[1] == 0x00);
    }

    SECTION("shift chooser rejects over-wide spreads") {
        int m = 12345;
        CHECK_FALSE(ChooseE8m0ScaleShift(100, 114, m));  // spread 14
        CHECK(ChooseE8m0ScaleShift(100, 113, m) == true);  // spread 13 OK
        CHECK(m == 100 - 121);
        CHECK_FALSE(ChooseE8m0ScaleShift(0, 5, m));  // zero byte: no range
    }

    SECTION("high E8M0 bytes shift down exactly (no upper clamp needed)") {
        // b = 200 (2^73) with m = 200-121 = 79: e4 = 1, gs = 2^79.
        int m = 0;
        REQUIRE(ChooseE8m0ScaleShift(200, 200, m));
        CHECK(m == 79);
        uint8_t p[2];
        E8m0ToE4m3Pow2Pair(200, m, p);
        CHECK(p[0] == 0x08);
        const float got = Pow2E4m3Decode(p[0]) * GsFromShift(m);
        const float want = E8m0Decode(200);
        uint32_t g, w;
        __builtin_memcpy(&g, &got, sizeof(g));
        __builtin_memcpy(&w, &want, sizeof(w));
        CHECK(g == w);
    }
}
