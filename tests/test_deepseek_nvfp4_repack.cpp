// tests/test_deepseek_nvfp4_repack.cpp — host unit tests for the E8M0 ->
// pow2-E4M3 per-16 scale pair primitive (DeepSeek vault NVFP4 repack,
// docs/superpowers/plans/2026-09-11-deepseek-nvfp4-migration.md Task 1).
#include <catch2/catch_test_macros.hpp>

#include "nvfp4_scale.h"

#include <cstdint>

namespace {

// Host decode of the target pow2-E4M3 payload — must match the runtime
// gpu::nvfp4::E4M3ScaleToFloat (nvfp4_codec.h) and E4m3P2ToFloat semantics:
// decode depends on the e4 field only, byte 0x00 (e4==0) decodes to 0.
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
// (lsx_kernels_quant.cuh E8m0ClampedToFloat): 0 -> 0, 0xFF -> FLT_MAX,
// otherwise 2^(b-127).
float E8m0Decode(uint8_t b)
{
    if (b == 0) return 0.0f;
    if (b == 0xFF) return 3.402823466e+38f;
    const uint32_t bits = static_cast<uint32_t>(b) << 23;
    float v;
    __builtin_memcpy(&v, &bits, sizeof(v));
    return v;
}

} // namespace

TEST_CASE("e8m0 scale byte -> pow2 e4m3 per-16 pair", "[fast][deepseek]")
{
    using lsxcommon::nvfp4::E8m0ToE4m3Pow2Pair;

    // Exhaustive 256-byte contract.
    for (int b = 0; b < 256; ++b) {
        const uint8_t src = static_cast<uint8_t>(b);
        uint8_t pair[2] = {0xEE, 0xEE};
        E8m0ToE4m3Pow2Pair(src, pair);

        // Both halves of the 32-element block must carry the same scale.
        REQUIRE(pair[0] == pair[1]);

        if (src == 0x00) {
            CHECK(pair[0] == 0x00);  // zero scale stays zero (decode: 0.0f)
            continue;
        }
        // Decoded target must be a nonzero power of two. Representable
        // range maps bit-exactly; outside it saturates to the nearest
        // bound (finite-clamp flavor — see the sections below).
        const float got = Pow2E4m3Decode(pair[0]);
        const float want = E8m0Decode(src);
        CHECK(got != 0.0f);
        CHECK(got <= 128.0f);
        CHECK(got >= 0.015625f);  // 2^-6 min scale
        // Result is exactly 2^k: bit pattern has mantissa 0.
        uint32_t bits;
        __builtin_memcpy(&bits, &got, sizeof(bits));
        CHECK((bits & 0x007FFFFFu) == 0);
    }

    SECTION("exact representable range b in [121..134] maps onto itself") {
        for (int b = 121; b <= 134; ++b) {
            uint8_t pair[2];
            lsxcommon::nvfp4::E8m0ToE4m3Pow2Pair(static_cast<uint8_t>(b), pair);
            const float got = Pow2E4m3Decode(pair[0]);
            const float want = E8m0Decode(static_cast<uint8_t>(b));
            // Bit-exact (both are pure powers of two).
            uint32_t g, w;
            __builtin_memcpy(&g, &got, sizeof(g));
            __builtin_memcpy(&w, &want, sizeof(w));
            CHECK(g == w);
        }
    }

    SECTION("below-range saturates to min scale 2^-6") {
        uint8_t p[2];
        lsxcommon::nvfp4::E8m0ToE4m3Pow2Pair(120, p);
        CHECK(p[0] == 0x08);  // e4=1 -> 2^-6
    }

    SECTION("above-range saturates to max scale 2^7") {
        uint8_t p[2];
        lsxcommon::nvfp4::E8m0ToE4m3Pow2Pair(135, p);
        CHECK(p[0] == 0x70);  // e4=14 -> 2^7
        lsxcommon::nvfp4::E8m0ToE4m3Pow2Pair(0xFF, p);
        CHECK(p[0] == 0x70);  // FLT_MAX flavor clamps finite
    }
}
