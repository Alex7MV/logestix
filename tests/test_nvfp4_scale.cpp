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
#include <filesystem>
#include <fstream>
#include <iterator>
#include <regex>
#include <set>
#include <string>
#include <vector>

#ifndef LSX_SOURCE_DIR
#define LSX_SOURCE_DIR "."
#endif

namespace fs = std::filesystem;

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

// ── Single-source guard: no test re-implements the block-scale decode ───────
//
// The block-scale codec is single-source (nvfp4_scale.h) precisely because the
// value table is subtle — a mantissa bit dropped, or `e == 0` mapped to zero
// instead of the subnormal m*2^-9, is a silent numerical change in whichever
// test happens to carry a stale copy. The E4M3FN alignment had exactly that:
// a dozen host references in tests/ still carried the retired field-zero /
// pow2-only contract while the runtime had moved on.
//
// A test that genuinely needs an INDEPENDENT oracle (this file's RefDecode,
// the nemotron converter's reference) is not a violation — independence is the
// point there. What is forbidden is a second copy that claims to mirror the
// runtime decoder, because that is what silently diverges. The discriminator
// used here is the signed/unsigned split: the unsigned block-scale type has NO
// sign bit, so a replica that extracts one is decoding a different format
// (signed F8_E4M3) and is legitimately separate.
TEST_CASE("no test re-implements the unsigned NVFP4 block-scale decode",
          "[nvfp4][codec][guard]")
{
    // Files whose `(b >> 3) & 0xF` exponent extraction decodes a SIGNED
    // F8_E4M3 tensor (they extract `(b >> 7) & 1`), or is an intentional
    // independent oracle for the codec itself.
    static const std::set<std::string> kAllowed = {
        // Signed F8_E4M3 weight tensors — a different type from the unsigned
        // UE4M3 block scale, with a real sign bit.
        "test_deepseek_moe_moe.cu",
        "test_deepseek_moe_v41_dequant.cu",
        "test_gigachat_moe_kernels.cpp",
        "test_mimo_moe_dequant.cpp",
        // Deliberately independent oracles for the codec under test.
        "test_nvfp4_scale.cpp",
        "test_nemotron_moe_nvfp4_convert.cpp",
        // pow2-only payload by construction: E8m0ToE4m3Pow2Pair emits
        // (e4 << 3), so the mantissa is always 0 and the pow2 and full-E4M3
        // decodes agree on every byte this file can see.
        "test_deepseek_moe_moe.cu",
        "test_deepseek_nvfp4_repack.cpp",
    };

    const fs::path dir(LSX_SOURCE_DIR "/tests");
    std::vector<std::string> offenders;
    for (const auto &e : fs::directory_iterator(dir)) {
        if (e.path().extension() != ".cu" && e.path().extension() != ".cpp")
            continue;
        const std::string name = e.path().filename().string();
        if (kAllowed.count(name)) continue;

        std::ifstream in(e.path());
        if (!in.good()) continue;
        const std::string text((std::istreambuf_iterator<char>(in)),
                               std::istreambuf_iterator<char>());

        // Extracts an E4M3 exponent field...
        const bool extracts_exp =
            std::regex_search(text, std::regex(R"(\(\s*\w+\s*>>\s*3\s*\)\s*&\s*0x0?F\s*)"));
        if (!extracts_exp) continue;
        // ...but is NOT calling the canonical codec (either the public
        // lsxcommon::e4m3_to_float wrapper or nvfp4_scale.h's E4m3ToFloat),
        // and takes no sign bit (so it is not a signed F8_E4M3 replica either).
        const bool canonical = text.find("E4m3ToFloat") != std::string::npos ||
                              text.find("e4m3_to_float") != std::string::npos;
        const bool signed_fp8 =
            std::regex_search(text, std::regex(R"(\(\s*\w+\s*>>\s*7\s*\)\s*&\s*1\s*)"));
        if (canonical || signed_fp8) continue;

        offenders.push_back(name);
    }

    for (const auto &x : offenders)
        WARN("stale local E4M3 block-scale decode in tests/" << x
              << " - call lsxcommon::nvfp4::E4m3ToFloat instead");
    CHECK(offenders.empty());
}
