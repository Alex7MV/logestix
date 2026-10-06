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

#include "lsxcommon/deepseek_moe_repack.h"
#include "lsxcommon/schema.h"
#include "nvfp4_scale.h"

#include <cstdint>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iterator>
#include <filesystem>

namespace {

// Host decode of the target pow2-E4M3 payload — matches the runtime
// gpu::nvfp4::E4M3ScaleToFloat (nvfp4_codec.h) / E4m3ToFloat semantics:
// decode depends on the e4 field only, e4 == 0 decodes to 0.
float Pow2E4m3Decode(uint8_t b)
{
    const uint32_t e4 = (b >> 3) & 0x0Fu;
    if (e4 == 0) return 0.0f;
    const uint32_t bits = (e4 + 120u) << 23;
    float v;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

// Host decode of the source E8M0 scale byte — clamped flavor
// (lsx_kernels_quant.cuh E8m0ClampedToFloat): 0 -> 0, else 2^(b-127).
float E8m0Decode(uint8_t b)
{
    if (b == 0) return 0.0f;
    const uint32_t bits = static_cast<uint32_t>(b) << 23;
    float v;
    std::memcpy(&v, &bits, sizeof(v));
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
                std::memcpy(&g, &got, sizeof(g));
                std::memcpy(&w, &want, sizeof(w));
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
            std::memcpy(&g, &got, sizeof(g));
            std::memcpy(&w, &want, sizeof(w));
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
        std::memcpy(&g, &got, sizeof(g));
        std::memcpy(&w, &want, sizeof(w));
        CHECK(g == w);
    }
}

TEST_CASE("deepseek nvfp4 repack tool: micro-vault e2e", "[tool][deepseek]")
{
    using lsxcommon::PackedExpert;
    namespace fs = std::filesystem;

    const fs::path root =
        fs::temp_directory_path() /
        ("lsx_ds_repack_" + std::to_string(::getpid()));
    const fs::path src = root / "in";
    const fs::path dst = root / "out";
    fs::create_directories(src / "logestix" / "experts" / "layer_0");
    fs::create_directories(src / "logestix" / "pinned");

    // Minimal v1 index (num_layers/experts_per_layer drive the tool walk).
    {
        std::ofstream f(src / "logestix" / "model.index.json");
        f << "{\"format\": \"logestix-v1\", \"model\": \"DeepSeek-V4-Flash\","
             " \"num_layers\": 1, \"experts_per_layer\": 2, \"files\": {}}\n";
    }
    // Verbatim-carry probe file.
    {
        std::ofstream f(src / "logestix" / "pinned" / "dummy.bin");
        f << "pinned-payload";
    }

    // Expert fixture: real-vault window [119..122] + a zero byte; distinct
    // packed payload per expert so verbatim-carry is provable.
    auto make_expert = [](uint8_t packed_seed) {
        PackedExpert e;
        e.w1_packed = {packed_seed, 0x11, 0x22, 0x33};
        e.w2_packed = {packed_seed, 0x44, 0x55, 0x66};
        e.w3_packed = {packed_seed, 0x77, 0x88, 0x99};
        e.w1_scales = {119, 120, 121, 122, 0x00};
        e.w2_scales = {120, 121};
        e.w3_scales = {122};
        e.w1_global_scale = 1.0f;
        e.w2_global_scale = 1.0f;
        e.w3_global_scale = 1.0f;
        return e;
    };
    for (int exp = 0; exp < 2; ++exp) {
        auto batch = lsxcommon::ExpertSchema::MakeBatch(make_expert(
            static_cast<uint8_t>(exp ? 0xA0 : 0xB0)));
        REQUIRE(batch.ok());
        auto w = lsxcommon::WriteRecordBatchToFile(
            *batch, src / "logestix" / "experts" / "layer_0" /
                        ("exp_" + std::to_string(exp) + ".arrow"));
        REQUIRE(w.ok());
    }

    REQUIRE(lsxcommon::deepseek::RepackDeepseekExpertsNvfp4(
                src.string(), dst.string())
                .ok());

    // Index format swapped, payload carried.
    {
        std::ifstream f(dst / "logestix" / "model.index.json");
        std::string json((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
        CHECK(json.find("\"logestix-v1-nvfp4\"") != std::string::npos);
        CHECK(json.find("logestix-v1\"") == std::string::npos);
    }
    {
        std::ifstream f(dst / "logestix" / "pinned" / "dummy.bin");
        std::string payload((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());
        CHECK(payload == "pinned-payload");
    }

    // Experts: packed verbatim, scales doubled + mapped, gs = 2^m.
    for (int exp = 0; exp < 2; ++exp) {
        const uint8_t packed_seed = exp ? 0xA0 : 0xB0;
        auto batch = lsxcommon::ReadRecordBatchFromFile(
            dst / "logestix" / "experts" / "layer_0" /
                ("exp_" + std::to_string(exp) + ".arrow"));
        REQUIRE(batch.ok());
        auto pe = lsxcommon::ExpertSchema::UnpackBatch(*batch);
        REQUIRE(pe.ok());
        const PackedExpert &e = *pe;
        CHECK(e.w1_packed == std::vector<uint8_t>(
                                  {packed_seed, 0x11, 0x22, 0x33}));
        // w1: bytes [119..122] exact with m = -2 (gs = 0.25); the zero
        // byte stays zero (per-16 byte 0x00).
        CHECK(e.w1_global_scale == 0.25f);
        REQUIRE(e.w1_scales.size() == 10);
        CHECK(e.w1_scales[0] == (1 << 3));   // 119 -> e4=1
        CHECK(e.w1_scales[1] == (1 << 3));
        CHECK(e.w1_scales[2] == (2 << 3));   // 120 -> e4=2
        CHECK(e.w1_scales[3] == (2 << 3));
        CHECK(e.w1_scales[4] == (3 << 3));   // 121 -> e4=3
        CHECK(e.w1_scales[5] == (3 << 3));
        CHECK(e.w1_scales[6] == (4 << 3));   // 122 -> e4=4
        CHECK(e.w1_scales[7] == (4 << 3));
        CHECK(e.w1_scales[8] == 0x00);       // zero byte
        CHECK(e.w1_scales[9] == 0x00);
        // w2: window [120..121] -> m = -1 (gs = 0.5), b-119.
        CHECK(e.w2_global_scale == 0.5f);
        REQUIRE(e.w2_scales.size() == 4);
        CHECK(e.w2_scales[0] == (1 << 3));   // 120 -> e4=1
        CHECK(e.w2_scales[1] == (1 << 3));
        CHECK(e.w2_scales[2] == (2 << 3));   // 121 -> e4=2
        CHECK(e.w2_scales[3] == (2 << 3));
        // w3: single byte 122 -> m = 1 (gs = 2.0), b-121 -> e4=1.
        CHECK(e.w3_global_scale == 2.0f);
        REQUIRE(e.w3_scales.size() == 2);
        CHECK(e.w3_scales[0] == (1 << 3));
        CHECK(e.w3_scales[1] == (1 << 3));
    }

    fs::remove_all(root);
}
