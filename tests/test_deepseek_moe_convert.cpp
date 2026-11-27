// tests/test_deepseek_moe_convert.cpp — synthetic e2e for the DeepSeek
// converter pipeline (logestix-v1-nvfp4, migration plan Task 3):
// routed experts arrive as E2M1-packed I8 + F8_E8M0 per-32 scales and must
// land as NVFP4 pow2-E4M3 per-16 pairs with the per-tensor global_scale
// folded in; dense/pinned tensors and the index pass through.
#include <catch2/catch_test_macros.hpp>

#include <lsxcommon/converter.h>
#include <lsxcommon/schema.h>

#include <arrow/api.h>

#include "test_synth_checkpoint.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

namespace fs = std::filesystem;
namespace {

using synthtest::FakeTensor;

// The real-vault E8M0 window (2^-8..2^-5, gs folded per tensor).
constexpr uint8_t kS1[4] = {119, 120, 121, 122};

FakeTensor MakeExpertWeight(const std::string &name, int rows,
                            int packed_cols, uint8_t fill)
{
    FakeTensor t;
    t.name = name;
    t.dtype = "I8";
    t.shape = {rows, packed_cols};
    t.data.assign(static_cast<size_t>(rows) * packed_cols, fill);
    return t;
}

FakeTensor MakeExpertScale(const std::string &name, int rows,
                           const uint8_t *bytes, int n)
{
    FakeTensor t;
    t.name = name;
    t.dtype = "F8_E8M0";
    t.shape = {rows, 1};
    t.data.assign(bytes, bytes + n);
    return t;
}

fs::path BuildDeepseekSource(const fs::path &in)
{
    fs::create_directories(in);
    std::vector<FakeTensor> tensors;
    // Globals (BF16 pass-through).
    tensors.push_back({"embed.weight", "BF16", {128, 64},
                       synthtest::MakeBf16(128 * 64)});
    tensors.push_back({"head.weight", "BF16", {128, 64},
                       synthtest::MakeBf16(128 * 64)});
    tensors.push_back({"norm.weight", "BF16", {64},
                       synthtest::MakeBf16(64)});
    // Attention FP8 + E8M0 tile scale (pass-through).
    tensors.push_back({"layers.0.attn.wq_a.weight", "F8_E4M3", {16, 64},
                       std::vector<uint8_t>(16 * 64, 0xAA)});
    const uint8_t tile = 120;
    tensors.push_back({"layers.0.attn.wq_a.scale", "F8_E8M0", {1, 1},
                       std::vector<uint8_t>{tile}});
    // Router (BF16).
    tensors.push_back({"layers.0.ffn.gate.weight", "BF16", {2, 64},
                       synthtest::MakeBf16(2 * 64)});
    // Two routed experts (packed E2M1 I8 + E8M0 per-32 scales).
    for (int e = 0; e < 2; ++e) {
        const std::string p = "layers.0.ffn.experts." + std::to_string(e);
        tensors.push_back(MakeExpertWeight(
            p + ".w1.weight", 4, 16, static_cast<uint8_t>(0xB0 + e)));
        // Window [119..122] (+ a zero block for expert 1) -> m = -2.
        std::vector<uint8_t> s1(kS1, kS1 + 4);
        int s1_rows = 4;
        if (e == 1) {
            s1.push_back(0x00);
            s1_rows = 5;
        }
        tensors.push_back(MakeExpertScale(p + ".w1.scale", s1_rows,
                                          s1.data(), (int)s1.size()));
        tensors.push_back(MakeExpertWeight(p + ".w2.weight", 8, 8, 0xC1));
        const uint8_t s2[8] = {120, 120, 120, 120, 120, 120, 120, 120};
        tensors.push_back(MakeExpertScale(p + ".w2.scale", 8, s2, 8));
        tensors.push_back(MakeExpertWeight(p + ".w3.weight", 4, 16, 0xC2));
        tensors.push_back(MakeExpertScale(p + ".w3.scale", 4, kS1, 4));
    }
    synthtest::WriteSafetensors(in, "model-00001-of-00001.safetensors",
                                tensors);

    std::ofstream cfg(in / "config.json");
    cfg << "{\"num_hidden_layers\": 1, \"hidden_size\": 64,"
           " \"vocab_size\": 128, \"num_local_experts\": 2}\n";
    fs::copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
                  in / "tokenizer.json");
    return in;
}

} // namespace

TEST_CASE("deepseek converter emits nvfp4 experts from e8m0 source",
          "[deepseek][convert][pipeline]")
{
    namespace fs = std::filesystem;
    const auto base =
        fs::temp_directory_path() / "logestix_ds_convert_pipeline";
    fs::remove_all(base);
    const auto in = BuildDeepseekSource(base / "model");
    const auto out = base / "converted";

    lsxcommon::Converter conv;
    conv.SetInputPath(in.string()).SetOutputPath(out.string()).SetVerbose(true);
    const auto st = conv.Run();
    INFO("converter status: " << st.ToString());
    REQUIRE(st.ok());

    // Index: NVFP4 format tag.
    {
        std::ifstream f(out / "model.index.json");
        REQUIRE(f.good());
        std::string json((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
        CHECK(json.find("\"logestix-v1-nvfp4\"") != std::string::npos);
    }

    // Expert 0: w1 window [119..122] -> m=-2 (gs 0.25), scales doubled.
    {
        auto batch = lsxcommon::ReadRecordBatchFromFile(
            out / "experts" / "layer_0" / "exp_0.arrow");
        REQUIRE(batch.ok());
        auto pe = lsxcommon::ExpertSchema::UnpackBatch(*batch);
        REQUIRE(pe.ok());
        CHECK(pe->w1_global_scale == 0.25f);
        REQUIRE(pe->w1_scales.size() == 8);
        CHECK(pe->w1_scales[0] == (1 << 3));   // 119 -> e4=1
        CHECK(pe->w1_scales[1] == (1 << 3));
        CHECK(pe->w1_scales[2] == (2 << 3));   // 120 -> e4=2
        CHECK(pe->w1_scales[3] == (2 << 3));
        CHECK(pe->w1_scales[4] == (3 << 3));   // 121 -> e4=3
        CHECK(pe->w1_scales[5] == (3 << 3));
        CHECK(pe->w1_scales[6] == (4 << 3));   // 122 -> e4=4
        CHECK(pe->w1_scales[7] == (4 << 3));
        // Single-byte window {120} -> m=-1 (gs 0.5), e4=1.
        CHECK(pe->w2_global_scale == 0.5f);
        REQUIRE(pe->w2_scales.size() == 16);
        CHECK(pe->w2_scales[0] == (1 << 3));
        CHECK(pe->w2_scales[1] == (1 << 3));
        // Packed nibbles verbatim.
        CHECK(pe->w1_packed[0] == 0xB0);
    }

    // Expert 1: w1 has the extra zero block (passes through as zero pairs).
    {
        auto batch = lsxcommon::ReadRecordBatchFromFile(
            out / "experts" / "layer_0" / "exp_1.arrow");
        REQUIRE(batch.ok());
        auto pe = lsxcommon::ExpertSchema::UnpackBatch(*batch);
        REQUIRE(pe.ok());
        REQUIRE(pe->w1_scales.size() == 10);
        CHECK(pe->w1_scales[8] == 0x00);
        CHECK(pe->w1_scales[9] == 0x00);
        CHECK(pe->w1_global_scale == 0.25f);
    }

    fs::remove_all(base);
}
