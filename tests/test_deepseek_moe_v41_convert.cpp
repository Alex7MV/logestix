// tests/test_deepseek_moe_v41_convert.cpp — synthetic e2e for the
// DeepSeek-V4.1-Flash converter (logestix-v41-nvfp4): routed experts arrive as
// E2M1-packed I8 + F8_E8M0 per-32 scales and must land as NVFP4 pow2-E4M3
// per-16 pairs with the per-tensor global_scale folded in; dense FP8/norms pass
// through; vision/engram/MTP tensors are excluded with counted records.
#include <catch2/catch_test_macros.hpp>

#include <lsxcommon/converter.h>
#include <lsxcommon/deepseek_moe_v41_convert.h>
#include <lsxcommon/schema.h>

#include <arrow/api.h>

#include "test_synth_checkpoint.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace {

using synthtest::FakeTensor;

// Real-vault E8M0 window (2^-8..2^-5) -> shift m = -2, global_scale 0.25.
constexpr uint8_t kS1[4] = {119, 120, 121, 122};

FakeTensor ExpertWeight(const std::string &name, int rows, int packed_cols,
                        uint8_t fill)
{
    FakeTensor t;
    t.name = name;
    t.dtype = "I8";
    t.shape = {rows, packed_cols};
    t.data.assign(static_cast<size_t>(rows) * packed_cols, fill);
    return t;
}

FakeTensor ExpertScale(const std::string &name, int rows, int cols,
                       const uint8_t *bytes, int n)
{
    FakeTensor t;
    t.name = name;
    t.dtype = "F8_E8M0";
    t.shape = {rows, cols};
    t.data.assign(bytes, bytes + n);
    return t;
}

FakeTensor F8(const std::string &name, std::vector<int64_t> shape)
{
    FakeTensor t;
    t.name = name;
    t.dtype = "F8_E4M3";
    t.shape = std::move(shape);
    int64_t n = 1;
    for (auto d : t.shape) n *= d;
    t.data.assign(static_cast<size_t>(n), 0xAA);
    return t;
}

void WriteCfg(const fs::path &dir)
{
    std::ofstream f(dir / "config.json");
    f << "{\n"
         "  \"model_type\": \"deepseek_v41\",\n"
         "  \"text_config\": {\n"
         "    \"hidden_size\": 64,\n"
         "    \"num_hidden_layers\": 1,\n"
         "    \"vocab_size\": 128,\n"
         "    \"n_routed_experts\": 2,\n"
         "    \"max_position_embeddings\": 256,\n"
         "    \"compress_ratios\": [0],\n"
         "    \"kv_source_layer_ids\": [],\n"
         "    \"index_source_layer_ids\": [],\n"
         "    \"candidate_source_layer_id\": -1\n"
         "  }\n"
         "}\n";
}

fs::path BuildSource(const fs::path &in,
                     const std::vector<FakeTensor> &extra = {})
{
    fs::create_directories(in);
    std::vector<FakeTensor> t;
    // Globals.
    t.push_back({"embed.weight", "BF16", {128, 64}, synthtest::MakeBf16(128 * 64)});
    t.push_back({"head.weight", "BF16", {128, 64}, synthtest::MakeBf16(128 * 64)});
    t.push_back({"norm.weight", "BF16", {64}, synthtest::MakeBf16(64)});
    // Layer 0 dense.
    t.push_back({"layers.0.attn_norm.weight", "BF16", {64},
                 synthtest::MakeBf16(64)});
    t.push_back(F8("layers.0.attn.wq_a.weight", {64, 64}));
    t.push_back(ExpertScale("layers.0.attn.wq_a.scale", 2, 2, kS1, 4));
    t.back().dtype = "F8_E8M0";
    t.push_back({"layers.0.attn.q_norm.weight", "BF16", {64},
                 synthtest::MakeBf16(64)});
    t.push_back({"layers.0.attn.kv_norm.weight", "BF16", {64},
                 synthtest::MakeBf16(64)});
    t.push_back({"layers.0.attn.attn_sink", "F32", {8},
                 synthtest::MakeF32(8)});
    t.push_back({"layers.0.ffn_norm.weight", "BF16", {64},
                 synthtest::MakeBf16(64)});
    t.push_back({"layers.0.ffn.gate.weight", "BF16", {2, 64},
                 synthtest::MakeBf16(128)});
    t.push_back({"layers.0.ffn.gate.bias", "F32", {2}, synthtest::MakeF32(2)});
    t.push_back({"layers.0.ffn.gate.bias_vl", "F32", {2},
                 synthtest::MakeF32(2)});
    t.push_back({"layers.0.hc_attn_fn", "F32", {24, 256},
                 synthtest::MakeF32(24 * 256)});
    t.push_back({"layers.0.hc_attn_base", "F32", {24}, synthtest::MakeF32(24)});
    t.push_back({"layers.0.hc_attn_scale", "F32", {3}, synthtest::MakeF32(3)});
    t.push_back({"layers.0.hc_ffn_fn", "F32", {24, 256},
                 synthtest::MakeF32(24 * 256)});
    t.push_back({"layers.0.hc_ffn_base", "F32", {24}, synthtest::MakeF32(24)});
    t.push_back({"layers.0.hc_ffn_scale", "F32", {3}, synthtest::MakeF32(3)});
    // Routed experts (I8 E2M1 + E8M0/32).
    for (int e = 0; e < 2; ++e) {
        const std::string p = "layers.0.ffn.experts." + std::to_string(e);
        t.push_back(ExpertWeight(p + ".w1.weight", 8, 32,
                                 static_cast<uint8_t>(0xB0 + e)));
        std::vector<uint8_t> s1(16, 0);
        for (int i = 0; i < 16; ++i) s1[i] = kS1[i % 4];
        t.push_back(ExpertScale(p + ".w1.scale", 8, 2, s1.data(), 16));
        t.push_back(ExpertWeight(p + ".w2.weight", 64, 4, 0xC1));
        std::vector<uint8_t> s2(64, 120);
        t.push_back(ExpertScale(p + ".w2.scale", 64, 1, s2.data(), 64));
        t.push_back(ExpertWeight(p + ".w3.weight", 8, 32, 0xC2));
        t.push_back(ExpertScale(p + ".w3.scale", 8, 2, s1.data(), 16));
    }
    // Excluded families (counted).
    t.push_back({"vision.blocks.0.attn.wqkv.weight", "BF16", {4, 4},
                 synthtest::MakeBf16(16)});
    t.push_back({"mtp.0.attn_norm.weight", "BF16", {64},
                 synthtest::MakeBf16(64)});
    t.push_back({"layers.0.engram.q_weight", "BF16", {4, 64},
                 synthtest::MakeBf16(256)});
    for (const auto &x : extra) t.push_back(x);
    synthtest::WriteSafetensors(in, "model-00001-of-00001.safetensors", t);
    WriteCfg(in);
    fs::copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
                  in / "tokenizer.json");
    return in;
}

} // namespace

TEST_CASE("deepseek-v41 converter emits the logestix-v41-nvfp4 layout",
          "[deepseek-v41][convert][pipeline]")
{
    const auto base = fs::temp_directory_path() / "lsx_v41_convert";
    fs::remove_all(base);
    // Named after the card: the converter stamps the input directory's name as
    // the index `model` field (see the sibling case below).
    const auto in = BuildSource(base / "DeepSeek-V4.1-Flash");
    const auto out = base / "converted";

    const auto st =
        lsxcommon::deepseek_v41::ConvertDeepSeekV41Model(in.string(),
                                                         out.string());
    INFO("convert status: " << st.ToString());
    REQUIRE(st.ok());

    {
        std::ifstream f(out / "model.index.json");
        REQUIRE(f.good());
        std::string json((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
        CHECK(json.find("\"logestix-v41-nvfp4\"") != std::string::npos);
        CHECK(json.find("\"DeepSeek-V4.1-Flash\"") != std::string::npos);
        CHECK(json.find("\"num_layers\": 1") != std::string::npos);
        CHECK(json.find("\"total_experts\": 2") != std::string::npos);
        CHECK(json.find("\"engram\": 1") != std::string::npos);
        CHECK(json.find("\"mtp\": 1") != std::string::npos);
        CHECK(json.find("\"vision\": 1") != std::string::npos);
    }

    {
        const auto names = synthtest::TensorNamesIn(
            out / "pinned" / "global.arrow");
        CHECK(std::find(names.begin(), names.end(), "embed.weight") !=
              names.end());
        CHECK(std::find(names.begin(), names.end(), "norm.weight") !=
              names.end());
    }

    // Tokenizer bridge (BPE vocab + merges).
    CHECK(fs::exists(out / "pinned" / "tokenizer_vocab.arrow"));
    CHECK(fs::exists(out / "pinned" / "tokenizer_merges.arrow"));
    {
        const auto names = synthtest::TensorNamesIn(
            out / "pinned" / "layer_000.arrow");
        CHECK(std::find(names.begin(), names.end(),
                        "layers.0.attn.wq_a.weight") != names.end());
        CHECK(std::find(names.begin(), names.end(),
                        "layers.0.ffn.gate.bias_vl") != names.end());
        CHECK(std::find(names.begin(), names.end(),
                        "layers.0.hc_attn_fn") != names.end());
    }

    // Routed expert 0: E8M0 window [119..122] -> m=-2 (global 0.25), pairs.
    {
        auto batch = lsxcommon::ReadRecordBatchFromFile(
            out / "experts" / "layer_0" / "exp_0.arrow");
        REQUIRE(batch.ok());
        auto pe = lsxcommon::ExpertSchema::UnpackBatch(*batch);
        REQUIRE(pe.ok());
        CHECK(pe->w1_global_scale == 0.25f);
        REQUIRE(pe->w1_scales.size() == 32);
        CHECK(pe->w1_scales[0] == (1 << 3));
        CHECK(pe->w1_scales[2] == (2 << 3));
        CHECK(pe->w2_global_scale == 0.5f);
        CHECK(pe->w3_global_scale == 0.25f);
        CHECK(pe->w1_packed[0] == 0xB0);
    }

    fs::remove_all(base);
}

TEST_CASE("deepseek-v41 converter stamps the input card name",
          "[deepseek-v41][convert][pipeline][sibling]")
{
    // The V4.1 contour has two checkpoints of the same architecture
    // (DeepSeek-V4.1-Flash and its 1017 update). The index's `model` field is
    // what `gpu::detect_model_type` normalizes into the LsxModelFactory key, so
    // it must be the CARD name of the tree being converted: a hardcoded parent
    // name makes the sibling tree resolve to the PARENT's kType, which loads
    // the wrong class (and, once the parent's gates are re-seeded, silently
    // compares two models' outputs).
    const auto base = fs::temp_directory_path() / "lsx_v41_convert_card";
    fs::remove_all(base);
    const auto in = BuildSource(base / "DeepSeek-V4.1-Flash-1017");
    const auto out = base / "converted";

    const auto st =
        lsxcommon::deepseek_v41::ConvertDeepSeekV41Model(in.string(),
                                                         out.string());
    INFO("convert status: " << st.ToString());
    REQUIRE(st.ok());

    std::ifstream f(out / "model.index.json");
    REQUIRE(f.good());
    const std::string json((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
    CHECK(json.find("\"model\": \"DeepSeek-V4.1-Flash-1017\"") !=
          std::string::npos);

    fs::remove_all(base);
}

TEST_CASE("deepseek-v41 converter rejects a malformed layer index without "
          "throwing",
          "[deepseek-v41][convert]")
{
    const auto base = fs::temp_directory_path() / "lsx_v41_badidx";
    fs::remove_all(base);
    // A corrupt expert name whose \d+ layer/expert fields exceed int range
    // must be rejected as unclassified, not raise std::out_of_range.
    const auto in = BuildSource(
        base / "model",
        {{"layers.99999999999999999999.ffn.experts.0.w1.weight", "I8", {8, 32},
          std::vector<uint8_t>(8 * 32, 0xB0)}});
    const auto out = base / "converted";

    arrow::Status st;
    CHECK_NOTHROW(st = lsxcommon::deepseek_v41::ConvertDeepSeekV41Model(
                      in.string(), out.string()));
    CHECK_FALSE(st.ok());

    fs::remove_all(base);
}
