// TDD — NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 converter
// (Model Ingestion Protocol stages 1-2). Synthetic PRE-QUANTIZED checkpoint
// only: the real model is never converted in a test, and the ctest TIMEOUT is
// the mechanical guard against that ever happening.
//
// The load-bearing assertion of this suite is the codec one: the vault's NVFP4
// block scales use the FULL E4M3FN value table (mantissa bits set, and
// subnormal bytes on the tiny blocks). The repo's canonical decoder now
// implements exactly that table, so the test compares the converter's
// dequantized-then-requantized output against a host reference using the same
// complete E4M3FN decode; the residual is the requantization noise.
#include <catch2/catch_test_macros.hpp>

#include "test_nemotron_moe_nvfp4_fixture.h"

#include "lsxcommon/nemotron_moe_nvfp4_convert.h"
#include "lsxcommon/safetensors.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace lsxcommon::nemotron_nvfp4;

namespace {

std::string ReadFile(const fs::path &p)
{
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Complete E4M3FN decode (mantissa bits AND subnormals) - the host mirror of
// the converter's dequant kernel. It matches the repo's canonical
// E4M3ScaleToFloat over the valid scale domain; kept local so the test does not
// depend on the CUDA header.
float E4m3FullToFloat(uint8_t b)
{
    const float sign = (b >> 7) ? -1.0f : 1.0f;
    const int e = (b >> 3) & 0xF;
    const int m = b & 0x7;
    if (e == 0) return sign * m * std::pow(2.0f, -9.0f);
    return sign * (1.0f + m / 8.0f) * std::pow(2.0f, static_cast<float>(e - 7));
}

float E2m1Value(int nib)
{
    static const float kLv[8] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
    const float v = kLv[nib & 7];
    return (nib & 8) ? -v : v;
}

// Dequantize one source NVFP4 tensor triple to the logical [R, C] matrix.
std::vector<float> DequantSourceNvfp4(const std::vector<uint8_t> &packed,
                                      const std::vector<uint8_t> &scale,
                                      float global, int64_t R, int64_t C)
{
    std::vector<float> out(static_cast<size_t>(R * C));
    for (int64_t i = 0; i < R * C; ++i) {
        const uint8_t byte = packed[static_cast<size_t>(i / 2)];
        const int nib = (i & 1) ? (byte >> 4) : (byte & 0xF);
        out[static_cast<size_t>(i)] =
            E2m1Value(nib) * E4m3FullToFloat(scale[static_cast<size_t>(i / 16)]) * global;
    }
    return out;
}

} // namespace

TEST_CASE("nvfp4 converter classifies the pre-quantized source layout",
          "[nemotron][nvfp4][convert]")
{
    // Every companion scale tensor of the vault must classify - never Unknown,
    // because Unknown is a hard converter error (a leftover tensor must fail
    // the conversion, not pass).
    struct { const char *name; NemotronNvfp4Kind kind; } kExpect[] = {
        {"backbone.embeddings.weight", NemotronNvfp4Kind::Embed},
        {"lm_head.weight", NemotronNvfp4Kind::LmHead},
        {"lm_head.weight_scale", NemotronNvfp4Kind::Nvfp4BlockScale},
        {"lm_head.weight_scale_2", NemotronNvfp4Kind::Nvfp4GlobalScale},
        {"backbone.norm_f.weight", NemotronNvfp4Kind::FinalNorm},
        {"backbone.layers.0.norm.weight", NemotronNvfp4Kind::LayerNorm},
        {"backbone.layers.0.mixer.in_proj.weight", NemotronNvfp4Kind::MambaInProj},
        {"backbone.layers.0.mixer.in_proj.weight_scale",
         NemotronNvfp4Kind::Fp8WeightScale},
        {"backbone.layers.0.mixer.in_proj.input_scale",
         NemotronNvfp4Kind::Fp8InputScale},
        {"backbone.layers.0.mixer.out_proj.weight", NemotronNvfp4Kind::MambaOutProj},
        {"backbone.layers.0.mixer.conv1d.weight", NemotronNvfp4Kind::MambaConv},
        {"backbone.layers.0.mixer.conv1d.bias", NemotronNvfp4Kind::MambaConvBias},
        {"backbone.layers.0.mixer.A_log", NemotronNvfp4Kind::MambaALog},
        {"backbone.layers.0.mixer.D", NemotronNvfp4Kind::MambaD},
        {"backbone.layers.0.mixer.dt_bias", NemotronNvfp4Kind::MambaDtBias},
        {"backbone.layers.0.mixer.norm.weight", NemotronNvfp4Kind::MambaNorm},
        {"backbone.layers.1.mixer.gate.weight", NemotronNvfp4Kind::RouterGate},
        {"backbone.layers.1.mixer.gate.e_score_correction_bias",
         NemotronNvfp4Kind::RouterBias},
        {"backbone.layers.1.mixer.shared_experts.up_proj.weight",
         NemotronNvfp4Kind::SharedUpProj},
        {"backbone.layers.1.mixer.shared_experts.up_proj.weight_scale",
         NemotronNvfp4Kind::Nvfp4BlockScale},
        {"backbone.layers.1.mixer.experts.3.down_proj.weight",
         NemotronNvfp4Kind::ExpertDownProj},
        {"backbone.layers.1.mixer.experts.3.down_proj.weight_scale_2",
         NemotronNvfp4Kind::Nvfp4GlobalScale},
        {"backbone.layers.2.mixer.q_proj.weight", NemotronNvfp4Kind::AttnQ},
        {"backbone.layers.2.mixer.k_proj.k_scale", NemotronNvfp4Kind::Fp8KvScale},
        {"backbone.layers.2.mixer.v_proj.v_scale", NemotronNvfp4Kind::Fp8KvScale},
        {"mtp.layers.0.mixer.q_proj.weight", NemotronNvfp4Kind::Mtp},
    };
    for (const auto &e : kExpect) {
        INFO("tensor " << e.name);
        CHECK(classify_nemotron_nvfp4_tensor(e.name).kind == e.kind);
    }

    // Expert index and layer index are carried, not just the kind.
    const auto ex = classify_nemotron_nvfp4_tensor(
        "backbone.layers.1.mixer.experts.3.down_proj.weight");
    CHECK(ex.layer == 1);
    CHECK(ex.expert == 3);

    // Malformed indices stay Unknown and never throw (the parent's contract).
    CHECK_NOTHROW(classify_nemotron_nvfp4_tensor("backbone.layers..mixer.norm.weight"));
    CHECK(classify_nemotron_nvfp4_tensor("backbone.layers..mixer.norm.weight").kind ==
          NemotronNvfp4Kind::Unknown);
    CHECK_NOTHROW(classify_nemotron_nvfp4_tensor(
        "backbone.layers.0.mixer.experts..up_proj.weight"));
    CHECK(classify_nemotron_nvfp4_tensor(
              "backbone.layers.0.mixer.experts..up_proj.weight").kind ==
          NemotronNvfp4Kind::Unknown);
    // A genuinely unknown tensor is still Unknown (the converter errors on it).
    CHECK(classify_nemotron_nvfp4_tensor("backbone.layers.0.mixer.mystery.weight").kind ==
          NemotronNvfp4Kind::Unknown);
}

TEST_CASE("nvfp4 converter reads the real vault's eos list from generation_config",
          "[nemotron][nvfp4][convert]")
{
    // The vault's generation_config.json declares eos_token_id [2, 11]: id 11
    // is a real second stop. A converter that only ever read config.json's
    // scalar eos_token_id would silently lose it.
    const auto src = nemotronv4test::MakeTempDir("nmv4_gen");
    nemotronv4test::WriteNemotronNvfp4Checkpoint(src);
    {
        std::ofstream f(src / "generation_config.json");
        f << R"({"bos_token_id": 1, "eos_token_id": [2, 11], "pad_token_id": 0})";
    }
    const auto dst = nemotronv4test::MakeTempDir("nmv4_gen_dst");
    REQUIRE(ConvertNemotronNvfp4Model(src.string(), dst.string()).ok());
    const std::string idx = ReadFile(dst / "logestix" / "model.index.json");
    CHECK(idx.find("\"eos_token_id\": [2, 11]") != std::string::npos);
    CHECK(idx.find("\"bos_token_id\": 1") != std::string::npos);
    CHECK(idx.find("\"pad_token_id\": 0") != std::string::npos);
}

TEST_CASE("nvfp4 converter emits the logestix-nemotron-nvfp4 layout",
          "[nemotron][nvfp4][convert]")
{
    const auto src = nemotronv4test::MakeTempDir("nmv4_src");
    const auto dst = nemotronv4test::MakeTempDir("nmv4_dst");
    nemotronv4test::WriteNemotronNvfp4Checkpoint(src);

    auto st = ConvertNemotronNvfp4Model(src.string(), dst.string());
    INFO(st.ToString());
    REQUIRE(st.ok());

    const std::string idx = ReadFile(dst / "logestix" / "model.index.json");
    CHECK(idx.find("\"format\": \"logestix-nemotron-nvfp4\"") != std::string::npos);
    CHECK(idx.find("\"attn_layer_idx\": [2]") != std::string::npos);
    CHECK(idx.find("\"experts_no_gate\": true") != std::string::npos);
    // MTP is a COUNTED cut: 3 tensors in the fixture, recorded, never applied.
    CHECK(idx.find("\"deferred\": {\"mtp\": \"cut\"") != std::string::npos);
    CHECK(idx.find("\"mtp_tensors\": 3") != std::string::npos);
    // The FP8 KV-cache scales are dropped, not stored as weights.
    CHECK(idx.find("\"fp8_kv_scales_dropped\": 2") != std::string::npos);

    // Per-expert arrow files for the single MoE layer (layer 1), w1 unused.
    for (int e = 0; e < nemotronv4test::kTExperts; ++e)
        CHECK(fs::exists(dst / "logestix" / "experts" / "layer_1" /
                         ("exp_" + std::to_string(e) + ".arrow")));
    CHECK(fs::exists(dst / "logestix" / "pinned" / "global.arrow"));
    CHECK(fs::exists(dst / "logestix" / "pinned" / "layer_0.arrow"));
    CHECK(fs::exists(dst / "logestix" / "pinned" / "layer_0_nvfp4.arrow"));
    CHECK(fs::exists(dst / "logestix" / "pinned" / "layer_2_nvfp4.arrow"));
}

TEST_CASE("nvfp4 converter dequantizes the source full-E4M3 block scales",
          "[nemotron][nvfp4][convert]")
{
    // THE load-bearing assertion. The source block scale carries mantissa bits
    // that the repo decoder ignores, so the converter must dequantize with the
    // COMPLETE E4M3 decode and requantize through the repo's own quantizer.
    // Comparing the stored expert arrow against a host reference built from the
    // full-E4M3 dequant is what pins that; a byte-passthrough converter fails
    // this by a wide margin.
    const auto src = nemotronv4test::MakeTempDir("nmv4_codec_src");
    const auto dst = nemotronv4test::MakeTempDir("nmv4_codec_dst");
    nemotronv4test::WriteNemotronNvfp4Checkpoint(src);
    REQUIRE(ConvertNemotronNvfp4Model(src.string(), dst.string()).ok());

    // Reference: the ACTUAL source tensor the converter consumed, read back out
    // of the fixture's safetensors (not a re-derived copy - a re-derived one
    // would silently drift from the file and make this assertion vacuous).
    const int64_t R = nemotronv4test::kTI, C = nemotronv4test::kTH;
    const auto src_model = lsxcommon::scan_directory(src.string());
    REQUIRE(!src_model.shards.empty());
    auto grab = [&](const char *n) -> const lsxcommon::TensorInfo * {
        for (const auto &sh : src_model.shards)
            for (const auto &t : sh.tensors)
                if (t.name == n) return &t;
        return nullptr;
    };
    auto bytes_of = [&](const lsxcommon::TensorInfo &t) {
        std::ifstream f(src / src_model.shards[0].filename, std::ios::binary);
        uint64_t hl = 0;
        f.read(reinterpret_cast<char *>(&hl), 8);
        std::vector<uint8_t> buf(static_cast<size_t>(t.byte_size()));
        f.seekg(static_cast<std::streamoff>(8 + hl + t.data_begin), std::ios::beg);
        f.read(reinterpret_cast<char *>(buf.data()),
               static_cast<std::streamsize>(buf.size()));
        return buf;
    };
    const auto *w_info =
        grab("backbone.layers.1.mixer.experts.0.up_proj.weight");
    const auto *s_info = grab("backbone.layers.1.mixer.experts.0.up_proj.weight_scale");
    const auto *g_info = grab("backbone.layers.1.mixer.experts.0.up_proj.weight_scale_2");
    REQUIRE(w_info != nullptr);
    REQUIRE(s_info != nullptr);
    REQUIRE(g_info != nullptr);
    const auto packed = bytes_of(*w_info);
    const auto bscale = bytes_of(*s_info);
    const auto gbytes = bytes_of(*g_info);
    float gs = 1.0f;
    std::memcpy(&gs, gbytes.data(), sizeof(float));
    const std::vector<float> truth =
        DequantSourceNvfp4(packed, bscale, gs, R, C);
    double truth_amax = 0.0;
    for (float v : truth) truth_amax = std::max<double>(truth_amax, std::abs(v));
    REQUIRE(truth_amax > 0.0);  // a non-degenerate reference

    // The PREMISE of this whole test, asserted rather than assumed: the source
    // block scales really do carry E4M3 mantissa bits. If they were all
    // pow2-shaped, a byte passthrough would be correct and the measurement
    // below would be meaningless.
    int with_mantissa = 0;
    for (uint8_t b : bscale)
        if ((b & 0x07) != 0) ++with_mantissa;
    INFO("source block scales with a non-zero mantissa: " << with_mantissa
         << " / " << bscale.size());
    CHECK(with_mantissa > static_cast<int>(bscale.size()) / 4);

    // The converted expert arrow: w3 (up) packed by the repo's NVFP4 codec.
    const auto rec = ReadDenseNvfp4Records(
        dst / "logestix" / "experts" / "layer_1" / "exp_0.arrow");
    REQUIRE(rec.ok());
    const lsxcommon::PackedExpert &r = *rec;
    REQUIRE(r.w3_packed.size() == truth.size() / 2);
    REQUIRE(r.w3_scales.size() == (truth.size() + 15) / 16);

    // Decode the repo record exactly as the runtime does
    // (nvfp4_codec.h::E2M1Value * E4M3ScaleToFloat * global_scale).
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < truth.size(); ++i) {
        const uint8_t byte = r.w3_packed[i / 2];
        const int nib = (i & 1) ? (byte >> 4) : (byte & 0xF);
        const uint8_t sb = r.w3_scales[i / 16];
        // The runtime's E4M3ScaleToFloat now IS this complete E4M3FN decode
        // (mantissa + subnormals), so the host reference is identical.
        const float s = E4m3FullToFloat(sb);
        const float got = E2m1Value(nib) * s * r.w3_global_scale;
        const double d = static_cast<double>(got) - truth[i];
        num += d * d;
        den += static_cast<double>(truth[i]) * truth[i];
    }
    const double rel_l2 = std::sqrt(num / den);
    INFO("relative L2 vs the source NVFP4 dequant: " << rel_l2);
    // A correct dequant+requantize lands in the ~11% band (the repo's
    // documented pow2-only scale deviation). A byte passthrough is ~30%, and
    // a naive exponent-only decode of the source scales is worse.
    CHECK(rel_l2 < 0.20);
    // And it must not be "exact by accident" either - i.e. this really did
    // requantize rather than copy.
    CHECK(rel_l2 > 0.0);
}

TEST_CASE("nvfp4 converter refuses a scale tensor with no weight",
          "[nemotron][nvfp4][convert]")
{
    const auto src = nemotronv4test::MakeTempDir("nmv4_orphan_src");
    const auto dst = nemotronv4test::MakeTempDir("nmv4_orphan_dst");
    nemotronv4test::WriteNemotronNvfp4Checkpoint(src);

    // A block scale for a module in a layer that has NO weight (the fixture has
    // 3 layers; layer 9 does not exist). An orphan scale is a hard error, never
    // a silently ignored leftover - and it must be the ORPHAN error, not some
    // incidental one.
    auto orphan = nemotronv4test::MakeNvfp4(
        "backbone.layers.9.mixer.q_proj.weight", 4, 16, 3);
    synthtest::WriteSafetensors(src, "orphan.safetensors",
                                {orphan.weight_scale});

    auto st = ConvertNemotronNvfp4Model(src.string(), dst.string());
    CHECK_FALSE(st.ok());
    CHECK(st.ToString().find("no matching weight") != std::string::npos);
}

TEST_CASE("nvfp4 converter rejects an unrecognized tensor",
          "[nemotron][nvfp4][convert]")
{
    const auto src = nemotronv4test::MakeTempDir("nmv4_bad_src");
    const auto dst = nemotronv4test::MakeTempDir("nmv4_bad_dst");
    nemotronv4test::WriteNemotronNvfp4Checkpoint(src);

    std::vector<synthtest::FakeTensor> extra{
        {"backbone.layers.0.mixer.mystery.weight", "BF16", {2, 2},
         synthtest::MakeBf16(4)}};
    synthtest::WriteSafetensors(src, "bogus.safetensors", extra);

    CHECK_FALSE(ConvertNemotronNvfp4Model(src.string(), dst.string()).ok());
}

TEST_CASE("nvfp4 converter rejects a missing routed expert",
          "[nemotron][nvfp4][convert]")
{
    const auto src = nemotronv4test::MakeTempDir("nmv4_gap_src");
    const auto dst = nemotronv4test::MakeTempDir("nmv4_gap_dst");
    nemotronv4test::WriteNemotronNvfp4Checkpoint(src);

    // A routed expert index beyond n_routed_experts is not "an expert we do
    // not have" - it is a geometry error, and must fail rather than be ignored.
    auto extra = nemotronv4test::MakeNvfp4(
        "backbone.layers.1.mixer.experts.9.up_proj.weight",
        nemotronv4test::kTI, nemotronv4test::kTH, 9);
    synthtest::WriteSafetensors(src, "gap.safetensors",
                                {extra.weight, extra.weight_scale,
                                 extra.weight_scale_2});

    CHECK_FALSE(ConvertNemotronNvfp4Model(src.string(), dst.string()).ok());
}