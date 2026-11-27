// TDD — Ornith-1.5-35B-A3B standalone converter tests (label convert): the
// MLX export layout and the transformers (HF) release layout.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include <lsxcommon/ornith_moe_convert.h>
#include <lsxcommon/schema.h>

#include "test_ornith_moe_fixture.h"

#include <cstring>
#include <optional>

namespace fs = std::filesystem;

namespace {

std::string ReadText(const fs::path &p)
{
    std::ifstream f(p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Reads one named high-precision record out of a converted pinned/*.arrow.
std::optional<lsxcommon::TensorRecord> ReadPinnedTensor(const fs::path &p,
                                                        const std::string &name)
{
    auto batch = lsxcommon::ReadRecordBatchFromFile(p);
    if (!batch.ok()) return std::nullopt;
    auto recs = lsxcommon::PinnedSchema::UnpackBatch(*batch);
    if (!recs.ok()) return std::nullopt;
    for (auto &r : *recs)
        if (r.name == name) return r;
    return std::nullopt;
}

float Bf16At(const std::vector<uint8_t> &data, size_t i)
{
    uint16_t bits = static_cast<uint16_t>(data[i * 2]) |
                    (static_cast<uint16_t>(data[i * 2 + 1]) << 8);
    uint32_t u = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

float F32At(const std::vector<uint8_t> &data, size_t i)
{
    float f;
    std::memcpy(&f, data.data() + i * 4, 4);
    return f;
}

class TempDir {
public:
    explicit TempDir(const char *tag)
    {
        path = fs::temp_directory_path() /
               std::string("lsx_ornith_convert_") /
               (std::string(tag) + "_" +
                std::to_string(
                    std::chrono::steady_clock::now().time_since_epoch().count()));
    }
    ~TempDir()
    {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
    fs::path path;
};

} // namespace

TEST_CASE("ornith MLX classifier maps every tensor kind",
          "[ornith][convert]")
{
    using lsxcommon::ornith::OrnithMoeTensorKind;
    using lsxcommon::ornith::classify_ornith_moe_tensor;

    CHECK(classify_ornith_moe_tensor("language_model.model.embed_tokens.weight")
              .kind == OrnithMoeTensorKind::Embed);
    CHECK(classify_ornith_moe_tensor("language_model.lm_head.weight").kind ==
          OrnithMoeTensorKind::LmHead);
    CHECK(classify_ornith_moe_tensor("language_model.model.norm.weight").kind ==
          OrnithMoeTensorKind::FinalNorm);

    auto layer = classify_ornith_moe_tensor(
        "language_model.model.layers.7.mlp.switch_mlp.gate_proj.weight");
    CHECK(layer.kind == OrnithMoeTensorKind::SwitchGateProj);
    CHECK(layer.layer == 7);
    CHECK(classify_ornith_moe_tensor(
              "language_model.model.layers.7.mlp.switch_mlp.up_proj.weight")
              .kind == OrnithMoeTensorKind::SwitchUpProj);
    CHECK(classify_ornith_moe_tensor(
              "language_model.model.layers.7.mlp.switch_mlp.down_proj.weight")
              .kind == OrnithMoeTensorKind::SwitchDownProj);
    CHECK(classify_ornith_moe_tensor(
              "language_model.model.layers.3.self_attn.q_proj.weight")
              .kind == OrnithMoeTensorKind::FullAttnQ);
    CHECK(classify_ornith_moe_tensor(
              "language_model.model.layers.0.linear_attn.conv1d.weight")
              .kind == OrnithMoeTensorKind::LinearAttnConv);
    CHECK(classify_ornith_moe_tensor("bogus.name").kind ==
          OrnithMoeTensorKind::Unknown);
}

TEST_CASE("ornith classifier tolerates malformed layer indices",
          "[ornith][convert]")
{
    using lsxcommon::ornith::OrnithMoeTensorKind;
    using lsxcommon::ornith::classify_ornith_moe_tensor;
    CHECK_NOTHROW(classify_ornith_moe_tensor(
        "language_model.model.layers..mlp.switch_mlp.gate_proj.weight"));
    CHECK_NOTHROW(classify_ornith_moe_tensor(
        "language_model.model.layers.self_attn.q_proj.weight"));
    CHECK(classify_ornith_moe_tensor(
              "language_model.model.layers..mlp.switch_mlp.gate_proj.weight")
              .kind == OrnithMoeTensorKind::Unknown);
    CHECK(classify_ornith_moe_tensor(
              "language_model.model.layers.self_attn.q_proj.weight")
              .kind == OrnithMoeTensorKind::Unknown);
}

TEST_CASE("ornith converter synthetic roundtrip", "[ornith][convert]")
{
    TempDir src("src"), out("out");
    const int n = ornithtest::BuildSyntheticOrnithCheckpoint(src.path);
    REQUIRE(n > 0);

    const auto st = lsxcommon::ornith::ConvertOrnithMoeModel(
        src.path.string(), out.path.string());
    INFO("convert status: " << st.ToString());
    REQUIRE(st.ok());

    const fs::path base = out.path;
    REQUIRE(fs::exists(base / "model.index.json"));
    REQUIRE(fs::exists(base / "pinned" / "global_nvfp4.arrow"));
    REQUIRE(fs::exists(base / "router" / "layer_0.arrow"));
    REQUIRE(fs::exists(base / "shared" / "layer_0_nvfp4.arrow"));
    REQUIRE(fs::exists(base / "experts" / "layer_0" / "exp_0.arrow"));

    const std::string idx = ReadText(base / "model.index.json");
    CHECK(idx.find("\"layout\": 2") != std::string::npos);
    CHECK(idx.find("\"model\": \"Ornith-1.5-35B-A3B\"") != std::string::npos);
    CHECK(idx.find("\"bos_token_id\": 248044") != std::string::npos);
    CHECK(idx.find("\"num_experts\": 8") != std::string::npos);
    // Conv normalization [C,K,1] -> [C,1,K] is asserted in the loader test.

    // MLX absolute norm (fixture value 1.0) becomes the runtime offset form
    // 1.0 - 1.0 = 0.0 stored as F32.
    const auto mlx_norm =
        ReadPinnedTensor(base / "pinned" / "layer_0.arrow",
                         "language_model.model.layers.0.input_layernorm.weight");
    REQUIRE(mlx_norm.has_value());
    CHECK(mlx_norm->dtype == "F32");
    CHECK(F32At(mlx_norm->data, 0) == Catch::Approx(0.0f));
}

// ── HF (transformers) export layout ────────────────────────────────────────

TEST_CASE("ornith classifier maps the transformers layout", "[ornith][convert]")
{
    using lsxcommon::ornith::OrnithMoeTensorKind;
    using lsxcommon::ornith::OrnithMoeSourceLayout;
    using lsxcommon::ornith::classify_ornith_moe_tensor;

    const auto embed = classify_ornith_moe_tensor(
        "model.language_model.embed_tokens.weight");
    CHECK(embed.kind == OrnithMoeTensorKind::Embed);
    CHECK(embed.layout == OrnithMoeSourceLayout::HuggingFace);

    const auto head = classify_ornith_moe_tensor("lm_head.weight");
    CHECK(head.kind == OrnithMoeTensorKind::LmHead);
    CHECK(head.layout == OrnithMoeSourceLayout::HuggingFace);

    const auto final_norm =
        classify_ornith_moe_tensor("model.language_model.norm.weight");
    CHECK(final_norm.kind == OrnithMoeTensorKind::FinalNorm);
    CHECK(final_norm.layout == OrnithMoeSourceLayout::HuggingFace);

    auto fused = classify_ornith_moe_tensor(
        "model.language_model.layers.7.mlp.experts.gate_up_proj");
    CHECK(fused.kind == OrnithMoeTensorKind::FusedExpertsGateUp);
    CHECK(fused.layer == 7);
    auto fused_down = classify_ornith_moe_tensor(
        "model.language_model.layers.7.mlp.experts.down_proj");
    CHECK(fused_down.kind == OrnithMoeTensorKind::FusedExpertsDown);
    CHECK(fused_down.layer == 7);

    CHECK(lsxcommon::ornith::ornith_moe_tensor_is_nvfp4(
        OrnithMoeTensorKind::FusedExpertsGateUp));
    CHECK(lsxcommon::ornith::ornith_moe_tensor_is_nvfp4(
        OrnithMoeTensorKind::FusedExpertsDown));
    CHECK_FALSE(lsxcommon::ornith::ornith_moe_tensor_is_nvfp4(
        OrnithMoeTensorKind::RouterGate));

    // Ballast classes.
    CHECK(classify_ornith_moe_tensor("model.visual.patch_embed.proj.weight")
              .kind == OrnithMoeTensorKind::Visual);
    CHECK(classify_ornith_moe_tensor("mtp.fc.weight").kind ==
          OrnithMoeTensorKind::Mtp);

    // The MLX layout must keep reporting itself.
    CHECK(classify_ornith_moe_tensor("language_model.model.embed_tokens.weight")
              .layout == OrnithMoeSourceLayout::Mlx);
    CHECK(classify_ornith_moe_tensor("bogus.name").kind ==
          OrnithMoeTensorKind::Unknown);
}

TEST_CASE("ornith HF converter roundtrip strips vision and defers MTP",
          "[ornith][convert]")
{
    TempDir src("hf_src"), out("hf_out");
    const int n = ornithtest::BuildSyntheticOrnithHfCheckpoint(src.path);
    REQUIRE(n > 0);

    const auto st = lsxcommon::ornith::ConvertOrnithMoeModel(
        src.path.string(), out.path.string());
    INFO("convert status: " << st.ToString());
    REQUIRE(st.ok());

    const fs::path base = out.path;
    REQUIRE(fs::exists(base / "model.index.json"));
    REQUIRE(fs::exists(base / "pinned" / "global_nvfp4.arrow"));
    REQUIRE(fs::exists(base / "experts" / "layer_0" / "exp_0.arrow"));
    REQUIRE(fs::exists(base / "experts" /
                       ("layer_" + std::to_string(
                                      ornithtest::OrnithCfg().num_layers - 1))));
    // MTP: stored, never applied.
    REQUIRE(fs::exists(base / "mtp" / "mtp_nvfp4.arrow"));
    REQUIRE(fs::exists(base / "mtp" / "mtp_raw.arrow"));

    const std::string idx = ReadText(base / "model.index.json");
    CHECK(idx.find("\"layout\": 2") != std::string::npos);
    CHECK(idx.find("\"skipped_visual_tensors\": 3") != std::string::npos);
    CHECK(idx.find("\"deferred\": {\"mtp\": \"stored\"") != std::string::npos);
    CHECK(idx.find("\"mtp_tensors\": 6") != std::string::npos);

    // The HF plain norm is ALREADY the runtime offset form: it must pass
    // through untouched (no -1.0 shift, no F32 upcast).
    const auto norm = ReadPinnedTensor(
        base / "pinned" / "layer_0.arrow",
        "model.language_model.layers.0.input_layernorm.weight");
    REQUIRE(norm.has_value());
    CHECK(norm->dtype == "BF16");
    CHECK(Bf16At(norm->data, 0) == Catch::Approx(0.05f).margin(1e-3f));
    CHECK(Bf16At(norm->data, 1) == Catch::Approx(0.06f).margin(1e-3f));

    // The HF conv1d layout [C,1,K] is already normalized.
    const auto conv = ReadPinnedTensor(
        base / "pinned" / "layer_0.arrow",
        "model.language_model.layers.0.linear_attn.conv1d.weight");
    REQUIRE(conv.has_value());
    CHECK(conv->shape ==
          std::vector<int64_t>{ornithtest::OrnithCfg().linear_num_key_heads *
                                   ornithtest::OrnithCfg().linear_key_head_dim * 2 +
                                   ornithtest::OrnithCfg().linear_num_value_heads *
                                       ornithtest::OrnithCfg().linear_value_head_dim,
                               1,
                               ornithtest::OrnithCfg().linear_conv_kernel_dim});
}

TEST_CASE("ornith HF fused experts: rows [0,I) become w1, rows [I,2I) become w3",
          "[ornith][convert]")
{
    const auto &c = ornithtest::OrnithCfg();
    const int E = c.num_experts;
    const int I = c.moe_intermediate_size;
    const int H = c.hidden_size;

    // The fused gate_up_proj [E,2I,H] split is the one numeric transform whose
    // result nothing read back: the other converter tests stop at "the file
    // exists". Stamp the halves so they are structurally DISTINGUISHABLE in
    // the NVFP4 scale bytes -- no FP4 decoding, and no dependence on the
    // global-scale convention:
    //   rows [0,I)  (gate) constant 1.0  -> every block shares one amax, so
    //                                       all w1 block-scale bytes are equal
    //   rows [I,2I) (up)   varies per 16 -> w3 block-scale bytes are not
    // The asymmetry is the point: a gate/up swap makes w1 the varying half,
    // which a symmetric A/B comparison could not detect.
    constexpr int64_t kBlock = 16;
    auto tensors = ornithtest::BuildSyntheticOrnithHfTensors();
    for (auto &t : tensors) {
        if (t.name.find("mlp.experts.gate_up_proj") == std::string::npos)
            continue;
        t.data = synthtest::MakeBf16Values(
            static_cast<int64_t>(E) * 2 * I * H, [=](int64_t i) {
                const int64_t row = (i / H) % (2 * I);
                if (row < I) return 1.0f;  // uniform gate half
                return 0.1f + 0.2f * static_cast<float>((i / kBlock) % 4);
            });
    }

    TempDir src("split_src"), out("split_out");
    fs::create_directories(src.path);
    synthtest::WriteSafetensors(src.path, "model.safetensors", tensors);
    ornithtest::WriteOrnithConfig(src.path);
    const auto st = lsxcommon::ornith::ConvertOrnithMoeModel(
        src.path.string(), out.path.string());
    INFO("convert status: " << st.ToString());
    REQUIRE(st.ok());

    auto batch = lsxcommon::ReadRecordBatchFromFile(
        out.path / "experts" / "layer_0" / "exp_0.arrow");
    REQUIRE(batch.ok());
    auto pe = lsxcommon::ExpertSchema::UnpackBatch(*batch);
    REQUIRE(pe.ok());
    REQUIRE(!pe->w1_scales.empty());
    REQUIRE(!pe->w3_scales.empty());

    const bool w1_uniform =
        std::all_of(pe->w1_scales.begin(), pe->w1_scales.end(),
                    [&](uint8_t b) { return b == pe->w1_scales.front(); });
    const bool w3_uniform =
        std::all_of(pe->w3_scales.begin(), pe->w3_scales.end(),
                    [&](uint8_t b) { return b == pe->w3_scales.front(); });

    // w1 came from the uniform gate half; w3 from the varying up half.
    CHECK(w1_uniform);
    CHECK_FALSE(w3_uniform);
}

TEST_CASE("ornith frozen gate accepts real geometry and rejects a mutation",
          "[ornith][convert]")
{
    using lsxcommon::ornith::CheckOrnithFrozenGeometry;
    using lsxcommon::ornith::OrnithMoeTopology;

    OrnithMoeTopology top;
    top.hidden_size = 2048;
    top.num_layers = 40;
    top.num_attention_heads = 16;
    top.num_key_value_heads = 2;
    top.head_dim = 256;
    top.linear_num_key_heads = 16;
    top.linear_num_value_heads = 32;
    top.linear_key_head_dim = 128;
    top.linear_value_head_dim = 128;
    top.vocab_size = 248320;
    top.num_experts = 256;
    top.num_experts_per_tok = 8;
    top.moe_intermediate_size = 512;
    top.shared_expert_intermediate_size = 512;
    top.bos_token_id = 248044;
    top.layer_types.assign(40, "linear_attention");
    for (int i = 3; i < 40; i += 4) top.layer_types[static_cast<size_t>(i)] =
        "full_attention";

    std::string why;
    CHECK(CheckOrnithFrozenGeometry(top, why));

    top.num_experts = 128;
    CHECK_FALSE(CheckOrnithFrozenGeometry(top, why));
    CHECK_FALSE(why.empty());
}
