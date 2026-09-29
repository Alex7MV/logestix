// TDD — Ornith-1.5-35B-A3B standalone MLX converter tests (label convert).
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include <lsxcommon/ornith_moe_convert.h>

#include "test_ornith_moe_fixture.h"

namespace fs = std::filesystem;

namespace {

std::string ReadText(const fs::path &p)
{
    std::ifstream f(p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
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

    const fs::path base = out.path / "logestix";
    REQUIRE(fs::exists(base / "model.index.json"));
    REQUIRE(fs::exists(base / "pinned" / "global_nvfp4.arrow"));
    REQUIRE(fs::exists(base / "router" / "layer_0.arrow"));
    REQUIRE(fs::exists(base / "shared" / "layer_0_nvfp4.arrow"));
    REQUIRE(fs::exists(base / "experts" / "layer_0" / "exp_0.arrow"));

    const std::string idx = ReadText(base / "model.index.json");
    CHECK(idx.find("\"format\": \"logestix-ornithmoe\"") != std::string::npos);
    CHECK(idx.find("\"model\": \"Ornith-1.5-35B-A3B\"") != std::string::npos);
    CHECK(idx.find("\"bos_token_id\": 248044") != std::string::npos);
    CHECK(idx.find("\"num_experts\": 8") != std::string::npos);
    // Conv normalization [C,K,1] -> [C,1,K] is asserted in the loader test.
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
