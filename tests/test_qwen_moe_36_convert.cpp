#include <catch2/catch_test_macros.hpp>
#include "test_qwen_moe_fixture.h"
#include "lsxcommon/qwen_moe_convert.h"
#include "lsxcommon/qwen_moe_36_convert.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

using namespace lsxcommon::qwen_moe;
using namespace lsxcommon::qwen_moe_36;
namespace fs = std::filesystem;

namespace {

// Appends an explicit bos_token_id into the synthetic text_config (mirrors
// the real Qwen3.6-35B-A3B config.json layout).
void PatchConfigBos(const fs::path &dir, int bos_id)
{
    const fs::path cfg = dir / "config.json";
    std::ifstream f(cfg);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string j = ss.str();
    const std::string needle = ",\"hidden_act\":\"silu\"";
    const auto pos = j.find(needle);
    REQUIRE(pos != std::string::npos);
    j.replace(pos, needle.size(),
              ",\"hidden_act\":\"silu\",\"bos_token_id\":" +
                  std::to_string(bos_id));
    std::ofstream g(cfg, std::ios::trunc);
    g << j;
}

std::string ReadIndex(const fs::path &out)
{
    std::ifstream f(out / "logestix" / "model.index.json");
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

} // namespace

TEST_CASE("Qwen3.6 MTP fused-expert tensor names classify as Mtp",
          "[qwen36-convert]")
{
    using K = MoeTensorKind;
    CHECK(classify_qwen_moe_tensor(
              "mtp.layers.0.mlp.experts.gate_up_proj").kind == K::Mtp);
    CHECK(classify_qwen_moe_tensor(
              "mtp.layers.0.mlp.experts.down_proj").kind == K::Mtp);
    CHECK(classify_qwen_moe_tensor(
              "mtp.layers.0.self_attn.q_proj.weight").kind == K::Mtp);
    // Main-line grammar unchanged from 3.5.
    CHECK(classify_qwen_moe_tensor(
              "model.language_model.layers.4.mlp.experts.gate_up_proj")
              .kind == K::FusedExpertsGateUp);
    CHECK(classify_qwen_moe_tensor(
              "model.visual.blocks.0.attn.qkv.weight").kind == K::Visual);
}

TEST_CASE("ParseQwenMoeConfig reads explicit bos_token_id", "[qwen36-convert]")
{
    fs::path dir = fs::temp_directory_path() / "lsx_qwen36_cfg";
    fs::remove_all(dir);
    fs::create_directories(dir);
    qwen_moe_test::BuildSyntheticMoeCheckpoint(dir);
    PatchConfigBos(dir, 5);
    auto top = ParseQwenMoeConfig(dir.string());
    REQUIRE(top.error_msg.empty());
    REQUIRE(top.bos_token_id == 5);
    std::error_code ec;
    fs::remove_all(dir, ec);
}

TEST_CASE("ConvertQwen36MoeModel synthetic roundtrip (format/model/bos)",
          "[qwen36-convert]")
{
    fs::path in = fs::temp_directory_path() / "lsx_qwen36_in";
    fs::path out = fs::temp_directory_path() / "lsx_qwen36_out";
    fs::remove_all(in);
    fs::remove_all(out);
    REQUIRE(qwen_moe_test::BuildSyntheticMoeCheckpoint(in) > 0);

    auto st = ConvertQwen36MoeModel(in.string(), out.string());
    if (!st.ok()) WARN(st.ToString());
    REQUIRE(st.ok());

    const std::string j = ReadIndex(out);
    CHECK(j.find("\"logestix-qwen36moe\"") != std::string::npos);
    CHECK(j.find("\"Qwen3.6-35B-A3B\"") != std::string::npos);
    // Synthetic config carries no bos: record_bos must NOT emit the field.
    CHECK(j.find("bos_token_id") == std::string::npos);

    // Unified layout: expert split + MTP store.
    CHECK(fs::exists(out / "logestix" / "experts" / "layer_0" / "exp_0.arrow"));
    CHECK(fs::exists(out / "logestix" / "mtp" / "mtp_nvfp4.arrow"));

    // Determinism: byte-identical index on re-conversion.
    fs::path out2 = fs::temp_directory_path() / "lsx_qwen36_out2";
    fs::remove_all(out2);
    REQUIRE(ConvertQwen36MoeModel(in.string(), out2.string()).ok());
    CHECK(ReadIndex(out) == ReadIndex(out2));
    fs::remove_all(out2);

    std::error_code ec;
    fs::remove_all(in, ec);
    fs::remove_all(out, ec);
}

TEST_CASE("ConvertQwen36MoeModel writes config bos_token_id into the index",
          "[qwen36-convert]")
{
    fs::path in = fs::temp_directory_path() / "lsx_qwen36_bos_in";
    fs::path out = fs::temp_directory_path() / "lsx_qwen36_bos_out";
    fs::remove_all(in);
    fs::remove_all(out);
    REQUIRE(qwen_moe_test::BuildSyntheticMoeCheckpoint(in) > 0);
    PatchConfigBos(in, 5);

    REQUIRE(ConvertQwen36MoeModel(in.string(), out.string()).ok());
    CHECK(ReadIndex(out).find("\"bos_token_id\": 5") != std::string::npos);

    std::error_code ec;
    fs::remove_all(in, ec);
    fs::remove_all(out, ec);
}

TEST_CASE("ConvertQwen35MoeModel output stays bos-free (byte-identity guard)",
          "[qwen36-convert]")
{
    fs::path in = fs::temp_directory_path() / "lsx_qwen35_bos_in";
    fs::path out = fs::temp_directory_path() / "lsx_qwen35_bos_out";
    fs::remove_all(in);
    fs::remove_all(out);
    REQUIRE(qwen_moe_test::BuildSyntheticMoeCheckpoint(in) > 0);
    PatchConfigBos(in, 5);

    REQUIRE(ConvertQwen35MoeModel(in.string(), out.string()).ok());
    CHECK(ReadIndex(out).find("bos_token_id") == std::string::npos);
    CHECK(ReadIndex(out).find("\"logestix-qwen35moe\"") != std::string::npos);

    std::error_code ec;
    fs::remove_all(in, ec);
    fs::remove_all(out, ec);
}
