// Ling-3.1-flash converter stage (Model Ingestion Protocol stages 1/2):
// config parsing, tensor classification, FP8 tile handling, and a synthetic
// end-to-end conversion (label `convert`, bounded TIMEOUT). The frozen
// passport gate is relaxed with LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY so the
// fixture never touches the real 535 GB checkpoint.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "ling_moe_constants.h"
#include "lsxcommon/ling_moe_convert.h"
#include "test_ling_moe_fixture.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace lsxcommon::ling;
namespace fs = std::filesystem;

namespace {

std::string ReadAll(const fs::path &p)
{
    std::ifstream f(p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::vector<std::string> DenseNamesIn(const fs::path &p)
{
    auto batch = lsxcommon::ReadRecordBatchFromFile(p);
    if (!batch.ok()) return {};
    auto recs = lsxcommon::DenseNvfp4Schema::UnpackBatch(*batch);
    if (!recs.ok()) return {};
    std::vector<std::string> names;
    for (auto &r : *recs) names.push_back(r.name);
    return names;
}

bool Contains(const std::vector<std::string> &v, const std::string &s)
{
    for (const auto &x : v)
        if (x == s) return true;
    return false;
}

} // namespace

TEST_CASE("ParseLingConfig reads the geometry and derives the layer tables",
          "[ling-moe-convert]")
{
    fs::path dir = fs::temp_directory_path() / "lsx_ling_cfg";
    fs::remove_all(dir);
    fs::create_directories(dir);
    ling_test::WriteConfig(dir);
    auto top = ParseLingConfig((dir / "config.json").string());
    INFO(top.error_msg);
    REQUIRE(top.error_msg.empty());
    const auto &c = ling_test::TestCfg();
    CHECK(top.hidden == c.hidden);
    CHECK(top.layers == c.layers);
    CHECK(top.vocab == c.vocab);
    CHECK(top.heads == c.heads);
    CHECK(top.kv_heads == c.kv_heads);
    CHECK(top.head_dim == c.head_dim);
    CHECK(top.qk_head_dim == c.qk_head_dim);
    CHECK(top.qk_nope == c.qk_nope);
    CHECK(top.qk_rope == c.qk_rope);
    CHECK(top.v_head == c.v_head);
    CHECK(top.q_lora == c.q_lora);
    CHECK(top.kv_lora == c.kv_lora);
    CHECK(top.conv_kernel == c.conv);
    CHECK(top.num_experts == c.experts);
    CHECK(top.top_k == c.top_k);
    CHECK(top.n_group == c.n_group);
    CHECK(top.topk_group == c.topk_group);
    CHECK(top.moe_intermediate == c.moe_inter);
    CHECK(top.shared_intermediate == c.shared_inter);
    CHECK(top.dense_intermediate == c.dense_inter);
    CHECK(top.first_k_dense == c.first_k_dense);
    CHECK(top.rms_eps == Catch::Approx(1e-6));
    CHECK(top.rope_theta == Catch::Approx(10000.0));
    CHECK(top.routed_scale == Catch::Approx(2.5));
    CHECK(top.eos_id == c.eos);
    CHECK(top.pad_id == c.pad);
    REQUIRE(top.mla_layers.size() == 1);
    CHECK(top.mla_layers[0] == 7);
    CHECK(top.kda_layers.size() == 7);
    CHECK(top.moe_layers.size() == 7);
    CHECK(top.dense_layers.size() == 1);
    std::error_code ec;
    fs::remove_all(dir, ec);
}

TEST_CASE("ParseLingConfig fails on a missing config", "[ling-moe-convert]")
{
    auto top = ParseLingConfig("/nonexistent_dir_lsx_ling");
    REQUIRE_FALSE(top.error_msg.empty());
}

TEST_CASE("ParseLingConfig rejects semantics drift the kernels cannot express",
          "[ling-moe-convert]")
{
    const auto rewrite = [](const char *needle, const char *replacement) {
        fs::path dir = fs::temp_directory_path() / "lsx_ling_cfg_drift";
        fs::remove_all(dir);
        fs::create_directories(dir);
        ling_test::WriteConfig(dir);
        std::string j = ReadAll(dir / "config.json");
        const auto at = j.find(needle);
        REQUIRE(at != std::string::npos);
        j.replace(at, std::char_traits<char>::length(needle), replacement);
        std::ofstream(dir / "config.json") << j;
        auto top = ParseLingConfig((dir / "config.json").string());
        std::error_code ec;
        fs::remove_all(dir, ec);
        return top;
    };
    CHECK_FALSE(rewrite("\"no_kda_lora\":true", "\"no_kda_lora\":false")
                    .error_msg.empty());
    CHECK_FALSE(rewrite("\"kda_safe_gate\":true", "\"kda_safe_gate\":false")
                    .error_msg.empty());
    CHECK_FALSE(rewrite("\"rope_interleave\":true",
                        "\"rope_interleave\":false")
                    .error_msg.empty());
    CHECK_FALSE(rewrite("\"norm_topk_prob\":true",
                        "\"norm_topk_prob\":false")
                    .error_msg.empty());
    CHECK_FALSE(rewrite("\"tie_word_embeddings\":false",
                        "\"tie_word_embeddings\":true")
                    .error_msg.empty());
    CHECK_FALSE(rewrite("\"topk_method\":\"noaux_tc\"",
                        "\"topk_method\":\"greedy\"")
                    .error_msg.empty());
}

TEST_CASE("ling tensor classification", "[ling-moe-convert]")
{
    using K = LingTensorKind;
    auto cl = [](const std::string &n) { return classify_ling_tensor(n); };
    CHECK(cl("model.word_embeddings.weight").kind == K::Embed);
    CHECK(cl("lm_head.weight").kind == K::LmHead);
    CHECK(cl("model.norm.weight").kind == K::FinalNorm);
    CHECK(cl("model.layers.0.attention.q_proj.weight").kind == K::KdaQ);
    CHECK(cl("model.layers.0.attention.f_proj.weight").kind == K::KdaF);
    CHECK(cl("model.layers.0.attention.g_proj.weight").kind == K::KdaG);
    CHECK(cl("model.layers.0.attention.q_conv1d.weight").kind == K::KdaConv);
    CHECK(cl("model.layers.0.attention.A_log").kind == K::KdaALog);
    CHECK(cl("model.layers.0.attention.dt_bias").kind == K::KdaDtBias);
    CHECK(cl("model.layers.0.attention.b_proj.weight").kind == K::KdaB);
    CHECK(cl("model.layers.0.attention.o_norm.weight").kind == K::KdaONorm);
    CHECK(cl("model.layers.7.attention.q_a_proj.weight").kind == K::MlaQa);
    CHECK(cl("model.layers.7.attention.q_b_proj.weight").kind == K::MlaQb);
    CHECK(cl("model.layers.7.attention.kv_a_proj_with_mqa.weight").kind ==
          K::MlaKvA);
    CHECK(cl("model.layers.7.attention.kv_b_proj.weight").kind == K::MlaKvB);
    CHECK(cl("model.layers.7.attention.dense.weight").kind == K::MlaDense);
    CHECK(cl("model.layers.7.attention.g_proj.weight").kind == K::MlaG);
    CHECK(cl("model.layers.7.attention.q_a_layernorm.weight").kind ==
          K::MlaNorm);
    CHECK(cl("model.layers.4.mlp.gate.weight").kind == K::RouterGate);
    CHECK(cl("model.layers.4.mlp.gate.expert_bias").kind == K::RouterBias);
    CHECK(cl("model.layers.4.mlp.shared_experts.gate_proj.weight").kind ==
          K::SharedGate);
    CHECK(cl("model.layers.4.mlp.experts.17.gate_proj.weight").kind ==
          K::RoutedGate);
    CHECK(cl("model.layers.4.mlp.experts.17.down_proj.weight").kind ==
          K::RoutedDown);
    CHECK(cl("model.layers.4.mlp.experts.17.up_proj.weight").kind ==
          K::RoutedUp);
    CHECK(cl("model.layers.72.eh_proj.weight").kind == K::Mtp);
    CHECK(cl("something.else").kind == K::Unknown);
    CHECK(cl("model.layers.4.mlp.experts.17.gate_proj.weight_scale_inv").kind ==
          K::Unknown);
}

TEST_CASE("ling quantization policy", "[ling-moe-convert]")
{
    using K = LingTensorKind;
    CHECK(ling_tensor_is_nvfp4(K::KdaQ));
    CHECK(ling_tensor_is_nvfp4(K::KdaF));
    CHECK(ling_tensor_is_nvfp4(K::KdaG));
    CHECK(ling_tensor_is_nvfp4(K::MlaQa));
    CHECK(ling_tensor_is_nvfp4(K::MlaDense));
    CHECK(ling_tensor_is_nvfp4(K::DenseMlpGate));
    CHECK(ling_tensor_is_nvfp4(K::SharedDown));
    CHECK(ling_tensor_is_nvfp4(K::RoutedUp));
    CHECK_FALSE(ling_tensor_is_nvfp4(K::KdaConv));
    CHECK_FALSE(ling_tensor_is_nvfp4(K::KdaALog));
    CHECK_FALSE(ling_tensor_is_nvfp4(K::KdaB));
    CHECK_FALSE(ling_tensor_is_nvfp4(K::MlaNorm));
    CHECK_FALSE(ling_tensor_is_nvfp4(K::MlaG));
    CHECK_FALSE(ling_tensor_is_nvfp4(K::RouterGate));
    CHECK_FALSE(ling_tensor_is_nvfp4(K::Embed));
    CHECK_FALSE(ling_tensor_is_nvfp4(K::LmHead));
}

TEST_CASE("LingDequantFp8Block128 clamps the partial last tile",
          "[ling-moe-convert]")
{
    // 130 rows -> two scale rows; the second tile only covers 2 rows.
    const std::vector<int64_t> shape{130, 128};
    const std::vector<int64_t> sshape{2, 1};
    std::vector<uint8_t> fp8(130 * 128, 0x38);  // E4M3 1.0
    std::vector<uint8_t> scale(synthtest::MakeF32Values(
        2, [](int64_t i) { return i == 0 ? 1.0f : 0.25f; }));
    std::vector<uint8_t> bf16;
    auto st = LingDequantFp8Block128ToBf16Gpu(fp8, shape, scale, sshape, bf16);
    REQUIRE(st.ok());
    REQUIRE(bf16.size() == static_cast<size_t>(130 * 128 * 2));
    const auto read_bf16 = [&](int64_t i) {
        uint16_t b = static_cast<uint16_t>(bf16[static_cast<size_t>(i) * 2]) |
                     (static_cast<uint16_t>(
                          bf16[static_cast<size_t>(i) * 2 + 1])
                      << 8);
        uint32_t u = static_cast<uint32_t>(b) << 16;
        float v;
        std::memcpy(&v, &u, sizeof(v));
        return v;
    };
    CHECK(read_bf16(0) == Catch::Approx(1.0f));
    CHECK(read_bf16(127 * 128) == Catch::Approx(1.0f));
    CHECK(read_bf16(128 * 128) == Catch::Approx(0.25f));
    CHECK(read_bf16(129 * 128 + 127) == Catch::Approx(0.25f));
}

TEST_CASE("ConvertLingModel synthetic end to end", "[ling-moe-convert]")
{
    setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1", 1);
    fs::path in = fs::temp_directory_path() / "lsx_ling_convert_in";
    fs::path out = fs::temp_directory_path() / "lsx_ling_convert_out";
    fs::remove_all(in);
    fs::remove_all(out);
    ling_test::WriteCheckpoint(in);
    auto st = ConvertLingModel(in.string(), out.string());
    INFO(st.ToString());
    REQUIRE(st.ok());

    const fs::path base = out / "logestix";
    const std::string idx = ReadAll(base / "model.index.json");
    CHECK(idx.find("\"format\": \"logestix-ling-nvfp4\"") != std::string::npos);
    CHECK(idx.find("\"model\": \"Ling-3.1-flash\"") != std::string::npos);
    CHECK(idx.find("\"mla_layers\": [7]") != std::string::npos);
    CHECK(idx.find("\"deferred\": {\"mtp\": 5}") != std::string::npos);
    CHECK(idx.find("\"total_experts\": 56") != std::string::npos);

    auto global = synthtest::TensorNamesIn(base / "pinned" / "global.arrow");
    CHECK(global.size() == 3);
    CHECK(Contains(global, "model.word_embeddings.weight"));
    CHECK(Contains(global, "lm_head.weight"));
    CHECK(Contains(global, "model.norm.weight"));

    auto dense = DenseNamesIn(base / "pinned" / "layer_7_nvfp4.arrow");
    CHECK(Contains(dense, "model.layers.7.attention.q_a_proj.weight"));
    CHECK(Contains(dense, "model.layers.7.attention.dense.weight"));
    CHECK_FALSE(Contains(dense, "model.layers.7.attention.g_proj.weight"));

    auto raw = synthtest::TensorNamesIn(base / "pinned" / "layer_0.arrow");
    CHECK(Contains(raw, "model.layers.0.attention.A_log"));
    CHECK(Contains(raw, "model.layers.0.attention.q_conv1d.weight"));
    CHECK(Contains(raw, "model.layers.0.attention.b_proj.weight"));
    CHECK(Contains(raw, "model.layers.0.attention.o_norm.weight"));

    auto router = synthtest::TensorNamesIn(base / "router" / "layer_1.arrow");
    CHECK(router.size() == 2);
    CHECK(Contains(router, "model.layers.1.mlp.gate.weight"));
    CHECK(Contains(router, "model.layers.1.mlp.gate.expert_bias"));

    auto shared = DenseNamesIn(base / "shared" / "layer_1_nvfp4.arrow");
    CHECK(shared.size() == 3);

    CHECK(fs::exists(base / "experts" / "layer_1" / "exp_0.arrow"));
    CHECK(fs::exists(base / "experts" / "layer_7" / "exp_7.arrow"));
    CHECK_FALSE(fs::exists(base / "pinned" / "layer_8.arrow"));
    CHECK_FALSE(fs::exists(base / "pinned" / "layer_8_nvfp4.arrow"));

    auto pe_batch = lsxcommon::ReadRecordBatchFromFile(
        base / "experts" / "layer_1" / "exp_0.arrow");
    REQUIRE(pe_batch.ok());
    auto pe = lsxcommon::ExpertSchema::UnpackBatch(*pe_batch);
    REQUIRE(pe.ok());
    CHECK(pe->w1_packed.size() == static_cast<size_t>(128 * 256 / 2));
    CHECK(pe->w2_packed.size() == static_cast<size_t>(128 * 256 / 2));
    CHECK(pe->w3_packed.size() == static_cast<size_t>(128 * 256 / 2));
    CHECK(pe->w1_scales.size() == static_cast<size_t>(128 * 256 / 16));
    CHECK(pe->w1_global_scale != 1.0f);

    std::error_code ec;
    fs::remove_all(in, ec);
    fs::remove_all(out, ec);
}
