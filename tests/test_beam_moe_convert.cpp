// Beam converter stage (Model Ingestion Protocol stages 1/2): config parsing,
// tensor classification, quantization policy, and a synthetic end-to-end
// conversion (label `convert`, bounded TIMEOUT — the frozen geometry gate is
// relaxed with LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY so the fixture never touches
// the real 934 GB checkpoint).
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "beam_moe_constants.h"
#include "lsxcommon/beam_moe_convert.h"
#include "test_beam_moe_fixture.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

using namespace lsxcommon::beam;
namespace fs = std::filesystem;

namespace {

std::string ReadAll(const fs::path &p)
{
    std::ifstream f(p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

} // namespace

TEST_CASE("ParseBeamConfig reads geometry, gate/router fields and layer tables",
          "[beam-moe-convert]")
{
    fs::path dir = fs::temp_directory_path() / "lsx_beam_cfg";
    fs::remove_all(dir);
    fs::create_directories(dir);
    beam_test::WriteConfig(dir);
    auto top = ParseBeamConfig(dir.string());
    REQUIRE(top.error_msg.empty());
    const auto &c = beam_test::TestCfg();
    CHECK(top.hidden_size == c.hidden);
    CHECK(top.num_layers == c.layers);
    CHECK(top.num_attention_heads == c.heads);
    CHECK(top.num_key_value_heads == c.kv_heads);
    CHECK(top.head_dim == c.head_dim);
    CHECK(top.vocab_size == c.vocab);
    CHECK(top.dense_intermediate_size == c.dense_inter);
    CHECK(top.moe_intermediate_size == c.moe_inter);
    CHECK(top.num_experts == c.experts);
    CHECK(top.num_experts_per_tok == c.top_k);
    CHECK(top.n_group == c.n_group);
    CHECK(top.topk_group == c.topk_group);
    CHECK(top.routed_scaling_factor == Catch::Approx(2.5));
    CHECK(top.final_logit_softcapping == Catch::Approx(30.0));
    CHECK(top.sliding_window == c.window);
    CHECK(top.rms_norm_eps == Catch::Approx(1e-6));
    CHECK(top.rope_theta == Catch::Approx(10000.0));
    REQUIRE(top.layer_types.size() == static_cast<size_t>(c.layers));
    CHECK(top.layer_types[static_cast<size_t>(c.full_attn_layer)] ==
          "full_attention");
    CHECK(top.layer_types[0] == "sliding_attention");
    REQUIRE(top.mlp_layer_types.size() == static_cast<size_t>(c.layers));
    CHECK(top.mlp_layer_types[0] == "dense");
    CHECK(top.mlp_layer_types[1] == "sparse");
    REQUIRE(top.eos_ids.size() == 2);
    CHECK(top.eos_ids[0] == c.eos);
    CHECK(top.eos_ids[1] == c.eos2);
    CHECK(top.bos_token_id == -1);
    CHECK(top.pad_token_id == c.pad);
    std::error_code ec;
    fs::remove_all(dir, ec);
}

TEST_CASE("ParseBeamConfig fails on a missing config", "[beam-moe-convert]")
{
    auto top = ParseBeamConfig("/nonexistent_dir_lsx_beam");
    REQUIRE_FALSE(top.error_msg.empty());
}

TEST_CASE("ParseBeamConfig rejects semantics drift the kernels cannot express",
          "[beam-moe-convert]")
{
    const auto rewrite = [](const char *needle, const char *replacement) {
        fs::path dir = fs::temp_directory_path() / "lsx_beam_cfg_drift";
        fs::remove_all(dir);
        fs::create_directories(dir);
        beam_test::WriteConfig(dir);
        std::string j = ReadAll(dir / "config.json");
        const auto at = j.find(needle);
        REQUIRE(at != std::string::npos);
        j.replace(at, std::char_traits<char>::length(needle), replacement);
        std::ofstream(dir / "config.json") << j;
        return ParseBeamConfig(dir.string());
    };
    CHECK_FALSE(rewrite("\"hidden_act\":\"silu\"", "\"hidden_act\":\"gelu\"")
                    .error_msg.empty());
    CHECK_FALSE(rewrite("\"full_attention\":null",
                        "\"full_attention\":{\"rope_type\":\"default\"}")
                    .error_msg.empty());
    // Frozen router geometry is checked by Parse only when the synthetic
    // override is off (the CTest environment sets it for the E2E case).
    unsetenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY");
    CHECK_FALSE(rewrite("\"n_group\":2", "\"n_group\":4").error_msg.empty());
    CHECK_FALSE(rewrite("\"topk_group\":1", "\"topk_group\":4").error_msg.empty());
    CHECK_FALSE(
        rewrite("\"sliding_window\":8", "\"sliding_window\":16").error_msg.empty());
    setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1", 1);
    CHECK_FALSE(rewrite("\"routed_scaling_factor\":2.5",
                        "\"routed_scaling_factor\":1.0")
                    .error_msg.empty());
    CHECK_FALSE(rewrite("\"norm_topk_prob\":true", "\"norm_topk_prob\":false")
                    .error_msg.empty());
    CHECK_FALSE(rewrite("\"attention_bias\":false", "\"attention_bias\":true")
                    .error_msg.empty());
    CHECK_FALSE(rewrite("\"tie_word_embeddings\":false",
                        "\"tie_word_embeddings\":true")
                    .error_msg.empty());
    std::error_code ec;
    fs::remove_all(fs::temp_directory_path() / "lsx_beam_cfg_drift", ec);
}

TEST_CASE("ParseBeamConfig rejects a wrong-length layer table",
          "[beam-moe-convert]")
{
    fs::path dir = fs::temp_directory_path() / "lsx_beam_cfg_bad";
    fs::remove_all(dir);
    fs::create_directories(dir);
    beam_test::WriteConfig(dir);
    std::ofstream f(dir / "config.json");
    f << R"({"model_type":"beam","hidden_size":64,"num_hidden_layers":4,)"
      << R"("num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,)"
      << R"("vocab_size":64,"intermediate_size":32,"moe_intermediate_size":16,)"
      << R"("num_experts":8,"num_experts_per_tok":2,"n_group":2,)"
      << R"("topk_group":1,"routed_scaling_factor":2.5,"sliding_window":8,)"
      << R"("rope_parameters":{"sliding_attention":{"rope_type":"default",)"
      << R"("rope_theta":10000.0},"full_attention":null},)"
      << R"("layer_types":["sliding_attention","full_attention"],)"
      << R"("mlp_layer_types":["dense","sparse","sparse","sparse"]})"
      << "\n";
    f.close();
    auto top = ParseBeamConfig(dir.string());
    CHECK_FALSE(top.error_msg.empty());
    std::error_code ec;
    fs::remove_all(dir, ec);
}

TEST_CASE("beam tensor classification", "[beam-moe-convert]")
{
    using K = BeamTensorKind;
    auto cl = [](const std::string &n) { return classify_beam_tensor(n); };
    CHECK(cl("model.embed_tokens.weight").kind == K::Embed);
    CHECK(cl("lm_head.weight").kind == K::LmHead);
    CHECK(cl("model.norm.weight").kind == K::FinalNorm);
    CHECK(cl("model.layers.3.input_layernorm.weight").kind == K::InputNorm);
    CHECK(cl("model.layers.3.post_attention_layernorm.weight").kind ==
          K::PostAttnNorm);
    CHECK(cl("model.layers.3.pre_feedforward_layernorm.weight").kind ==
          K::PreFfnNorm);
    CHECK(cl("model.layers.3.post_feedforward_layernorm.weight").kind ==
          K::PostFfnNorm);
    CHECK(cl("model.layers.3.self_attn.q_proj.weight").kind == K::AttnQ);
    CHECK(cl("model.layers.3.self_attn.gate_proj.weight").kind == K::AttnGate);
    CHECK(cl("model.layers.3.self_attn.gate_bias").kind == K::AttnGateBias);
    CHECK(cl("model.layers.3.self_attn.q_norm.weight").kind == K::AttnQNorm);
    CHECK(cl("model.layers.3.mlp.gate_proj.weight").kind == K::DenseMlpGate);
    CHECK(cl("model.layers.3.mlp.gate.weight").kind == K::RouterGate);
    CHECK(cl("model.layers.3.mlp.gate.e_score_correction_bias").kind ==
          K::RouterBias);
    auto e = cl("model.layers.3.mlp.experts.17.up_proj.weight");
    CHECK(e.kind == K::ExpertUp);
    CHECK(e.layer == 3);
    CHECK(e.expert == 17);
    CHECK(cl("model.visual.patch_embed.weight").kind == K::Unknown);
    CHECK(cl("model.layers.3.self_attn.rotary_emb.inv_freq").kind == K::Unknown);

    CHECK(beam_tensor_is_nvfp4(K::AttnGate));
    CHECK(beam_tensor_is_nvfp4(K::ExpertDown));
    CHECK_FALSE(beam_tensor_is_nvfp4(K::RouterGate));
    CHECK(beam_tensor_is_routed_expert(K::ExpertGate));
    CHECK_FALSE(beam_tensor_is_routed_expert(K::AttnQ));
}

TEST_CASE("synthetic Beam checkpoint converts to the NVFP4 tree",
          "[beam-moe-convert]")
{
    setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1", 1);
    fs::path src = fs::temp_directory_path() / "lsx_beam_src";
    fs::path out = fs::temp_directory_path() / "lsx_beam_out";
    fs::remove_all(src);
    fs::remove_all(out);
    const int n = beam_test::BuildSyntheticCheckpoint(src);
    REQUIRE(n > 0);

    auto st = ConvertBeamModel(src.string(), out.string());
    INFO(st.ToString());
    REQUIRE(st.ok());

    const fs::path base = out / "logestix";
    REQUIRE(fs::exists(base / "pinned" / "global.arrow"));
    REQUIRE(fs::exists(base / "pinned" / "layer_0_nvfp4.arrow"));
    REQUIRE(fs::exists(base / "pinned" / "layer_1_nvfp4.arrow"));
    REQUIRE(fs::exists(base / "pinned" / "layer_1.arrow"));
    REQUIRE(fs::exists(base / "model.index.json"));

    const auto &c = beam_test::TestCfg();
    for (int l = c.first_moe_layer; l < c.layers; ++l)
        for (int e = 0; e < c.experts; ++e)
            REQUIRE(fs::exists(base / "experts" /
                               absl::StrCat("layer_", l) /
                               absl::StrCat("exp_", e, ".arrow")));

    // Dense NVFP4 records: layer 0 has the dense MLP + 5 attention matrices.
    auto l0 = lsxcommon::ReadRecordBatchFromFile(base / "pinned" /
                                                 "layer_0_nvfp4.arrow");
    REQUIRE(l0.ok());
    auto l0r = lsxcommon::DenseNvfp4Schema::UnpackBatch(*l0);
    REQUIRE(l0r.ok());
    CHECK(l0r->size() == 8);
    for (const auto &r : *l0r) {
        CHECK(r.global_scale > 0.0f);
        CHECK_FALSE(r.packed.empty());
        REQUIRE(r.shape.size() == 2);
        CHECK(r.shape[0] * r.shape[1] / 2 ==
              static_cast<int64_t>(r.packed.size()));
    }
    // A MoE layer has only the 5 attention matrices.
    auto l1 = lsxcommon::ReadRecordBatchFromFile(base / "pinned" /
                                                 "layer_1_nvfp4.arrow");
    REQUIRE(l1.ok());
    auto l1r = lsxcommon::DenseNvfp4Schema::UnpackBatch(*l1);
    REQUIRE(l1r.ok());
    CHECK(l1r->size() == 5);

    // Raw plane: 4 norms + q/k norm + gate_bias + router + correction bias.
    auto raw = lsxcommon::ReadRecordBatchFromFile(base / "pinned" /
                                                  "layer_1.arrow");
    REQUIRE(raw.ok());
    auto rawr = lsxcommon::PinnedSchema::UnpackBatch(*raw);
    REQUIRE(rawr.ok());
    CHECK(rawr->size() == 9);
    bool saw_router = false, saw_bias = false, saw_gate_bias = false;
    for (const auto &r : *rawr) {
        if (r.name == "model.layers.1.mlp.gate.weight") {
            saw_router = true;
            CHECK(r.dtype == "F32");
        }
        if (r.name == "model.layers.1.mlp.gate.e_score_correction_bias") {
            saw_bias = true;
            CHECK(r.dtype == "F32");
        }
        if (r.name == "model.layers.1.self_attn.gate_bias") {
            saw_gate_bias = true;
            CHECK(r.shape == std::vector<int64_t>{c.heads});
        }
    }
    CHECK(saw_router);
    CHECK(saw_bias);
    CHECK(saw_gate_bias);

    auto glob = lsxcommon::ReadRecordBatchFromFile(base / "pinned" /
                                                   "global.arrow");
    REQUIRE(glob.ok());
    auto globr = lsxcommon::PinnedSchema::UnpackBatch(*glob);
    REQUIRE(globr.ok());
    CHECK(globr->size() == 3);

    const std::string idx = ReadAll(base / "model.index.json");
    CHECK(idx.find("\"format\": \"logestix-beam-nvfp4\"") != std::string::npos);
    CHECK(idx.find("\"model_type\": \"beam\"") != std::string::npos);
    CHECK(idx.find("\"num_layers\": 4") != std::string::npos);
    CHECK(idx.find("\"first_moe_layer\": 1") != std::string::npos);
    CHECK(idx.find("\"total_experts\": 24") != std::string::npos);
    CHECK(idx.find("\"n_group\": 2") != std::string::npos);
    CHECK(idx.find("\"topk_group\": 1") != std::string::npos);
    CHECK(idx.find("\"routed_scaling_factor\": 2.5") != std::string::npos);
    CHECK(idx.find("\"full_attention_layers\": [3]") != std::string::npos);

    std::error_code ec;
    fs::remove_all(src, ec);
    fs::remove_all(out, ec);
}
