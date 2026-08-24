#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "qwen_moe_test_fixture.h"
#include "lsxcommon/qwen_moe_convert.h"

#include <filesystem>

using namespace lsxcommon::qwen_moe;
namespace fs = std::filesystem;

TEST_CASE("ParseQwenMoeConfig reads text_config MoE fields", "[qwen-moe-convert]")
{
    fs::path dir = fs::temp_directory_path() / "lsx_qwen_moe_cfg";
    fs::create_directories(dir);
    qwen_moe_test::WriteMoeConfig(dir);
    auto top = ParseQwenMoeConfig(dir.string());
    REQUIRE(top.error_msg.empty());
    REQUIRE(top.hidden_size == 64);
    REQUIRE(top.num_layers == 4);
    REQUIRE(top.num_experts == 8);
    REQUIRE(top.num_experts_per_tok == 2);
    REQUIRE(top.moe_intermediate_size == 16);
    REQUIRE(top.shared_expert_intermediate_size == 16);
    REQUIRE(top.num_attention_heads == 4);
    REQUIRE(top.num_key_value_heads == 1);
    REQUIRE(top.head_dim == 32);
    REQUIRE(top.linear_num_key_heads == 2);
    REQUIRE(top.linear_num_value_heads == 4);
    REQUIRE(top.linear_key_head_dim == 8);
    REQUIRE(top.linear_value_head_dim == 8);
    REQUIRE(top.vocab_size == 64);
    REQUIRE(top.eos_token_id == 7);
    REQUIRE(top.rms_norm_eps == Catch::Approx(1e-6));
    REQUIRE(top.partial_rotary_factor == Catch::Approx(0.25));
    REQUIRE(top.rope_theta == Catch::Approx(10000.0));
    REQUIRE(top.layer_types.size() == 4);
    REQUIRE(top.layer_types[3] == "full_attention");
    std::error_code ec;
    fs::remove_all(dir, ec);
}

TEST_CASE("ParseQwenMoeConfig fails on missing config", "[qwen-moe-convert]")
{
    auto top = ParseQwenMoeConfig("/nonexistent_dir_lsx_qwen_moe");
    REQUIRE_FALSE(top.error_msg.empty());
}

TEST_CASE("MoE tensor classification", "[qwen-moe-convert]")
{
    using K = lsxcommon::qwen_moe::MoeTensorKind;
    auto cl = [](const std::string &n) {
        return lsxcommon::qwen_moe::classify_qwen_moe_tensor(n);
    };
    CHECK(cl("model.language_model.embed_tokens.weight").kind == K::Embed);
    CHECK(cl("lm_head.weight").kind == K::LmHead);
    CHECK(cl("model.language_model.norm.weight").kind == K::FinalNorm);
    CHECK(cl("model.language_model.layers.3.self_attn.q_proj.weight").kind ==
          K::FullAttnQ);
    CHECK(cl("model.language_model.layers.3.self_attn.q_proj.weight").layer == 3);
    CHECK(cl("model.language_model.layers.0.linear_attn.in_proj_qkv.weight")
              .kind == K::LinearAttnQkv);
    CHECK(cl("model.language_model.layers.0.linear_attn.A_log").layer == 0);
    CHECK(cl("model.language_model.layers.7.self_attn.q_norm.weight").kind ==
          K::FullAttnQNorm);
    CHECK(cl("model.language_model.layers.2.input_layernorm.weight").kind ==
          K::InputNorm);
    CHECK(cl("model.language_model.layers.2.post_attention_layernorm.weight")
              .kind == K::PostNorm);
    CHECK(cl("model.language_model.layers.5.mlp.gate.weight").kind ==
          K::RouterGate);
    CHECK(cl("model.language_model.layers.5.mlp.experts.gate_up_proj").kind ==
          K::FusedExpertsGateUp);
    CHECK(cl("model.language_model.layers.5.mlp.experts.gate_up_proj").layer == 5);
    CHECK(cl("model.language_model.layers.5.mlp.experts.down_proj").kind ==
          K::FusedExpertsDown);
    CHECK(cl("model.language_model.layers.1.mlp.shared_expert.gate_proj.weight")
              .kind == K::SharedExpertGateProj);
    CHECK(cl("model.language_model.layers.1.mlp.shared_expert.up_proj.weight")
              .kind == K::SharedExpertUpProj);
    CHECK(cl("model.language_model.layers.1.mlp.shared_expert.down_proj.weight")
              .kind == K::SharedExpertDownProj);
    CHECK(cl("model.language_model.layers.1.mlp.shared_expert_gate.weight")
              .kind == K::SharedExpertGate);
    CHECK(cl("mtp.fc.weight").kind == K::Mtp);
    CHECK(cl("model.visual.blocks.0.attn.qkv.weight").kind == K::Visual);
    CHECK(cl("totally.unknown.name").kind == K::Unknown);
}

TEST_CASE("MoE quantization policy keeps norms/router/gates high precision",
          "[qwen-moe-convert]")
{
    using K = lsxcommon::qwen_moe::MoeTensorKind;
    using lsxcommon::qwen_moe::qwen_moe_tensor_is_nvfp4;
    CHECK(qwen_moe_tensor_is_nvfp4(K::FullAttnQ));
    CHECK(qwen_moe_tensor_is_nvfp4(K::FullAttnK));
    CHECK(qwen_moe_tensor_is_nvfp4(K::FullAttnV));
    CHECK(qwen_moe_tensor_is_nvfp4(K::FullAttnO));
    CHECK(qwen_moe_tensor_is_nvfp4(K::LinearAttnOut));
    CHECK(qwen_moe_tensor_is_nvfp4(K::FusedExpertsGateUp));
    CHECK(qwen_moe_tensor_is_nvfp4(K::FusedExpertsDown));
    CHECK(qwen_moe_tensor_is_nvfp4(K::SharedExpertGateProj));
    CHECK(qwen_moe_tensor_is_nvfp4(K::SharedExpertUpProj));
    CHECK(qwen_moe_tensor_is_nvfp4(K::SharedExpertDownProj));
    CHECK(qwen_moe_tensor_is_nvfp4(K::Embed));
    CHECK(qwen_moe_tensor_is_nvfp4(K::LmHead));
    CHECK_FALSE(qwen_moe_tensor_is_nvfp4(K::RouterGate));
    CHECK_FALSE(qwen_moe_tensor_is_nvfp4(K::SharedExpertGate));
    CHECK_FALSE(qwen_moe_tensor_is_nvfp4(K::InputNorm));
    CHECK_FALSE(qwen_moe_tensor_is_nvfp4(K::PostNorm));
    CHECK_FALSE(qwen_moe_tensor_is_nvfp4(K::FullAttnQNorm));
    CHECK_FALSE(qwen_moe_tensor_is_nvfp4(K::FullAttnKNorm));
    CHECK_FALSE(qwen_moe_tensor_is_nvfp4(K::FinalNorm));
    CHECK_FALSE(qwen_moe_tensor_is_nvfp4(K::LinearAttnConv));
    CHECK_FALSE(qwen_moe_tensor_is_nvfp4(K::LinearAttnALog));
    CHECK_FALSE(qwen_moe_tensor_is_nvfp4(K::LinearAttnDt));
}
