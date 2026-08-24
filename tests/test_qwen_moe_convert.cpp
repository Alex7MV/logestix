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
