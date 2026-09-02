#include <catch2/catch_test_macros.hpp>
#include "lsxcommon/model_factory.h"
#include "qwen_moe_exp_constants.h"
#include "lsxcommon/qwen_moe_exp_convert.h"

#include <filesystem>

using namespace lsxcommon;

TEST_CASE("registry resolves qwen3.8-flash-next by kType", "[qwen-exp-model]")
{
    ModelInitConfig cfg;
    cfg.model_type = "qwen3.8-flash-next";
    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Type() == "qwen3.8-flash-next");
}

TEST_CASE("frozen geometry differs from synthetic fixture", "[qwen-exp-model]")
{
    // Reflex: the frozen passport constants describe the real 48-layer model,
    // not the tiny synthetic fixture geometry.
    qwen_moe_exp::Qwen4ExpTopology bad{};
    bad.num_layers = 4;
    bad.hidden_size = 64;
    bad.num_experts = 8;
    bad.hc_count = 2;
    REQUIRE(bad.num_layers != qwen_moe_exp::frozen::kLayers);
    REQUIRE(bad.hidden_size != qwen_moe_exp::frozen::kHidden);
    REQUIRE(bad.num_experts != qwen_moe_exp::frozen::kNumExperts);
    REQUIRE(bad.hc_count != qwen_moe_exp::frozen::kHcCount);
}
