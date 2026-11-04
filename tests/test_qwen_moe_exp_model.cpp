#include <catch2/catch_test_macros.hpp>
#include "lsxcommon/model_factory.h"
#include "qwen_moe_exp_constants.h"
#include "lsxcommon/qwen_moe_exp_convert.h"
#include "lsxcommon/qwen_tool_format.h"

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

// Stage 4b: the model must frame with its own template, not the 3.5/3.6 one.
//
// The codec carries the deltas behind QwenRenderPolicy, and each flag is
// exercised individually in test_qwen_tool_framing against the checkpoint's
// `chat_template.jinja` rendered by transformers 5.15.1 -- but a codepath can
// still forget to switch a flag on, and that is exactly what happened first
// here: `render_content(...)|trim` (template lines 70/78/92/103) was covered by
// a test and not enabled for the model, so a client message with leading or
// trailing whitespace framed with bytes the reference never produces.
// Pin the whole policy the model renders with, so a default flip or a new flag
// added without a decision fails here instead of silently changing the wire.
TEST_CASE("qwen4_exp model frames with the checkpoint's own template deltas",
          "[qwen-exp-model]")
{
    const QwenRenderPolicy &p = kQwen38RenderPolicy;
    // The goldens (FSI/CODE/DET) and the tool round-trip are captured with
    // thinking disabled: the generation prompt closes the empty think block.
    REQUIRE(p.enable_thinking == false);
    // Every assistant *history* turn keeps the closed think block:
    // `assistant\n<think>\n{reasoning}\n</think>\n\n{content}`.
    REQUIRE(p.history_think_block);
    // Tool definitions serialise like jinja's `tojson`
    // (json.dumps(obj, ensure_ascii=False): ", " / ": " separators).
    REQUIRE(p.jinja_tool_json);
    // Message bodies go through `render_content(...)|trim`.
    REQUIRE(p.trim_content);
}
