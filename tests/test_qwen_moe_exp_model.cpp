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
// Only the deltas that genuinely vary per family live in QwenRenderPolicy; the
// shared rules (`render_content|trim` and `tool|tojson`) are unconditional in
// the codec and are pinned by test_qwen_tool_framing against the template
// rendered by transformers. Pin what the model does choose, so a default flip
// added without a decision fails here instead of silently changing the wire.
TEST_CASE("qwen4_exp model frames with the checkpoint's own template deltas",
          "[qwen-exp-model]")
{
    const QwenRenderPolicy &p = kQwen38RenderPolicy;
    // The goldens (FSI/CODE/DET) and the tool round-trip are captured with
    // thinking disabled: the generation prompt closes the empty think block.
    REQUIRE(p.enable_thinking == false);
    // Every assistant *history* turn keeps the closed think block:
    // `assistant\n<think>\n{reasoning}\n</think>\n\n{content}`. This is the 3.8
    // template's preserve_thinking default; 3.5/3.6 do not have it.
    REQUIRE(p.history_think_block);
    // A leading system turn with empty (trimmed) content is dropped -- the 3.8
    // template wraps that turn in `{%- if content %}`.
    REQUIRE(p.suppress_empty_system);
}
