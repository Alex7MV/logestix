// TDD — GigaChat3.5 converter (Model Ingestion Protocol step 1). Synthetic
// checkpoint only: the real model is never converted in a test. The ctest
// TIMEOUT is the mechanical guard against a real full conversion.
#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/gigachat_moe_convert.h"

#include <string>

using namespace lsxcommon::gigachat;

TEST_CASE("gigachat tensor classifier", "[gigachat][convert]")
{
    auto k = [](const char *n) { return classify_gigachat_tensor(n).kind; };
    CHECK(k("model.embed_tokens.weight") == GigaChatTensorKind::Embed);
    CHECK(k("lm_head.weight") == GigaChatTensorKind::LmHead);
    CHECK(k("model.norm.weight") == GigaChatTensorKind::FinalNormWeight);
    CHECK(k("model.norm.gate_up_projection.weight") ==
          GigaChatTensorKind::FinalNormGateUp);
    CHECK(k("model.norm.gate_down_projection.weight") ==
          GigaChatTensorKind::FinalNormGateDown);
    CHECK(k("model.layers.3.input_layernorm.weight") ==
          GigaChatTensorKind::LayerNormWeight);
    CHECK(k("model.layers.7.post_feedforward_layernorm.gate_down_projection.weight") ==
          GigaChatTensorKind::LayerNormGateDown);
    CHECK(k("model.layers.3.self_attn.q_a_proj.weight") ==
          GigaChatTensorKind::QaProj);
    CHECK(k("model.layers.3.self_attn.kv_a_proj_with_mqa.weight") ==
          GigaChatTensorKind::KvaProj);
    CHECK(k("model.layers.3.self_attn.gate_proj.weight") ==
          GigaChatTensorKind::AttnGateProj);
    CHECK(k("model.layers.3.self_attn.q_a_layernorm.weight") ==
          GigaChatTensorKind::LayerNormWeight);
    CHECK(k("model.layers.4.self_attn.in_proj_qkvz.weight") ==
          GigaChatTensorKind::GdnInProjQkvz);
    CHECK(k("model.layers.4.self_attn.in_proj_ba.weight") ==
          GigaChatTensorKind::GdnInProjBa);
    CHECK(k("model.layers.4.self_attn.conv1d.weight") ==
          GigaChatTensorKind::GdnConv1d);
    CHECK(k("model.layers.4.self_attn.A_log") == GigaChatTensorKind::GdnALog);
    CHECK(k("model.layers.4.self_attn.norm.weight") ==
          GigaChatTensorKind::GdnNormWeight);
    CHECK(k("model.layers.5.mlp.gate.weight") ==
          GigaChatTensorKind::RouterWeight);
    CHECK(k("model.layers.5.mlp.gate.e_score_correction_bias") ==
          GigaChatTensorKind::RouterBias);
    CHECK(k("model.layers.5.mlp.shared_experts.gate_proj.weight") ==
          GigaChatTensorKind::SharedGate);
    CHECK(k("model.layers.5.mlp.experts.200.up_proj.weight") ==
          GigaChatTensorKind::ExpertUp);
    CHECK(k("model.layers.0.mlp.down_proj.weight") ==
          GigaChatTensorKind::MlpDown);
    CHECK(k("model.layers.40.eh_proj.weight") == GigaChatTensorKind::Mtp);
    CHECK(k("model.layers.41.shared_head.head.weight") == GigaChatTensorKind::Mtp);
    CHECK(k("garbage") == GigaChatTensorKind::Unknown);

    auto e = classify_gigachat_tensor("model.layers.5.mlp.experts.200.up_proj.weight");
    CHECK(e.layer == 5);
    CHECK(e.expert == 200);
}
