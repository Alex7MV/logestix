#include <catch2/catch_test_macros.hpp>
#include "lsxcommon/kimi_inference_load.h"
#include "lsxcommon/kimi_convert.h"

using lsxcommon::kimi::kimi_classify;
using K = lsxcommon::kimi::KimiTensorKind;


TEST_CASE("Kimi geometry is compile-time locked", "[kimi][convert]") {
    using namespace lsxcommon::kimi;
    STATIC_REQUIRE(kKimiHidden == 7168);
    STATIC_REQUIRE(kKimiNumLayers == 61);
    STATIC_REQUIRE(kKimiNumHeads == 64);
    STATIC_REQUIRE(kKimiKvLoraRank == 512);
    STATIC_REQUIRE(kKimiQLoraRank == 1536);
    STATIC_REQUIRE(kKimiQkNope == 128);
    STATIC_REQUIRE(kKimiQkRope == 64);
    STATIC_REQUIRE(kKimiHeadV == 128);
    STATIC_REQUIRE(kKimiMoeIntermediate == 2048);
    STATIC_REQUIRE(kKimiDenseIntermediate == 18432);
    STATIC_REQUIRE(kKimiNumRouted == 384);
    STATIC_REQUIRE(kKimiNumShared == 1);
    STATIC_REQUIRE(kKimiTopK == 8);
    STATIC_REQUIRE(kKimiVocab == 163840);
    STATIC_REQUIRE(kKimiFirstKDenseReplace == 1);
}

TEST_CASE("Kimi classifier strips language_model prefix", "[kimi][convert]") {
    auto t = kimi_classify("language_model.model.layers.3.self_attn.q_a_proj.weight");
    REQUIRE(t.kind == K::Attention);
    REQUIRE(t.layer == 3);
    REQUIRE(t.base == "q_a_proj");

    t = kimi_classify("language_model.model.layers.7.mlp.experts.42.gate_proj.weight_packed");
    REQUIRE(t.kind == K::RoutedExpert);
    REQUIRE(t.layer == 7);
    REQUIRE(t.expert == 42);
    REQUIRE(t.weight_id == 1);

    t = kimi_classify("language_model.model.layers.7.mlp.experts.42.down_proj.weight_scale");
    REQUIRE(t.kind == K::RoutedExpert);
    REQUIRE(t.expert == 42);
    REQUIRE(t.weight_id == 3);

    REQUIRE(kimi_classify("vision_tower.encoder.layers.0.attn.qkv.weight").kind == K::Unknown);
    REQUIRE(kimi_classify("mm_projector.proj.0.weight").kind == K::Unknown);
}

TEST_CASE("Kimi classifier covers every tensor family", "[kimi][convert]") {
    REQUIRE(kimi_classify("language_model.model.embed_tokens.weight").kind == K::Global);
    REQUIRE(kimi_classify("language_model.lm_head.weight").kind == K::Global);
    REQUIRE(kimi_classify("language_model.model.norm.weight").kind == K::Global);
    REQUIRE(kimi_classify("language_model.model.layers.0.mlp.gate_proj.weight").kind == K::DenseMlp);
    REQUIRE(kimi_classify("language_model.model.layers.0.mlp.down_proj.weight_packed").kind == K::DenseMlp);
    REQUIRE(kimi_classify("language_model.model.layers.5.mlp.gate.weight").kind == K::Router);
    REQUIRE(kimi_classify("language_model.model.layers.5.mlp.gate.e_score_correction_bias").kind == K::Router);
    REQUIRE(kimi_classify("language_model.model.layers.5.mlp.shared_experts.up_proj.weight_scale").kind == K::SharedExpert);
    REQUIRE(kimi_classify("language_model.model.layers.5.input_layernorm.weight").kind == K::LayerNorm);
    REQUIRE(kimi_classify("language_model.model.layers.5.post_attention_layernorm.weight").kind == K::LayerNorm);
}
