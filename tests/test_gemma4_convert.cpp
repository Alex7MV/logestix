// Тест Gemma-4 конвертера: классификатор имён тензоров + NVFP4-политика.
#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/gemma4_convert.h"

using lsxcommon::gemma::classify_gemma_tensor;
using K = lsxcommon::gemma::TensorKind;

TEST_CASE("gemma tensor classifier", "[gemma][convert]") {
    SECTION("globals") {
        CHECK(classify_gemma_tensor("model.language_model.embed_tokens.weight").kind == K::Embed);
        CHECK(classify_gemma_tensor("model.language_model.norm.weight").kind == K::FinalNorm);
    }
    SECTION("vision/audio fat skipped") {
        CHECK(classify_gemma_tensor("model.vision_tower.patch_embedder.input_proj.weight").kind == K::Visual);
        CHECK(classify_gemma_tensor("model.embed_vision.embedding_projection.weight").kind == K::Visual);
        CHECK(classify_gemma_tensor("model.vision_tower.encoder.layers.5.self_attn.q_proj.linear.weight").kind == K::Visual);
        CHECK(classify_gemma_tensor("model.vision_tower.std_scale").kind == K::Visual);
        CHECK(classify_gemma_tensor("model.vision_tower.std_bias").kind == K::Visual);
    }
    SECTION("attention") {
        auto q = classify_gemma_tensor("model.language_model.layers.3.self_attn.q_proj.weight");
        CHECK(q.kind == K::AttnQ);
        CHECK(q.layer == 3);
        CHECK(classify_gemma_tensor("model.language_model.layers.0.self_attn.k_norm.weight").kind == K::AttnKNorm);
        CHECK(classify_gemma_tensor("model.language_model.layers.0.self_attn.q_norm.weight").kind == K::AttnQNorm);
        CHECK(classify_gemma_tensor("model.language_model.layers.0.self_attn.o_proj.weight").kind == K::AttnO);
    }
    SECTION("dense mlp") {
        auto g = classify_gemma_tensor("model.language_model.layers.7.mlp.gate_proj.weight");
        CHECK(g.kind == K::MlpGate);
        CHECK(g.layer == 7);
        CHECK(classify_gemma_tensor("model.language_model.layers.7.mlp.up_proj.weight").kind == K::MlpUp);
        CHECK(classify_gemma_tensor("model.language_model.layers.7.mlp.down_proj.weight").kind == K::MlpDown);
    }
    SECTION("moe") {
        auto e = classify_gemma_tensor("model.language_model.layers.9.experts.gate_up_proj");
        CHECK(e.kind == K::ExpertsGateUp);
        CHECK(e.layer == 9);
        CHECK(classify_gemma_tensor("model.language_model.layers.9.experts.down_proj").kind == K::ExpertsDown);
        CHECK(classify_gemma_tensor("model.language_model.layers.9.router.proj.weight").kind == K::RouterProj);
        CHECK(classify_gemma_tensor("model.language_model.layers.9.router.scale").kind == K::RouterScale);
        CHECK(classify_gemma_tensor("model.language_model.layers.9.router.per_expert_scale").kind == K::RouterPerExpertScale);
    }
    SECTION("norms + scalar") {
        CHECK(classify_gemma_tensor("model.language_model.layers.1.input_layernorm.weight").kind == K::InputNorm);
        CHECK(classify_gemma_tensor("model.language_model.layers.1.post_attention_layernorm.weight").kind == K::PostNorm);
        CHECK(classify_gemma_tensor("model.language_model.layers.1.pre_feedforward_layernorm.weight").kind == K::PreFFN);
        CHECK(classify_gemma_tensor("model.language_model.layers.1.post_feedforward_layernorm.weight").kind == K::PostFFN);
        CHECK(classify_gemma_tensor("model.language_model.layers.1.pre_feedforward_layernorm_2.weight").kind == K::PreFFN2);
        CHECK(classify_gemma_tensor("model.language_model.layers.1.post_feedforward_layernorm_2.weight").kind == K::PostFFN2);
        CHECK(classify_gemma_tensor("model.language_model.layers.1.layer_scalar").kind == K::LayerScalar);
    }
    SECTION("unknown") {
        CHECK(classify_gemma_tensor("something.else.weight").kind == K::Unknown);
    }
}

TEST_CASE("nvfp4 policy: experts only", "[gemma][convert]") {
    using lsxcommon::gemma::gemma_tensor_is_nvfp4;
    CHECK(gemma_tensor_is_nvfp4(K::ExpertsGateUp));
    CHECK(gemma_tensor_is_nvfp4(K::ExpertsDown));
    CHECK_FALSE(gemma_tensor_is_nvfp4(K::AttnQ));
    CHECK_FALSE(gemma_tensor_is_nvfp4(K::Embed));
    CHECK_FALSE(gemma_tensor_is_nvfp4(K::FinalNorm));
    CHECK_FALSE(gemma_tensor_is_nvfp4(K::MlpGate));
    CHECK_FALSE(gemma_tensor_is_nvfp4(K::RouterProj));
}
