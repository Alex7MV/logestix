// GLM-5.3-Flash converter suite. Part 1: frozen geometry + tensor
// classifier. Real-checkpoint tests SKIP without the model dir.
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <string>
#include <vector>

#include "glm_moe_flash_constants.h"

namespace fs = std::filesystem;

using namespace lsxcommon::glm_moe_flash;

TEST_CASE("frozen geometry passport matches GLM-5.3-Flash config")
{
    using namespace frozen;
    STATIC_REQUIRE(kHidden == 4096);
    STATIC_REQUIRE(kLayers == 45);
    STATIC_REQUIRE(kMtpLayer == 45);
    STATIC_REQUIRE(kHeads == 64);
    STATIC_REQUIRE(kHeadDim == 128);
    STATIC_REQUIRE(kQkHeadDim == 256);
    STATIC_REQUIRE(kQLora == 1536);
    STATIC_REQUIRE(kKvLora == 512);
    STATIC_REQUIRE(kNumExperts == 288);
    STATIC_REQUIRE(kTopK == 8);
    STATIC_REQUIRE(kMoeIntermediate == 2048);
    STATIC_REQUIRE(kDenseIntermediate == 12288);
    STATIC_REQUIRE(kHcCount == 4);
    STATIC_REQUIRE(kConvKernel == 4);
    STATIC_REQUIRE(kGateLowrank == 128);
    STATIC_REQUIRE(kIndexerHeads == 32);
    STATIC_REQUIRE(kIndexerTopk == 2048);
    STATIC_REQUIRE(kVocab == 154880);
    STATIC_REQUIRE(kFirstKDense == 3);
    // 34 KDA + 11 DSA == 45
    STATIC_REQUIRE(kKdaLayerCount == 34);
    STATIC_REQUIRE(kDsaLayerCount == 11);
    REQUIRE(IsKdaLayer(0));
    REQUIRE(IsKdaLayer(44));
    REQUIRE_FALSE(IsKdaLayer(3));
    REQUIRE(IsDsaLayer(43));
    REQUIRE_FALSE(IsDsaLayer(42));
    REQUIRE(IsMoeLayer(3));
    REQUIRE(IsMoeLayer(45));
    REQUIRE_FALSE(IsMoeLayer(2));
}

TEST_CASE("classify_flash_tensor: KDA layer tensors")
{
    auto p = classify_flash_tensor(
        "model.language_model.layers.0.self_attn.q_proj.weight");
    REQUIRE(p.kind == FlashTensorKind::KdaQ);
    REQUIRE(p.layer == 0);
    p = classify_flash_tensor(
        "model.language_model.layers.4.self_attn.k_conv1d.weight");
    REQUIRE(p.kind == FlashTensorKind::KdaConv);
    p = classify_flash_tensor(
        "model.language_model.layers.4.self_attn.dt_bias");
    REQUIRE(p.kind == FlashTensorKind::KdaDtBias);
    p = classify_flash_tensor(
        "model.language_model.layers.4.self_attn.b_proj.weight");
    REQUIRE(p.kind == FlashTensorKind::KdaB);
}

TEST_CASE("classify_flash_tensor: DSA layer tensors")
{
    auto p = classify_flash_tensor(
        "model.language_model.layers.3.self_attn.q_a_proj.weight");
    REQUIRE(p.kind == FlashTensorKind::DsaQa);
    p = classify_flash_tensor(
        "model.language_model.layers.3.self_attn.indexer.wq_b.weight");
    REQUIRE(p.kind == FlashTensorKind::Indexer);
    REQUIRE(p.layer == 3);
}

TEST_CASE("classify_flash_tensor: MoE + hc + global")
{
    auto p = classify_flash_tensor(
        "model.language_model.layers.5.mlp.experts.42.gate_proj.weight");
    REQUIRE(p.kind == FlashTensorKind::RoutedExpert);
    REQUIRE(p.layer == 5);
    REQUIRE(p.expert == 42);
    REQUIRE(p.proj_id == 1);
    p = classify_flash_tensor(
        "model.language_model.layers.5.mlp.gate.e_score_correction_bias");
    REQUIRE(p.kind == FlashTensorKind::RouterBias);
    p = classify_flash_tensor("model.language_model.layers.5.hc_attn_fn");
    REQUIRE(p.kind == FlashTensorKind::HcAttnFn);
    p = classify_flash_tensor("lm_head.weight");
    REQUIRE(p.kind == FlashTensorKind::LmHead);
    p = classify_flash_tensor("model.visual.patch_embed.proj.weight");
    REQUIRE(p.kind == FlashTensorKind::Visual);
    p = classify_flash_tensor("model.language_model.layers.45.eh_proj.weight");
    REQUIRE(p.kind == FlashTensorKind::MtpEhProj);
    p = classify_flash_tensor(
        "model.language_model.layers.45.self_attn.indexer.wq_b.weight");
    REQUIRE(p.kind == FlashTensorKind::Indexer);
    REQUIRE(p.layer == 45);
    p = classify_flash_tensor(
        "model.language_model.layers.45.mlp.experts.0.gate_proj.weight");
    REQUIRE(p.kind == FlashTensorKind::RoutedExpert);
    REQUIRE(p.layer == 45);
}

TEST_CASE("flash_tensor_is_nvfp4 policy")
{
    REQUIRE(flash_tensor_is_nvfp4(FlashTensorKind::RoutedExpert));
    REQUIRE(flash_tensor_is_nvfp4(FlashTensorKind::SharedExpert));
    REQUIRE(flash_tensor_is_nvfp4(FlashTensorKind::DsaQa));
    REQUIRE(flash_tensor_is_nvfp4(FlashTensorKind::DsaOBig));
    REQUIRE(flash_tensor_is_nvfp4(FlashTensorKind::DenseMlp));
    // Norms/biases/conv/hc/gates/embed stay high precision.
    REQUIRE_FALSE(flash_tensor_is_nvfp4(FlashTensorKind::KdaQ));
    REQUIRE_FALSE(flash_tensor_is_nvfp4(FlashTensorKind::KdaConv));
    REQUIRE_FALSE(flash_tensor_is_nvfp4(FlashTensorKind::HcAttnFn));
    REQUIRE_FALSE(flash_tensor_is_nvfp4(FlashTensorKind::RouterGate));
    REQUIRE_FALSE(flash_tensor_is_nvfp4(FlashTensorKind::LmHead));
    REQUIRE_FALSE(flash_tensor_is_nvfp4(FlashTensorKind::DsaKvB));
    REQUIRE_FALSE(flash_tensor_is_nvfp4(FlashTensorKind::Indexer));
}
