// Mistral-Small-4-119B synthetic end-to-end converter test (Model Ingestion
// Protocol step 1). Runs the real converter pipeline on a synthetic checkpoint
// — no real weights are read (Converter test rules 1-5): label `convert`,
// bounded TIMEOUT (tests/CMakeLists.txt), LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY as a
// RUNTIME env var.
//
// What the fixture pins beyond the generic E2E:
//   - attention/router/norms/embed/head stay BF16 (the checkpoint's own
//     quantization ignore-list excludes them);
//   - routed + shared experts are ALREADY NVFP4 and are repacked verbatim
//     (packed nibbles + per-16 E4M3 scales + per-tensor F32 global scale);
//   - the vision tower is stripped and counted.

#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/mistral_moe_convert.h"
#include "mistral_moe_constants.h"
#include "test_mistral_moe_fixture.h"

#include <lsxcommon/schema.h>

#include <absl/strings/str_cat.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

using lsxcommon::mistral_moe::mistral_classify;
using K = lsxcommon::mistral_moe::MistralTensorKind;
namespace fs = std::filesystem;

TEST_CASE("Mistral geometry is compile-time locked", "[mistral][convert]")
{
    using namespace lsxcommon::mistral_moe::frozen;
    STATIC_REQUIRE(kHidden == 4096);
    STATIC_REQUIRE(kLayers == 36);
    STATIC_REQUIRE(kVocab == 131072);
    STATIC_REQUIRE(kHeads == 32);
    STATIC_REQUIRE(kKvHeads == 32);
    STATIC_REQUIRE(kQLoraRank == 1024);
    STATIC_REQUIRE(kKvLoraRank == 256);
    STATIC_REQUIRE(kQkNopeHeadDim == 64);
    STATIC_REQUIRE(kQkRopeHeadDim == 64);
    STATIC_REQUIRE(kQkHeadDim == 128);
    STATIC_REQUIRE(kVHeadDim == 128);
    STATIC_REQUIRE(kNumRouted == 128);
    STATIC_REQUIRE(kTopK == 4);
    STATIC_REQUIRE(kSharedExperts == 1);
    STATIC_REQUIRE(kMoeIntermediate == 2048);
    STATIC_REQUIRE(kRopeDim == 64);
    STATIC_REQUIRE(kEosId == 2);
    STATIC_REQUIRE(kPadId == 11);
}

TEST_CASE("Mistral classifier covers every tensor family", "[mistral][convert]")
{
    // Attention projections.
    auto t = mistral_classify("layers.7.attention.wq_a.weight");
    REQUIRE(t.kind == K::Attention);
    REQUIRE(t.layer == 7);
    REQUIRE(t.base == "wq_a");
    REQUIRE(mistral_classify("layers.7.attention.wq_b.weight").base == "wq_b");
    REQUIRE(mistral_classify("layers.7.attention.wkv_a_with_mqa.weight").base ==
            "wkv_a_with_mqa");
    REQUIRE(mistral_classify("layers.7.attention.wkv_b.weight").base == "wkv_b");
    REQUIRE(mistral_classify("layers.7.attention.wo.weight").base == "wo");

    // Norms.
    REQUIRE(mistral_classify("layers.7.attention_norm.weight").kind ==
            K::LayerNorm);
    REQUIRE(mistral_classify("layers.7.ffn_norm.weight").kind == K::LayerNorm);
    REQUIRE(mistral_classify("layers.7.attention.q_a_norm.weight").kind ==
            K::LayerNorm);
    REQUIRE(mistral_classify("layers.7.attention.kv_a_norm.weight").kind ==
            K::LayerNorm);

    // Router.
    REQUIRE(mistral_classify("layers.5.gate.weight").kind == K::Router);

    // Routed experts: weight + companions.
    t = mistral_classify("layers.5.experts.42.w1.weight_packed");
    REQUIRE(t.kind == K::RoutedExpert);
    REQUIRE(t.expert == 42);
    REQUIRE(t.weight_id == 1);
    REQUIRE(t.companion.empty());
    REQUIRE(mistral_classify("layers.5.experts.42.w1.weight_scale").companion ==
            "scale");
    REQUIRE(mistral_classify("layers.5.experts.42.w2.weight_global_scale")
                .companion == "global_scale");
    REQUIRE(mistral_classify("layers.5.experts.42.w3.input_global_scale")
                .companion == "input_global_scale");
    REQUIRE(mistral_classify("layers.5.experts.42.w2.weight_packed").weight_id == 2);
    REQUIRE(mistral_classify("layers.5.experts.42.w3.weight_packed").weight_id == 3);

    // Shared expert.
    REQUIRE(mistral_classify("layers.5.shared_experts.w1.weight_packed").kind ==
            K::SharedExpert);
    REQUIRE(mistral_classify("layers.5.shared_experts.w2.weight_scale").kind ==
            K::SharedExpert);

    // Globals.
    REQUIRE(mistral_classify("tok_embeddings.weight").kind == K::Global);
    REQUIRE(mistral_classify("output.weight").kind == K::Global);
    REQUIRE(mistral_classify("norm.weight").kind == K::Global);

    // Vision ballast.
    REQUIRE(mistral_classify("vision_encoder.patch_conv.weight").kind == K::Vision);
    REQUIRE(mistral_classify("vision_language_adapter.w_in.weight").kind == K::Vision);
    REQUIRE(mistral_classify("patch_merger.merging_layer.weight").kind == K::Vision);
    REQUIRE(mistral_classify("pre_mm_projector_norm.weight").kind == K::Vision);
}

TEST_CASE("Mistral synthetic convert emits logestix-mistral-nvfp4",
          "[mistral][convert]")
{
    auto in = fs::temp_directory_path() / "mistral_conv_fixture";
    auto out = fs::temp_directory_path() / "mistral_conv_out";
    fs::remove_all(in);
    fs::remove_all(out);
    fs::create_directories(in);

    mistraltest::WriteMistralParams(in);
    synthtest::WriteSafetensors(in, "consolidated-00001-of-00001.safetensors",
                                mistraltest::BuildSyntheticMistralModel());

    const auto st = lsxcommon::mistral_moe::ConvertMistralModel(in.string(), out);
    INFO("converter status: " << st.ToString());
    REQUIRE(st.ok());

    const auto log_root = out / "logestix";
    REQUIRE(fs::exists(log_root / "model.index.json"));
    REQUIRE(fs::exists(log_root / "pinned" / "global.arrow"));
    REQUIRE(fs::exists(log_root / "pinned" / "shared_experts.arrow"));
    for (int l = 0; l < 2; ++l)
        REQUIRE(fs::exists(log_root / "pinned" /
                           absl::StrCat("layer_", l, ".arrow")));
    for (int e = 0; e < 2; ++e)
        REQUIRE(fs::exists(log_root / "experts" / "layer_0" /
                           absl::StrCat("exp_", e, ".arrow")));

    std::ifstream f(log_root / "model.index.json");
    const std::string json((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
    REQUIRE(json.find("\"logestix-mistral-nvfp4\"") != std::string::npos);
    REQUIRE(json.find("\"moe_layers\": [0,1]") != std::string::npos);
    REQUIRE(json.find("\"dense_layers\": []") != std::string::npos);
    // Vision ballast counted, not silently dropped.
    REQUIRE(json.find("vision_encoder.patch_conv.weight") != std::string::npos);
}

TEST_CASE("Mistral converter passes already-quantized experts through verbatim",
          "[mistral][convert]")
{
    auto in = fs::temp_directory_path() / "mistral_conv_ex_fixture";
    auto out = fs::temp_directory_path() / "mistral_conv_ex_out";
    fs::remove_all(in);
    fs::remove_all(out);
    fs::create_directories(in);

    mistraltest::WriteMistralParams(in);
    synthtest::WriteSafetensors(in, "consolidated-00001-of-00001.safetensors",
                                mistraltest::BuildSyntheticMistralModel());

    const auto st = lsxcommon::mistral_moe::ConvertMistralModel(in.string(), out);
    INFO("converter status: " << st.ToString());
    REQUIRE(st.ok());

    const auto log_root = out / "logestix";
    auto batch = lsxcommon::ReadRecordBatchFromFile(
        log_root / "experts" / "layer_0" / "exp_0.arrow");
    REQUIRE(batch.ok());
    auto pe = lsxcommon::ExpertSchema::UnpackBatch(*batch);
    REQUIRE(pe.ok());
    // Already-quantized expert passthrough: nibbles + per-16 scales survive.
    mistraltest::MistralSynthGeom g;
    REQUIRE(pe->w1_packed.size() ==
            static_cast<size_t>(g.expert_hidden * g.hidden / 2));
    REQUIRE(pe->w1_scales.size() ==
            static_cast<size_t>(g.expert_hidden * g.hidden / 16));
    REQUIRE(pe->w3_packed == pe->w1_packed);
    REQUIRE(pe->w3_scales == pe->w1_scales);
    REQUIRE(pe->w2_packed.size() ==
            static_cast<size_t>(g.hidden * g.expert_hidden / 2));
    REQUIRE(pe->w2_scales.size() ==
            static_cast<size_t>(g.hidden * g.expert_hidden / 16));

    // Attention stays BF16 in the per-layer raw batch.
    auto raw = lsxcommon::LoadTensorBatchViews(log_root / "pinned" / "layer_0.arrow");
    REQUIRE(raw.ok());
    bool found_attn = false;
    for (const auto &v : raw->tensors)
        if (v.name == "layers.0.attention.wq_a.weight")
        {
            found_attn = true;
            REQUIRE(v.dtype == "BF16");
        }
    REQUIRE(found_attn);
}
