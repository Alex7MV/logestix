#include <catch2/catch_test_macros.hpp>
#include "lsxcommon/kimi_moe_load.h"
#include "lsxcommon/kimi_moe_convert.h"
#include "lsxcommon/model_integrity.h"
#include "lsxcommon/path_registry.h"

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
    REQUIRE(t.weight_id == 2);

    t = kimi_classify("language_model.model.layers.7.mlp.experts.42.up_proj.weight_packed");
    REQUIRE(t.kind == K::RoutedExpert);
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

#include "test_kimi_moe_fixture.h"

#include <absl/strings/str_cat.h>

#include <filesystem>
#include <fstream>

TEST_CASE("ConvertKimiModel produces logestix-kimi layout", "[kimi][convert]") {
    namespace fs = std::filesystem;
    auto in = fs::temp_directory_path() / "kimi_conv_fixture";
    auto out = fs::temp_directory_path() / "kimi_conv_out";
    fs::remove_all(in);
    fs::remove_all(out);
    fs::create_directories(in);

    kimitest::WriteKimiConfig(in);
    kimitest::WriteSafetensors(in, "model-00001-of-00001.safetensors",
                               kimitest::BuildSyntheticKimiModel(3));
    kimitest::WriteKimiTokenizerFiles(in);

    REQUIRE(lsxcommon::kimi::ConvertKimiModel(in.string(), out).ok());

    const auto log_root = out / "logestix";
    REQUIRE(fs::exists(log_root / "model.index.json"));
    REQUIRE(fs::exists(log_root / "pinned" / "global.arrow"));
    // INT4 routed experts packed to per-expert arrow files.
    for (int l = 1; l <= 2; ++l)
        for (int e = 0; e < 2; ++e)
            REQUIRE(fs::exists(log_root / "experts"
                               / absl::StrCat("layer_", l)
                               / absl::StrCat("exp_", e, ".arrow")));

    std::ifstream f(log_root / "model.index.json");
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    REQUIRE(json.find("\"logestix-kimi-r2\"") != std::string::npos);
    REQUIRE(json.find("\"moe_layers\": [1,2]") != std::string::npos);
    REQUIRE(json.find("\"dense_layers\": [0]") != std::string::npos);

    // Pinned layers: dense layer 0 and MoE layers 1-2 (attention/router/norm).
    REQUIRE(fs::exists(log_root / "pinned" / "layer_0.arrow"));
    REQUIRE(fs::exists(log_root / "pinned" / "layer_1.arrow"));
    REQUIRE(fs::exists(log_root / "pinned" / "layer_2.arrow"));
}

#include <cstdlib>

TEST_CASE("Kimi vault conversion passes integrity verification",
          "[kimi][convert][real-model]")
{
    const char *env = std::getenv("LSX_TEST_KIMI_MODEL_PATH");
    auto dir = env ? std::filesystem::path(env)
                   : std::filesystem::path(
                         lsxcommon::lsx_model_models_dir()) /
                         "Kimi-K2.7-Code";
    if (!std::filesystem::exists(dir / "logestix" / "model.index.json"))
        SKIP("converted Kimi model not present");

    auto vres = lsxcommon::integrity::VerifyConvertedModel(dir);
    if (!vres.ok()) FAIL(vres.status().ToString());
    REQUIRE(vres.ok());
}

// Round-2 Task 3: MLA projections are quantized to NVFP4 (packed + scale
// records + global_scale in the index); layernorms stay BF16; format rev
// bumps to logestix-kimi-r2.
TEST_CASE("Kimi converter r2: MLA projections quantized, norms untouched",
          "[kimi][convert][r2]")
{
    namespace fs = std::filesystem;
    auto in = fs::temp_directory_path() / "kimi_conv_r2_fixture";
    auto out = fs::temp_directory_path() / "kimi_conv_r2_out";
    fs::remove_all(in);
    fs::remove_all(out);
    fs::create_directories(in);

    kimitest::WriteKimiConfig(in);
    kimitest::WriteSafetensors(in, "model-00001-of-00001.safetensors",
                               kimitest::BuildSyntheticKimiModel(3));
    kimitest::WriteKimiTokenizerFiles(in);

    REQUIRE(lsxcommon::kimi::ConvertKimiModel(in.string(), out).ok());

    const auto log_root = out / "logestix";
    std::ifstream f(log_root / "model.index.json");
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    REQUIRE(json.find("\"logestix-kimi-r2\"") != std::string::npos);
    // mla_nvfp4 block carries per-layer global scales.
    REQUIRE(json.find("\"mla_nvfp4\"") != std::string::npos);
    REQUIRE(json.find("\"q_a_proj\"") != std::string::npos);
    REQUIRE(json.find("\"o_proj\"") != std::string::npos);

    // Load the produced layer file and verify record shapes/names.
    auto tensors = lsxcommon::LoadTensorBatchViews(
        log_root / "pinned" / "layer_1.arrow");
    REQUIRE(tensors.ok());
    const auto &views = *tensors;
    auto find = [&](std::string_view suffix) -> const lsxcommon::TensorView * {
        for (const auto &t : views.tensors) {
            if (t.name.size() >= suffix.size() &&
                t.name.compare(t.name.size() - suffix.size(), suffix.size(),
                               suffix) == 0)
                return &t;
        }
        return nullptr;
    };
    // q_a packed/scale pair exists.
    const auto *qa_pk = find("self_attn.q_a_proj.weight_packed");
    const auto *qa_sc = find("self_attn.q_a_proj.weight_scale");
    REQUIRE(qa_pk != nullptr);
    REQUIRE(qa_sc != nullptr);
    // BF16 [16,16] = 256 elems -> 128 packed bytes + 16 scale bytes.
    REQUIRE(qa_pk->bytes == 128);
    REQUIRE(qa_sc->bytes == 16);
    // No raw BF16 q_a projection remains.
    REQUIRE(find("self_attn.q_a_proj.weight") == nullptr);
    // o_proj pair exists.
    REQUIRE(find("self_attn.o_proj.weight_packed") != nullptr);
    REQUIRE(find("self_attn.o_proj.weight_scale") != nullptr);
    // Layernorms remain BF16 single records.
    const auto *qa_ln = find("self_attn.q_a_layernorm.weight");
    REQUIRE(qa_ln != nullptr);
    REQUIRE(qa_ln->bytes == 32);   // 16 BF16 elems
    REQUIRE(qa_ln->dtype == "BF16");
}
