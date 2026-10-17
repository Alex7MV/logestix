// Gemma-4 converter test: tensor name classifier + NVFP4 policy.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <fstream>
#include <vector>

#include "test_gemma_moe_fixture.h"

#include <filesystem>

#include "lsxcommon/gemma_moe_convert.h"
#include "lsxcommon/model_integrity.h"
#include "lsxcommon/schema.h"

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

TEST_CASE("gemma topology parse + convert happy path", "[gemma][convert]") {
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "gemma4_conv_test";
    remove_all(tmp);
    create_directories(tmp);

    auto tensors = gemmatest::BuildSyntheticGemmaModel();
    gemmatest::WriteSafetensors(tmp, "model.safetensors", tensors);
    gemmatest::WriteConfig(tmp);
    std::filesystem::copy_file(
        TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
        tmp / "tokenizer.json");

    auto top = lsxcommon::gemma::load_gemma_topology((tmp / "config.json").string());
    REQUIRE(top.error_msg.empty());
    CHECK(top.num_layers == 4);
    CHECK(top.hidden_size == 64);
    CHECK(top.vocab_size == 64);
    CHECK(top.num_experts == 8);
    CHECK(top.top_k_experts == 2);
    CHECK(top.moe_intermediate_size == 32);  // 32-multiple: routed-GEMV K
    CHECK(top.num_global_key_value_heads == 1);
    CHECK(top.global_head_dim == 32);
    CHECK(top.sliding_window == 8);
    CHECK(top.bos_id == 2);
    // Full eos set from the top-level config array [1, 106] plus the sibling
    // generation_config's [1, 106, 50] (the model ends a turn on 106).
    CHECK(top.eos_ids == std::vector<int>({1, 106, 50}));
    CHECK(top.layer_types.size() == 4);

    auto out = tmp / "out";
    auto cst = lsxcommon::gemma::ConvertGemmaModel(tmp.string(), out.string());
    if (!cst.ok()) FAIL("ConvertGemmaModel: " << cst.ToString());
    REQUIRE(cst.ok());

    // GLM-style layout: pinned raw + per-expert NVFP4 files.
    CHECK(exists(out / "logestix" / "pinned" / "global.arrow"));
    CHECK(exists(out / "logestix" / "pinned" / "layer_0.arrow"));
    CHECK_FALSE(exists(out / "logestix" / "pinned" / "layer_0_nvfp4.arrow"));
    for (int l = 0; l < 4; ++l)
        for (int e = 0; e < 8; ++e)
            CHECK(exists(out / "logestix" / "experts" /
                         ("layer_" + std::to_string(l)) /
                         ("exp_" + std::to_string(e) + ".arrow")));

    // Index round-trip.
    auto top2 = lsxcommon::gemma::load_gemma_topology(
        (out / "logestix" / "model.index.json").string());
    // The index has no text_config nesting; parse must still succeed via
    // top-level keys written by WriteGemmaModelIndex.
    CHECK(top2.error_msg.empty());
    CHECK(top2.num_layers == 4);
    CHECK(top2.num_experts == 8);
    CHECK(top2.eos_ids == std::vector<int>({1, 106, 50}));
    CHECK(top2.vram.min_bytes > 0);
    CHECK(top2.total_pinned_tensors > 0);

    remove_all(tmp);
}

TEST_CASE("convert fails without config.json", "[gemma][convert]") {
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "gemma4_conv_noconf";
    remove_all(tmp);
    create_directories(tmp);

    auto tensors = gemmatest::BuildSyntheticGemmaModel();
    gemmatest::WriteSafetensors(tmp, "model.safetensors", tensors);
    std::filesystem::copy_file(
        TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
        tmp / "tokenizer.json");

    auto out = tmp / "out";
    auto st = lsxcommon::gemma::ConvertGemmaModel(tmp.string(), out.string());
    REQUIRE_FALSE(st.ok());
    remove_all(tmp);
}

TEST_CASE("convert fails on layer_types contradicting tensor shapes",
          "[gemma][convert]") {
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "gemma4_conv_badlt";
    remove_all(tmp);
    create_directories(tmp);

    auto tensors = gemmatest::BuildSyntheticGemmaModel();
    gemmatest::WriteSafetensors(tmp, "model.safetensors", tensors);
    gemmatest::WriteConfig(tmp);
    std::filesystem::copy_file(
        TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
        tmp / "tokenizer.json");

    // Flip layer 3 to sliding in the config while its q_proj keeps the
    // full-attention width -> geometry gate must reject.
    auto cfg = tmp / "config.json";
    std::ifstream in(cfg);
    std::string json((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
    in.close();
    auto pos = json.find("\"full_attention\"");
    REQUIRE(pos != std::string::npos);
    json.replace(pos, strlen("\"full_attention\""), "\"sliding_attention\"");
    { std::ofstream out(cfg, std::ios::trunc); out << json; }

    auto out = tmp / "out";
    auto st = lsxcommon::gemma::ConvertGemmaModel(tmp.string(), out.string());
    REQUIRE_FALSE(st.ok());
    remove_all(tmp);
}

TEST_CASE("gemma converter emits NVFP4 records for dense MLP",
          "[gemma][convert][nvfp4]") {
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "gemma4_conv_mlp_nvfp4";
    remove_all(tmp);
    create_directories(tmp);

    auto tensors = gemmatest::BuildSyntheticGemmaModel();
    gemmatest::WriteSafetensors(tmp, "model.safetensors", tensors);
    gemmatest::WriteConfig(tmp);
    std::filesystem::copy_file(
        TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
        tmp / "tokenizer.json");

    auto out = tmp / "out";
    auto cst = lsxcommon::gemma::ConvertGemmaModel(tmp.string(), out.string());
    if (!cst.ok()) FAIL("ConvertGemmaModel: " << cst.ToString());
    REQUIRE(cst.ok());

    auto batch_res = lsxcommon::LoadTensorBatchViews(
        out / "logestix" / "pinned" / "layer_0.arrow");
    REQUIRE(batch_res.ok());

    auto find = [&](std::string_view name) -> const lsxcommon::TensorView * {
        for (const auto &t : batch_res->tensors)
            if (t.name == name) return &t;
        return nullptr;
    };
    const auto &cfg = gemmatest::TestCfg();
    const size_t numel =
        static_cast<size_t>(cfg.intermediate_size) * cfg.hidden_size;

    const auto *pk = find("model.language_model.layers.0.mlp.gate_proj.weight.packed");
    const auto *sc = find("model.language_model.layers.0.mlp.gate_proj.weight.scales");
    const auto *gs = find("model.language_model.layers.0.mlp.gate_proj.weight.global_scale");
    REQUIRE(pk != nullptr);
    REQUIRE(sc != nullptr);
    REQUIRE(gs != nullptr);
    CHECK(pk->bytes == numel / 2);         // E2M1 nibble-packed, 2 elems/byte
    CHECK(sc->bytes == numel / 16);        // one E8M0-style scale byte per 16 elems
    CHECK(gs->bytes == 4);                 // F32 global scale
    // up/down triples exist too
    CHECK(find("model.language_model.layers.0.mlp.up_proj.weight.packed") != nullptr);
    CHECK(find("model.language_model.layers.0.mlp.down_proj.weight.packed") != nullptr);
    // the raw BF16 MLP tensors are gone
    CHECK(find("model.language_model.layers.0.mlp.gate_proj.weight") == nullptr);

    // Index advertises the mlp_nvfp4 flag.
    {
        std::ifstream idx(out / "logestix" / "model.index.json");
        std::string json((std::istreambuf_iterator<char>(idx)),
                         std::istreambuf_iterator<char>());
        CHECK(json.find("\"mlp_nvfp4\": true") != std::string::npos);
    }

    // Release every handle before the cleanup: Windows refuses to delete a
    // mapped or open file, while Linux happily unlinks both and hides the
    // ordering bug.
    batch_res->keepalive.reset();

    remove_all(tmp);
}
