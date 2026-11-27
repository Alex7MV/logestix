// Muse-Glimmer-30B converter test: tensor classifier + NVFP4 policy +
// topology parse (Model Ingestion Protocol step 1). The synthetic E2E
// pipeline cases land with the converter itself.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "lsxcommon/muse_dense_constants.h"
#include "lsxcommon/muse_dense_convert.h"
#include "lsxcommon/model_integrity.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/tokenizer.h"

#include "test_muse_dense_fixture.h"

using lsxcommon::muse_dense::classify_muse_dense_tensor;
using lsxcommon::muse_dense::muse_dense_tensor_is_nvfp4;
using K = lsxcommon::muse_dense::TensorKind;

TEST_CASE("muse dense tensor classifier", "[muse_dense][convert]") {
    SECTION("globals") {
        CHECK(classify_muse_dense_tensor(
                  "model.language_model.embed_tokens.weight").kind == K::Embed);
        CHECK(classify_muse_dense_tensor("lm_head.weight").kind == K::LmHead);
        CHECK(classify_muse_dense_tensor(
                  "model.language_model.norm.weight").kind == K::FinalNorm);
    }
    SECTION("vision ballast is counted, never applied") {
        CHECK(classify_muse_dense_tensor(
                  "model.vision_tower.layers.0.attn.q_proj.weight").kind ==
              K::Visual);
        CHECK(classify_muse_dense_tensor(
                  "model.vision_tower.patch_embedder.patch_embedding.weight")
                  .kind == K::Visual);
        CHECK(classify_muse_dense_tensor(
                  "model.vision_adapter.fc1.weight").kind == K::Visual);
        CHECK(classify_muse_dense_tensor(
                  "model.vision_projection.weight").kind == K::Visual);
    }
    SECTION("attention projections incl. the output gate") {
        const auto q = classify_muse_dense_tensor(
            "model.language_model.layers.3.self_attn.q_proj.weight");
        CHECK(q.kind == K::AttnQ);
        CHECK(q.layer == 3);
        CHECK(classify_muse_dense_tensor(
                  "model.language_model.layers.0.self_attn.k_proj.weight").kind ==
              K::AttnK);
        CHECK(classify_muse_dense_tensor(
                  "model.language_model.layers.0.self_attn.v_proj.weight").kind ==
              K::AttnV);
        CHECK(classify_muse_dense_tensor(
                  "model.language_model.layers.0.self_attn.o_proj.weight").kind ==
              K::AttnO);
        // self_attn.gate_proj is the ATTENTION output gate, not the MLP gate.
        const auto g = classify_muse_dense_tensor(
            "model.language_model.layers.51.self_attn.gate_proj.weight");
        CHECK(g.kind == K::AttnGate);
        CHECK(g.layer == 51);
    }
    SECTION("dense MLP - no experts, no router") {
        const auto g = classify_muse_dense_tensor(
            "model.language_model.layers.7.mlp.gate_proj.weight");
        CHECK(g.kind == K::MlpGate);
        CHECK(g.layer == 7);
        CHECK(classify_muse_dense_tensor(
                  "model.language_model.layers.7.mlp.up_proj.weight").kind ==
              K::MlpUp);
        CHECK(classify_muse_dense_tensor(
                  "model.language_model.layers.7.mlp.down_proj.weight").kind ==
              K::MlpDown);
        CHECK(classify_muse_dense_tensor(
                  "model.language_model.layers.9.experts.gate_up_proj").kind ==
              K::Unknown);
        CHECK(classify_muse_dense_tensor(
                  "model.language_model.layers.9.router.proj.weight").kind ==
              K::Unknown);
    }
    SECTION("four centered norms per layer") {
        CHECK(classify_muse_dense_tensor(
                  "model.language_model.layers.1.input_layernorm.weight").kind ==
              K::InputNorm);
        CHECK(classify_muse_dense_tensor(
                  "model.language_model.layers.1.post_attention_layernorm."
                  "weight").kind == K::PostAttnNorm);
        CHECK(classify_muse_dense_tensor(
                  "model.language_model.layers.1.pre_feedforward_layernorm."
                  "weight").kind == K::PreFfnNorm);
        CHECK(classify_muse_dense_tensor(
                  "model.language_model.layers.1.post_feedforward_layernorm."
                  "weight").kind == K::PostFfnNorm);
    }
    SECTION("unknown tensors are never silently skipped") {
        CHECK(classify_muse_dense_tensor("something.else.weight").kind ==
              K::Unknown);
        // gemma-style per-head q/k norm weights do not exist here (the QK
        // norm is weightless); their presence means a wrong checkpoint.
        CHECK(classify_muse_dense_tensor(
                  "model.language_model.layers.1.self_attn.q_norm.weight").kind ==
              K::Unknown);
        CHECK(classify_muse_dense_tensor(
                  "model.language_model.layers.1.layer_scalar").kind ==
              K::Unknown);
    }
}

TEST_CASE("muse dense nvfp4 policy: the 8 projections only",
          "[muse_dense][convert]") {
    CHECK(muse_dense_tensor_is_nvfp4(K::AttnQ));
    CHECK(muse_dense_tensor_is_nvfp4(K::AttnK));
    CHECK(muse_dense_tensor_is_nvfp4(K::AttnV));
    CHECK(muse_dense_tensor_is_nvfp4(K::AttnO));
    CHECK(muse_dense_tensor_is_nvfp4(K::AttnGate));
    CHECK(muse_dense_tensor_is_nvfp4(K::MlpGate));
    CHECK(muse_dense_tensor_is_nvfp4(K::MlpUp));
    CHECK(muse_dense_tensor_is_nvfp4(K::MlpDown));
    // embed + lm_head stay FP16 (untied head, exact greedy); norms never
    // quantized.
    CHECK_FALSE(muse_dense_tensor_is_nvfp4(K::Embed));
    CHECK_FALSE(muse_dense_tensor_is_nvfp4(K::LmHead));
    CHECK_FALSE(muse_dense_tensor_is_nvfp4(K::FinalNorm));
    CHECK_FALSE(muse_dense_tensor_is_nvfp4(K::InputNorm));
    CHECK_FALSE(muse_dense_tensor_is_nvfp4(K::PostAttnNorm));
    CHECK_FALSE(muse_dense_tensor_is_nvfp4(K::PreFfnNorm));
    CHECK_FALSE(muse_dense_tensor_is_nvfp4(K::PostFfnNorm));
    CHECK_FALSE(muse_dense_tensor_is_nvfp4(K::Visual));
}

TEST_CASE("muse dense topology parse: synthetic config", "[muse_dense][convert]") {
    namespace fs = std::filesystem;
    const auto tmp = fs::temp_directory_path() / "muse_dense_topo";
    fs::remove_all(tmp);
    fs::create_directories(tmp);
    {
        std::ofstream f(tmp / "config.json");
        f << R"({"model_type":"muse_glimmer","text_config":{)"
          << R"("hidden_size":64,"num_hidden_layers":4,"intermediate_size":32,)"
          << R"("vocab_size":128,"num_attention_heads":4,"num_key_value_heads":2,)"
          << R"("head_dim":16,"sliding_window":8,"rms_norm_eps":1e-5,)"
          << R"("post_norm_eps":1e-8,"qk_scale_factor":3.87,)"
          << R"("output_multiplier":0.5,"final_logit_softcapping":20.0,)"
          << R"("rope_parameters":{"rope_theta":500000.0,"rope_type":"default"},)"
          << R"("layer_rope_theta":[500000.0,500000.0,500000.0,0],)"
          << R"("layer_types":["sliding_attention","sliding_attention",)"
          << R"("sliding_attention","full_attention"],)"
          << R"("bos_token_id":100,"eos_token_id":101,"pad_token_id":null,)"
          << R"("tie_word_embeddings":false}})";
    }
    {
        std::ofstream f(tmp / "generation_config.json");
        f << R"({"bos_token_id":100,"eos_token_id":[101,108],"pad_token_id":118})";
    }

    auto top = lsxcommon::muse_dense::load_muse_dense_topology(
        (tmp / "config.json").string());
    REQUIRE(top.error_msg.empty());
    CHECK(top.num_layers == 4);
    CHECK(top.hidden_size == 64);
    CHECK(top.intermediate_size == 32);
    CHECK(top.vocab_size == 128);
    CHECK(top.num_attention_heads == 4);
    CHECK(top.num_key_value_heads == 2);
    CHECK(top.head_dim == 16);
    CHECK(top.sliding_window == 8);
    CHECK(top.rms_norm_eps == Catch::Approx(1e-5f));
    CHECK(top.post_norm_eps == Catch::Approx(1e-8f));
    CHECK(top.qk_scale_factor == Catch::Approx(3.87f));
    CHECK(top.output_multiplier == Catch::Approx(0.5f));
    CHECK(top.final_logit_softcapping == Catch::Approx(20.0f));
    CHECK(top.rope_theta == Catch::Approx(500000.0f));
    CHECK(top.bos_id == 100);
    // generation_config adds the turn-end eos and supplies the pad id.
    CHECK(top.eos_ids == std::vector<int>({101, 108}));
    CHECK(top.pad_id == 118);
    REQUIRE(top.layer_types.size() == 4);
    CHECK_FALSE(top.IsFull(0));
    CHECK(top.IsFull(3));
    CHECK(top.Window(0) == 8);
    CHECK(top.Window(3) == 0);
    CHECK(top.RopeTheta(0) == 500000.0);
    CHECK(top.RopeTheta(3) == 0.0);   // NoPE

    fs::remove_all(tmp);
}

TEST_CASE("muse dense topology parse: missing file errors",
          "[muse_dense][convert]") {
    auto top = lsxcommon::muse_dense::load_muse_dense_topology(
        "/nonexistent/muse/config.json");
    CHECK_FALSE(top.error_msg.empty());
}

TEST_CASE("muse dense topology parse: real checkpoint config",
          "[muse_dense][convert]") {
    namespace fs = std::filesystem;
    std::string root;
    if (const char *env = std::getenv("LSX_TEST_MUSE_GLIMMER_30B_ROOT");
        env && *env)
        root = env;
    else
        root = "/mnt/fast_storage/lsx_model/meta-models/Muse-Glimmer-30B";
    if (!fs::exists(fs::path(root) / "config.json"))
        SKIP("Muse-Glimmer-30B source checkpoint not present");

    auto top = lsxcommon::muse_dense::load_muse_dense_topology(
        (fs::path(root) / "config.json").string());
    REQUIRE(top.error_msg.empty());
    namespace fz = lsxcommon::muse_dense::frozen;
    CHECK(top.num_layers == fz::kNumLayers);
    CHECK(top.hidden_size == fz::kHiddenSize);
    CHECK(top.intermediate_size == fz::kIntermediateSize);
    CHECK(top.vocab_size == fz::kVocabSize);
    CHECK(top.num_attention_heads == fz::kNumHeads);
    CHECK(top.num_key_value_heads == fz::kNumKVHeads);
    CHECK(top.head_dim == fz::kHeadDim);
    CHECK(top.sliding_window == fz::kSlidingWindow);
    CHECK(top.rms_norm_eps == Catch::Approx(fz::kRmsNormEps));
    CHECK(top.post_norm_eps == Catch::Approx(fz::kPostNormEps));
    CHECK(top.qk_scale_factor == Catch::Approx(fz::kQkScaleFactor));
    CHECK(top.output_multiplier == Catch::Approx(fz::kOutputMultiplier));
    CHECK(top.final_logit_softcapping == Catch::Approx(fz::kFinalLogitSoftcap));
    CHECK(top.rope_theta == Catch::Approx(fz::kRopeTheta));
    CHECK(top.bos_id == fz::kBosId);
    CHECK(top.pad_id == fz::kPadId);
    CHECK(top.eos_ids == std::vector<int>({fz::kEosIdText, fz::kEosIdTurn}));
    REQUIRE(top.layer_types.size() == static_cast<size_t>(fz::kNumLayers));
    for (int l = 0; l < fz::kNumLayers; ++l) {
        CHECK(top.IsFull(l) == fz::IsFullLayer(l));
        CHECK(top.RopeTheta(l) == fz::RopeThetaForLayer(l));
    }
}
TEST_CASE("muse dense convert happy path", "[muse_dense][convert]") {
    namespace fs = std::filesystem;
    const auto tmp = fs::temp_directory_path() / "muse_dense_conv";
    fs::remove_all(tmp);
    fs::create_directories(tmp);

    synthtest::WriteSafetensors(tmp, "model.safetensors",
                                muse_dense_test::BuildSyntheticMuseDenseModel());
    muse_dense_test::WriteConfig(tmp);
    muse_dense_test::WriteTokenizerJson(tmp);

    const auto out = tmp / "out";
    auto st = lsxcommon::muse_dense::ConvertMuseDenseModel(tmp.string(),
                                                           out.string());
    if (!st.ok()) FAIL("ConvertMuseDenseModel: " << st.ToString());
    REQUIRE(st.ok());

    const auto pinned = out / "pinned";
    SECTION("index: format, geometry, ballast count") {
        std::ifstream f(out / "model.index.json");
        REQUIRE(f.good());
        const std::string json((std::istreambuf_iterator<char>(f)),
                               std::istreambuf_iterator<char>());
        CHECK(json.find("\"format\": \"logestix-muse-nvfp4\"") !=
              std::string::npos);
        CHECK(json.find("\"model\": \"muse-glimmer-30b\"") != std::string::npos);
        CHECK(json.find("\"skipped_visual_tensors\": 4") != std::string::npos);
        CHECK(json.find("\"head_fp16\": true") != std::string::npos);
        CHECK(json.find("\"vram\"") != std::string::npos);
        // The loader re-reads the topology from the index.
        auto top = lsxcommon::muse_dense::load_muse_dense_topology(
            (out / "model.index.json").string());
        REQUIRE(top.error_msg.empty());
        CHECK(top.num_layers == 4);
        CHECK(top.hidden_size == 64);
        CHECK(top.eos_ids == std::vector<int>({260, 261}));
        CHECK(top.pad_id == 263);
        CHECK(top.IsFull(3));
        CHECK(top.RopeTheta(3) == 0.0);
    }
    SECTION("globals: raw BF16 embed + untied lm_head + final norm") {
        const auto names = synthtest::TensorNamesIn(pinned / "global.arrow");
        CHECK(std::find(names.begin(), names.end(),
                        "model.language_model.embed_tokens.weight") !=
              names.end());
        CHECK(std::find(names.begin(), names.end(), "lm_head.weight") !=
              names.end());
        CHECK(std::find(names.begin(), names.end(),
                        "model.language_model.norm.weight") != names.end());
        // No NVFP4 globals: embed/head stay high precision in this family.
        CHECK_FALSE(fs::exists(pinned / "global_nvfp4.arrow"));
    }
    SECTION("layer: 8 NVFP4 triples + 4 raw norms") {
        const auto names = synthtest::TensorNamesIn(pinned / "layer_0.arrow");
        REQUIRE(names.size() == 8 * 3 + 4);
        for (const char *proj : {"self_attn.q_proj.weight",
                                 "self_attn.k_proj.weight",
                                 "self_attn.v_proj.weight",
                                 "self_attn.o_proj.weight",
                                 "self_attn.gate_proj.weight",
                                 "mlp.gate_proj.weight", "mlp.up_proj.weight",
                                 "mlp.down_proj.weight"}) {
            const std::string base =
                std::string("model.language_model.layers.0.") + proj;
            for (const char *suf : {".packed", ".scales", ".global_scale"}) {
                INFO(base + suf);
                CHECK(std::find(names.begin(), names.end(), base + suf) !=
                      names.end());
            }
        }
        for (const char *norm : {"input_layernorm.weight",
                                 "post_attention_layernorm.weight",
                                 "pre_feedforward_layernorm.weight",
                                 "post_feedforward_layernorm.weight"}) {
            const std::string n =
                std::string("model.language_model.layers.0.") + norm;
            INFO(n);
            CHECK(std::find(names.begin(), names.end(), n) != names.end());
        }
        // All four layers exist (the full layer 3 has the SAME tensor set).
        CHECK(fs::exists(pinned / "layer_3.arrow"));
        CHECK(synthtest::TensorNamesIn(pinned / "layer_3.arrow").size() ==
              8 * 3 + 4);
    }
    SECTION("tokenizer bridge: kind 3 with the ATEM specials") {
        REQUIRE(fs::exists(pinned / "tokenizer_vocab.arrow"));
        REQUIRE(fs::exists(pinned / "tokenizer_merges.arrow"));
        auto vf = arrow::io::MemoryMappedFile::Open(
            (pinned / "tokenizer_vocab.arrow").string(),
            arrow::io::FileMode::READ);
        auto mf = arrow::io::MemoryMappedFile::Open(
            (pinned / "tokenizer_merges.arrow").string(),
            arrow::io::FileMode::READ);
        REQUIRE(vf.ok());
        REQUIRE(mf.ok());
        lsxcommon::ArrowTokenizerDb db;
        REQUIRE(lsxcommon::load_tokenizer_metadata(*vf, *mf, db).ok());
        CHECK(db.tokenizer_kind == 3);
        CHECK(db.base_vocab_size == 259);
        CHECK(db.bos_id == 259);
        CHECK(db.eos_id == 260);
        CHECK(db.pad_id == 263);
    }
    fs::remove_all(tmp);
}

TEST_CASE("muse dense convert refuses unknown tensors", "[muse_dense][convert]") {
    namespace fs = std::filesystem;
    const auto tmp = fs::temp_directory_path() / "muse_dense_unknown";
    fs::remove_all(tmp);
    fs::create_directories(tmp);

    auto tensors = muse_dense_test::BuildSyntheticMuseDenseModel();
    tensors.push_back({"model.language_model.layers.0.self_attn.q_norm.weight",
                       "BF16", {16}, synthtest::MakeBf16(16)});
    synthtest::WriteSafetensors(tmp, "model.safetensors", tensors);
    muse_dense_test::WriteConfig(tmp);
    muse_dense_test::WriteTokenizerJson(tmp);

    auto st = lsxcommon::muse_dense::ConvertMuseDenseModel(
        tmp.string(), (tmp / "out").string());
    CHECK_FALSE(st.ok());
    CHECK(st.message().find("q_norm") != std::string::npos);
    fs::remove_all(tmp);
}

TEST_CASE("muse dense convert refuses a missing projection",
          "[muse_dense][convert]") {
    namespace fs = std::filesystem;
    const auto tmp = fs::temp_directory_path() / "muse_dense_missing";
    fs::remove_all(tmp);
    fs::create_directories(tmp);

    auto tensors = muse_dense_test::BuildSyntheticMuseDenseModel();
    // Drop the attention gate of layer 1: the converter must error, not
    // silently produce a model that decodes without the gate.
    tensors.erase(std::remove_if(tensors.begin(), tensors.end(),
                                 [](const synthtest::FakeTensor &t) {
                                     return t.name ==
                                            "model.language_model.layers.1."
                                            "self_attn.gate_proj.weight";
                                 }),
                  tensors.end());
    synthtest::WriteSafetensors(tmp, "model.safetensors", tensors);
    muse_dense_test::WriteConfig(tmp);
    muse_dense_test::WriteTokenizerJson(tmp);

    auto st = lsxcommon::muse_dense::ConvertMuseDenseModel(
        tmp.string(), (tmp / "out").string());
    CHECK_FALSE(st.ok());
    fs::remove_all(tmp);
}
