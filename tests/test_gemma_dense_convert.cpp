// Gemma-4-31B-it (dense) converter test: tensor classifier + NVFP4 policy +
// the synthetic end-to-end pipeline (Model Ingestion Protocol step 1).
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "test_gemma_dense_fixture.h"

#include "lsxcommon/gemma_dense_convert.h"
#include "lsxcommon/model_integrity.h"
#include "lsxcommon/schema.h"

using lsxcommon::gemma_dense::classify_gemma_dense_tensor;
using lsxcommon::gemma_dense::gemma_dense_tensor_is_nvfp4;
using K = lsxcommon::gemma_dense::TensorKind;

TEST_CASE("gemma dense tensor classifier", "[gemma_dense][convert]") {
    SECTION("globals") {
        CHECK(classify_gemma_dense_tensor(
                  "model.language_model.embed_tokens.weight").kind == K::Embed);
        CHECK(classify_gemma_dense_tensor(
                  "model.language_model.norm.weight").kind == K::FinalNorm);
    }
    SECTION("vision ballast is skipped, never applied") {
        CHECK(classify_gemma_dense_tensor(
                  "model.vision_tower.patch_embedder.input_proj.weight").kind ==
              K::Visual);
        CHECK(classify_gemma_dense_tensor(
                  "model.embed_vision.embedding_projection.weight").kind ==
              K::Visual);
        CHECK(classify_gemma_dense_tensor(
                  "model.vision_tower.encoder.layers.5.self_attn.q_proj.linear."
                  "weight").kind == K::Visual);
        CHECK(classify_gemma_dense_tensor("model.vision_tower.std_scale").kind ==
              K::Visual);
        CHECK(classify_gemma_dense_tensor("model.vision_tower.std_bias").kind ==
              K::Visual);
    }
    SECTION("attention projections") {
        const auto q = classify_gemma_dense_tensor(
            "model.language_model.layers.3.self_attn.q_proj.weight");
        CHECK(q.kind == K::AttnQ);
        CHECK(q.layer == 3);
        CHECK(classify_gemma_dense_tensor(
                  "model.language_model.layers.0.self_attn.k_proj.weight").kind ==
              K::AttnK);
        CHECK(classify_gemma_dense_tensor(
                  "model.language_model.layers.0.self_attn.v_proj.weight").kind ==
              K::AttnV);
        CHECK(classify_gemma_dense_tensor(
                  "model.language_model.layers.0.self_attn.o_proj.weight").kind ==
              K::AttnO);
        CHECK(classify_gemma_dense_tensor(
                  "model.language_model.layers.0.self_attn.q_norm.weight").kind ==
              K::AttnQNorm);
        CHECK(classify_gemma_dense_tensor(
                  "model.language_model.layers.0.self_attn.k_norm.weight").kind ==
              K::AttnKNorm);
    }
    SECTION("dense MLP - no experts, no router") {
        const auto g = classify_gemma_dense_tensor(
            "model.language_model.layers.7.mlp.gate_proj.weight");
        CHECK(g.kind == K::MlpGate);
        CHECK(g.layer == 7);
        CHECK(classify_gemma_dense_tensor(
                  "model.language_model.layers.7.mlp.up_proj.weight").kind ==
              K::MlpUp);
        CHECK(classify_gemma_dense_tensor(
                  "model.language_model.layers.7.mlp.down_proj.weight").kind ==
              K::MlpDown);
        // A routed-expert tensor is a different family: it must NOT be claimed
        // here, so a MoE checkpoint handed to this converter errors out.
        CHECK(classify_gemma_dense_tensor(
                  "model.language_model.layers.9.experts.gate_up_proj").kind ==
              K::Unknown);
        CHECK(classify_gemma_dense_tensor(
                  "model.language_model.layers.9.router.proj.weight").kind ==
              K::Unknown);
    }
    SECTION("norms + layer_scalar") {
        CHECK(classify_gemma_dense_tensor(
                  "model.language_model.layers.1.input_layernorm.weight").kind ==
              K::InputNorm);
        CHECK(classify_gemma_dense_tensor(
                  "model.language_model.layers.1.post_attention_layernorm."
                  "weight").kind == K::PostNorm);
        CHECK(classify_gemma_dense_tensor(
                  "model.language_model.layers.1.pre_feedforward_layernorm."
                  "weight").kind == K::PreFFN);
        CHECK(classify_gemma_dense_tensor(
                  "model.language_model.layers.1.post_feedforward_layernorm."
                  "weight").kind == K::PostFFN);
        CHECK(classify_gemma_dense_tensor(
                  "model.language_model.layers.1.layer_scalar").kind ==
              K::LayerScalar);
    }
    SECTION("unknown tensors are never silently skipped") {
        CHECK(classify_gemma_dense_tensor("something.else.weight").kind ==
              K::Unknown);
        CHECK(classify_gemma_dense_tensor(
                  "model.language_model.layers.1.self_attn.bias.weight").kind ==
              K::Unknown);
    }
}

TEST_CASE("nvfp4 policy: every linear projection + the tied table",
          "[gemma_dense][convert]") {
    CHECK(gemma_dense_tensor_is_nvfp4(K::AttnQ));
    CHECK(gemma_dense_tensor_is_nvfp4(K::AttnK));
    CHECK(gemma_dense_tensor_is_nvfp4(K::AttnV));
    CHECK(gemma_dense_tensor_is_nvfp4(K::AttnO));
    CHECK(gemma_dense_tensor_is_nvfp4(K::MlpGate));
    CHECK(gemma_dense_tensor_is_nvfp4(K::MlpUp));
    CHECK(gemma_dense_tensor_is_nvfp4(K::MlpDown));
    CHECK(gemma_dense_tensor_is_nvfp4(K::Embed));
    // Norms and the per-layer scalar stay high precision.
    CHECK_FALSE(gemma_dense_tensor_is_nvfp4(K::InputNorm));
    CHECK_FALSE(gemma_dense_tensor_is_nvfp4(K::PostNorm));
    CHECK_FALSE(gemma_dense_tensor_is_nvfp4(K::PreFFN));
    CHECK_FALSE(gemma_dense_tensor_is_nvfp4(K::PostFFN));
    CHECK_FALSE(gemma_dense_tensor_is_nvfp4(K::AttnQNorm));
    CHECK_FALSE(gemma_dense_tensor_is_nvfp4(K::AttnKNorm));
    CHECK_FALSE(gemma_dense_tensor_is_nvfp4(K::FinalNorm));
    CHECK_FALSE(gemma_dense_tensor_is_nvfp4(K::LayerScalar));
    CHECK_FALSE(gemma_dense_tensor_is_nvfp4(K::Visual));
}

TEST_CASE("gemma dense topology parse", "[gemma_dense][convert]") {
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "gemma4dense_topo";
    remove_all(tmp);
    create_directories(tmp);
    gemma_dense_test::WriteConfig(tmp);

    auto top = lsxcommon::gemma_dense::load_gemma_dense_topology(
        (tmp / "config.json").string());
    REQUIRE(top.error_msg.empty());
    CHECK(top.num_layers == 4);
    CHECK(top.hidden_size == 64);
    CHECK(top.intermediate_size == 32);
    CHECK(top.vocab_size == 64);
    CHECK(top.num_attention_heads == 4);
    CHECK(top.num_key_value_heads == 2);
    CHECK(top.num_global_key_value_heads == 1);
    CHECK(top.head_dim == 16);
    CHECK(top.global_head_dim == 64);
    CHECK(top.sliding_window == 8);
    CHECK(top.rms_norm_eps == Catch::Approx(1e-6f));
    CHECK(top.partial_rotary_factor == Catch::Approx(0.25f));
    CHECK(top.rope_theta_sliding == Catch::Approx(10000.0f));
    CHECK(top.rope_theta_full == Catch::Approx(1000000.0f));
    CHECK(top.final_logit_softcapping == Catch::Approx(30.0f));
    CHECK(top.bos_id == 2);
    // config.json carries [1, 106]; generation_config.json adds 50.
    CHECK(top.eos_ids == std::vector<int>({1, 106, 50}));
    REQUIRE(top.layer_types.size() == 4);
    CHECK_FALSE(top.IsFull(0));
    CHECK(top.IsFull(3));
    CHECK(top.HeadDim(0) == 16);
    CHECK(top.HeadDim(3) == 64);
    CHECK(top.KvHeads(0) == 2);
    CHECK(top.KvHeads(3) == 1);
    CHECK(top.QRows(3) == 4 * 64);
    CHECK(top.KvRows(0) == 2 * 16);
    CHECK(top.Window(0) == 8);
    CHECK(top.Window(3) == 0);

    remove_all(tmp);
}

TEST_CASE("gemma dense convert happy path", "[gemma_dense][convert]") {
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "gemma4dense_conv";
    remove_all(tmp);
    create_directories(tmp);

    synthtest::WriteSafetensors(
        tmp, "model.safetensors",
        gemma_dense_test::BuildSyntheticGemmaDenseModel());
    gemma_dense_test::WriteConfig(tmp);
    std::filesystem::copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_gemma_tok.json",
                               tmp / "tokenizer.json");

    const auto out = tmp / "out";
    auto cst = lsxcommon::gemma_dense::ConvertGemmaDenseModel(tmp.string(),
                                                              out.string());
    if (!cst.ok()) FAIL("ConvertGemmaDenseModel: " << cst.ToString());
    REQUIRE(cst.ok());

    // Layout: pinned globals (raw norms) + the NVFP4 tied table + per-layer.
    CHECK(exists(out / "pinned" / "global.arrow"));
    CHECK(exists(out / "pinned" / "global_nvfp4.arrow"));
    for (int l = 0; l < gemma_dense_test::TestCfg().num_layers; ++l)
        CHECK(exists(out / "pinned" /
                     ("layer_" + std::to_string(l) + ".arrow")));
    // A dense checkpoint has no routed experts at all.
    CHECK_FALSE(exists(out / "experts"));
    // Tokenizer bridge (stage 4): the vocabulary/merges arrows plus the
    // metadata the loader re-mmaps.
    CHECK(exists(out / "pinned" / "tokenizer_vocab.arrow"));
    CHECK(exists(out / "pinned" / "tokenizer_merges.arrow"));

    // Global raw holds the norms/scalars, never a quantized projection.
    {
        const auto names = synthtest::TensorNamesIn(
            out / "pinned" / "global.arrow");
        CHECK(std::find(names.begin(), names.end(),
                        "model.language_model.norm.weight") != names.end());
    }

    // Per-layer layout: projections as the NVFP4 triple, norms/scalar raw.
    auto batch_res = lsxcommon::LoadTensorBatchViews(
        out / "pinned" / "layer_0.arrow");
    REQUIRE(batch_res.ok());
    auto find = [&](std::string_view name) -> const lsxcommon::TensorView * {
        for (const auto &t : batch_res->tensors)
            if (t.name == name) return &t;
        return nullptr;
    };
    const auto &cfg = gemma_dense_test::TestCfg();
    const size_t mlp_numel =
        static_cast<size_t>(cfg.intermediate_size) * cfg.hidden_size;
    const auto *pk =
        find("model.language_model.layers.0.mlp.gate_proj.weight.packed");
    const auto *sc =
        find("model.language_model.layers.0.mlp.gate_proj.weight.scales");
    const auto *gs =
        find("model.language_model.layers.0.mlp.gate_proj.weight.global_scale");
    REQUIRE(pk != nullptr);
    REQUIRE(sc != nullptr);
    REQUIRE(gs != nullptr);
    CHECK(pk->bytes == mlp_numel / 2);    // E2M1 nibbles, 2 elems/byte
    CHECK(sc->bytes == mlp_numel / 16);   // one E4M3 scale per 16 elements
    CHECK(gs->bytes == 4);                // F32 global scale
    CHECK(find("model.language_model.layers.0.mlp.gate_proj.weight") ==
          nullptr);                       // the raw BF16 twin is gone
    // Attention projections are NVFP4 too (full hidden-accuracy chain).
    CHECK(find("model.language_model.layers.0.self_attn.q_proj.weight.packed") !=
          nullptr);
    CHECK(find("model.language_model.layers.0.self_attn.o_proj.weight.packed") !=
          nullptr);
    // Norms + the per-layer scalar stay raw BF16.
    CHECK(find("model.language_model.layers.0.input_layernorm.weight") !=
          nullptr);
    CHECK(find("model.language_model.layers.0.post_attention_layernorm.weight") !=
          nullptr);
    CHECK(find("model.language_model.layers.0.pre_feedforward_layernorm.weight") !=
          nullptr);
    CHECK(find("model.language_model.layers.0.post_feedforward_layernorm.weight") !=
          nullptr);
    CHECK(find("model.language_model.layers.0.layer_scalar") != nullptr);
    CHECK(find("model.language_model.layers.0.self_attn.q_norm.weight") !=
          nullptr);

    // The full-attention layer stores NO v_proj (V is tied to K).
    {
        auto full_res = lsxcommon::LoadTensorBatchViews(
            out / "pinned" / "layer_3.arrow");
        REQUIRE(full_res.ok());
        bool has_v = false;
        for (const auto &t : full_res->tensors)
            if (t.name == "model.language_model.layers.3.self_attn.v_proj.weight")
                has_v = true;
        CHECK_FALSE(has_v);
        full_res->keepalive.reset();
    }

    // Index round-trip through the loader-side parser.
    {
        std::ifstream idx(out / "model.index.json");
        const std::string json((std::istreambuf_iterator<char>(idx)),
                               std::istreambuf_iterator<char>());
        CHECK(json.find("\"layout\": 2") !=
              std::string::npos);
        CHECK(json.find("\"mlp_nvfp4\": true") != std::string::npos);
        CHECK(json.find("\"attn_nvfp4\": true") != std::string::npos);
        CHECK(json.find("\"head_nvfp4\": true") != std::string::npos);
    }
    auto top2 = lsxcommon::gemma_dense::load_gemma_dense_topology(
        (out / "model.index.json").string());
    CHECK(top2.error_msg.empty());
    CHECK(top2.num_layers == 4);
    CHECK(top2.hidden_size == 64);
    CHECK(top2.vocab_size == 64);
    CHECK(top2.layer_types.size() == 4);
    CHECK(top2.eos_ids == std::vector<int>({1, 106, 50}));
    CHECK(top2.vram.min_bytes > 0);
    CHECK(top2.total_pinned_tensors > 0);

    // Release every handle before the cleanup: Windows refuses to delete a
    // mapped or open file while Linux unlinks happily and hides the ordering.
    batch_res->keepalive.reset();
    remove_all(tmp);
}

TEST_CASE("convert rejects a checkpoint without config.json",
          "[gemma_dense][convert]") {
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "gemma4dense_noconf";
    remove_all(tmp);
    create_directories(tmp);
    synthtest::WriteSafetensors(
        tmp, "model.safetensors",
        gemma_dense_test::BuildSyntheticGemmaDenseModel());
    std::filesystem::copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_gemma_tok.json",
                               tmp / "tokenizer.json");

    auto st = lsxcommon::gemma_dense::ConvertGemmaDenseModel(
        tmp.string(), (tmp / "out").string());
    REQUIRE_FALSE(st.ok());
    remove_all(tmp);
}

TEST_CASE("convert rejects layer_types contradicting the tensor shapes",
          "[gemma_dense][convert]") {
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "gemma4dense_badlt";
    remove_all(tmp);
    create_directories(tmp);
    synthtest::WriteSafetensors(
        tmp, "model.safetensors",
        gemma_dense_test::BuildSyntheticGemmaDenseModel());
    gemma_dense_test::WriteConfig(tmp);
    std::filesystem::copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_gemma_tok.json",
                               tmp / "tokenizer.json");

    // Flip layer 3 to sliding in the config while its q_proj keeps the
    // full-attention width.
    const auto cfg = tmp / "config.json";
    std::ifstream in(cfg);
    std::string json((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
    in.close();
    const auto pos = json.find("\"full_attention\"");
    REQUIRE(pos != std::string::npos);
    json.replace(pos, std::strlen("\"full_attention\""), "\"sliding_attention\"");
    { std::ofstream o(cfg, std::ios::trunc); o << json; }

    auto st = lsxcommon::gemma_dense::ConvertGemmaDenseModel(
        tmp.string(), (tmp / "out").string());
    REQUIRE_FALSE(st.ok());
    remove_all(tmp);
}

TEST_CASE("convert rejects a full-attention layer that ships a v_proj",
          "[gemma_dense][convert]") {
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "gemma4dense_vextraneous";
    remove_all(tmp);
    create_directories(tmp);

    auto tensors = gemma_dense_test::BuildSyntheticGemmaDenseModel();
    // attention_k_eq_v: a v_proj on layer 3 (full attention) contradicts the
    // frozen contract and must be an error, not a silently stored tensor.
    const int64_t H = gemma_dense_test::TestCfg().hidden_size;
    const int64_t kv = gemma_dense_test::FullKVRows();
    tensors.push_back({"model.language_model.layers.3.self_attn.v_proj.weight",
                       "BF16", {kv, H}, synthtest::MakeBf16(kv * H)});
    synthtest::WriteSafetensors(tmp, "model.safetensors", tensors);
    gemma_dense_test::WriteConfig(tmp);
    std::filesystem::copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_gemma_tok.json",
                               tmp / "tokenizer.json");

    auto st = lsxcommon::gemma_dense::ConvertGemmaDenseModel(
        tmp.string(), (tmp / "out").string());
    REQUIRE_FALSE(st.ok());
    remove_all(tmp);
}

TEST_CASE("convert rejects a sliding layer missing its v_proj",
          "[gemma_dense][convert]") {
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "gemma4dense_vmissing";
    remove_all(tmp);
    create_directories(tmp);

    auto tensors = gemma_dense_test::BuildSyntheticGemmaDenseModel();
    const std::string victim =
        "model.language_model.layers.1.self_attn.v_proj.weight";
    tensors.erase(std::remove_if(tensors.begin(), tensors.end(),
                                 [&](const synthtest::FakeTensor &t) {
                                     return t.name == victim;
                                 }),
                  tensors.end());
    synthtest::WriteSafetensors(tmp, "model.safetensors", tensors);
    gemma_dense_test::WriteConfig(tmp);
    std::filesystem::copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_gemma_tok.json",
                               tmp / "tokenizer.json");

    auto st = lsxcommon::gemma_dense::ConvertGemmaDenseModel(
        tmp.string(), (tmp / "out").string());
    REQUIRE_FALSE(st.ok());
    remove_all(tmp);
}

TEST_CASE("convert rejects a MoE checkpoint (this family is dense only)",
          "[gemma_dense][convert]") {
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "gemma4dense_moeinput";
    remove_all(tmp);
    create_directories(tmp);

    auto tensors = gemma_dense_test::BuildSyntheticGemmaDenseModel();
    tensors.push_back({"model.language_model.layers.1.experts.gate_up_proj",
                       "BF16", {4, 32, 64}, synthtest::MakeBf16(4 * 32 * 64)});
    synthtest::WriteSafetensors(tmp, "model.safetensors", tensors);
    gemma_dense_test::WriteConfig(tmp);
    std::filesystem::copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_gemma_tok.json",
                               tmp / "tokenizer.json");

    auto st = lsxcommon::gemma_dense::ConvertGemmaDenseModel(
        tmp.string(), (tmp / "out").string());
    REQUIRE_FALSE(st.ok());
    remove_all(tmp);
}
