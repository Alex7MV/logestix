// Gemma-4-31B-it (dense) converter test: tensor classifier + NVFP4 policy +
// the synthetic end-to-end pipeline (Model Ingestion Protocol step 1).
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "test_gemma_dense_fixture.h"

#include "lsxstorage/artifact.h"

#include "lsxcommon/gemma_dense_convert.h"
#include "lsxcommon/model_integrity.h"
#include "lsxcommon/schema.h"

using lsxcommon::gemma_dense::classify_gemma_dense_tensor;
using lsxcommon::gemma_dense::gemma_dense_tensor_is_nvfp4;
using K = lsxcommon::gemma_dense::TensorKind;

namespace {

// E2M1 (NVFP4 nibble) and E4M3 (block scale) decode tables. Host-side copies of
// the codec, used to decode what the converter WROTE and compare it with the
// source bytes — the same host-side decode that caught gemma-4-26b's 7.01x-hot
// logits. A test that trusts the engine's own dequant cannot see that bug.
float E2M1Of(int nib)
{
    static const float kVal[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,
                                   4.0f,  6.0f,  -0.0f, -0.5f, -1.0f, -1.5f,
                                   -2.0f, -3.0f, -4.0f, -6.0f};
    return kVal[nib & 0xF];
}

float E4M3Of(std::uint8_t b)
{
    if ((b & 0x7F) == 0) return 0.f;
    const int s = (b >> 7) & 1;
    const int e = (b >> 3) & 0x0F;
    const int m = b & 0x07;
    if (e == 0) return s ? -m * 0.001953125f : m * 0.001953125f;
    const float v = std::ldexp(1.f + m / 8.f, e - 7);
    return s ? -v : v;
}

} // namespace

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
    SECTION("unified gemma-4 ballast: vision_embedder + audio projection") {
        // google/gemma-4-12B-it is a `gemma4_unified` tree: no vision
        // transformer and no audio tower, only a patch projection + position
        // table and ONE audio embedding projection. Every one of those names
        // must be claimed as ballast, or the strict coverage check rejects the
        // checkpoint ("unclassified tensor(s)").
        for (const char *n : {
                 "model.vision_embedder.patch_dense.weight",
                 "model.vision_embedder.patch_dense.bias",
                 "model.vision_embedder.patch_ln1.weight",
                 "model.vision_embedder.patch_ln1.bias",
                 "model.vision_embedder.patch_ln2.weight",
                 "model.vision_embedder.patch_ln2.bias",
                 "model.vision_embedder.pos_embedding",
                 "model.vision_embedder.pos_norm.weight",
                 "model.vision_embedder.pos_norm.bias",
                 "model.embed_audio.embedding_projection.weight",
             }) {
            INFO(n);
            CHECK(classify_gemma_dense_tensor(n).kind == K::Visual);
        }
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
        // No codec keys in the index (design §2.4.3 rule 5): the per-layer map
        // is artifact.fb::layer_codec, derived from the bytes. The index keeps
        // geometry, so the assertion is that the keys are ABSENT — a test that
        // keeps demanding them would pin the second source of truth back in.
        CHECK(json.find("\"mlp_nvfp4\"") == std::string::npos);
        CHECK(json.find("\"attn_nvfp4\"") == std::string::npos);
        CHECK(json.find("\"head_nvfp4\"") == std::string::npos);
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

// ── gemma-4-12B-it (`gemma4_unified` wrapper) ────────────────────────────
//
// ONE converter serves the whole Gemma-4 dense family, so the 12B checkpoint is
// accepted here rather than through a second pipeline. What differs is the
// multimodal ballast: the unified tree ships `model.vision_embedder.*` and a
// single `model.embed_audio.embedding_projection` instead of a vision
// transformer, and its special ids live in `text_config` only.
TEST_CASE("gemma-4-12B unified checkpoint converts (synthetic E2E)",
          "[gemma_dense][convert]") {
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "gemma4_u12b_synth";
    remove_all(tmp);
    create_directories(tmp);

    synthtest::WriteSafetensors(tmp, "model.safetensors",
                                gemma_dense_test::BuildSyntheticGemmaUnifiedModel());
    gemma_dense_test::WriteUnifiedConfig(tmp);
    std::filesystem::copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_gemma_tok.json",
                               tmp / "tokenizer.json");

    const auto out = tmp / "out";
    auto cst = lsxcommon::gemma_dense::ConvertGemmaDenseModel(tmp.string(),
                                                              out.string());
    if (!cst.ok()) FAIL("ConvertGemmaDenseModel: " << cst.ToString());
    REQUIRE(cst.ok());

    // The text tower is complete and the ballast is nowhere in the artifact.
    CHECK(exists(out / "pinned" / "global_nvfp4.arrow"));
    for (int l = 0; l < gemma_dense_test::TestCfg().num_layers; ++l)
        CHECK(exists(out / "pinned" /
                     ("layer_" + std::to_string(l) + ".arrow")));
    for (const auto &arrow : {out / "pinned" / "global.arrow",
                              out / "pinned" / "layer_0.arrow"}) {
        auto batch = lsxcommon::LoadTensorBatchViews(arrow);
        REQUIRE(batch.ok());
        for (const auto &t : batch->tensors) {
            INFO(t.name);
            CHECK(t.name.rfind("model.vision_embedder.", 0) != 0);
            CHECK(t.name.rfind("model.embed_audio.", 0) != 0);
            CHECK(t.name.rfind("model.embed_vision.", 0) != 0);
        }
        batch->keepalive.reset();
    }
    {
        auto g = lsxcommon::LoadDenseNvfp4BatchViews(out / "pinned" /
                                                     "global_nvfp4.arrow");
        REQUIRE(g.ok());
        REQUIRE(g->records.size() == 1);
        CHECK(g->records[0].name == "model.language_model.embed_tokens.weight");
        g->keepalive.reset();
    }

    // The index names the model it actually converted (the input directory's
    // leaf) and counts the ballast it stripped: an artifact that still says
    // "gemma-4-31B-it" cannot be told apart from the 31B tree by a reader.
    {
        std::ifstream idx(out / "model.index.json");
        const std::string json((std::istreambuf_iterator<char>(idx)),
                               std::istreambuf_iterator<char>());
        CHECK(json.find("\"model\": \"gemma4_u12b_synth\"") !=
              std::string::npos);
        CHECK(json.find("\"skipped_visual_tensors\": 11") != std::string::npos);
    }

    // Round-trip through the loader-side parser: the ids come from
    // text_config + generation_config (no top-level bos/pad in this nesting).
    auto top2 = lsxcommon::gemma_dense::load_gemma_dense_topology(
        (out / "model.index.json").string());
    CHECK(top2.error_msg.empty());
    CHECK(top2.num_layers == 4);
    CHECK(top2.hidden_size == 64);
    CHECK(top2.bos_id == 2);
    CHECK(top2.pad_id == 0);
    CHECK(top2.eos_ids == std::vector<int>({1, 106, 50}));
    CHECK(top2.layer_types.size() == 4);
    CHECK(top2.IsFull(3));

    remove_all(tmp);
}

// The unified tree's special ids are nested one level deeper than the 31B's.
// Reading them from the wrong level yields bos = -1 / an empty eos set, which
// the framing and the eos stop condition then silently inherit.
TEST_CASE("gemma-4-12B config parses text_config ids and geometry",
          "[gemma_dense][convert]") {
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "gemma4_u12b_cfg";
    remove_all(tmp);
    create_directories(tmp);
    gemma_dense_test::WriteUnifiedConfig(tmp);

    auto top = lsxcommon::gemma_dense::load_gemma_dense_topology(
        (tmp / "config.json").string());
    REQUIRE(top.error_msg.empty());
    CHECK(top.hidden_size == 64);
    CHECK(top.num_layers == 4);
    CHECK(top.bos_id == 2);
    CHECK(top.pad_id == 0);
    CHECK(top.eos_ids == std::vector<int>({1, 106, 50}));
    CHECK(top.sliding_window == 8);
    CHECK(top.partial_rotary_factor == Catch::Approx(0.25f));
    remove_all(tmp);
}

// ── The converted tree must be a VALID ARTIFACT, not just a loadable layout ──
//
// `lsxforge --codec-profile nvfp4` publishes an artifact only after
// ScanArtifactTree accepts what the converter wrote (verifier rule D1: a weight
// plane names its codec, and `U8` is the retired ambiguous spelling of NVFP4
// nibbles). Before this test the dense Gemma-4 converter wrote its NVFP4 triple
// as `U8`, so every conversion through the codec-profile path failed at the
// publish step with "the converted tree is not a valid artifact (nothing was
// published)" — after the whole tree had been quantised on the GPU.
TEST_CASE("gemma-4-12B converted tree passes the artifact scan",
          "[gemma_dense][convert]") {
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "gemma4_u12b_scan";
    remove_all(tmp);
    create_directories(tmp);

    synthtest::WriteSafetensors(
        tmp, "model.safetensors",
        gemma_dense_test::BuildSyntheticGemmaUnifiedModel());
    gemma_dense_test::WriteUnifiedConfig(tmp);
    std::filesystem::copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_gemma_tok.json",
                               tmp / "tokenizer.json");

    const auto out = tmp / "out";
    auto cst = lsxcommon::gemma_dense::ConvertGemmaDenseModel(tmp.string(),
                                                              out.string());
    if (!cst.ok()) FAIL("ConvertGemmaDenseModel: " << cst.ToString());
    REQUIRE(cst.ok());

    const auto scan = lsx::storage::ScanArtifactTree(out);
    std::string joined;
    for (const auto &p : scan.problems) joined += "  " + p + "\n";
    INFO("artifact scan problems:\n" << joined);
    REQUIRE(scan.problems.empty());
    CHECK(scan.ok);
    // The tree really is 4-bit where it claims to be: the scan derives the mask
    // from the bytes, so a converter that quietly wrote BF16 would show here.
    REQUIRE(scan.mask != 0);
    remove_all(tmp);
}

// A dense Gemma-4 config that omits a geometry field must be refused, not
// filled in from a default: this family has SEVERAL sizes served by one
// converter, so a default is one sibling's number, and a tree written with it
// decodes at the wrong width. The geometry gate would catch it at Load, but a
// conversion that produces a tree nobody can load is a wasted GPU pass.
TEST_CASE("convert rejects a config missing a geometry field",
          "[gemma_dense][convert]") {
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "gemma4_missing_field";
    remove_all(tmp);
    create_directories(tmp);

    synthtest::WriteSafetensors(
        tmp, "model.safetensors",
        gemma_dense_test::BuildSyntheticGemmaUnifiedModel());
    gemma_dense_test::WriteUnifiedConfig(tmp);
    std::filesystem::copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_gemma_tok.json",
                               tmp / "tokenizer.json");

    // Drop `intermediate_size` from text_config.
    const auto cfg = tmp / "config.json";
    std::ifstream in(cfg);
    std::string json((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
    in.close();
    const std::string key = "\"intermediate_size\":32,";
    const auto pos = json.find(key);
    REQUIRE(pos != std::string::npos);
    json.replace(pos, key.size(), "");
    {
        std::ofstream o(cfg, std::ios::trunc);
        o << json;
    }

    auto st = lsxcommon::gemma_dense::ConvertGemmaDenseModel(
        tmp.string(), (tmp / "out").string());
    INFO(st.ToString());
    REQUIRE_FALSE(st.ok());
    CHECK(st.ToString().find("intermediate_size") != std::string::npos);
    remove_all(tmp);
}

// The same for the loader-side parser: an index without `hidden_size` must
// report an error, because every arithmetic constant this family derives
// (embedding scale, KV geometry, window) comes from it.
TEST_CASE("index parser reports a missing geometry field",
          "[gemma_dense][convert]") {
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "gemma4_index_missing";
    remove_all(tmp);
    create_directories(tmp);
    {
        std::ofstream o(tmp / "model.index.json");
        o << "{\"layout\":2,\"model\":\"x\",\"num_layers\":4,"
             "\"vocab_size\":64,\"intermediate_size\":32,"
             "\"num_attention_heads\":4,\"num_key_value_heads\":2,"
             "\"num_global_key_value_heads\":1,\"head_dim\":16,"
             "\"global_head_dim\":64}";
    }
    auto top = lsxcommon::gemma_dense::load_gemma_dense_topology(
        (tmp / "model.index.json").string());
    CHECK_FALSE(top.error_msg.empty());
    CHECK(top.error_msg.find("hidden_size") != std::string::npos);
    remove_all(tmp);
}

// ── The residual plane (stage 2 "Parity fix") ──────────────────────────────
//
// The dense Gemma-4 members are ingested at 4 bits per weight, and their FSI
// gate is a near-tie decision: the accumulated body error of ONE plane moves the
// fork. The fix that clears it (phi-4, gemma-4-26b) is a SECOND NVFP4 plane
// carrying `w - dequant(base)`, decoded as one more GEMV with acc=true: ~8
// effective bits for 2x the bytes, no new codec, no new kernel family.
//
// A 12B fits that budget (the 31B does not: every linear at 8 bits is 33.7 GiB
// and this card has 32). The tied embed/lm_head table stays 4-bit — the 26B's
// measured recipe left it 4-bit and still went green, and it is the single
// largest plane in the tree.
TEST_CASE("dense gemma conversions carry a residual plane on every projection",
          "[gemma_dense][convert]") {
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "gemma4_u12b_residual";
    remove_all(tmp);
    create_directories(tmp);

    synthtest::WriteSafetensors(
        tmp, "model.safetensors",
        gemma_dense_test::BuildSyntheticGemmaUnifiedModel());
    gemma_dense_test::WriteUnifiedConfig(tmp);
    std::filesystem::copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_gemma_tok.json",
                               tmp / "tokenizer.json");

    const auto out = tmp / "out";
    auto cst = lsxcommon::gemma_dense::ConvertGemmaDenseModel(
        tmp.string(), out.string(),
        lsxcommon::gemma_dense::GemmaDenseConvertOptions{.residual_plane = true});
    if (!cst.ok()) FAIL("ConvertGemmaDenseModel: " << cst.ToString());
    REQUIRE(cst.ok());

    auto find = [](const std::vector<lsxcommon::TensorView> &tensors,
                   std::string_view name) -> const lsxcommon::TensorView * {
        for (const auto &t : tensors)
            if (t.name == name) return &t;
        return nullptr;
    };

    auto batch = lsxcommon::LoadTensorBatchViews(out / "pinned" / "layer_0.arrow");
    REQUIRE(batch.ok());

    // Every projection of a layer carries base + residual; the norms do not.
    for (const char *proj : {
             "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
             "self_attn.q_proj.weight", "self_attn.k_proj.weight",
             "self_attn.v_proj.weight", "self_attn.o_proj.weight"}) {
        const std::string base =
            std::string("model.language_model.layers.0.") + proj;
        for (const char *suffix : {".packed", ".scales", ".global_scale"}) {
            INFO(base + suffix);
            REQUIRE(find(batch->tensors, base + suffix) != nullptr);
            INFO(base + ".res" + suffix);
            REQUIRE(find(batch->tensors, base + ".res" + suffix) != nullptr);
        }
    }
    CHECK(find(batch->tensors,
               "model.language_model.layers.0.input_layernorm.weight.res.packed") ==
          nullptr);

    // base + res reconstructs the source at least 2x better than base alone.
    // Decoded on the HOST from the written records, against the fixture's own
    // BF16 bytes — the same check that found gemma-4-26b's 7.01x logits bug.
    {
        const auto &cfg = gemma_dense_test::TestCfg();
        const int64_t I = cfg.intermediate_size;
        const int64_t H = cfg.hidden_size;
        const std::string base = "model.language_model.layers.0.mlp.gate_proj.weight";
        const auto *pk = find(batch->tensors, base + ".packed");
        const auto *sc = find(batch->tensors, base + ".scales");
        const auto *gs = find(batch->tensors, base + ".global_scale");
        const auto *rpk = find(batch->tensors, base + ".res.packed");
        const auto *rsc = find(batch->tensors, base + ".res.scales");
        const auto *rgs = find(batch->tensors, base + ".res.global_scale");
        REQUIRE(rpk != nullptr);
        REQUIRE(rsc != nullptr);
        REQUIRE(rgs != nullptr);
        REQUIRE(pk != nullptr);

        auto decode = [](const std::vector<uint8_t> &packed,
                         const std::vector<uint8_t> &scales, float gs,
                         int64_t rows, int64_t cols) {
            std::vector<float> out(static_cast<size_t>(rows * cols));
            for (int64_t i = 0; i < rows * cols; ++i) {
                const uint8_t byte = packed[static_cast<size_t>(i) >> 1];
                const int nib = (i & 1) ? (byte >> 4) : (byte & 0xF);
                out[static_cast<size_t>(i)] =
                    E2M1Of(nib) * E4M3Of(scales[static_cast<size_t>(i) >> 4]) * gs;
            }
            return out;
        };
        std::vector<uint8_t> bpk(pk->data, pk->data + pk->bytes);
        std::vector<uint8_t> bsc(sc->data, sc->data + sc->bytes);
        std::vector<uint8_t> rpkv(rpk->data, rpk->data + rpk->bytes);
        std::vector<uint8_t> rscv(rsc->data, rsc->data + rsc->bytes);
        float bgs = 0.f, rgsv = 0.f;
        std::memcpy(&bgs, gs->data, 4);
        std::memcpy(&rgsv, rgs->data, 4);
        const auto base_w = decode(bpk, bsc, bgs, I, H);
        const auto res_w = decode(rpkv, rscv, rgsv, I, H);

        // The fixture's own BF16 rows for this tensor: MakeBf16 writes
        // element i's BF16 bits as `i % 65536`, so the source decodes exactly.
        std::vector<float> src(static_cast<size_t>(I * H));
        for (size_t i = 0; i < src.size(); ++i) {
            const std::uint32_t wide =
                static_cast<std::uint32_t>(synthtest::MakeBf16BitsAt(
                    static_cast<int64_t>(i)))
                << 16;
            std::memcpy(&src[i], &wide, 4);
        }

        auto rel = [&](bool with_res) {
            double num = 0.0, den = 0.0;
            for (size_t i = 0; i < src.size(); ++i) {
                const double w = with_res ? base_w[i] + res_w[i] : base_w[i];
                num += (w - src[i]) * (w - src[i]);
                den += static_cast<double>(src[i]) * src[i];
            }
            return std::sqrt(num / den);
        };
        const double e_base = rel(false);
        const double e_res = rel(true);
        INFO("relL2 base=" << e_base << " base+res=" << e_res);
        CHECK(e_base > 0.02);           // 4 bits really is lossy
        CHECK(e_res < e_base * 0.5);    // the residual is worth >= 1 bit
    }

    // The index records the choice, so the VRAM estimate and a reader can see it.
    {
        std::ifstream idx(out / "model.index.json");
        const std::string json((std::istreambuf_iterator<char>(idx)),
                               std::istreambuf_iterator<char>());
        CHECK(json.find("\"residual\": \"attn+mlp\"") != std::string::npos);
    }

    batch->keepalive.reset();
    remove_all(tmp);
}

// The tied embed/lm_head table is NOT doubled. Measured 2026-11-28 on the real
// 12B tree: giving the head its own residual plane (+0.6 GiB, 13 GB artifact)
// left FSI at 2/3 with byte-identical output, so it was reverted. The plane that
// matters is the projections', which took FSI-01 from red to green.
TEST_CASE("dense gemma residual plane excludes the tied embed/lm_head table",
          "[gemma_dense][convert]") {
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "gemma4_u12b_res_head_off";
    remove_all(tmp);
    create_directories(tmp);
    synthtest::WriteSafetensors(
        tmp, "model.safetensors",
        gemma_dense_test::BuildSyntheticGemmaUnifiedModel());
    gemma_dense_test::WriteUnifiedConfig(tmp);
    std::filesystem::copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_gemma_tok.json",
                               tmp / "tokenizer.json");
    const auto out = tmp / "out";
    REQUIRE(lsxcommon::gemma_dense::ConvertGemmaDenseModel(
                tmp.string(), out.string(),
                lsxcommon::gemma_dense::GemmaDenseConvertOptions{
                    .residual_plane = true})
                .ok());
    auto g = lsxcommon::LoadDenseNvfp4BatchViews(out / "pinned" /
                                                 "global_nvfp4.arrow");
    REQUIRE(g.ok());
    REQUIRE(g->records.size() == 1);
    CHECK(g->records[0].name == "model.language_model.embed_tokens.weight");
    CHECK(g->records[0].name.find(".res") == std::string::npos);
    g->keepalive.reset();
    remove_all(tmp);
}

// ── The residual plane is a per-model decision, not a converter constant ────
//
// ONE converter serves every dense Gemma-4 size, and the plane's cost is
// proportional to the model: ~+5.7 GB at 12B (7.0 -> 12.0 GiB, fits a 32 GB card)
// and ~+16.7 GB at 31B (17 -> 33.7 GiB, which does NOT: the 31B row records that
// measurement). Writing it unconditionally would make a fresh 31B conversion
// produce a tree nobody can load, so the choice travels from the model's
// manifest row into the converter as an explicit option.
//
// The DEFAULT is 4-bit only on purpose. Forgetting to declare the option costs
// precision — which the FSI gate reports — and never produces an unloadable
// tree; the opposite default would fail silently in the expensive direction.
TEST_CASE("residual plane off: a 4-bit-only tree, no .res records",
          "[gemma_dense][convert]") {
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "gemma4_no_residual";
    remove_all(tmp);
    create_directories(tmp);
    synthtest::WriteSafetensors(
        tmp, "model.safetensors",
        gemma_dense_test::BuildSyntheticGemmaUnifiedModel());
    gemma_dense_test::WriteUnifiedConfig(tmp);
    std::filesystem::copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_gemma_tok.json",
                               tmp / "tokenizer.json");
    const auto out = tmp / "out";
    // ResidualPlane off — the 31B shape.
    auto st = lsxcommon::gemma_dense::ConvertGemmaDenseModel(
        tmp.string(), out.string(),
        lsxcommon::gemma_dense::GemmaDenseConvertOptions{.residual_plane = false});
    if (!st.ok()) FAIL("ConvertGemmaDenseModel: " << st.ToString());
    REQUIRE(st.ok());

    // No projection carries a second plane.
    for (int layer : {0, 3}) {
        auto batch = lsxcommon::LoadTensorBatchViews(
            out / "pinned" / ("layer_" + std::to_string(layer) + ".arrow"));
        REQUIRE(batch.ok());
        for (const auto &t : batch->tensors) {
            INFO(t.name);
            CHECK(std::string(t.name).find(".res.") == std::string::npos);
        }
        // The base planes are still there, so "off" means 4 bits, not "empty".
        CHECK(batch->tensors.size() > 10);
        batch->keepalive.reset();
    }
    {
        auto g = lsxcommon::LoadDenseNvfp4BatchViews(out / "pinned" /
                                                     "global_nvfp4.arrow");
        REQUIRE(g.ok());
        REQUIRE(g->records.size() == 1);   // the tied table, base only
        g->keepalive.reset();
    }

    // The index says so, instead of claiming a plane the tree does not hold.
    {
        std::ifstream idx(out / "model.index.json");
        const std::string json((std::istreambuf_iterator<char>(idx)),
                               std::istreambuf_iterator<char>());
        CHECK(json.find("\"residual\": \"none\"") != std::string::npos);
        CHECK(json.find("attn+mlp") == std::string::npos);
    }

    // And the artifact scan still accepts it: a 4-bit-only tree is a valid
    // artifact, not a half-written one.
    {
        const auto scan = lsx::storage::ScanArtifactTree(out);
        std::string joined;
        for (const auto &q : scan.problems) joined += "  " + q + "\n";
        INFO("artifact scan problems:\n" << joined);
        REQUIRE(scan.problems.empty());
    }
    remove_all(tmp);
}

// The shipped default is 4-bit only, and that is a safety property, not an
// accident of the code: forgetting the option costs precision (visible in the
// FSI gate) and can never produce the unloadable tree.
TEST_CASE("residual plane defaults to off (the safe direction)",
          "[gemma_dense][convert]") {
    lsxcommon::gemma_dense::GemmaDenseConvertOptions dflt{};
    CHECK_FALSE(dflt.residual_plane);
}
