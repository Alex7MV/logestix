// MiniMax-M3 converter stage (Model Ingestion Protocol stage 1/2): config
// parsing, tensor classification, quantization policy, and a synthetic
// end-to-end conversion (label `convert`, bounded TIMEOUT — the frozen
// geometry gate is relaxed with LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY so the
// fixture never touches the real 800 GB checkpoint).
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/minimax_moe_convert.h"
#include "test_minimax_moe_fixture.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

using namespace lsxcommon::minimax;
namespace fs = std::filesystem;

namespace {

const char *kGoldenDebt = nullptr;  // reserved for parity-stage bookkeeping

std::string ReadAll(const fs::path &p)
{
    std::ifstream f(p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

} // namespace

TEST_CASE("ParseMinimaxConfig reads the text_config fields and dispatch arrays",
          "[minimax-moe-convert]")
{
    fs::path dir = fs::temp_directory_path() / "lsx_minimax_cfg";
    fs::remove_all(dir);
    fs::create_directories(dir);
    minimax_test::WriteConfig(dir);
    auto top = ParseMinimaxConfig(dir.string());
    REQUIRE(top.error_msg.empty());
    const auto &c = minimax_test::TestCfg();
    CHECK(top.hidden_size == c.hidden);
    CHECK(top.num_layers == c.layers);
    CHECK(top.num_attention_heads == c.heads);
    CHECK(top.num_key_value_heads == c.kv_heads);
    CHECK(top.head_dim == c.head_dim);
    CHECK(top.rotary_dim == c.rotary_dim);
    CHECK(top.vocab_size == c.vocab);
    CHECK(top.moe_intermediate_size == c.moe_inter);
    CHECK(top.shared_intermediate_size == c.shared_inter);
    CHECK(top.dense_intermediate_size == c.dense_inter);
    CHECK(top.num_experts == c.experts);
    CHECK(top.num_experts_per_tok == c.top_k);
    CHECK(top.index_n_heads == c.index_n_heads);
    CHECK(top.index_head_dim == c.index_head_dim);
    CHECK(top.index_block_size == c.block_size);
    CHECK(top.index_topk_blocks == c.topk_blocks);
    CHECK(top.index_local_blocks == c.local_blocks);
    CHECK(top.rms_norm_eps == Catch::Approx(1e-6));
    CHECK(top.rope_theta == Catch::Approx(5000000.0));
    CHECK(top.routed_scaling_factor == Catch::Approx(2.0));
    CHECK(top.swiglu_alpha == Catch::Approx(1.702));
    CHECK(top.swiglu_limit == Catch::Approx(7.0));
    // Token ids come from generation_config.json for this checkpoint.
    CHECK(top.bos_token_id == 1);
    CHECK(top.eos_token_id == 2);
    CHECK(top.pad_token_id == 0);
    REQUIRE(top.layer_types.size() == static_cast<size_t>(c.layers));
    REQUIRE(top.mlp_types.size() == static_cast<size_t>(c.layers));
    CHECK(top.layer_types[0] == "dense_attention");
    CHECK(top.layer_types[1] == "sparse_attention");
    CHECK(top.mlp_types[0] == "dense");
    CHECK(top.mlp_types[1] == "sparse");
    (void)kGoldenDebt;
    std::error_code ec;
    fs::remove_all(dir, ec);
}

TEST_CASE("ParseMinimaxConfig fails on a missing config", "[minimax-moe-convert]")
{
    auto top = ParseMinimaxConfig("/nonexistent_dir_lsx_minimax");
    REQUIRE_FALSE(top.error_msg.empty());
}

TEST_CASE("MiniMax-M3 tensor classification", "[minimax-moe-convert]")
{
    using K = MinimaxTensorKind;
    auto cl = [](const std::string &n) { return classify_minimax_tensor(n); };
    CHECK(cl("language_model.model.embed_tokens.weight").kind == K::Embed);
    CHECK(cl("language_model.lm_head.weight").kind == K::LmHead);
    CHECK(cl("language_model.model.norm.weight").kind == K::FinalNorm);
    CHECK(cl("language_model.model.layers.3.self_attn.q_proj.weight").kind ==
          K::AttnQ);
    CHECK(cl("language_model.model.layers.3.self_attn.q_proj.weight").layer == 3);
    CHECK(cl("language_model.model.layers.3.self_attn.index_q_proj.weight").kind ==
          K::IndexQ);
    CHECK(cl("language_model.model.layers.3.self_attn.index_k_norm.weight").kind ==
          K::IndexKNorm);
    CHECK(cl("language_model.model.layers.0.mlp.gate_proj.weight").kind ==
          K::DenseMlpGate);
    CHECK(cl("language_model.model.layers.3.block_sparse_moe.gate.weight").kind ==
          K::RouterGate);
    CHECK(cl("language_model.model.layers.3.block_sparse_moe."
             "e_score_correction_bias").kind == K::RouterBias);
    CHECK(cl("language_model.model.layers.3.block_sparse_moe."
             "shared_experts.down_proj.weight").kind == K::SharedDownProj);
    {
        auto p = cl("language_model.model.layers.5.block_sparse_moe."
                    "experts.42.w2.weight");
        CHECK(p.kind == K::ExpertW2);
        CHECK(p.layer == 5);
        CHECK(p.expert == 42);
    }
    CHECK(cl("vision_tower.vision_model.pre_layrnorm.weight").kind == K::Visual);
    CHECK(cl("multi_modal_projector.linear_2.bias").kind == K::Visual);
    CHECK(cl("patch_merge_mlp.linear_1.weight").kind == K::Visual);
    CHECK(cl("totally.unknown.name").kind == K::Unknown);
}

TEST_CASE("MiniMax-M3 quantization policy keeps norms and the router exact",
          "[minimax-moe-convert]")
{
    using K = MinimaxTensorKind;
    for (K k : {K::AttnQ, K::AttnK, K::AttnV, K::AttnO, K::IndexQ, K::IndexK,
                K::DenseMlpGate, K::DenseMlpUp, K::DenseMlpDown, K::ExpertW1,
                K::ExpertW2, K::ExpertW3, K::SharedGateProj,
                K::SharedUpProj, K::SharedDownProj})
        CHECK(minimax_tensor_is_nvfp4(k));
    // Embed/head stay FP16 on purpose (see the convert test's 4b block).
    CHECK_FALSE(minimax_tensor_is_nvfp4(K::Embed));
    CHECK_FALSE(minimax_tensor_is_nvfp4(K::LmHead));
    for (K k : {K::FinalNorm, K::InputNorm, K::PostNorm, K::AttnQNorm,
                K::AttnKNorm, K::IndexQNorm, K::IndexKNorm, K::RouterGate,
                K::RouterBias, K::Visual, K::Unknown})
        CHECK_FALSE(minimax_tensor_is_nvfp4(k));
    for (K k : {K::ExpertW1, K::ExpertW2, K::ExpertW3})
        CHECK(minimax_tensor_is_routed_expert(k));
    CHECK_FALSE(minimax_tensor_is_routed_expert(K::SharedGateProj));
}

TEST_CASE("ConvertMinimaxMoeModel converts a synthetic checkpoint",
          "[minimax-moe-convert]")
{
    const char *prev = std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY");
    setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1", 1);
    fs::path in = fs::temp_directory_path() / "lsx_minimax_in";
    fs::path out = fs::temp_directory_path() / "lsx_minimax_out";
    fs::remove_all(in);
    fs::remove_all(out);
    REQUIRE(minimax_test::BuildSyntheticCheckpoint(in) > 0);

    auto st = ConvertMinimaxMoeModel(in.string(), out.string());
    if (!st.ok()) WARN(st.ToString());
    REQUIRE(st.ok());

    const fs::path base = out;
    const auto &c = minimax_test::TestCfg();

    // 1) Index carries the format tag, dispatch arrays and frozen numbers.
    REQUIRE(fs::exists(base / "model.index.json"));
    {
        const std::string j = ReadAll(base / "model.index.json");
        CHECK(j.find("\"logestix-minimax-nvfp4\"") != std::string::npos);
        CHECK(j.find("\"model_type\": \"minimax-m3\"") != std::string::npos);
        CHECK(j.find("\"num_experts\": " + std::to_string(c.experts)) !=
              std::string::npos);
        CHECK(j.find("\"hidden_size\": " + std::to_string(c.hidden)) !=
              std::string::npos);
        CHECK(j.find("\"mlp_layer_types\"") != std::string::npos);
        CHECK(j.find("\"sparse_attention\"") != std::string::npos);
        CHECK(j.find("\"skipped_visual_tensors\": 2") != std::string::npos);
        CHECK(j.find("\"bos_token_id\": 1") != std::string::npos);
        CHECK(j.find("\"eos_token_id\": 2") != std::string::npos);
        CHECK(j.find("\"vram\"") != std::string::npos);
    }

    // 2) Attention projections and the scaled-identity norms are where the
    //    loader expects them; norms stay raw BF16.
    {
        auto names = synthtest::TensorNamesIn(base / "pinned" / "layer_2.arrow");
        bool has_q_norm = false, has_index_norm = false;
        for (auto &n : names) {
            if (n.find("self_attn.q_norm.weight") != std::string::npos)
                has_q_norm = true;
            if (n.find("self_attn.index_k_norm.weight") != std::string::npos)
                has_index_norm = true;
        }
        CHECK(has_q_norm);
        CHECK(has_index_norm);
        auto nv = lsxcommon::ReadRecordBatchFromFile(
                      base / "pinned" / "layer_2_nvfp4.arrow")
                      .ValueOrDie();
        auto recs = lsxcommon::DenseNvfp4Schema::UnpackBatch(nv).ValueOrDie();
        bool has_index_q = false;
        for (auto &r : recs) {
            if (r.name.find("self_attn.index_q_proj.weight") !=
                std::string::npos) {
                has_index_q = true;
                // [index_n_heads*index_head_dim, H] -> numel/2 packed bytes.
                CHECK(static_cast<int64_t>(r.packed.size()) ==
                      static_cast<int64_t>(c.index_n_heads) *
                          c.index_head_dim * c.hidden / 2);
            }
        }
        CHECK(has_index_q);
    }

    // 3) Dense-MLP layers carry the MLP triple in the same nvfp4 batch, and
    //    MoE layers do not.
    {
        auto dense = lsxcommon::DenseNvfp4Schema::UnpackBatch(
                         lsxcommon::ReadRecordBatchFromFile(
                             base / "pinned" / "layer_0_nvfp4.arrow")
                             .ValueOrDie())
                         .ValueOrDie();
        bool has_mlp = false;
        for (auto &r : dense)
            if (r.name.find("mlp.gate_proj.weight") != std::string::npos)
                has_mlp = true;
        CHECK(has_mlp);
        auto moe = lsxcommon::DenseNvfp4Schema::UnpackBatch(
                       lsxcommon::ReadRecordBatchFromFile(
                           base / "pinned" / "layer_2_nvfp4.arrow")
                           .ValueOrDie())
                       .ValueOrDie();
        bool moe_has_mlp = false;
        for (auto &r : moe)
            if (r.name.find("mlp.gate_proj.weight") != std::string::npos)
                moe_has_mlp = true;
        CHECK_FALSE(moe_has_mlp);
    }

    // 4) Router is stored F32 (both tensors), never quantized.
    {
        auto recs = lsxcommon::PinnedSchema::UnpackBatch(
                        lsxcommon::ReadRecordBatchFromFile(
                            base / "router" / "layer_2.arrow")
                            .ValueOrDie())
                        .ValueOrDie();
        REQUIRE(recs.size() == 2);
        bool gate = false, bias = false;
        for (auto &r : recs) {
            CHECK(r.dtype == "F32");
            if (r.name.find("gate.weight") != std::string::npos) {
                gate = true;
                CHECK(static_cast<int64_t>(r.data.size()) ==
                      static_cast<int64_t>(c.experts) * c.hidden * 4);
            }
            if (r.name.find("e_score_correction_bias") != std::string::npos) {
                bias = true;
                CHECK(static_cast<int64_t>(r.data.size()) ==
                      static_cast<int64_t>(c.experts) * 4);
            }
        }
        CHECK(gate);
        CHECK(bias);
    }

    // 4b) Embed + head stay FP16: BF16 raw in global.arrow, NOT in
    //     global_nvfp4.arrow (a 4-bit head row moves every logit by ~3 at this
    //     model's logit scale while the first-token margin is 0.4).
    {
        auto names = synthtest::TensorNamesIn(base / "pinned" / "global.arrow");
        bool emb = false, hd = false;
        for (auto &n : names) {
            if (n.find("embed_tokens.weight") != std::string::npos) emb = true;
            if (n.find("lm_head.weight") != std::string::npos) hd = true;
        }
        CHECK(emb);
        CHECK(hd);
        auto nv = lsxcommon::ReadRecordBatchFromFile(
                      base / "pinned" / "global_nvfp4.arrow");
        if (nv.ok()) {
            auto recs =
                lsxcommon::DenseNvfp4Schema::UnpackBatch(nv.ValueOrDie())
                    .ValueOrDie();
            for (auto &r : recs)
                CHECK(r.name.find("embed_tokens") == std::string::npos);
        }
    }

    // 5) Shared experts + expert files exist; dense-MLP layers have no experts.
    CHECK(fs::exists(base / "shared" / "layer_2_nvfp4.arrow"));
    CHECK_FALSE(fs::exists(base / "shared" / "layer_0_nvfp4.arrow"));
    for (int l = 0; l < c.layers; ++l) {
        for (int e = 0; e < c.experts; ++e) {
            const bool want = l >= c.first_moe_layer;
            CHECK(fs::exists(base / "experts" / ("layer_" + std::to_string(l)) /
                             ("exp_" + std::to_string(e) + ".arrow")) == want);
        }
    }
    {
        auto batch = lsxcommon::ReadRecordBatchFromFile(
                         base / "experts" / "layer_1" / "exp_0.arrow")
                         .ValueOrDie();
        auto pe = lsxcommon::ExpertSchema::UnpackBatch(batch).ValueOrDie();
        const int64_t n = static_cast<int64_t>(c.moe_inter) * c.hidden;
        CHECK(static_cast<int64_t>(pe.w1_packed.size()) == n / 2);
        CHECK(static_cast<int64_t>(pe.w2_packed.size()) == n / 2);
        CHECK(static_cast<int64_t>(pe.w3_packed.size()) == n / 2);
        // Different source bytes per projection => different payloads.
        CHECK(pe.w1_packed != pe.w3_packed);
    }

    // 6) Determinism: a second conversion produces a byte-identical index.
    {
        fs::path out2 = fs::temp_directory_path() / "lsx_minimax_out2";
        fs::remove_all(out2);
        REQUIRE(ConvertMinimaxMoeModel(in.string(), out2.string()).ok());
        CHECK(ReadAll(base / "model.index.json") ==
              ReadAll(out2 / "model.index.json"));
        fs::remove_all(out2);
    }

    std::error_code ec;
    fs::remove_all(in, ec);
    fs::remove_all(out, ec);
    if (prev) setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", prev, 1);
    else unsetenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY");
}
