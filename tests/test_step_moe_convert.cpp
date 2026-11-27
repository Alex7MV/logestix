// Step-5-Preview converter stage (Model Ingestion Protocol stage 1/2):
// config parsing, tensor classification, the fused FP8 expert split, the
// deferred MTP/vision ballast and the synthetic end-to-end conversion
// (label `convert`, bounded TIMEOUT — the frozen geometry gate is relaxed with
// LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY so the fixture never touches the real
// 606 GB checkpoint).
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/step_moe_convert.h"
#include "test_step_moe_fixture.h"

#include <arrow/io/file.h>

#include <lsxcommon/tokenizer.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

using namespace lsxcommon::step;
namespace fs = std::filesystem;

namespace {

std::string ReadAll(const fs::path &p)
{
    std::ifstream f(p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

} // namespace

TEST_CASE("ParseStepConfig reads the step5 text_config and the DSA schedule",
          "[step-moe-convert]")
{
    fs::path dir = fs::temp_directory_path() / "lsx_step_cfg";
    fs::remove_all(dir);
    fs::create_directories(dir);
    step_test::WriteConfig(dir);
    auto top = ParseStepConfig(dir.string());
    REQUIRE(top.error_msg.empty());
    const auto &c = step_test::TestCfg();
    CHECK(top.hidden_size == c.hidden);
    CHECK(top.num_layers == c.layers);
    CHECK(top.num_attention_heads == c.heads);
    CHECK(top.num_key_value_heads == c.kv_heads);
    CHECK(top.head_dim == c.head_dim);
    CHECK(top.vocab_size == c.vocab);
    CHECK(top.moe_intermediate_size == c.moe_inter);
    CHECK(top.shared_intermediate_size == c.shared_inter);
    CHECK(top.dense_intermediate_size == c.dense_inter);
    CHECK(top.num_experts == c.experts);
    CHECK(top.num_experts_per_tok == c.top_k);
    CHECK(top.first_moe_layer == c.first_moe_layer);
    CHECK(top.num_mtp_layers == c.mtp_layers);
    CHECK(top.sliding_window == c.sliding_window);
    CHECK(top.router_scaling_factor == Catch::Approx(c.router_scaling));
    CHECK(top.eos_token_id == 1);
    CHECK(top.eos_token_id2 == 2);
    CHECK(top.pad_token_id == 2);
    REQUIRE(top.mlp_types.size() == static_cast<size_t>(c.layers));
    CHECK(top.mlp_types[0] == "dense");
    CHECK(top.mlp_types[1] == "sparse");
    REQUIRE(top.layer_types.size() == static_cast<size_t>(c.layers));
    CHECK(top.layer_types[0] == "sliding_attention");
    CHECK(top.layer_types[3] == "full_attention");
    std::error_code ec;
    fs::remove_all(dir, ec);
}

TEST_CASE("ParseStepConfig fails on a missing config", "[step-moe-convert]")
{
    auto top = ParseStepConfig("/nonexistent_dir_lsx_step");
    REQUIRE_FALSE(top.error_msg.empty());
}

TEST_CASE("Step-5 tensor classification", "[step-moe-convert]")
{
    using K = StepTensorKind;
    auto cl = [](const std::string &n) { return classify_step_tensor(n); };
    CHECK(cl("model.embed_tokens.weight").kind == K::Embed);
    CHECK(cl("lm_head.weight").kind == K::LmHead);
    CHECK(cl("model.norm.weight").kind == K::FinalNorm);
    CHECK(cl("model.layers.3.self_attn.q_proj.weight").kind == K::AttnQ);
    CHECK(cl("model.layers.3.self_attn.q_proj.weight").layer == 3);
    CHECK(cl("model.layers.3.self_attn.g_proj.weight").kind == K::AttnG);
    CHECK(cl("model.layers.3.self_attn.q_norm.weight").kind == K::AttnQNorm);
    CHECK(cl("model.layers.3.self_attn.sparse_indexer_q.weight").kind == K::IndexQ);
    CHECK(cl("model.layers.3.self_attn.sparse_indexer_k_norm.bias").kind ==
          K::IndexKNormBias);
    CHECK(cl("model.layers.3.self_attn.ssmax_s").kind == K::IndexSsmax);
    CHECK(cl("model.layers.0.mlp.gate_proj.weight").kind == K::DenseMlpGate);
    CHECK(cl("model.layers.3.moe.gate.weight").kind == K::RouterGate);
    CHECK(cl("model.layers.3.moe.router_bias").kind == K::RouterBias);
    CHECK(cl("model.layers.3.moe.gate_proj.weight").kind == K::ExpertGate);
    CHECK(cl("model.layers.3.moe.gate_proj.weight_scale_inv").kind ==
          K::ExpertGateScale);
    CHECK(cl("model.layers.3.moe.down_proj.weight").kind == K::ExpertDown);
    CHECK(cl("model.layers.3.share_expert.down_proj.weight").kind ==
          K::SharedDownProj);
    // Vision ballast and the three MTP layers are stripped, never applied.
    CHECK(cl("vision_model.conv1.weight").kind == K::VisionBallast);
    CHECK(cl("vit_large_projector.weight").kind == K::VisionBallast);
    CHECK(cl("model.layers.92.enorm.weight").kind == K::MtpBallast);
    CHECK(cl("model.layers.92.transformer.shared_head.output.weight").kind ==
          K::MtpBallast);
    CHECK(cl("totally.unknown.name").kind == K::Unknown);
}

TEST_CASE("Step-5 quantization policy keeps norms, embed and the router exact",
          "[step-moe-convert]")
{
    using K = StepTensorKind;
    for (K k : {K::AttnQ, K::AttnK, K::AttnV, K::AttnO, K::AttnG, K::DenseMlpGate,
                K::DenseMlpUp, K::DenseMlpDown, K::ExpertGate, K::ExpertDown,
                K::ExpertUp, K::SharedGateProj, K::SharedUpProj,
                K::SharedDownProj})
        CHECK(step_tensor_is_nvfp4(k));
    for (K k : {K::Embed, K::LmHead, K::FinalNorm, K::InputNorm, K::PostNorm,
                K::AttnQNorm, K::AttnKNorm, K::RouterGate, K::RouterBias,
                K::IndexQ, K::IndexW, K::Unknown})
        CHECK_FALSE(step_tensor_is_nvfp4(k));
    for (K k : {K::ExpertGate, K::ExpertDown, K::ExpertUp, K::ExpertGateScale})
        CHECK(step_tensor_is_routed_expert(k));
    CHECK_FALSE(step_tensor_is_routed_expert(K::SharedGateProj));
    CHECK_FALSE(step_tensor_is_routed_expert(K::AttnQ));
}

TEST_CASE("ConvertStepMoeModel converts a synthetic checkpoint",
          "[step-moe-convert]")
{
    const char *prev = std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY");
    setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1", 1);
    fs::path in = fs::temp_directory_path() / "lsx_step_in";
    fs::path out = fs::temp_directory_path() / "lsx_step_out";
    fs::remove_all(in);
    fs::remove_all(out);
    REQUIRE(step_test::BuildSyntheticCheckpoint(
                in, true, TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json") > 0);

    auto st = ConvertStepMoeModel(in.string(), out.string());
    if (!st.ok()) WARN(st.ToString());
    REQUIRE(st.ok());

    const fs::path base = out;
    const auto &c = step_test::TestCfg();

    // 1) Index carries the format tag, dispatch, hybrid schedule and stripped
    // ballast counts.
    {
        REQUIRE(fs::exists(base / "model.index.json"));
        const std::string j = ReadAll(base / "model.index.json");
        CHECK(j.find("\"layout\": 2") != std::string::npos);
        CHECK(j.find("\"model_type\": \"step-5-preview\"") != std::string::npos);
        CHECK(j.find("\"num_experts\": " + std::to_string(c.experts)) !=
              std::string::npos);
        CHECK(j.find("\"hidden_size\": " + std::to_string(c.hidden)) !=
              std::string::npos);
        CHECK(j.find("\"mlp_layer_types\"") != std::string::npos);
        CHECK(j.find("\"layer_types\"") != std::string::npos);
        CHECK(j.find("\"first_moe_layer\": 1") != std::string::npos);
        CHECK(j.find("\"bos_token_id\": 0") != std::string::npos);
        CHECK(j.find("\"eos_token_id\": 1") != std::string::npos);
        CHECK(j.find("\"eos_token_id2\": 2") != std::string::npos);
        CHECK(j.find("\"num_mtp_layers_stripped\": 1") != std::string::npos);
        CHECK(j.find("\"skipped_visual_tensors\": 2") != std::string::npos);
        CHECK(j.find("\"skipped_mtp_tensors\": 4") != std::string::npos);
        CHECK(j.find("\"vram\"") != std::string::npos);
    }

    // 1b) Tokenizer policy. Step-5's tokenizer.json is the DeepSeek three-stage
    // Sequence (\p{N}{1,3} + CJK Isolated, then the six-alternative main
    // pattern), NOT the Flint/o200k case-run pattern: the arrows must carry
    // tokenizer_kind 0.
    {
        auto vf = arrow::io::MemoryMappedFile::Open(
            (base / "pinned" / "tokenizer_vocab.arrow").string(),
            arrow::io::FileMode::READ);
        auto mf = arrow::io::MemoryMappedFile::Open(
            (base / "pinned" / "tokenizer_merges.arrow").string(),
            arrow::io::FileMode::READ);
        REQUIRE(vf.ok());
        REQUIRE(mf.ok());
        lsxcommon::ArrowTokenizerDb db;
        REQUIRE(lsxcommon::load_tokenizer_metadata(*vf, *mf, db).ok());
        CHECK(db.tokenizer_kind == 0);
    }

    // 2) Attention projections (incl. the per-head gate) are NVFP4; q/k norms
    // stay raw F32.
    {
        auto names = synthtest::TensorNamesIn(base / "pinned" / "layer_3.arrow");
        bool has_q_norm = false, has_k_norm = false;
        for (auto &n : names) {
            if (n.find("self_attn.q_norm.weight") != std::string::npos)
                has_q_norm = true;
            if (n.find("self_attn.k_norm.weight") != std::string::npos)
                has_k_norm = true;
            CHECK(n.find("self_attn.q_proj.weight") == std::string::npos);
        }
        CHECK(has_q_norm);
        CHECK(has_k_norm);
        auto recs = lsxcommon::DenseNvfp4Schema::UnpackBatch(
                        lsxcommon::ReadRecordBatchFromFile(
                            base / "pinned" / "layer_3_nvfp4.arrow")
                            .ValueOrDie())
                        .ValueOrDie();
        bool q = false, o = false, g = false;
        for (auto &r : recs) {
            if (r.name.find("self_attn.q_proj.weight") != std::string::npos) {
                q = true;
                CHECK(static_cast<int64_t>(r.packed.size()) ==
                      static_cast<int64_t>(c.heads) * c.head_dim * c.hidden / 2);
            }
            if (r.name.find("self_attn.o_proj.weight") != std::string::npos)
                o = true;
            if (r.name.find("self_attn.g_proj.weight") != std::string::npos) {
                g = true;
                CHECK(static_cast<int64_t>(r.packed.size()) ==
                      static_cast<int64_t>(c.heads) * c.hidden / 2);
            }
        }
        CHECK(q);
        CHECK(o);
        CHECK(g);
    }

    // 3) Dense-MLP layer 0 carries the MLP triple; MoE layers do not.
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
                           base / "pinned" / "layer_1_nvfp4.arrow")
                           .ValueOrDie())
                       .ValueOrDie();
        for (auto &r : moe)
            CHECK(r.name.find("mlp.gate_proj.weight") == std::string::npos);
    }

    // 4) Router gate (BF16) + routing bias (F32) are stored raw.
    {
        auto recs = lsxcommon::PinnedSchema::UnpackBatch(
                        lsxcommon::ReadRecordBatchFromFile(
                            base / "router" / "layer_1.arrow")
                            .ValueOrDie())
                        .ValueOrDie();
        REQUIRE(recs.size() == 2);
        bool gate = false, bias = false;
        for (auto &r : recs) {
            if (r.name.find("moe.gate.weight") != std::string::npos) {
                gate = true;
                CHECK(r.dtype == "BF16");
                CHECK(static_cast<int64_t>(r.data.size()) ==
                      static_cast<int64_t>(c.experts) * c.hidden * 2);
            }
            if (r.name.find("router_bias") != std::string::npos) {
                bias = true;
                CHECK(r.dtype == "F32");
                CHECK(static_cast<int64_t>(r.data.size()) ==
                      static_cast<int64_t>(c.experts) * 4);
            }
        }
        CHECK(gate);
        CHECK(bias);
    }

    // 5) Embed/head/final norm stay raw in global.arrow.
    {
        auto names = synthtest::TensorNamesIn(base / "pinned" / "global.arrow");
        bool emb = false, hd = false, fn = false;
        for (auto &n : names) {
            if (n.find("embed_tokens.weight") != std::string::npos) emb = true;
            if (n.find("lm_head.weight") != std::string::npos) hd = true;
            if (n.find("model.norm.weight") != std::string::npos) fn = true;
        }
        CHECK(emb);
        CHECK(hd);
        CHECK(fn);
    }

    // 6) Fused FP8 stacks are split per expert into the neutral store layout.
    CHECK(fs::exists(base / "shared" / "layer_1_nvfp4.arrow"));
    CHECK_FALSE(fs::exists(base / "shared" / "layer_0_nvfp4.arrow"));
    CHECK_FALSE(fs::exists(base / "router" / "layer_0.arrow"));
    // DSA indexer tensors are stored for the >4096 stage on full layers only.
    CHECK(fs::exists(base / "indexer" / "layer_3.arrow"));
    CHECK_FALSE(fs::exists(base / "indexer" / "layer_1.arrow"));
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
        CHECK(static_cast<int64_t>(pe.w1_scales.size()) == n / 16);
        CHECK(pe.w1_packed != pe.w3_packed);
    }

    // 7) Determinism: a second conversion produces a byte-identical index.
    {
        fs::path out2 = fs::temp_directory_path() / "lsx_step_out2";
        fs::remove_all(out2);
        REQUIRE(ConvertStepMoeModel(in.string(), out2.string()).ok());
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
