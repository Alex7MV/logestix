// Hy-Memory-30B-A3B converter stage (Model Ingestion Protocol stage 1/2):
// config parsing, tensor classification, quantization policy, and a synthetic
// end-to-end conversion (label `convert`, bounded TIMEOUT — the frozen
// geometry gate is relaxed with LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY so the
// fixture never touches the real 60 GB checkpoint).
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/hy_moe_convert.h"
#include "test_hy_moe_fixture.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

using namespace lsxcommon::hy;
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

TEST_CASE("ParseHyConfig reads the flat hy_v3 fields and the MLP dispatch",
          "[hy-moe-convert]")
{
    fs::path dir = fs::temp_directory_path() / "lsx_hy_cfg";
    fs::remove_all(dir);
    fs::create_directories(dir);
    hy_test::WriteConfig(dir);
    auto top = ParseHyConfig(dir.string());
    REQUIRE(top.error_msg.empty());
    const auto &c = hy_test::TestCfg();
    CHECK(top.hidden_size == c.hidden);
    CHECK(top.num_layers == c.layers);
    CHECK(top.num_attention_heads == c.heads);
    CHECK(top.num_key_value_heads == c.kv_heads);
    CHECK(top.head_dim == c.head_dim);
    CHECK(top.rotary_dim == c.rotary_dim);
    CHECK(top.vocab_size == c.vocab);
    // The expert width comes from `expert_hidden_dim`, NOT the vestigial
    // `moe_intermediate_size` (1536 in the shipped config vs 768 tensors).
    CHECK(top.moe_intermediate_size == c.moe_inter);
    CHECK(top.shared_intermediate_size == c.shared_inter);
    CHECK(top.dense_intermediate_size == c.dense_inter);
    CHECK(top.num_experts == c.experts);
    CHECK(top.num_experts_per_tok == c.top_k);
    CHECK(top.rms_norm_eps == Catch::Approx(1e-5));
    CHECK(top.rope_theta == Catch::Approx(11158840.0));
    CHECK(top.router_scaling_factor == Catch::Approx(c.router_scaling_factor));
    CHECK(top.bos_token_id == 3);
    CHECK(top.eos_token_id == 4);
    CHECK(top.pad_token_id == 5);
    REQUIRE(top.mlp_types.size() == static_cast<size_t>(c.layers));
    CHECK(top.mlp_types[0] == "dense");
    CHECK(top.mlp_types[1] == "sparse");
    std::error_code ec;
    fs::remove_all(dir, ec);
}

TEST_CASE("ParseHyConfig fails on a missing config", "[hy-moe-convert]")
{
    auto top = ParseHyConfig("/nonexistent_dir_lsx_hy");
    REQUIRE_FALSE(top.error_msg.empty());
}

TEST_CASE("ParseHyConfig rejects a layer dispatch of the wrong length",
          "[hy-moe-convert]")
{
    fs::path dir = fs::temp_directory_path() / "lsx_hy_cfg_bad";
    fs::remove_all(dir);
    fs::create_directories(dir);
    hy_test::WriteConfig(dir);
    std::ofstream f(dir / "config.json");
    f << R"({"model_type":"hy_v3","hidden_size":64,"num_hidden_layers":4,)"
      << R"("num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,)"
      << R"("vocab_size":64,"intermediate_size":32,"expert_hidden_dim":16,)"
      << R"("num_experts":8,"num_experts_per_tok":2,)"
      << R"("mlp_layer_types":["dense","sparse"]})"
      << "\n";
    f.close();
    auto top = ParseHyConfig(dir.string());
    CHECK_FALSE(top.error_msg.empty());
    std::error_code ec;
    fs::remove_all(dir, ec);
}

TEST_CASE("Hy-Memory tensor classification", "[hy-moe-convert]")
{
    using K = HyTensorKind;
    auto cl = [](const std::string &n) { return classify_hy_tensor(n); };
    CHECK(cl("model.embed_tokens.weight").kind == K::Embed);
    CHECK(cl("lm_head.weight").kind == K::LmHead);
    CHECK(cl("model.norm.weight").kind == K::FinalNorm);
    CHECK(cl("model.layers.3.self_attn.q_proj.weight").kind == K::AttnQ);
    CHECK(cl("model.layers.3.self_attn.q_proj.weight").layer == 3);
    CHECK(cl("model.layers.3.self_attn.k_norm.weight").kind == K::AttnKNorm);
    CHECK(cl("model.layers.0.mlp.gate_proj.weight").kind == K::DenseMlpGate);
    CHECK(cl("model.layers.0.mlp.down_proj.weight").kind == K::DenseMlpDown);
    CHECK(cl("model.layers.3.mlp.router.gate.weight").kind == K::RouterGate);
    CHECK(cl("model.layers.3.mlp.expert_bias").kind == K::RouterBias);
    CHECK(cl("model.layers.3.mlp.shared_mlp.down_proj.weight").kind ==
          K::SharedDownProj);
    {
        auto p = cl("model.layers.5.mlp.experts.42.gate_proj.weight");
        CHECK(p.kind == K::ExpertGate);
        CHECK(p.layer == 5);
        CHECK(p.expert == 42);
    }
    CHECK(cl("model.layers.5.mlp.experts.42.down_proj.weight").kind ==
          K::ExpertDown);
    CHECK(cl("model.layers.5.mlp.experts.42.up_proj.weight").kind ==
          K::ExpertUp);
    // A routed-expert leaf clashing with the dense MLP leaf must be told apart
    // by the path, not the suffix.
    CHECK(cl("model.layers.5.mlp.gate_proj.weight").kind == K::DenseMlpGate);
    CHECK(cl("mtp.0.whatever.weight").kind == K::Unknown);
    CHECK(cl("model.visual.blocks.0.weight").kind == K::Unknown);
    CHECK(cl("totally.unknown.name").kind == K::Unknown);
}

TEST_CASE("Hy-Memory quantization policy keeps norms, embed and the router exact",
          "[hy-moe-convert]")
{
    using K = HyTensorKind;
    for (K k : {K::AttnQ, K::AttnK, K::AttnV, K::AttnO, K::DenseMlpGate,
                K::DenseMlpUp, K::DenseMlpDown, K::ExpertGate, K::ExpertDown,
                K::ExpertUp, K::SharedGateProj, K::SharedUpProj,
                K::SharedDownProj})
        CHECK(hy_tensor_is_nvfp4(k));
    // Embed/head stay FP16 on purpose (phi-4 / gemma-4 / MiniMax rule).
    CHECK_FALSE(hy_tensor_is_nvfp4(K::Embed));
    CHECK_FALSE(hy_tensor_is_nvfp4(K::LmHead));
    for (K k : {K::FinalNorm, K::InputNorm, K::PostNorm, K::AttnQNorm,
                K::AttnKNorm, K::RouterGate, K::RouterBias, K::Unknown})
        CHECK_FALSE(hy_tensor_is_nvfp4(k));
    for (K k : {K::ExpertGate, K::ExpertDown, K::ExpertUp})
        CHECK(hy_tensor_is_routed_expert(k));
    CHECK_FALSE(hy_tensor_is_routed_expert(K::SharedGateProj));
    CHECK_FALSE(hy_tensor_is_routed_expert(K::AttnQ));
}

TEST_CASE("ConvertHyMoeModel converts a synthetic checkpoint",
          "[hy-moe-convert]")
{
    const char *prev = std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY");
    setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1", 1);
    fs::path in = fs::temp_directory_path() / "lsx_hy_in";
    fs::path out = fs::temp_directory_path() / "lsx_hy_out";
    fs::remove_all(in);
    fs::remove_all(out);
    REQUIRE(hy_test::BuildSyntheticCheckpoint(in) > 0);

    auto st = ConvertHyMoeModel(in.string(), out.string());
    if (!st.ok()) WARN(st.ToString());
    REQUIRE(st.ok());

    const fs::path base = out / "logestix";
    const auto &c = hy_test::TestCfg();

    // 1) Index carries the format tag, dispatch and frozen numbers.
    {
        REQUIRE(fs::exists(base / "model.index.json"));
        const std::string j = ReadAll(base / "model.index.json");
        CHECK(j.find("\"logestix-hy-nvfp4\"") != std::string::npos);
        CHECK(j.find("\"model_type\": \"hy-memory-30b-a3b\"") !=
              std::string::npos);
        CHECK(j.find("\"num_experts\": " + std::to_string(c.experts)) !=
              std::string::npos);
        CHECK(j.find("\"hidden_size\": " + std::to_string(c.hidden)) !=
              std::string::npos);
        CHECK(j.find("\"rotary_dim\": " + std::to_string(c.rotary_dim)) !=
              std::string::npos);
        CHECK(j.find("\"mlp_layer_types\"") != std::string::npos);
        CHECK(j.find("\"first_moe_layer\": 1") != std::string::npos);
        CHECK(j.find("\"bos_token_id\": 3") != std::string::npos);
        CHECK(j.find("\"eos_token_id\": 4") != std::string::npos);
        CHECK(j.find("\"vram\"") != std::string::npos);
    }

    // 2) Attention projections are NVFP4; the per-head norms stay raw BF16.
    {
        auto names = synthtest::TensorNamesIn(base / "pinned" / "layer_2.arrow");
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
                            base / "pinned" / "layer_2_nvfp4.arrow")
                            .ValueOrDie())
                        .ValueOrDie();
        bool q = false, o = false;
        for (auto &r : recs) {
            if (r.name.find("self_attn.q_proj.weight") != std::string::npos) {
                q = true;
                CHECK(static_cast<int64_t>(r.packed.size()) ==
                      static_cast<int64_t>(c.heads) * c.head_dim * c.hidden / 2);
            }
            if (r.name.find("self_attn.o_proj.weight") != std::string::npos)
                o = true;
        }
        CHECK(q);
        CHECK(o);
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
                           base / "pinned" / "layer_2_nvfp4.arrow")
                           .ValueOrDie())
                       .ValueOrDie();
        bool moe_has_mlp = false;
        for (auto &r : moe)
            if (r.name.find("mlp.gate_proj.weight") != std::string::npos)
                moe_has_mlp = true;
        CHECK_FALSE(moe_has_mlp);
    }

    // 4) Router gate (BF16) + expert bias (F32) are stored raw, never quantized.
    {
        auto recs = lsxcommon::PinnedSchema::UnpackBatch(
                        lsxcommon::ReadRecordBatchFromFile(
                            base / "router" / "layer_2.arrow")
                            .ValueOrDie())
                        .ValueOrDie();
        REQUIRE(recs.size() == 2);
        bool gate = false, bias = false;
        for (auto &r : recs) {
            if (r.name.find("router.gate.weight") != std::string::npos) {
                gate = true;
                CHECK(r.dtype == "BF16");
                CHECK(static_cast<int64_t>(r.data.size()) ==
                      static_cast<int64_t>(c.experts) * c.hidden * 2);
            }
            if (r.name.find("expert_bias") != std::string::npos) {
                bias = true;
                CHECK(r.dtype == "F32");
                CHECK(static_cast<int64_t>(r.data.size()) ==
                      static_cast<int64_t>(c.experts) * 4);
            }
        }
        CHECK(gate);
        CHECK(bias);
    }

    // 5) Embed + head stay FP16 (raw BF16 in global.arrow, no NVFP4 record).
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

    // 6) Shared expert + routed experts exist where the loader expects them.
    CHECK(fs::exists(base / "shared" / "layer_2_nvfp4.arrow"));
    CHECK_FALSE(fs::exists(base / "shared" / "layer_0_nvfp4.arrow"));
    CHECK_FALSE(fs::exists(base / "router" / "layer_0.arrow"));
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
        // ExpertSchema carries no shape columns; the byte counts pin the
        // geometry the loader derives from the passport.
        CHECK(static_cast<int64_t>(pe.w1_scales.size()) ==
              (n / 16) * 1);
        // gate and up are distinct source tensors.
        CHECK(pe.w1_packed != pe.w3_packed);
    }

    // 7) Determinism: a second conversion produces a byte-identical index.
    {
        fs::path out2 = fs::temp_directory_path() / "lsx_hy_out2";
        fs::remove_all(out2);
        REQUIRE(ConvertHyMoeModel(in.string(), out2.string()).ok());
        CHECK(ReadAll(base / "model.index.json") ==
              ReadAll(out2 / "logestix" / "model.index.json"));
        fs::remove_all(out2);
    }

    std::error_code ec;
    fs::remove_all(in, ec);
    fs::remove_all(out, ec);
    if (prev) setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", prev, 1);
    else unsetenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY");
}
