// RecursiveSI/flint converter stage (Model Ingestion Protocol stages 1/2):
// config parsing, tensor classification, quantization policy, and a synthetic
// end-to-end conversion (label `convert`, bounded TIMEOUT — the frozen
// geometry gate is relaxed with LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY so the
// fixture never touches the real 75 GB checkpoint).
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "flint_moe_constants.h"
#include "lsxcommon/flint_moe_convert.h"
#include "test_flint_moe_fixture.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

using namespace lsxcommon::flint;
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

TEST_CASE("ParseFlintConfig reads the geometry, the layer tables and the eos list",
          "[flint-moe-convert]")
{
    fs::path dir = fs::temp_directory_path() / "lsx_flint_cfg";
    fs::remove_all(dir);
    fs::create_directories(dir);
    flint_test::WriteConfig(dir);
    auto top = ParseFlintConfig(dir.string());
    REQUIRE(top.error_msg.empty());
    const auto &c = flint_test::TestCfg();
    CHECK(top.hidden_size == c.hidden);
    CHECK(top.num_layers == c.layers);
    CHECK(top.num_attention_heads == c.heads);
    CHECK(top.num_key_value_heads == c.kv_heads);
    CHECK(top.head_dim == c.head_dim);
    CHECK(top.vocab_size == c.vocab);
    CHECK(top.dense_intermediate_size == c.dense_inter);
    CHECK(top.moe_intermediate_size == c.moe_inter);
    CHECK(top.shared_intermediate_size == c.shared_inter);
    CHECK(top.num_experts == c.experts);
    CHECK(top.num_experts_per_tok == c.top_k);
    CHECK(top.sliding_window == c.window);
    CHECK(top.rms_norm_eps == Catch::Approx(1e-6));
    CHECK(top.rope_theta == Catch::Approx(10000.0));
    REQUIRE(top.layer_types.size() == static_cast<size_t>(c.layers));
    CHECK(top.layer_types[static_cast<size_t>(c.full_attn_layer)] ==
          "full_attention");
    CHECK(top.layer_types[0] == "sliding_attention");
    REQUIRE(top.no_rope_layers.size() == 1);
    CHECK(top.no_rope_layers[0] == c.full_attn_layer);
    REQUIRE(top.mlp_only_layers.size() == 1);
    CHECK(top.mlp_only_layers[0] == 0);
    REQUIRE(top.eos_ids.size() == 2);
    CHECK(top.eos_ids[0] == c.eos);
    CHECK(top.eos_ids[1] == c.eos2);
    CHECK(top.bos_token_id == c.bos);
    CHECK(top.pad_token_id == c.pad);
    std::error_code ec;
    fs::remove_all(dir, ec);
}

TEST_CASE("ParseFlintConfig fails on a missing config", "[flint-moe-convert]")
{
    auto top = ParseFlintConfig("/nonexistent_dir_lsx_flint");
    REQUIRE_FALSE(top.error_msg.empty());
}

TEST_CASE("ParseFlintConfig rejects a wrong-length layer_types table",
          "[flint-moe-convert]")
{
    fs::path dir = fs::temp_directory_path() / "lsx_flint_cfg_bad";
    fs::remove_all(dir);
    fs::create_directories(dir);
    flint_test::WriteConfig(dir);
    std::ofstream f(dir / "config.json");
    f << R"({"model_type":"flint","hidden_size":64,"num_hidden_layers":4,)"
      << R"("num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,)"
      << R"("vocab_size":64,"intermediate_size":32,"moe_intermediate_size":16,)"
      << R"("shared_expert_intermediate_size":16,"num_experts":8,)"
      << R"("num_experts_per_tok":2,"sliding_window":8,)"
      << R"("layer_types":["sliding_attention","full_attention"]})"
      << "\n";
    f.close();
    auto top = ParseFlintConfig(dir.string());
    CHECK_FALSE(top.error_msg.empty());
    std::error_code ec;
    fs::remove_all(dir, ec);
}

TEST_CASE("ParseFlintConfig rejects a semantics drift the kernels cannot express",
          "[flint-moe-convert]")
{
    const auto rewrite = [](const char *needle, const char *replacement) {
        fs::path dir = fs::temp_directory_path() / "lsx_flint_cfg_drift";
        fs::remove_all(dir);
        fs::create_directories(dir);
        flint_test::WriteConfig(dir);
        std::string j = ReadAll(dir / "config.json");
        const auto at = j.find(needle);
        REQUIRE(at != std::string::npos);
        j.replace(at, std::char_traits<char>::length(needle), replacement);
        std::ofstream(dir / "config.json") << j;
        return ParseFlintConfig(dir.string());
    };
    // hidden_act != silu would mean a different activation than the SwiGLU
    // kernel implements.
    CHECK_FALSE(rewrite("\"hidden_act\":\"silu\"", "\"hidden_act\":\"gelu\"")
                    .error_msg.empty());
    // softmax_type != learnable means the checkpoint's sink has no defined role.
    CHECK_FALSE(
        rewrite("\"softmax_type\":\"learnable\"", "\"softmax_type\":\"softmax\"")
            .error_msg.empty());
    // norm_topk_prob false would leave the routed weights un-renormalized.
    CHECK_FALSE(rewrite("\"norm_topk_prob\":true", "\"norm_topk_prob\":false")
                    .error_msg.empty());
    // attention_bias true would need bias tensors the checkpoint does not carry.
    CHECK_FALSE(rewrite("\"attention_bias\":false", "\"attention_bias\":true")
                    .error_msg.empty());
    std::error_code ec;
    fs::remove_all(fs::temp_directory_path() / "lsx_flint_cfg_drift", ec);
}

TEST_CASE("flint tensor classification", "[flint-moe-convert]")
{
    using K = FlintTensorKind;
    auto cl = [](const std::string &n) { return classify_flint_tensor(n); };
    CHECK(cl("model.embed_tokens.weight").kind == K::Embed);
    CHECK(cl("lm_head.weight").kind == K::LmHead);
    CHECK(cl("model.norm.weight").kind == K::FinalNorm);
    CHECK(cl("model.layers.3.self_attn.q_proj.weight").kind == K::AttnQ);
    CHECK(cl("model.layers.3.self_attn.q_proj.weight").layer == 3);
    CHECK(cl("model.layers.3.self_attn.k_norm.weight").kind == K::AttnKNorm);
    // The learned sink is its own kind, never a norm and never quantized.
    CHECK(cl("model.layers.3.self_attn.softmax_offset").kind == K::SinkLogit);
    CHECK(cl("model.layers.3.self_attn.softmax_offset").layer == 3);
    CHECK(cl("model.layers.0.mlp.gate_proj.weight").kind == K::DenseMlpGate);
    CHECK(cl("model.layers.0.mlp.down_proj.weight").kind == K::DenseMlpDown);
    // Flint's router tensor is `mlp.gate.weight` (no `router.` segment).
    CHECK(cl("model.layers.3.mlp.gate.weight").kind == K::RouterGate);
    CHECK(cl("model.layers.3.mlp.shared_expert.down_proj.weight").kind ==
          K::SharedDownProj);
    CHECK(cl("model.layers.3.mlp.shared_expert.up_proj.weight").kind ==
          K::SharedUpProj);
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
    CHECK(cl("model.layers.5.mlp.experts.0.softmax_offset").kind ==
          K::Unknown);
    CHECK(cl("mtp.0.whatever.weight").kind == K::Unknown);
    CHECK(cl("model.visual.blocks.0.weight").kind == K::Unknown);
    CHECK(cl("totally.unknown.name").kind == K::Unknown);
}

TEST_CASE("flint quantization policy keeps sinks, norms, router and edges exact",
          "[flint-moe-convert]")
{
    using K = FlintTensorKind;
    for (K k : {K::AttnQ, K::AttnK, K::AttnV, K::AttnO, K::DenseMlpGate,
                K::DenseMlpUp, K::DenseMlpDown, K::ExpertGate, K::ExpertDown,
                K::ExpertUp, K::SharedGateProj, K::SharedUpProj,
                K::SharedDownProj})
        CHECK(flint_tensor_is_nvfp4(k));
    // Embed/head stay FP16 on purpose (phi-4 / gemma-4 rule, and the vendor
    // serves the head in FP32).
    CHECK_FALSE(flint_tensor_is_nvfp4(K::Embed));
    CHECK_FALSE(flint_tensor_is_nvfp4(K::LmHead));
    for (K k : {K::FinalNorm, K::InputNorm, K::PostNorm, K::AttnQNorm,
                K::AttnKNorm, K::SinkLogit, K::RouterGate, K::Unknown})
        CHECK_FALSE(flint_tensor_is_nvfp4(k));
    for (K k : {K::ExpertGate, K::ExpertDown, K::ExpertUp})
        CHECK(flint_tensor_is_routed_expert(k));
    CHECK_FALSE(flint_tensor_is_routed_expert(K::SharedGateProj));
    CHECK_FALSE(flint_tensor_is_routed_expert(K::AttnQ));
}

TEST_CASE("the frozen passport's layer tables match the checkpoint",
          "[flint-moe-convert]")
{
    using namespace lsxcommon::flint::frozen;
    CHECK(kLayers == 49);
    CHECK(kNumFullAttnLayers == 14);
    for (int l = 0; l < kLayers; ++l) {
        // Full attention and no-RoPE are the SAME set in this checkpoint.
        CHECK(IsFullAttn(l) == SkipsRope(l));
        CHECK(IsMoeLayer(l) == (l >= 2));
    }
    CHECK(IsFullAttn(0));
    CHECK(IsFullAttn(1));
    CHECK_FALSE(IsFullAttn(2));
    CHECK(IsFullAttn(48));
    CHECK(IsEosId(150001));
    CHECK(IsEosId(150018));
    CHECK_FALSE(IsEosId(150000));
}

TEST_CASE("ConvertFlintModel converts a synthetic checkpoint",
          "[flint-moe-convert]")
{
    const char *prev = std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY");
    setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1", 1);
    fs::path in = fs::temp_directory_path() / "lsx_flint_in";
    fs::path out = fs::temp_directory_path() / "lsx_flint_out";
    fs::remove_all(in);
    fs::remove_all(out);
    REQUIRE(flint_test::BuildSyntheticCheckpoint(in) > 0);

    auto st = ConvertFlintModel(in.string(), out.string());
    if (!st.ok()) WARN(st.ToString());
    REQUIRE(st.ok());

    const fs::path base = out;
    const auto &c = flint_test::TestCfg();

    // 1) Index carries the format tag, the layer tables and the frozen numbers.
    {
        REQUIRE(fs::exists(base / "model.index.json"));
        const std::string j = ReadAll(base / "model.index.json");
        CHECK(j.find("\"layout\": 2") != std::string::npos);
        CHECK(j.find("\"model_type\": \"flint\"") != std::string::npos);
        CHECK(j.find("\"num_experts\": " + std::to_string(c.experts)) !=
              std::string::npos);
        CHECK(j.find("\"hidden_size\": " + std::to_string(c.hidden)) !=
              std::string::npos);
        CHECK(j.find("\"sliding_window\": " + std::to_string(c.window)) !=
              std::string::npos);
        CHECK(j.find("\"full_attention_layers\": [3]") != std::string::npos);
        CHECK(j.find("\"no_rope_layers\": [3]") != std::string::npos);
        CHECK(j.find("\"mlp_only_layers\": [0]") != std::string::npos);
        CHECK(j.find("\"eos_token_ids\": [269,271]") != std::string::npos);
        CHECK(j.find("\"vram\"") != std::string::npos);
        CHECK(j.find("\"total_experts\": " +
                     std::to_string(c.experts * (c.layers - c.first_moe_layer))) !=
              std::string::npos);
    }

    // 2) Attention projections are NVFP4; the head norms and the learned sink
    // stay raw BF16 in the pinned layer file.
    {
        auto names = synthtest::TensorNamesIn(base / "pinned" / "layer_3.arrow");
        bool has_q_norm = false, has_k_norm = false, has_sink = false;
        for (auto &n : names) {
            if (n.find("self_attn.q_norm.weight") != std::string::npos)
                has_q_norm = true;
            if (n.find("self_attn.k_norm.weight") != std::string::npos)
                has_k_norm = true;
            if (n.find("self_attn.softmax_offset") != std::string::npos)
                has_sink = true;
        }
        CHECK(has_q_norm);
        CHECK(has_k_norm);
        CHECK(has_sink);
        auto recs = lsxcommon::DenseNvfp4Schema::UnpackBatch(
                        lsxcommon::ReadRecordBatchFromFile(
                            base / "pinned" / "layer_3_nvfp4.arrow")
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
            CHECK(r.name.find("softmax_offset") == std::string::npos);
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

    // 4) Router gate is stored raw (BF16) and never quantized.
    {
        auto recs = lsxcommon::PinnedSchema::UnpackBatch(
                        lsxcommon::ReadRecordBatchFromFile(
                            base / "router" / "layer_2.arrow")
                            .ValueOrDie())
                        .ValueOrDie();
        REQUIRE(recs.size() == 1);
        CHECK(recs[0].name.find("mlp.gate.weight") != std::string::npos);
        CHECK(recs[0].dtype == "BF16");
        CHECK(static_cast<int64_t>(recs[0].data.size()) ==
              static_cast<int64_t>(c.experts) * c.hidden * 2);
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
    }

    // 6) Shared expert + routed experts exist only on MoE layers.
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
        CHECK(static_cast<int64_t>(pe.w1_scales.size()) == (n / 16) * 1);
        CHECK(pe.w1_packed != pe.w3_packed);
    }

    // 7) Residual plane placement: attention + dense MLP + shared expert carry
    // a second NVFP4 plane (`<name>.res`), and the routed experts do NOT (the
    // decode read volume stays untouched).
    {
        auto recs = lsxcommon::DenseNvfp4Schema::UnpackBatch(
                        lsxcommon::ReadRecordBatchFromFile(
                            base / "pinned" / "layer_3_nvfp4.arrow")
                            .ValueOrDie())
                        .ValueOrDie();
        bool q_res = false, o_res = false;
        for (auto &r : recs) {
            if (r.name.find("self_attn.q_proj.weight.res") != std::string::npos)
                q_res = true;
            if (r.name.find("self_attn.o_proj.weight.res") != std::string::npos)
                o_res = true;
        }
        CHECK(q_res);
        CHECK(o_res);
        auto shared = lsxcommon::DenseNvfp4Schema::UnpackBatch(
                          lsxcommon::ReadRecordBatchFromFile(
                              base / "shared" / "layer_2_nvfp4.arrow")
                              .ValueOrDie())
                          .ValueOrDie();
        bool sh_res = false;
        for (auto &r : shared)
            if (r.name.find("shared_expert.down_proj.weight.res") !=
                std::string::npos)
                sh_res = true;
        CHECK(sh_res);
        auto experts = lsxcommon::ExpertSchema::UnpackBatch(
                           lsxcommon::ReadRecordBatchFromFile(
                               base / "experts" / "layer_1" / "exp_0.arrow")
                               .ValueOrDie())
                           .ValueOrDie();
        CHECK(static_cast<int64_t>(experts.w1_packed.size()) ==
              static_cast<int64_t>(c.moe_inter) * c.hidden / 2);
        // Placement policy is exposed by the converter itself.
        CHECK(flint_tensor_has_residual(FlintTensorKind::AttnQ));
        CHECK(flint_tensor_has_residual(FlintTensorKind::SharedDownProj));
        CHECK_FALSE(flint_tensor_has_residual(FlintTensorKind::ExpertGate));
        CHECK_FALSE(flint_tensor_has_residual(FlintTensorKind::RouterGate));
        CHECK_FALSE(flint_tensor_has_residual(FlintTensorKind::SinkLogit));
        CHECK_FALSE(flint_tensor_has_residual(FlintTensorKind::Embed));
    }

    // 8) Determinism: a second conversion produces a byte-identical index.
    {
        fs::path out2 = fs::temp_directory_path() / "lsx_flint_out2";
        fs::remove_all(out2);
        REQUIRE(ConvertFlintModel(in.string(), out2.string()).ok());
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
