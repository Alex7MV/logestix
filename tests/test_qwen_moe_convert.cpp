#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "test_qwen_moe_fixture.h"
#include "lsxcommon/qwen_moe_convert.h"

#include <filesystem>

using namespace lsxcommon::qwen_moe;
namespace fs = std::filesystem;

TEST_CASE("ParseQwenMoeConfig reads text_config MoE fields", "[qwen-moe-convert]")
{
    fs::path dir = fs::temp_directory_path() / "lsx_qwen_moe_cfg";
    fs::create_directories(dir);
    qwen_moe_test::WriteMoeConfig(dir);
    auto top = ParseQwenMoeConfig(dir.string());
    REQUIRE(top.error_msg.empty());
    REQUIRE(top.hidden_size == 64);
    REQUIRE(top.num_layers == 4);
    REQUIRE(top.num_experts == 8);
    REQUIRE(top.num_experts_per_tok == 2);
    REQUIRE(top.moe_intermediate_size == 16);
    REQUIRE(top.shared_expert_intermediate_size == 16);
    REQUIRE(top.num_attention_heads == 4);
    REQUIRE(top.num_key_value_heads == 1);
    REQUIRE(top.head_dim == 32);
    REQUIRE(top.linear_num_key_heads == 2);
    REQUIRE(top.linear_num_value_heads == 4);
    REQUIRE(top.linear_key_head_dim == 8);
    REQUIRE(top.linear_value_head_dim == 8);
    REQUIRE(top.vocab_size == 64);
    REQUIRE(top.eos_token_id == 7);
    REQUIRE(top.rms_norm_eps == Catch::Approx(1e-6));
    REQUIRE(top.partial_rotary_factor == Catch::Approx(0.25));
    REQUIRE(top.rope_theta == Catch::Approx(10000.0));
    REQUIRE(top.layer_types.size() == 4);
    REQUIRE(top.layer_types[3] == "full_attention");
    std::error_code ec;
    fs::remove_all(dir, ec);
}

TEST_CASE("ParseQwenMoeConfig fails on missing config", "[qwen-moe-convert]")
{
    auto top = ParseQwenMoeConfig("/nonexistent_dir_lsx_qwen_moe");
    REQUIRE_FALSE(top.error_msg.empty());
}

TEST_CASE("MoE tensor classification", "[qwen-moe-convert]")
{
    using K = lsxcommon::qwen_moe::MoeTensorKind;
    auto cl = [](const std::string &n) {
        return lsxcommon::qwen_moe::classify_qwen_moe_tensor(n);
    };
    CHECK(cl("model.language_model.embed_tokens.weight").kind == K::Embed);
    CHECK(cl("lm_head.weight").kind == K::LmHead);
    CHECK(cl("model.language_model.norm.weight").kind == K::FinalNorm);
    CHECK(cl("model.language_model.layers.3.self_attn.q_proj.weight").kind ==
          K::FullAttnQ);
    CHECK(cl("model.language_model.layers.3.self_attn.q_proj.weight").layer == 3);
    CHECK(cl("model.language_model.layers.0.linear_attn.in_proj_qkv.weight")
              .kind == K::LinearAttnQkv);
    CHECK(cl("model.language_model.layers.0.linear_attn.A_log").layer == 0);
    CHECK(cl("model.language_model.layers.7.self_attn.q_norm.weight").kind ==
          K::FullAttnQNorm);
    CHECK(cl("model.language_model.layers.2.input_layernorm.weight").kind ==
          K::InputNorm);
    CHECK(cl("model.language_model.layers.2.post_attention_layernorm.weight")
              .kind == K::PostNorm);
    CHECK(cl("model.language_model.layers.5.mlp.gate.weight").kind ==
          K::RouterGate);
    CHECK(cl("model.language_model.layers.5.mlp.experts.gate_up_proj").kind ==
          K::FusedExpertsGateUp);
    CHECK(cl("model.language_model.layers.5.mlp.experts.gate_up_proj").layer == 5);
    CHECK(cl("model.language_model.layers.5.mlp.experts.down_proj").kind ==
          K::FusedExpertsDown);
    CHECK(cl("model.language_model.layers.1.mlp.shared_expert.gate_proj.weight")
              .kind == K::SharedExpertGateProj);
    CHECK(cl("model.language_model.layers.1.mlp.shared_expert.up_proj.weight")
              .kind == K::SharedExpertUpProj);
    CHECK(cl("model.language_model.layers.1.mlp.shared_expert.down_proj.weight")
              .kind == K::SharedExpertDownProj);
    CHECK(cl("model.language_model.layers.1.mlp.shared_expert_gate.weight")
              .kind == K::SharedExpertGate);
    CHECK(cl("mtp.fc.weight").kind == K::Mtp);
    CHECK(cl("model.visual.blocks.0.attn.qkv.weight").kind == K::Visual);
    CHECK(cl("totally.unknown.name").kind == K::Unknown);
}

TEST_CASE("MoE quantization policy keeps norms/router/gates high precision",
          "[qwen-moe-convert]")
{
    using K = lsxcommon::qwen_moe::MoeTensorKind;
    using lsxcommon::qwen_moe::qwen_moe_tensor_is_nvfp4;
    CHECK(qwen_moe_tensor_is_nvfp4(K::FullAttnQ));
    CHECK(qwen_moe_tensor_is_nvfp4(K::FullAttnK));
    CHECK(qwen_moe_tensor_is_nvfp4(K::FullAttnV));
    CHECK(qwen_moe_tensor_is_nvfp4(K::FullAttnO));
    CHECK(qwen_moe_tensor_is_nvfp4(K::LinearAttnOut));
    CHECK(qwen_moe_tensor_is_nvfp4(K::FusedExpertsGateUp));
    CHECK(qwen_moe_tensor_is_nvfp4(K::FusedExpertsDown));
    CHECK(qwen_moe_tensor_is_nvfp4(K::SharedExpertGateProj));
    CHECK(qwen_moe_tensor_is_nvfp4(K::SharedExpertUpProj));
    CHECK(qwen_moe_tensor_is_nvfp4(K::SharedExpertDownProj));
    CHECK(qwen_moe_tensor_is_nvfp4(K::Embed));
    CHECK(qwen_moe_tensor_is_nvfp4(K::LmHead));
    CHECK_FALSE(qwen_moe_tensor_is_nvfp4(K::RouterGate));
    CHECK_FALSE(qwen_moe_tensor_is_nvfp4(K::SharedExpertGate));
    CHECK_FALSE(qwen_moe_tensor_is_nvfp4(K::InputNorm));
    CHECK_FALSE(qwen_moe_tensor_is_nvfp4(K::PostNorm));
    CHECK_FALSE(qwen_moe_tensor_is_nvfp4(K::FullAttnQNorm));
    CHECK_FALSE(qwen_moe_tensor_is_nvfp4(K::FullAttnKNorm));
    CHECK_FALSE(qwen_moe_tensor_is_nvfp4(K::FinalNorm));
    CHECK_FALSE(qwen_moe_tensor_is_nvfp4(K::LinearAttnConv));
    CHECK_FALSE(qwen_moe_tensor_is_nvfp4(K::LinearAttnALog));
    CHECK_FALSE(qwen_moe_tensor_is_nvfp4(K::LinearAttnDt));
}

TEST_CASE("ConvertQwen35MoeModel converts synthetic checkpoint",
          "[qwen-moe-convert]")
{
    fs::path in = fs::temp_directory_path() / "lsx_qwen_moe_in";
    fs::path out = fs::temp_directory_path() / "lsx_qwen_moe_out";
    fs::remove_all(in); fs::remove_all(out);
    REQUIRE(qwen_moe_test::BuildSyntheticMoeCheckpoint(in) > 0);

    auto st = ConvertQwen35MoeModel(in.string(), out.string());
    if (!st.ok()) WARN(st.ToString());
    REQUIRE(st.ok());

    // Unified vault convention (matches dense converters): <out>/logestix.
    const fs::path base = out / "logestix";

    // 1) Index with format tag and MoE fields.
    REQUIRE(fs::exists(base / "model.index.json"));
    {
        std::ifstream f(base / "model.index.json");
        std::stringstream ss; ss << f.rdbuf();
        const std::string j = ss.str();
        CHECK(j.find("\"logestix-qwen35moe\"") != std::string::npos);
        CHECK(j.find("\"num_experts\": 8") != std::string::npos);
        CHECK(j.find("\"hidden_size\": 64") != std::string::npos);
        CHECK(j.find("\"skipped_visual_tensors\": 2") != std::string::npos);
    }

    // 2) Expert split: gate/up halves differ (fused rows were split).
    {
        auto batch = lsxcommon::ReadRecordBatchFromFile(
            base / "experts" / "layer_0" / "exp_3.arrow").ValueOrDie();
        auto pe = lsxcommon::ExpertSchema::UnpackBatch(batch).ValueOrDie();
        const int64_t elems = 16 * 64;  // I*H
        // ExpertSchema stores packed payloads only; shapes come from topology.
        CHECK(pe.w1_packed.size() * 2 >= static_cast<size_t>(elems));
        CHECK(pe.w2_packed.size() * 2 >= static_cast<size_t>(elems));
        // Distinct deterministic input data => halves must differ.
        CHECK(pe.w1_packed != pe.w3_packed);
    }

    // 3) Dense layer records keep original names.
    {
        auto names = qwentest::TensorNamesIn(base / "pinned" / "layer_0.arrow");
        bool has_norm = false;
        for (auto &n : names)
            if (n.find("input_layernorm") != std::string::npos) has_norm = true;
        CHECK(has_norm);
    }

    // 4) Router stored raw BF16: E*H elements x 2 bytes.
    {
        auto batch = lsxcommon::ReadRecordBatchFromFile(
            base / "router" / "layer_0.arrow").ValueOrDie();
        auto recs = lsxcommon::PinnedSchema::UnpackBatch(batch).ValueOrDie();
        REQUIRE(recs.size() == 1);
        CHECK(recs[0].name.find("mlp.gate.weight") != std::string::npos);
        CHECK(static_cast<int64_t>(recs[0].data.size()) == 8 * 64 * 2);
    }

    // 5) Norms stay BF16 (dtype string preserved, not nvfp4-packed).
    {
        auto batch = lsxcommon::ReadRecordBatchFromFile(
            base / "pinned" / "layer_0.arrow").ValueOrDie();
        auto recs = lsxcommon::PinnedSchema::UnpackBatch(batch).ValueOrDie();
        bool checked = false;
        for (auto &r : recs) {
            if (r.name.find("input_layernorm") != std::string::npos) {
                CHECK(r.dtype == "BF16");
                CHECK(r.data.size() == 64 * 2);
                checked = true;
            }
        }
        CHECK(checked);
    }

    // 6) Shared expert files present.
    CHECK(fs::exists(base / "shared" / "layer_0_nvfp4.arrow"));
    CHECK(fs::exists(base / "shared" / "layer_0_gate.arrow"));

    // 7) MTP stored but separate from the live path.
    CHECK(fs::exists(base / "mtp" / "mtp_nvfp4.arrow"));

    // 8) Determinism: byte-identical index on re-conversion.
    {
        fs::path out2 = fs::temp_directory_path() / "lsx_qwen_moe_out2";
        fs::remove_all(out2);
        REQUIRE(ConvertQwen35MoeModel(in.string(), out2.string()).ok());
        std::string index_a, index_b;
        {
            // Streams must CLOSE before remove_all: Windows refuses to delete
            // files with open handles (POSIX unlinks them silently).
            std::ifstream a(base / "model.index.json"),
                b(out2 / "logestix" / "model.index.json");
            std::stringstream sa, sb; sa << a.rdbuf(); sb << b.rdbuf();
            index_a = sa.str();
            index_b = sb.str();
        }
        CHECK(index_a == index_b);
        fs::remove_all(out2);
    }

    // 9) All 4x8 expert files exist.
    for (int l = 0; l < 4; ++l)
        for (int e = 0; e < 8; ++e)
            CHECK(fs::exists(base / "experts" / ("layer_" + std::to_string(l)) /
                             ("exp_" + std::to_string(e) + ".arrow")));

    std::error_code ec;
    fs::remove_all(in, ec);
    fs::remove_all(out, ec);
}
