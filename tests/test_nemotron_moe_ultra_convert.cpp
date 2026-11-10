// TDD — Nemotron-3-Ultra-550B-A55B converter (Model Ingestion Protocol step 1).
//
// Synthetic checkpoint only: the 1.1 TB source is never converted in a test.
// The ctest TIMEOUT (120 s) is the mechanical guard against a full conversion
// sneaking in.
//
// What is Ultra-specific and therefore what these cases exist to prove:
//   * the latent MoE bridge (fc1/fc2) is converted and is mandatory exactly when
//     the config declares `moe_latent_size`;
//   * the routed experts live at the LATENT width, not at hidden;
//   * the index carries the grouped-router fields (n_group / topk_group) and the
//     model's own routed_scaling_factor instead of a frozen constant;
//   * experts stream per layer, so peak host residency is one layer, not 290 GB;
//   * a layer window converts a slice and marks the tree partial.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "test_nemotron_moe_ultra_fixture.h"

#include "lsxcommon/nemotron_moe_ultra_convert.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace lsxcommon::nemotron_ultra;

namespace {

std::string ReadIndexJson(const fs::path &dst)
{
    std::ifstream in(dst / "logestix" / "model.index.json");
    REQUIRE(in.good());
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

fs::path MakeSource(const std::string &tag)
{
    const auto src = ultratest::MakeTempDir(tag);
    ultratest::WriteUltraCheckpoint(src);
    return src;
}

} // namespace

TEST_CASE("ultra classifier: tensor names map to Ultra kinds", "[ultra][convert]")
{
    using K = UltraTensorKind;

    CHECK(classify_ultra_tensor("backbone.layers.0.mixer.in_proj.weight").kind ==
          K::MambaInProj);
    CHECK(classify_ultra_tensor("backbone.layers.0.mixer.in_proj.weight").layer == 0);
    CHECK(classify_ultra_tensor("backbone.layers.0.mixer.out_proj.weight").kind ==
          K::MambaOutProj);
    CHECK(classify_ultra_tensor("backbone.layers.0.mixer.conv1d.weight").kind ==
          K::MambaConv);
    CHECK(classify_ultra_tensor("backbone.layers.0.mixer.conv1d.bias").kind ==
          K::MambaConvBias);
    CHECK(classify_ultra_tensor("backbone.layers.0.mixer.A_log").kind == K::MambaALog);
    CHECK(classify_ultra_tensor("backbone.layers.0.mixer.D").kind == K::MambaD);
    CHECK(classify_ultra_tensor("backbone.layers.0.mixer.dt_bias").kind ==
          K::MambaDtBias);
    CHECK(classify_ultra_tensor("backbone.layers.0.mixer.norm.weight").kind ==
          K::MambaNorm);
    CHECK(classify_ultra_tensor("backbone.layers.0.norm.weight").kind == K::LayerNorm);
    CHECK(classify_ultra_tensor("backbone.embeddings.weight").kind == K::Embed);
    CHECK(classify_ultra_tensor("lm_head.weight").kind == K::LmHead);
    CHECK(classify_ultra_tensor("backbone.norm_f.weight").kind == K::FinalNorm);

    // The latent bridge is its own kind, never confused with the shared expert.
    CHECK(classify_ultra_tensor("backbone.layers.1.mixer.fc1_latent_proj.weight").kind ==
          K::MoeLatentUp);
    CHECK(classify_ultra_tensor("backbone.layers.1.mixer.fc1_latent_proj.weight").layer == 1);
    CHECK(classify_ultra_tensor("backbone.layers.1.mixer.fc2_latent_proj.weight").kind ==
          K::MoeLatentDown);
    CHECK(classify_ultra_tensor(
              "backbone.layers.1.mixer.shared_experts.up_proj.weight").kind ==
          K::SharedUpProj);
    CHECK(classify_ultra_tensor(
              "backbone.layers.1.mixer.shared_experts.down_proj.weight").kind ==
          K::SharedDownProj);

    // Routed experts carry layer + expert.
    auto up = classify_ultra_tensor("backbone.layers.1.mixer.experts.7.up_proj.weight");
    CHECK(up.kind == K::ExpertUpProj);
    CHECK(up.expert == 7);
    auto dn = classify_ultra_tensor("backbone.layers.1.mixer.experts.3.down_proj.weight");
    CHECK(dn.kind == K::ExpertDownProj);
    CHECK(dn.expert == 3);

    // Attention projections.
    CHECK(classify_ultra_tensor("backbone.layers.2.mixer.q_proj.weight").kind == K::AttnQ);
    CHECK(classify_ultra_tensor("backbone.layers.2.mixer.k_proj.weight").kind == K::AttnK);
    CHECK(classify_ultra_tensor("backbone.layers.2.mixer.v_proj.weight").kind == K::AttnV);
    CHECK(classify_ultra_tensor("backbone.layers.2.mixer.o_proj.weight").kind == K::AttnO);

    // Router + the counted cut.
    CHECK(classify_ultra_tensor("backbone.layers.1.mixer.gate.weight").kind == K::RouterGate);
    CHECK(classify_ultra_tensor("backbone.layers.1.mixer.gate.e_score_correction_bias").kind ==
          K::RouterBias);
    CHECK(classify_ultra_tensor("mtp.layers.0.mixer.q_proj.weight").kind == K::Mtp);
    CHECK(classify_ultra_tensor("totally.unknown.tensor").kind == K::Unknown);
}

TEST_CASE("ultra classifier: malformed indices are Unknown, never a throw",
          "[ultra][convert]")
{
    using K = UltraTensorKind;
    CHECK_NOTHROW(classify_ultra_tensor("backbone.layers..mixer.norm.weight"));
    CHECK_NOTHROW(classify_ultra_tensor("backbone.layers.0.mixer.experts..up_proj.weight"));
    CHECK(classify_ultra_tensor("backbone.layers..mixer.norm.weight").kind == K::Unknown);
    CHECK(classify_ultra_tensor("backbone.layers.0.mixer.experts..up_proj.weight").kind ==
          K::Unknown);
}

TEST_CASE("ultra config parse: the Ultra fields are read", "[ultra][convert]")
{
    const auto src = MakeSource("nu_cfg");
    const auto top = ParseUltraConfig(src.string());
    REQUIRE(top.error_msg.empty());
    CHECK(top.hidden_size == ultratest::kTH);
    CHECK(top.num_layers == 3);
    CHECK(top.num_experts == ultratest::kTExperts);
    CHECK(top.num_experts_per_tok == ultratest::kTTopK);
    CHECK(top.moe_latent_size == ultratest::kTLatent);
    CHECK(top.n_group == ultratest::kTGroup);
    CHECK(top.topk_group == ultratest::kTTopkGroup);
    CHECK(top.routed_scaling_factor == Catch::Approx(5.0));
    CHECK(top.norm_topk_prob);
    CHECK(top.layer_types == std::vector<std::string>{"mamba", "moe", "attention"});
}

TEST_CASE("ultra config parse: a group count that does not divide the experts is rejected",
          "[ultra][convert]")
{
    const auto src = ultratest::CopyUltraCheckpoint(MakeSource("nu_badgrp"), "nu_badgrp");

    // n_group 3 cannot divide 8 experts: the reference would crash in the group
    // stage, so the converter must reject it at parse time by name.
    {
        std::ifstream in(src / "config.json");
        std::stringstream ss;
        ss << in.rdbuf();
        std::string cfg = ss.str();
        const std::string from = "\"n_group\": 4";
        const auto at = cfg.find(from);
        REQUIRE(at != std::string::npos);
        cfg.replace(at, from.size(), "\"n_group\": 3");
        std::ofstream out(src / "config.json");
        out << cfg;
    }

    const auto top = ParseUltraConfig(src.string());
    CHECK_FALSE(top.error_msg.empty());
    CHECK(top.error_msg.find("group") != std::string::npos);
}

TEST_CASE("ultra convert: the synthetic checkpoint produces the Ultra index",
          "[ultra][convert]")
{
    const auto src = MakeSource("nu_conv_src");
    const auto dst = ultratest::MakeTempDir("nu_conv_dst");

    const auto st = ConvertNemotronUltraModel(src.string(), dst.string());
    REQUIRE(st.ok());

    const std::string json = ReadIndexJson(dst);
    CHECK(json.find("\"format\": \"logestix-nemotron-ultra-nvfp4\"") != std::string::npos);
    CHECK(json.find("\"moe_latent_size\": 8") != std::string::npos);
    CHECK(json.find("\"n_group\": 4") != std::string::npos);
    CHECK(json.find("\"topk_group\": 1") != std::string::npos);
    CHECK(json.find("\"mtp_tensors\": 3") != std::string::npos);
    CHECK(json.find("\"partial\": false") != std::string::npos);
    // Expert records are streamed one layer at a time: the peak is this model's
    // experts per layer (8), never every expert in the checkpoint. Without this
    // the real 24 576-expert pool would be ~290 GB of host records.
    CHECK(json.find("\"peak_expert_records\": 8") != std::string::npos);

    CHECK(fs::exists(dst / "logestix" / "pinned" / "global.arrow"));
    CHECK(fs::exists(dst / "logestix" / "pinned" / "layer_0.arrow"));
    CHECK(fs::exists(dst / "logestix" / "pinned" / "layer_1_nvfp4.arrow"));
    CHECK(fs::exists(dst / "logestix" / "pinned" / "layer_2_nvfp4.arrow"));
    // One expert arrow per expert of the single MoE layer; no layer_0 experts.
    for (int e = 0; e < ultratest::kTExperts; ++e)
        CHECK(fs::exists(dst / "logestix" / "experts" / "layer_1" /
                         ("exp_" + std::to_string(e) + ".arrow")));
    CHECK_FALSE(fs::exists(dst / "logestix" / "experts" / "layer_0"));
}

TEST_CASE("ultra convert: the latent projections are mandatory when declared",
          "[ultra][convert]")
{
    const auto src = ultratest::MakeTempDir("nu_nolat");
    ultratest::WriteUltraConfig(src);
    // Drop fc1_latent_proj from the inventory: the config still declares
    // moe_latent_size, so the converter must fail naming the tensor instead of
    // silently converting a model whose latent bridge is missing.
    auto tensors = ultratest::MakeUltraTensors();
    tensors.erase(std::remove_if(tensors.begin(), tensors.end(),
                                 [](const synthtest::FakeTensor &t) {
                                     return t.name.find("fc1_latent_proj") !=
                                            std::string::npos;
                                 }),
                  tensors.end());
    synthtest::WriteSafetensors(src, "model.safetensors", tensors);

    const auto dst = ultratest::MakeTempDir("nu_nolat_dst");
    const auto st = ConvertNemotronUltraModel(src.string(), dst.string());
    REQUIRE_FALSE(st.ok());
    CHECK(st.message().find("fc1_latent_proj") != std::string::npos);
}

TEST_CASE("ultra convert: an unclassified tensor is an error, never ignored",
          "[ultra][convert]")
{
    const auto src = MakeSource("nu_bad_src");
    const auto dst = ultratest::MakeTempDir("nu_bad_dst");
    std::vector<synthtest::FakeTensor> extra;
    extra.push_back({"backbone.layers.0.mixer.mystery.weight", "BF16", {2, 2},
                     synthtest::MakeBf16(4)});
    synthtest::WriteSafetensors(src, "bogus.safetensors", extra);

    const auto st = ConvertNemotronUltraModel(src.string(), dst.string());
    REQUIRE_FALSE(st.ok());
    CHECK(st.message().find("mystery") != std::string::npos);
}

TEST_CASE("ultra convert: a layer window converts only that layer and marks the tree partial",
          "[ultra][convert]")
{
    const auto src = MakeSource("nu_win_src");
    const auto dst = ultratest::MakeTempDir("nu_win_dst");

    ConversionWindow w;
    w.first_layer = 1;
    w.last_layer = 1;
    const auto st = ConvertNemotronUltraModel(src.string(), dst.string(), w);
    REQUIRE(st.ok());

    const std::string json = ReadIndexJson(dst);
    CHECK(json.find("\"partial\": true") != std::string::npos);
    CHECK(json.find("\"layers_converted\": [1]") != std::string::npos);
    CHECK(fs::exists(dst / "logestix" / "experts" / "layer_1" / "exp_0.arrow"));
    CHECK_FALSE(fs::exists(dst / "logestix" / "pinned" / "layer_0.arrow"));
    CHECK_FALSE(fs::exists(dst / "logestix" / "pinned" / "layer_2_nvfp4.arrow"));
}

TEST_CASE("ultra convert: a second window extends the tree without rewriting layer 1",
          "[ultra][convert]")
{
    const auto src = MakeSource("nu_resume_src");
    const auto dst = ultratest::MakeTempDir("nu_resume_dst");

    ConversionWindow first;
    first.first_layer = 1;
    first.last_layer = 1;
    REQUIRE(ConvertNemotronUltraModel(src.string(), dst.string(), first).ok());
    const auto stamp_file = dst / "logestix" / "experts" / "layer_1" / "exp_0.arrow";
    const auto before = fs::last_write_time(stamp_file);

    ConversionWindow second;
    second.first_layer = 2;
    second.last_layer = 2;
    REQUIRE(ConvertNemotronUltraModel(src.string(), dst.string(), second).ok());

    CHECK(fs::last_write_time(stamp_file) == before);  // untouched, not rewritten
    CHECK(fs::exists(dst / "logestix" / "pinned" / "layer_2_nvfp4.arrow"));
    // The index is rewritten to name every converted layer.
    const std::string json = ReadIndexJson(dst);
    CHECK(json.find("\"layers_converted\": [1,2]") != std::string::npos);
    CHECK(json.find("\"partial\": true") != std::string::npos);
}
