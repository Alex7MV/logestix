#include <catch2/catch_all.hpp>

#include <lsxcommon/granite_dense_switch_convert.h>
#include <lsxcommon/schema.h>

#include "test_granite_switch_fixture.h"

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

using namespace lsxcommon::granite_switch;

namespace {

std::vector<std::string> DenseNvfp4NamesIn(const std::filesystem::path &p)
{
    auto batch = lsxcommon::ReadRecordBatchFromFile(p);
    if (!batch.ok()) return {};
    auto recs = lsxcommon::DenseNvfp4Schema::UnpackBatch(*batch);
    if (!recs.ok()) return {};
    std::vector<std::string> names;
    for (auto &r : *recs) names.push_back(r.name);
    return names;
}

bool Contains(const std::vector<std::string> &v, const std::string &s)
{
    return std::find(v.begin(), v.end(), s) != v.end();
}

} // namespace

TEST_CASE("granite-switch tensor classifier", "[granite_switch][convert]")
{
    auto check = [](const std::string &n, GraniteSwitchTensorKind k,
                    int layer = -1, int slice = -1) {
        auto p = classify_granite_switch_tensor(n);
        CHECK(p.kind == k);
        if (layer >= 0) CHECK(p.layer == layer);
        if (slice >= 0) CHECK(p.slice == slice);
    };

    check("model.embed_tokens.weight", GraniteSwitchTensorKind::Embed);
    check("model.norm.weight", GraniteSwitchTensorKind::FinalNorm);
    check("model.switch.codebook", GraniteSwitchTensorKind::SwitchCodebook);
    check("model.switch.control_to_substitute_lut",
          GraniteSwitchTensorKind::SwitchLut);
    check("model.layers.3.input_layernorm.weight",
          GraniteSwitchTensorKind::InputNorm, 3);
    check("model.layers.3.post_attention_layernorm.weight",
          GraniteSwitchTensorKind::PostNorm, 3);
    check("model.layers.3.self_attn.qkv_proj.base_layer.weight",
          GraniteSwitchTensorKind::AttnQkvBase, 3);
    check("model.layers.3.self_attn.qkv_proj.lora_A_slices.2",
          GraniteSwitchTensorKind::AttnQkvLoraA, 3, 2);
    check("model.layers.3.self_attn.qkv_proj.lora_B_slices.1",
          GraniteSwitchTensorKind::AttnQkvLoraB, 3, 1);
    check("model.layers.3.self_attn.o_proj.base_layer.weight",
          GraniteSwitchTensorKind::AttnOBase, 3);
    check("model.layers.3.self_attn.o_proj.lora_A",
          GraniteSwitchTensorKind::AttnOLoraA, 3);
    check("model.layers.3.self_attn.o_proj.lora_B",
          GraniteSwitchTensorKind::AttnOLoraB, 3);
    check("model.layers.3.shared_mlp.input_linear.base_layer.weight",
          GraniteSwitchTensorKind::MlpGateUpBase, 3);
    check("model.layers.3.shared_mlp.input_linear.lora_A_slices.1",
          GraniteSwitchTensorKind::MlpGateUpLoraA, 3, 1);
    check("model.layers.3.shared_mlp.input_linear.lora_B_slices.0",
          GraniteSwitchTensorKind::MlpGateUpLoraB, 3, 0);
    check("model.layers.3.shared_mlp.output_linear.base_layer.weight",
          GraniteSwitchTensorKind::MlpDownBase, 3);
    check("model.layers.3.shared_mlp.output_linear.lora_A",
          GraniteSwitchTensorKind::MlpDownLoraA, 3);
    check("model.layers.3.shared_mlp.output_linear.lora_B",
          GraniteSwitchTensorKind::MlpDownLoraB, 3);

    check("model.layers.3.self_attn.q_norm.weight",
          GraniteSwitchTensorKind::Unknown);
    check("lm_head.weight", GraniteSwitchTensorKind::Unknown);

    CHECK(granite_switch_tensor_is_nvfp4(GraniteSwitchTensorKind::Embed));
    CHECK(granite_switch_tensor_is_nvfp4(GraniteSwitchTensorKind::AttnQkvBase));
    CHECK_FALSE(granite_switch_tensor_is_nvfp4(
        GraniteSwitchTensorKind::AttnQkvLoraA));
    CHECK_FALSE(granite_switch_tensor_is_nvfp4(
        GraniteSwitchTensorKind::InputNorm));
    CHECK_FALSE(granite_switch_tensor_is_nvfp4(
        GraniteSwitchTensorKind::SwitchCodebook));
}

TEST_CASE("granite-switch converter builds the pinned layout",
          "[granite_switch][convert]")
{
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "granite_switch_conv_test";
    remove_all(tmp);
    create_directories(tmp);

    gswtest::BuildConverted(tmp);
    copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
              tmp / "tokenizer.json");

    REQUIRE(ConvertGraniteSwitchModel(tmp.string(), tmp.string()).ok());

    auto top = load_granite_switch_topology(
        (tmp / "logestix" / "model.index.json").string());
    INFO(top.error_msg);
    CHECK(top.error_msg.empty());
    CHECK(top.num_layers == 2);
    CHECK(top.switch_cache_layers == 2);
    CHECK(top.hidden_size == 64);
    CHECK(top.intermediate_size == 128);
    CHECK(top.vocab_size == 72);
    CHECK(top.num_attention_heads == 4);
    CHECK(top.num_key_value_heads == 2);
    CHECK(top.head_dim == 16);
    CHECK(top.adapter_ranks == std::vector<int>({4, 8}));
    CHECK(top.adapter_token_ids == std::vector<int>({64, 65}));
    CHECK(top.adapter_substitute_token_ids == std::vector<int>({6, 7}));
    CHECK(top.adapter_names ==
          std::vector<std::string>({"guardian-core", "decision"}));
    CHECK(top.tie_word_embeddings);
    CHECK(top.embedding_multiplier == Catch::Approx(12.0));
    CHECK(top.residual_multiplier == Catch::Approx(0.175));
    CHECK(top.logits_scaling == Catch::Approx(16.0));
    CHECK(top.vram.min_bytes > 0);

    const auto pinned = tmp / "logestix" / "pinned";
    CHECK(exists(pinned / "global.arrow"));
    CHECK(exists(pinned / "global_nvfp4.arrow"));
    CHECK(exists(pinned / "layer_0_nvfp4.arrow"));
    CHECK(exists(pinned / "layer_1_nvfp4.arrow"));
    CHECK(exists(pinned / "layer_0_adapters.arrow"));

    auto gn = DenseNvfp4NamesIn(pinned / "global_nvfp4.arrow");
    CHECK(Contains(gn, "model.embed_tokens.weight"));
    CHECK_FALSE(Contains(gn, "lm_head.weight"));

    auto ln = DenseNvfp4NamesIn(pinned / "layer_0_nvfp4.arrow");
    CHECK(Contains(ln, "model.layers.0.self_attn.qkv_proj.weight"));
    CHECK(Contains(ln, "model.layers.0.self_attn.o_proj.weight"));
    CHECK(Contains(ln, "model.layers.0.shared_mlp.gate_up_proj.weight"));
    CHECK(Contains(ln, "model.layers.0.shared_mlp.down_proj.weight"));

    auto an = gswtest::TensorNamesIn(pinned / "layer_0_adapters.arrow");
    CHECK(Contains(an,
                   "model.layers.0.self_attn.qkv_proj.lora_A_slices.0.adapter0"));
    CHECK(Contains(an,
                   "model.layers.0.self_attn.qkv_proj.lora_B_slices.2.adapter1"));
    CHECK_FALSE(Contains(
        an, "model.layers.0.shared_mlp.input_linear.lora_A_slices.0.adapter0"));
    CHECK(Contains(
        an, "model.layers.0.shared_mlp.input_linear.lora_A_slices.0.adapter1"));
    CHECK(Contains(an,
                   "model.layers.0.shared_mlp.output_linear.lora_B.adapter1"));

    auto rn = gswtest::TensorNamesIn(pinned / "global.arrow");
    CHECK(Contains(rn, "model.norm.weight"));
    CHECK(Contains(rn, "model.switch.codebook"));

    remove_all(tmp);
}
