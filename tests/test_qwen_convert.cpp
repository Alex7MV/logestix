#include <catch2/catch_all.hpp>

#include <lsxcommon/qwen_convert.h>
#include <lsxcommon/schema.h>

#include "qwen_test_fixture.h"

#include <arrow/io/file.h>
#include <filesystem>
#include <cstdint>

using namespace lsxcommon::qwen;

TEST_CASE("Qwen tensor name classifier")
{
    auto check = [](const std::string &name, TensorKind expected, int layer = -1) {
        auto p = classify_qwen_tensor(name);
        CHECK(p.kind == expected);
        if (layer >= 0) CHECK(p.layer == layer);
    };
    check("model.language_model.embed_tokens.weight", TensorKind::Embed);
    check("lm_head.weight", TensorKind::LmHead);
    check("model.language_model.norm.weight", TensorKind::FinalNorm);
    check("model.language_model.layers.0.linear_attn.in_proj_qkv.weight", TensorKind::LinearAttnQkv, 0);
    check("model.language_model.layers.3.self_attn.q_proj.weight", TensorKind::FullAttnQ, 3);
    check("model.language_model.layers.12.linear_attn.out_proj.weight", TensorKind::LinearAttnOut, 12);
    check("model.language_model.layers.1.mlp.gate_proj.weight", TensorKind::MlpGate, 1);
    check("model.language_model.layers.5.input_layernorm.weight", TensorKind::InputNorm, 5);
    check("model.language_model.layers.7.self_attn.o_proj.weight", TensorKind::FullAttnO, 7);
    check("model.language_model.layers.0.linear_attn.A_log", TensorKind::LinearAttnA_log, 0);
    check("model.language_model.layers.0.linear_attn.dt_bias", TensorKind::LinearAttnDt, 0);
    check("model.language_model.layers.0.linear_attn.conv1d.weight", TensorKind::LinearAttnConv, 0);
    check("mtp.layers.0.self_attn.q_proj.weight", TensorKind::MTP);
    check("mtp.norm.weight", TensorKind::MTP);
    check("visual.blocks.0.norm1.weight", TensorKind::Visual);
    check("unknown_tensor", TensorKind::Unknown);
}

TEST_CASE("DenseNvfp4Schema round-trip")
{
    lsxcommon::DenseNvfp4Record rec;
    rec.name = "test";
    rec.shape = {64, 128};
    rec.dtype = "E2M1";
    rec.packed = std::vector<uint8_t>(64 * 128 / 2, 0x12);
    rec.scales = std::vector<uint8_t>((64 * 128 + 15) / 16, 0x28);
    rec.global_scale = 1.5f;

    auto batch = lsxcommon::DenseNvfp4Schema::MakeBatch({rec});
    REQUIRE(batch.ok());
    auto out = lsxcommon::DenseNvfp4Schema::UnpackBatch(*batch);
    REQUIRE(out.ok());
    CHECK(out->size() == 1);
    CHECK((*out)[0].name == "test");
    CHECK((*out)[0].shape == std::vector<int64_t>({64, 128}));
    CHECK((*out)[0].packed.size() == 64 * 128 / 2);
    CHECK((*out)[0].scales.size() == (64 * 128 + 15) / 16);
    CHECK((*out)[0].global_scale == 1.5f);
}

TEST_CASE("Qwen converter builds logestix-qwen index and filters MTP/Visual")
{
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "qwen_conv_test";
    remove_all(tmp);
    create_directories(tmp);

    auto tensors = qwentest::BuildSyntheticQwenModel();
    qwentest::WriteSafetensors(tmp, "model.safetensors", tensors);
    qwentest::WriteConfig(tmp);
    std::filesystem::copy_file(
        TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
        tmp / "tokenizer.json");

    auto st = ConvertQwenModel(tmp.string(), tmp.string());
    REQUIRE(st.ok());

    auto top = load_qwen_topology((tmp / "logestix" / "model.index.json").string());
    CHECK(top.num_layers == 4);
    CHECK(top.layer_types[0] == "linear_attention");
    CHECK(top.layer_types[1] == "linear_attention");
    CHECK(top.layer_types[2] == "linear_attention");
    CHECK(top.layer_types[3] == "full_attention");
    CHECK(top.hidden_size == 64);
    CHECK(top.intermediate_size == 128);
    CHECK(top.vocab_size == 64);
    CHECK(top.linear_num_value_heads == 4);
    CHECK(top.linear_value_head_dim == 8);
    CHECK(top.num_attention_heads == 2);
    CHECK(top.head_dim == 32);
    CHECK(top.error_msg.empty());
    // VRAM profile: mandatory, internally consistent.
    CHECK(top.vram.min_bytes > 0);
    CHECK(top.vram.optimal_bytes >= top.vram.min_bytes);
    CHECK(top.vram.max_bytes >= top.vram.optimal_bytes); // dense: max == optimal
    CHECK(top.vram.context_tokens > 0);

    // global.arrow (raw norms) and global_nvfp4.arrow (embed/head)
    CHECK(exists(tmp / "logestix" / "pinned" / "global.arrow"));
    CHECK(exists(tmp / "logestix" / "pinned" / "global_nvfp4.arrow"));
    // per-layer
    CHECK(exists(tmp / "logestix" / "pinned" / "layer_0.arrow"));
    CHECK(exists(tmp / "logestix" / "pinned" / "layer_0_nvfp4.arrow"));

    remove_all(tmp);
}