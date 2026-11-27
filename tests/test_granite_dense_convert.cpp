#include <catch2/catch_all.hpp>

#include <lsxcommon/granite_dense_convert.h>
#include <lsxcommon/schema.h>

#include "test_granite_dense_fixture.h"

#include <arrow/io/file.h>
#include <cstdint>
#include <filesystem>

using namespace lsxcommon::granite;

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

TEST_CASE("Granite tensor name classifier", "[granite][convert]")
{
    auto check = [](const std::string &name, TensorKind expected,
                    int layer = -1) {
        auto p = classify_granite_tensor(name);
        CHECK(p.kind == expected);
        if (layer >= 0) CHECK(p.layer == layer);
    };
    check("model.embed_tokens.weight", TensorKind::Embed);
    check("lm_head.weight", TensorKind::LmHead);
    check("model.norm.weight", TensorKind::FinalNorm);
    check("model.layers.0.input_layernorm.weight", TensorKind::InputNorm, 0);
    check("model.layers.7.post_attention_layernorm.weight",
          TensorKind::PostNorm, 7);
    check("model.layers.3.self_attn.q_proj.weight", TensorKind::AttnQ, 3);
    check("model.layers.3.self_attn.k_proj.weight", TensorKind::AttnK, 3);
    check("model.layers.3.self_attn.v_proj.weight", TensorKind::AttnV, 3);
    check("model.layers.3.self_attn.o_proj.weight", TensorKind::AttnO, 3);
    check("model.layers.1.mlp.gate_proj.weight", TensorKind::MlpGate, 1);
    check("model.layers.1.mlp.up_proj.weight", TensorKind::MlpUp, 1);
    check("model.layers.1.mlp.down_proj.weight", TensorKind::MlpDown, 1);
    check("model.layers.0.self_attn.q_norm.weight", TensorKind::Unknown);
    check("unknown_tensor", TensorKind::Unknown);

    CHECK(granite_tensor_is_nvfp4(TensorKind::AttnQ));
    CHECK(granite_tensor_is_nvfp4(TensorKind::MlpDown));
    CHECK(granite_tensor_is_nvfp4(TensorKind::Embed));
    CHECK_FALSE(granite_tensor_is_nvfp4(TensorKind::InputNorm));
    CHECK_FALSE(granite_tensor_is_nvfp4(TensorKind::FinalNorm));
}

TEST_CASE("Granite converter builds logestix-granite index and arrow layout",
          "[granite][convert]")
{
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "granite_conv_test";
    remove_all(tmp);
    create_directories(tmp);

    auto tensors = grantest::BuildSyntheticGraniteModel();
    grantest::WriteSafetensors(tmp, "model.safetensors", tensors);
    grantest::WriteConfig(tmp);
    copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
              tmp / "tokenizer.json");

    auto st = ConvertGraniteModel(tmp.string(), tmp.string());
    REQUIRE(st.ok());

    auto top = load_granite_topology(
        (tmp / "model.index.json").string());
    CHECK(top.error_msg.empty());
    CHECK(top.num_layers == 2);
    CHECK(top.hidden_size == 64);
    CHECK(top.intermediate_size == 128);
    CHECK(top.vocab_size == 64);
    CHECK(top.num_attention_heads == 4);
    CHECK(top.num_key_value_heads == 2);
    CHECK(top.head_dim == 16);
    CHECK(top.bos_id == 0);
    CHECK(top.eos_id == 1);

    // VRAM profile: mandatory, internally consistent.
    CHECK(top.vram.min_bytes > 0);
    CHECK(top.vram.optimal_bytes >= top.vram.min_bytes);
    CHECK(top.vram.max_bytes >= top.vram.optimal_bytes);
    CHECK(top.vram.context_tokens > 0);

    // global.arrow (final norm raw) + global_nvfp4.arrow (embed/head).
    CHECK(exists(tmp / "pinned" / "global.arrow"));
    CHECK(exists(tmp / "pinned" / "global_nvfp4.arrow"));
    CHECK(exists(tmp / "pinned" / "layer_0.arrow"));
    CHECK(exists(tmp / "pinned" / "layer_0_nvfp4.arrow"));
    CHECK(exists(tmp / "pinned" / "layer_1_nvfp4.arrow"));

    // Per-layer NVFP4 records: q/k/v/o + fused gate_up + down.
    auto names = DenseNvfp4NamesIn(tmp / "pinned" /
                                   "layer_0_nvfp4.arrow");
    // q/k/v are fused into one jointly-quantized record.
    CHECK(Contains(names, "model.layers.0.self_attn.qkv_proj.weight"));
    CHECK(Contains(names, "model.layers.0.self_attn.o_proj.weight"));
    CHECK(Contains(names, "model.layers.0.mlp.gate_up_proj.weight"));
    CHECK(Contains(names, "model.layers.0.mlp.down_proj.weight"));

    // Global NVFP4 records: embed + lm_head.
    auto gnames = DenseNvfp4NamesIn(tmp / "pinned" /
                                    "global_nvfp4.arrow");
    CHECK(Contains(gnames, "model.embed_tokens.weight"));
    CHECK(Contains(gnames, "lm_head.weight"));

    // Raw global record: final norm.
    auto rnames = synthtest::TensorNamesIn(tmp / "pinned" /
                                           "global.arrow");
    CHECK(Contains(rnames, "model.norm.weight"));

    remove_all(tmp);
}
