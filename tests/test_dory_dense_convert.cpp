#include <catch2/catch_all.hpp>

#include <lsxcommon/dory_dense_convert.h>
#include <lsxcommon/schema.h>

#include "nvfp4_scale.h"

#include "test_dory_dense_fixture.h"

#include <arrow/io/file.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>

using namespace lsxcommon::dory;

namespace {

std::vector<std::string> DenseNamesIn(const std::filesystem::path &p)
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

std::string ReadFile(const std::filesystem::path &p)
{
    std::ifstream f(p);
    return std::string((std::istreambuf_iterator<char>(f)),
                       std::istreambuf_iterator<char>());
}

std::vector<float> DecodeRecord(const lsxcommon::DenseNvfp4Record &r)
{
    static const float kMag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const int64_t n = static_cast<int64_t>(r.packed.size()) * 2;
    std::vector<float> out(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        const uint8_t byte = r.packed[static_cast<size_t>(i >> 1)];
        const int nib = (i & 1) ? (byte >> 4) : (byte & 0x0F);
        const float mag = kMag[nib & 7] * ((nib & 8) ? -1.0f : 1.0f);
        const float scale = lsxcommon::nvfp4::E4m3ToFloat(
            r.scales[static_cast<size_t>(i >> 4)]);
        out[static_cast<size_t>(i)] = mag * scale * r.global_scale;
    }
    return out;
}

float RelL2(const std::vector<float> &a, const std::vector<float> &b)
{
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < b.size(); ++i) {
        const double d = static_cast<double>(a[i]) - b[i];
        num += d * d;
        den += static_cast<double>(b[i]) * b[i];
    }
    return den > 0.0 ? static_cast<float>(std::sqrt(num / den)) : 0.0f;
}

struct ConvertedFixture {
    std::filesystem::path dir;
};

ConvertedFixture BuildConverted()
{
    namespace fs = std::filesystem;
    auto tmp = fs::temp_directory_path() / "dory_convert_test";
    fs::remove_all(tmp);
    fs::create_directories(tmp);
    dorytest::WriteSafetensors(tmp, "model.safetensors",
                               dorytest::BuildSyntheticDoryModel());
    dorytest::WriteConfig(tmp);
    fs::copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
                  tmp / "tokenizer.json");
    auto st = ConvertDoryModel(tmp.string(), tmp.string());
    REQUIRE(st.ok());
    return {tmp};
}

} // namespace

TEST_CASE("dory tensor name classifier", "[dory][convert]")
{
    auto check = [](const std::string &name, TensorKind expected,
                    int layer = -1) {
        const auto p = classify_dory_tensor(name);
        CHECK(p.kind == expected);
        if (layer >= 0) CHECK(p.layer == layer);
    };
    check("backbone.embeddings.weight", TensorKind::Embed);
    check("lm_head.weight", TensorKind::LmHead);
    check("lm_head.logit_scale", TensorKind::LogitScale);
    check("backbone.layers.0.mixer.q_proj.weight", TensorKind::MixerQ, 0);
    check("backbone.layers.0.mixer.k_proj.weight", TensorKind::MixerK, 0);
    check("backbone.layers.0.mixer.v_proj.weight", TensorKind::MixerV, 0);
    check("backbone.layers.0.mixer.o_proj.weight", TensorKind::MixerO, 0);
    check("backbone.layers.0.mixer.g_proj.weight", TensorKind::MixerG, 0);
    check("backbone.layers.1.mixer.gate_proj.weight", TensorKind::MixerGate, 1);
    check("backbone.layers.1.mixer.up_proj.weight", TensorKind::MixerUp, 1);
    check("backbone.layers.1.mixer.down_proj.weight", TensorKind::MixerDown, 1);
    check("backbone.layers.0.mixer.sqk", TensorKind::MixerSqk, 0);
    check("backbone.layers.0.mixer.s_gate", TensorKind::MixerSGate, 0);
    check("backbone.layers.1.mixer.suv_gate", TensorKind::MixerSuvGate, 1);
    check("backbone.layers.1.mixer.suv_up", TensorKind::MixerSuvUp, 1);
    check("backbone.layers.0.attn_alpha", TensorKind::AttnAlpha, 0);
    check("backbone.layers.1.mlp_alpha", TensorKind::MlpAlpha, 1);
    check("backbone.layers.0.input_layernorm.weight", TensorKind::Unknown);
    check("backbone.norm.weight", TensorKind::Unknown);
    check("", TensorKind::Unknown);

    // Every projection is quantized; the nGPT vectors, the residual alphas and
    // the edges stay raw.
    CHECK(dory_tensor_is_nvfp4(TensorKind::MixerQ));
    CHECK(dory_tensor_is_nvfp4(TensorKind::MixerG));
    CHECK(dory_tensor_is_nvfp4(TensorKind::MixerDown));
    CHECK_FALSE(dory_tensor_is_nvfp4(TensorKind::MixerSqk));
    CHECK_FALSE(dory_tensor_is_nvfp4(TensorKind::MixerSGate));
    CHECK_FALSE(dory_tensor_is_nvfp4(TensorKind::MixerSuvGate));
    CHECK_FALSE(dory_tensor_is_nvfp4(TensorKind::AttnAlpha));
    CHECK_FALSE(dory_tensor_is_nvfp4(TensorKind::MlpAlpha));
    CHECK_FALSE(dory_tensor_is_nvfp4(TensorKind::Embed));
    CHECK_FALSE(dory_tensor_is_nvfp4(TensorKind::LmHead));
    CHECK_FALSE(dory_tensor_is_nvfp4(TensorKind::LogitScale));

    // Exactly the projections the converter fuses: q|k|v and gate|up. The
    // decode issues ONE GEMV per mixer input, so these are the tensors whose
    // record name gains a `qkv_proj`/`gate_up_proj` spelling.
    CHECK(dory_fused_kind(TensorKind::MixerQ));
    CHECK(dory_fused_kind(TensorKind::MixerK));
    CHECK(dory_fused_kind(TensorKind::MixerV));
    CHECK(dory_fused_kind(TensorKind::MixerGate));
    CHECK(dory_fused_kind(TensorKind::MixerUp));
    CHECK_FALSE(dory_fused_kind(TensorKind::MixerO));
    CHECK_FALSE(dory_fused_kind(TensorKind::MixerG));
    CHECK_FALSE(dory_fused_kind(TensorKind::MixerDown));
    CHECK_FALSE(dory_fused_kind(TensorKind::MixerSqk));
}

TEST_CASE("dory converter builds the logestix-dory-nvfp4 layout",
          "[dory][convert]")
{
    auto fx = BuildConverted();
    auto tmp = fx.dir;

    const auto top =
        load_dory_topology((tmp / "logestix" / "model.index.json").string());
    INFO("topology error: " << top.error_msg);
    REQUIRE(top.error_msg.empty());

    const auto &cfg = dorytest::TestCfg();
    CHECK(top.num_layers == cfg.num_layers);
    CHECK(top.hidden_size == cfg.hidden_size);
    CHECK(top.intermediate_size == cfg.intermediate_size);
    CHECK(top.vocab_size == cfg.vocab_size);
    CHECK(top.num_attention_heads == cfg.num_attention_heads);
    CHECK(top.num_key_value_heads == cfg.num_key_value_heads);
    CHECK(top.head_dim == cfg.head_dim);
    CHECK(top.input_layers == cfg.input_layers);
    CHECK(top.recurrent_layers == cfg.recurrent_layers);
    CHECK(top.output_layers == cfg.output_layers);
    CHECK(top.recurrent_loops == cfg.recurrent_loops);
    CHECK(top.sliding_window == cfg.sliding_window);
    CHECK(top.num_attention_layers == cfg.num_layers / 2);
    CHECK(top.bos_id == cfg.bos_id);
    CHECK(top.eos_id == cfg.eos_id);
    CHECK(std::abs(top.initializer_range - cfg.initializer_range) < 1e-9);

    CHECK(top.vram.min_bytes > 0);
    CHECK(top.vram.optimal_bytes >= top.vram.min_bytes);
    CHECK(top.vram.max_bytes >= top.vram.optimal_bytes);
    CHECK(top.vram.context_tokens > 0);

    const auto json = ReadFile(tmp / "logestix" / "model.index.json");
    CHECK(json.find("\"format\": \"logestix-dory-nvfp4\"") !=
          std::string::npos);
    CHECK(json.find("\"edges\": \"fp16\"") != std::string::npos);
    CHECK(json.find("\"residual\": true") != std::string::npos);
    CHECK(json.find("\"num_kv_slots\": " + std::to_string(
                                             dorytest::CountKvSlots())) !=
          std::string::npos);
    // detect_model_type normalizes this field: it must fold to the kType.
    CHECK(json.find("\"model\": \"Dory-1.0-4B\"") != std::string::npos);

    // Attention layers carry the FUSED q|k|v record plus the gate and output
    // projections; mlp layers carry the FUSED gate|up plus down.
    const auto attn_names =
        DenseNamesIn(tmp / "logestix" / "pinned" / "layer_0_nvfp4.arrow");
    CHECK(Contains(attn_names, "backbone.layers.0.mixer.qkv_proj.weight"));
    CHECK(Contains(attn_names, "backbone.layers.0.mixer.g_proj.weight"));
    CHECK(Contains(attn_names, "backbone.layers.0.mixer.o_proj.weight"));
    CHECK_FALSE(Contains(attn_names, "backbone.layers.0.mixer.q_proj.weight"));
    for (const char *n : {"mixer.qkv_proj.weight", "mixer.g_proj.weight",
                          "mixer.o_proj.weight"})
        CHECK(Contains(attn_names, std::string("backbone.layers.0.") + n +
                                       ".res"));

    const auto mlp_names =
        DenseNamesIn(tmp / "logestix" / "pinned" / "layer_1_nvfp4.arrow");
    CHECK(Contains(mlp_names, "backbone.layers.1.mixer.gate_up_proj.weight"));
    CHECK(Contains(mlp_names, "backbone.layers.1.mixer.down_proj.weight"));
    CHECK(Contains(mlp_names,
                   "backbone.layers.1.mixer.gate_up_proj.weight.res"));
    CHECK(Contains(mlp_names, "backbone.layers.1.mixer.down_proj.weight.res"));

    // The nGPT vectors / alphas and the edges are raw records, never NVFP4.
    const auto raw_attn =
        synthtest::TensorNamesIn(tmp / "logestix" / "pinned" / "layer_0.arrow");
    CHECK(Contains(raw_attn, "backbone.layers.0.mixer.sqk"));
    CHECK(Contains(raw_attn, "backbone.layers.0.mixer.s_gate"));
    CHECK(Contains(raw_attn, "backbone.layers.0.attn_alpha"));
    const auto raw_mlp =
        synthtest::TensorNamesIn(tmp / "logestix" / "pinned" / "layer_1.arrow");
    CHECK(Contains(raw_mlp, "backbone.layers.1.mixer.suv_gate"));
    CHECK(Contains(raw_mlp, "backbone.layers.1.mixer.suv_up"));
    CHECK(Contains(raw_mlp, "backbone.layers.1.mlp_alpha"));
    const auto raw_global =
        synthtest::TensorNamesIn(tmp / "logestix" / "pinned" / "global.arrow");
    CHECK(Contains(raw_global, "backbone.embeddings.weight"));
    CHECK(Contains(raw_global, "lm_head.weight"));
    CHECK(Contains(raw_global, "lm_head.logit_scale"));
    CHECK_FALSE(std::filesystem::exists(tmp / "logestix" / "pinned" /
                                        "global_nvfp4.arrow"));

    // Fused q|k|v has q_size + 2*kv_size rows.
    auto batch = lsxcommon::ReadRecordBatchFromFile(
        tmp / "logestix" / "pinned" / "layer_0_nvfp4.arrow");
    REQUIRE(batch.ok());
    auto recs = lsxcommon::DenseNvfp4Schema::UnpackBatch(*batch);
    REQUIRE(recs.ok());
    for (const auto &r : *recs) {
        if (r.name != "backbone.layers.0.mixer.qkv_proj.weight") continue;
        REQUIRE(r.shape.size() == 2);
        CHECK(r.shape[0] == dorytest::QSize() + 2 * dorytest::KVSize());
        CHECK(r.shape[1] == dorytest::TestCfg().hidden_size);
    }

    std::filesystem::remove_all(tmp);
}

// The residual plane must actually buy precision: base+residual must
// reconstruct the source projection at least twice as well as the 4-bit base.
TEST_CASE("dory residual plane halves the 4-bit reconstruction error",
          "[dory][convert]")
{
    auto fx = BuildConverted();
    auto tmp = fx.dir;

    const auto tensors = dorytest::BuildSyntheticDoryModel();
    // Source rows of the FUSED qkv record: q|k|v concatenated.
    std::vector<float> src_qkv;
    for (const char *suffix :
         {"mixer.q_proj.weight", "mixer.k_proj.weight",
          "mixer.v_proj.weight"}) {
        const std::string name = std::string("backbone.layers.0.") + suffix;
        for (const auto &t : tensors) {
            if (t.name != name) continue;
            const size_t n = t.data.size() / 2;
            for (size_t i = 0; i < n; ++i) {
                const uint16_t bits = static_cast<uint16_t>(
                    t.data[2 * i] |
                    (static_cast<uint16_t>(t.data[2 * i + 1]) << 8));
                const uint32_t f32 = static_cast<uint32_t>(bits) << 16;
                float v;
                std::memcpy(&v, &f32, sizeof(v));
                src_qkv.push_back(v);
            }
        }
    }
    REQUIRE(!src_qkv.empty());

    auto batch = lsxcommon::ReadRecordBatchFromFile(
        tmp / "logestix" / "pinned" / "layer_0_nvfp4.arrow");
    REQUIRE(batch.ok());
    auto recs = lsxcommon::DenseNvfp4Schema::UnpackBatch(*batch);
    REQUIRE(recs.ok());
    const lsxcommon::DenseNvfp4Record *base = nullptr;
    const lsxcommon::DenseNvfp4Record *res = nullptr;
    for (const auto &r : *recs) {
        if (r.name == "backbone.layers.0.mixer.qkv_proj.weight") base = &r;
        if (r.name == "backbone.layers.0.mixer.qkv_proj.weight.res") res = &r;
    }
    REQUIRE(base != nullptr);
    REQUIRE(res != nullptr);

    const auto dq_base = DecodeRecord(*base);
    const auto dq_res = DecodeRecord(*res);
    REQUIRE(dq_base.size() == src_qkv.size());
    std::vector<float> sum(src_qkv.size());
    for (size_t i = 0; i < sum.size(); ++i) sum[i] = dq_base[i] + dq_res[i];
    const float e_base = RelL2(dq_base, src_qkv);
    const float e_res = RelL2(sum, src_qkv);
    INFO("base relL2=" << e_base << " base+res=" << e_res);
    CHECK(e_base > 0.0f);
    CHECK(e_res < e_base * 0.5f);

    std::filesystem::remove_all(tmp);
}

// A leftover tensor must fail the conversion: Dory carries no ballast, so an
// unclassified name means the classifier is stale, not that something can be
// skipped.
TEST_CASE("dory converter rejects an unclassified tensor", "[dory][convert]")
{
    namespace fs = std::filesystem;
    auto tmp = fs::temp_directory_path() / "dory_convert_reject";
    fs::remove_all(tmp);
    fs::create_directories(tmp);
    auto tensors = dorytest::BuildSyntheticDoryModel();
    tensors.push_back({"backbone.mtp_head.weight", "BF16", {8, 8},
                       synthtest::MakeBf16(64)});
    dorytest::WriteSafetensors(tmp, "model.safetensors", tensors);
    dorytest::WriteConfig(tmp);
    const auto st = ConvertDoryModel(tmp.string(), tmp.string());
    CHECK_FALSE(st.ok());
    CHECK(st.ToString().find("unclassified") != std::string::npos);
    fs::remove_all(tmp);
}
