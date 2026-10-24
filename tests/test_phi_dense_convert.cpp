#include <catch2/catch_all.hpp>

#include <lsxcommon/phi_dense_convert.h>
#include <lsxcommon/schema.h>

#include "nvfp4_scale.h"

#include "test_phi_dense_fixture.h"

#include <arrow/io/file.h>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>

using namespace lsxcommon::phi;

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


namespace {

// Host reference decode of one NVFP4 record (mirrors nvfp4_codec.h E2M1Value +
// nvfp4_scale.h E4m3ToFloat). Test-side on purpose: the point of the residual
// case is an independent check of what the converter wrote.
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

std::vector<float> SourceBf16(const std::vector<phitest::FakeTensor> &ts,
                              const std::string &name)
{
    for (const auto &t : ts) {
        if (t.name != name) continue;
        std::vector<float> out(static_cast<size_t>(t.data.size() / 2));
        for (size_t i = 0; i < out.size(); ++i) {
            const uint16_t bits = static_cast<uint16_t>(
                t.data[2 * i] | (static_cast<uint16_t>(t.data[2 * i + 1]) << 8));
            const uint32_t f32 = static_cast<uint32_t>(bits) << 16;
            float v;
            std::memcpy(&v, &f32, sizeof(v));
            out[i] = v;
        }
        return out;
    }
    return {};
}

} // namespace

TEST_CASE("phi tensor name classifier", "[phi][convert]")
{
    auto check = [](const std::string &name, TensorKind expected,
                    int layer = -1) {
        auto p = classify_phi_tensor(name);
        CHECK(p.kind == expected);
        if (layer >= 0) CHECK(p.layer == layer);
    };
    check("model.embed_tokens.weight", TensorKind::Embed);
    check("lm_head.weight", TensorKind::LmHead);
    check("model.norm.weight", TensorKind::FinalNorm);
    check("model.layers.0.input_layernorm.weight", TensorKind::InputNorm, 0);
    check("model.layers.7.post_attention_layernorm.weight",
          TensorKind::PostNorm, 7);
    check("model.layers.3.self_attn.qkv_proj.weight", TensorKind::AttnQKV, 3);
    check("model.layers.3.self_attn.o_proj.weight", TensorKind::AttnO, 3);
    check("model.layers.1.mlp.gate_up_proj.weight", TensorKind::MlpGateUp, 1);
    check("model.layers.1.mlp.down_proj.weight", TensorKind::MlpDown, 1);
    check("model.layers.0.self_attn.q_norm.weight", TensorKind::Unknown);
    check("unknown_tensor", TensorKind::Unknown);

    CHECK(phi_tensor_is_nvfp4(TensorKind::AttnQKV));
    CHECK(phi_tensor_is_nvfp4(TensorKind::MlpDown));
    // The embedding and the LM head stay high precision: quantizing them to
    // 4 bits cost 0.12 perplexity and two of nine near-tie answers against the
    // same body (measured 2026-10-24, see the fps8/residual study in
    // docs/open-tasks), and it is the one thing every standard 4-bit recipe
    // (NF4/AWQ/GPTQ) refuses to do.
    CHECK_FALSE(phi_tensor_is_nvfp4(TensorKind::Embed));
    CHECK_FALSE(phi_tensor_is_nvfp4(TensorKind::LmHead));
    CHECK_FALSE(phi_tensor_is_nvfp4(TensorKind::InputNorm));
    CHECK_FALSE(phi_tensor_is_nvfp4(TensorKind::FinalNorm));
}


// The 4-bit body cannot express every near-tie decision (measured 2026-10-24:
// DET-TEST-02 flips on the base plane alone, and no single layer recovers it —
// the error is spread over all 40). Storing a second NVFP4 plane for the
// residual makes the effective precision ~8 bits while reusing the same codec,
// schema and kernels: the decode is one extra GEMV with acc=true.
TEST_CASE("phi residual plane halves the 4-bit reconstruction error",
          "[phi][convert]")
{
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "phi_residual_test";
    remove_all(tmp);
    create_directories(tmp);

    auto tensors = phitest::BuildSyntheticPhiModel();
    phitest::WriteSafetensors(tmp, "model.safetensors", tensors);
    phitest::WriteConfig(tmp);
    copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
              tmp / "tokenizer.json");
    REQUIRE(ConvertPhiModel(tmp.string(), tmp.string()).ok());

    const path nv = tmp / "logestix" / "pinned" / "layer_0_nvfp4.arrow";
    auto names = DenseNvfp4NamesIn(nv);
    for (const char *t : {"self_attn.qkv_proj.weight", "self_attn.o_proj.weight",
                          "mlp.gate_up_proj.weight", "mlp.down_proj.weight"}) {
        const std::string base = std::string("model.layers.0.") + t;
        CHECK(Contains(names, base));
        CHECK(Contains(names, base + ".res"));
    }

    auto batch = lsxcommon::ReadRecordBatchFromFile(nv);
    REQUIRE(batch.ok());
    auto recs = lsxcommon::DenseNvfp4Schema::UnpackBatch(*batch);
    REQUIRE(recs.ok());

    for (const char *t : {"self_attn.qkv_proj.weight", "self_attn.o_proj.weight",
                          "mlp.gate_up_proj.weight", "mlp.down_proj.weight"}) {
        const std::string base = std::string("model.layers.0.") + t;
        const lsxcommon::DenseNvfp4Record *b = nullptr;
        const lsxcommon::DenseNvfp4Record *r = nullptr;
        for (const auto &x : *recs) {
            if (x.name == base) b = &x;
            if (x.name == base + ".res") r = &x;
        }
        REQUIRE(b != nullptr);
        REQUIRE(r != nullptr);

        const auto src = SourceBf16(tensors, base);
        REQUIRE(!src.empty());
        const auto dq_base = DecodeRecord(*b);
        const auto dq_res = DecodeRecord(*r);
        REQUIRE(dq_base.size() == src.size());
        REQUIRE(dq_res.size() == src.size());
        std::vector<float> sum(src.size());
        for (size_t i = 0; i < sum.size(); ++i) sum[i] = dq_base[i] + dq_res[i];

        const float e_base = RelL2(dq_base, src);
        const float e_res = RelL2(sum, src);
        INFO(base << ": base relL2=" << e_base << " base+res=" << e_res);
        CHECK(e_base > 0.0f);
        // A second plane must visibly improve the reconstruction; 2x is the
        // floor a 4-bit residual buys on synthetic weights (measured 2.4-3.2x).
        CHECK(e_res < e_base * 0.5f);
    }

    remove_all(tmp);
}

TEST_CASE("phi converter builds logestix-phi index and arrow layout",
          "[phi][convert]")
{
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "phi_conv_test";
    remove_all(tmp);
    create_directories(tmp);

    auto tensors = phitest::BuildSyntheticPhiModel();
    phitest::WriteSafetensors(tmp, "model.safetensors", tensors);
    phitest::WriteConfig(tmp);
    copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
              tmp / "tokenizer.json");

    auto st = ConvertPhiModel(tmp.string(), tmp.string());
    REQUIRE(st.ok());

    auto top = load_phi_topology(
        (tmp / "logestix" / "model.index.json").string());
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
    CHECK(top.attention_scale > 0.0);

    CHECK(top.vram.min_bytes > 0);
    CHECK(top.vram.optimal_bytes >= top.vram.min_bytes);
    CHECK(top.vram.max_bytes >= top.vram.optimal_bytes);
    CHECK(top.vram.context_tokens > 0);

    CHECK(exists(tmp / "logestix" / "pinned" / "global.arrow"));
    // With FP16 edges the fixture has no global NVFP4 tensors at all, so the
    // file is legitimately absent (the loader treats it as optional).
    CHECK_FALSE(exists(tmp / "logestix" / "pinned" / "global_nvfp4.arrow"));
    CHECK(exists(tmp / "logestix" / "pinned" / "layer_0.arrow"));
    CHECK(exists(tmp / "logestix" / "pinned" / "layer_0_nvfp4.arrow"));
    CHECK(exists(tmp / "logestix" / "pinned" / "layer_1_nvfp4.arrow"));

    auto names = DenseNvfp4NamesIn(tmp / "logestix" / "pinned" /
                                   "layer_0_nvfp4.arrow");
    CHECK(Contains(names, "model.layers.0.self_attn.qkv_proj.weight"));
    CHECK(Contains(names, "model.layers.0.self_attn.o_proj.weight"));
    CHECK(Contains(names, "model.layers.0.mlp.gate_up_proj.weight"));
    CHECK(Contains(names, "model.layers.0.mlp.down_proj.weight"));

    // Edges: embed + lm_head are FP16 (raw BF16 records), never NVFP4.
    auto gnames = DenseNvfp4NamesIn(tmp / "logestix" / "pinned" /
                                    "global_nvfp4.arrow");
    CHECK_FALSE(Contains(gnames, "model.embed_tokens.weight"));
    CHECK_FALSE(Contains(gnames, "lm_head.weight"));

    auto rnames = synthtest::TensorNamesIn(tmp / "logestix" / "pinned" /
                                           "global.arrow");
    CHECK(Contains(rnames, "model.norm.weight"));
    CHECK(Contains(rnames, "model.embed_tokens.weight"));
    CHECK(Contains(rnames, "lm_head.weight"));

    {
        // The index records the weight format so a loader can refuse a tree
        // it does not understand.
        std::ifstream f(tmp / "logestix" / "model.index.json");
        std::string json((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
        CHECK(json.find("\"edges\": \"fp16\"") != std::string::npos);
    }

    remove_all(tmp);
}
