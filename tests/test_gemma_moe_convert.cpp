// Gemma-4 converter test: tensor name classifier + NVFP4 policy.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "test_gemma_moe_fixture.h"

#include <filesystem>

#include "nvfp4_scale.h"

#include "lsxcommon/gemma_moe_convert.h"
#include "lsxcommon/model_integrity.h"
#include "lsxcommon/schema.h"

using lsxcommon::gemma::classify_gemma_tensor;
using K = lsxcommon::gemma::TensorKind;

TEST_CASE("gemma tensor classifier", "[gemma][convert]") {
    SECTION("globals") {
        CHECK(classify_gemma_tensor("model.language_model.embed_tokens.weight").kind == K::Embed);
        CHECK(classify_gemma_tensor("model.language_model.norm.weight").kind == K::FinalNorm);
    }
    SECTION("vision/audio fat skipped") {
        CHECK(classify_gemma_tensor("model.vision_tower.patch_embedder.input_proj.weight").kind == K::Visual);
        CHECK(classify_gemma_tensor("model.embed_vision.embedding_projection.weight").kind == K::Visual);
        CHECK(classify_gemma_tensor("model.vision_tower.encoder.layers.5.self_attn.q_proj.linear.weight").kind == K::Visual);
        CHECK(classify_gemma_tensor("model.vision_tower.std_scale").kind == K::Visual);
        CHECK(classify_gemma_tensor("model.vision_tower.std_bias").kind == K::Visual);
    }
    SECTION("attention") {
        auto q = classify_gemma_tensor("model.language_model.layers.3.self_attn.q_proj.weight");
        CHECK(q.kind == K::AttnQ);
        CHECK(q.layer == 3);
        CHECK(classify_gemma_tensor("model.language_model.layers.0.self_attn.k_norm.weight").kind == K::AttnKNorm);
        CHECK(classify_gemma_tensor("model.language_model.layers.0.self_attn.q_norm.weight").kind == K::AttnQNorm);
        CHECK(classify_gemma_tensor("model.language_model.layers.0.self_attn.o_proj.weight").kind == K::AttnO);
    }
    SECTION("dense mlp") {
        auto g = classify_gemma_tensor("model.language_model.layers.7.mlp.gate_proj.weight");
        CHECK(g.kind == K::MlpGate);
        CHECK(g.layer == 7);
        CHECK(classify_gemma_tensor("model.language_model.layers.7.mlp.up_proj.weight").kind == K::MlpUp);
        CHECK(classify_gemma_tensor("model.language_model.layers.7.mlp.down_proj.weight").kind == K::MlpDown);
    }
    SECTION("moe") {
        auto e = classify_gemma_tensor("model.language_model.layers.9.experts.gate_up_proj");
        CHECK(e.kind == K::ExpertsGateUp);
        CHECK(e.layer == 9);
        CHECK(classify_gemma_tensor("model.language_model.layers.9.experts.down_proj").kind == K::ExpertsDown);
        CHECK(classify_gemma_tensor("model.language_model.layers.9.router.proj.weight").kind == K::RouterProj);
        CHECK(classify_gemma_tensor("model.language_model.layers.9.router.scale").kind == K::RouterScale);
        CHECK(classify_gemma_tensor("model.language_model.layers.9.router.per_expert_scale").kind == K::RouterPerExpertScale);
    }
    SECTION("norms + scalar") {
        CHECK(classify_gemma_tensor("model.language_model.layers.1.input_layernorm.weight").kind == K::InputNorm);
        CHECK(classify_gemma_tensor("model.language_model.layers.1.post_attention_layernorm.weight").kind == K::PostNorm);
        CHECK(classify_gemma_tensor("model.language_model.layers.1.pre_feedforward_layernorm.weight").kind == K::PreFFN);
        CHECK(classify_gemma_tensor("model.language_model.layers.1.post_feedforward_layernorm.weight").kind == K::PostFFN);
        CHECK(classify_gemma_tensor("model.language_model.layers.1.pre_feedforward_layernorm_2.weight").kind == K::PreFFN2);
        CHECK(classify_gemma_tensor("model.language_model.layers.1.post_feedforward_layernorm_2.weight").kind == K::PostFFN2);
        CHECK(classify_gemma_tensor("model.language_model.layers.1.layer_scalar").kind == K::LayerScalar);
    }
    SECTION("unknown") {
        CHECK(classify_gemma_tensor("something.else.weight").kind == K::Unknown);
    }
}

TEST_CASE("nvfp4 policy: experts only", "[gemma][convert]") {
    using lsxcommon::gemma::gemma_tensor_is_nvfp4;
    CHECK(gemma_tensor_is_nvfp4(K::ExpertsGateUp));
    CHECK(gemma_tensor_is_nvfp4(K::ExpertsDown));
    CHECK_FALSE(gemma_tensor_is_nvfp4(K::AttnQ));
    CHECK_FALSE(gemma_tensor_is_nvfp4(K::Embed));
    CHECK_FALSE(gemma_tensor_is_nvfp4(K::FinalNorm));
    CHECK_FALSE(gemma_tensor_is_nvfp4(K::MlpGate));
    CHECK_FALSE(gemma_tensor_is_nvfp4(K::RouterProj));
}

TEST_CASE("gemma topology parse + convert happy path", "[gemma][convert]") {
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "gemma4_conv_test";
    remove_all(tmp);
    create_directories(tmp);

    auto tensors = gemmatest::BuildSyntheticGemmaModel();
    gemmatest::WriteSafetensors(tmp, "model.safetensors", tensors);
    gemmatest::WriteConfig(tmp);
    std::filesystem::copy_file(
        TOKENIZER_TEST_DATA_DIR "/synthetic_gemma_tok.json",
        tmp / "tokenizer.json");

    auto top = lsxcommon::gemma::load_gemma_topology((tmp / "config.json").string());
    REQUIRE(top.error_msg.empty());
    CHECK(top.num_layers == 4);
    CHECK(top.hidden_size == 64);
    CHECK(top.vocab_size == 64);
    CHECK(top.num_experts == 8);
    CHECK(top.top_k_experts == 2);
    CHECK(top.moe_intermediate_size == 32);  // 32-multiple: routed-GEMV K
    CHECK(top.num_global_key_value_heads == 1);
    CHECK(top.global_head_dim == 32);
    CHECK(top.sliding_window == 8);
    CHECK(top.bos_id == 2);
    // Full eos set from the top-level config array [1, 106] plus the sibling
    // generation_config's [1, 106, 50] (the model ends a turn on 106).
    CHECK(top.eos_ids == std::vector<int>({1, 106, 50}));
    CHECK(top.layer_types.size() == 4);

    auto out = tmp / "out";
    auto cst = lsxcommon::gemma::ConvertGemmaModel(tmp.string(), out.string());
    if (!cst.ok()) FAIL("ConvertGemmaModel: " << cst.ToString());
    REQUIRE(cst.ok());

    // GLM-style layout: pinned raw + per-expert NVFP4 files.
    CHECK(exists(out / "logestix" / "pinned" / "global.arrow"));
    CHECK(exists(out / "logestix" / "pinned" / "layer_0.arrow"));
    CHECK_FALSE(exists(out / "logestix" / "pinned" / "layer_0_nvfp4.arrow"));
    for (int l = 0; l < 4; ++l)
        for (int e = 0; e < 8; ++e)
            CHECK(exists(out / "logestix" / "experts" /
                         ("layer_" + std::to_string(l)) /
                         ("exp_" + std::to_string(e) + ".arrow")));

    // Index round-trip.
    auto top2 = lsxcommon::gemma::load_gemma_topology(
        (out / "logestix" / "model.index.json").string());
    // The index has no text_config nesting; parse must still succeed via
    // top-level keys written by WriteGemmaModelIndex.
    CHECK(top2.error_msg.empty());
    CHECK(top2.num_layers == 4);
    CHECK(top2.num_experts == 8);
    CHECK(top2.eos_ids == std::vector<int>({1, 106, 50}));
    CHECK(top2.vram.min_bytes > 0);
    CHECK(top2.total_pinned_tensors > 0);

    remove_all(tmp);
}

TEST_CASE("convert fails without config.json", "[gemma][convert]") {
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "gemma4_conv_noconf";
    remove_all(tmp);
    create_directories(tmp);

    auto tensors = gemmatest::BuildSyntheticGemmaModel();
    gemmatest::WriteSafetensors(tmp, "model.safetensors", tensors);
    std::filesystem::copy_file(
        TOKENIZER_TEST_DATA_DIR "/synthetic_gemma_tok.json",
        tmp / "tokenizer.json");

    auto out = tmp / "out";
    auto st = lsxcommon::gemma::ConvertGemmaModel(tmp.string(), out.string());
    REQUIRE_FALSE(st.ok());
    remove_all(tmp);
}

TEST_CASE("convert fails on layer_types contradicting tensor shapes",
          "[gemma][convert]") {
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "gemma4_conv_badlt";
    remove_all(tmp);
    create_directories(tmp);

    auto tensors = gemmatest::BuildSyntheticGemmaModel();
    gemmatest::WriteSafetensors(tmp, "model.safetensors", tensors);
    gemmatest::WriteConfig(tmp);
    std::filesystem::copy_file(
        TOKENIZER_TEST_DATA_DIR "/synthetic_gemma_tok.json",
        tmp / "tokenizer.json");

    // Flip layer 3 to sliding in the config while its q_proj keeps the
    // full-attention width -> geometry gate must reject.
    auto cfg = tmp / "config.json";
    std::ifstream in(cfg);
    std::string json((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
    in.close();
    auto pos = json.find("\"full_attention\"");
    REQUIRE(pos != std::string::npos);
    json.replace(pos, strlen("\"full_attention\""), "\"sliding_attention\"");
    { std::ofstream out(cfg, std::ios::trunc); out << json; }

    auto out = tmp / "out";
    auto st = lsxcommon::gemma::ConvertGemmaModel(tmp.string(), out.string());
    REQUIRE_FALSE(st.ok());
    remove_all(tmp);
}

namespace {

std::vector<float> Bf16Floats(const std::vector<uint8_t> &bytes)
{
    std::vector<float> out(bytes.size() / 2);
    for (size_t i = 0; i < out.size(); ++i) {
        const uint16_t bits = static_cast<uint16_t>(
            bytes[2 * i] | (static_cast<uint16_t>(bytes[2 * i + 1]) << 8));
        const uint32_t f32 = static_cast<uint32_t>(bits) << 16;
        std::memcpy(&out[i], &f32, sizeof(float));
    }
    return out;
}

// Host decode of one NVFP4 record triad (packed + scales + global_scale) as the
// converter wrote it. Test-side on purpose: the residual case is an
// independent check of the stored bits, not of the decoder.
std::vector<float> DecodeTriad(const std::vector<lsxcommon::TensorView> &ts,
                               const std::string &base,
                               const std::string &suffix)
{
    static const float kMag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const lsxcommon::TensorView *pk = nullptr;
    const lsxcommon::TensorView *sc = nullptr;
    const lsxcommon::TensorView *gs = nullptr;
    for (const auto &t : ts) {
        if (t.name == base + suffix + ".packed") pk = &t;
        if (t.name == base + suffix + ".scales") sc = &t;
        if (t.name == base + suffix + ".global_scale") gs = &t;
    }
    if (!pk || !sc || !gs) return {};
    float g = 0.f;
    std::memcpy(&g, gs->data, sizeof(float));
    const int64_t n = static_cast<int64_t>(pk->bytes) * 2;
    std::vector<float> out(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        const auto *bytes = static_cast<const uint8_t *>(pk->data);
        const uint8_t byte = bytes[i >> 1];
        const int nib = (i & 1) ? (byte >> 4) : (byte & 0x0F);
        const float mag = kMag[nib & 7] * ((nib & 8) ? -1.0f : 1.0f);
        const float scale = lsxcommon::nvfp4::E4m3ToFloat(
            static_cast<const uint8_t *>(sc->data)[i >> 4]);
        out[static_cast<size_t>(i)] = mag * scale * g;
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

} // namespace

// The 4-bit body drifts from the bf16 source by cos 0.965 at the last layer
// (measured 2026-10-24), i.e. ~3.5 logits rms on the head, and that flips the
// FSI/CODE template near-ties. An offline recipe ablation showed the cheapest
// sufficient fix is a second NVFP4 plane (w - dequant(base), decoded with
// acc=true) on the attention projections and the dense MLP: experts stay 4-bit
// so the streamed expert bytes and decode t/s are untouched.
TEST_CASE("gemma MoE residual plane covers attn + dense MLP, never the experts",
          "[gemma][convert][nvfp4][residual]") {
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "gemma4_conv_residual";
    remove_all(tmp);
    create_directories(tmp);

    auto tensors = gemmatest::BuildSyntheticGemmaModel();
    // The shared fixture builds projection weights from raw 16-bit patterns
    // (NaN/Inf-heavy) which the quantizer flattens to zeros -- useless for a
    // reconstruction comparison. Re-seed those tensors with a smooth,
    // deterministic, NVFP4-imperfect signal so the plane's effect is visible.
    for (auto &t : tensors) {
        if (t.name.find(".weight") == std::string::npos) continue;
        if (t.name.find("norm") != std::string::npos) continue;
        if (t.name.find("router") != std::string::npos) continue;
        if (t.name.find("embed_tokens") != std::string::npos) continue;
        const int64_t n = static_cast<int64_t>(t.data.size() / 2);
        t.data = gemmatest::MakeBf16Values(n, [](int64_t i) {
            return 0.3f * std::sin(0.37f * static_cast<float>(i)) +
                   0.15f * std::sin(1.9f * static_cast<float>(i));
        });
    }
    gemmatest::WriteSafetensors(tmp, "model.safetensors", tensors);
    gemmatest::WriteConfig(tmp);
    std::filesystem::copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_gemma_tok.json",
                               tmp / "tokenizer.json");

    auto out = tmp / "out";
    auto cst = lsxcommon::gemma::ConvertGemmaModel(tmp.string(), out.string());
    if (!cst.ok()) FAIL("ConvertGemmaModel: " << cst.ToString());
    REQUIRE(cst.ok());

    auto batch_res = lsxcommon::LoadTensorBatchViews(
        out / "logestix" / "pinned" / "layer_0.arrow");
    REQUIRE(batch_res.ok());
    const auto &ts = batch_res->tensors;

    for (const char *t : {"self_attn.q_proj.weight", "self_attn.k_proj.weight",
                          "self_attn.o_proj.weight",
                          "mlp.gate_proj.weight", "mlp.up_proj.weight",
                          "mlp.down_proj.weight"}) {
        const std::string base = std::string("model.language_model.layers.0.") +
                                 t;
        const std::string res = base + ".res";
        bool have_res = false;
        for (const auto &v : ts)
            if (v.name == res + ".packed") have_res = true;
        INFO(base);
        REQUIRE(have_res);
        const auto dq_base = DecodeTriad(ts, base, "");
        const auto dq_res = DecodeTriad(ts, base, ".res");
        REQUIRE(!dq_base.empty());
        REQUIRE(dq_res.size() == dq_base.size());
        std::vector<float> sum(dq_base.size());
        for (size_t i = 0; i < sum.size(); ++i) sum[i] = dq_base[i] + dq_res[i];
        std::vector<float> src;
        for (const auto &ft : tensors)
            if (ft.name == base) src = Bf16Floats(ft.data);
        REQUIRE(src.size() == sum.size());
        const float e_base = RelL2(dq_base, src);
        const float e_res = RelL2(sum, src);
        INFO(base << ": base relL2=" << e_base << " base+res=" << e_res);
        CHECK(e_base > 0.0f);
        CHECK(e_res < e_base * 0.5f);
    }

    // The router stays high precision (a 4-bit router swaps the expert set).
    for (const auto &v : ts)
        CHECK(v.name.find("router.proj.weight.res") == std::string::npos);
    // The tied table keeps its 4-bit plane: measured, the head costs < 1
    // logit while the body owns the drift, so a residual there is pure cost.
    {
        auto g = lsxcommon::ReadRecordBatchFromFile(
            out / "logestix" / "pinned" / "global_nvfp4.arrow");
        REQUIRE(g.ok());
        auto recs = lsxcommon::DenseNvfp4Schema::UnpackBatch(*g);
        REQUIRE(recs.ok());
        for (const auto &r : *recs)
            CHECK(r.name.find(".res") == std::string::npos);
    }

    batch_res->keepalive.reset();
    remove_all(tmp);
}

TEST_CASE("gemma converter emits NVFP4 records for dense MLP",
          "[gemma][convert][nvfp4]") {
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "gemma4_conv_mlp_nvfp4";
    remove_all(tmp);
    create_directories(tmp);

    auto tensors = gemmatest::BuildSyntheticGemmaModel();
    gemmatest::WriteSafetensors(tmp, "model.safetensors", tensors);
    gemmatest::WriteConfig(tmp);
    std::filesystem::copy_file(
        TOKENIZER_TEST_DATA_DIR "/synthetic_gemma_tok.json",
        tmp / "tokenizer.json");

    auto out = tmp / "out";
    auto cst = lsxcommon::gemma::ConvertGemmaModel(tmp.string(), out.string());
    if (!cst.ok()) FAIL("ConvertGemmaModel: " << cst.ToString());
    REQUIRE(cst.ok());

    auto batch_res = lsxcommon::LoadTensorBatchViews(
        out / "logestix" / "pinned" / "layer_0.arrow");
    REQUIRE(batch_res.ok());

    auto find = [&](std::string_view name) -> const lsxcommon::TensorView * {
        for (const auto &t : batch_res->tensors)
            if (t.name == name) return &t;
        return nullptr;
    };
    const auto &cfg = gemmatest::TestCfg();
    const size_t numel =
        static_cast<size_t>(cfg.intermediate_size) * cfg.hidden_size;

    const auto *pk = find("model.language_model.layers.0.mlp.gate_proj.weight.packed");
    const auto *sc = find("model.language_model.layers.0.mlp.gate_proj.weight.scales");
    const auto *gs = find("model.language_model.layers.0.mlp.gate_proj.weight.global_scale");
    REQUIRE(pk != nullptr);
    REQUIRE(sc != nullptr);
    REQUIRE(gs != nullptr);
    CHECK(pk->bytes == numel / 2);         // E2M1 nibble-packed, 2 elems/byte
    CHECK(sc->bytes == numel / 16);        // one E8M0-style scale byte per 16 elems
    CHECK(gs->bytes == 4);                 // F32 global scale
    // up/down triples exist too
    CHECK(find("model.language_model.layers.0.mlp.up_proj.weight.packed") != nullptr);
    CHECK(find("model.language_model.layers.0.mlp.down_proj.weight.packed") != nullptr);
    // the raw BF16 MLP tensors are gone
    CHECK(find("model.language_model.layers.0.mlp.gate_proj.weight") == nullptr);

    // Index advertises the mlp_nvfp4 flag.
    {
        std::ifstream idx(out / "logestix" / "model.index.json");
        std::string json((std::istreambuf_iterator<char>(idx)),
                         std::istreambuf_iterator<char>());
        CHECK(json.find("\"mlp_nvfp4\": true") != std::string::npos);
    }

    // Release every handle before the cleanup: Windows refuses to delete a
    // mapped or open file, while Linux happily unlinks both and hides the
    // ordering bug.
    batch_res->keepalive.reset();

    remove_all(tmp);
}
