// Xing4.0-29B-A4B synthetic end-to-end converter test (Model Ingestion
// Protocol step 1). Runs the real converter pipeline on a synthetic
// checkpoint — no real weights are read here (Converter test rules 1-5): label
// `convert`, bounded TIMEOUT (set in tests/CMakeLists.txt), and
// LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY as a RUNTIME env var.
//
// What the fixture pins beyond the generic E2E:
//   - the mHC block (attn_hc/ffn_hc .hc_fn/.hc_base/.hc_scale) survives as
//     BF16/F32 records — it is a residual router, not a weight matrix, and
//     quantizing it would break the Sinkhorn mixing;
//   - model.layers.40 (MTP) is stored under mtp/ and counted, not dropped;
//   - the MLA projections are NVFP4 (packed + per-block scales) while the two
//     inner norms stay BF16.

#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/xing_moe_convert.h"
#include "xing_moe_constants.h"


#include <absl/strings/str_cat.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

using lsxcommon::xing::xing_classify;
using X = lsxcommon::xing::XingTensorKind;
namespace fs = std::filesystem;

TEST_CASE("Xing geometry is compile-time locked", "[xing][convert]")
{
    using namespace lsxcommon::xing::frozen;
    STATIC_REQUIRE(kHidden == 3584);
    STATIC_REQUIRE(kLayers == 40);
    STATIC_REQUIRE(kVocab == 131072);
    STATIC_REQUIRE(kHcMult == 4);
    STATIC_REQUIRE(kHcMix == 24);
    STATIC_REQUIRE(kHcFlat == 14336);
    STATIC_REQUIRE(kHeads == 32);
    STATIC_REQUIRE(kKvHeads == 32);
    STATIC_REQUIRE(kQLoraRank == 768);
    STATIC_REQUIRE(kKvLoraRank == 512);
    STATIC_REQUIRE(kQkNopeHeadDim == 128);
    STATIC_REQUIRE(kQkRopeHeadDim == 64);
    STATIC_REQUIRE(kQkHeadDim == 192);
    STATIC_REQUIRE(kVHeadDim == 128);
    STATIC_REQUIRE(kFirstKDense == 2);
    STATIC_REQUIRE(kDenseIntermediate == 9216);
    STATIC_REQUIRE(kMoeIntermediate == 1024);
    STATIC_REQUIRE(kNumRouted == 64);
    STATIC_REQUIRE(kTopK == 4);
    STATIC_REQUIRE(kSharedExperts == 1);
    STATIC_REQUIRE(kRopeDim == 64);
    STATIC_REQUIRE(kMtpLayers[0] == 40);
    STATIC_REQUIRE(kEosId == 2);
}

TEST_CASE("Xing classifier covers every tensor family", "[xing][convert]")
{
    // MLA projections.
    auto t = xing_classify("model.layers.7.self_attn.q_a_proj.weight");
    REQUIRE(t.kind == X::Attention);
    REQUIRE(t.layer == 7);
    REQUIRE(t.base == "q_a_proj");
    REQUIRE(xing_classify("model.layers.7.self_attn.kv_a_proj_with_mqa.weight").kind ==
            X::Attention);
    REQUIRE(xing_classify("model.layers.7.self_attn.kv_b_proj.weight").kind ==
            X::Attention);
    REQUIRE(xing_classify("model.layers.7.self_attn.o_proj.weight").kind == X::Attention);

    // mHC — two blocks per layer, no input_norm tensor in the checkpoint.
    t = xing_classify("model.layers.7.attn_hc.hc_fn");
    REQUIRE(t.kind == X::HyperConnection);
    REQUIRE(t.layer == 7);
    REQUIRE(t.base == "attn_hc.hc_fn");
    REQUIRE(xing_classify("model.layers.7.attn_hc.hc_base").kind == X::HyperConnection);
    REQUIRE(xing_classify("model.layers.7.attn_hc.hc_scale").kind == X::HyperConnection);
    REQUIRE(xing_classify("model.layers.7.ffn_hc.hc_fn").kind == X::HyperConnection);
    // A weight on the unweighted norm would mean a different architecture.
    REQUIRE(xing_classify("model.layers.7.attn_hc.input_norm.weight").kind ==
            X::Unknown);

    // Norms.
    REQUIRE(xing_classify("model.layers.7.input_layernorm.weight").kind == X::LayerNorm);
    REQUIRE(xing_classify("model.layers.7.post_attention_layernorm.weight").kind ==
            X::LayerNorm);
    REQUIRE(xing_classify("model.layers.7.self_attn.q_a_layernorm.weight").kind ==
            X::LayerNorm);
    REQUIRE(xing_classify("model.layers.7.self_attn.kv_a_layernorm.weight").kind ==
            X::LayerNorm);

    // Dense MLP prefix / MoE body.
    REQUIRE(xing_classify("model.layers.1.mlp.gate_proj.weight").kind == X::DenseMlp);
    REQUIRE(xing_classify("model.layers.1.mlp.up_proj.weight").kind == X::DenseMlp);
    REQUIRE(xing_classify("model.layers.1.mlp.down_proj.weight").kind == X::DenseMlp);

    REQUIRE(xing_classify("model.layers.5.mlp.gate.weight").kind == X::Router);
    REQUIRE(xing_classify("model.layers.5.mlp.gate.e_score_correction_bias").kind ==
            X::Router);
    t = xing_classify("model.layers.5.mlp.experts.42.gate_proj.weight");
    REQUIRE(t.kind == X::RoutedExpert);
    REQUIRE(t.expert == 42);
    REQUIRE(t.weight_id == 1);
    REQUIRE(xing_classify("model.layers.5.mlp.experts.42.up_proj.weight").weight_id == 2);
    REQUIRE(xing_classify("model.layers.5.mlp.experts.42.down_proj.weight").weight_id == 3);
    REQUIRE(xing_classify("model.layers.5.mlp.shared_experts.gate_proj.weight").kind ==
            X::SharedExpert);
    REQUIRE(xing_classify("model.layers.5.mlp.shared_experts.down_proj.weight").kind ==
            X::SharedExpert);

    // Globals.
    REQUIRE(xing_classify("model.embed_tokens.weight").kind == X::Global);
    REQUIRE(xing_classify("lm_head.weight").kind == X::Global);
    REQUIRE(xing_classify("model.norm.weight").kind == X::Global);

    // Stored MTP block: layers.40 is a FULL decoder layer plus the speculative
    // head, so its attention/MoE tensors must be Mtp too - keying the block by
    // tensor identity would scatter it into the resident graph.
    REQUIRE(xing_classify("model.layers.40.embed_tokens.weight").kind == X::Mtp);
    REQUIRE(xing_classify("model.layers.40.eh_proj.weight").kind == X::Mtp);
    REQUIRE(xing_classify("model.layers.40.shared_head.head.weight").kind == X::Mtp);
    REQUIRE(xing_classify("model.layers.40.enorm.weight").kind == X::Mtp);
    REQUIRE(xing_classify("model.layers.40.self_attn.q_a_proj.weight").kind == X::Mtp);
    REQUIRE(xing_classify("model.layers.40.mlp.experts.7.gate_proj.weight").kind ==
            X::Mtp);
    REQUIRE(xing_classify("model.layers.40.mlp.gate.weight").kind == X::Mtp);
    REQUIRE(xing_classify("model.layers.40.input_layernorm.weight").kind == X::Mtp);
    // The last resident layer is still a normal layer.
    REQUIRE(xing_classify("model.layers.39.self_attn.q_a_proj.weight").kind ==
            X::Attention);

    // Foreign towers classify Unknown (stripped ballast).
    REQUIRE(xing_classify("vision_tower.blocks.0.attn.qkv.weight").kind == X::Unknown);
    REQUIRE(xing_classify("mm_projector.linear.weight").kind == X::Unknown);
}

#include "test_xing_moe_fixture.h"

TEST_CASE("Xing synthetic convert emits logestix-xing-nvfp4", "[xing][convert]")
{
    auto in = fs::temp_directory_path() / "xing_conv_fixture";
    auto out = fs::temp_directory_path() / "xing_conv_out";
    fs::remove_all(in);
    fs::remove_all(out);
    fs::create_directories(in);

    xingtest::WriteXingConfig(in);
    xingtest::WriteSafetensors(in, "model-00001-of-00001.safetensors",
                               xingtest::BuildSyntheticXingModel(3, 2));

    const auto st = lsxcommon::xing::ConvertXingModel(in.string(), out);
    INFO("converter status: " << st.ToString());
    REQUIRE(st.ok());

    const auto log_root = out / "logestix";
    REQUIRE(fs::exists(log_root / "model.index.json"));
    REQUIRE(fs::exists(log_root / "pinned" / "global_nvfp4.arrow"));
    // Raw (norms/mHC/router bias) and NVFP4 (linears) live in sibling files —
    // one Arrow schema each, so they cannot share a batch.
    for (int l = 0; l <= 2; ++l)
    {
        REQUIRE(fs::exists(log_root / "pinned" / absl::StrCat("layer_", l, ".arrow")));
        REQUIRE(fs::exists(log_root / "pinned" /
                           absl::StrCat("layer_", l, "_nvfp4.arrow")));
    }
    // Routed experts: one arrow per (layer, expert).
    for (int e = 0; e < 2; ++e)
        REQUIRE(fs::exists(log_root / "experts" / "layer_2" /
                           absl::StrCat("exp_", e, ".arrow")));
    // MTP is stored, not dropped.
    REQUIRE(fs::exists(log_root / "mtp" / "raw.arrow"));

    std::ifstream f(log_root / "model.index.json");
    const std::string json((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
    REQUIRE(json.find("\"logestix-xing-nvfp4\"") != std::string::npos);
    REQUIRE(json.find("\"dense_layers\": [0,1]") != std::string::npos);
    REQUIRE(json.find("\"moe_layers\": [2]") != std::string::npos);
    REQUIRE(json.find("\"mtp_layers\": [40]") != std::string::npos);
}

TEST_CASE("Xing converter: mHC stays BF16, MLA projections go NVFP4",
          "[xing][convert]")
{
    auto in = fs::temp_directory_path() / "xing_conv_hc_fixture";
    auto out = fs::temp_directory_path() / "xing_conv_hc_out";
    fs::remove_all(in);
    fs::remove_all(out);
    fs::create_directories(in);

    xingtest::WriteXingConfig(in);
    xingtest::WriteSafetensors(in, "model-00001-of-00001.safetensors",
                               xingtest::BuildSyntheticXingModel(3, 2));

    const auto st = lsxcommon::xing::ConvertXingModel(in.string(), out);
    INFO("converter status: " << st.ToString());
    REQUIRE(st.ok());

    const auto log_root = fs::temp_directory_path() / "xing_conv_hc_out" / "logestix";
    auto raw = lsxcommon::LoadTensorBatchViews(log_root / "pinned" / "layer_2.arrow");
    REQUIRE(raw.ok());
    auto nvfp4 = lsxcommon::LoadDenseNvfp4BatchViews(
        log_root / "pinned" / "layer_2_nvfp4.arrow");
    REQUIRE(nvfp4.ok());
    const auto find_raw = [&](std::string_view n) -> const lsxcommon::TensorView * {
        for (const auto &v : raw->tensors)
            if (v.name == n) return &v;
        return nullptr;
    };
    const auto find_nvfp4 =
        [&](std::string_view n) -> const lsxcommon::DenseNvfp4View * {
        for (const auto &v : nvfp4->records)
            if (v.name == n) return &v;
        return nullptr;
    };

    // mHC: hc_fn/hc_base BF16, hc_scale F32 - the residual mixing parameters are
    // never quantized. hc_fn is [24, 4*16] = 1536 elems -> 3072 B.
    const auto *hc_fn = find_raw("model.layers.2.attn_hc.hc_fn");
    REQUIRE(hc_fn != nullptr);
    REQUIRE(hc_fn->dtype == "BF16");
    REQUIRE(hc_fn->bytes == 3072);
    const auto *hc_base = find_raw("model.layers.2.attn_hc.hc_base");
    REQUIRE(hc_base != nullptr);
    REQUIRE(hc_base->dtype == "BF16");
    REQUIRE(hc_base->bytes == 48);  // [24]
    const auto *hc_scale = find_raw("model.layers.2.ffn_hc.hc_scale");
    REQUIRE(hc_scale != nullptr);
    REQUIRE(hc_scale->dtype == "F32");
    REQUIRE(hc_scale->bytes == 12);  // [3]

    // MLA q_a_proj: BF16 [16,16] -> 128 packed bytes + 16 E4M3 scale bytes.
    const auto *qa = find_nvfp4("model.layers.2.self_attn.q_a_proj.weight");
    REQUIRE(qa != nullptr);
    REQUIRE(qa->packed_bytes == 128);
    REQUIRE(qa->scale_bytes == 16);
    REQUIRE(qa->rank == 2);
    // The raw BF16 copy must NOT survive: quantization replaced it.
    REQUIRE(find_raw("model.layers.2.self_attn.q_a_proj.weight") == nullptr);

    // Inner norms stay BF16 (norms are never quantized).
    const auto *qa_ln = find_raw("model.layers.2.self_attn.q_a_layernorm.weight");
    REQUIRE(qa_ln != nullptr);
    REQUIRE(qa_ln->dtype == "BF16");
    REQUIRE(qa_ln->bytes == 32);  // [16]

    // BOTH router tensors stay high precision: the weight because its argmax
    // picks the expert set (NVFP4 rounding there swaps experts instead of
    // blurring an output), the bias because upstream lists it in
    // _keep_in_fp32_modules_strict.
    const auto *bias = find_raw("model.layers.2.mlp.gate.e_score_correction_bias");
    REQUIRE(bias != nullptr);
    REQUIRE(bias->dtype == "F32");
    const auto *router = find_raw("model.layers.2.mlp.gate.weight");
    REQUIRE(router != nullptr);
    REQUIRE(router->dtype == "BF16");
    REQUIRE(find_nvfp4("model.layers.2.mlp.gate.weight") == nullptr);
}
