// GLM-5.3-Flash converter suite: frozen geometry, tensor classifier,
// FP8->NVFP4 requant, config parser, conversion pipeline.
// The end-to-end converter test runs on a synthetic tiny checkpoint
// (unified converter-suite discipline: no real-model conversions).
#include <catch2/catch_test_macros.hpp>

#include <simdjson.h>

#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "test_manifest_query_root.h"
#include "glm_moe_flash_constants.h"
#include "test_glm_moe_flash_fixture.h"

namespace fs = std::filesystem;

using namespace lsxcommon::glm_moe_flash;

TEST_CASE("frozen geometry passport matches GLM-5.3-Flash config")
{
    using namespace frozen;
    STATIC_REQUIRE(kHidden == 4096);
    STATIC_REQUIRE(kLayers == 45);
    STATIC_REQUIRE(kMtpLayer == 45);
    STATIC_REQUIRE(kHeads == 64);
    STATIC_REQUIRE(kHeadDim == 128);
    STATIC_REQUIRE(kQkHeadDim == 256);
    STATIC_REQUIRE(kQLora == 1536);
    STATIC_REQUIRE(kKvLora == 512);
    STATIC_REQUIRE(kNumExperts == 288);
    STATIC_REQUIRE(kTopK == 8);
    STATIC_REQUIRE(kMoeIntermediate == 2048);
    STATIC_REQUIRE(kDenseIntermediate == 12288);
    STATIC_REQUIRE(kHcCount == 4);
    STATIC_REQUIRE(kConvKernel == 4);
    STATIC_REQUIRE(kGateLowrank == 128);
    STATIC_REQUIRE(kIndexerHeads == 32);
    STATIC_REQUIRE(kIndexerTopk == 2048);
    STATIC_REQUIRE(kVocab == 154880);
    STATIC_REQUIRE(kFirstKDense == 3);
    // 34 KDA + 11 DSA == 45
    STATIC_REQUIRE(kKdaLayerCount == 34);
    STATIC_REQUIRE(kDsaLayerCount == 11);
    REQUIRE(IsKdaLayer(0));
    REQUIRE(IsKdaLayer(44));
    REQUIRE_FALSE(IsKdaLayer(3));
    REQUIRE(IsDsaLayer(43));
    REQUIRE_FALSE(IsDsaLayer(42));
    REQUIRE(IsMoeLayer(3));
    REQUIRE(IsMoeLayer(45));
    REQUIRE_FALSE(IsMoeLayer(2));
}

TEST_CASE("classify_flash_tensor: KDA layer tensors")
{
    auto p = classify_flash_tensor(
        "model.language_model.layers.0.self_attn.q_proj.weight");
    REQUIRE(p.kind == FlashTensorKind::KdaQ);
    REQUIRE(p.layer == 0);
    p = classify_flash_tensor(
        "model.language_model.layers.4.self_attn.k_conv1d.weight");
    REQUIRE(p.kind == FlashTensorKind::KdaConv);
    p = classify_flash_tensor(
        "model.language_model.layers.4.self_attn.dt_bias");
    REQUIRE(p.kind == FlashTensorKind::KdaDtBias);
    p = classify_flash_tensor(
        "model.language_model.layers.4.self_attn.b_proj.weight");
    REQUIRE(p.kind == FlashTensorKind::KdaB);
}

TEST_CASE("classify_flash_tensor: malformed indices are Unknown")
{
    // Digit runs are unbounded in the name; a corrupt (over-range) index must
    // not be read by atoi (UB on overflow) but rejected as Unknown.
    CHECK(classify_flash_tensor(
              "model.language_model.layers.99999999999999999999."
              "self_attn.q_proj.weight")
              .kind == FlashTensorKind::Unknown);
    CHECK(classify_flash_tensor(
              "model.language_model.layers.5.mlp.experts.99999999999999999999."
              "gate_proj.weight")
              .kind == FlashTensorKind::Unknown);
}

TEST_CASE("classify_flash_tensor: DSA layer tensors")
{
    auto p = classify_flash_tensor(
        "model.language_model.layers.3.self_attn.q_a_proj.weight");
    REQUIRE(p.kind == FlashTensorKind::DsaQa);
    p = classify_flash_tensor(
        "model.language_model.layers.3.self_attn.indexer.wq_b.weight");
    REQUIRE(p.kind == FlashTensorKind::Indexer);
    REQUIRE(p.layer == 3);
}

TEST_CASE("classify_flash_tensor: MoE + hc + global")
{
    auto p = classify_flash_tensor(
        "model.language_model.layers.5.mlp.experts.42.gate_proj.weight");
    REQUIRE(p.kind == FlashTensorKind::RoutedExpert);
    REQUIRE(p.layer == 5);
    REQUIRE(p.expert == 42);
    REQUIRE(p.proj_id == 1);
    p = classify_flash_tensor(
        "model.language_model.layers.5.mlp.gate.e_score_correction_bias");
    REQUIRE(p.kind == FlashTensorKind::RouterBias);
    p = classify_flash_tensor("model.language_model.layers.5.hc_attn_fn");
    REQUIRE(p.kind == FlashTensorKind::HcAttnFn);
    p = classify_flash_tensor("lm_head.weight");
    REQUIRE(p.kind == FlashTensorKind::LmHead);
    p = classify_flash_tensor("model.visual.patch_embed.proj.weight");
    REQUIRE(p.kind == FlashTensorKind::Visual);
    p = classify_flash_tensor("model.language_model.layers.45.eh_proj.weight");
    REQUIRE(p.kind == FlashTensorKind::MtpEhProj);
    p = classify_flash_tensor(
        "model.language_model.layers.45.self_attn.indexer.wq_b.weight");
    REQUIRE(p.kind == FlashTensorKind::Indexer);
    REQUIRE(p.layer == 45);
    p = classify_flash_tensor(
        "model.language_model.layers.45.mlp.experts.0.gate_proj.weight");
    REQUIRE(p.kind == FlashTensorKind::RoutedExpert);
    REQUIRE(p.layer == 45);
}

TEST_CASE("flash_tensor_is_nvfp4 policy")
{
    REQUIRE(flash_tensor_is_nvfp4(FlashTensorKind::RoutedExpert));
    REQUIRE(flash_tensor_is_nvfp4(FlashTensorKind::SharedExpert));
    REQUIRE(flash_tensor_is_nvfp4(FlashTensorKind::DsaQa));
    REQUIRE(flash_tensor_is_nvfp4(FlashTensorKind::DsaOBig));
    REQUIRE(flash_tensor_is_nvfp4(FlashTensorKind::DenseMlp));
    // KDA attention projections joined the NVFP4 set (perf: the four
    // [4096,8192] fp16 pairs dominated device-dense; commit 24deae1).
    REQUIRE(flash_tensor_is_nvfp4(FlashTensorKind::KdaQ));
    REQUIRE(flash_tensor_is_nvfp4(FlashTensorKind::KdaK));
    REQUIRE(flash_tensor_is_nvfp4(FlashTensorKind::KdaV));
    REQUIRE(flash_tensor_is_nvfp4(FlashTensorKind::KdaO));
    // Norms/biases/conv/hc/gates/embed stay high precision.
    REQUIRE_FALSE(flash_tensor_is_nvfp4(FlashTensorKind::KdaConv));
    REQUIRE_FALSE(flash_tensor_is_nvfp4(FlashTensorKind::HcAttnFn));
    REQUIRE_FALSE(flash_tensor_is_nvfp4(FlashTensorKind::RouterGate));
    REQUIRE_FALSE(flash_tensor_is_nvfp4(FlashTensorKind::LmHead));
    REQUIRE_FALSE(flash_tensor_is_nvfp4(FlashTensorKind::DsaKvB));
    REQUIRE_FALSE(flash_tensor_is_nvfp4(FlashTensorKind::Indexer));
}

// ── Task 2: FP8→NVFP4 requant + config parser ────────────────────────────
#include "lsxcommon/glm_moe_flash_convert.h"
#include "glm_moe_flash_fp8.h"

#include <cmath>

TEST_CASE("ParseFlashConfig: real checkpoint text_config")
{
    manifest_query::RequireSourceRoot();
    const fs::path dir = manifest_query::SourceDirFor("glm-5.3-flash");
    if (!fs::exists(dir / "config.json")) {
        SKIP(std::string("GLM-5.3-Flash checkpoint not present: ") +
             dir.string());
    }
    auto top = ParseFlashConfig(dir);
    REQUIRE(top.error_msg.empty());
    REQUIRE(top.hidden == 4096);
    REQUIRE(top.layers == 45);
    REQUIRE(top.vocab == 154880);
    REQUIRE(top.num_experts == 288);
    REQUIRE(top.top_k == 8);
    REQUIRE(top.hc_count == 4);
    REQUIRE(top.q_lora == 1536);
    REQUIRE(top.kv_lora == 512);
    REQUIRE(top.moe_intermediate == 2048);
    REQUIRE(top.dense_intermediate == 12288);
    REQUIRE(top.routed_scaling == 2.5);
    REQUIRE(top.eos_ids ==
            (std::vector<int>{154820, 154827, 154829}));
    REQUIRE(top.pad_id == 154820);
}

TEST_CASE("RequantFp8ToNvfp4Host: full-E4M3-scale NVFP4 convention parity")
{
    using lsxcommon::nvfp4::E4m3ToFloat;
    // 16x16 FP8 block: all 0x38 (E4M3 = 1.0: exp field 7, bias 7) with one
    // 128-grid scale 2.0. Dequantized source = 2.0 for every element.
    std::vector<uint8_t> fp8(256, 0x38);
    std::vector<float> scale_inv(1, 2.0f);  // [1,1] 128x128 grid
    auto blob = RequantFp8ToNvfp4Host(fp8.data(), scale_inv.data(), 16, 16);
    REQUIRE(blob.rows == 16);
    REQUIRE(blob.cols == 16);
    REQUIRE(blob.packed.size() == 128);  // 256 nibbles / 2
    REQUIRE(blob.scales.size() == 16);   // 256 / 16-element blocks
    // Per-block scale byte: pow2 ceil of (block_max / global_max).
    // block_max = global_max = 2.0 -> ratio 1.0 -> FloatToE4m3(1.0)
    // = 2^0 -> e4 field 7 -> byte 0x38.
    for (uint8_t s : blob.scales) REQUIRE(s == 0x38);
    // global_scale = global_max_abs / 6 == 2/6 == 1/3.
    REQUIRE(blob.global_scale == 1.0f / 3.0f);
    // Roundtrip: dequant(E2M1(nib) * e4m3_p2(s) * gs) must reproduce 2.0
    // (grid hit exact: 2.0 / (1.0 * 1/3) = 6.0 = E2M1 top level).
    for (uint8_t byte : blob.packed) {
        const uint8_t lo = byte & 0x0F, hi = (byte >> 4) & 0x0F;
        REQUIRE(lo == 0x07);  // magnitude 7 = 6.0, positive
        REQUIRE(hi == 0x07);
    }
    const float deq = 6.0f * E4m3ToFloat(0x38) * blob.global_scale;
    REQUIRE(std::fabs(deq - 2.0f) < 1e-6f);
}

TEST_CASE("RequantFp8ToNvfp4Host: mixed magnitudes + zero block")
{
    using lsxcommon::glm_moe_flash::FlashFp8E4m3Decode;
    // 2 rows x 16 cols; row 0: ramp values, row 1: all zeros.
    std::vector<uint8_t> fp8(32, 0x00);
    for (int c = 0; c < 16; ++c)
        fp8[c] = static_cast<uint8_t>(0x30 + c);  // varying magnitudes
    std::vector<float> scale_inv(1, 1.0f);
    auto blob = RequantFp8ToNvfp4Host(fp8.data(), scale_inv.data(), 2, 16);
    REQUIRE(blob.rows == 2);
    REQUIRE(blob.cols == 16);
    // Row 1 all-zero: quantized to zero nibbles (block scale encodes to
    // the reserved zero convention used by the GPU path: max=0 -> s
    // fallback, nibbles stay 0 either way).
    for (int i = 16; i < 32; ++i) {
        const uint8_t byte = blob.packed[i / 2];
        const uint8_t nib = (i % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
        REQUIRE(nib == 0x00);
    }
    // Row 0: values are powers-of-two multiples of 0.5 in E4M3 (0x30..0x3F
    // span 0.5..~57); requant must round-trip within one E2M1 step.
    // Verify no NaNs and global_scale positive finite.
    REQUIRE(std::isfinite(blob.global_scale));
    REQUIRE(blob.global_scale > 0.0f);
    // Roundtrip fidelity: |dequant - source| within one E2M1 step at the
    // block scale (relative tolerance 2/6 of the value). Local E2M1 decode
    // twin (magnitude LUT) — device twin lives in nvfp4_codec.h.
    auto e2m1_value = [](uint8_t nib) {
        static const float kMag[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
        const float v = kMag[nib & 7];
        return (nib & 8) ? -v : v;
    };
    for (int i = 0; i < 16; ++i) {
        const float src = FlashFp8E4m3Decode(fp8[i]);
        const uint8_t byte = blob.packed[i / 2];
        const uint8_t nib = (i % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
        const int blk = i / 16;
        const float s = lsxcommon::nvfp4::E4m3ToFloat(blob.scales[blk])
                      * blob.global_scale;
        const float deq = e2m1_value(nib) * s;
        const float tol = std::max(std::fabs(src) * (2.0f / 6.0f), 1e-6f);
        INFO("i=" << i << " src=" << src << " deq=" << deq);
        REQUIRE(std::fabs(deq - src) <= tol);
    }
}

// ── Task 3: converter pipeline ───────────────────────────────────────────
#include "lsxcommon/glm_moe_flash_convert.h"

TEST_CASE("DequantFp8TilesToBf16Gpu: parity vs host dequant")
{
    // 128 rows x 128 cols FP8 + 1x1 F32 tile scale (one 128x128 tile).
    // The kernel requires exact 128-multiple geometry (matches the
    // checkpoint: e.g. [2048, 4096] weights with [16, 32] scale tiles).
    constexpr int R = 128, C = 128;
    std::vector<uint8_t> fp8(static_cast<size_t>(R) * C);
    for (size_t i = 0; i < fp8.size(); ++i)
        fp8[i] = static_cast<uint8_t>(0x08 + (i % 0x70));  // varied values
    const float scale = 1.5f;
    std::vector<uint8_t> scale_bytes(reinterpret_cast<const uint8_t *>(&scale),
                                     reinterpret_cast<const uint8_t *>(&scale)
                                         + 4);
    std::vector<uint8_t> bf16;
    auto st = lsxcommon::glm_moe_flash::DequantFp8TilesToBf16Gpu(
        fp8, {R, C}, scale_bytes, {1, 1}, bf16);
    INFO("dequant status: " << st.ToString());
    REQUIRE(st.ok());
    REQUIRE(bf16.size() == static_cast<size_t>(R) * C * 2);
    // Host reference on a sample of elements.
    // BF16 bit pattern -> F32: a BF16 value is the top 16 bits of the
    // F32 bit pattern, so widening is a 16-bit left shift.
    auto bf16_to_float = [](const uint8_t *p) {
        const uint32_t bits = (static_cast<uint32_t>(p[0])
                               | (static_cast<uint32_t>(p[1]) << 8))
                              << 16;
        float v;
        std::memcpy(&v, &bits, 4);
        return v;
    };
    for (int i = 0; i < R * C; i += 997) {
        const float src =
            lsxcommon::glm_moe_flash::FlashFp8E4m3Decode(fp8[i]) * scale;
        const float got =
            bf16_to_float(bf16.data() + static_cast<size_t>(i) * 2);
        // BF16 has 8 mantissa bits; fp8 value has at most 3 -> exact.
        INFO("i=" << i << " src=" << src << " got=" << got);
        REQUIRE(std::fabs(got - src) <= 1e-6f * std::max(1.0f, src));
    }
}

TEST_CASE("ConvertGlm5NextModel: synthetic tiny checkpoint — layout + index")
{
    // Converter tests are unified on synthetic fixtures: no real-checkpoint
    // conversions in the suite (the e2e path used to convert the whole
    // GLM-5.3-Flash vault here — minutes of GPU work "as a test").
    // The checkpoint builder is the family fixture (test_glm_moe_flash_
    // fixture.h) — the loader suite converts the SAME fixture, so converter
    // and loader always agree on the produced layout.
    glmflash::ConvertedFlashFixture fix;
    fix.in = fs::temp_directory_path() / "glm53flash_conv_synth_in";
    fix.out = fs::temp_directory_path() / "glm53flash_conv_synth_out";
    fs::remove_all(fix.in);
    fs::remove_all(fix.out);
    auto st = fix.Convert();
    INFO("convert status: " << st.ToString());
    REQUIRE(st.ok());

    // Vault layout: globals + per-layer raw/nvfp4 + per-expert arrows.
    const fs::path base = fix.out;
    REQUIRE(fs::exists(base / "model.index.json"));
    REQUIRE(fs::exists(base / "pinned" / "global.arrow"));
    REQUIRE(fs::exists(base / "pinned" / "layer_000.arrow"));
    REQUIRE(fs::exists(base / "pinned" / "layer_000_nvfp4.arrow"));
    REQUIRE(fs::exists(base / "pinned" / "layer_003.arrow"));
    REQUIRE(fs::exists(base / "pinned" / "layer_003_nvfp4.arrow"));
    // Frozen-range layers without tensors still flush empty raw batches.
    REQUIRE(fs::exists(base / "pinned" / "layer_045.arrow"));
    REQUIRE(fs::exists(base / "experts" / "layer_003" / "exp_000.arrow"));
    REQUIRE(fs::exists(base / "experts" / "layer_003" / "exp_001.arrow"));

    // model.index.json: format tag + topology lists (frozen-driven).
    // The stream must be closed before the cleanup below: Windows refuses to
    // delete an open file, while Linux unlinks it and hides the ordering bug.
    {
        std::ifstream f(base / "model.index.json");
        std::stringstream ss;
        ss << f.rdbuf();
        const std::string idx = ss.str();
        // A LAYOUT generation, not a per-model `format` tag (design §2.4.3).
        REQUIRE(idx.find("\"layout\": 2") != std::string::npos);
        // The retired field must not come back.
        REQUIRE(idx.find("\"format\"") == std::string::npos);
        REQUIRE(idx.find("\"hidden_size\": 64") != std::string::npos);
        REQUIRE(idx.find("\"num_experts\": 2") != std::string::npos);
        REQUIRE(idx.find("\"kda_layers\"") != std::string::npos);
        REQUIRE(idx.find("\"dsa_layers\"") != std::string::npos);
        REQUIRE(idx.find("\"moe_layers\": [3,") != std::string::npos);
    }

    fs::remove_all(fix.in);
    fs::remove_all(fix.out);
}

// ── Audit instrument: the real checkpoint (light files only) ─────────────
// The converter hard-errors on an unclassified tensor and silently ignores
// nothing, so the classifier's coverage of the *actual* inventory is the
// difference between a conversion failure and a model that lost a tensor.
// These cases read config.json + model.safetensors.index.json (never tensor
// data: the converter-suite discipline forbids real conversions) and SKIP on
// a machine whose vault does not have the checkpoint.
namespace {

fs::path GlmFlashSourceDir()
{
    return manifest_query::SourceDirFor("glm-5.3-flash");
}

} // namespace

TEST_CASE("real checkpoint: hybrid pattern + semantics match the passport")
{
    const fs::path dir = GlmFlashSourceDir();
    const std::string cfg = dir.string() + "/config.json";
    if (!fs::exists(cfg))
        SKIP(std::string("GLM-5.3-Flash checkpoint not present: ") + cfg);

    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    REQUIRE(parser.load(cfg.c_str()).get(doc) == simdjson::SUCCESS);
    simdjson::dom::object tc;
    REQUIRE(doc["text_config"].get_object().get(tc) == simdjson::SUCCESS);

    using namespace frozen;
    auto iv = [&](const char *k) {
        int64_t v = 0;
        REQUIRE(tc[k].get_int64().get(v) == simdjson::SUCCESS);
        return static_cast<int>(v);
    };
    auto dv = [&](const char *k) {
        double v = 0.0;
        REQUIRE(tc[k].get_double().get(v) == simdjson::SUCCESS);
        return v;
    };
    // Geometry the kernels are compiled against.
    CHECK(iv("hidden_size") == kHidden);
    CHECK(iv("num_hidden_layers") == kLayers);
    CHECK(iv("num_attention_heads") == kHeads);
    CHECK(iv("qk_head_dim") == kQkHeadDim);
    CHECK(iv("q_lora_rank") == kQLora);
    CHECK(iv("kv_lora_rank") == kKvLora);
    CHECK(iv("moe_intermediate_size") == kMoeIntermediate);
    CHECK(iv("intermediate_size") == kDenseIntermediate);
    CHECK(iv("n_routed_experts") == kNumExperts);
    CHECK(iv("num_experts_per_tok") == kTopK);
    CHECK(iv("hc_mult") == kHcCount);
    CHECK(iv("vocab_size") == kVocab);
    CHECK(iv("first_k_dense_replace") == kFirstKDense);
    CHECK(iv("num_nextn_predict_layers") == 1);

    // Hybrid pattern: the one item that silently decodes garbage if a
    // single layer_types assumption is used instead.
    simdjson::dom::object lac;
    REQUIRE(tc["linear_attn_config"].get_object().get(lac) == simdjson::SUCCESS);
    simdjson::dom::array kda_arr, full_arr;
    REQUIRE(lac["kda_layers"].get_array().get(kda_arr) == simdjson::SUCCESS);
    REQUIRE(lac["full_attn_layers"].get_array().get(full_arr) ==
            simdjson::SUCCESS);
    std::vector<int> got_kda, got_full;
    for (auto e : kda_arr) {
        int64_t v = 0;
        REQUIRE(e.get_int64().get(v) == simdjson::SUCCESS);
        got_kda.push_back(static_cast<int>(v));
    }
    for (auto e : full_arr) {
        int64_t v = 0;
        REQUIRE(e.get_int64().get(v) == simdjson::SUCCESS);
        got_full.push_back(static_cast<int>(v));
    }
    CHECK(got_kda == std::vector<int>(std::begin(kKdaLayerIds),
                                      std::end(kKdaLayerIds)));
    CHECK(got_full == std::vector<int>(std::begin(kDsaLayerIds),
                                       std::end(kDsaLayerIds)));
    {
        int64_t conv = 0;
        REQUIRE(lac["short_conv_kernel_size"].get_int64().get(conv) ==
                simdjson::SUCCESS);
        CHECK(static_cast<int>(conv) == kConvKernel);
        int64_t heads = 0;
        REQUIRE(lac["num_heads"].get_int64().get(heads) == simdjson::SUCCESS);
        CHECK(static_cast<int>(heads) == kHeads);
        int64_t hd = 0;
        REQUIRE(lac["head_dim"].get_int64().get(hd) == simdjson::SUCCESS);
        CHECK(static_cast<int>(hd) == kHeadDim);
    }

    // Semantics that the kernels hard-code (a mismatch here is not a
    // performance question: it is a wrong-answer question).
    CHECK(dv("swiglu_limit") == 10.0);          // clamped SwiGLU
    CHECK(dv("routed_scaling_factor") == 2.5);  // kRoutedScalingNum/Den
    CHECK(dv("hc_eps") == 1e-6);
    CHECK(iv("hc_sinkhorn_iters") == 20);
    CHECK(iv("index_head_dim") == kIndexerHeadDim);
    CHECK(iv("index_n_heads") == kIndexerHeads);
    CHECK(iv("index_topk") == kIndexerTopk);
    CHECK(iv("index_kpool") == kIndexerKpool);
    CHECK(iv("n_shared_experts") == 1);
    CHECK(tc["mla_use_nope"].get_bool().value());
    CHECK(tc["norm_topk_prob"].get_bool().value());
    {
        std::string_view scoring, method;
        REQUIRE(tc["scoring_func"].get_string().get(scoring) ==
                simdjson::SUCCESS);
        REQUIRE(tc["topk_method"].get_string().get(method) ==
                simdjson::SUCCESS);
        CHECK(scoring == "sigmoid");
        CHECK(method == "noaux_tc");
    }

    // ParseFlashConfig must agree with the passport it is gated against.
    auto top = ParseFlashConfig(dir.string());
    REQUIRE(top.error_msg.empty());
    CHECK(top.qk_head_dim == kQkHeadDim);
    CHECK(top.indexer_topk == kIndexerTopk);
}

TEST_CASE("real checkpoint: every tensor in the index is classified")
{
    const fs::path dir = GlmFlashSourceDir();
    const std::string index_path = dir.string() + "/model.safetensors.index.json";
    if (!fs::exists(index_path))
        SKIP(std::string("GLM-5.3-Flash checkpoint not present: ") + index_path);

    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    REQUIRE(parser.load(index_path.c_str()).get(doc) == simdjson::SUCCESS);
    simdjson::dom::object weight_map;
    REQUIRE(doc["weight_map"].get_object().get(weight_map) == simdjson::SUCCESS);

    std::set<std::string> names;      // classified payload tensors
    std::vector<std::string> unknown;
    size_t scale_inv = 0;
    std::map<int, std::set<int>> moe_experts;
    std::map<int, std::map<int, std::array<bool, 4>>> expert_projs;
    std::map<int, std::set<std::string>> layer_suffixes;  // non-expert names

    for (auto [key, shard] : weight_map) {
        const std::string name(key);
        (void)shard;
        if (name.find(".weight_scale_inv") != std::string::npos) {
            ++scale_inv;
            continue;
        }
        const ParsedFlashTensor p = classify_flash_tensor(name);
        names.insert(name);
        if (p.kind == FlashTensorKind::Unknown) {
            unknown.push_back(name);
            continue;
        }
        if (p.kind == FlashTensorKind::RoutedExpert) {
            // Only MoE layers own routed experts, each expert exactly once
            // per projection.
            CHECK(frozen::IsMoeLayer(p.layer));
            moe_experts[p.layer].insert(p.expert);
            expert_projs[p.layer][p.expert][static_cast<size_t>(p.proj_id)] =
                true;
        } else if (p.layer >= 0) {
            const std::string prefix = ".layers." + std::to_string(p.layer) + ".";
            const size_t at = name.find(prefix);
            layer_suffixes[p.layer].insert(
                at == std::string::npos ? name : name.substr(at + prefix.size()));
        }
    }

    INFO("unclassified: " << (unknown.empty() ? std::string("none")
                                              : unknown.front()));
    CHECK(unknown.empty());
    CHECK(scale_inv == 37338);  // FP8 scale payloads, consumed as pairs
    CHECK(names.size() == 38770);

    // Routed-expert completeness: 288 experts x 3 projections on every MoE
    // layer, and no expert tensor anywhere else.
    CHECK(moe_experts.size() == static_cast<size_t>(frozen::kLayers -
                                                    frozen::kFirstKDense + 1));
    for (const auto &[layer, experts] : moe_experts) {
        INFO("layer " << layer);
        CHECK(experts.size() == static_cast<size_t>(frozen::kNumExperts));
        CHECK(*experts.begin() == 0);
        CHECK(*experts.rbegin() == frozen::kNumExperts - 1);
        size_t missing = 0;
        for (const auto &[e, projs] : expert_projs[layer]) {
            for (int pid = 1; pid <= 3; ++pid)
                if (!projs[static_cast<size_t>(pid)]) ++missing;
        }
        CHECK(missing == 0);
    }

    // Per-layer structure. Layer tables come from the checkpoint itself,
    // not from the passport, so a passport/config drift fails here.
    static const std::array<std::pair<const char *, FlashTensorKind>, 14>
        kKdaSuffix = {{
            {"self_attn.q_proj.weight", FlashTensorKind::KdaQ},
            {"self_attn.k_proj.weight", FlashTensorKind::KdaK},
            {"self_attn.v_proj.weight", FlashTensorKind::KdaV},
            {"self_attn.o_proj.weight", FlashTensorKind::KdaO},
            {"self_attn.q_conv1d.weight", FlashTensorKind::KdaConv},
            {"self_attn.k_conv1d.weight", FlashTensorKind::KdaConv},
            {"self_attn.v_conv1d.weight", FlashTensorKind::KdaConv},
            {"self_attn.A_log", FlashTensorKind::KdaALog},
            {"self_attn.dt_bias", FlashTensorKind::KdaDtBias},
            {"self_attn.b_proj.weight", FlashTensorKind::KdaB},
            {"self_attn.f_a_proj.weight", FlashTensorKind::KdaFa},
            {"self_attn.f_b_proj.weight", FlashTensorKind::KdaFb},
            {"self_attn.g_a_proj.weight", FlashTensorKind::KdaGa},
            {"self_attn.g_b_proj.weight", FlashTensorKind::KdaGb},
        }};
    static const std::array<std::pair<const char *, FlashTensorKind>, 14>
        kDsaSuffix = {{
            {"self_attn.q_a_proj.weight", FlashTensorKind::DsaQa},
            {"self_attn.q_b_proj.weight", FlashTensorKind::DsaQb},
            {"self_attn.kv_a_proj_with_mqa.weight", FlashTensorKind::DsaKvA},
            {"self_attn.kv_b_proj.weight", FlashTensorKind::DsaKvB},
            {"self_attn.o_proj.weight", FlashTensorKind::DsaOBig},
            {"self_attn.q_a_layernorm.weight", FlashTensorKind::DsaNorms},
            {"self_attn.kv_a_layernorm.weight", FlashTensorKind::DsaNorms},
            {"self_attn.indexer.wq_b.weight", FlashTensorKind::Indexer},
            {"self_attn.indexer.wk.weight", FlashTensorKind::Indexer},
            {"self_attn.indexer.weights_proj.weight", FlashTensorKind::Indexer},
            {"self_attn.indexer.k_norm.weight", FlashTensorKind::Indexer},
            {"self_attn.indexer.k_norm.bias", FlashTensorKind::Indexer},
            {"self_attn.indexer.index_kpool_compress_ape",
             FlashTensorKind::Indexer},
            {"self_attn.indexer.index_kpool_compress_gate",
             FlashTensorKind::Indexer},
        }};

    const std::string lprefix = "model.language_model.layers.";
    auto check_layer = [&](int layer, const auto &suffixes,
                           const char *what) {
        for (const auto &[suffix, want] : suffixes) {
            const std::string full =
                lprefix + std::to_string(layer) + "." + suffix;
            INFO(what << " layer " << layer << ": " << suffix);
            CHECK(names.count(full) == 1);
            CHECK(classify_flash_tensor(full).kind == want);
        }
    };

    for (size_t i = 0; i < frozen::kKdaLayerCount; ++i) {
        const int layer = frozen::kKdaLayerIds[i];
        check_layer(layer, kKdaSuffix, "KDA");
        INFO("KDA layer " << layer);
        CHECK(layer_suffixes[layer].count("self_attn.o_norm.weight") == 1);
        CHECK(frozen::IsKdaLayer(layer));
    }
    for (size_t i = 0; i < frozen::kDsaLayerCount; ++i) {
        const int layer = frozen::kDsaLayerIds[i];
        check_layer(layer, kDsaSuffix, "DSA");
        CHECK(frozen::IsDsaLayer(layer));
    }
    // MTP nextn layer: a full DSA + MoE layer, but the checkpoint carries no
    // hyper-connection tensors for it (mHC starts at layer 0 and covers the
    // 45 resident layers only). It is stored and never applied, and the
    // loader builds layers [0, num_layers) == [0, 45).
    check_layer(frozen::kMtpLayer, kDsaSuffix, "MTP");
    for (int l = 0; l < frozen::kLayers; ++l) {
        const std::string prefix = lprefix + std::to_string(l) + ".";
        INFO("layer " << l);
        CHECK(names.count(prefix + "input_layernorm.weight") == 1);
        CHECK(names.count(prefix + "post_attention_layernorm.weight") == 1);
        for (const char *hc : {"hc_attn_base", "hc_attn_fn", "hc_attn_scale",
                               "hc_ffn_base", "hc_ffn_fn", "hc_ffn_scale"})
            CHECK(names.count(prefix + hc) == 1);
    }
    CHECK(names.count(lprefix + std::to_string(frozen::kMtpLayer) +
                      ".hc_attn_fn") == 0);

    // Globals + multimodal ballast (stripped, counted by the converter).
    CHECK(classify_flash_tensor("lm_head.weight").kind ==
          FlashTensorKind::LmHead);
    CHECK(classify_flash_tensor("model.language_model.embed_tokens.weight")
              .kind == FlashTensorKind::Embed);
    CHECK(classify_flash_tensor("model.language_model.norm.weight").kind ==
          FlashTensorKind::FinalNorm);
    size_t visual = 0;
    for (const auto &n : names) {
        const auto p = classify_flash_tensor(n);
        if (p.kind != FlashTensorKind::Visual) continue;
        ++visual;
        CHECK(p.layer == -1);
    }
    CHECK(visual == 347);  // 24 vision blocks + merger/patch_embed
}
