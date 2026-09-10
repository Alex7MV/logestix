// GLM-5.3-Flash converter suite: frozen geometry, tensor classifier,
// FP8->NVFP4 requant, config parser, conversion pipeline.
// The end-to-end converter test runs on a synthetic tiny checkpoint
// (unified converter-suite discipline: no real-model conversions).
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

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
#include "lsxnetwork/lsx_token_translator.h"

#include <cmath>

TEST_CASE("ParseFlashConfig: real checkpoint text_config")
{
    const std::string dir = "/mnt/llm_models/models/GLM-5.3-Flash";
    if (!fs::exists(dir + "/config.json")) {
        SKIP("GLM-5.3-Flash checkpoint not present");
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

TEST_CASE("RequantFp8ToNvfp4Host: pow2-scale NVFP4 convention parity")
{
    using lsxcommon::nvfp4::E4m3P2ToFloat;
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
    // block_max = global_max = 2.0 -> ratio 1.0 -> FloatToE4m3P2(1.0)
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
    const float deq = 6.0f * E4m3P2ToFloat(0x38) * blob.global_scale;
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
        const float s = lsxcommon::nvfp4::E4m3P2ToFloat(blob.scales[blk])
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
    const fs::path base = fix.out / "logestix";
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
    std::ifstream f(base / "model.index.json");
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string idx = ss.str();
    REQUIRE(idx.find("logestix-glm5next") != std::string::npos);
    REQUIRE(idx.find("\"hidden_size\": 64") != std::string::npos);
    REQUIRE(idx.find("\"num_experts\": 2") != std::string::npos);
    REQUIRE(idx.find("\"kda_layers\"") != std::string::npos);
    REQUIRE(idx.find("\"dsa_layers\"") != std::string::npos);
    REQUIRE(idx.find("\"moe_layers\": [3,") != std::string::npos);

    fs::remove_all(fix.in);
    fs::remove_all(fix.out);
}

TEST_CASE("BuildFlashPassportTokens: byte-0 unk anchors + DS bridge side")
{
    namespace fs = std::filesystem;
    // Real converted flash tokenizer when present; the DS side depends on
    // the DeepSeek-V4-Flash vault vocabulary, so this test exercises both
    // the complete-bridge and glm-only outcomes deterministically.
    const char *env = std::getenv("LSX_TEST_GLM_FLASH_ROOT");
    if (!env) SKIP("GLM-5.3-Flash converted model not present");
    const fs::path pinned = fs::path(env) / "logestix" / "pinned";
    if (!fs::exists(pinned / "tokenizer_vocab.arrow"))
        SKIP("flash tokenizer arrows not present");

    const auto pt = lsxcommon::glm_moe_flash::BuildFlashPassportTokens(pinned);
    REQUIRE(pt.glm_tokens.size() >= 154000);
    // Byte-0 anchor exists in a byte-level BPE vocabulary and must be found
    // as the unk id (exactly one '\0' single-byte token).
    REQUIRE(pt.glm_tokens[pt.glm_unk_id] == std::string("\0", 1));
    if (pt.deepseek_tokens.empty()) {
        SKIP("DeepSeek-V4-Flash vault vocabulary not present");
    }
    REQUIRE(pt.deepseek_tokens.size() >= 128000);
    REQUIRE(pt.deepseek_tokens[pt.deepseek_unk_id] == std::string("\0", 1));

    // Bridge smoke through the real passport codec: a handful of shared
    // byte tokens must translate exactly both ways.
    lsxnetwork::BuildPassportOptions opts;
    opts.glm_tokens = pt.glm_tokens;
    opts.deepseek_tokens = pt.deepseek_tokens;
    opts.glm_unk_id = pt.glm_unk_id;
    opts.deepseek_unk_id = pt.deepseek_unk_id;
    const auto bytes = lsxnetwork::BuildPassport(opts);
    REQUIRE(bytes.size() > 64);
    lsxnetwork::token_passport::PassportHeader hdr;
    std::memcpy(&hdr, bytes.data(), sizeof(hdr));
    REQUIRE(hdr.magic == lsxnetwork::token_passport::kMagic);
    REQUIRE(hdr.glm_vocab_size == pt.glm_tokens.size());
    REQUIRE(hdr.deepseek_vocab_size == pt.deepseek_tokens.size());
    REQUIRE(hdr.glm_unk_id == pt.glm_unk_id);
    REQUIRE(hdr.deepseek_unk_id == pt.deepseek_unk_id);
}
