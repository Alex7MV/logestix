#pragma once

// Mistral-Small-4-119B synthetic checkpoint fixture (Model Ingestion Protocol
// step 1). Family-owned: includes only the family-neutral
// test_synth_checkpoint.h primitives.

#include "test_synth_checkpoint.h"

#include <absl/strings/str_cat.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace mistraltest {

using synthtest::FakeTensor;

// One byte per element (U8 packed E2M1 nibbles). Every nibble pattern is a
// valid E2M1 value, so arbitrary bytes are safe here.
inline std::vector<uint8_t> MakeRaw(int64_t n)
{
    std::vector<uint8_t> out(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        out[static_cast<size_t>(i)] = static_cast<uint8_t>((i * 7 + 1) & 0xff);
    return out;
}

// F8_E4M3 block scales. MUST be finite: 0x7F is the E4M3 NaN and would poison
// the whole dequant (NaN * global_scale is still NaN). 0x38..0x3F is a stable
// run of finite E4M3 values (2^0 near 1.0).
inline std::vector<uint8_t> MakeScales(int64_t n)
{
    std::vector<uint8_t> out(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        out[static_cast<size_t>(i)] =
            static_cast<uint8_t>(0x38 + (i & 0x7));
    return out;
}

// BF16 weights with NORMAL magnitudes (the raw MakeBf16 bit pattern is a
// denormal that underflows to zero once converted to FP16).
inline std::vector<uint8_t> MakeBf16Small(int64_t n)
{
    return synthtest::MakeBf16Values(
        n, [](int64_t i) { return 0.05f * static_cast<float>((i % 13) - 6); });
}

inline std::vector<int64_t> Sh(int64_t a, int64_t b) { return {a, b}; }
inline std::vector<int64_t> Sh1(int64_t a) { return {a}; }

// Small synthetic geometry used by the converter/loader tests. It does NOT
// match the frozen passport; LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1 relaxes the
// gate (set as a RUNTIME env var by tests/CMakeLists.txt).
struct MistralSynthGeom {
    // hidden and expert_hidden are multiples of 16 so the NVFP4 per-16 block
    // scales are non-empty (a 0-block tensor is a malformed fixture, not an
    // architecture).
    // hidden and expert_hidden are multiples of 32 so the neutral NVFP4 MoE
    // GEMV path (K % 32) is eligible; all widths are multiples of 16 for the
    // NVFP4 per-16 block scales.
    int64_t hidden = 64;
    int64_t layers = 2;
    int64_t heads = 2;
    int64_t q_lora = 16;
    int64_t kv_lora = 16;
    int64_t qk_nope = 8;
    int64_t qk_rope = 8;
    int64_t v_head = 8;
    int64_t expert_hidden = 32;
    int64_t experts = 2;
    int64_t vocab = 64;
};

inline void WriteMistralParams(const std::filesystem::path &dir,
                               const MistralSynthGeom &g = {})
{
    std::ofstream f(dir / "params.json");
    f << "{\n";
    f << "  \"dim\": " << g.hidden << ",\n";
    f << "  \"head_dim\": 128,\n";
    f << "  \"hidden_dim\": 12288,\n";
    f << "  \"kv_lora_rank\": " << g.kv_lora << ",\n";
    f << "  \"max_position_embeddings\": 1048576,\n";
    f << "  \"moe\": {\"expert_hidden_dim\": " << g.expert_hidden
      << ", \"num_experts\": " << g.experts << ", \"num_experts_per_tok\": 1,"
      << " \"num_shared_experts\": 1, \"first_k_dense_replace\": 0},\n";
    f << "  \"n_heads\": " << g.heads << ",\n";
    f << "  \"n_kv_heads\": " << g.heads << ",\n";
    f << "  \"n_layers\": " << g.layers << ",\n";
    f << "  \"norm_eps\": 1e-06,\n";
    f << "  \"q_lora_rank\": " << g.q_lora << ",\n";
    f << "  \"qk_nope_head_dim\": " << g.qk_nope << ",\n";
    f << "  \"qk_rope_head_dim\": " << g.qk_rope << ",\n";
    f << "  \"rope_theta\": 10000.0,\n";
    f << "  \"tied_embeddings\": false,\n";
    f << "  \"v_head_dim\": " << g.v_head << ",\n";
    f << "  \"vocab_size\": " << g.vocab << "\n";
    f << "}\n";
}

// One already-quantized expert/shared projection family (w1/w2/w3) for a layer.
inline void AddExpertTensors(std::vector<FakeTensor> &out,
                             const std::string &prefix,
                             const MistralSynthGeom &g, int64_t w1_rows)
{
    // w1/w3: [expert_hidden, hidden]; w2: [hidden, expert_hidden].
    const int64_t I = g.expert_hidden;
    const int64_t H = g.hidden;
    const int64_t w1_bytes = I * (H / 2);
    const int64_t w2_bytes = H * (I / 2);
    const int64_t w1_scales = I * (H / 16);
    const int64_t w2_scales = H * (I / 16);
    (void)w1_rows;
    struct Spec {
        const char *name;
        int64_t packed;
        int64_t packed_rows;
        int64_t cols_half;
        int64_t scales;
        int64_t scale_rows;
        int64_t scale_cols;
    };
    const Spec specs[3] = {
        {"w1", w1_bytes, I, H / 2, w1_scales, I, H / 16},
        {"w2", w2_bytes, H, I / 2, w2_scales, H, I / 16},
        {"w3", w1_bytes, I, H / 2, w1_scales, I, H / 16},
    };
    for (const auto &s : specs)
    {
        out.push_back(FakeTensor{absl::StrCat(prefix, ".", s.name, ".weight_packed"),
                                 "U8", Sh(s.packed_rows, s.cols_half),
                                 MakeRaw(s.packed)});
        out.push_back(FakeTensor{absl::StrCat(prefix, ".", s.name, ".weight_scale"),
                                 "F8_E4M3", Sh(s.scale_rows, s.scale_cols),
                                 MakeScales(s.scales)});
        out.push_back(FakeTensor{
            absl::StrCat(prefix, ".", s.name, ".weight_global_scale"), "F32",
            Sh1(1), synthtest::MakeF32Values(1, [](int64_t) { return 1.0f; })});
        out.push_back(FakeTensor{
            absl::StrCat(prefix, ".", s.name, ".input_global_scale"), "F32",
            Sh1(1), synthtest::MakeF32Values(1, [](int64_t) { return 1.0f; })});
    }
}

inline std::vector<FakeTensor> BuildSyntheticMistralModel(
    const MistralSynthGeom &g = {})
{
    std::vector<FakeTensor> out;
    const int64_t H = g.hidden;
    const int64_t qk = g.qk_nope + g.qk_rope;

    // Globals.
    out.push_back(FakeTensor{"tok_embeddings.weight", "BF16", Sh(g.vocab, H),
                             MakeBf16Small(g.vocab * H)});
    out.push_back(FakeTensor{"output.weight", "BF16", Sh(g.vocab, H),
                             MakeBf16Small(g.vocab * H)});
    out.push_back(FakeTensor{"norm.weight", "BF16", Sh1(H),
                             MakeBf16Small(H)});

    for (int64_t l = 0; l < g.layers; ++l)
    {
        const std::string p = absl::StrCat("layers.", l, ".");
        out.push_back(FakeTensor{p + "attention.wq_a.weight", "BF16",
                                 Sh(g.q_lora, H),
                                 MakeBf16Small(g.q_lora * H)});
        out.push_back(FakeTensor{p + "attention.q_a_norm.weight", "BF16",
                                 Sh1(g.q_lora),
                                 MakeBf16Small(g.q_lora)});
        out.push_back(FakeTensor{p + "attention.wq_b.weight", "BF16",
                                 Sh(g.heads * qk, g.q_lora),
                                 MakeBf16Small(g.heads * qk * g.q_lora)});
        out.push_back(FakeTensor{p + "attention.wkv_a_with_mqa.weight", "BF16",
                                 Sh(g.kv_lora + g.qk_rope, H),
                                 MakeBf16Small((g.kv_lora + g.qk_rope) * H)});
        out.push_back(FakeTensor{p + "attention.kv_a_norm.weight", "BF16",
                                 Sh1(g.kv_lora),
                                 MakeBf16Small(g.kv_lora)});
        out.push_back(FakeTensor{p + "attention.wkv_b.weight", "BF16",
                                 Sh(g.heads * (g.qk_nope + g.v_head), g.kv_lora),
                                 MakeBf16Small(g.heads *
                                                     (g.qk_nope + g.v_head) *
                                                     g.kv_lora)});
        out.push_back(FakeTensor{p + "attention.wo.weight", "BF16",
                                 Sh(H, g.heads * g.v_head),
                                 MakeBf16Small(H * g.heads * g.v_head)});
        out.push_back(FakeTensor{p + "attention_norm.weight", "BF16", Sh1(H),
                                 MakeBf16Small(H)});
        out.push_back(FakeTensor{p + "ffn_norm.weight", "BF16", Sh1(H),
                                 MakeBf16Small(H)});
        out.push_back(FakeTensor{p + "gate.weight", "BF16", Sh(g.experts, H),
                                 MakeBf16Small(g.experts * H)});
        for (int64_t e = 0; e < g.experts; ++e)
            AddExpertTensors(out,
                             absl::StrCat(p, "experts.", e), g, g.expert_hidden);
        AddExpertTensors(out, p + "shared_experts", g, g.expert_hidden);
    }

    // Vision ballast (stripped + counted).
    out.push_back(FakeTensor{"vision_encoder.patch_conv.weight", "BF16",
                             Sh(4, 4), MakeBf16Small(16)});
    out.push_back(FakeTensor{"vision_language_adapter.w_in.weight", "BF16",
                             Sh(4, 4), MakeBf16Small(16)});
    out.push_back(FakeTensor{"patch_merger.merging_layer.weight", "BF16",
                             Sh(4, 4), MakeBf16Small(16)});
    out.push_back(FakeTensor{"pre_mm_projector_norm.weight", "BF16", Sh1(4),
                             MakeBf16Small(4)});
    return out;
}

} // namespace mistraltest
