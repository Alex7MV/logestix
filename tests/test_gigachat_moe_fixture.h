#pragma once

// GigaChat3.5 synthetic checkpoint fixture. Family-owned: re-exports the
// family-neutral primitives of test_synth_checkpoint.h under `gigachattest::`
// and builds a tiny hybrid (MLA + GatedDeltaNet + MoE + MTP) checkpoint that
// exercises every converter path without touching the real vault.
//
// Frozen gates are bypassed in converter tests via
// LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1 (runtime ENV, never a compile define).

#include "test_synth_checkpoint.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace gigachattest {

using synthtest::FakeTensor;
using synthtest::MakeBf16;
using synthtest::WriteSafetensors;

// Tiny synthetic geometry.
inline constexpr int kTH = 128;        // hidden
inline constexpr int kTVocab = 128;
inline constexpr int kTLayers = 2;     // layer 0 GDN-dense, layer 1 MLA-MoE
inline constexpr int kTFullAttn = 1;   // layer 1 is full attention
inline constexpr int kTExperts = 2;
inline constexpr int kTMoeI = 256;
inline constexpr int kTDenseI = 256;
inline constexpr int kTGateRank = 16;

// GatedDeltaNet.
inline constexpr int kTGdnQkvz = 768;   // 2*128 + 2*256
inline constexpr int kTGdnValue = 256;  // 2 * 128
inline constexpr int kTGdnConv = 512;   // 2*128 + 256
inline constexpr int kTGdnBa = 4;       // 2 * 2 value heads

// MLA.
inline constexpr int kTHeads = 2;
inline constexpr int kTQk = 192;        // nope 128 + rope 64
inline constexpr int kTQkOut = 384;     // heads * qk
inline constexpr int kTKvA = 192;       // kv_lora 128 + rope 64 (partial tile)
inline constexpr int kTKvB = 512;       // heads * (128 + 128)
inline constexpr int kVOut = 256;       // heads * v_head_dim

inline std::filesystem::path MakeTempDir(const std::string &tag)
{
    static std::atomic<unsigned> counter{0};
    const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
    auto p = std::filesystem::temp_directory_path() /
             (tag + "_" + std::to_string(ticks) + "_" +
              std::to_string(counter.fetch_add(1)));
    std::filesystem::remove_all(p);
    std::filesystem::create_directories(p);
    return p;
}

// Valid E4M3 byte pattern (exponent nibble 1, sign 0).
inline std::vector<uint8_t> MakeFp8(int64_t n)
{
    std::vector<uint8_t> out(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        out[static_cast<size_t>(i)] =
            static_cast<uint8_t>((i % 15) | 0x08);
    return out;
}

inline std::vector<uint8_t> MakeOnesF32(int64_t n)
{
    return synthtest::MakeF32Values(
        n, [](int64_t) { return 1.0f; });
}

inline void AddFp8(std::vector<FakeTensor> &ts, const std::string &base,
                   int64_t rows, int64_t cols, bool scale_inv = true)
{
    ts.push_back(FakeTensor{base + ".weight", "F8_E4M3", {rows, cols},
                            MakeFp8(rows * cols)});
    if (scale_inv)
    {
        const int64_t sr = (rows + 127) / 128, sc = (cols + 127) / 128;
        ts.push_back(FakeTensor{base + ".weight_scale_inv", "F32", {sr, sc},
                                MakeOnesF32(sr * sc)});
    }
}

inline void AddBf16(std::vector<FakeTensor> &ts, const std::string &name,
                    std::vector<int64_t> shape)
{
    int64_t n = 1;
    for (auto d : shape) n *= d;
    ts.push_back(FakeTensor{name, "BF16", std::move(shape), MakeBf16(n)});
}

// ZeroCenteredGatedNorm triple (weight + gate_up + gate_down).
inline void AddGatedNorm(std::vector<FakeTensor> &ts, const std::string &base,
                         int64_t dim)
{
    AddBf16(ts, base + ".weight", {dim});
    AddBf16(ts, base + ".gate_up_projection.weight", {kTGateRank, dim});
    AddBf16(ts, base + ".gate_down_projection.weight", {dim, kTGateRank});
}

inline void WriteGigaChatCheckpoint(const std::filesystem::path &dir)
{
    {
        std::ofstream f(dir / "config.json");
        f << R"({
  "architectures": ["GigaChat35ForCausalLM"],
  "model_type": "gigachat3_5",
  "hidden_size": 128,
  "num_hidden_layers": 2,
  "vocab_size": 128,
  "num_attention_heads": 2,
  "num_key_value_heads": 2,
  "q_lora_rank": 128,
  "kv_lora_rank": 128,
  "qk_nope_head_dim": 128,
  "qk_rope_head_dim": 64,
  "v_head_dim": 128,
  "n_routed_experts": 2,
  "num_experts_per_tok": 1,
  "moe_intermediate_size": 256,
  "intermediate_size": 256,
  "first_k_dense_replace": 1,
  "rms_norm_eps": 1e-06,
  "routed_scaling_factor": 2.5,
  "swiglu_limit": 10,
  "linear_attention_type": "GigaChat35GatedDeltaNet",
  "full_attention_layers": [1],
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 2
})";
    }

    std::vector<FakeTensor> ts;

    // Globals.
    AddBf16(ts, "model.embed_tokens.weight", {kTVocab, kTH});
    AddBf16(ts, "lm_head.weight", {kTVocab, kTH});
    AddGatedNorm(ts, "model.norm", kTH);

    // Layer 0: GatedDeltaNet + dense MLP.
    for (const char *n : {"input_layernorm", "post_attention_layernorm",
                          "post_self_attn_layernorm",
                          "post_feedforward_layernorm"})
        AddGatedNorm(ts, std::string("model.layers.0.") + n, kTH);
    AddFp8(ts, "model.layers.0.self_attn.in_proj_qkvz", kTGdnQkvz, kTH);
    AddFp8(ts, "model.layers.0.self_attn.out_proj", kTH, kTGdnValue);
    AddBf16(ts, "model.layers.0.self_attn.conv1d.weight", {kTGdnConv, 1, 4});
    AddBf16(ts, "model.layers.0.self_attn.A_log", {2});
    AddBf16(ts, "model.layers.0.self_attn.dt_bias", {2});
    AddBf16(ts, "model.layers.0.self_attn.in_proj_ba.weight", {kTGdnBa, kTH});
    AddBf16(ts, "model.layers.0.self_attn.norm.weight", {128});
    AddFp8(ts, "model.layers.0.mlp.gate_proj", kTDenseI, kTH);
    AddFp8(ts, "model.layers.0.mlp.up_proj", kTDenseI, kTH);
    AddFp8(ts, "model.layers.0.mlp.down_proj", kTH, kTDenseI);

    // Layer 1: MLA + MoE.
    for (const char *n : {"input_layernorm", "post_attention_layernorm",
                          "post_self_attn_layernorm",
                          "post_feedforward_layernorm"})
        AddGatedNorm(ts, std::string("model.layers.1.") + n, kTH);
    AddFp8(ts, "model.layers.1.self_attn.q_a_proj", 128, kTH);
    AddGatedNorm(ts, "model.layers.1.self_attn.q_a_layernorm", 128);
    AddFp8(ts, "model.layers.1.self_attn.q_b_proj", kTQkOut, 128);
    AddFp8(ts, "model.layers.1.self_attn.kv_a_proj_with_mqa", kTKvA, kTH);
    AddGatedNorm(ts, "model.layers.1.self_attn.kv_a_layernorm", 128);
    AddFp8(ts, "model.layers.1.self_attn.kv_b_proj", kTKvB, 128);
    AddFp8(ts, "model.layers.1.self_attn.o_proj", kTH, kVOut);
    AddFp8(ts, "model.layers.1.self_attn.gate_proj", kVOut, kTH);
    AddBf16(ts, "model.layers.1.mlp.gate.weight", {kTExperts, kTH});
    ts.push_back(FakeTensor{"model.layers.1.mlp.gate.e_score_correction_bias",
                            "F32", {kTExperts}, MakeOnesF32(kTExperts)});
    AddFp8(ts, "model.layers.1.mlp.shared_experts.gate_proj", kTMoeI, kTH);
    AddFp8(ts, "model.layers.1.mlp.shared_experts.up_proj", kTMoeI, kTH);
    AddFp8(ts, "model.layers.1.mlp.shared_experts.down_proj", kTH, kTMoeI);
    for (int e = 0; e < kTExperts; ++e)
    {
        const std::string p = "model.layers.1.mlp.experts." + std::to_string(e);
        AddFp8(ts, p + ".gate_proj", kTMoeI, kTH);
        AddFp8(ts, p + ".up_proj", kTMoeI, kTH);
        AddFp8(ts, p + ".down_proj", kTH, kTMoeI);
    }

    // MTP layer 40: stored, never applied.
    AddBf16(ts, "model.layers.40.eh_proj.weight", {kTH, 2 * kTH});

    WriteSafetensors(dir, "model.safetensors", ts);
}

// Real frozen-geometry full-attention layer set.
inline const std::vector<int> &RealFullAttn()
{
    static const std::vector<int> v = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39};
    return v;
}

// Writes a real-geometry model.index.json (no weights) so the loader's frozen
// passport can be exercised without the synthetic-geometry bypass.
inline void WriteGigaChatIndex(
    const std::filesystem::path &model_root, int hidden_override = 7168,
    const std::string &format_tag = "logestix-gigachat-nvfp4")
{
    std::filesystem::create_directories(model_root / "logestix");
    std::ofstream f(model_root / "logestix" / "model.index.json");
    f << "{\n  \"format\": \"" << format_tag << "\",\n"
      << "  \"model\": \"GigaChat3.5-432B-A28B-Reasoning\",\n"
      << "  \"num_layers\": 40,\n  \"full_attention_layers\": [";
    const auto &fa = RealFullAttn();
    for (size_t i = 0; i < fa.size(); ++i)
    {
        if (i) f << ",";
        f << fa[i];
    }
    f << "],\n  \"hidden_size\": " << hidden_override
      << ",\n  \"vocab_size\": 128256,\n  \"num_attention_heads\": 64,\n"
      << "  \"num_key_value_heads\": 64,\n  \"num_experts\": 256,\n"
      << "  \"num_experts_per_tok\": 8,\n  \"moe_intermediate_size\": 2048,\n"
      << "  \"first_k_dense_replace\": 3,\n  \"rms_norm_eps\": 1e-06,\n"
      << "  \"router_scaling_factor\": 2.5,\n  \"swiglu_limit\": 10,\n"
      << "  \"bos_token_id\": 1,\n  \"eos_token_id\": 2,\n  \"pad_token_id\": 2,\n"
      << "  \"deferred\": {\"mtp\": \"stored\", \"mtp_layers\": [40,41,42], "
         "\"mtp_tensors\": 129},\n  \"total_pinned_tensors\": 0\n}\n";
}

} // namespace gigachattest
