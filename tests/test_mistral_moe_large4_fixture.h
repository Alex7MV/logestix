#pragma once

// Mistral-Large-4-1T-A52B synthetic checkpoint fixture (Model Ingestion
// Protocol stage 1). Family-owned: includes only the family-neutral
// test_synth_checkpoint.h primitives.
//
// The fixture reproduces the topology features the converter must branch on:
//   * a RoPE layer (layer 0) and a NoPE layer (layer 3),
//   * a KV-sharing layer (layer 3 reuses layer 2, both NoPE),
//   * a shared-indexer layer (layer 1) and a full-indexer layer (layer 3),
//   * a dense MLP prefix (layer 0) and MoE layers (1..3),
//   * FP8-E4M3 linears with F32 128x128 tile scales and BF16 edges,
//   * pixtral vision ballast and a `mtp_dspark` head (stripped + counted).

#include "test_synth_checkpoint.h"

#include <absl/strings/str_cat.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace mistrallarge4test {

using synthtest::FakeTensor;

inline std::vector<int64_t> Sh2(int64_t a, int64_t b) { return {a, b}; }
inline std::vector<int64_t> Sh1(int64_t a) { return {a}; }

inline std::vector<uint8_t> Fp8Bytes(int64_t n)
{
    // E4M3FN finite magnitudes: 0x30..0x3f is 2^-3..~1.9. All finite.
    std::vector<uint8_t> out(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        out[static_cast<size_t>(i)] =
            static_cast<uint8_t>(0x30 + ((i * 7 + 1) % 15));
    return out;
}

inline std::vector<uint8_t> TileScales(int64_t rows, int64_t cols,
                                       int64_t* /*unused*/ = nullptr)
{
    const int64_t sr = (rows + 127) / 128;
    const int64_t sc = (cols + 127) / 128;
    return synthtest::MakeF32Values(sr * sc, [](int64_t i) {
        return 0.02f + 0.001f * static_cast<float>(i % 5);
    });
}

inline std::vector<uint8_t> Bf16Vals(int64_t n)
{
    return synthtest::MakeBf16Values(
        n, [](int64_t i) { return 0.05f * static_cast<float>((i % 13) - 6); });
}

inline std::vector<uint8_t> F32Vals(int64_t n, float base = 0.01f)
{
    return synthtest::MakeF32Values(n, [base](int64_t i) {
        return base * (1.0f + 0.1f * static_cast<float>(i % 7));
    });
}

// Small synthetic geometry. It does NOT match the frozen passport;
// LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1 relaxes the gate (a RUNTIME env var set by
// tests/CMakeLists.txt).
struct Large4SynthGeom {
    int hidden = 64;
    int layers = 4;
    int heads = 2;
    int kv_lora = 16;
    int qk_nope = 8;
    int qk_rope = 8;
    int v_head = 8;
    int index_heads = 2;
    int index_head_dim = 16;
    int moe_inter = 32;
    int dense_inter = 48;
    int experts = 2;
    int vocab = 64;

    int qk_head() const { return qk_nope + qk_rope; }
    bool rope(int l) const { return l % 4 != 3; }
    bool moe(int l) const { return l != 0; }
    bool full_indexer(int l) const
    {
        return l == 0 || l == 3;
    }
    int kv_source(int l) const { return l == 3 ? 2 : l; }
};

inline void WriteConfig(const std::filesystem::path &dir,
                        const Large4SynthGeom &g)
{
    std::ofstream f(dir / "config.json");
    f << "{\n  \"model_type\": \"mistral_large4\",\n";
    f << "  \"text_config\": {\n";
    f << "    \"hidden_size\": " << g.hidden << ",\n";
    f << "    \"num_hidden_layers\": " << g.layers << ",\n";
    f << "    \"num_attention_heads\": " << g.heads << ",\n";
    f << "    \"num_key_value_heads\": " << g.heads << ",\n";
    f << "    \"kv_lora_rank\": " << g.kv_lora << ",\n";
    f << "    \"qk_nope_head_dim\": " << g.qk_nope << ",\n";
    f << "    \"qk_rope_head_dim\": " << g.qk_rope << ",\n";
    f << "    \"v_head_dim\": " << g.v_head << ",\n";
    f << "    \"vocab_size\": " << g.vocab << ",\n";
    f << "    \"moe_intermediate_size\": " << g.moe_inter << ",\n";
    f << "    \"intermediate_size\": " << g.dense_inter << ",\n";
    f << "    \"n_routed_experts\": " << g.experts << ",\n";
    f << "    \"num_experts_per_tok\": 2,\n";
    f << "    \"index_n_heads\": " << g.index_heads << ",\n";
    f << "    \"index_head_dim\": " << g.index_head_dim << ",\n";
    f << "    \"index_topk\": 2048,\n";
    f << "    \"max_position_embeddings\": 4096,\n";
    f << "    \"no_rope_layers\": [";
    for (int l = 0; l < g.layers; ++l) f << (l ? "," : "") << (g.rope(l) ? 1 : 0);
    f << "],\n    \"kv_source_layers\": [";
    for (int l = 0; l < g.layers; ++l) f << (l ? "," : "") << g.kv_source(l);
    f << "],\n    \"indexer_types\": [";
    for (int l = 0; l < g.layers; ++l)
        f << (l ? "," : "") << "\"" << (g.full_indexer(l) ? "full" : "shared")
          << "\"";
    f << "],\n    \"mlp_layer_types\": [";
    for (int l = 0; l < g.layers; ++l)
        f << (l ? "," : "") << "\"" << (g.moe(l) ? "sparse" : "dense") << "\"";
    f << "]\n  }\n}\n";
}

inline void AddFp8(std::vector<FakeTensor> &ts, const std::string &prefix,
                   const std::string &proj, int64_t rows, int64_t cols)
{
    ts.push_back({absl::StrCat(prefix, proj, ".weight"), "F8_E4M3",
                  Sh2(rows, cols), Fp8Bytes(rows * cols)});
    ts.push_back({absl::StrCat(prefix, proj, ".qscale_weight"), "F32",
                  Sh2((rows + 127) / 128, (cols + 127) / 128),
                  TileScales(rows, cols)});
}

inline void AddBf16(std::vector<FakeTensor> &ts, const std::string &name,
                    const std::vector<int64_t> &shape)
{
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    ts.push_back({name, "BF16", shape, Bf16Vals(n)});
}

// Writes one synthetic shard for the whole model (small geometry).
inline void WriteShards(const std::filesystem::path &dir,
                        const Large4SynthGeom &g)
{
    std::vector<FakeTensor> ts;
    const int64_t qk = g.qk_head();
    AddBf16(ts, "tok_embeddings.weight", Sh2(g.vocab, g.hidden));
    AddBf16(ts, "output.weight", Sh2(g.vocab, g.hidden));
    AddBf16(ts, "norm.weight", Sh1(g.hidden));
    AddBf16(ts, "lerope.alpha", Sh1(g.qk_rope / 2));

    for (int l = 0; l < g.layers; ++l)
    {
        const std::string p = absl::StrCat("layers.", l, ".");
        const bool rope = g.rope(l);
        const bool own_kv = g.kv_source(l) == l;
        const bool moe = g.moe(l);
        const int64_t kv_rows =
            rope ? g.kv_lora + g.qk_rope : g.kv_lora;
        const int64_t kvb_rows =
            g.heads * ((rope ? g.qk_nope : qk) + g.v_head);

        AddBf16(ts, absl::StrCat(p, "attention_norm.weight"), Sh1(g.hidden));
        AddBf16(ts, absl::StrCat(p, "ffn_norm.weight"), Sh1(g.hidden));
        AddFp8(ts, p, "attention.wq", g.heads * qk, g.hidden);
        AddFp8(ts, p, "attention.wo", g.hidden, g.heads * g.v_head);
        if (own_kv)
        {
            AddBf16(ts, absl::StrCat(p, "attention.wkv_a_with_mqa.weight"),
                    Sh2(kv_rows, g.hidden));
            AddBf16(ts, absl::StrCat(p, "attention.kv_a_norm.weight"),
                    Sh1(g.kv_lora));
        }
        AddBf16(ts, absl::StrCat(p, "attention.wkv_b.weight"),
                Sh2(kvb_rows, g.kv_lora));
        AddBf16(ts, absl::StrCat(p, "attention.ssmax_scaling.s"),
                Sh1(g.heads));
        AddBf16(ts, absl::StrCat(p, "attention.ssmax_scaling.b"),
                Sh1(g.heads));
        if (g.full_indexer(l))
        {
            AddFp8(ts, p, "attention.indexer.wq_b",
                   g.index_heads * g.index_head_dim, g.hidden);
            AddBf16(ts, absl::StrCat(p, "attention.indexer.wk.weight"),
                    Sh2(g.index_head_dim, g.hidden));
            AddBf16(ts, absl::StrCat(p, "attention.indexer.weights_proj.weight"),
                    Sh2(g.index_heads, g.hidden));
            AddBf16(ts, absl::StrCat(p, "attention.indexer.q_input_norm.weight"),
                    Sh1(g.hidden));
            AddBf16(ts, absl::StrCat(p, "attention.indexer.q_input_norm.bias"),
                    Sh1(g.hidden));
            AddBf16(ts, absl::StrCat(p, "attention.indexer.k_norm.weight"),
                    Sh1(g.index_head_dim));
            AddBf16(ts, absl::StrCat(p, "attention.indexer.k_norm.bias"),
                    Sh1(g.index_head_dim));
            AddBf16(ts, absl::StrCat(p, "attention.indexer.ssmax_scaling.s"),
                    Sh1(g.index_heads));
            AddBf16(ts, absl::StrCat(p, "attention.indexer.ssmax_scaling.b"),
                    Sh1(g.index_heads));
        }
        if (!moe)
        {
            AddFp8(ts, p, "feed_forward.w1", g.dense_inter, g.hidden);
            AddFp8(ts, p, "feed_forward.w2", g.hidden, g.dense_inter);
            AddFp8(ts, p, "feed_forward.w3", g.dense_inter, g.hidden);
        }
        else
        {
            AddBf16(ts, absl::StrCat(p, "gate.weight"),
                    Sh2(g.experts, g.hidden));
            AddFp8(ts, p, "shared_experts.w1", g.moe_inter, g.hidden);
            AddFp8(ts, p, "shared_experts.w2", g.hidden, g.moe_inter);
            AddFp8(ts, p, "shared_experts.w3", g.moe_inter, g.hidden);
            for (int e = 0; e < g.experts; ++e)
            {
                const std::string ep =
                    absl::StrCat(p, "experts.", e, ".");
                AddFp8(ts, ep, "w1", g.moe_inter, g.hidden);
                AddFp8(ts, ep, "w2", g.hidden, g.moe_inter);
                AddFp8(ts, ep, "w3", g.moe_inter, g.hidden);
            }
        }
    }

    // Ballast: vision tower + adapter + patch merger + DSpark MTP head.
    AddBf16(ts, "vision_encoder.patch_conv.weight", {16, 3, 2, 2});
    AddBf16(ts, "vision_encoder.ln_pre.weight", Sh1(16));
    AddBf16(ts, "vision_language_adapter.w_in.weight", Sh2(g.hidden, 16));
    AddBf16(ts, "patch_merger.merging_layer.weight", Sh2(16, 64));
    AddBf16(ts, "pre_mm_projector_norm.weight", Sh1(16));
    AddBf16(ts, "mtp_dspark.backbone.final_norm.weight", Sh1(g.hidden));
    AddBf16(ts, "mtp_dspark.backbone.layers.0.attention.wq.weight",
            Sh2(g.heads * qk, g.hidden));

    synthtest::WriteSafetensors(dir, "synthetic-00001-of-00001.safetensors", ts);
}

inline void WriteCheckpoint(const std::filesystem::path &dir,
                            const Large4SynthGeom &g = {})
{
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    WriteConfig(dir, g);
    WriteShards(dir, g);
}

} // namespace mistrallarge4test
