#pragma once

// Shared fixture builder for Qwen3.8-Flash-Next (qwen4_exp) converter /
// loader / kernel tests. Builds a synthetic qwen4_exp safetensors model
// directory on disk. Dense attention geometry mirrors the qwen hybrid
// layout so the dense-part arrow records stay schema-compatible; adds the
// qwen4_exp block inventory: GatedResidual hyper-connections (attn+mlp per
// layer + global mixer), QSA indexer on full-attention layers, PLE n-gram
// projections on the PLE layer, MoE 512-expert analog, MTP and visual decoys.
// Tensor-writing primitives come from the family-neutral
// test_synth_checkpoint.h (suite-autonomy invariant).

#include "test_synth_checkpoint.h"

#include <algorithm>
#include <cstring>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace qwen_moe_exp_test {

using synthtest::FakeTensor;
using synthtest::MakeBf16;
using synthtest::MakeBf16Values;
using synthtest::MakeF32Values;
using synthtest::WriteSafetensors;

// Cross-platform env gate set/clear (etalon: test_qwen_moe_fixture.h).
inline void ExpSetEnv(const char *name, const char *value)
{
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

inline void ExpUnsetEnv(const char *name)
{
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

// Tiny qwen4_exp geometry. Dense attention side mirrors the shared qwen
// hybrid layout defaults
// (hidden=64, layers=4, full attention at index 3, heads=4/kv=1/head_dim=32,
// GDN nk=2/kd=8, nv=4/vd=8, conv=4, vocab=64).
// qwen4_exp side: experts=8, top_k=2, I=16, SI=16, hc_count=2 (shrunken from
// 4 so hc_hidden=128), hc_lowrank=8, indexer 2Q/1KV hd=8 budget=8 compress=2,
// PLE ngram_size=3, heads_per_ngram=2 (4 heads, head_dim=16), split_parts=2
// shards, seed=1234.
struct QwenExpTestConfig {
    // Dense (lockstep with the shared qwen hybrid layout defaults).
    int hidden_size = 64;
    int vocab_size = 64;
    int num_layers = 4;
    std::vector<std::string> layer_types = {
        "linear_attention", "linear_attention", "linear_attention", "full_attention"};
    int linear_num_key_heads = 2;
    int linear_num_value_heads = 4;
    int linear_key_head_dim = 8;
    int linear_value_head_dim = 8;
    int linear_conv_kernel_dim = 4;
    int num_attention_heads = 4;
    int num_key_value_heads = 1;
    int head_dim = 32;

    // MoE.
    int num_experts = 8;
    int num_experts_per_tok = 2;
    int moe_intermediate_size = 16;
    int shared_expert_intermediate_size = 16;

    // Hyper-connections.
    int hc_count = 2;
    int hc_lowrank = 8;

    // QSA indexer.
    int indexer_n_heads = 2;
    int indexer_kv_heads = 1;
    int indexer_head_dim = 8;
    int indexer_budget = 8;
    int indexer_compress_ratio = 2;

    // PLE.
    int ngram_size = 3;
    int heads_per_ngram = 2;
    int ple_embed_dim = 64;
    int ple_conv_kernel_size = 4;
    int split_ngram_parts = 2;
    int64_t ngram_vocab_size_base = 1000000;
    int64_t seed = 1234;
    // PLE ids follow the HF convention: id = tensor layer idx + 1.
    // Fixture tensors live under layers.1.ple (matching the real model),
    // so the id is 2.
    std::vector<int> ple_layer_ids = {2};

    int eos_token_id = 7;
};

inline const QwenExpTestConfig &ExpCfg()
{
    static QwenExpTestConfig cfg;
    return cfg;
}

inline void WriteExpConfig(const std::filesystem::path &dir)
{
    const auto &c = ExpCfg();
    std::string layer_types = "[";
    for (size_t i = 0; i < c.layer_types.size(); ++i) {
        if (i) layer_types += ",";
        layer_types += "\"" + c.layer_types[i] + "\"";
    }
    layer_types += "]";

    std::string ple_ids = "[";
    for (size_t i = 0; i < c.ple_layer_ids.size(); ++i) {
        if (i) ple_ids += ",";
        ple_ids += std::to_string(c.ple_layer_ids[i]);
    }
    ple_ids += "]";

    std::string json =
        "{\"architectures\":[\"Qwen4ExpForConditionalGeneration\"],"
        "\"model_type\":\"qwen4_exp\","
        "\"eos_token_id\":" + std::to_string(c.eos_token_id) + ","
        "\"text_config\":{"
        "\"model_type\":\"qwen4_exp_text\""
        ",\"hidden_size\":" + std::to_string(c.hidden_size) +
        ",\"num_hidden_layers\":" + std::to_string(c.num_layers) +
        ",\"vocab_size\":" + std::to_string(c.vocab_size) +
        ",\"num_attention_heads\":" + std::to_string(c.num_attention_heads) +
        ",\"num_key_value_heads\":" + std::to_string(c.num_key_value_heads) +
        ",\"head_dim\":" + std::to_string(c.head_dim) +
        ",\"linear_num_key_heads\":" + std::to_string(c.linear_num_key_heads) +
        ",\"linear_num_value_heads\":" + std::to_string(c.linear_num_value_heads) +
        ",\"linear_key_head_dim\":" + std::to_string(c.linear_key_head_dim) +
        ",\"linear_value_head_dim\":" + std::to_string(c.linear_value_head_dim) +
        ",\"linear_conv_kernel_dim\":" + std::to_string(c.linear_conv_kernel_dim) +
        ",\"rms_norm_eps\":1e-06"
        ",\"partial_rotary_factor\":0.25"
        ",\"rope_theta\":10000.0"
        ",\"layer_types\":" + layer_types +
        ",\"num_experts\":" + std::to_string(c.num_experts) +
        ",\"num_experts_per_tok\":" + std::to_string(c.num_experts_per_tok) +
        ",\"moe_intermediate_size\":" + std::to_string(c.moe_intermediate_size) +
        ",\"shared_expert_intermediate_size\":" +
            std::to_string(c.shared_expert_intermediate_size) +
        ",\"output_gate_type\":\"sigmoid\""
        ",\"output_router_logits\":false"
        ",\"hidden_act\":\"silu\""
        ",\"eos_token_id\":" + std::to_string(c.eos_token_id) +
        // Hyper-connections.
        ",\"hc_count\":" + std::to_string(c.hc_count) +
        ",\"hc_lowrank\":" + std::to_string(c.hc_lowrank) +
        // QSA indexer.
        ",\"indexer_n_heads\":" + std::to_string(c.indexer_n_heads) +
        ",\"indexer_kv_heads\":" + std::to_string(c.indexer_kv_heads) +
        ",\"indexer_head_dim\":" + std::to_string(c.indexer_head_dim) +
        ",\"indexer_budget\":" + std::to_string(c.indexer_budget) +
        ",\"indexer_compress_ratio\":" + std::to_string(c.indexer_compress_ratio) +
        // PLE.
        ",\"ngram_size\":" + std::to_string(c.ngram_size) +
        ",\"heads_per_ngram\":" + std::to_string(c.heads_per_ngram) +
        ",\"ple_embed_dim\":" + std::to_string(c.ple_embed_dim) +
        ",\"ple_conv_kernel_size\":" + std::to_string(c.ple_conv_kernel_size) +
        ",\"ple_layer_ids\":" + ple_ids +
        ",\"split_ngram_parts\":" + std::to_string(c.split_ngram_parts) +
        ",\"ngram_vocab_size_base\":" + std::to_string(c.ngram_vocab_size_base) +
        ",\"make_ngram_vocab_size_divisible_by\":128"
        ",\"seed\":" + std::to_string(c.seed) +
        ",\"rope_parameters\":{\"rope_type\":\"default\",\"rope_theta\":10000.0,"
        "\"partial_rotary_factor\":0.25}"
        "}}";
    std::ofstream f(dir / "config.json");
    f << json;
}

inline int64_t FindNthPrimeAfterFixture(int64_t start, int64_t count)
{
    auto is_prime = [](int64_t v) {
        if (v < 2) return false;
        if (v % 2 == 0) return v == 2;
        for (int64_t d = 3; d * d <= v; d += 2)
            if (v % d == 0) return false;
        return true;
    };
    int64_t p = start;
    for (int64_t i = 0; i < count; ++i) {
        ++p;
        while (!is_prime(p)) ++p;
    }
    return p;
}

// Full synthetic checkpoint mirroring the real qwen4_exp checkpoint tensor
// inventory: globals (embed/lm_head/global hc mixer), per-layer attention
// (GDN or QSA + indexer) + attn/mlp hyper-connections, MoE block on EVERY
// layer, PLE projections + n-gram shards on the PLE layer, MTP and visual
// decoy tensors the converter must account for explicitly.
inline std::vector<FakeTensor> BuildSyntheticExpTensors()
{
    const auto &c = ExpCfg();
    const int H = c.hidden_size;
    const int V = c.vocab_size;
    const int E = c.num_experts;
    const int I = c.moe_intermediate_size;
    const int SI = c.shared_expert_intermediate_size;
    const int HC = c.hc_count;
    const int HCH = H * HC;
    const int LR = c.hc_lowrank;

    // Derived dense dims — identical formulas to qwentest.
    const int KD = c.linear_num_key_heads * c.linear_key_head_dim;
    const int VD = c.linear_num_value_heads * c.linear_value_head_dim;
    const int CD = KD * 2 + VD;
    const int NH = c.num_attention_heads * c.head_dim * 2;  // gate doubled
    const int KV = c.num_key_value_heads * c.head_dim;
    // Indexer projections: (n_heads + kv_heads) * head_dim.
    const int IDX = (c.indexer_n_heads + c.indexer_kv_heads) * c.indexer_head_dim;
    // PLE heads.
    const int PLE_HEADS = (c.ngram_size - 1) * c.heads_per_ngram;
    const int PLE_HDIM = c.ple_embed_dim / PLE_HEADS;

    // Weight-like float patterns (denormal-heavy index patterns collapse to
    // identical NVFP4 nibble packs; etalon weight_pattern).
    auto weight_pattern = [](int64_t i) {
        return ((static_cast<float>(i * 37 % 251)) / 251.0f) - 0.5f;
    };

    std::vector<FakeTensor> t;
    t.push_back({"model.language_model.embed_tokens.weight", "BF16",
                 {V, H}, MakeBf16(V * H)});
    t.push_back({"lm_head.weight", "BF16", {V, H}, MakeBf16(V * H)});
    // Global hc mixer (use_combine=false: hc_norm + input_mix only).
    t.push_back({"model.language_model.hyper_connection_mixer.hc_norm.weight",
                 "BF16", {HCH}, MakeBf16(HCH)});
    t.push_back({"model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
                 "BF16", {LR, HCH}, MakeBf16Values(static_cast<int64_t>(LR) * HCH, weight_pattern)});
    t.push_back({"model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
                 "BF16", {HCH, LR}, MakeBf16Values(static_cast<int64_t>(HCH) * LR, weight_pattern)});

    for (int l = 0; l < c.num_layers; ++l) {
        const std::string lp =
            "model.language_model.layers." + std::to_string(l) + ".";
        const bool full = c.layer_types[l] == "full_attention";

        if (full) {
            const auto fa = lp + "self_attn.";
            t.push_back({fa + "q_proj.weight", "BF16", {NH, H}, MakeBf16(NH * H)});
            t.push_back({fa + "k_proj.weight", "BF16", {KV, H}, MakeBf16(KV * H)});
            t.push_back({fa + "v_proj.weight", "BF16", {KV, H}, MakeBf16(KV * H)});
            t.push_back({fa + "o_proj.weight", "BF16",
                         {H, c.num_attention_heads * c.head_dim},
                         MakeBf16(H * c.num_attention_heads * c.head_dim)});
            t.push_back({fa + "q_norm.weight", "BF16", {c.head_dim},
                         MakeBf16(c.head_dim)});
            t.push_back({fa + "k_norm.weight", "BF16", {c.head_dim},
                         MakeBf16(c.head_dim)});
            t.push_back({fa + "indexer.index_qk_proj.weight", "BF16",
                         {IDX, H}, MakeBf16(IDX * H)});
            t.push_back({fa + "indexer.q_layernorm.weight", "BF16",
                         {c.indexer_head_dim}, MakeBf16(c.indexer_head_dim)});
            t.push_back({fa + "indexer.k_layernorm.weight", "BF16",
                         {c.indexer_head_dim}, MakeBf16(c.indexer_head_dim)});
        } else {
            const auto la = lp + "linear_attn.";
            t.push_back({la + "in_proj_qkv.weight", "BF16", {KD * 2 + VD, H},
                         MakeBf16((KD * 2 + VD) * H)});
            t.push_back({la + "in_proj_a.weight", "BF16",
                         {c.linear_num_value_heads, H},
                         MakeBf16(c.linear_num_value_heads * H)});
            t.push_back({la + "in_proj_b.weight", "BF16",
                         {c.linear_num_value_heads, H},
                         MakeBf16(c.linear_num_value_heads * H)});
            t.push_back({la + "in_proj_z.weight", "BF16", {VD, H},
                         MakeBf16(VD * H)});
            t.push_back({la + "out_proj.weight", "BF16", {H, VD},
                         MakeBf16(H * VD)});
            t.push_back({la + "conv1d.weight", "BF16",
                         {CD, c.linear_conv_kernel_dim},
                         MakeBf16(CD * c.linear_conv_kernel_dim)});
            t.push_back({la + "norm.weight", "BF16",
                         {c.linear_value_head_dim},
                         MakeBf16(c.linear_value_head_dim)});
            t.push_back({la + "A_log", "F32", {c.linear_num_value_heads},
                         MakeF32Values(c.linear_num_value_heads,
                                       [](int64_t i) { return (float)i * 0.5f + 1.0f; })});
            t.push_back({la + "dt_bias", "F32", {c.linear_num_value_heads},
                         MakeF32Values(c.linear_num_value_heads,
                                       [](int64_t i) { return (float)i * 0.1f; })});
        }

        // Hyper-connections (attn + mlp) per layer.
        for (const char *which : {"attn_hyper_connection", "mlp_hyper_connection"}) {
            const auto hp = lp + which + ".";
            t.push_back({hp + "hc_norm.weight", "BF16", {HCH}, MakeBf16(HCH)});
            t.push_back({hp + "input_mix_weight_down.weight", "BF16",
                         {LR, HCH},
                         MakeBf16Values(static_cast<int64_t>(LR) * HCH, weight_pattern)});
            t.push_back({hp + "input_mix_weight_up.weight", "BF16",
                         {HCH, LR},
                         MakeBf16Values(static_cast<int64_t>(HCH) * LR, weight_pattern)});
            t.push_back({hp + "block_inject_weight.weight", "BF16",
                         {HC, HCH},
                         MakeBf16Values(static_cast<int64_t>(HC) * HCH, weight_pattern)});
        }

        // MoE block (all layers).
        t.push_back({lp + "mlp.gate.weight", "BF16", {E, H},
                     MakeBf16Values(E * H, weight_pattern)});
        t.push_back({lp + "mlp.experts.gate_up_proj", "BF16", {E, 2 * I, H},
                     MakeBf16Values(static_cast<int64_t>(E) * 2 * I * H,
                                    weight_pattern)});
        t.push_back({lp + "mlp.experts.down_proj", "BF16", {E, H, I},
                     MakeBf16Values(static_cast<int64_t>(E) * H * I,
                                    weight_pattern)});
        t.push_back({lp + "mlp.shared_expert.gate_proj.weight", "BF16", {SI, H},
                     MakeBf16Values(SI * H, weight_pattern)});
        t.push_back({lp + "mlp.shared_expert.up_proj.weight", "BF16", {SI, H},
                     MakeBf16Values(SI * H, weight_pattern)});
        t.push_back({lp + "mlp.shared_expert.down_proj.weight", "BF16", {H, SI},
                     MakeBf16Values(H * SI, weight_pattern)});
        t.push_back({lp + "mlp.shared_expert_gate.weight", "BF16", {1, H},
                     MakeBf16Values(H, [](int64_t i) { return -0.5f + (float)i * 0.01f; })});

        // PLE layer: projections + n-gram shards + meta buffers.
        if (l == 1) {
            const auto pp = lp + "ple.";
            t.push_back({pp + "key_proj.weight", "BF16", {HCH, c.ple_embed_dim},
                         MakeBf16Values(static_cast<int64_t>(HCH) * c.ple_embed_dim, weight_pattern)});
            t.push_back({pp + "value_proj.weight", "BF16", {H, c.ple_embed_dim},
                         MakeBf16Values(static_cast<int64_t>(H) * c.ple_embed_dim, weight_pattern)});
            t.push_back({pp + "norm_key.weight", "BF16", {HCH}, MakeBf16(HCH)});
            t.push_back({pp + "norm_query.weight", "BF16", {HCH}, MakeBf16(HCH)});
            t.push_back({pp + "norm_conv.weight", "BF16", {HCH}, MakeBf16(HCH)});
            t.push_back({pp + "conv1d.weight", "BF16",
                         {HCH, 1, c.ple_conv_kernel_size},
                         MakeBf16(HCH * c.ple_conv_kernel_size)});
            // N-gram embedding shards: [vocab_base-ish rows, PLE_HDIM].
            // Fixture-scale: 1000 rows per shard.
            for (int s = 0; s < c.split_ngram_parts; ++s) {
                t.push_back({pp + "ple_embedding.ngram_embedding.shard_" +
                                 std::to_string(s) + ".weight",
                             "BF16", {1000, PLE_HDIM},
                             MakeBf16(1000 * PLE_HDIM)});
            }
            // Meta buffers (I64): multipliers [ngram_size], offsets + sizes
            // [PLE_HEADS] — the converter recomputes and cross-checks.
            const int64_t vb = c.ngram_vocab_size_base;
            std::vector<int64_t> sizes;
            for (int h = 0; h < PLE_HEADS; ++h)
                sizes.push_back(FindNthPrimeAfterFixture(vb - 1, h + 1));
            std::vector<int64_t> offs;
            int64_t tot = 0;
            for (int64_t s : sizes) { offs.push_back(tot); tot += s; }
            // Multipliers: port of _build_layer_multipliers(seed=1234).
            {
                const uint64_t MASK = ~0ull;
                const uint64_t GAMMA = 0x9E3779B97F4A7C15ull;
                const uint64_t M1 = 0xBF58476D1CE4E5B9ull;
                const uint64_t M2 = 0x94D049BB133111EBull;
                auto sm64 = [&](uint64_t v) {
                    v = (v + GAMMA) & MASK;
                    v = ((v ^ (v >> 30)) * M1) & MASK;
                    v = ((v ^ (v >> 27)) * M2) & MASK;
                    return (v ^ (v >> 31)) & MASK;
                };
                const int64_t max_long = (1ll << 63) - 1;
                const int64_t half = std::max<int64_t>(1, (max_long / V) / 2);
                const int64_t base = c.seed + 10007 * 0;  // ple_layer_index 0
                std::vector<int64_t> mult;
                for (int i = 0; i < c.ngram_size; ++i) {
                    const uint64_t v = static_cast<uint64_t>(base) +
                                       GAMMA * static_cast<uint64_t>(i + 1);
                    mult.push_back(static_cast<int64_t>(2 * (sm64(v) % static_cast<uint64_t>(half)) + 1));
                }
                std::vector<uint8_t> raw(mult.size() * 8);
                std::memcpy(raw.data(), mult.data(), raw.size());
                t.push_back({pp + "ple_embedding.layer_multipliers", "I64",
                             {static_cast<int64_t>(mult.size())}, std::move(raw)});
            }
            std::vector<uint8_t> raw_off(offs.size() * 8);
            std::memcpy(raw_off.data(), offs.data(), raw_off.size());
            t.push_back({pp + "ple_embedding.ngram_heads_offsets", "I64",
                         {static_cast<int64_t>(offs.size())}, std::move(raw_off)});
            std::vector<uint8_t> raw_sz(sizes.size() * 8);
            std::memcpy(raw_sz.data(), sizes.data(), raw_sz.size());
            t.push_back({pp + "ple_embedding.ngram_heads_vocab_sizes", "I64",
                         {static_cast<int64_t>(sizes.size())}, std::move(raw_sz)});
        }
    }

    // Decoys: vision tower skipped-and-counted; MTP stored, never applied.
    t.push_back({"model.visual.blocks.0.attn.qkv.weight", "BF16", {H, H},
                 MakeBf16(H * H)});
    t.push_back({"visual.blocks.0.norm1.weight", "BF16", {H}, MakeBf16(H)});
    t.push_back({"mtp.fc.weight", "BF16", {H, H}, MakeBf16(H * H)});
    t.push_back({"mtp.norm.weight", "BF16", {H}, MakeBf16(H)});
    return t;
}

// Builds <dir>/model.safetensors + config.json. Returns tensor count written.
inline int BuildSyntheticExpCheckpoint(const std::filesystem::path &dir)
{
    std::filesystem::create_directories(dir);
    auto tensors = BuildSyntheticExpTensors();
    WriteSafetensors(dir, "model.safetensors", tensors);
    WriteExpConfig(dir);
    return static_cast<int>(tensors.size());
}

} // namespace qwen_moe_exp_test
