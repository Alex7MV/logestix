#pragma once

// Shared fixture builder for Qwen3.5-35B-A3B MoE converter / loader / kernel
// tests. Builds a synthetic qwen3_5_moe safetensors model directory on disk.
// Attention geometry mirrors qwentest (qwen_dense_test_fixture.h) exactly so the
// dense-part arrow records stay schema-compatible; adds the MoE block
// (router / fused experts / shared expert) present in every layer.

#include "qwen_dense_test_fixture.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace qwen_moe_test {

using qwentest::FakeTensor;
using qwentest::MakeBf16;
using qwentest::MakeBf16Values;
using qwentest::MakeF32Values;
using qwentest::WriteSafetensors;

// Cross-platform env gate set/clear: setenv/unsetenv are POSIX-only; MSVC uses
// _putenv_s, where an empty value removes the variable (verified 2026-08; the
// LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY gate only checks getenv() presence).
inline void MoeSetEnv(const char *name, const char *value)
{
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

inline void MoeUnsetEnv(const char *name)
{
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

// Tiny MoE geometry. Attention side matches qwentest::TestCfg() defaults so
// both fixtures agree on dense record shapes:
//   hidden=64, layers=4 (full attention at index 3), heads=4/kv=1/head_dim=32
//   linear nk=2/kd=8, nv=4/vd=8, conv=4, vocab=64
// MoE side: experts=8, top_k=2, moe_inter=16, shared_inter=16.
struct QwenMoeTestConfig {
    // Dense (kept in lockstep with qwentest::QwenTestConfig defaults).
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

    int eos_token_id = 7;
};

inline const QwenMoeTestConfig &MoeCfg()
{
    static QwenMoeTestConfig cfg;
    return cfg;
}

inline void WriteMoeConfig(const std::filesystem::path &dir)
{
    const auto &c = MoeCfg();
    std::string layer_types = "[";
    for (size_t i = 0; i < c.layer_types.size(); ++i) {
        if (i) layer_types += ",";
        layer_types += "\"" + c.layer_types[i] + "\"";
    }
    layer_types += "]";

    std::string json =
        "{\"architectures\":[\"Qwen3_5MoeForConditionalGeneration\"],"
        "\"model_type\":\"qwen3_5_moe\","
        "\"eos_token_id\":" + std::to_string(c.eos_token_id) + ","
        "\"text_config\":{"
        "\"hidden_size\":" + std::to_string(c.hidden_size) +
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
        ",\"mlp_only_layers\":[]"
        ",\"hidden_act\":\"silu\""
        ",\"eos_token_id\":" + std::to_string(c.eos_token_id) +
        ",\"rope_parameters\":{\"rope_type\":\"default\",\"rope_theta\":10000.0,"
        "\"partial_rotary_factor\":0.25}"
        "}}";
    std::ofstream f(dir / "config.json");
    f << json;
}

// Full synthetic checkpoint mirroring the real checkpoint tensor inventory:
// globals, per-layer attention/norms, MoE block on EVERY layer, plus MTP and
// visual decoy tensors that the converter must account for explicitly.
inline std::vector<FakeTensor> BuildSyntheticMoeTensors()
{
    namespace fs = std::filesystem;
    const auto &c = MoeCfg();
    const int H = c.hidden_size;
    const int V = c.vocab_size;
    const int E = c.num_experts;
    const int I = c.moe_intermediate_size;
    const int SI = c.shared_expert_intermediate_size;

    // Derived dense dims — identical formulas to qwentest.
    const auto &q = qwentest::TestCfg();
    const int KD = q.linear_num_key_heads * q.linear_key_head_dim;
    const int VD = q.linear_num_value_heads * q.linear_value_head_dim;
    const int CD = KD * 2 + VD;
    const int NH = q.num_attention_heads * q.head_dim * 2;  // gate doubled
    const int KV = q.num_key_value_heads * q.head_dim;

    std::vector<FakeTensor> t;
    t.push_back({"model.language_model.embed_tokens.weight", "BF16",
                 {V, H}, MakeBf16(V * H)});
    t.push_back({"lm_head.weight", "BF16", {V, H}, MakeBf16(V * H)});
    t.push_back({"model.language_model.norm.weight", "BF16", {H},
                 MakeBf16Values(H, [](int64_t i) { return (float)i * 0.01f; })});

    for (int l = 0; l < c.num_layers; ++l) {
        const std::string lp =
            "model.language_model.layers." + std::to_string(l) + ".";
        const bool full = c.layer_types[l] == "full_attention";

        t.push_back({lp + "input_layernorm.weight", "BF16", {H}, MakeBf16(H)});
        t.push_back({lp + "post_attention_layernorm.weight", "BF16", {H},
                     MakeBf16(H)});

        if (full) {
            const auto fa = lp + "self_attn.";
            t.push_back({fa + "q_proj.weight", "BF16", {NH, H}, MakeBf16(NH * H)});
            t.push_back({fa + "k_proj.weight", "BF16", {KV, H}, MakeBf16(KV * H)});
            t.push_back({fa + "v_proj.weight", "BF16", {KV, H}, MakeBf16(KV * H)});
            t.push_back({fa + "o_proj.weight", "BF16",
                         {H, q.num_attention_heads * q.head_dim},
                         MakeBf16(H * q.num_attention_heads * q.head_dim)});
            t.push_back({fa + "q_norm.weight", "BF16", {q.head_dim},
                         MakeBf16(q.head_dim)});
            t.push_back({fa + "k_norm.weight", "BF16", {q.head_dim},
                         MakeBf16(q.head_dim)});
        } else {
            const auto la = lp + "linear_attn.";
            t.push_back({la + "in_proj_qkv.weight", "BF16", {KD * 2 + VD, H},
                         MakeBf16((KD * 2 + VD) * H)});
            t.push_back({la + "in_proj_a.weight", "BF16",
                         {q.linear_num_value_heads, H},
                         MakeBf16(q.linear_num_value_heads * H)});
            t.push_back({la + "in_proj_b.weight", "BF16",
                         {q.linear_num_value_heads, H},
                         MakeBf16(q.linear_num_value_heads * H)});
            t.push_back({la + "in_proj_z.weight", "BF16", {VD, H},
                         MakeBf16(VD * H)});
            t.push_back({la + "out_proj.weight", "BF16", {H, VD},
                         MakeBf16(H * VD)});
            t.push_back({la + "conv1d.weight", "BF16",
                         {CD, q.linear_conv_kernel_dim},
                         MakeBf16(CD * q.linear_conv_kernel_dim)});
            t.push_back({la + "norm.weight", "BF16",
                         {q.linear_value_head_dim},
                         MakeBf16(q.linear_value_head_dim)});
            t.push_back({la + "A_log", "F32", {q.linear_num_value_heads},
                         MakeF32Values(q.linear_num_value_heads,
                                       [](int64_t i) { return (float)i * 0.5f + 1.0f; })});
            t.push_back({la + "dt_bias", "F32", {q.linear_num_value_heads},
                         MakeF32Values(q.linear_num_value_heads,
                                       [](int64_t i) { return (float)i * 0.1f; })});
        }

        // MoE block (all layers). Weight-like float patterns (denormal-heavy
        // index patterns collapse to identical NVFP4 nibble packs, which
        // would defeat the split-distinctness assertions).
        auto weight_pattern = [](int64_t i) {
            return ((static_cast<float>(i * 37 % 251)) / 251.0f) - 0.5f;
        };
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
    }

    // Decoys: vision tower must be skipped-and-counted; MTP block stored but
    // never applied (HF parity decision).
    t.push_back({"model.visual.blocks.0.attn.qkv.weight", "BF16", {H, H},
                 MakeBf16(H * H)});
    t.push_back({"visual.blocks.0.norm1.weight", "BF16", {H}, MakeBf16(H)});
    t.push_back({"mtp.fc.weight", "BF16", {H, H}, MakeBf16(H * H)});
    t.push_back({"mtp.norm.weight", "BF16", {H}, MakeBf16(H)});
    return t;
}

// Builds <dir>/model.safetensors + config.json. Returns tensor count written.
inline int BuildSyntheticMoeCheckpoint(const std::filesystem::path &dir)
{
    std::filesystem::create_directories(dir);
    auto tensors = BuildSyntheticMoeTensors();
    WriteSafetensors(dir, "model.safetensors", tensors);
    WriteMoeConfig(dir);
    return static_cast<int>(tensors.size());
}

} // namespace qwen_moe_test
