#pragma once

// Shared fixture builder for Ornith-1.5-35B-A3B converter / loader tests.
// Builds a synthetic qwen3_5_moe checkpoint on disk in either supported source
// layout:
//   MLX  — prefix language_model.model.* / language_model.lm_head.weight,
//          mlp.switch_mlp.{gate,up,down}_proj stacked experts, plain norms in
//          ABSOLUTE form, linear_attn.conv1d.weight [C, K, 1];
//   HF   — prefix model.language_model.* / lm_head.weight, fused
//          mlp.experts.{gate_up_proj,down_proj}, plain norms already in the
//          runtime OFFSET form, conv1d [C, 1, K], plus visual/MTP ballast.
// Family-neutral primitives come from test_synth_checkpoint.h (suite autonomy).

#include "test_synth_checkpoint.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace ornithtest {

using synthtest::FakeTensor;
using synthtest::MakeBf16;
using synthtest::MakeBf16Values;
using synthtest::WriteSafetensors;

struct OrnithTestConfig {
    int hidden_size = 64;
    int vocab_size = 64;
    int num_layers = 4;
    std::vector<std::string> layer_types = {
        "linear_attention", "linear_attention", "linear_attention",
        "full_attention"};
    int linear_num_key_heads = 2;
    int linear_num_value_heads = 4;
    int linear_key_head_dim = 8;
    int linear_value_head_dim = 8;
    int linear_conv_kernel_dim = 4;
    int num_attention_heads = 2;
    int num_key_value_heads = 1;
    int head_dim = 32;
    int num_experts = 8;
    int num_experts_per_tok = 2;
    int moe_intermediate_size = 16;
    int shared_expert_intermediate_size = 16;
    int eos_token_id = 7;
    int bos_token_id = 248044;
};

inline const OrnithTestConfig &OrnithCfg()
{
    static OrnithTestConfig cfg;
    return cfg;
}

inline void WriteOrnithConfig(const std::filesystem::path &dir)
{
    const auto &c = OrnithCfg();
    std::string layer_types = "[";
    for (size_t i = 0; i < c.layer_types.size(); ++i) {
        if (i) layer_types += ",";
        layer_types += "\"" + c.layer_types[i] + "\"";
    }
    layer_types += "]";
    std::ofstream f(dir / "config.json");
    f << "{\"architectures\":[\"Qwen3_5MoeForConditionalGeneration\"],"
      << "\"model_type\":\"qwen3_5_moe\","
      << "\"eos_token_id\":" << c.eos_token_id << ","
      << "\"text_config\":{"
      << "\"hidden_size\":" << c.hidden_size
      << ",\"num_hidden_layers\":" << c.num_layers
      << ",\"vocab_size\":" << c.vocab_size
      << ",\"num_attention_heads\":" << c.num_attention_heads
      << ",\"num_key_value_heads\":" << c.num_key_value_heads
      << ",\"head_dim\":" << c.head_dim
      << ",\"linear_num_key_heads\":" << c.linear_num_key_heads
      << ",\"linear_num_value_heads\":" << c.linear_num_value_heads
      << ",\"linear_key_head_dim\":" << c.linear_key_head_dim
      << ",\"linear_value_head_dim\":" << c.linear_value_head_dim
      << ",\"linear_conv_kernel_dim\":" << c.linear_conv_kernel_dim
      << ",\"rms_norm_eps\":1e-06"
      << ",\"partial_rotary_factor\":0.25"
      << ",\"rope_theta\":10000.0"
      << ",\"layer_types\":" << layer_types
      << ",\"num_experts\":" << c.num_experts
      << ",\"num_experts_per_tok\":" << c.num_experts_per_tok
      << ",\"moe_intermediate_size\":" << c.moe_intermediate_size
      << ",\"shared_expert_intermediate_size\":"
      << c.shared_expert_intermediate_size
      << ",\"bos_token_id\":" << c.bos_token_id
      << ",\"eos_token_id\":" << c.eos_token_id << "}}";
}

inline std::vector<FakeTensor> BuildSyntheticOrnithTensors()
{
    const auto &c = OrnithCfg();
    const int H = c.hidden_size;
    const int V = c.vocab_size;
    const int E = c.num_experts;
    const int I = c.moe_intermediate_size;
    const int SI = c.shared_expert_intermediate_size;
    const int KD = c.linear_num_key_heads * c.linear_key_head_dim;
    const int VD = c.linear_num_value_heads * c.linear_value_head_dim;
    const int CD = KD * 2 + VD;
    const int NH = c.num_attention_heads * c.head_dim * 2;  // gate doubled
    const int KV = c.num_key_value_heads * c.head_dim;

    auto weight_pattern = [](int64_t i) {
        return ((static_cast<float>(i * 37 % 251)) / 251.0f) - 0.5f;
    };

    // Plain RMSNorm weights are stored by the MLX export in ABSOLUTE form
    // (multiplier ~1.0); the converter subtracts 1 into the runtime offset
    // form, so the fixture must use realistic near-ones (not raw bit
    // patterns, which would become degenerate).
    auto norm_pattern = [](int64_t) { return 1.0f; };

    std::vector<FakeTensor> t;
    t.push_back({"language_model.model.embed_tokens.weight", "BF16", {V, H},
                 MakeBf16(V * H)});
    t.push_back({"language_model.lm_head.weight", "BF16", {V, H},
                 MakeBf16(V * H)});
    t.push_back({"language_model.model.norm.weight", "BF16", {H},
                 MakeBf16Values(H, norm_pattern)});

    for (int l = 0; l < c.num_layers; ++l) {
        const std::string lp =
            "language_model.model.layers." + std::to_string(l) + ".";
        const bool full = c.layer_types[static_cast<size_t>(l)] ==
                          "full_attention";
        t.push_back({lp + "input_layernorm.weight", "BF16", {H},
                     MakeBf16Values(H, norm_pattern)});
        t.push_back({lp + "post_attention_layernorm.weight", "BF16", {H},
                     MakeBf16Values(H, norm_pattern)});
        if (full) {
            const auto fa = lp + "self_attn.";
            t.push_back({fa + "q_proj.weight", "BF16", {NH, H}, MakeBf16(NH * H)});
            t.push_back({fa + "k_proj.weight", "BF16", {KV, H}, MakeBf16(KV * H)});
            t.push_back({fa + "v_proj.weight", "BF16", {KV, H}, MakeBf16(KV * H)});
            t.push_back({fa + "o_proj.weight", "BF16",
                         {H, c.num_attention_heads * c.head_dim},
                         MakeBf16(H * c.num_attention_heads * c.head_dim)});
            t.push_back({fa + "q_norm.weight", "BF16", {c.head_dim},
                         MakeBf16Values(c.head_dim, norm_pattern)});
            t.push_back({fa + "k_norm.weight", "BF16", {c.head_dim},
                         MakeBf16Values(c.head_dim, norm_pattern)});
        } else {
            const auto la = lp + "linear_attn.";
            t.push_back({la + "in_proj_qkv.weight", "BF16", {CD, H},
                         MakeBf16(CD * H)});
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
            // MLX conv1d layout: [C, K, 1].
            t.push_back({la + "conv1d.weight", "BF16",
                         {CD, c.linear_conv_kernel_dim, 1},
                         MakeBf16(CD * c.linear_conv_kernel_dim)});
            // RMSNormGated is absolute in both exporters: near-ones.
            t.push_back({la + "norm.weight", "BF16",
                         {c.linear_value_head_dim},
                         MakeBf16Values(c.linear_value_head_dim, norm_pattern)});
            t.push_back({la + "A_log", "BF16", {c.linear_num_value_heads},
                         MakeBf16Values(c.linear_num_value_heads,
                                        [](int64_t) { return 1.0f; })});
            t.push_back({la + "dt_bias", "BF16", {c.linear_num_value_heads},
                         MakeBf16Values(c.linear_num_value_heads,
                                        [](int64_t) { return 0.0f; })});
        }
        // MLX stacked experts: [E, I, H] gate/up, [E, H, I] down.
        t.push_back({lp + "mlp.gate.weight", "BF16", {E, H},
                     MakeBf16Values(E * H, weight_pattern)});
        t.push_back({lp + "mlp.switch_mlp.gate_proj.weight", "BF16", {E, I, H},
                     MakeBf16Values(static_cast<int64_t>(E) * I * H,
                                    weight_pattern)});
        t.push_back({lp + "mlp.switch_mlp.up_proj.weight", "BF16", {E, I, H},
                     MakeBf16Values(static_cast<int64_t>(E) * I * H,
                                    weight_pattern)});
        t.push_back({lp + "mlp.switch_mlp.down_proj.weight", "BF16", {E, H, I},
                     MakeBf16Values(static_cast<int64_t>(E) * H * I,
                                    weight_pattern)});
        t.push_back({lp + "mlp.shared_expert.gate_proj.weight", "BF16", {SI, H},
                     MakeBf16Values(SI * H, weight_pattern)});
        t.push_back({lp + "mlp.shared_expert.up_proj.weight", "BF16", {SI, H},
                     MakeBf16Values(SI * H, weight_pattern)});
        t.push_back({lp + "mlp.shared_expert.down_proj.weight", "BF16", {H, SI},
                     MakeBf16Values(H * SI, weight_pattern)});
        t.push_back({lp + "mlp.shared_expert_gate.weight", "BF16", {1, H},
                     MakeBf16Values(H, [](int64_t i) {
                         return -0.5f + (float)i * 0.01f;
                     })});
    }
    return t;
}

inline int BuildSyntheticOrnithCheckpoint(const std::filesystem::path &dir)
{
    std::filesystem::create_directories(dir);
    auto tensors = BuildSyntheticOrnithTensors();
    WriteSafetensors(dir, "model.safetensors", tensors);
    WriteOrnithConfig(dir);
    return static_cast<int>(tensors.size());
}

// ── HF (transformers) export variant ───────────────────────────────────────
// The upstream release re-exports the checkpoint in the transformers layout:
//   prefix model.language_model.*, lm_head.weight at the root,
//   mlp.experts.{gate_up_proj [E,2I,H], down_proj [E,H,I]} fused experts,
//   linear_attn.conv1d.weight [C,1,K], plain norms already in the runtime
//   OFFSET form, plus the multimodal/MTP ballast that must be stripped
//   (model.visual.*) and stored-but-never-applied (mtp.*).

inline std::vector<FakeTensor> BuildSyntheticOrnithHfTensors()
{
    const auto &c = OrnithCfg();
    const int H = c.hidden_size;
    const int V = c.vocab_size;
    const int E = c.num_experts;
    const int I = c.moe_intermediate_size;
    const int SI = c.shared_expert_intermediate_size;
    const int KD = c.linear_num_key_heads * c.linear_key_head_dim;
    const int VD = c.linear_num_value_heads * c.linear_value_head_dim;
    const int CD = KD * 2 + VD;
    const int NH = c.num_attention_heads * c.head_dim * 2;  // gate doubled
    const int KV = c.num_key_value_heads * c.head_dim;

    auto weight_pattern = [](int64_t i) {
        return ((static_cast<float>(i * 37 % 251)) / 251.0f) - 0.5f;
    };
    // HF stores the plain norms in the runtime offset form (w = true - 1),
    // i.e. small values around 0.0 — NOT the MLX near-ones.
    auto offset_norm_pattern = [](int64_t i) {
        return 0.05f + static_cast<float>(i % 7) * 0.01f;
    };

    std::vector<FakeTensor> t;
    t.push_back({"model.language_model.embed_tokens.weight", "BF16", {V, H},
                 MakeBf16(V * H)});
    t.push_back({"lm_head.weight", "BF16", {V, H}, MakeBf16(V * H)});
    t.push_back({"model.language_model.norm.weight", "BF16", {H},
                 MakeBf16Values(H, offset_norm_pattern)});

    for (int l = 0; l < c.num_layers; ++l) {
        const std::string lp =
            "model.language_model.layers." + std::to_string(l) + ".";
        const bool full = c.layer_types[static_cast<size_t>(l)] ==
                          "full_attention";
        t.push_back({lp + "input_layernorm.weight", "BF16", {H},
                     MakeBf16Values(H, offset_norm_pattern)});
        t.push_back({lp + "post_attention_layernorm.weight", "BF16", {H},
                     MakeBf16Values(H, offset_norm_pattern)});
        if (full) {
            const auto fa = lp + "self_attn.";
            t.push_back({fa + "q_proj.weight", "BF16", {NH, H}, MakeBf16(NH * H)});
            t.push_back({fa + "k_proj.weight", "BF16", {KV, H}, MakeBf16(KV * H)});
            t.push_back({fa + "v_proj.weight", "BF16", {KV, H}, MakeBf16(KV * H)});
            t.push_back({fa + "o_proj.weight", "BF16",
                         {H, c.num_attention_heads * c.head_dim},
                         MakeBf16(H * c.num_attention_heads * c.head_dim)});
            t.push_back({fa + "q_norm.weight", "BF16", {c.head_dim},
                         MakeBf16Values(c.head_dim, offset_norm_pattern)});
            t.push_back({fa + "k_norm.weight", "BF16", {c.head_dim},
                         MakeBf16Values(c.head_dim, offset_norm_pattern)});
        } else {
            const auto la = lp + "linear_attn.";
            t.push_back({la + "in_proj_qkv.weight", "BF16", {CD, H},
                         MakeBf16(CD * H)});
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
            // HF conv1d layout is already [C, 1, K].
            t.push_back({la + "conv1d.weight", "BF16",
                         {CD, 1, c.linear_conv_kernel_dim},
                         MakeBf16(CD * c.linear_conv_kernel_dim)});
            // RMSNormGated stays absolute in both exporters.
            t.push_back({la + "norm.weight", "BF16",
                         {c.linear_value_head_dim},
                         MakeBf16Values(c.linear_value_head_dim,
                                        [](int64_t) { return 1.0f; })});
            t.push_back({la + "A_log", "BF16", {c.linear_num_value_heads},
                         MakeBf16Values(c.linear_num_value_heads,
                                        [](int64_t) { return 1.0f; })});
            t.push_back({la + "dt_bias", "BF16", {c.linear_num_value_heads},
                         MakeBf16Values(c.linear_num_value_heads,
                                        [](int64_t) { return 0.0f; })});
        }
        // HF fused experts: gate_up [E, 2I, H] (gate rows [0,I), up [I,2I)),
        // down [E, H, I].
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
                     MakeBf16Values(H, [](int64_t i) {
                         return -0.5f + static_cast<float>(i) * 0.01f;
                     })});
    }

    // Vision ballast: skipped and counted, never converted.
    t.push_back({"model.visual.patch_embed.proj.weight", "BF16", {H, H},
                 MakeBf16(H * H)});
    t.push_back({"model.visual.blocks.0.attn.qkv.weight", "BF16", {H, H},
                 MakeBf16(H * H)});
    t.push_back({"model.visual.merger.linear_fc1.weight", "BF16", {H, H},
                 MakeBf16(H * H)});

    // MTP subtree: converted and stored, never applied.
    t.push_back({"mtp.fc.weight", "BF16", {H, 2 * VD}, MakeBf16(H * 2 * VD)});
    t.push_back({"mtp.norm.weight", "BF16", {H}, MakeBf16Values(H, offset_norm_pattern)});
    t.push_back({"mtp.layers.0.input_layernorm.weight", "BF16", {H},
                 MakeBf16Values(H, offset_norm_pattern)});
    t.push_back({"mtp.layers.0.self_attn.q_proj.weight", "BF16", {NH, H},
                 MakeBf16(NH * H)});
    t.push_back({"mtp.layers.0.mlp.gate.weight", "BF16", {E, H},
                 MakeBf16Values(E * H, weight_pattern)});
    t.push_back({"mtp.layers.0.mlp.experts.0.gate_proj.weight", "BF16", {I, H},
                 MakeBf16Values(I * H, weight_pattern)});
    return t;
}

inline int BuildSyntheticOrnithHfCheckpoint(const std::filesystem::path &dir)
{
    std::filesystem::create_directories(dir);
    auto tensors = BuildSyntheticOrnithHfTensors();
    WriteSafetensors(dir, "model.safetensors", tensors);
    WriteOrnithConfig(dir);
    return static_cast<int>(tensors.size());
}

// Cross-platform env gate set/clear (setenv/unsetenv are POSIX-only; MSVC uses
// _putenv_s, where an empty value removes the variable).
inline void OrnithSetEnv(const char *name, const char *value)
{
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

inline void OrnithUnsetEnv(const char *name)
{
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

} // namespace ornithtest
