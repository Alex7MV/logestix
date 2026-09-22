#pragma once

// Shared fixture builder for Granite-4.2-30B converter / loader / kernel /
// model tests. Builds a synthetic dense Granite safetensors model directory on
// disk. Tensor-writing primitives live in the family-neutral
// test_synth_checkpoint.h (suite-autonomy invariant) and are re-exported below
// under grantest.

#include "test_synth_checkpoint.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace grantest {

using synthtest::FakeTensor;
using synthtest::MakeBf16;
using synthtest::MakeBf16Values;
using synthtest::MakeF32Values;
using synthtest::TensorNamesIn;
using synthtest::WriteSafetensors;

// Synthetic geometry — tiny so CPU/CUDA references run fast. Mirror the real
// model proportionally (GQA 4:1, head_dim = hidden / heads).
struct GraniteTestConfig {
    int hidden_size = 64;
    int intermediate_size = 128;
    int vocab_size = 64;
    int num_layers = 2;
    int num_attention_heads = 4;
    int num_key_value_heads = 2;
    int head_dim = 16;  // hidden / num_attention_heads

    double rms_norm_eps = 1e-5;
    double rope_theta = 10000.0;
    double attention_multiplier = 0.0625;  // 1 / head_dim

    int bos_id = 0;
    int eos_id = 1;
    int pad_id = 2;
};

inline const GraniteTestConfig &TestCfg()
{
    static GraniteTestConfig cfg;
    return cfg;
}

inline int QProjOut() { return TestCfg().num_attention_heads * TestCfg().head_dim; }
inline int KVProjOut() { return TestCfg().num_key_value_heads * TestCfg().head_dim; }

inline std::vector<FakeTensor> BuildSyntheticGraniteModel()
{
    const auto &c = TestCfg();
    const int H = c.hidden_size;
    const int I = c.intermediate_size;
    const int V = c.vocab_size;

    std::vector<FakeTensor> t;

    // Global.
    t.push_back({"model.embed_tokens.weight", "BF16", {V, H}, MakeBf16(V * H)});
    t.push_back({"lm_head.weight", "BF16", {V, H}, MakeBf16(V * H)});
    t.push_back({"model.norm.weight", "BF16", {H},
                 MakeBf16Values(H, [](int64_t i) { return (float)i * 0.01f; })});

    for (int L = 0; L < c.num_layers; ++L) {
        std::string lp = "model.layers." + std::to_string(L) + ".";

        t.push_back({lp + "input_layernorm.weight", "BF16", {H}, MakeBf16(H)});
        t.push_back(
            {lp + "post_attention_layernorm.weight", "BF16", {H}, MakeBf16(H)});

        auto sa = lp + "self_attn.";
        t.push_back({sa + "q_proj.weight", "BF16", {QProjOut(), H},
                     MakeBf16(H * QProjOut())});
        t.push_back({sa + "k_proj.weight", "BF16", {KVProjOut(), H},
                     MakeBf16(H * KVProjOut())});
        t.push_back({sa + "v_proj.weight", "BF16", {KVProjOut(), H},
                     MakeBf16(H * KVProjOut())});
        t.push_back({sa + "o_proj.weight", "BF16", {H, QProjOut()},
                     MakeBf16(QProjOut() * H)});

        // MLP (SwiGLU, no bias).
        t.push_back({lp + "mlp.gate_proj.weight", "BF16", {I, H},
                     MakeBf16(H * I)});
        t.push_back({lp + "mlp.up_proj.weight", "BF16", {I, H},
                     MakeBf16(H * I)});
        t.push_back({lp + "mlp.down_proj.weight", "BF16", {H, I},
                     MakeBf16(I * H)});
    }

    return t;
}

// Writes a config.json matching BuildSyntheticGraniteModel so the converter's
// geometry inference reads authoritative values.
inline void WriteConfig(const std::filesystem::path &dir)
{
    const auto &c = TestCfg();
    std::string json =
        "{\"architectures\":[\"GraniteForCausalLM\"],\"model_type\":\"granite\","
        "\"hidden_size\":" + std::to_string(c.hidden_size) +
        ",\"intermediate_size\":" + std::to_string(c.intermediate_size) +
        ",\"num_hidden_layers\":" + std::to_string(c.num_layers) +
        ",\"vocab_size\":" + std::to_string(c.vocab_size) +
        ",\"num_attention_heads\":" + std::to_string(c.num_attention_heads) +
        ",\"num_key_value_heads\":" + std::to_string(c.num_key_value_heads) +
        ",\"head_dim\":" + std::to_string(c.head_dim) +
        ",\"rms_norm_eps\":" + std::to_string(c.rms_norm_eps) +
        ",\"rope_theta\":" + std::to_string(c.rope_theta) +
        ",\"attention_multiplier\":" + std::to_string(c.attention_multiplier) +
        ",\"embedding_multiplier\":1.0,\"residual_multiplier\":1.0,"
        "\"logits_scaling\":1.0,\"tie_word_embeddings\":false,"
        "\"hidden_act\":\"silu\",\"attention_bias\":false,\"mlp_bias\":false,"
        "\"bos_token_id\":" + std::to_string(c.bos_id) +
        ",\"eos_token_id\":" + std::to_string(c.eos_id) +
        ",\"pad_token_id\":" + std::to_string(c.pad_id) +
        ",\"max_position_embeddings\":131072}";
    std::ofstream f(dir / "config.json");
    f << json;
}

} // namespace grantest
