#pragma once

// Shared fixture builder for phi-4 converter / loader / kernel / model tests.
// Tensor-writing primitives live in the family-neutral test_synth_checkpoint.h
// (suite-autonomy invariant) and are re-exported under phitest.

#include "test_synth_checkpoint.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace phitest {

using synthtest::FakeTensor;
using synthtest::MakeBf16;
using synthtest::MakeBf16Values;
using synthtest::MakeF32Values;
using synthtest::TensorNamesIn;
using synthtest::WriteSafetensors;

// Synthetic geometry — tiny so CPU/CUDA references run fast. Mirror the real
// model proportionally (GQA 4:1, head_dim = hidden / heads) and store the
// projections PRE-FUSED exactly like the real phi-4 checkpoint.
struct PhiTestConfig {
    int hidden_size = 64;
    int intermediate_size = 128;
    int vocab_size = 64;
    int num_layers = 2;
    int num_attention_heads = 4;
    int num_key_value_heads = 2;
    int head_dim = 16;  // hidden / num_attention_heads

    double rms_norm_eps = 1e-5;
    double rope_theta = 10000.0;
    double attention_scale = 0.25;  // 1 / sqrt(head_dim)

    int bos_id = 0;
    int eos_id = 1;
    int pad_id = 2;
};

inline const PhiTestConfig &TestCfg()
{
    static PhiTestConfig cfg;
    return cfg;
}

inline int QProjOut() { return TestCfg().num_attention_heads * TestCfg().head_dim; }
inline int KVProjOut() { return TestCfg().num_key_value_heads * TestCfg().head_dim; }

inline std::vector<FakeTensor> BuildSyntheticPhiModel()
{
    const auto &c = TestCfg();
    const int H = c.hidden_size;
    const int I = c.intermediate_size;
    const int V = c.vocab_size;
    const int QKV = QProjOut() + 2 * KVProjOut();

    std::vector<FakeTensor> t;

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
        // Fused qkv (q | k | v), like the real checkpoint.
        t.push_back({sa + "qkv_proj.weight", "BF16", {QKV, H},
                     MakeBf16(QKV * H)});
        t.push_back({sa + "o_proj.weight", "BF16", {H, QProjOut()},
                     MakeBf16(QProjOut() * H)});

        // Fused gate_up (gate | up), like the real checkpoint.
        t.push_back({lp + "mlp.gate_up_proj.weight", "BF16", {2 * I, H},
                     MakeBf16(2 * I * H)});
        t.push_back({lp + "mlp.down_proj.weight", "BF16", {H, I},
                     MakeBf16(I * H)});
    }

    return t;
}

// Writes a config.json matching BuildSyntheticPhiModel so the converter's
// geometry inference reads authoritative values.
inline void WriteConfig(const std::filesystem::path &dir)
{
    const auto &c = TestCfg();
    std::string json =
        "{\"architectures\":[\"Phi3ForCausalLM\"],\"model_type\":\"phi3\","
        "\"hidden_size\":" + std::to_string(c.hidden_size) +
        ",\"intermediate_size\":" + std::to_string(c.intermediate_size) +
        ",\"num_hidden_layers\":" + std::to_string(c.num_layers) +
        ",\"vocab_size\":" + std::to_string(c.vocab_size) +
        ",\"num_attention_heads\":" + std::to_string(c.num_attention_heads) +
        ",\"num_key_value_heads\":" + std::to_string(c.num_key_value_heads) +
        ",\"head_dim\":" + std::to_string(c.head_dim) +
        ",\"rms_norm_eps\":" + std::to_string(c.rms_norm_eps) +
        ",\"rope_theta\":" + std::to_string(c.rope_theta) +
        ",\"tie_word_embeddings\":false,\"hidden_act\":\"silu\","
        "\"attention_bias\":false,\"attention_dropout\":0.0,"
        "\"bos_token_id\":" + std::to_string(c.bos_id) +
        ",\"eos_token_id\":" + std::to_string(c.eos_id) +
        ",\"pad_token_id\":" + std::to_string(c.pad_id) +
        ",\"max_position_embeddings\":16384}";
    std::ofstream f(dir / "config.json");
    f << json;
}

} // namespace phitest
