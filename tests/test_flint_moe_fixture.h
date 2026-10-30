#pragma once

// RecursiveSI/flint synthetic-checkpoint fixture (converter/loader/model
// tests). Builds the family-neutral primitives from test_synth_checkpoint.h;
// geometry is deliberately tiny so the whole pipeline runs in a bounded
// TIMEOUT, and the frozen passport is relaxed with
// LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY.
//
// The fixture keeps the three things that make Flint a family of its own even
// at toy size: a hybrid `layer_types` table with a no-RoPE full-attention
// layer, the per-head `softmax_offset` sink, and the dense-prefix + MoE +
// ungated-shared-expert split.
#include "test_synth_checkpoint.h"

#include <absl/strings/str_cat.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace flint_test {

struct Cfg {
    int hidden = 64;
    int layers = 4;
    int heads = 2;
    int kv_heads = 1;
    int head_dim = 32;
    int vocab = 64;
    int dense_inter = 32;
    int moe_inter = 16;
    int shared_inter = 16;
    int experts = 8;
    int top_k = 2;
    int window = 8;
    int first_moe_layer = 1;   // mlp_only_layers = {0}
    int full_attn_layer = 3;   // also the no-RoPE layer
    int bos = 268, eos = 269, eos2 = 271, pad = 269;
};

inline const Cfg &TestCfg()
{
    static const Cfg c;
    return c;
}

inline std::string LayerTypesJson()
{
    const Cfg &c = TestCfg();
    std::string s = "[";
    for (int l = 0; l < c.layers; ++l) {
        if (l) s += ",";
        s += (l == c.full_attn_layer) ? "\"full_attention\""
                                      : "\"sliding_attention\"";
    }
    s += "]";
    return s;
}

inline std::string MlpTypesJson()
{
    const Cfg &c = TestCfg();
    std::string s = "[";
    for (int l = 0; l < c.layers; ++l) {
        if (l) s += ",";
        s += (l < c.first_moe_layer) ? "\"dense\"" : "\"sparse\"";
    }
    s += "]";
    return s;
}

inline void WriteConfig(const std::filesystem::path &dir)
{
    const Cfg &c = TestCfg();
    std::ofstream f(dir / "config.json");
    f << R"({"architectures":["FlintForCausalLM"],"model_type":"flint",)"
      << "\"hidden_size\":" << c.hidden
      << ",\"intermediate_size\":" << c.dense_inter
      << ",\"num_hidden_layers\":" << c.layers
      << ",\"num_attention_heads\":" << c.heads
      << ",\"num_key_value_heads\":" << c.kv_heads
      << ",\"head_dim\":" << c.head_dim
      << ",\"vocab_size\":" << c.vocab
      << ",\"rms_norm_eps\":1e-06,\"rope_theta\":10000.0"
      << ",\"max_position_embeddings\":2048"
      << ",\"hidden_act\":\"silu\",\"attention_bias\":false"
      << ",\"tie_word_embeddings\":false"
      << ",\"softmax_type\":\"learnable\",\"router_score_function\":\"sigmoid\""
      << ",\"norm_topk_prob\":true"
      << ",\"use_sliding_window\":true,\"sliding_window\":" << c.window
      << ",\"moe_intermediate_size\":" << c.moe_inter
      << ",\"shared_expert_intermediate_size\":" << c.shared_inter
      << ",\"num_experts\":" << c.experts
      << ",\"num_experts_per_tok\":" << c.top_k
      << ",\"mlp_only_layers\":[0]"
      << ",\"no_rope_layer_indices\":[" << c.full_attn_layer << "]"
      << ",\"layer_types\":" << LayerTypesJson()
      << ",\"bos_token_id\":" << c.bos
      << ",\"eos_token_id\":[" << c.eos << "," << c.eos2 << "]"
      << ",\"pad_token_id\":" << c.pad << "}\n";
    f.close();
}

inline std::vector<synthtest::FakeTensor> Tensors()
{
    const Cfg &c = TestCfg();
    const int H = c.hidden;
    const int QW = c.heads * c.head_dim;
    const int KVW = c.kv_heads * c.head_dim;
    std::vector<synthtest::FakeTensor> t;
    auto bf16 = [](const std::string &name, std::vector<int64_t> shape,
                   int64_t seed) {
        int64_t n = 1;
        for (int64_t d : shape) n *= d;
        synthtest::FakeTensor f;
        f.name = name;
        f.dtype = "BF16";
        f.shape = std::move(shape);
        f.data = synthtest::MakeBf16Values(
            n, [seed](int64_t i) {
                const int64_t k = (i * 7 + seed * 13) % 23;
                return static_cast<float>((k - 11)) * 0.011f;
            });
        return f;
    };

    t.push_back(bf16("model.embed_tokens.weight", {c.vocab, H}, 1));
    t.push_back(bf16("lm_head.weight", {c.vocab, H}, 2));
    t.push_back(bf16("model.norm.weight", {H}, 3));

    for (int l = 0; l < c.layers; ++l) {
        const std::string p = absl::StrCat("model.layers.", l, ".");
        t.push_back(bf16(p + "input_layernorm.weight", {H}, 10 + l));
        t.push_back(bf16(p + "post_attention_layernorm.weight", {H}, 20 + l));
        t.push_back(bf16(p + "self_attn.q_proj.weight", {QW, H}, 30 + l));
        t.push_back(bf16(p + "self_attn.k_proj.weight", {KVW, H}, 40 + l));
        t.push_back(bf16(p + "self_attn.v_proj.weight", {KVW, H}, 50 + l));
        t.push_back(bf16(p + "self_attn.o_proj.weight", {H, QW}, 60 + l));
        t.push_back(bf16(p + "self_attn.q_norm.weight", {c.head_dim}, 70 + l));
        t.push_back(bf16(p + "self_attn.k_norm.weight", {c.head_dim}, 80 + l));
        t.push_back(bf16(p + "self_attn.softmax_offset", {c.heads}, 90 + l));
        if (l < c.first_moe_layer) {
            t.push_back(bf16(p + "mlp.gate_proj.weight", {c.dense_inter, H},
                             130 + l));
            t.push_back(bf16(p + "mlp.up_proj.weight", {c.dense_inter, H},
                             140 + l));
            t.push_back(bf16(p + "mlp.down_proj.weight", {H, c.dense_inter},
                             150 + l));
        } else {
            t.push_back(bf16(p + "mlp.gate.weight", {c.experts, H}, 160 + l));
            t.push_back(bf16(p + "mlp.shared_expert.gate_proj.weight",
                             {c.shared_inter, H}, 180 + l));
            t.push_back(bf16(p + "mlp.shared_expert.up_proj.weight",
                             {c.shared_inter, H}, 190 + l));
            t.push_back(bf16(p + "mlp.shared_expert.down_proj.weight",
                             {H, c.shared_inter}, 200 + l));
            for (int e = 0; e < c.experts; ++e) {
                const std::string pe = absl::StrCat(p, "mlp.experts.", e, ".");
                t.push_back(bf16(pe + "gate_proj.weight", {c.moe_inter, H},
                                 1000 + e * 3 + l));
                t.push_back(bf16(pe + "down_proj.weight", {H, c.moe_inter},
                                 1001 + e * 3 + l));
                t.push_back(bf16(pe + "up_proj.weight", {c.moe_inter, H},
                                 1002 + e * 3 + l));
            }
        }
    }
    return t;
}

// Two shards so the converter's cross-shard expert join is exercised.
inline int BuildSyntheticCheckpoint(const std::filesystem::path &dir,
                                    bool with_tokenizer = false,
                                    const std::filesystem::path &tok_fixture = {})
{
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    WriteConfig(dir);
    if (with_tokenizer) {
        std::filesystem::copy_file(
            tok_fixture, dir / "tokenizer.json",
            std::filesystem::copy_options::overwrite_existing, ec);
    }
    auto all = Tensors();
    std::vector<synthtest::FakeTensor> a, b;
    int64_t bytes_a = 0;
    const int64_t kSplit = 64 * 1024;
    for (auto &t : all) {
        if (bytes_a < kSplit) {
            bytes_a += static_cast<int64_t>(t.data.size());
            a.push_back(std::move(t));
        } else {
            b.push_back(std::move(t));
        }
    }
    if (!a.empty()) synthtest::WriteSafetensors(dir, "model-00001.safetensors", a);
    if (!b.empty()) synthtest::WriteSafetensors(dir, "model-00002.safetensors", b);
    return static_cast<int>(all.size());
}

} // namespace flint_test
