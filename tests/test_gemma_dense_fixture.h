#pragma once

// Synthetic gemma-4-31B-it (DENSE) checkpoint fixture.
//
// Family-specific geometry only: the byte-pattern / safetensors primitives are
// the model-neutral ones from test_synth_checkpoint.h (test-suite autonomy
// invariant - a family fixture never borrows another family's fixture).
//
// Shapes mirror the real checkpoint proportionally:
//   sliding layer: q 32x256, kv 16x256, o 32x256 -> here: q 4x16, kv 2x16
//   full layer:    q 32x512, kv  4x512, o 32x512 -> here: q 4x64, kv 1x64
// and there is NO v_proj on a full layer (attention_k_eq_v ties V to K), NO
// routed expert, and NO router - the dense family.

#include <absl/strings/str_cat.h>
#include <absl/strings/str_join.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "test_synth_checkpoint.h"

namespace gemma_dense_test {

struct GemmaDenseTestConfig {
    int hidden_size = 64;
    int intermediate_size = 32;  // dense MLP (32-multiple: NVFP4 GEMV)
    int vocab_size = 64;
    int num_layers = 4;
    std::vector<std::string> layer_types = {
        "sliding_attention", "sliding_attention", "sliding_attention",
        "full_attention"};

    // Attention.
    int num_attention_heads = 4;
    int num_key_value_heads = 2;          // sliding layers
    int head_dim = 16;
    int num_global_key_value_heads = 1;   // full-attention layers
    // 1 kv head x 64 = 64 KV rows, deliberately DIFFERENT from the sliding
    // layers' 2 x 16 = 32: the cache geometry per layer kind is part of the
    // contract this suite pins.
    int global_head_dim = 64;
    int sliding_window = 8;

    double rms_norm_eps = 1e-6;
    double partial_rotary_factor = 0.25;
    double rope_theta_full = 1000000.0;
    double rope_theta_sliding = 10000.0;
    double final_logit_softcapping = 30.0;
};

inline const GemmaDenseTestConfig &TestCfg()
{
    static GemmaDenseTestConfig cfg;
    return cfg;
}

inline int SlidingQRows()
{
    return TestCfg().num_attention_heads * TestCfg().head_dim;
}
inline int SlidingKVRows()
{
    return TestCfg().num_key_value_heads * TestCfg().head_dim;
}
inline int FullQRows()
{
    return TestCfg().num_attention_heads * TestCfg().global_head_dim;
}
inline int FullKVRows()
{
    return TestCfg().num_global_key_value_heads * TestCfg().global_head_dim;
}

// Every text tensor of the synthetic checkpoint, in the real safetensors
// layout (single shard; the converter is shard-agnostic).
inline std::vector<synthtest::FakeTensor> BuildSyntheticGemmaDenseModel()
{
    const auto &c = TestCfg();
    const int H = c.hidden_size;
    const int I = c.intermediate_size;
    const int V = c.vocab_size;

    std::vector<synthtest::FakeTensor> t;

    t.push_back({"model.language_model.embed_tokens.weight", "BF16", {V, H},
                 synthtest::MakeBf16(V * H)});
    t.push_back({"model.language_model.norm.weight", "BF16", {H},
                 synthtest::MakeBf16(H)});

    for (int L = 0; L < c.num_layers; ++L) {
        const std::string lp =
            "model.language_model.layers." + std::to_string(L) + ".";
        const bool full = c.layer_types[static_cast<size_t>(L)] ==
                          "full_attention";

        t.push_back({lp + "input_layernorm.weight", "BF16", {H},
                     synthtest::MakeBf16(H)});
        t.push_back({lp + "post_attention_layernorm.weight", "BF16", {H},
                     synthtest::MakeBf16(H)});
        t.push_back({lp + "pre_feedforward_layernorm.weight", "BF16", {H},
                     synthtest::MakeBf16(H)});
        t.push_back({lp + "post_feedforward_layernorm.weight", "BF16", {H},
                     synthtest::MakeBf16(H)});
        // BF16 in the real checkpoint (a regression class of its own).
        t.push_back({lp + "layer_scalar", "BF16", {1},
                     synthtest::MakeBf16Values(1, [](int64_t) { return 1.0f; })});

        t.push_back({lp + "mlp.gate_proj.weight", "BF16", {I, H},
                     synthtest::MakeBf16(I * H)});
        t.push_back({lp + "mlp.up_proj.weight", "BF16", {I, H},
                     synthtest::MakeBf16(I * H)});
        t.push_back({lp + "mlp.down_proj.weight", "BF16", {H, I},
                     synthtest::MakeBf16(H * I)});

        const int q_rows = full ? FullQRows() : SlidingQRows();
        const int kv_rows = full ? FullKVRows() : SlidingKVRows();
        const std::string sa = lp + "self_attn.";
        t.push_back({sa + "q_proj.weight", "BF16", {q_rows, H},
                     synthtest::MakeBf16(q_rows * H)});
        t.push_back({sa + "k_proj.weight", "BF16", {kv_rows, H},
                     synthtest::MakeBf16(kv_rows * H)});
        if (!full)  // attention_k_eq_v: global layers have no v_proj
            t.push_back({sa + "v_proj.weight", "BF16", {kv_rows, H},
                         synthtest::MakeBf16(kv_rows * H)});
        t.push_back({sa + "o_proj.weight", "BF16", {H, q_rows},
                     synthtest::MakeBf16(H * q_rows)});
        const int norm_len = full ? c.global_head_dim : c.head_dim;
        t.push_back({sa + "q_norm.weight", "BF16", {norm_len},
                     synthtest::MakeBf16(norm_len)});
        t.push_back({sa + "k_norm.weight", "BF16", {norm_len},
                     synthtest::MakeBf16(norm_len)});
    }

    // Vision fat - stored-and-counted, never applied.
    t.push_back({"model.vision_tower.patch_embedder.input_proj.weight", "BF16",
                 {H, H}, synthtest::MakeBf16(H * H)});
    t.push_back({"model.embed_vision.embedding_projection.weight", "BF16",
                 {H, H}, synthtest::MakeBf16(H * H)});
    t.push_back({"model.vision_tower.std_scale", "BF16", {1},
                 synthtest::MakeBf16(1)});
    t.push_back({"model.vision_tower.std_bias", "BF16", {1},
                 synthtest::MakeBf16(1)});

    return t;
}

// config.json in the real HF nesting (model_type gemma4 + text_config).
inline void WriteConfig(const std::filesystem::path &dir)
{
    const auto &c = TestCfg();
    std::string layer_types = "[";
    for (size_t i = 0; i < c.layer_types.size(); ++i) {
        if (i) layer_types += ",";
        layer_types += "\"" + c.layer_types[i] + "\"";
    }
    layer_types += "]";

    const std::string json =
        "{\"model_type\":\"gemma4\","
        "\"bos_token_id\":2,\"eos_token_id\":[1,106],\"pad_token_id\":0,"
        "\"text_config\":{"
        "\"model_type\":\"gemma4_text\","
        "\"hidden_size\":" + std::to_string(c.hidden_size) +
        ",\"intermediate_size\":" + std::to_string(c.intermediate_size) +
        ",\"num_hidden_layers\":" + std::to_string(c.num_layers) +
        ",\"vocab_size\":" + std::to_string(c.vocab_size) +
        ",\"num_attention_heads\":" + std::to_string(c.num_attention_heads) +
        ",\"num_key_value_heads\":" + std::to_string(c.num_key_value_heads) +
        ",\"num_global_key_value_heads\":" +
        std::to_string(c.num_global_key_value_heads) +
        ",\"head_dim\":" + std::to_string(c.head_dim) +
        ",\"global_head_dim\":" + std::to_string(c.global_head_dim) +
        ",\"sliding_window\":" + std::to_string(c.sliding_window) +
        ",\"attention_k_eq_v\":true"
        ",\"enable_moe_block\":false"
        ",\"rms_norm_eps\":1e-06"
        ",\"final_logit_softcapping\":30.0"
        ",\"tie_word_embeddings\":true"
        ",\"hidden_activation\":\"gelu_pytorch_tanh\""
        ",\"layer_types\":" + layer_types +
        ",\"rope_parameters\":{"
        "\"full_attention\":{\"partial_rotary_factor\":0.25,"
        "\"rope_theta\":1000000.0,\"rope_type\":\"proportional\"},"
        "\"sliding_attention\":{\"rope_theta\":10000.0,"
        "\"rope_type\":\"default\"}}"
        "}}";
    std::ofstream f(dir / "config.json");
    f << json;

    // The real checkpoint ends a turn on 106 and the tool channel on 50.
    std::ofstream g(dir / "generation_config.json");
    g << "{\"eos_token_id\":[1,106,50]}";
}

} // namespace gemma_dense_test
