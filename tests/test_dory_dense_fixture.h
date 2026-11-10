#pragma once

// Shared fixture builder for the Dory1.0-4B converter / loader / model / kernel
// tests. Tensor-writing primitives live in the family-neutral
// test_synth_checkpoint.h (suite-autonomy invariant) and are re-exported here.
//
// The synthetic geometry mirrors the REAL layer pattern at a tiny scale:
// mixers alternate attention (even layer) / mlp (odd layer), the stack is
// partitioned input / recurrent / output, the recurrent block is applied twice,
// `full_attention` sits at layer % 8 == 6, and attention is RECTANGULAR
// (num_heads*head_dim != hidden_size) exactly like the checkpoint.

#include "test_synth_checkpoint.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace dorytest {

using synthtest::FakeTensor;
using synthtest::MakeBf16;
using synthtest::MakeBf16Values;
using synthtest::MakeF32Values;
using synthtest::TensorNamesIn;
using synthtest::WriteSafetensors;

struct DoryTestConfig {
    int hidden_size = 64;
    int intermediate_size = 128;
    int vocab_size = 64;
    int num_layers = 8;
    int num_attention_heads = 4;
    int num_key_value_heads = 2;
    int head_dim = 8;             // 4*8 = 32 != 64: rectangular attention
    int input_layers = 2;
    int recurrent_layers = 4;
    int output_layers = 2;
    int recurrent_loops = 2;
    int sliding_window = 4;
    double rope_theta_sliding = 100.0;
    double rope_theta_full = 10000.0;
    double initializer_range = 0.02;
    double ngpt_alpha_init = 0.05;
    int bos_id = 0;
    int eos_id = 1;
    int pad_id = 2;
};

inline const DoryTestConfig &TestCfg()
{
    static DoryTestConfig cfg;
    return cfg;
}

inline int QSize() { return TestCfg().num_attention_heads * TestCfg().head_dim; }
inline int KVSize() { return TestCfg().num_key_value_heads * TestCfg().head_dim; }

inline bool LayerIsAttention(int layer) { return (layer & 1) == 0; }
inline bool LayerIsFull(int layer) { return layer % 8 == 6; }
inline bool LayerIsRecurrent(int layer)
{
    const auto &c = TestCfg();
    return layer >= c.input_layers && layer < c.input_layers + c.recurrent_layers;
}

inline int CountKvSlots()
{
    const auto &c = TestCfg();
    int n = 0;
    for (int l = 0; l < c.num_layers; ++l) {
        if (!LayerIsAttention(l)) continue;
        n += LayerIsRecurrent(l) ? c.recurrent_loops : 1;
    }
    return n;
}

inline std::vector<std::string> LayerTypes()
{
    const auto &c = TestCfg();
    std::vector<std::string> types;
    for (int l = 0; l < c.num_layers; ++l) {
        if (!LayerIsAttention(l)) types.push_back("mlp");
        else if (LayerIsFull(l)) types.push_back("full_attention");
        else types.push_back("sliding_attention");
    }
    return types;
}

// Small deterministic non-degenerate values: the nGPT vectors must not be zero
// (they are squared and normalized against).
inline float Pattern(int64_t i, float scale)
{
    return scale * (0.5f + 0.5f * static_cast<float>((i * 37) % 23) / 22.0f);
}

inline std::vector<FakeTensor> BuildSyntheticDoryModel()
{
    const auto &c = TestCfg();
    const int H = c.hidden_size;
    const int I = c.intermediate_size;
    const int V = c.vocab_size;
    const int Q = QSize();
    const int KV = KVSize();

    std::vector<FakeTensor> t;
    t.push_back({"backbone.embeddings.weight", "BF16", {V, H},
                 MakeBf16Values(V * H, [](int64_t i) {
                     return Pattern(i, 0.05f);
                 })});
    t.push_back({"lm_head.weight", "BF16", {V, H},
                 MakeBf16Values(V * H, [](int64_t i) {
                     return Pattern(i + 7, 0.05f);
                 })});
    t.push_back({"lm_head.logit_scale", "BF16", {V},
                 MakeBf16Values(V, [](int64_t i) {
                     return Pattern(i, 0.03f);
                 })});

    for (int l = 0; l < c.num_layers; ++l) {
        const std::string lp = "backbone.layers." + std::to_string(l) + ".";
        if (LayerIsAttention(l)) {
            const std::string mx = lp + "mixer.";
            t.push_back({mx + "q_proj.weight", "BF16", {Q, H},
                         MakeBf16Values(static_cast<int64_t>(Q) * H,
                                        [](int64_t i) {
                                            return Pattern(i, 0.04f);
                                        })});
            t.push_back({mx + "k_proj.weight", "BF16", {KV, H},
                         MakeBf16Values(static_cast<int64_t>(KV) * H,
                                        [](int64_t i) {
                                            return Pattern(i + 3, 0.04f);
                                        })});
            t.push_back({mx + "v_proj.weight", "BF16", {KV, H},
                         MakeBf16Values(static_cast<int64_t>(KV) * H,
                                        [](int64_t i) {
                                            return Pattern(i + 5, 0.04f);
                                        })});
            t.push_back({mx + "o_proj.weight", "BF16", {H, Q},
                         MakeBf16Values(static_cast<int64_t>(H) * Q,
                                        [](int64_t i) {
                                            return Pattern(i + 11, 0.04f);
                                        })});
            t.push_back({mx + "g_proj.weight", "BF16", {Q, H},
                         MakeBf16Values(static_cast<int64_t>(Q) * H,
                                        [](int64_t i) {
                                            return Pattern(i + 13, 0.04f);
                                        })});
            t.push_back({mx + "sqk", "BF16", {Q},
                         MakeBf16Values(Q, [](int64_t i) {
                             return Pattern(i, 0.02f);
                         })});
            t.push_back({mx + "s_gate", "BF16", {Q},
                         MakeBf16Values(Q, [](int64_t i) {
                             return Pattern(i + 1, 0.02f);
                         })});
            t.push_back({lp + "attn_alpha", "BF16", {H},
                         MakeBf16Values(H, [](int64_t i) {
                             return Pattern(i, 0.05f);
                         })});
        } else {
            const std::string mx = lp + "mixer.";
            t.push_back({mx + "gate_proj.weight", "BF16", {I, H},
                         MakeBf16Values(static_cast<int64_t>(I) * H,
                                        [](int64_t i) {
                                            return Pattern(i, 0.04f);
                                        })});
            t.push_back({mx + "up_proj.weight", "BF16", {I, H},
                         MakeBf16Values(static_cast<int64_t>(I) * H,
                                        [](int64_t i) {
                                            return Pattern(i + 2, 0.04f);
                                        })});
            t.push_back({mx + "down_proj.weight", "BF16", {H, I},
                         MakeBf16Values(static_cast<int64_t>(H) * I,
                                        [](int64_t i) {
                                            return Pattern(i + 9, 0.04f);
                                        })});
            t.push_back({mx + "suv_gate", "BF16", {I},
                         MakeBf16Values(I, [](int64_t i) {
                             return Pattern(i, 0.02f);
                         })});
            t.push_back({mx + "suv_up", "BF16", {I},
                         MakeBf16Values(I, [](int64_t i) {
                             return Pattern(i + 4, 0.02f);
                         })});
            t.push_back({lp + "mlp_alpha", "BF16", {H},
                         MakeBf16Values(H, [](int64_t i) {
                             return Pattern(i + 6, 0.05f);
                         })});
        }
    }
    return t;
}

inline void WriteConfig(const std::filesystem::path &dir)
{
    const auto &c = TestCfg();
    std::ofstream f(dir / "config.json");
    f << "{\n"
      << "  \"architectures\": [\"DoryForCausalLM\"],\n"
      << "  \"model_type\": \"dory\",\n"
      << "  \"vocab_size\": " << c.vocab_size << ",\n"
      << "  \"hidden_size\": " << c.hidden_size << ",\n"
      << "  \"num_hidden_layers\": " << c.num_layers << ",\n"
      << "  \"intermediate_size\": " << c.intermediate_size << ",\n"
      << "  \"num_attention_heads\": " << c.num_attention_heads << ",\n"
      << "  \"num_key_value_heads\": " << c.num_key_value_heads << ",\n"
      << "  \"head_dim\": " << c.head_dim << ",\n"
      << "  \"max_position_embeddings\": 128,\n"
      << "  \"tie_word_embeddings\": false,\n"
      << "  \"ngpt_alpha_init\": " << c.ngpt_alpha_init << ",\n"
      << "  \"attention_output_gate_logit_scale_init\": null,\n"
      << "  \"mlp_bias\": false,\n"
      << "  \"attention_bias\": false,\n"
      << "  \"n_input_layers\": " << c.input_layers << ",\n"
      << "  \"n_recurrent_layers\": " << c.recurrent_layers << ",\n"
      << "  \"n_output_layers\": " << c.output_layers << ",\n"
      << "  \"n_recurrent_loops\": " << c.recurrent_loops << ",\n"
      << "  \"recurrent_kv_cache_mode\": \"per_loop\",\n"
      << "  \"hidden_act\": \"silu\",\n"
      << "  \"initializer_range\": " << c.initializer_range << ",\n"
      << "  \"dtype\": \"bfloat16\",\n"
      << "  \"sliding_window\": " << c.sliding_window << ",\n"
      << "  \"rope_parameters\": {\n"
      << "    \"sliding_attention\": {\"rope_type\": \"default\", "
         "\"rope_theta\": "
      << c.rope_theta_sliding << ", \"partial_rotary_factor\": 1.0},\n"
      << "    \"full_attention\": {\"rope_type\": \"default\", "
         "\"rope_theta\": "
      << c.rope_theta_full << ", \"partial_rotary_factor\": 1.0}\n"
      << "  },\n"
      << "  \"layer_types\": [";
    const auto types = LayerTypes();
    for (size_t i = 0; i < types.size(); ++i) {
        if (i) f << ", ";
        f << "\"" << types[i] << "\"";
    }
    f << "]\n}\n";

    std::ofstream g(dir / "generation_config.json");
    g << "{\n"
      << "  \"bos_token_id\": " << c.bos_id << ",\n"
      << "  \"eos_token_id\": [" << (c.eos_id - 1) << ", " << c.eos_id
      << "],\n"
      << "  \"pad_token_id\": " << c.pad_id << ",\n"
      << "  \"do_sample\": true,\n"
      << "  \"temperature\": 1.0\n}\n";
}

} // namespace dorytest
