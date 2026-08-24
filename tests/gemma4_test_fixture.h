#pragma once

// Shared fixture builder for Gemma-4-26B-A4B converter / loader / kernel tests.
// Builds a synthetic gemma4 MoE safetensors model directory on disk
// (sliding_attention + full_attention layers, dense MLP + routed-expert MoE).
// Shapes mirror the real checkpoint proportionally:
//   sliding layer: q 16x256, kv 8x256  ->  here: q 4x16, kv 2x16
//   full layer:    q 16x512, kv 2x512  ->  here: q 4x32, kv 1x32

#include <absl/strings/str_cat.h>
#include <absl/strings/str_join.h>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

namespace gemmatest {

struct GemmaTestConfig {
    int hidden_size = 64;
    int intermediate_size = 24;       // dense MLP
    int vocab_size = 64;
    int num_layers = 4;
    std::vector<std::string> layer_types = {
        "sliding_attention", "sliding_attention", "sliding_attention",
        "full_attention"};

    // Attention.
    int num_attention_heads = 4;
    int num_key_value_heads = 2;      // sliding layers
    int head_dim = 16;
    int num_global_key_value_heads = 1; // full-attention layers
    int global_head_dim = 32;
    int sliding_window = 8;

    // MoE.
    int num_experts = 8;
    int top_k_experts = 2;
    int moe_intermediate_size = 16;

    double rms_norm_eps = 1e-6;
    double partial_rotary_factor = 0.25;
    double rope_theta_full = 1000000.0;
    double rope_theta_sliding = 10000.0;
};

inline const GemmaTestConfig &TestCfg()
{
    static GemmaTestConfig cfg;
    return cfg;
}

inline std::vector<uint8_t> MakeBf16(int64_t n)
{
    std::vector<uint8_t> out(static_cast<size_t>(n) * 2);
    for (int64_t i = 0; i < n; ++i) {
        uint16_t bits = static_cast<uint16_t>((i * 2654435761u) % 65536);
        out[static_cast<size_t>(i) * 2 + 0] = static_cast<uint8_t>(bits & 0xff);
        out[static_cast<size_t>(i) * 2 + 1] = static_cast<uint8_t>(bits >> 8);
    }
    return out;
}

inline std::vector<uint8_t> MakeF32Values(
    int64_t n, const std::function<float(int64_t)> &pattern)
{
    std::vector<uint8_t> out(static_cast<size_t>(n) * 4);
    for (int64_t i = 0; i < n; ++i) {
        float v = pattern(i);
        std::memcpy(out.data() + static_cast<size_t>(i) * 4, &v, 4);
    }
    return out;
}

// BF16 payload from an f32 pattern (round-to-nearest-even via bit shift is
// fine for the values used here).
inline std::vector<uint8_t> MakeBf16Values(
    int64_t n, const std::function<float(int64_t)> &pattern)
{
    std::vector<uint8_t> out(static_cast<size_t>(n) * 2);
    for (int64_t i = 0; i < n; ++i) {
        uint32_t u;
        float v = pattern(i);
        std::memcpy(&u, &v, 4);
        const uint16_t bits = static_cast<uint16_t>((u + 0x7fffU) >> 16);
        out[static_cast<size_t>(i) * 2 + 0] =
            static_cast<uint8_t>(bits & 0xff);
        out[static_cast<size_t>(i) * 2 + 1] =
            static_cast<uint8_t>(bits >> 8);
    }
    return out;
}

struct FakeTensor {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    std::vector<uint8_t> data;
};

inline void WriteSafetensors(const std::filesystem::path &dir,
                             const std::string &fname,
                             const std::vector<FakeTensor> &tensors)
{
    std::string header = R"({"__metadata__":{})";
    int64_t data_pos = 0;
    for (const auto &t : tensors) {
        std::string shape_s = absl::StrCat("[", absl::StrJoin(t.shape, ","), "]");
        int64_t begin = data_pos;
        int64_t end = data_pos + static_cast<int64_t>(t.data.size());
        absl::StrAppend(&header, ",\"", t.name, "\":{\"dtype\":\"", t.dtype,
                        "\",\"shape\":", shape_s, ",\"data_offsets\":[", begin,
                        ",", end, "]}");
        data_pos = end;
    }
    header += "}";
    while ((header.size() + 8) % 8 != 0)
        header += " ";

    std::ofstream f(dir / fname, std::ios::binary);
    uint64_t hlen = static_cast<uint64_t>(header.size());
    uint8_t lenbuf[8];
    for (int i = 0; i < 8; ++i)
        lenbuf[i] = static_cast<uint8_t>((hlen >> (8 * i)) & 0xff);
    f.write(reinterpret_cast<char *>(lenbuf), 8);
    f.write(header.data(), static_cast<std::streamsize>(header.size()));
    for (const auto &t : tensors)
        f.write(reinterpret_cast<const char *>(t.data.data()),
                static_cast<std::streamsize>(t.data.size()));
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

inline std::vector<FakeTensor> BuildSyntheticGemmaModel()
{
    const auto &c = TestCfg();
    const int H = c.hidden_size;
    const int I = c.intermediate_size;
    const int V = c.vocab_size;
    const int MI = c.moe_intermediate_size;
    const int E = c.num_experts;

    std::vector<FakeTensor> t;

    t.push_back({"model.language_model.embed_tokens.weight", "BF16", {V, H},
                 MakeBf16(V * H)});
    t.push_back({"model.language_model.norm.weight", "BF16", {H}, MakeBf16(H)});

    for (int L = 0; L < c.num_layers; ++L) {
        std::string lp = "model.language_model.layers." + std::to_string(L) + ".";
        bool full = c.layer_types[L] == "full_attention";

        t.push_back({lp + "input_layernorm.weight", "BF16", {H}, MakeBf16(H)});
        t.push_back({lp + "post_attention_layernorm.weight", "BF16", {H},
                     MakeBf16(H)});
        t.push_back({lp + "pre_feedforward_layernorm.weight", "BF16", {H},
                     MakeBf16(H)});
        t.push_back({lp + "post_feedforward_layernorm.weight", "BF16", {H},
                     MakeBf16(H)});
        t.push_back({lp + "post_feedforward_layernorm_1.weight", "BF16", {H},
                     MakeBf16(H)});
        t.push_back({lp + "pre_feedforward_layernorm_2.weight", "BF16", {H},
                     MakeBf16(H)});
        t.push_back({lp + "post_feedforward_layernorm_2.weight", "BF16", {H},
                     MakeBf16(H)});
        // Real checkpoints store layer_scalar / per_expert_scale as BF16;
        // keep the fixture faithful to that (regression: the loader used to
        // skip BF16 scalars due to an F32-sized guard).
        t.push_back({lp + "layer_scalar", "BF16", {1},
                     MakeBf16Values(1, [](int64_t) { return 1.0f; })});

        // Dense MLP branch.
        t.push_back({lp + "mlp.gate_proj.weight", "BF16", {I, H}, MakeBf16(I * H)});
        t.push_back({lp + "mlp.up_proj.weight", "BF16", {I, H}, MakeBf16(I * H)});
        t.push_back({lp + "mlp.down_proj.weight", "BF16", {H, I}, MakeBf16(H * I)});

        // MoE branch.
        t.push_back({lp + "experts.gate_up_proj", "BF16", {E, 2 * MI, H},
                     MakeBf16(E * 2 * MI * H)});
        t.push_back({lp + "experts.down_proj", "BF16", {E, H, MI},
                     MakeBf16(E * H * MI)});
        t.push_back({lp + "router.proj.weight", "BF16", {E, H}, MakeBf16(E * H)});
        t.push_back({lp + "router.scale", "BF16", {H}, MakeBf16(H)});
        t.push_back({lp + "router.per_expert_scale", "BF16", {E},
                     MakeBf16Values(E, [](int64_t i) {
                         return 0.5f + 0.01f * static_cast<float>(i);
                     })});

        // Attention: geometry depends on the layer type. Full-attention
        // layers tie V to K (attention_k_eq_v): no v_proj tensor.
        int q_rows = full ? FullQRows() : SlidingQRows();
        int kv_rows = full ? FullKVRows() : SlidingKVRows();
        auto sa = lp + "self_attn.";
        t.push_back({sa + "q_proj.weight", "BF16", {q_rows, H},
                     MakeBf16(q_rows * H)});
        t.push_back({sa + "k_proj.weight", "BF16", {kv_rows, H},
                     MakeBf16(kv_rows * H)});
        if (!full)
            t.push_back({sa + "v_proj.weight", "BF16", {kv_rows, H},
                         MakeBf16(kv_rows * H)});
        t.push_back({sa + "o_proj.weight", "BF16", {H, q_rows},
                     MakeBf16(H * q_rows)});
        int norm_len = full ? c.global_head_dim : c.head_dim;
        t.push_back({sa + "q_norm.weight", "BF16", {norm_len}, MakeBf16(norm_len)});
        t.push_back({sa + "k_norm.weight", "BF16", {norm_len}, MakeBf16(norm_len)});
    }

    // Vision fat — must be skipped by the converter.
    t.push_back({"model.vision_tower.patch_embedder.input_proj.weight", "BF16",
                 {H, H}, MakeBf16(H * H)});
    t.push_back({"model.embed_vision.embedding_projection.weight", "BF16",
                 {H, H}, MakeBf16(H * H)});
    t.push_back({"model.vision_tower.std_scale", "BF16", {1}, MakeBf16(1)});
    t.push_back({"model.vision_tower.std_bias", "BF16", {1}, MakeBf16(1)});

    return t;
}

inline void WriteConfig(const std::filesystem::path &dir)
{
    const auto &c = TestCfg();
    std::string layer_types = "[";
    for (size_t i = 0; i < c.layer_types.size(); ++i) {
        if (i) layer_types += ",";
        layer_types += "\"" + c.layer_types[i] + "\"";
    }
    layer_types += "]";

    std::string json =
        "{\"model_type\":\"gemma4\","
        "\"bos_token_id\":2,\"eos_token_id\":[1,106],\"pad_token_id\":0,"
        "\"text_config\":{"
        "\"model_type\":\"gemma4_text\","
        "\"hidden_size\":" + std::to_string(c.hidden_size) +
        ",\"intermediate_size\":" + std::to_string(c.intermediate_size) +
        ",\"moe_intermediate_size\":" + std::to_string(c.moe_intermediate_size) +
        ",\"num_hidden_layers\":" + std::to_string(c.num_layers) +
        ",\"vocab_size\":" + std::to_string(c.vocab_size) +
        ",\"num_attention_heads\":" + std::to_string(c.num_attention_heads) +
        ",\"num_key_value_heads\":" + std::to_string(c.num_key_value_heads) +
        ",\"num_global_key_value_heads\":" +
        std::to_string(c.num_global_key_value_heads) +
        ",\"head_dim\":" + std::to_string(c.head_dim) +
        ",\"global_head_dim\":" + std::to_string(c.global_head_dim) +
        ",\"sliding_window\":" + std::to_string(c.sliding_window) +
        ",\"num_experts\":" + std::to_string(c.num_experts) +
        ",\"top_k_experts\":" + std::to_string(c.top_k_experts) +
        ",\"enable_moe_block\":true"
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
}

} // namespace gemmatest
