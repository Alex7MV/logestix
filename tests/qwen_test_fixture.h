#pragma once

// Shared fixture builder for Qwen3.5-9B converter / loader / kernel tests.
// Builds a synthetic Qwen3-Next hybrid safetensors model directory on disk
// (mixed linear-attention + full-attention layers). Mirrors glm_test_fixture.h.

#include <lsxcommon/schema.h>
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/ipc/writer.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

namespace qwentest {

// Synthetic geometry — kept tiny so CUDA kernels and CPU FP64 references run
// fast. Shapes mirror the real 9B model proportionally.
struct QwenTestConfig {
    int hidden_size = 64;
    int intermediate_size = 128;
    int vocab_size = 64;
    int num_layers = 4;

    // Hybrid block types (index == layer idx).
    std::vector<std::string> layer_types = {
        "linear_attention", "linear_attention", "linear_attention", "full_attention"};

    // Linear attention (Qwen3_5GatedDeltaNet).
    int linear_num_key_heads   = 2;
    int linear_num_value_heads = 4;
    int linear_key_head_dim    = 8;
    int linear_value_head_dim  = 8;
    int linear_conv_kernel_dim = 4;

    // Full attention (Qwen3_5Attention).
    int num_attention_heads = 2;
    int num_key_value_heads = 1;
    int head_dim = 32;   // hidden / num_attention_heads
    int rotary_dim = 8;  // (int)(0.25 * head_dim)

    double rms_norm_eps       = 1e-6;
    double partial_rotary_factor = 0.25;
    double rope_theta          = 10000.0;
    std::string model_name     = "qwen3.5-9b";
};

inline const QwenTestConfig &TestCfg()
{
    static QwenTestConfig cfg;
    return cfg;
}

// Derived geometry.
inline int KeyDim()  { return TestCfg().linear_num_key_heads * TestCfg().linear_key_head_dim; }
inline int ValueDim(){ return TestCfg().linear_num_value_heads * TestCfg().linear_value_head_dim; }
inline int ConvDim() { return KeyDim() * 2 + ValueDim(); }
inline int QProjOut(){ return TestCfg().num_attention_heads * TestCfg().head_dim * 2; }
inline int KVProjOut(){ return TestCfg().num_key_value_heads * TestCfg().head_dim; }

inline std::vector<uint8_t> MakeBf16(int64_t n)
{
    std::vector<uint8_t> out(static_cast<size_t>(n) * 2);
    for (int64_t i = 0; i < n; ++i) {
        uint16_t bits = static_cast<uint16_t>(i % 65536);
        out[static_cast<size_t>(i) * 2 + 0] = static_cast<uint8_t>(bits & 0xff);
        out[static_cast<size_t>(i) * 2 + 1] = static_cast<uint8_t>(bits >> 8);
    }
    return out;
}

inline std::vector<uint8_t> MakeBf16Values(
    int64_t n, const std::function<float(int64_t)> &pattern)
{
    std::vector<uint8_t> out(static_cast<size_t>(n) * 2);
    for (int64_t i = 0; i < n; ++i) {
        float v = pattern(i);
        uint32_t u;
        std::memcpy(&u, &v, sizeof(u));
        if ((u & 0x7f800000U) == 0x7f800000U) {
            uint16_t b = static_cast<uint16_t>(u >> 16);
            out[static_cast<size_t>(i) * 2 + 0] = static_cast<uint8_t>(b & 0xff);
            out[static_cast<size_t>(i) * 2 + 1] = static_cast<uint8_t>(b >> 8);
            continue;
        }
        uint32_t rounding_bias = 0x7fffU + ((u >> 16) & 1U);
        u += rounding_bias;
        uint16_t b = static_cast<uint16_t>(u >> 16);
        out[static_cast<size_t>(i) * 2 + 0] = static_cast<uint8_t>(b & 0xff);
        out[static_cast<size_t>(i) * 2 + 1] = static_cast<uint8_t>(b >> 8);
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
    std::vector<std::pair<int64_t, int64_t>> offsets;
    for (const auto &t : tensors) {
        std::string shape_s = "[";
        for (size_t i = 0; i < t.shape.size(); ++i) {
            if (i) shape_s += ",";
            shape_s += std::to_string(t.shape[i]);
        }
        shape_s += "]";
        int64_t begin = data_pos;
        int64_t end = data_pos + static_cast<int64_t>(t.data.size());
        header += ",\"" + t.name + "\":{\"dtype\":\"" + t.dtype +
                  "\",\"shape\":" + shape_s +
                  ",\"data_offsets\":[" + std::to_string(begin) + "," +
                  std::to_string(end) + "]}";
        data_pos = end;
        offsets.push_back({begin, end});
    }
    header += "}";
    while ((header.size() + 8) % 8 != 0)
        header += " ";

    std::ofstream f(dir / fname, std::ios::binary);
    uint64_t hlen = static_cast<uint64_t>(header.size());
    uint8_t lenbuf[8];
    for (int i = 0; i < 8; ++i) lenbuf[i] = static_cast<uint8_t>((hlen >> (8 * i)) & 0xff);
    f.write(reinterpret_cast<char*>(lenbuf), 8);
    f.write(header.data(), static_cast<std::streamsize>(header.size()));
    for (const auto &t : tensors)
        f.write(reinterpret_cast<const char*>(t.data.data()),
                static_cast<std::streamsize>(t.data.size()));
}

inline std::vector<FakeTensor> BuildSyntheticQwenModel()
{
    const auto &c = TestCfg();
    const int H = c.hidden_size;
    const int I = c.intermediate_size;
    const int V = c.vocab_size;
    const int KD = KeyDim();
    const int VD = ValueDim();
    const int CD = ConvDim();

    std::vector<FakeTensor> t;

    // Global
    t.push_back({"model.language_model.embed_tokens.weight", "BF16", {V, H}, MakeBf16(V * H)});
    t.push_back({"lm_head.weight", "BF16", {V, H}, MakeBf16(V * H)});
    t.push_back({"model.language_model.norm.weight", "BF16", {H}, MakeBf16Values(H, [](int64_t i){ return (float)(i) * 0.01f; })});

    for (int L = 0; L < c.num_layers; ++L) {
        std::string lp = "model.language_model.layers." + std::to_string(L) + ".";
        bool linear = c.layer_types[L] == "linear_attention";

        t.push_back({lp + "input_layernorm.weight",  "BF16", {H}, MakeBf16(H)});
        t.push_back({lp + "post_attention_layernorm.weight", "BF16", {H}, MakeBf16(H)});

        if (linear) {
            auto la = lp + "linear_attn.";
            t.push_back({la + "in_proj_qkv.weight", "BF16", {KD * 2 + VD, H}, MakeBf16(H * KD * 2 + VD * H)});
            t.push_back({la + "in_proj_a.weight",   "BF16", {c.linear_num_value_heads, H}, MakeBf16(H * c.linear_num_value_heads)});
            t.push_back({la + "in_proj_b.weight",   "BF16", {c.linear_num_value_heads, H}, MakeBf16(H * c.linear_num_value_heads)});
            t.push_back({la + "in_proj_z.weight",   "BF16", {VD, H}, MakeBf16(H * VD)});
            t.push_back({la + "out_proj.weight",    "BF16", {H, VD}, MakeBf16(H * VD)});
            t.push_back({la + "conv1d.weight",      "BF16", {CD, c.linear_conv_kernel_dim}, MakeBf16(CD * c.linear_conv_kernel_dim)});
            t.push_back({la + "norm.weight",        "BF16", {c.linear_value_head_dim}, MakeBf16(c.linear_value_head_dim)});
            t.push_back({la + "A_log",              "F32",  {c.linear_num_value_heads}, MakeF32Values(c.linear_num_value_heads, [](int64_t i){ return (float)i * 0.5f + 1.0f; })});
            t.push_back({la + "dt_bias",            "F32",  {c.linear_num_value_heads}, MakeF32Values(c.linear_num_value_heads, [](int64_t i){ return (float)i * 0.1f; })});
        } else {
            auto fa = lp + "self_attn.";
            t.push_back({fa + "q_proj.weight", "BF16", {QProjOut(), H}, MakeBf16(H * QProjOut())});
            t.push_back({fa + "k_proj.weight", "BF16", {KVProjOut(), H}, MakeBf16(H * KVProjOut())});
            t.push_back({fa + "v_proj.weight", "BF16", {KVProjOut(), H}, MakeBf16(H * KVProjOut())});
            t.push_back({fa + "o_proj.weight", "BF16", {H, c.num_attention_heads * c.head_dim}, MakeBf16(c.num_attention_heads * c.head_dim * H)});
            t.push_back({fa + "q_norm.weight", "BF16", {c.head_dim}, MakeBf16(c.head_dim)});
            t.push_back({fa + "k_norm.weight", "BF16", {c.head_dim}, MakeBf16(c.head_dim)});
        }

        // MLP (SwiGLU)
        t.push_back({lp + "mlp.gate_proj.weight", "BF16", {I, H}, MakeBf16(H * I)});
        t.push_back({lp + "mlp.up_proj.weight",   "BF16", {I, H}, MakeBf16(H * I)});
        t.push_back({lp + "mlp.down_proj.weight", "BF16", {H, I}, MakeBf16(I * H)});
    }

    // MTP + Visual tensors — must be skipped by the converter.
    t.push_back({"mtp.norm.weight", "BF16", {H}, MakeBf16(H)});
    t.push_back({"mtp.layers.0.mlp.gate_proj.weight", "BF16", {H, I}, MakeBf16(H * I)});
    t.push_back({"visual.blocks.0.norm1.weight", "BF16", {H}, MakeBf16(H)});

    return t;
}

inline std::vector<std::string> TensorNamesIn(const std::filesystem::path &arrow_path)
{
    auto batch_res = lsxcommon::ReadRecordBatchFromFile(arrow_path);
    if (!batch_res.ok()) return {};
    auto t_res = lsxcommon::PinnedSchema::UnpackBatch(*batch_res);
    if (!t_res.ok()) return {};
    std::vector<std::string> names;
    for (auto &t : *t_res) names.push_back(t.name);
    return names;
}

// Writes a config.json matching BuildSyntheticQwenModel so the converter's
// geometry inference reads authoritative values.
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
        "{\"model_type\":\"qwen3_5\"," 
        "\"text_config\":{"
        "\"hidden_size\":" + std::to_string(c.hidden_size) +
        ",\"intermediate_size\":" + std::to_string(c.intermediate_size) +
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
        ",\"rms_norm_eps\":" + std::to_string(c.rms_norm_eps) +
        ",\"layer_types\":" + layer_types +
        ",\"rope_parameters\":{\"theta\":1,\"rope_theta\":" + std::to_string(c.rope_theta) +
        ",\"partial_rotary_factor\":" + std::to_string(c.partial_rotary_factor) +
        ",\"mrope_section\":[11,11,10],\"rope_type\":\"default\"}"
        "}}";
    std::ofstream f(dir / "config.json");
    f << json;
}

} // namespace qwentest