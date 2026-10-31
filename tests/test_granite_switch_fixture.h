#pragma once

// Synthetic Granite-Switch checkpoint builder for converter / loader / model
// tests. Tensor-writing primitives come from the family-neutral
// test_synth_checkpoint.h (test-suite autonomy invariant).

#include "test_synth_checkpoint.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace gswtest {

using synthtest::FakeTensor;
using synthtest::MakeBf16Values;
using synthtest::MakeF32Values;
using synthtest::TensorNamesIn;
using synthtest::WriteSafetensors;

struct SwitchTestConfig {
    int hidden_size = 64;
    int intermediate_size = 128;
    int vocab_size = 72;
    int num_layers = 2;
    int num_attention_heads = 4;
    int num_key_value_heads = 2;
    int head_dim = 16;
    int max_lora_rank = 8;
    int switch_cache_layers = 2;
    std::vector<int> adapter_ranks = {4, 8};
    std::vector<int> adapter_token_ids = {64, 65};
    std::vector<int> adapter_substitute_token_ids = {6, 7};
    double rms_norm_eps = 1e-5;
    double rope_theta = 10000.0;
    double attention_multiplier = 0.0625;
    double embedding_multiplier = 12.0;
    double residual_multiplier = 0.175;
    double logits_scaling = 16.0;
    int bos_id = 0;
    int eos_id = 1;
    int pad_id = 2;
};

inline const SwitchTestConfig &Cfg()
{
    static SwitchTestConfig c;
    return c;
}

inline int QOut()
{
    return Cfg().num_attention_heads * Cfg().head_dim;
}

inline int KVOut()
{
    return Cfg().num_key_value_heads * Cfg().head_dim;
}

// Deterministic, finite BF16 fill (never NaN/Inf, unlike raw bit patterns).
inline std::vector<uint8_t> Fill(int64_t n, float scale = 0.02f)
{
    return MakeBf16Values(n, [scale](int64_t i) {
        return (static_cast<float>((i * 37) % 101) - 50.0f) * scale;
    });
}

inline std::vector<uint8_t> FillNorm(int64_t n)
{
    return MakeBf16Values(n, [](int64_t i) {
        return 1.0f + (static_cast<float>(i % 7) - 3.0f) * 0.01f;
    });
}

// A stacked LoRA A tensor is [num_adapters, 1, max_rank, in]. The source
// zero-pads rows beyond the adapter's rank; the converter asserts that.
inline void ZeroATail(FakeTensor &t, const std::vector<int> &ranks,
                      int max_rank, int in)
{
    const int64_t adapter_bytes = static_cast<int64_t>(max_rank) * in * 2;
    for (int a = 0; a < static_cast<int>(ranks.size()); ++a) {
        const int r = ranks[a];
        if (r >= max_rank) continue;
        const int64_t begin = a * adapter_bytes + static_cast<int64_t>(r) * in * 2;
        const int64_t end = (a + 1) * adapter_bytes;
        std::fill(t.data.begin() + begin, t.data.begin() + end, 0);
    }
}

inline std::vector<FakeTensor> BuildModel()
{
    const auto &c = Cfg();
    const int H = c.hidden_size;
    const int I = c.intermediate_size;
    const int V = c.vocab_size;
    const int A = static_cast<int>(c.adapter_ranks.size());
    const int R = c.max_lora_rank;

    std::vector<FakeTensor> t;
    t.push_back({"model.embed_tokens.weight", "BF16", {V, H}, Fill(V * H)});
    t.push_back({"model.norm.weight", "BF16", {H}, FillNorm(H)});
    t.push_back({"model.switch.codebook", "F32", {2048, 64},
                 MakeF32Values(2048 * 64, [](int64_t i) {
                     return (static_cast<float>(i % 7) - 3.0f) * 0.125f;
                 })});

    std::vector<int64_t> lut(V, -1);
    for (int a = 0; a < A; ++a)
        lut[c.adapter_token_ids[a]] = c.adapter_substitute_token_ids[a];
    std::vector<uint8_t> lut_bytes(lut.size() * sizeof(int64_t));
    std::memcpy(lut_bytes.data(), lut.data(), lut_bytes.size());
    t.push_back({"model.switch.control_to_substitute_lut", "I64", {V},
                 std::move(lut_bytes)});

    for (int L = 0; L < c.num_layers; ++L) {
        const std::string lp = "model.layers." + std::to_string(L) + ".";
        t.push_back({lp + "input_layernorm.weight", "BF16", {H}, FillNorm(H)});
        t.push_back(
            {lp + "post_attention_layernorm.weight", "BF16", {H}, FillNorm(H)});

        const std::string sa = lp + "self_attn.";
        t.push_back({sa + "qkv_proj.base_layer.weight", "BF16",
                     {QOut() + 2 * KVOut(), H},
                     Fill((QOut() + 2 * KVOut()) * H)});
        t.push_back({sa + "o_proj.base_layer.weight", "BF16", {H, QOut()},
                     Fill(H * QOut())});
        for (int s = 0; s < 3; ++s) {
            const int out = s == 0 ? QOut() : KVOut();
            const std::string slice = std::to_string(s);
            FakeTensor a{sa + "qkv_proj.lora_A_slices." + slice, "BF16",
                         {A, 1, R, H}, Fill(A * R * H, 0.01f)};
            ZeroATail(a, c.adapter_ranks, R, H);
            t.push_back(std::move(a));
            t.push_back({sa + "qkv_proj.lora_B_slices." + slice, "BF16",
                         {A, 1, out, R}, Fill(A * out * R, 0.01f)});
        }
        FakeTensor oa{sa + "o_proj.lora_A", "BF16", {A, 1, R, QOut()},
                      Fill(A * R * QOut(), 0.01f)};
        ZeroATail(oa, c.adapter_ranks, R, QOut());
        t.push_back(std::move(oa));
        t.push_back({sa + "o_proj.lora_B", "BF16", {A, 1, H, R},
                     Fill(A * H * R, 0.01f)});

        const std::string ml = lp + "shared_mlp.";
        t.push_back({ml + "input_linear.base_layer.weight", "BF16", {2 * I, H},
                     Fill(2 * I * H)});
        t.push_back({ml + "output_linear.base_layer.weight", "BF16", {H, I},
                     Fill(H * I)});
        for (int s = 0; s < 2; ++s) {
            const std::string slice = std::to_string(s);
            FakeTensor a{ml + "input_linear.lora_A_slices." + slice, "BF16",
                         {A, 1, R, H}, Fill(A * R * H, 0.01f)};
            ZeroATail(a, c.adapter_ranks, R, H);
            t.push_back(std::move(a));
            t.push_back({ml + "input_linear.lora_B_slices." + slice, "BF16",
                         {A, 1, I, R}, Fill(A * I * R, 0.01f)});
        }
        FakeTensor da{ml + "output_linear.lora_A", "BF16", {A, 1, R, I},
                      Fill(A * R * I, 0.01f)};
        ZeroATail(da, c.adapter_ranks, R, I);
        t.push_back(std::move(da));
        t.push_back({ml + "output_linear.lora_B", "BF16", {A, 1, H, R},
                     Fill(A * H * R, 0.01f)});
    }

    // Adapter 0 targets attention only: zero its MLP deltas like the real
    // guardian-core. The adapter axis is outermost, so adapter 0 is the first
    // half of each stacked LoRA tensor.
    for (auto &ft : t) {
        if (ft.name.find("shared_mlp.") != std::string::npos &&
            (ft.name.find("lora_A") != std::string::npos ||
             ft.name.find("lora_B") != std::string::npos)) {
            std::fill(ft.data.begin(), ft.data.begin() + ft.data.size() / 2, 0);
        }
    }
    return t;
}

inline void WriteConfig(const std::filesystem::path &dir)
{
    const auto &c = Cfg();
    std::string j =
        "{\"architectures\":[\"GraniteSwitchForCausalLM\"],"
        "\"model_type\":\"granite_switch\",\"hidden_size\":" +
        std::to_string(c.hidden_size) + ",\"intermediate_size\":" +
        std::to_string(c.intermediate_size) + ",\"shared_intermediate_size\":" +
        std::to_string(c.intermediate_size) + ",\"num_hidden_layers\":" +
        std::to_string(c.num_layers + c.switch_cache_layers) +
        ",\"vocab_size\":" + std::to_string(c.vocab_size) +
        ",\"num_attention_heads\":" + std::to_string(c.num_attention_heads) +
        ",\"num_key_value_heads\":" + std::to_string(c.num_key_value_heads) +
        ",\"projection_head_dim\":" + std::to_string(c.head_dim) +
        ",\"num_adapters\":2,\"max_lora_rank\":" +
        std::to_string(c.max_lora_rank) +
        ",\"adapter_names\":[\"guardian-core\",\"decision\"],"
        "\"adapter_ranks\":[4,8],\"adapter_token_ids\":[64,65],"
        "\"adapter_substitute_token_ids\":[6,7],"
        "\"tie_word_embeddings\":true,\"embedding_multiplier\":12.0,"
        "\"residual_multiplier\":0.175,\"logits_scaling\":16.0,"
        "\"attention_multiplier\":0.0625,\"rms_norm_eps\":1e-5,"
        "\"rope_theta\":10000.0,\"hidden_act\":\"silu\","
        "\"bos_token_id\":0,\"eos_token_id\":1,\"pad_token_id\":2,"
        "\"max_position_embeddings\":4096}";
    std::ofstream(dir / "config.json") << j;
}

// Writes the synthetic checkpoint and converts it; returns the output root.
inline bool BuildConverted(const std::filesystem::path &dir)
{
    if (!std::filesystem::exists(dir)) std::filesystem::create_directories(dir);
    WriteSafetensors(dir, "model.safetensors", BuildModel());
    WriteConfig(dir);
    return true;
}

} // namespace gswtest
