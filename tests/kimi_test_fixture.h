#pragma once

// Shared fixture builder for Kimi-K2.7-Code converter/loader tests.
// Reuses the GLM fixture primitives (FakeTensor, WriteSafetensors, MakeBf16,
// MakeF32) and adds Kimi-specific pieces:
//   - "language_model.model." checkpoint prefix on every tensor name
//   - routed experts as compressed-tensors INT4 group-32 pairs
//     (weight_packed I32 [rows, cols/8] + weight_scale BF16 [rows, cols/32])
//   - dense MLP only on layer 0 (first_k_dense_replace=1)
//   - multimodal config.json wrapper with text_config

#include "glm_test_fixture.h"

#include <absl/strings/str_cat.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace kimitest {

using glmtest::FakeTensor;
using glmtest::MakeBf16;
using glmtest::MakeF32;
using glmtest::WriteSafetensors;

// Deterministic INT4 nibble pattern: signed value in [-7, 7].
inline int NibbleAt(int r, int c) { return ((r * 131 + c * 17) % 15); }

// Builds weight_packed (I32 [rows, cols/8], low-nibble-first) and
// weight_scale (BF16 [rows, cols/32]) pair for one projection.
inline void AppendInt4ExpertProj(std::vector<FakeTensor> &t,
                                 const std::string &base,
                                 const std::string &proj,
                                 int rows, int cols)
{
    const int words_per_row = cols / 8;
    const int groups = cols / 32;
    std::vector<uint8_t> packed(static_cast<size_t>(rows) * words_per_row * 4,
                                0);
    for (int r = 0; r < rows; ++r) {
        for (int w = 0; w < words_per_row; ++w) {
            int32_t word = 0;
            for (int k = 0; k < 8; ++k)
                word |= (NibbleAt(r, w * 8 + k) & 0xF) << (4 * k);
            const size_t off =
                (static_cast<size_t>(r) * words_per_row + w) * 4;
            std::memcpy(packed.data() + off, &word, 4);
        }
    }
    t.push_back({base + proj + ".weight_packed", "I32",
                 {rows, words_per_row}, std::move(packed)});
    t.push_back({base + proj + ".weight_scale", "BF16",
                 {rows, groups},
                 MakeBf16(static_cast<int64_t>(rows) * groups)});
}

inline void AppendKimiAttention(std::vector<FakeTensor> &t, int l)
{
    auto p = absl::StrCat("language_model.model.layers.", l, ".");
    t.push_back({p + "input_layernorm.weight",          "BF16", {16}, MakeBf16(16)});
    t.push_back({p + "post_attention_layernorm.weight", "BF16", {16}, MakeBf16(16)});
    t.push_back({p + "self_attn.q_a_proj.weight",           "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({p + "self_attn.q_a_layernorm.weight",      "BF16", {16},    MakeBf16(16)});
    t.push_back({p + "self_attn.q_b_proj.weight",           "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({p + "self_attn.kv_a_proj_with_mqa.weight", "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({p + "self_attn.kv_a_layernorm.weight",     "BF16", {16},    MakeBf16(16)});
    t.push_back({p + "self_attn.kv_b_proj.weight",          "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({p + "self_attn.o_proj.weight",             "BF16", {16, 16}, MakeBf16(256)});
}

inline void AppendKimiMoeFfn(std::vector<FakeTensor> &t, int l)
{
    auto p = absl::StrCat("language_model.model.layers.", l, ".mlp.");
    t.push_back({p + "gate.weight",                  "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({p + "gate.e_score_correction_bias", "F32",  {16},    MakeF32(16)});
    for (int e = 0; e < 2; ++e) {
        auto ep = absl::StrCat(p, "experts.", e, ".");
        AppendInt4ExpertProj(t, ep, "gate_proj", 4, 32);
        AppendInt4ExpertProj(t, ep, "down_proj", 4, 32);
        AppendInt4ExpertProj(t, ep, "up_proj", 4, 32);
    }
    t.push_back({p + "shared_experts.gate_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({p + "shared_experts.down_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({p + "shared_experts.up_proj.weight",   "BF16", {16, 16}, MakeBf16(256)});
}

inline void AppendKimiDenseMlp(std::vector<FakeTensor> &t, int l)
{
    auto p = absl::StrCat("language_model.model.layers.", l, ".");
    t.push_back({p + "mlp.gate_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({p + "mlp.down_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({p + "mlp.up_proj.weight",   "BF16", {16, 16}, MakeBf16(256)});
}

inline std::vector<FakeTensor> BuildSyntheticKimiModel(int num_layers = 3)
{
    // Layers: 0 dense, 1..num_layers-1 MoE (2 experts each). Globals +
    // attention everywhere. INT4-packed routed experts.
    std::vector<FakeTensor> t;
    t.push_back({"language_model.model.embed_tokens.weight", "BF16", {64, 16},
                 MakeBf16(64 * 16)});
    t.push_back({"language_model.lm_head.weight",            "BF16", {64, 16},
                 MakeBf16(64 * 16)});
    t.push_back({"language_model.model.norm.weight",         "BF16", {16},
                 MakeBf16(16)});

    for (int l = 0; l < num_layers; ++l) {
        AppendKimiAttention(t, l);
        if (l == 0)
            AppendKimiDenseMlp(t, l);
        else
            AppendKimiMoeFfn(t, l);
    }
    return t;
}

inline void WriteKimiConfig(const std::filesystem::path &dir)
{
    // text_config carries no geometry keys here: the converter only enforces
    // locked values that are present (fixture uses tiny shapes).
    std::ofstream f(dir / "config.json");
    f << R"({"architectures":["KimiK25ForConditionalGeneration"],)"
      << R"("model_type":"kimi_k25",)"
      << R"("text_config":{"model_type":"kimi_k2","hidden_act":"silu"}})";
}

// Writes the minimal tiktoken ranks + tokenizer_config pair the converter's
// mandatory tokenizer step needs (complete byte alphabet, no merges needed).
inline void WriteKimiTokenizerFiles(const std::filesystem::path &dir)
{
    const char *b64 =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    auto enc1 = [&](uint8_t b) {
        std::string out;
        out += b64[(b >> 2) & 63];
        out += b64[(b & 3) << 4];
        out += "==";
        return out;
    };
    {
        std::ofstream f(dir / "tiktoken.model");
        for (int b = 0; b < 256; ++b)
            f << enc1(static_cast<uint8_t>(b)) << " " << b << "\n";
    }
    {
        std::ofstream f(dir / "tokenizer_config.json");
        f << "{\"bos_token\":\"[BOS]\",\"eos_token\":\"[EOS]\","
             "\"pad_token\":\"[PAD]\",\"added_tokens_decoder\":{"
             "\"256\":{\"content\":\"[BOS]\",\"special\":true},"
             "\"257\":{\"content\":\"[EOS]\",\"special\":true},"
             "\"258\":{\"content\":\"[PAD]\",\"special\":true}}}";
    }
}

} // namespace kimitest
