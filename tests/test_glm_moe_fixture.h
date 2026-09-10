#pragma once

// Shared fixture builder for GLM-5.2 converter + loader tests.
// Builds a synthetic 79-layer safetensors model directory on disk and reads
// logged logestix layout files back, so both the converter and the loader
// tests exercise the same realistic layout. Tensor-writing primitives live in
// the family-neutral test_synth_checkpoint.h (suite-autonomy invariant) and
// are re-exported below under glmtest for call sites.

#include "test_synth_checkpoint.h"

#include <absl/strings/str_cat.h>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace glmtest {

using synthtest::FakeTensor;
using synthtest::FloatToBf16;
using synthtest::MakeBf16;
using synthtest::MakeBf16BitsAt;
using synthtest::MakeBf16Values;
using synthtest::MakeF32;
using synthtest::MakeF32Values;
using synthtest::TensorNamesIn;
using synthtest::WriteSafetensors;

inline void AppendAttention(std::vector<FakeTensor> &t, int l)
{
    auto p = absl::StrCat("model.layers.", l, ".");
    t.push_back({p + "input_layernorm.weight",          "BF16", {16}, MakeBf16(16)});
    t.push_back({p + "post_attention_layernorm.weight", "BF16", {16}, MakeBf16(16)});
    t.push_back({p + "self_attn.q_a_proj.weight",            "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({p + "self_attn.q_a_layernorm.weight",       "BF16", {16},    MakeBf16(16)});
    t.push_back({p + "self_attn.q_b_proj.weight",            "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({p + "self_attn.kv_a_proj_with_mqa.weight",  "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({p + "self_attn.kv_a_layernorm.weight",      "BF16", {16},    MakeBf16(16)});
    t.push_back({p + "self_attn.kv_b_proj.weight",           "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({p + "self_attn.o_proj.weight",              "BF16", {16, 16}, MakeBf16(256)});
}

inline void AppendMoeFfn(std::vector<FakeTensor> &t, int l)
{
    auto p = absl::StrCat("model.layers.", l, ".mlp.");
    t.push_back({p + "gate.weight",                "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({p + "gate.e_score_correction_bias", "F32", {16},   MakeF32(16)});
    for (int e = 0; e < 2; ++e) {
        auto ep = absl::StrCat(p, "experts.", e, ".");
        t.push_back({ep + "gate_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
        t.push_back({ep + "down_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
        t.push_back({ep + "up_proj.weight",   "BF16", {16, 16}, MakeBf16(256)});
    }
    t.push_back({p + "shared_experts.gate_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({p + "shared_experts.down_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({p + "shared_experts.up_proj.weight",   "BF16", {16, 16}, MakeBf16(256)});
}

inline void AppendDenseMlp(std::vector<FakeTensor> &t, int l)
{
    auto p = absl::StrCat("model.layers.", l, ".");
    t.push_back({p + "mlp.gate_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({p + "mlp.down_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({p + "mlp.up_proj.weight",   "BF16", {16, 16}, MakeBf16(256)});
}

inline void AppendIndexer(std::vector<FakeTensor> &t, int l)
{
    auto p = absl::StrCat("model.layers.", l, ".self_attn.indexer.");
    t.push_back({p + "wq_b.weight",          "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({p + "wk.weight",            "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({p + "weights_proj.weight",  "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({p + "k_norm.weight",        "BF16", {16},    MakeBf16(16)});
    t.push_back({p + "k_norm.bias",          "F32",  {16},    MakeF32(16)});
}

inline std::vector<FakeTensor> BuildSyntheticModel()
{
    // Full heterogeneous grid matching a real GLM-5.2 topology:
    // layers 0-2 dense (attention+norm+MLP), layers 3-78 MoE
    // (attention+norm+router+bias, indexer when l % 4 == 0),
    // MTP extras on layer 78. Global embed/lm_head/model.norm.
    std::vector<FakeTensor> t;
    t.push_back({"model.embed_tokens.weight", "BF16", {64, 16}, MakeBf16(64 * 16)});
    t.push_back({"lm_head.weight",            "BF16", {64, 16}, MakeBf16(64 * 16)});
    t.push_back({"model.norm.weight",         "BF16", {16},     MakeBf16(16)});

    for (int l = 0; l <= 78; ++l) {
        AppendAttention(t, l);
        if (l < 3) {
            AppendDenseMlp(t, l);
        } else {
            AppendMoeFfn(t, l);
            if (l % 4 == 0) AppendIndexer(t, l);
        }
    }

    t.push_back({"model.layers.78.enorm.weight",             "BF16", {16},     MakeBf16(16)});
    t.push_back({"model.layers.78.hnorm.weight",             "BF16", {16},     MakeBf16(16)});
    t.push_back({"model.layers.78.eh_proj.weight",           "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({"model.layers.78.shared_head.norm.weight",  "BF16", {16},     MakeBf16(16)});

    return t;
}

// Writes the config.json the converter's integrity checks expect. Defaults
// match BuildSyntheticModel(): 79 layers, hidden 16, vocab 64, 2 local experts.
inline void WriteConfig(const std::filesystem::path &dir,
                        int num_layers = 79, int hidden = 16,
                        int vocab = 64, int experts = 2)
{
    std::string json = absl::StrCat(
        "{\"model_type\":\"glm\",",
        "\"num_layers\":", num_layers, ",",
        "\"hidden_size\":", hidden, ",",
        "\"vocab_size\":", vocab, ",",
        "\"num_local_experts\":", experts, ",",
        "\"num_experts\":", experts,
        "}");
    std::ofstream f(dir / "config.json");
    f << json;
}

// Copies the shared synthetic BPE tokenizer.json into `dir` so converters
// that make the tokenizer mandatory can build+verify it.
inline void WriteTokenizer(const std::filesystem::path &dir)
{
    std::filesystem::copy_file(
        TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
        dir / "tokenizer.json");
}

} // namespace glmtest