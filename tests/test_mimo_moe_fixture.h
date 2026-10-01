#pragma once

// MiMo-V2.6-Flash-RL family test fixture. Re-exports the family-neutral
// synthetic-checkpoint primitives (test-suite autonomy: never borrow another
// family's fixture) and adds MiMo-specific helpers.
#include "test_synth_checkpoint.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace mimotest {

namespace fs = std::filesystem;

// Geometric shape of the synthetic MiMo checkpoint. Small enough for a
// bounded convert test; realistic enough to exercise every pipeline path:
// one GA layer with dense FFN, one SWA layer with MoE experts.
struct SynthShape {
    int hidden = 64;
    int layers = 2;
    int vocab = 128;
    int dense_intermediate = 256;
    int moe_intermediate = 32;
    int num_experts = 4;
    int top_k = 2;
    int heads = 2;
    int kv_heads_ga = 1;
    int kv_heads_swa = 2;
    int head_dim = 8;
    int v_head_dim = 4;
};

inline int GaQkvRows(const SynthShape &s)
{
    return s.heads * s.head_dim + s.kv_heads_ga * s.head_dim +
           s.kv_heads_ga * s.v_head_dim;
}
inline int SwaQkvRows(const SynthShape &s)
{
    return s.heads * s.head_dim + s.kv_heads_swa * s.head_dim +
           s.kv_heads_swa * s.v_head_dim;
}

inline void WriteConfig(const fs::path &in, const SynthShape &s)
{
    std::ofstream cfg(in / "config.json");
    cfg << "{\"hidden_size\": " << s.hidden
        << ", \"num_hidden_layers\": " << s.layers
        << ", \"vocab_size\": " << s.vocab
        << ", \"intermediate_size\": " << s.dense_intermediate
        << ", \"moe_intermediate_size\": " << s.moe_intermediate
        << ", \"n_routed_experts\": " << s.num_experts
        << ", \"num_experts_per_tok\": " << s.top_k
        << ", \"num_attention_heads\": " << s.heads
        << ", \"num_key_value_heads\": " << s.kv_heads_ga
        << ", \"swa_num_key_value_heads\": " << s.kv_heads_swa
        << ", \"head_dim\": " << s.head_dim
        << ", \"v_head_dim\": " << s.v_head_dim
        << ", \"swa_head_dim\": " << s.head_dim
        << ", \"swa_v_head_dim\": " << s.v_head_dim
        << ", \"layernorm_epsilon\": 1e-6"
        << ", \"rope_theta\": 10000000.0"
        << ", \"swa_rope_theta\": 10000.0"
        << ", \"partial_rotary_factor\": 0.5"
        << ", \"sliding_window\": 8"
        << ", \"attention_value_scale\": 0.707"
        << ", \"n_group\": 1, \"topk_group\": 1, \"norm_topk_prob\": true"
        << ", \"scoring_func\": \"sigmoid\", \"topk_method\": \"noaux_tc\""
        << ", \"hybrid_layer_pattern\": [";
    for (int i = 0; i < s.layers; ++i) cfg << (i == 0 ? 0 : 1) << (i + 1 < s.layers ? "," : "");
    cfg << "], \"moe_layer_freq\": [";
    for (int i = 0; i < s.layers; ++i) cfg << (i == 0 ? 0 : 1) << (i + 1 < s.layers ? "," : "");
    cfg << "], \"eos_token_id\": 2, \"pad_token_id\": 2, \"bos_token_id\": 1}\n";
}

// Builds a synthetic MiMo HF checkpoint (config + one safetensors shard +
// tokenizer.json) covering globals, a GA layer, a dense FFN, a SWA MoE layer
// with all experts, an MXFP4 shift-failure fallback expert, a stripped
// multimodal tensor and an MTP tensor.
inline fs::path BuildSyntheticMiMoCheckpoint(const fs::path &in,
                                             const SynthShape &s = {})
{
    fs::create_directories(in);
    using synthtest::FakeTensor;
    std::vector<FakeTensor> tensors;

    auto bf16 = [&](const std::string &n, std::vector<int64_t> sh) {
        FakeTensor t; t.name = n; t.dtype = "BF16"; t.shape = sh;
        int64_t numel = 1; for (auto d : sh) numel *= d;
        t.data = synthtest::MakeBf16(numel);
        return t;
    };
    auto fp8 = [&](const std::string &n, std::vector<int64_t> sh) {
        FakeTensor t; t.name = n; t.dtype = "F8_E4M3"; t.shape = sh;
        int64_t numel = 1; for (auto d : sh) numel *= d;
        t.data.assign(static_cast<size_t>(numel), 0x30);  // ~0.5 in E4M3
        return t;
    };
    auto f32 = [&](const std::string &n, std::vector<int64_t> sh, float v) {
        FakeTensor t; t.name = n; t.dtype = "F32"; t.shape = sh;
        t.data = synthtest::MakeF32Values(
            1, [v](int64_t) { return v; });
        (void)sh;
        return t;
    };
    auto scale_f32 = [&](const std::string &n, int rows, int cols) {
        FakeTensor t; t.name = n; t.dtype = "F32"; t.shape = {rows, cols};
        t.data = synthtest::MakeF32Values(
            static_cast<int64_t>(rows) * cols, [](int64_t) { return 1.0f; });
        return t;
    };
    auto packed = [&](const std::string &n, int rows, int packed_cols, uint8_t fill) {
        FakeTensor t; t.name = n; t.dtype = "U8"; t.shape = {rows, packed_cols};
        t.data.assign(static_cast<size_t>(rows) * packed_cols, fill);
        return t;
    };
    auto e8m0 = [&](const std::string &n, int rows, int blocks, uint8_t b) {
        FakeTensor t; t.name = n; t.dtype = "U8"; t.shape = {rows, blocks};
        t.data.assign(static_cast<size_t>(rows) * blocks, b);
        return t;
    };

    // Globals.
    tensors.push_back(bf16("model.embed_tokens.weight", {s.vocab, s.hidden}));
    tensors.push_back(bf16("lm_head.weight", {s.vocab, s.hidden}));
    tensors.push_back(bf16("model.norm.weight", {s.hidden}));

    const int ga_rows = GaQkvRows(s);
    const int swa_rows = SwaQkvRows(s);
    const int C128 = 128;
    auto srows = [](int r) { return (r + 127) / 128; };

    // Layer 0: GA + dense FFN.
    tensors.push_back(bf16("model.layers.0.input_layernorm.weight", {s.hidden}));
    tensors.push_back(bf16("model.layers.0.post_attention_layernorm.weight", {s.hidden}));
    tensors.push_back(fp8("model.layers.0.self_attn.qkv_proj.weight", {ga_rows, s.hidden}));
    // GA quirk: one spare scale row block.
    tensors.push_back(scale_f32("model.layers.0.self_attn.qkv_proj.weight_scale_inv",
                                srows(ga_rows) + 1, (s.hidden + C128 - 1) / C128));
    tensors.push_back(bf16("model.layers.0.self_attn.o_proj.weight",
                           {s.hidden, s.heads * s.v_head_dim}));
    tensors.push_back(fp8("model.layers.0.mlp.gate_proj.weight", {s.dense_intermediate, s.hidden}));
    tensors.push_back(scale_f32("model.layers.0.mlp.gate_proj.weight_scale_inv",
                                srows(s.dense_intermediate), (s.hidden + C128 - 1) / C128));
    tensors.push_back(fp8("model.layers.0.mlp.up_proj.weight", {s.dense_intermediate, s.hidden}));
    tensors.push_back(scale_f32("model.layers.0.mlp.up_proj.weight_scale_inv",
                                srows(s.dense_intermediate), (s.hidden + C128 - 1) / C128));
    tensors.push_back(fp8("model.layers.0.mlp.down_proj.weight", {s.hidden, s.dense_intermediate}));
    tensors.push_back(scale_f32("model.layers.0.mlp.down_proj.weight_scale_inv",
                                srows(s.hidden), (s.dense_intermediate + C128 - 1) / C128));

    // Layer 1: SWA + MoE.
    tensors.push_back(bf16("model.layers.1.input_layernorm.weight", {s.hidden}));
    tensors.push_back(bf16("model.layers.1.post_attention_layernorm.weight", {s.hidden}));
    tensors.push_back(fp8("model.layers.1.self_attn.qkv_proj.weight", {swa_rows, s.hidden}));
    tensors.push_back(scale_f32("model.layers.1.self_attn.qkv_proj.weight_scale_inv",
                                srows(swa_rows), (s.hidden + C128 - 1) / C128));
    tensors.push_back(bf16("model.layers.1.self_attn.o_proj.weight",
                           {s.hidden, s.heads * s.v_head_dim}));
    tensors.push_back(bf16("model.layers.1.self_attn.attention_sink_bias", {s.heads}));
    tensors.push_back(bf16("model.layers.1.mlp.gate.weight", {s.num_experts, s.hidden}));
    tensors.push_back(f32("model.layers.1.mlp.gate.e_score_correction_bias",
                          {s.num_experts}, 0.0f));

    const int mi = s.moe_intermediate;
    for (int e = 0; e < s.num_experts; ++e) {
        const std::string p = "model.layers.1.mlp.experts." + std::to_string(e);
        tensors.push_back(packed(p + ".gate_proj.weight", mi, s.hidden / 2,
                                 static_cast<uint8_t>(0xB0 + e)));
        tensors.push_back(packed(p + ".up_proj.weight", mi, s.hidden / 2,
                                 static_cast<uint8_t>(0xC0 + e)));
        tensors.push_back(packed(p + ".down_proj.weight", s.hidden, mi / 2, 0xD0));
        // Expert 0: uniform scale -> exact NVFP4 repack.
        // Expert 1: spread 20 -> ChooseE8m0ScaleShift fails -> dequant fallback.
        const uint8_t gb = (e == 1) ? 120 : 120;
        tensors.push_back(e8m0(p + ".gate_proj.weight_scale", mi, s.hidden / 32, gb));
        if (e == 1) {
            // Overwrite one byte to create a spread > 13.
            tensors.back().data[0] = 100;
        }
        tensors.push_back(e8m0(p + ".up_proj.weight_scale", mi, s.hidden / 32, 120));
        tensors.push_back(e8m0(p + ".down_proj.weight_scale", s.hidden, mi / 32, 120));
    }

    // Stripped multimodal ballast + counted MTP.
    tensors.push_back(bf16("visual.blocks.0.attn.proj.weight", {8, 8}));
    tensors.push_back(bf16("model.mtp.layers.0.mlp.gate_proj.weight", {8, 8}));

    synthtest::WriteSafetensors(in, "model-00001-of-00001.safetensors", tensors);

    fs::copy_file(std::string(TOKENIZER_TEST_DATA_DIR) + "/synthetic_tok.json",
                  in / "tokenizer.json", fs::copy_options::overwrite_existing);

    // NOTE: config.json is written last so callers can override `in` layout.
    WriteConfig(in, s);
    return in;
}

inline std::string ReadFileText(const fs::path &p)
{
    std::ifstream f(p);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

inline bool ExpertFileExists(const fs::path &out, int layer, int expert)
{
    return fs::exists(fs::path(out) / "logestix" / "experts" /
                      ("layer_" + std::to_string(layer)) /
                      ("exp_" + std::to_string(expert) + ".arrow"));
}

} // namespace mimotest
