#pragma once

// MiniMax-M3 synthetic-checkpoint fixture (converter/loader/kernel tests).
// Builds the family-neutral primitives from test_synth_checkpoint.h; geometry
// is deliberately tiny so the whole pipeline runs in a bounded TIMEOUT, and
// the frozen passport is relaxed with LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY.
#include "test_synth_checkpoint.h"

#include <absl/strings/str_cat.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace minimax_test {

struct Cfg {
    int hidden = 64;
    int layers = 4;
    int heads = 2;
    int kv_heads = 1;
    int head_dim = 32;
    int rotary_dim = 16;
    int vocab = 64;
    int dense_inter = 32;
    int moe_inter = 16;
    int shared_inter = 16;
    int experts = 8;
    int top_k = 2;
    int index_n_heads = 1;   // == kv_heads (one indexer head per GQA group)
    int index_head_dim = 32;
    int block_size = 8;
    int topk_blocks = 2;
    int local_blocks = 1;
    int first_moe_layer = 1;
};

inline const Cfg &TestCfg()
{
    static const Cfg c;
    return c;
}

inline const char *kMoeFreq()
{
    // 0 -> dense MLP, 1 -> MoE. Layers 0 dense, 1..3 MoE.
    return "[0,1,1,1]";
}

inline const char *kSparseFreq()
{
    // 0 -> dense attention, 1 -> sparse (MSA indexer).
    return "[0,1,1,1]";
}

inline void WriteConfig(const std::filesystem::path &dir)
{
    const Cfg &c = TestCfg();
    std::ofstream f(dir / "config.json");
    f << R"({"architectures":["MiniMaxM3SparseForConditionalGeneration"],)"
      << R"("model_type":"minimax_m3_vl","text_config":{)"
      << "\"hidden_size\":" << c.hidden
      << ",\"intermediate_size\":" << c.moe_inter
      << ",\"num_hidden_layers\":" << c.layers
      << ",\"num_attention_heads\":" << c.heads
      << ",\"num_key_value_heads\":" << c.kv_heads
      << ",\"head_dim\":" << c.head_dim
      << ",\"rotary_dim\":" << c.rotary_dim
      << ",\"vocab_size\":" << c.vocab
      << ",\"rms_norm_eps\":1e-06,\"rope_theta\":5000000"
      << ",\"dense_intermediate_size\":" << c.dense_inter
      << ",\"shared_intermediate_size\":" << c.shared_inter
      << ",\"num_local_experts\":" << c.experts
      << ",\"num_experts_per_tok\":" << c.top_k
      << ",\"routed_scaling_factor\":2.0"
      << ",\"swiglu_alpha\":1.702,\"swiglu_limit\":7.0"
      << ",\"moe_layer_freq\":" << kMoeFreq()
      << R"(,"sparse_attention_config":{"sparse_num_index_heads":)"
      << c.index_n_heads << ",\"sparse_index_dim\":" << c.index_head_dim
      << ",\"sparse_block_size\":" << c.block_size
      << ",\"sparse_topk_blocks\":" << c.topk_blocks
      << ",\"sparse_local_block\":" << c.local_blocks
      << ",\"sparse_attention_freq\":" << kSparseFreq() << "}}," 
      << R"("generation_config":{"bos_token_id":1,"eos_token_id":2,"pad_token_id":0}})"
      << "\n";
    f.close();
    std::ofstream g(dir / "generation_config.json");
    g << R"({"bos_token_id":1,"eos_token_id":2,"pad_token_id":0})"
      << "\n";
}

inline std::vector<synthtest::FakeTensor> Tensors()
{
    const Cfg &c = TestCfg();
    const int H = c.hidden;
    const int QW = c.heads * c.head_dim;
    const int KVW = c.kv_heads * c.head_dim;
    const int IDXQ = c.index_n_heads * c.index_head_dim;
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
                // Small deterministic values with a shape-dependent sign so
                // w1 != w3 and norms are non-degenerate.
                const int64_t k = (i * 7 + seed * 13) % 23;
                return static_cast<float>((k - 11)) * 0.011f;
            });
        return f;
    };
    auto f32 = [](const std::string &name, std::vector<int64_t> shape,
                  int64_t seed) {
        int64_t n = 1;
        for (int64_t d : shape) n *= d;
        synthtest::FakeTensor f;
        f.name = name;
        f.dtype = "F32";
        f.shape = std::move(shape);
        f.data = synthtest::MakeF32Values(
            n, [seed](int64_t i) {
                const int64_t k = (i * 5 + seed * 3) % 17;
                return static_cast<float>(k - 8) * 0.004f;
            });
        return f;
    };

    t.push_back(bf16("language_model.model.embed_tokens.weight",
                     {c.vocab, H}, 1));
    t.push_back(bf16("language_model.lm_head.weight", {c.vocab, H}, 2));
    t.push_back(bf16("language_model.model.norm.weight", {H}, 3));
    // Vision ballast: counted, never applied.
    t.push_back(bf16("vision_tower.vision_model.embeddings.patch_embedding.weight",
                     {8, 3, 2, 2, 2}, 4));
    t.push_back(bf16("multi_modal_projector.linear_1.weight", {8, 8}, 5));

    for (int l = 0; l < c.layers; ++l) {
        const std::string p =
            absl::StrCat("language_model.model.layers.", l, ".");
        t.push_back(bf16(p + "input_layernorm.weight", {H}, 10 + l));
        t.push_back(bf16(p + "post_attention_layernorm.weight", {H}, 20 + l));
        t.push_back(bf16(p + "self_attn.q_proj.weight", {QW, H}, 30 + l));
        t.push_back(bf16(p + "self_attn.k_proj.weight", {KVW, H}, 40 + l));
        t.push_back(bf16(p + "self_attn.v_proj.weight", {KVW, H}, 50 + l));
        t.push_back(bf16(p + "self_attn.o_proj.weight", {H, QW}, 60 + l));
        t.push_back(bf16(p + "self_attn.q_norm.weight", {c.head_dim}, 70 + l));
        t.push_back(bf16(p + "self_attn.k_norm.weight", {c.head_dim}, 80 + l));
        if (l >= c.first_moe_layer) {  // sparse attention layers carry the indexer
            t.push_back(bf16(p + "self_attn.index_q_proj.weight", {IDXQ, H},
                             90 + l));
            t.push_back(bf16(p + "self_attn.index_k_proj.weight",
                             {c.index_head_dim, H}, 100 + l));
            t.push_back(bf16(p + "self_attn.index_q_norm.weight",
                             {c.index_head_dim}, 110 + l));
            t.push_back(bf16(p + "self_attn.index_k_norm.weight",
                             {c.index_head_dim}, 120 + l));
        }
        if (l < c.first_moe_layer) {
            t.push_back(bf16(p + "mlp.gate_proj.weight", {c.dense_inter, H},
                             130 + l));
            t.push_back(bf16(p + "mlp.up_proj.weight", {c.dense_inter, H},
                             140 + l));
            t.push_back(bf16(p + "mlp.down_proj.weight", {H, c.dense_inter},
                             150 + l));
        } else {
            t.push_back(f32(p + "block_sparse_moe.gate.weight",
                            {c.experts, H}, 160 + l));
            t.push_back(f32(p + "block_sparse_moe.e_score_correction_bias",
                            {c.experts}, 170 + l));
            t.push_back(bf16(p + "block_sparse_moe.shared_experts.gate_proj.weight",
                             {c.shared_inter, H}, 180 + l));
            t.push_back(bf16(p + "block_sparse_moe.shared_experts.up_proj.weight",
                             {c.shared_inter, H}, 190 + l));
            t.push_back(bf16(p + "block_sparse_moe.shared_experts.down_proj.weight",
                             {H, c.shared_inter}, 200 + l));
            for (int e = 0; e < c.experts; ++e) {
                const std::string pe = absl::StrCat(
                    p, "block_sparse_moe.experts.", e, ".");
                t.push_back(bf16(pe + "w1.weight", {c.moe_inter, H},
                                 1000 + e * 3 + l));
                t.push_back(bf16(pe + "w2.weight", {H, c.moe_inter},
                                 1001 + e * 3 + l));
                t.push_back(bf16(pe + "w3.weight", {c.moe_inter, H},
                                 1002 + e * 3 + l));
            }
        }
    }
    return t;
}

// Two shards so the converter's cross-shard expert join is exercised.
// `with_tokenizer` copies the MiniMax synthetic BPE fixture in, so the
// tokenizer bridge + the model/loader stages can run without the real
// checkpoint (the fixture's markers are the checkpoint's own sentinels).
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

} // namespace minimax_test
