#pragma once

// Family-neutral synthetic checkpoint primitives re-exported for the Alice
// suite (test-suite autonomy invariant: never borrow another family's
// fixture). Defines a tiny legal AliceAI geometry used by the converter,
// loader and model-stage tests. All dims that feed NVFP4 packing are multiples
// of 16.

#include "test_synth_checkpoint.h"

#include <fstream>
#include <string>
#include <vector>

namespace alicetest {

namespace fs = std::filesystem;

inline constexpr int kHidden = 64;
inline constexpr int kLayers = 8;           // full attention at i%4==3
inline constexpr int kFullAttnInterval = 4;
inline constexpr int kBlockAttnRes = 4;
inline constexpr int kHeads = 4;
inline constexpr int kKVHeads = 2;
inline constexpr int kHeadDim = 16;
inline constexpr int kRotaryDim = 4;
inline constexpr int kLinearKeyHeads = 4;
inline constexpr int kLinearValueHeads = 4;
inline constexpr int kLinearKeyHeadDim = 16;
inline constexpr int kLinearValueHeadDim = 16;
inline constexpr int kLinearConv = 4;
inline constexpr int kKeyDim = kLinearKeyHeads * kLinearKeyHeadDim;      // 64
inline constexpr int kValueDim = kLinearValueHeads * kLinearValueHeadDim; // 64
inline constexpr int kExperts = 4;
inline constexpr int kTopK = 2;
inline constexpr int kMoeInter = 16;
inline constexpr int kSharedInter = 16;
inline constexpr int kVocab = 64;

inline int64_t Numel(const std::vector<int64_t> &shape)
{
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

inline void Add(std::vector<synthtest::FakeTensor> &ts, const std::string &name,
                std::vector<int64_t> shape)
{
    synthtest::FakeTensor t;
    t.name = name;
    t.dtype = "BF16";
    t.shape = std::move(shape);
    t.data = synthtest::MakeBf16(Numel(t.shape));
    ts.push_back(std::move(t));
}

// `mtp_layers` > 0 asks for a checkpoint that carries the multi-token-prediction
// head. The default is 0 because most Alice tests are about the live path; the
// no-ballast case (Task 8b) is the one that needs it.
inline void WriteAliceConfigJson(const fs::path &dir, int mtp_layers = 0)
{
    std::ofstream f(dir / "config.json");
    f << "{\n"
      << "  \"architectures\": [\"AliceAIForCausalLM\"],\n"
      << "  \"model_type\": \"alice_ai\",\n"
      << "  \"hidden_size\": " << kHidden << ",\n"
      << "  \"num_hidden_layers\": " << kLayers << ",\n"
      << "  \"num_attention_heads\": " << kHeads << ",\n"
      << "  \"num_key_value_heads\": " << kKVHeads << ",\n"
      << "  \"head_dim\": " << kHeadDim << ",\n"
      << "  \"linear_num_key_heads\": " << kLinearKeyHeads << ",\n"
      << "  \"linear_num_value_heads\": " << kLinearValueHeads << ",\n"
      << "  \"linear_key_head_dim\": " << kLinearKeyHeadDim << ",\n"
      << "  \"linear_value_head_dim\": " << kLinearValueHeadDim << ",\n"
      << "  \"linear_conv_kernel_dim\": " << kLinearConv << ",\n"
      << "  \"num_experts\": " << kExperts << ",\n"
      << "  \"num_experts_per_tok\": " << kTopK << ",\n"
      << "  \"moe_intermediate_size\": " << kMoeInter << ",\n"
      << "  \"shared_expert_intermediate_size\": " << kSharedInter << ",\n"
      << "  \"block_attn_res_block_size\": " << kBlockAttnRes << ",\n"
      << "  \"mtp_num_hidden_layers\": " << mtp_layers << ",\n"
      << "  \"rms_norm_eps\": 1e-6,\n"
      << "  \"partial_rotary_factor\": 0.25,\n"
      << "  \"rope_theta\": 1000000.0,\n"
      << "  \"vocab_size\": " << kVocab << ",\n"
      << "  \"bos_token_id\": 1,\n"
      << "  \"eos_token_id\": 2,\n"
      << "  \"max_position_embeddings\": 2048,\n"
      << "  \"layer_types\": [";
    for (int i = 0; i < kLayers; ++i) {
        if (i) f << ", ";
        f << "\"" << ((i % kFullAttnInterval == kFullAttnInterval - 1)
                          ? "full_attention" : "linear_attention")
          << "\"";
    }
    f << "]\n}\n";
}

inline void WriteAliceShard(const fs::path &dir, bool with_mtp = false)
{
    std::vector<synthtest::FakeTensor> ts;
    Add(ts, "model.embed_tokens.weight", {kVocab, kHidden});
    Add(ts, "lm_head.weight", {kVocab, kHidden});
    Add(ts, "model.norm.weight", {kHidden});
    Add(ts, "model.attnres_final.res_proj.weight", {1, kHidden});
    Add(ts, "model.attnres_final.res_norm_weight", {kHidden});

    for (int l = 0; l < kLayers; ++l) {
        const std::string b = "model.layers." + std::to_string(l) + ".";
        Add(ts, b + "input_layernorm.weight", {kHidden});
        Add(ts, b + "post_attention_layernorm.weight", {kHidden});
        Add(ts, b + "mlp_res_proj.weight", {1, kHidden});
        Add(ts, b + "mlp_res_norm_weight", {kHidden});
        if (l != 0) {
            Add(ts, b + "attn_res_proj.weight", {1, kHidden});
            Add(ts, b + "attn_res_norm_weight", {kHidden});
        }
        if (l % kFullAttnInterval == kFullAttnInterval - 1) {
            Add(ts, b + "self_attn.q_proj.weight",
                {2 * kHeads * kHeadDim, kHidden});
            Add(ts, b + "self_attn.k_proj.weight", {kKVHeads * kHeadDim, kHidden});
            Add(ts, b + "self_attn.v_proj.weight", {kKVHeads * kHeadDim, kHidden});
            Add(ts, b + "self_attn.o_proj.weight", {kHidden, kHeads * kHeadDim});
            Add(ts, b + "self_attn.q_norm.weight", {kHeadDim});
            Add(ts, b + "self_attn.k_norm.weight", {kHeadDim});
        } else {
            Add(ts, b + "linear_attn.q_proj.weight", {kKeyDim, kHidden});
            Add(ts, b + "linear_attn.k_proj.weight", {kKeyDim, kHidden});
            Add(ts, b + "linear_attn.v_proj.weight", {kValueDim, kHidden});
            Add(ts, b + "linear_attn.o_proj.weight", {kHidden, kValueDim});
            Add(ts, b + "linear_attn.f_a_proj.weight",
                {kLinearValueHeadDim, kHidden});
            Add(ts, b + "linear_attn.f_b_proj.weight",
                {kKeyDim, kLinearValueHeadDim});
            Add(ts, b + "linear_attn.g_a_proj.weight",
                {kLinearValueHeadDim, kHidden});
            Add(ts, b + "linear_attn.g_b_proj.weight",
                {kValueDim, kLinearValueHeadDim});
            Add(ts, b + "linear_attn.b_proj.weight", {kLinearKeyHeads, kHidden});
            Add(ts, b + "linear_attn.q_conv1d.weight", {kKeyDim, 1, kLinearConv});
            Add(ts, b + "linear_attn.k_conv1d.weight", {kKeyDim, 1, kLinearConv});
            Add(ts, b + "linear_attn.v_conv1d.weight", {kValueDim, 1, kLinearConv});
            Add(ts, b + "linear_attn.a_log_bias", {kLinearKeyHeads});
            Add(ts, b + "linear_attn.dt_bias", {kKeyDim});
            Add(ts, b + "linear_attn.o_norm.weight", {kLinearValueHeadDim});
        }
        Add(ts, b + "mlp.gate.weight", {kExperts, kHidden});
        Add(ts, b + "mlp.gate.e_score_correction_bias", {kExperts});
        Add(ts, b + "mlp.shared_expert.gate_proj.weight", {kSharedInter, kHidden});
        Add(ts, b + "mlp.shared_expert.up_proj.weight", {kSharedInter, kHidden});
        Add(ts, b + "mlp.shared_expert.down_proj.weight", {kHidden, kSharedInter});
        Add(ts, b + "mlp.shared_expert_gate.weight", {1, kHidden});
        Add(ts, b + "mlp.experts.gate_up_proj",
            {kExperts, 2 * kMoeInter, kHidden});
        Add(ts, b + "mlp.experts.down_proj", {kExperts, kHidden, kMoeInter});
    }
    if (with_mtp) {
        // The MTP head: one raw tensor (the router) and the fused expert pair,
        // which the converter would otherwise write under mtp/experts.
        Add(ts, "mtp.mlp.gate.weight", {kExperts, kHidden});
        Add(ts, "mtp.mlp.experts.gate_up_proj",
            {kExperts, 2 * kMoeInter, kHidden});
        Add(ts, "mtp.mlp.experts.down_proj", {kExperts, kHidden, kMoeInter});
    }
    synthtest::WriteSafetensors(dir, "model-00001-of-00001.safetensors", ts);
}

} // namespace alicetest
