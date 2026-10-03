#pragma once

// Shared fixture builder for Xing4.0-29B-A4B converter/loader tests.
// Uses the family-neutral synthetic-checkpoint primitives
// (test_synth_checkpoint.h — suite-autonomy invariant: no borrowing from other
// model families' fixtures) and adds Xing-specific pieces:
//   - flat "model.layers.N.*" names (no language_model prefix, no VL tower)
//   - mHC tensors per layer: attn_hc/ffn_hc.{hc_fn,hc_base,hc_scale}
//     (hc_fn [24, 4H], hc_base [24], hc_scale [3]) and NO input_norm tensor
//     (Xing4_0UnweightedRMSNorm is weight-free — its absence is a geometry
//     fact worth pinning, so the fixture never writes one)
//   - MLA projections q_a/q_b/kv_a_proj_with_mqa/kv_b/o + two inner norms
//   - dense SwiGLU on layers 0..1, 64 routed + 1 shared expert elsewhere
//   - the stored MTP block model.layers.40.*

#include "test_synth_checkpoint.h"

#include <absl/strings/str_cat.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace xingtest {

using synthtest::FakeTensor;
using synthtest::MakeBf16;
using synthtest::MakeF32;
using synthtest::WriteSafetensors;

// hc_fn [mix, hc_mult*H]; hc_base [mix]; hc_scale [3] (F32, like the source).
inline void AppendHyperConnection(std::vector<FakeTensor> &t,
                                  const std::string &prefix, int hc_mult,
                                  int hidden)
{
    const int mix = (2 + hc_mult) * hc_mult;
    t.push_back({prefix + ".hc_fn", "BF16", {mix, hc_mult * hidden},
                 MakeBf16(static_cast<int64_t>(mix) * hc_mult * hidden)});
    t.push_back({prefix + ".hc_base", "BF16", {mix}, MakeBf16(mix)});
    t.push_back({prefix + ".hc_scale", "F32", {3}, MakeF32(3)});
}

inline void AppendXingAttention(std::vector<FakeTensor> &t, int l)
{
    const auto p = absl::StrCat("model.layers.", l, ".");
    t.push_back({p + "self_attn.q_a_proj.weight",           "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({p + "self_attn.q_a_layernorm.weight",      "BF16", {16},    MakeBf16(16)});
    t.push_back({p + "self_attn.q_b_proj.weight",           "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({p + "self_attn.kv_a_proj_with_mqa.weight", "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({p + "self_attn.kv_a_layernorm.weight",     "BF16", {16},    MakeBf16(16)});
    t.push_back({p + "self_attn.kv_b_proj.weight",          "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({p + "self_attn.o_proj.weight",             "BF16", {16, 16}, MakeBf16(256)});
    AppendHyperConnection(t, p + "attn_hc", 4, 16);
    AppendHyperConnection(t, p + "ffn_hc", 4, 16);
    t.push_back({p + "input_layernorm.weight",          "BF16", {16}, MakeBf16(16)});
    t.push_back({p + "post_attention_layernorm.weight", "BF16", {16}, MakeBf16(16)});
}

inline void AppendXingDenseMlp(std::vector<FakeTensor> &t, int l)
{
    const auto p = absl::StrCat("model.layers.", l, ".mlp.");
    t.push_back({p + "gate_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({p + "up_proj.weight",   "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({p + "down_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
}

inline void AppendXingMoeFfn(std::vector<FakeTensor> &t, int l, int num_experts)
{
    const auto p = absl::StrCat("model.layers.", l, ".mlp.");
    t.push_back({p + "gate.weight",                  "BF16", {num_experts, 16},
                 MakeBf16(static_cast<int64_t>(num_experts) * 16)});
    t.push_back({p + "gate.e_score_correction_bias", "F32", {num_experts},
                 MakeF32(num_experts)});
    for (int e = 0; e < num_experts; ++e) {
        const auto ep = absl::StrCat(p, "experts.", e, ".");
        t.push_back({ep + "gate_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
        t.push_back({ep + "up_proj.weight",   "BF16", {16, 16}, MakeBf16(256)});
        t.push_back({ep + "down_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
    }
    t.push_back({p + "shared_experts.gate_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({p + "shared_experts.up_proj.weight",   "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({p + "shared_experts.down_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
}

// The MTP block the converter must store + count but never execute.
inline void AppendXingMtp(std::vector<FakeTensor> &t, int l)
{
    const auto p = absl::StrCat("model.layers.", l, ".");
    t.push_back({p + "embed_tokens.weight", "BF16", {64, 16}, MakeBf16(64 * 16)});
    t.push_back({p + "enorm.weight",        "BF16", {16},    MakeBf16(16)});
    t.push_back({p + "hnorm.weight",        "BF16", {16},    MakeBf16(16)});
    t.push_back({p + "eh_proj.weight",      "BF16", {16, 32}, MakeBf16(512)});
    t.push_back({p + "shared_head.head.weight", "BF16", {64, 16}, MakeBf16(64 * 16)});
    t.push_back({p + "shared_head.norm.weight", "BF16", {16},    MakeBf16(16)});
}

// num_layers model layers (2 dense then MoE, mirroring first_k_dense_replace),
// plus the stored MTP layer at <mtp_layer>.
inline std::vector<FakeTensor> BuildSyntheticXingModel(int num_layers = 3,
                                                       int num_experts = 2,
                                                       int mtp_layer = 40,
                                                       int hc_mult = 4,
                                                       int hidden = 16)
{
    std::vector<FakeTensor> t;
    t.push_back({"model.embed_tokens.weight", "BF16", {64, hidden},
                 MakeBf16(64 * hidden)});
    t.push_back({"lm_head.weight", "BF16", {64, hidden},
                 MakeBf16(64 * hidden)});
    t.push_back({"model.norm.weight", "BF16", {hidden}, MakeBf16(hidden)});
    for (int l = 0; l < num_layers; ++l) {
        AppendXingAttention(t, l);
        if (l < 2)
            AppendXingDenseMlp(t, l);
        else
            AppendXingMoeFfn(t, l, num_experts);
    }
    AppendXingMtp(t, mtp_layer);
    return t;
}

inline void WriteXingConfig(const std::filesystem::path &dir)
{
    // Frozen values that the converter cross-checks are present verbatim;
    // tensor shapes stay tiny (the fixture exercises the pipeline, not the
    // geometry).
    std::ofstream f(dir / "config.json");
    f << R"({"architectures":["Xing4_0ForCausalLM"],)"
      << R"("model_type":"xing4_0",)"
      << R"("hidden_size":3584,"num_hidden_layers":40,"vocab_size":131072,)"
      << R"("num_attention_heads":32,"num_key_value_heads":32,)"
      << R"("q_lora_rank":768,"kv_lora_rank":512,)"
      << R"("qk_nope_head_dim":128,"qk_rope_head_dim":64,"v_head_dim":128,)"
      << R"("first_k_dense_replace":2,"moe_intermediate_size":1024,)"
      << R"("n_routed_experts":64,"num_experts_per_tok":4,"n_shared_experts":1,)"
      << R"("routed_scaling_factor":2.0,"norm_topk_prob":true,)"
      << R"("scoring_func":"sigmoid","topk_method":"noaux_tc",)"
      << R"("hc_mult":4,"hc_sinkhorn_iters":20,"hc_eps":1e-06,)"
      << R"("mhc_h_res_clamp_min":-30,"mhc_h_res_clamp_max":30,)"
      << R"("bos_token_id":1,"eos_token_id":2,"rope_interleave":true})";
}

} // namespace xingtest