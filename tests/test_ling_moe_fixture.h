#pragma once

// Ling-3.1-flash synthetic checkpoint fixture (converter tests). Builds the
// family-neutral primitives from test_synth_checkpoint.h at toy geometry so
// the whole FP8 -> NVFP4 pipeline runs inside a bounded convert TIMEOUT.
//
// The fixture keeps the structural facts that make Ling a family of its own:
// the 1-in-`layer_group_size` gated-MLA split (KDA otherwise), the dense-MLP
// prefix, the group-limited top-k router with an expert bias, the ungated
// shared expert, FP8 block-128 linears with partial tiles, and the MTP block
// after the last main layer.
#include "test_synth_checkpoint.h"

#include <absl/strings/str_cat.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace ling_test {

struct Cfg {
    int hidden = 256;
    int layers = 8;
    int vocab = 512;
    int heads = 2;
    int kv_heads = 2;
    int head_dim = 64;
    int qk_head_dim = 96;
    int qk_nope = 64;
    int qk_rope = 32;
    int v_head = 64;
    int q_lora = 64;
    int kv_lora = 64;
    int conv = 4;
    int experts = 8;
    int top_k = 2;
    int n_group = 2;
    int topk_group = 1;
    int moe_inter = 128;
    int shared_inter = 128;
    int dense_inter = 256;
    int first_k_dense = 1;
    int layer_group_size = 8;
    int eos = 500;
    int pad = 501;
};

inline const Cfg &TestCfg()
{
    static const Cfg c;
    return c;
}

inline int KdaProj()
{
    return TestCfg().heads * TestCfg().head_dim;
}

inline void WriteConfig(const std::filesystem::path &dir)
{
    const Cfg &c = TestCfg();
    std::ofstream f(dir / "config.json");
    f << R"({"architectures":["BailingMoeV3ForCausalLM"],)"
      << R"("model_type":"bailing_hybrid",)"
      << "\"hidden_size\":" << c.hidden
      << ",\"num_hidden_layers\":" << c.layers
      << ",\"vocab_size\":" << c.vocab
      << ",\"num_attention_heads\":" << c.heads
      << ",\"num_key_value_heads\":" << c.kv_heads
      << ",\"head_dim\":" << c.head_dim
      << ",\"qk_head_dim\":" << c.qk_head_dim
      << ",\"qk_nope_head_dim\":" << c.qk_nope
      << ",\"qk_rope_head_dim\":" << c.qk_rope
      << ",\"v_head_dim\":" << c.v_head
      << ",\"q_lora_rank\":" << c.q_lora
      << ",\"kv_lora_rank\":" << c.kv_lora
      << ",\"short_conv_kernel_size\":" << c.conv
      << ",\"num_experts\":" << c.experts
      << ",\"num_experts_per_tok\":" << c.top_k
      << ",\"n_group\":" << c.n_group
      << ",\"topk_group\":" << c.topk_group
      << ",\"moe_intermediate_size\":" << c.moe_inter
      << ",\"moe_shared_expert_intermediate_size\":" << c.shared_inter
      << ",\"intermediate_size\":" << c.dense_inter
      << ",\"first_k_dense_replace\":" << c.first_k_dense
      << ",\"layer_group_size\":" << c.layer_group_size
      << R"(,"rms_norm_eps":1e-6,"rope_theta":10000.0,)"
      << R"("routed_scaling_factor":2.5,"hidden_act":"silu",)"
      << R"("scoring_func":"sigmoid","topk_method":"noaux_tc",)"
      << R"("gated_attention_proj_granularity_type":"head_wise",)"
      << R"("norm_topk_prob":true,"tie_word_embeddings":false,)"
      << R"("use_bias":false,"use_qkv_bias":false,"no_kda_lora":true,)"
      << R"("kda_safe_gate":true,"kda_lower_bound":-5.0,)"
      << R"("linear_silu":true,"moe_router_enable_expert_bias":true,)"
      << R"("rope_interleave":true,"partial_rotary_factor":0.5,)"
      << R"("rope_scaling":null,)"
      << "\"eos_token_id\":" << c.eos
      << ",\"pad_token_id\":" << c.pad << "}\n";
}

inline std::vector<uint8_t> Fp8Bytes(int64_t n)
{
    std::vector<uint8_t> out(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        out[static_cast<size_t>(i)] = static_cast<uint8_t>(0x08 + (i % 4));
    return out;
}

inline void AddFp8(std::vector<synthtest::FakeTensor> &ts,
                   const std::string &name, int rows, int cols)
{
    synthtest::FakeTensor w;
    w.name = name;
    w.dtype = "F8_E4M3";
    w.shape = {rows, cols};
    w.data = Fp8Bytes(static_cast<int64_t>(rows) * cols);
    ts.push_back(std::move(w));

    const int sr = (rows + 127) / 128;
    const int sc = (cols + 127) / 128;
    synthtest::FakeTensor s;
    s.name = name + "_scale_inv";
    s.dtype = "F32";
    s.shape = {sr, sc};
    s.data = synthtest::MakeF32Values(sr * sc, [](int64_t i) {
        return 0.5f + 0.0625f * static_cast<float>(i % 3);
    });
    ts.push_back(std::move(s));
}

inline void AddBf16(std::vector<synthtest::FakeTensor> &ts,
                    const std::string &name, std::vector<int64_t> shape,
                    int64_t n)
{
    synthtest::FakeTensor t;
    t.name = name;
    t.dtype = "BF16";
    t.shape = std::move(shape);
    t.data = synthtest::MakeBf16Values(
        n, [](int64_t i) { return 0.01f * static_cast<float>(i % 17); });
    ts.push_back(std::move(t));
}

inline void AddF32(std::vector<synthtest::FakeTensor> &ts,
                   const std::string &name, std::vector<int64_t> shape,
                   int64_t n)
{
    synthtest::FakeTensor t;
    t.name = name;
    t.dtype = "F32";
    t.shape = std::move(shape);
    t.data = synthtest::MakeF32Values(
        n, [](int64_t i) { return 0.001f * static_cast<float>(i % 13); });
    ts.push_back(std::move(t));
}

inline std::vector<synthtest::FakeTensor> MakeTensors()
{
    const Cfg &c = TestCfg();
    const int H = c.hidden;
    const int KD = KdaProj();
    const int QD = c.heads * (c.qk_nope + c.qk_rope);
    const int KVD = c.heads * (c.qk_nope + c.v_head);
    const int VD = c.heads * c.v_head;
    const int MOE_LAYERS = c.layers - c.first_k_dense;

    std::vector<synthtest::FakeTensor> ts;
    AddBf16(ts, "model.word_embeddings.weight", {c.vocab, H},
            static_cast<int64_t>(c.vocab) * H);
    AddBf16(ts, "lm_head.weight", {c.vocab, H},
            static_cast<int64_t>(c.vocab) * H);
    AddBf16(ts, "model.norm.weight", {H}, H);

    for (int l = 0; l < c.layers; ++l) {
        const std::string p = absl::StrCat("model.layers.", l, ".");
        AddBf16(ts, p + "input_layernorm.weight", {H}, H);
        AddBf16(ts, p + "post_attention_layernorm.weight", {H}, H);
        const bool mla = (l + 1) % c.layer_group_size == 0;
        if (mla) {
            AddFp8(ts, p + "attention.q_a_proj.weight", c.q_lora, H);
            AddFp8(ts, p + "attention.q_b_proj.weight", QD, c.q_lora);
            AddFp8(ts, p + "attention.kv_a_proj_with_mqa.weight",
                   c.kv_lora + c.qk_rope, H);
            AddFp8(ts, p + "attention.kv_b_proj.weight", KVD, c.kv_lora);
            AddFp8(ts, p + "attention.dense.weight", H, VD);
            AddBf16(ts, p + "attention.q_a_layernorm.weight", {c.q_lora},
                    c.q_lora);
            AddBf16(ts, p + "attention.kv_a_layernorm.weight", {c.kv_lora},
                    c.kv_lora);
            AddBf16(ts, p + "attention.g_proj.weight", {c.heads, H},
                    static_cast<int64_t>(c.heads) * H);
        } else {
            AddFp8(ts, p + "attention.q_proj.weight", KD, H);
            AddFp8(ts, p + "attention.k_proj.weight", KD, H);
            AddFp8(ts, p + "attention.v_proj.weight", KD, H);
            AddFp8(ts, p + "attention.o_proj.weight", H, KD);
            AddFp8(ts, p + "attention.f_proj.weight", KD, H);
            AddFp8(ts, p + "attention.g_proj.weight", KD, H);
            AddBf16(ts, p + "attention.q_conv1d.weight", {KD, 1, c.conv},
                    static_cast<int64_t>(KD) * c.conv);
            AddBf16(ts, p + "attention.k_conv1d.weight", {KD, 1, c.conv},
                    static_cast<int64_t>(KD) * c.conv);
            AddBf16(ts, p + "attention.v_conv1d.weight", {KD, 1, c.conv},
                    static_cast<int64_t>(KD) * c.conv);
            AddF32(ts, p + "attention.A_log", {c.heads}, c.heads);
            AddF32(ts, p + "attention.dt_bias", {KD}, KD);
            AddBf16(ts, p + "attention.b_proj.weight", {c.heads, H},
                    static_cast<int64_t>(c.heads) * H);
            AddBf16(ts, p + "attention.o_norm.weight", {c.head_dim},
                    c.head_dim);
        }
        if (l < c.first_k_dense) {
            AddFp8(ts, p + "mlp.gate_proj.weight", c.dense_inter, H);
            AddFp8(ts, p + "mlp.up_proj.weight", c.dense_inter, H);
            AddFp8(ts, p + "mlp.down_proj.weight", H, c.dense_inter);
        } else {
            AddBf16(ts, p + "mlp.gate.weight", {c.experts, H},
                    static_cast<int64_t>(c.experts) * H);
            AddF32(ts, p + "mlp.gate.expert_bias", {c.experts}, c.experts);
            AddFp8(ts, p + "mlp.shared_experts.gate_proj.weight",
                   c.shared_inter, H);
            AddFp8(ts, p + "mlp.shared_experts.up_proj.weight",
                   c.shared_inter, H);
            AddFp8(ts, p + "mlp.shared_experts.down_proj.weight", H,
                   c.shared_inter);
            for (int e = 0; e < c.experts; ++e) {
                const std::string ep =
                    absl::StrCat(p, "mlp.experts.", e, ".");
                AddFp8(ts, ep + "gate_proj.weight", c.moe_inter, H);
                AddFp8(ts, ep + "up_proj.weight", c.moe_inter, H);
                AddFp8(ts, ep + "down_proj.weight", H, c.moe_inter);
            }
        }
    }

    // MTP block after the last main layer: stored-not-applied.
    const std::string mp = absl::StrCat("model.layers.", c.layers, ".");
    AddFp8(ts, mp + "attention.q_proj.weight", KD, H);
    AddFp8(ts, mp + "attention.k_proj.weight", KD, H);
    AddBf16(ts, mp + "eh_proj.weight", {H, 2 * H}, static_cast<int64_t>(H) * 2 * H);
    AddBf16(ts, mp + "enorm.weight", {H}, H);
    AddBf16(ts, mp + "input_layernorm.weight", {H}, H);
    (void)MOE_LAYERS;
    return ts;
}

inline void WriteCheckpoint(const std::filesystem::path &dir)
{
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    WriteConfig(dir);
    synthtest::WriteSafetensors(dir, "model-00001-of-00001.safetensors",
                                MakeTensors());
}

} // namespace ling_test
