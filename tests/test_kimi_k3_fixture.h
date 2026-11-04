#pragma once

// Kimi-K3 synthetic checkpoint fixture (family-owned, built on the
// model-neutral synthtest primitives). Mimics the released layout:
// `language_model.model.layers.N.*`, MXFP4 (E2M1 + E8M0/32) routed experts,
// BF16 everything else, an MLA layer at 0-based N=2 and a KDA layer at N=0/1,
// plus vision/mm_projector ballast that the converter must strip and count.
//
// Geometry is deliberately tiny but every width is a multiple of 32 so the
// NVFP4 per-16 blocks and the MXFP4 per-32 scale groups are exact.

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include "test_synth_checkpoint.h"

namespace kimik3test {

namespace fs = std::filesystem;

// Tiny geometry (synthetic mode; the frozen passport is bypassed by
// LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY).
inline constexpr int kHidden = 64;
inline constexpr int kLayers = 4;
inline constexpr int kVocab = 48;
inline constexpr int kDenseLayers = 1;
inline constexpr int kAttnResBlock = 2;
inline constexpr int kMlaHeads = 2;
inline constexpr int kQLora = 16;
inline constexpr int kKvLora = 32;
inline constexpr int kQkNope = 8;
inline constexpr int kQkRope = 4;
inline constexpr int kVHead = 8;
inline constexpr int kKdaHeads = 2;
inline constexpr int kKdaHeadDim = 16;
inline constexpr int kConv = 4;
inline constexpr int kExperts = 4;
inline constexpr int kTopK = 2;
inline constexpr int kMoeInter = 32;
inline constexpr int kLatent = 64;
inline constexpr int kSharedInter = 64;
inline constexpr int kDenseInter = 64;

// 0-based MLA layers of the fixture: {2,3} (config lists 1-based {3,4}).
inline constexpr bool IsMla(int layer) { return layer == 2 || layer == 3; }

inline constexpr int QBrows() { return kMlaHeads * (kQkNope + kQkRope); }
inline constexpr int KvBrows() { return kMlaHeads * (kQkNope + kVHead); }
inline constexpr int KdaRows() { return kKdaHeads * kKdaHeadDim; }

inline void WriteConfigJson(const fs::path &dir)
{
    std::string j = R"({"architectures":["KimiK3ForConditionalGeneration"],)"
        R"("bos_token_id":163584,"eos_token_id":163586,"pad_token_id":163839,)"
        R"("model_type":"kimi_k3",)"
        R"("text_config":{)"
        R"("activation_situ_beta":4.0,"activation_situ_linear_beta":25.0,)"
        R"("attn_res_block_size":2,"first_k_dense_replace":1,)"
        R"("hidden_act":"situ","hidden_size":64,)"
        R"("intermediate_size":64,"kv_lora_rank":32,)"
        R"("latent_moe_use_norm":true,)"
        R"("linear_attn_config":{"full_attn_layers":[3,4],)"
        R"("gate_lower_bound":-5.0,"head_dim":16,)"
        R"("kda_layers":[1,2],"num_heads":2,"short_conv_kernel_size":4,)"
        R"("use_full_rank_gate":true},)"
        R"("mla_use_nope":true,"mla_use_output_gate":true,)"
        R"("model_type":"kimi_linear","moe_intermediate_size":32,)"
        R"("moe_layer_freq":1,"moe_renormalize":true,)"
        R"("moe_router_activation_func":"sigmoid",)"
        R"("num_attention_heads":2,"num_expert_group":1,"num_experts":4,)"
        R"("num_experts_per_token":2,"num_hidden_layers":4,)"
        R"("num_key_value_heads":2,"num_shared_experts":2,)"
        R"("q_lora_rank":16,"qk_nope_head_dim":8,"qk_rope_head_dim":4,)"
        R"("rms_norm_eps":1e-05,"routed_expert_hidden_size":64,)"
        R"("routed_scaling_factor":1.0,"tie_word_embeddings":false,)"
        R"("topk_group":1,"topk_method":"noaux_tc",)"
        R"("use_grouped_topk":true,"v_head_dim":8,"vocab_size":48},)"
        R"("tie_word_embeddings":false,)"
        R"("vision_config":{"vt_num_hidden_layers":1}})"
        ;
    std::ofstream f(dir / "config.json", std::ios::binary);
    f << j;
}

inline std::vector<uint8_t> MakeF32Scaled(int64_t n, float base)
{
    return synthtest::MakeF32Values(n, [base](int64_t i) {
        return base + static_cast<float>(i % 7) * 0.5f;
    });
}

// MXFP4 pack: element 2i in the low nibble, element 2i+1 in the high nibble.
inline std::vector<uint8_t> MakeMxfp4Packed(int64_t rows, int64_t cols)
{
    std::vector<uint8_t> out(static_cast<size_t>(rows * (cols / 2)));
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t c = 0; c < cols / 2; ++c) {
            const uint8_t lo = static_cast<uint8_t>((r + c) % 16);
            const uint8_t hi = static_cast<uint8_t>((r + 2 * c + 5) % 16);
            out[static_cast<size_t>(r * (cols / 2) + c)] =
                static_cast<uint8_t>(lo | (hi << 4));
        }
    return out;
}

// E8M0 exponent bytes with a spread well inside the exact-repack window (<=13).
inline std::vector<uint8_t> MakeE8m0Scales(int64_t rows, int64_t groups)
{
    std::vector<uint8_t> out(static_cast<size_t>(rows * groups));
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t g = 0; g < groups; ++g)
            out[static_cast<size_t>(r * groups + g)] =
                static_cast<uint8_t>(126 + (r + g) % 4);
    return out;
}

// Appended to the tensor list of every layer (before that layer's own
// tensors). A sibling variant (Kimi-K3.1-Pro's DSA indexer) injects its extra
// tensors through this hook instead of duplicating the whole shard writer; the
// K3 fixture itself never mentions them.
using LayerExtraFn =
    std::function<void(int layer, std::vector<synthtest::FakeTensor> &ts)>;

inline void WriteShard(const fs::path &dir, bool with_ballast = true,
                       const LayerExtraFn &extra_layer = {})
{
    using synthtest::FakeTensor;
    std::vector<FakeTensor> ts;

    auto bf16 = [](const std::string &name, int64_t rows, int64_t cols,
                   float base = 0.01f) {
        return FakeTensor{name, "BF16", {rows, cols},
                          synthtest::MakeBf16Values(rows * cols,
                                                    [base](int64_t i) {
                                                        return base * (1.0f +
                                                            static_cast<float>(i % 13));
                                                    })};
    };
    const std::string P = "language_model.";

    ts.push_back(bf16(P + "model.embed_tokens.weight", kVocab, kHidden));
    ts.push_back(bf16(P + "lm_head.weight", kVocab, kHidden));
    ts.push_back(synthtest::FakeTensor{P + "model.norm.weight", "BF16", {kHidden},
                                       synthtest::MakeBf16Values(kHidden,
                                           [](int64_t) { return 1.0f; })});
    ts.push_back(bf16(P + "model.output_attn_res_proj.weight", 1, kHidden));
    ts.push_back(synthtest::FakeTensor{P + "model.output_attn_res_norm.weight",
                                       "BF16", {kHidden},
                                       synthtest::MakeBf16Values(kHidden,
                                           [](int64_t) { return 1.0f; })});

    for (int l = 0; l < kLayers; ++l) {
        const std::string L = P + "model.layers." + std::to_string(l) + ".";
        if (extra_layer) extra_layer(l, ts);
        ts.push_back(synthtest::FakeTensor{L + "input_layernorm.weight", "BF16",
                                           {kHidden},
                                           synthtest::MakeBf16Values(kHidden,
                                               [](int64_t) { return 1.0f; })});
        ts.push_back(synthtest::FakeTensor{L + "post_attention_layernorm.weight",
                                           "BF16", {kHidden},
                                           synthtest::MakeBf16Values(kHidden,
                                               [](int64_t) { return 1.0f; })});
        ts.push_back(bf16(L + "self_attention_res_proj.weight", 1, kHidden));
        ts.push_back(bf16(L + "self_attention_res_norm.weight", 1, kHidden));
        ts.push_back(bf16(L + "mlp_res_proj.weight", 1, kHidden));
        ts.push_back(bf16(L + "mlp_res_norm.weight", 1, kHidden));

        if (IsMla(l)) {
            const std::string A = L + "self_attn.";
            ts.push_back(bf16(A + "q_a_proj.weight", kQLora, kHidden));
            ts.push_back(synthtest::FakeTensor{A + "q_a_layernorm.weight", "BF16",
                                               {kQLora},
                                               synthtest::MakeBf16Values(kQLora,
                                                   [](int64_t) { return 1.0f; })});
            ts.push_back(bf16(A + "q_b_proj.weight", QBrows(), kQLora));
            ts.push_back(bf16(A + "kv_a_proj_with_mqa.weight",
                              kKvLora + kQkRope, kHidden));
            ts.push_back(synthtest::FakeTensor{A + "kv_a_layernorm.weight", "BF16",
                                               {kKvLora},
                                               synthtest::MakeBf16Values(kKvLora,
                                                   [](int64_t) { return 1.0f; })});
            ts.push_back(bf16(A + "kv_b_proj.weight", KvBrows(), kKvLora));
            ts.push_back(bf16(A + "o_proj.weight", kHidden, kMlaHeads * kVHead));
            ts.push_back(bf16(A + "g_proj.weight", kMlaHeads * kVHead, kHidden));
        } else {
            const std::string A = L + "self_attn.";
            ts.push_back(bf16(A + "q_proj.weight", KdaRows(), kHidden));
            ts.push_back(bf16(A + "k_proj.weight", KdaRows(), kHidden));
            ts.push_back(bf16(A + "v_proj.weight", KdaRows(), kHidden));
            ts.push_back(bf16(A + "o_proj.weight", kHidden, KdaRows()));
            ts.push_back(bf16(A + "g_proj.weight", KdaRows(), kHidden));
            ts.push_back(bf16(A + "f_a_proj.weight", kKdaHeadDim, kHidden));
            ts.push_back(bf16(A + "f_b_proj.weight", KdaRows(), kKdaHeadDim));
            ts.push_back(bf16(A + "b_proj.weight", kKdaHeads, kHidden));
            ts.push_back(synthtest::FakeTensor{A + "q_conv1d.weight", "F32",
                                               {KdaRows(), 1, kConv},
                                               MakeF32Scaled(KdaRows() * kConv, 0.1f)});
            ts.push_back(synthtest::FakeTensor{A + "k_conv1d.weight", "F32",
                                               {KdaRows(), 1, kConv},
                                               MakeF32Scaled(KdaRows() * kConv, 0.1f)});
            ts.push_back(synthtest::FakeTensor{A + "v_conv1d.weight", "F32",
                                               {KdaRows(), 1, kConv},
                                               MakeF32Scaled(KdaRows() * kConv, 0.1f)});
            ts.push_back(synthtest::FakeTensor{A + "A_log", "F32", {kKdaHeadDim},
                                               MakeF32Scaled(kKdaHeadDim, 0.5f)});
            ts.push_back(synthtest::FakeTensor{A + "dt_bias", "F32", {KdaRows()},
                                               MakeF32Scaled(KdaRows(), 0.2f)});
            ts.push_back(synthtest::FakeTensor{A + "o_norm.weight", "F32",
                                               {kKdaHeadDim},
                                               MakeF32Scaled(kKdaHeadDim, 1.0f)});
        }

        if (l < kDenseLayers) {
            const std::string M = L + "mlp.";
            ts.push_back(bf16(M + "gate_proj.weight", kDenseInter, kHidden));
            ts.push_back(bf16(M + "up_proj.weight", kDenseInter, kHidden));
            ts.push_back(bf16(M + "down_proj.weight", kHidden, kDenseInter));
            continue;
        }

        const std::string B = L + "block_sparse_moe.";
        ts.push_back(bf16(B + "gate.weight", kExperts, kHidden));
        ts.push_back(synthtest::FakeTensor{B + "gate.e_score_correction_bias",
                                           "F32", {kExperts},
                                           MakeF32Scaled(kExperts, 0.05f)});
        ts.push_back(bf16(B + "routed_expert_down_proj.weight", kLatent, kHidden));
        ts.push_back(bf16(B + "routed_expert_up_proj.weight", kHidden, kLatent));
        ts.push_back(synthtest::FakeTensor{B + "routed_expert_norm.weight", "BF16",
                                           {kLatent},
                                           synthtest::MakeBf16Values(kLatent,
                                               [](int64_t) { return 1.0f; })});
        ts.push_back(bf16(B + "shared_experts.gate_proj.weight", kSharedInter, kHidden));
        ts.push_back(bf16(B + "shared_experts.up_proj.weight", kSharedInter, kHidden));
        ts.push_back(bf16(B + "shared_experts.down_proj.weight", kHidden, kSharedInter));
        for (int e = 0; e < kExperts; ++e) {
            const std::string E = B + "experts." + std::to_string(e) + ".";
            ts.push_back(synthtest::FakeTensor{E + "w1.weight_packed", "U8",
                                               {kMoeInter, kLatent / 2},
                                               MakeMxfp4Packed(kMoeInter, kLatent)});
            ts.push_back(synthtest::FakeTensor{E + "w1.weight_scale", "U8",
                                               {kMoeInter, kLatent / 32},
                                               MakeE8m0Scales(kMoeInter, kLatent / 32)});
            ts.push_back(synthtest::FakeTensor{E + "w3.weight_packed", "U8",
                                               {kMoeInter, kLatent / 2},
                                               MakeMxfp4Packed(kMoeInter, kLatent)});
            ts.push_back(synthtest::FakeTensor{E + "w3.weight_scale", "U8",
                                               {kMoeInter, kLatent / 32},
                                               MakeE8m0Scales(kMoeInter, kLatent / 32)});
            ts.push_back(synthtest::FakeTensor{E + "w2.weight_packed", "U8",
                                               {kLatent, kMoeInter / 2},
                                               MakeMxfp4Packed(kLatent, kMoeInter)});
            ts.push_back(synthtest::FakeTensor{E + "w2.weight_scale", "U8",
                                               {kLatent, kMoeInter / 32},
                                               MakeE8m0Scales(kLatent, kMoeInter / 32)});
        }
    }

    if (with_ballast) {
        ts.push_back(bf16("vision_tower.encoder.blocks.0.wo.weight", 8, 8));
        ts.push_back(bf16("mm_projector.proj.0.weight", 8, 8));
    }

    synthtest::WriteSafetensors(dir, "model-00001-of-00001.safetensors", ts);
}

inline void WriteCheckpoint(const fs::path &dir)
{
    fs::create_directories(dir);
    WriteConfigJson(dir);
    WriteShard(dir);
}

}  // namespace kimik3test
