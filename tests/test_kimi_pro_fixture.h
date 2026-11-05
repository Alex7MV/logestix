#pragma once

// Kimi-K3.1-Pro synthetic checkpoint fixture (family-owned: `kimi`).
//
// Kimi-K3.1-Pro is the Kimi-K3 tower with Gated DSA on every full-attention
// layer: the same tensors PLUS five BF16 indexer tensors per DSA layer. The
// fixture therefore reuses the K3 shard writer (`kimik3test::WriteShard`,
// family-internal) and injects only the indexer delta through its layer hook,
// then writes the `kimi_k31` / `dsa_config` variant of config.json.
//
// Geometry is tiny (LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY bypasses the frozen
// passport) but keeps every width NVFP4-aligned; `kIndexTopK` is deliberately
// 4 so the pipeline's DSA branch is reachable with an 8-token prompt.

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "test_kimi_k3_fixture.h"

namespace kimiprotest {

namespace fs = std::filesystem;

inline constexpr int kIndexHeads = 2;
inline constexpr int kIndexHeadDim = 16;
inline constexpr int kIndexTopK = 4;
inline constexpr int kIndexRopeDim = 0;
// wq_b reads the normalised q-lora vector; wk and weights_proj read the
// attention input.
inline constexpr int kIndexWqBRows = kIndexHeads * kIndexHeadDim;
inline constexpr int kIndexQLora = kimik3test::kQLora;

enum class Variant {
    kPro,              // kimi_k31 + indexer on every MLA layer (the good tree)
    kProNoIndexer,     // kimi_k31 without indexer tensors -> must fail
    kProIndexerOnKda,  // indexer tensors on a KDA layer -> must fail
    k3,                // kimi_k3 config + K3 tensors -> must fail
};

inline void WriteConfigJson(const fs::path &dir, Variant v)
{
    if (v == Variant::k3) {
        kimik3test::WriteConfigJson(dir);
        return;
    }
    // Byte-for-byte the K3 config with the wrapper model_type swapped and a
    // dsa_config block added (dsa_layers is 1-based, like full_attn_layers).
    std::string j = R"({"architectures":["KimiK31ForConditionalGeneration"],)"
        R"("bos_token_id":163584,"eos_token_id":163586,"pad_token_id":163839,)"
        R"("model_type":"kimi_k31",)"
        R"("text_config":{)"
        R"("activation_situ_beta":4.0,"activation_situ_linear_beta":25.0,)"
        R"("attn_res_block_size":2,"first_k_dense_replace":1,)"
        R"("dsa_config":{"dsa_layers":[3,4],)"
        R"("index_head_dim":16,"index_k_norm_type":"layernorm",)"
        R"("index_num_heads":2,"index_rope_dim":0,"index_skip_topk_offset":1,)"
        R"("index_topk":4,"index_topk_freq":1},)"
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
        R"("vision_config":{"vt_num_hidden_layers":1}})";
    std::ofstream f(dir / "config.json", std::ios::binary);
    f << j;
}

// The five indexer tensors of one DSA layer, in the released layout.
inline void AddIndexerTensors(int layer, std::vector<synthtest::FakeTensor> &ts)
{
    using synthtest::FakeTensor;
    const std::string A = "language_model.model.layers." +
                          std::to_string(layer) + ".self_attn.indexer.";
    auto bf16 = [](const std::string &name, int64_t rows, int64_t cols,
                   float base) {
        return FakeTensor{name, "BF16", {rows, cols},
                          synthtest::MakeBf16Values(rows * cols,
                                                    [base](int64_t i) {
                                                        return base * (1.0f +
                                                            static_cast<float>(i % 11));
                                                    })};
    };
    ts.push_back(bf16(A + "wq_b.weight", kIndexWqBRows, kIndexQLora, 0.02f));
    ts.push_back(bf16(A + "wk.weight", kIndexHeadDim, kimik3test::kHidden,
                      0.03f));
    ts.push_back(FakeTensor{A + "k_norm.weight", "BF16", {kIndexHeadDim},
                            synthtest::MakeBf16Values(kIndexHeadDim,
                                [](int64_t) { return 1.0f; })});
    ts.push_back(FakeTensor{A + "k_norm.bias", "BF16", {kIndexHeadDim},
                            synthtest::MakeBf16Values(kIndexHeadDim,
                                [](int64_t) { return 0.0f; })});
    ts.push_back(bf16(A + "weights_proj.weight", kIndexHeads,
                      kimik3test::kHidden, 0.04f));
}

inline void WriteCheckpoint(const fs::path &dir, Variant v)
{
    fs::create_directories(dir);
    WriteConfigJson(dir, v);
    kimik3test::LayerExtraFn hook;
    if (v == Variant::kPro)
        hook = [](int layer, std::vector<synthtest::FakeTensor> &ts) {
            if (kimik3test::IsMla(layer)) AddIndexerTensors(layer, ts);
        };
    else if (v == Variant::kProIndexerOnKda)
        hook = [](int layer, std::vector<synthtest::FakeTensor> &ts) {
            if (!kimik3test::IsMla(layer)) AddIndexerTensors(layer, ts);
        };
    kimik3test::WriteShard(dir, /*with_ballast=*/true, hook);
}

}  // namespace kimiprotest
