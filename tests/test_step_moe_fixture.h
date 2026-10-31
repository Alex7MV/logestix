#pragma once

// Step-5-Preview synthetic-checkpoint fixture (converter/loader/model tests).
// Builds the family-neutral primitives from test_synth_checkpoint.h; geometry
// is deliberately tiny so the whole pipeline runs in a bounded TIMEOUT, and the
// frozen passport is relaxed with LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY.
//
// The routed experts mirror the real checkpoint layout: FUSED [E, N, K]
// F8_E4M3 stacks with a [E, ceil(N/128), ceil(K/128)] F32 scale grid.
#include "test_synth_checkpoint.h"

#include <absl/strings/str_cat.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace step_test {

struct Cfg {
    int hidden = 64;
    int layers = 4;
    int heads = 2;
    int kv_heads = 1;
    int head_dim = 32;
    int vocab = 64;
    int dense_inter = 32;
    int moe_inter = 16;
    int shared_inter = 16;
    int experts = 8;
    int top_k = 2;
    int first_moe_layer = 1;
    int mtp_layers = 1;
    int sliding_window = 8;
    int index_proxy = 16;
    int index_heads = 4;
    double router_scaling = 3.0;
};

inline const Cfg &TestCfg()
{
    static const Cfg c;
    return c;
}

inline std::string LayerTypesJson()
{
    // Same schedule as production: every 4th layer (starting at 3) is full.
    const Cfg &c = TestCfg();
    std::string s = "[";
    for (int l = 0; l < c.layers + c.mtp_layers; ++l) {
        if (l) s += ",";
        const bool full = l >= 3 && (l - 3) % 4 == 0;
        s += full ? "\"full_attention\"" : "\"sliding_attention\"";
    }
    s += "]";
    return s;
}

inline std::string MoeLayerListJson()
{
    const Cfg &c = TestCfg();
    std::string s = "[";
    bool first = true;
    for (int l = c.first_moe_layer; l < c.layers; ++l) {
        if (!first) s += ",";
        first = false;
        s += std::to_string(l);
    }
    s += "]";
    return s;
}

inline void WriteConfig(const std::filesystem::path &dir)
{
    const Cfg &c = TestCfg();
    std::ofstream f(dir / "config.json");
    f << R"({"architectures":["Step5ForConditionalGeneration"],)"
      << R"("model_type":"step5_vl","text_config":{)"
      << R"("architectures":["Step5ForCausalLM"],"model_type":"step5",)"
      << "\"hidden_size\":" << c.hidden
      << ",\"intermediate_size\":" << c.dense_inter
      << ",\"num_hidden_layers\":" << c.layers
      << ",\"num_attention_heads\":" << c.heads
      << ",\"num_attention_groups\":" << c.kv_heads
      << ",\"head_dim\":" << c.head_dim
      << ",\"vocab_size\":" << c.vocab
      << ",\"rms_norm_eps\":1e-05"
      << ",\"sliding_window\":" << c.sliding_window
      << ",\"moe_intermediate_size\":" << c.moe_inter
      << ",\"share_expert_dim\":" << c.shared_inter
      << ",\"moe_num_experts\":" << c.experts
      << ",\"moe_top_k\":" << c.top_k
      << ",\"moe_layer_list\":" << MoeLayerListJson()
      << ",\"num_nextn_predict_layers\":" << c.mtp_layers
      << ",\"moe_router_scaling_factor\":" << c.router_scaling
      << ",\"tie_word_embeddings\":false"
      << ",\"bos_token_id\":0,\"eos_token_id\":[1,2],\"pad_token_id\":2"
      << ",\"layer_types\":" << LayerTypesJson()
      << R"(,"sparse_config":{"enabled":true,"proxy_dim":16,"topk":4,)"
      << R"("region_block_size":8,"apply_to_layer_types":["full_attention"]}}})"
      << "\n";
    f.close();
}

inline std::vector<uint8_t> MakeFp8Stack(int64_t n, int64_t seed)
{
    std::vector<uint8_t> out(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        // Small non-negative E4M3 bit patterns (0x00..0x38 are finite).
        out[static_cast<size_t>(i)] =
            static_cast<uint8_t>((i * 3 + seed * 5) % 0x39);
    }
    return out;
}

inline std::vector<synthtest::FakeTensor> Tensors()
{
    const Cfg &c = TestCfg();
    const int H = c.hidden;
    const int QW = c.heads * c.head_dim;
    const int KVW = c.kv_heads * c.head_dim;
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
                return static_cast<float>(k - 8) * 0.002f;
            });
        return f;
    };

    t.push_back(bf16("model.embed_tokens.weight", {c.vocab, H}, 1));
    t.push_back(bf16("lm_head.weight", {c.vocab, H}, 2));
    t.push_back(f32("model.norm.weight", {H}, 3));

    for (int l = 0; l < c.layers; ++l) {
        const std::string p = absl::StrCat("model.layers.", l, ".");
        t.push_back(f32(p + "input_layernorm.weight", {H}, 10 + l));
        t.push_back(f32(p + "post_attention_layernorm.weight", {H}, 20 + l));
        t.push_back(bf16(p + "self_attn.q_proj.weight", {QW, H}, 30 + l));
        t.push_back(bf16(p + "self_attn.k_proj.weight", {KVW, H}, 40 + l));
        t.push_back(bf16(p + "self_attn.v_proj.weight", {KVW, H}, 50 + l));
        t.push_back(bf16(p + "self_attn.o_proj.weight", {H, QW}, 60 + l));
        t.push_back(bf16(p + "self_attn.g_proj.weight", {c.heads, H}, 65 + l));
        t.push_back(f32(p + "self_attn.q_norm.weight", {c.head_dim}, 70 + l));
        t.push_back(f32(p + "self_attn.k_norm.weight", {c.head_dim}, 80 + l));
        if (l >= 3 && (l - 3) % 4 == 0) {
            t.push_back(bf16(p + "self_attn.sparse_indexer_q.weight",
                             {c.index_heads * c.index_proxy, H}, 90 + l));
            t.push_back(bf16(p + "self_attn.sparse_indexer_k.weight",
                             {c.index_proxy, H}, 95 + l));
            t.push_back(bf16(p + "self_attn.sparse_indexer_z.weight",
                             {c.index_proxy, H}, 96 + l));
            t.push_back(f32(p + "self_attn.sparse_indexer_w.weight",
                            {c.index_heads, H}, 97 + l));
            t.push_back(f32(p + "self_attn.sparse_indexer_q_norm.weight",
                            {c.index_proxy}, 98 + l));
            t.push_back(f32(p + "self_attn.sparse_indexer_k_norm.weight",
                            {c.index_proxy}, 99 + l));
            t.push_back(f32(p + "self_attn.sparse_indexer_k_norm.bias",
                            {c.index_proxy}, 100 + l));
            t.push_back(f32(p + "self_attn.ssmax_s", {c.heads}, 101 + l));
        }
        if (l < c.first_moe_layer) {
            t.push_back(bf16(p + "mlp.gate_proj.weight", {c.dense_inter, H},
                             130 + l));
            t.push_back(bf16(p + "mlp.up_proj.weight", {c.dense_inter, H},
                             140 + l));
            t.push_back(bf16(p + "mlp.down_proj.weight", {H, c.dense_inter},
                             150 + l));
        } else {
            t.push_back(bf16(p + "moe.gate.weight", {c.experts, H}, 160 + l));
            t.push_back(f32(p + "moe.router_bias", {c.experts}, 170 + l));
            t.push_back(bf16(p + "share_expert.gate_proj.weight",
                             {c.shared_inter, H}, 180 + l));
            t.push_back(bf16(p + "share_expert.up_proj.weight",
                             {c.shared_inter, H}, 190 + l));
            t.push_back(bf16(p + "share_expert.down_proj.weight",
                             {H, c.shared_inter}, 200 + l));
            // Fused FP8 expert stacks + their F32 scale grids.
            const int64_t up_n = static_cast<int64_t>(c.experts) * c.moe_inter * H;
            const int64_t dn_n = static_cast<int64_t>(c.experts) * H * c.moe_inter;
            const int64_t sc_n = static_cast<int64_t>(c.experts) *
                                 ((c.moe_inter + 127) / 128) * ((H + 127) / 128);
            const int64_t sc_dn = static_cast<int64_t>(c.experts) *
                                  ((H + 127) / 128) * ((c.moe_inter + 127) / 128);
            synthtest::FakeTensor g;
            g.name = p + "moe.gate_proj.weight";
            g.dtype = "F8_E4M3";
            g.shape = {c.experts, c.moe_inter, H};
            g.data = MakeFp8Stack(up_n, l + 1);
            t.push_back(g);
            synthtest::FakeTensor u = g;
            u.name = p + "moe.up_proj.weight";
            u.data = MakeFp8Stack(up_n, l + 2);
            t.push_back(u);
            synthtest::FakeTensor d;
            d.name = p + "moe.down_proj.weight";
            d.dtype = "F8_E4M3";
            d.shape = {c.experts, H, c.moe_inter};
            d.data = MakeFp8Stack(dn_n, l + 3);
            t.push_back(d);
            t.push_back(f32(p + "moe.gate_proj.weight_scale_inv",
                            {c.experts, (c.moe_inter + 127) / 128,
                             (H + 127) / 128}, l + 4));
            t.push_back(f32(p + "moe.up_proj.weight_scale_inv",
                            {c.experts, (c.moe_inter + 127) / 128,
                             (H + 127) / 128}, l + 5));
            // The scale values must be positive and small: the fixture writes
            // (k-8)*0.002 which can be negative; scales are magnitudes.
            auto &sg = t[t.size() - 2];
            auto &su = t[t.size() - 1];
            for (auto *rec : {&sg, &su}) {
                for (size_t i = 0; i < rec->data.size(); i += 4) {
                    float v = 1.0f + static_cast<float>(i % 5) * 0.01f;
                    std::memcpy(rec->data.data() + i, &v, 4);
                }
            }
            synthtest::FakeTensor ds;
            ds.name = p + "moe.down_proj.weight_scale_inv";
            ds.dtype = "F32";
            ds.shape = {c.experts, (H + 127) / 128, (c.moe_inter + 127) / 128};
            ds.data = synthtest::MakeF32Values(sc_dn, [](int64_t i) {
                return 1.0f + static_cast<float>(i % 5) * 0.01f;
            });
            t.push_back(std::move(ds));
            (void)sc_n;
        }
    }

    // Ballast: vision + one MTP layer (index == layers).
    t.push_back(bf16("vision_model.conv1.weight", {4, 3, 2, 2}, 900));
    t.push_back(bf16("vit_large_projector.weight", {H, 8}, 901));
    const std::string mp = absl::StrCat("model.layers.", c.layers, ".");
    t.push_back(f32(mp + "enorm.weight", {H}, 910));
    t.push_back(f32(mp + "hnorm.weight", {H}, 911));
    t.push_back(bf16(mp + "eh_proj.weight", {H, 2 * H}, 912));
    t.push_back(bf16(mp + "transformer.shared_head.output.weight",
                     {c.vocab, H}, 913));
    return t;
}

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

} // namespace step_test
