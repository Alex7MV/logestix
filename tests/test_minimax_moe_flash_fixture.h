#pragma once

// MiniMax-M3.1-Flash synthetic-checkpoint fixture (converter/loader/model
// stages). Same tiny geometry as the M3 fixture, but with the M3.1 SOURCE
// encoding: dense linear projections are MXFP8 (F8_E4M3 payload + per-32
// E8M0 exponents in `<name>.weight_scale_inv`), routed experts are already
// NVFP4 (`weight_packed` / `weight_scale` / `weight_global_scale` /
// `input_global_scale`), and sparse attention (the MSA indexer) is active on
// EVERY layer. Builds on the family-neutral primitives only.
#include "test_synth_checkpoint.h"

#include "lsxcommon/schema.h"

#include <absl/strings/str_cat.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace minimax_flash_test {

struct Cfg {
    int hidden = 64;
    int layers = 4;
    int heads = 2;
    int kv_heads = 1;
    int head_dim = 32;
    int rotary_dim = 16;
    int vocab = 64;
    int dense_inter = 32;
    int moe_inter = 32;   // MXFP8 needs cols % 32 == 0
    int shared_inter = 32;
    int experts = 8;
    int top_k = 2;
    int index_n_heads = 1;   // == kv_heads (one indexer head per GQA group)
    int index_head_dim = 32;
    int block_size = 8;
    int topk_blocks = 2;
    int local_blocks = 1;
    int first_moe_layer = 1;
    int first_sparse_layer = 0;  // M3.1: MSA on every layer
};

inline const Cfg &TestCfg()
{
    static const Cfg c;
    return c;
}

inline void WriteConfig(const std::filesystem::path &dir)
{
    const Cfg &c = TestCfg();
    std::string sparse;
    for (int l = 0; l < c.layers; ++l)
        sparse += (l ? ",1" : "1");
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
      << ",\"rms_norm_eps\":1e-06,\"rope_theta\":10000000"
      << ",\"dense_intermediate_size\":" << c.dense_inter
      << ",\"shared_intermediate_size\":" << c.shared_inter
      << ",\"num_local_experts\":" << c.experts
      << ",\"num_experts_per_tok\":" << c.top_k
      << ",\"routed_scaling_factor\":2.0"
      << ",\"swiglu_alpha\":1.702,\"swiglu_limit\":7.0"
      << ",\"moe_layer_freq\":[0,1,1,1]"
      << R"(,"sparse_attention_config":{"sparse_num_index_heads":)"
      << c.index_n_heads << ",\"sparse_index_dim\":" << c.index_head_dim
      << ",\"sparse_block_size\":" << c.block_size
      << ",\"sparse_topk_blocks\":" << c.topk_blocks
      << ",\"sparse_local_block\":" << c.local_blocks
      << ",\"sparse_attention_freq\":[" << sparse << "]}},"
      << R"("generation_config":{"bos_token_id":1,"eos_token_id":2,"pad_token_id":0}})"
      << "\n";
    f.close();
    std::ofstream g(dir / "generation_config.json");
    g << R"({"bos_token_id":1,"eos_token_id":2,"pad_token_id":0})" << "\n";
}

inline float Bf16ToFloat(uint16_t b)
{
    const uint32_t bits = static_cast<uint32_t>(b) << 16;
    float v;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

inline std::vector<float> DecodeBf16(const std::vector<uint8_t> &raw, int64_t n)
{
    std::vector<float> out(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        uint16_t b;
        std::memcpy(&b, raw.data() + static_cast<size_t>(i) * 2, 2);
        out[static_cast<size_t>(i)] = Bf16ToFloat(b);
    }
    return out;
}

inline float E8m0ToFloat(uint8_t u)
{
    return std::ldexp(1.0f, static_cast<int>(u) - 127);
}

inline uint8_t FloatToE8m0Pow2(float magnitude)
{
    if (!(magnitude > 0.0f)) return 127;  // scale 1.0
    int e = static_cast<int>(std::ceil(std::log2(magnitude)));
    int u = e + 127;
    if (u < 0) u = 0;
    if (u > 254) u = 254;
    return static_cast<uint8_t>(u);
}

// Row-major BF16 -> MXFP8 (E4M3 payload + per-32-column E8M0 exponents).
// The stored scale MULTIPLIES the payload: dequant = fp8 * 2^(u-127).
inline void QuantizeMxfp8(const std::vector<uint8_t> &bf16, int64_t rows,
                          int64_t cols, std::vector<uint8_t> &fp8,
                          std::vector<uint8_t> &scales)
{
    const std::vector<float> w = DecodeBf16(bf16, rows * cols);
    const int64_t nblocks = cols / 32;
    fp8.assign(static_cast<size_t>(rows * cols), 0);
    scales.assign(static_cast<size_t>(rows * nblocks), 0);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t b = 0; b < nblocks; ++b) {
            float mx = 0.0f;
            for (int64_t j = 0; j < 32; ++j)
                mx = std::fmax(mx, std::fabs(w[static_cast<size_t>(r * cols + b * 32 + j)]));
            // 448 (E4M3 max) is the payload ceiling.
            const uint8_t e = FloatToE8m0Pow2(mx / 400.0f);
            scales[static_cast<size_t>(r * nblocks + b)] = e;
            const float scale = E8m0ToFloat(e);
            for (int64_t j = 0; j < 32; ++j) {
                const size_t idx = static_cast<size_t>(r * cols + b * 32 + j);
                fp8[idx] = lsxcommon::float_to_e4m3(w[idx] / scale);
            }
        }
    }
}

inline int E2m1Nibble(float v)
{
    static const float kMag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const bool neg = v < 0.0f;
    const float a = std::fabs(v);
    int best = 0;
    float best_d = std::fabs(a - kMag[0]);
    for (int k = 1; k < 8; ++k) {
        const float d = std::fabs(a - kMag[k]);
        if (d < best_d) {
            best_d = d;
            best = k;
        }
    }
    return (neg ? 8 : 0) | best;
}

// Row-major BF16 -> source-form NVFP4: low-nibble-first packing, E4M3
// per-16 block scales, and an F32 global scale in compressed-tensors
// DIVISOR form (w = nibble * block_scale / weight_global_scale).
inline void QuantizeNvfp4Source(const std::vector<uint8_t> &bf16, int64_t rows,
                                int64_t cols, std::vector<uint8_t> &packed,
                                std::vector<uint8_t> &scales, float &global_scale)
{
    const std::vector<float> w = DecodeBf16(bf16, rows * cols);
    float amax = 0.0f;
    for (float v : w) amax = std::fmax(amax, std::fabs(v));
    if (!(amax > 0.0f)) amax = 1.0f;
    // 448 * 6 / amax is the ModelOpt convention; use it so block scales land
    // comfortably inside E4M3.
    global_scale = 448.0f * 6.0f / amax;
    const int64_t nblocks = cols / 16;
    packed.assign(static_cast<size_t>((rows * cols + 1) / 2), 0);
    scales.assign(static_cast<size_t>(rows * nblocks), 0);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t b = 0; b < nblocks; ++b) {
            float mx = 0.0f;
            for (int64_t j = 0; j < 16; ++j)
                mx = std::fmax(mx, std::fabs(w[static_cast<size_t>(r * cols + b * 16 + j)]));
            const float bs = mx * global_scale / 6.0f;
            scales[static_cast<size_t>(r * nblocks + b)] =
                lsxcommon::float_to_e4m3(bs);
            const float deq_scale = lsxcommon::e4m3_to_float(
                                        scales[static_cast<size_t>(r * nblocks + b)]) /
                                    global_scale;
            for (int64_t j = 0; j < 16; ++j) {
                const size_t idx = static_cast<size_t>(r * cols + b * 16 + j);
                const int nib = E2m1Nibble(w[idx] / deq_scale);
                const size_t byte = idx / 2;
                if (idx % 2 == 0)
                    packed[byte] = static_cast<uint8_t>(
                        (packed[byte] & 0xF0) | (nib & 0x0F));
                else
                    packed[byte] = static_cast<uint8_t>(
                        (packed[byte] & 0x0F) | ((nib & 0x0F) << 4));
            }
        }
    }
}

inline std::vector<synthtest::FakeTensor> Tensors()
{
    const Cfg &c = TestCfg();
    const int H = c.hidden;
    const int QW = c.heads * c.head_dim;
    const int KVW = c.kv_heads * c.head_dim;
    const int IDXQ = c.index_n_heads * c.index_head_dim;
    std::vector<synthtest::FakeTensor> t;

    auto bf16_vals = [](int64_t n, int64_t seed) {
        return synthtest::MakeBf16Values(
            n, [seed](int64_t i) {
                const int64_t k = (i * 7 + seed * 13) % 23;
                return static_cast<float>((k - 11)) * 0.011f;
            });
    };
    auto bf16 = [&](const std::string &name, std::vector<int64_t> shape,
                    int64_t seed) {
        int64_t n = 1;
        for (int64_t d : shape) n *= d;
        synthtest::FakeTensor f;
        f.name = name;
        f.dtype = "BF16";
        f.shape = std::move(shape);
        f.data = bf16_vals(n, seed);
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
    // One MXFP8 linear (payload + scale_inv companion).
    auto mxfp8 = [&](const std::string &name, int64_t rows, int64_t cols,
                     int64_t seed) {
        const auto raw = bf16_vals(rows * cols, seed);
        std::vector<uint8_t> fp8, scales;
        QuantizeMxfp8(raw, rows, cols, fp8, scales);
        synthtest::FakeTensor w;
        w.name = name + ".weight";
        w.dtype = "F8_E4M3";
        w.shape = {rows, cols};
        w.data = std::move(fp8);
        t.push_back(std::move(w));
        synthtest::FakeTensor s;
        s.name = name + ".weight_scale_inv";
        s.dtype = "U8";
        s.shape = {rows, cols / 32};
        s.data = std::move(scales);
        t.push_back(std::move(s));
    };
    // One already-quantized routed expert projection (ModelOpt NVFP4).
    auto nvfp4 = [&](const std::string &name, int64_t rows, int64_t cols,
                     int64_t seed) {
        const auto raw = bf16_vals(rows * cols, seed);
        std::vector<uint8_t> packed, scales;
        float gs = 0.0f;
        QuantizeNvfp4Source(raw, rows, cols, packed, scales, gs);
        synthtest::FakeTensor p;
        p.name = name + ".weight_packed";
        p.dtype = "U8";
        p.shape = {rows, cols / 2};
        p.data = std::move(packed);
        t.push_back(std::move(p));
        synthtest::FakeTensor s;
        s.name = name + ".weight_scale";
        s.dtype = "F8_E4M3";
        s.shape = {rows, cols / 16};
        s.data = std::move(scales);
        t.push_back(std::move(s));
        synthtest::FakeTensor g;
        g.name = name + ".weight_global_scale";
        g.dtype = "F32";
        g.shape = {};
        g.data.resize(4);
        std::memcpy(g.data.data(), &gs, 4);
        t.push_back(std::move(g));
        synthtest::FakeTensor ig;
        ig.name = name + ".input_global_scale";
        ig.dtype = "F32";
        ig.shape = {};
        const float one = 1.0f;
        ig.data.resize(4);
        std::memcpy(ig.data.data(), &one, 4);
        t.push_back(std::move(ig));
    };

    t.push_back(bf16("language_model.model.embed_tokens.weight", {c.vocab, H}, 1));
    t.push_back(bf16("language_model.lm_head.weight", {c.vocab, H}, 2));
    t.push_back(bf16("language_model.model.norm.weight", {H}, 3));
    // Ballast: counted, never applied.
    t.push_back(bf16("vision_tower.vision_model.embeddings.patch_embedding.weight",
                     {8, 3, 2, 2, 2}, 4));
    t.push_back(bf16("multi_modal_projector.linear_1.weight", {8, 8}, 5));

    for (int l = 0; l < c.layers; ++l) {
        const std::string p =
            absl::StrCat("language_model.model.layers.", l, ".");
        t.push_back(bf16(p + "input_layernorm.weight", {H}, 10 + l));
        t.push_back(bf16(p + "post_attention_layernorm.weight", {H}, 20 + l));
        mxfp8(p + "self_attn.q_proj", QW, H, 30 + l);
        mxfp8(p + "self_attn.k_proj", KVW, H, 40 + l);
        mxfp8(p + "self_attn.v_proj", KVW, H, 50 + l);
        mxfp8(p + "self_attn.o_proj", H, QW, 60 + l);
        t.push_back(bf16(p + "self_attn.q_norm.weight", {c.head_dim}, 70 + l));
        t.push_back(bf16(p + "self_attn.k_norm.weight", {c.head_dim}, 80 + l));
        // M3.1: the indexer exists on EVERY layer (sparse attention throughout).
        mxfp8(p + "self_attn.index_q_proj", IDXQ, H, 90 + l);
        mxfp8(p + "self_attn.index_k_proj", c.index_head_dim, H, 100 + l);
        t.push_back(bf16(p + "self_attn.index_q_norm.weight",
                         {c.index_head_dim}, 110 + l));
        t.push_back(bf16(p + "self_attn.index_k_norm.weight",
                         {c.index_head_dim}, 120 + l));
        if (l < c.first_moe_layer) {
            mxfp8(p + "mlp.gate_proj", c.dense_inter, H, 130 + l);
            mxfp8(p + "mlp.up_proj", c.dense_inter, H, 140 + l);
            mxfp8(p + "mlp.down_proj", H, c.dense_inter, 150 + l);
        } else {
            t.push_back(f32(p + "block_sparse_moe.gate.weight",
                            {c.experts, H}, 160 + l));
            t.push_back(f32(p + "block_sparse_moe.e_score_correction_bias",
                            {c.experts}, 170 + l));
            mxfp8(p + "block_sparse_moe.shared_experts.gate_proj",
                  c.shared_inter, H, 180 + l);
            mxfp8(p + "block_sparse_moe.shared_experts.up_proj",
                  c.shared_inter, H, 190 + l);
            mxfp8(p + "block_sparse_moe.shared_experts.down_proj",
                  H, c.shared_inter, 200 + l);
            for (int e = 0; e < c.experts; ++e) {
                const std::string pe = absl::StrCat(
                    p, "block_sparse_moe.experts.", e, ".", "w");
                nvfp4(pe + "1", c.moe_inter, H, 1000 + e * 3 + l);
                nvfp4(pe + "2", H, c.moe_inter, 1001 + e * 3 + l);
                nvfp4(pe + "3", c.moe_inter, H, 1002 + e * 3 + l);
            }
        }
    }
    return t;
}

// Two shards so the converter's cross-shard expert join is exercised.
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
    if (!a.empty())
        synthtest::WriteSafetensors(dir, "model-00001.safetensors", a);
    if (!b.empty())
        synthtest::WriteSafetensors(dir, "model-00002.safetensors", b);
    return static_cast<int>(all.size());
}

} // namespace minimax_flash_test
