#pragma once

// Inkling (base) synthetic-checkpoint fixture. Family-neutral primitives come
// from test_synth_checkpoint.h; this header only knows the base checkpoint's
// tensor names and a legal small geometry: 6 layers so the sliding/full split
// AND the dense-MLP prefix (layers 0..1) are both exercised, both kv widths
// (sliding 4 / full 2), 4 routed experts top-2 plus 2 shared-sink experts.
//
// The real checkpoint's numbers live in src/inkling_moe_base_constants.h; the
// converter relaxes its frozen cross-check under
// LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1 and reads the *parsed* geometry.
//
// ROW LAYOUTS (the trap this fixture exists to pin, verified against
// transformers' conversion_mapping.py "inkling_mm_model" on 2026-11-17):
//   * routed `mlp.experts.w13_weight [E, 2I, H]` is HALVES: rows [0,I) are the
//     gate plane, rows [I,2I) the up plane (the mapping applies NO Interleave
//     to it — the target gate_up_proj chunks in halves);
//   * `shared_experts.shared_w13_weight [S, 2I, H]` is ROW-INTERLEAVED
//     (Interleave(dim=1) + Chunk: source rows alternate gate, up);
//   * dense `mlp.w13_dn.weight [2*denseI, H]` is ROW-INTERLEAVED
//     (Interleave(dim=0) + Chunk).
// Every weight matrix carries a row-identifiable pattern (RowValue below) so
// a wrong split is detectable by reconstruction, not just by size.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <absl/strings/str_cat.h>

#include "test_synth_checkpoint.h"

namespace inklingbasetest {

// ── synthetic geometry ─────────────────────────────────────────────────────
inline constexpr int kHidden = 64;
inline constexpr int kLayers = 6;
inline constexpr int kHeads = 4;
inline constexpr int kKvHeadsFull = 2;
inline constexpr int kKvHeadsSliding = 4;
inline constexpr int kHeadDim = 16;
inline constexpr int kDRel = 4;
inline constexpr int kRelExtent = 8;
inline constexpr int kWindow = 8;
inline constexpr int kConvK = 4;
inline constexpr int kExperts = 4;
inline constexpr int kTopK = 2;
inline constexpr int kShared = 2;
inline constexpr int kMoeInter = 32;
inline constexpr int kDenseLayers = 2;   // layers 0..1 dense, 2..5 MoE
inline constexpr int kDenseInter = 128;
inline constexpr int kVocab = 256;
inline constexpr int kUnpadded = 200;
inline constexpr int kFullLayer = 5;

inline constexpr int kQDim = kHeads * kHeadDim;                 // 64
inline constexpr int kKvDimFull = kKvHeadsFull * kHeadDim;      // 32
inline constexpr int kKvDimSliding = kKvHeadsSliding * kHeadDim; // 64
inline constexpr int kRelDim = kHeads * kDRel;                  // 16

inline bool IsFull(int layer) { return layer == kFullLayer; }
inline bool IsDense(int layer) { return layer < kDenseLayers; }
inline int KvDim(int layer) { return IsFull(layer) ? kKvDimFull : kKvDimSliding; }
inline int RelExtentOf(int layer) { return IsFull(layer) ? kRelExtent : kWindow; }

inline const char *Shard0() { return "model-00001-of-00001.safetensors"; }

inline std::string Llm(const std::string &rest) { return "model.llm." + rest; }

inline std::string Layer(int l, const std::string &rest)
{
    return absl::StrCat("model.llm.layers.", l, ".", rest);
}

inline void Push(std::vector<synthtest::FakeTensor> &out, std::string name,
                 std::string dtype, std::vector<int64_t> shape,
                 std::vector<uint8_t> data)
{
    out.push_back(synthtest::FakeTensor{std::move(name), std::move(dtype),
                                        std::move(shape), std::move(data)});
}

// Row-identifiable weight value: row r, column c, tag t. Bounded
// (~±0.08) so the NVFP4 quantization is well-conditioned. The hundreds digit
// of the tag (the plane family: 1xx/2xx routed gate/up, 3xx/4xx shared
// gate/up, 5xx/6xx dense gate/up) flips the sign of the whole row pattern,
// so gate vs up planes are ~2.0 apart in relative L2 in EVERY row — a
// positive-only encoding leaves adjacent tags a few percent apart and a
// swapped split would pass the pins.
inline float RowValue(int64_t r, int64_t c, int64_t tag)
{
    const float sign = (((tag / 100) & 1) ? -1.0f : 1.0f);
    return sign * (0.02f + 0.004f * static_cast<float>(r % 16) +
                   0.0008f * static_cast<float>(c % 8));
}

// A [rows, cols] BF16 matrix whose element (r,c) is RowValue(r,c,tag).
inline std::vector<uint8_t> RowMatrix(int64_t rows, int64_t cols, int64_t tag)
{
    return synthtest::MakeBf16Values(rows * cols, [cols, tag](int64_t i) {
        return RowValue(i / cols, i % cols, tag);
    });
}

// The routed expert slab [E, 2I, H] ROW-INTERLEAVED: source row 2r is gate
// row r (tag 100+e), source row 2r+1 is up row r (tag 200+e). The release
// conversion_mapping applies Interleave(dim=1) to de-interleave it to
// [gate;up] halves before the forward chunks the output — the SAME layout
// family as shared_w13 / dense w13_dn (verified against the release op on
// the real checkpoint, 2026-11-18).
inline std::vector<uint8_t> RoutedW13()
{
    const int64_t E = kExperts, I = kMoeInter, H = kHidden;
    return synthtest::MakeBf16Values(E * 2 * I * H, [I, H](int64_t i) {
        const int64_t e = i / (2 * I * H);
        const int64_t r = (i / H) % (2 * I);
        const int64_t c = i % H;
        const bool up = (r & 1) != 0;
        return RowValue(r / 2, c, (up ? 200 : 100) + e);
    });
}

// The shared expert slab [S, 2I, H] ROW-INTERLEAVED: source row 2r is gate
// row r (tag 300+s), source row 2r+1 is up row r (tag 400+s).
inline std::vector<uint8_t> SharedW13()
{
    const int64_t S = kShared, I = kMoeInter, H = kHidden;
    return synthtest::MakeBf16Values(S * 2 * I * H, [I, H](int64_t i) {
        const int64_t s = i / (2 * I * H);
        const int64_t r = (i / H) % (2 * I);
        const int64_t c = i % H;
        const bool up = (r & 1) != 0;
        return RowValue(r / 2, c, (up ? 400 : 300) + s);
    });
}

// The dense w13_dn slab [2*denseI, H] ROW-INTERLEAVED: source row 2r is gate
// row r (tag 500+layer), source row 2r+1 is up row r (tag 600+layer).
inline std::vector<uint8_t> DenseW13(int layer)
{
    const int64_t I = kDenseInter, H = kHidden;
    return synthtest::MakeBf16Values(2 * I * H, [H, layer](int64_t i) {
        const int64_t r = i / H;
        const int64_t c = i % H;
        const bool up = (r & 1) != 0;
        return RowValue(r / 2, c, (up ? 600 : 500) + layer);
    });
}

inline std::vector<synthtest::FakeTensor> MakeLayerTensors(int layer)
{
    std::vector<synthtest::FakeTensor> t;
    const int kv = KvDim(layer);
    // Dense attention projections (BF16 -> NVFP4 in the artifact).
    Push(t, Layer(layer, "attn.wq_du.weight"), "BF16", {kQDim, kHidden},
         RowMatrix(kQDim, kHidden, 10 + layer));
    Push(t, Layer(layer, "attn.wk_dv.weight"), "BF16", {kv, kHidden},
         RowMatrix(kv, kHidden, 20 + layer));
    Push(t, Layer(layer, "attn.wv_dv.weight"), "BF16", {kv, kHidden},
         RowMatrix(kv, kHidden, 30 + layer));
    Push(t, Layer(layer, "attn.wo_ud.weight"), "BF16", {kHidden, kQDim},
         RowMatrix(kHidden, kQDim, 40 + layer));
    Push(t, Layer(layer, "attn.wr_du.weight"), "BF16", {kRelDim, kHidden},
         RowMatrix(kRelDim, kHidden, 50 + layer));
    // Raw (never quantized) side tensors.
    Push(t, Layer(layer, "attn.q_norm.weight"), "BF16", {kHeadDim},
         synthtest::MakeBf16Values(kHeadDim, [](int64_t i) {
             return 0.9f + 0.01f * static_cast<float>(i);
         }));
    Push(t, Layer(layer, "attn.k_norm.weight"), "BF16", {kHeadDim},
         synthtest::MakeBf16Values(kHeadDim, [](int64_t i) {
             return 0.8f + 0.01f * static_cast<float>(i);
         }));
    Push(t, Layer(layer, "attn.rel_logits_proj.proj"), "BF16",
         {kDRel, RelExtentOf(layer)},
         synthtest::MakeBf16Values(kDRel * RelExtentOf(layer), [](int64_t i) {
             return 0.01f * std::sin(0.3f * static_cast<float>(i));
         }));
    Push(t, Layer(layer, "attn.k_sconv.weight"), "BF16", {kv, 1, kConvK},
         synthtest::MakeBf16Values(kv * kConvK, [](int64_t i) {
             return 0.1f + 0.01f * static_cast<float>(i % 4);
         }));
    Push(t, Layer(layer, "attn.v_sconv.weight"), "BF16", {kv, 1, kConvK},
         synthtest::MakeBf16Values(kv * kConvK, [](int64_t i) {
             return 0.1f + 0.01f * static_cast<float>(i % 4);
         }));
    Push(t, Layer(layer, "attn_sconv.weight"), "BF16", {kHidden, 1, kConvK},
         synthtest::MakeBf16Values(kHidden * kConvK, [](int64_t i) {
             return 0.1f + 0.01f * static_cast<float>(i % 4);
         }));
    Push(t, Layer(layer, "mlp_sconv.weight"), "BF16", {kHidden, 1, kConvK},
         synthtest::MakeBf16Values(kHidden * kConvK, [](int64_t i) {
             return 0.1f + 0.01f * static_cast<float>(i % 4);
         }));
    Push(t, Layer(layer, "attn_norm.weight"), "BF16", {kHidden},
         synthtest::MakeBf16Values(kHidden, [](int64_t i) {
             return 0.9f + 0.005f * static_cast<float>(i % 8);
         }));
    Push(t, Layer(layer, "mlp_norm.weight"), "BF16", {kHidden},
         synthtest::MakeBf16Values(kHidden, [](int64_t i) {
             return 0.9f + 0.005f * static_cast<float>(i % 8);
         }));
    if (IsDense(layer)) {
        Push(t, Layer(layer, "mlp.w13_dn.weight"), "BF16",
             {2 * kDenseInter, kHidden}, DenseW13(layer));
        Push(t, Layer(layer, "mlp.w2_md.weight"), "BF16",
             {kHidden, kDenseInter}, RowMatrix(kHidden, kDenseInter, 60 + layer));
        Push(t, Layer(layer, "mlp.global_scale"), "BF16", {1},
             synthtest::MakeBf16Values(1, [](int64_t) { return 1.0f; }));
    } else {
        Push(t, Layer(layer, "mlp.experts.w13_weight"), "BF16",
             {kExperts, 2 * kMoeInter, kHidden}, RoutedW13());
        Push(t, Layer(layer, "mlp.experts.w2_weight"), "BF16",
             {kExperts, kHidden, kMoeInter},
             synthtest::MakeBf16Values(
                 int64_t(kExperts) * kHidden * kMoeInter,
                 [](int64_t i) {
                     const int64_t e = i / (int64_t(kHidden) * kMoeInter);
                     const int64_t r = (i / kMoeInter) % kHidden;
                     const int64_t c = i % kMoeInter;
                     return RowValue(r, c, 700 + e);
                 }));
        Push(t, Layer(layer, "mlp.shared_experts.shared_w13_weight"), "BF16",
             {kShared, 2 * kMoeInter, kHidden}, SharedW13());
        Push(t, Layer(layer, "mlp.shared_experts.shared_w2_weight"), "BF16",
             {kShared, kHidden, kMoeInter},
             synthtest::MakeBf16Values(
                 int64_t(kShared) * kHidden * kMoeInter,
                 [](int64_t i) {
                     const int64_t s = i / (int64_t(kHidden) * kMoeInter);
                     const int64_t r = (i / kMoeInter) % kHidden;
                     const int64_t c = i % kMoeInter;
                     return RowValue(r, c, 800 + s);
                 }));
        // Router: 258-row analogue (E + shared), F32 selection bias over the
        // routed rows only, F32 global scale.
        Push(t, Layer(layer, "mlp.gate.weight"), "BF16",
             {kExperts + kShared, kHidden},
             RowMatrix(kExperts + kShared, kHidden, 70 + layer));
        Push(t, Layer(layer, "mlp.gate.bias"), "F32", {kExperts},
             synthtest::MakeF32Values(kExperts, [](int64_t i) {
                 return 0.01f * static_cast<float>(i);
             }));
        Push(t, Layer(layer, "mlp.gate.global_scale"), "F32", {1},
             synthtest::MakeF32Values(1, [](int64_t) { return 1.0f; }));
    }
    return t;
}

inline void WriteConfig(const std::filesystem::path &dir)
{
    std::string local;
    for (int l = 0; l < kLayers; ++l) {
        if (IsFull(l)) continue;
        if (!local.empty()) local += ",";
        local += std::to_string(l);
    }
    const std::string text = absl::StrCat(
        "{\n"
        "  \"architectures\": [\"InklingForConditionalGeneration\"],\n"
        "  \"model_type\": \"inkling_mm_model\",\n"
        "  \"eos_token_id\": 200006,\n"
        "  \"text_config\": {\n"
        "    \"hidden_size\": ", kHidden,
        ",\n    \"num_hidden_layers\": ", kLayers,
        ",\n    \"vocab_size\": ", kVocab,
        ",\n    \"unpadded_vocab_size\": ", kUnpadded,
        ",\n    \"num_attention_heads\": ", kHeads,
        ",\n    \"num_key_value_heads\": ", kKvHeadsFull,
        ",\n    \"head_dim\": ", kHeadDim,
        ",\n    \"d_rel\": ", kDRel,
        ",\n    \"rel_extent\": ", kRelExtent,
        ",\n    \"sliding_window_size\": ", kWindow,
        ",\n    \"sconv_kernel_size\": ", kConvK,
        ",\n    \"use_sconv\": true,\n"
        "    \"dense_mlp_idx\": ", kDenseLayers,
        ",\n    \"dense_intermediate_size\": ", kDenseInter,
        ",\n    \"intermediate_size\": ", kMoeInter,
        ",\n    \"n_routed_experts\": ", kExperts,
        ",\n    \"num_experts_per_tok\": ", kTopK,
        ",\n    \"n_shared_experts\": ", kShared,
        ",\n    \"shared_expert_sink\": true,\n"
        "    \"route_scale\": 8.0,\n"
        "    \"use_gate_bias\": true,\n"
        "    \"gate_activation\": \"sigmoid\",\n"
        "    \"norm_after_topk\": true,\n"
        "    \"use_global_scale\": true,\n"
        "    \"rms_norm_eps\": 1e-06,\n"
        "    \"use_embed_norm\": true,\n"
        "    \"log_scaling_n_floor\": 128000,\n"
        "    \"log_scaling_alpha\": 0.1,\n"
        "    \"logits_mup_width_multiplier\": 24.0,\n"
        "    \"final_logit_softcapping\": null,\n"
        "    \"swa_head_dim\": ", kHeadDim,
        ",\n    \"swa_num_attention_heads\": ", kHeads,
        ",\n    \"swa_num_key_value_heads\": ", kKvHeadsSliding,
        ",\n    \"local_layer_ids\": [", local, "]\n",
        "  }\n}\n");
    std::ofstream f(dir / "config.json");
    f << text;
}

inline void WriteCheckpoint(const std::filesystem::path &dir)
{
    WriteConfig(dir);
    std::vector<synthtest::FakeTensor> t;
    // Globals.
    Push(t, Llm("embed.weight"), "BF16", {kVocab, kHidden},
         RowMatrix(kVocab, kHidden, 1));
    Push(t, Llm("unembed.weight"), "BF16", {kVocab, kHidden},
         RowMatrix(kVocab, kHidden, 2));
    Push(t, Llm("embed_norm.weight"), "BF16", {kHidden},
         synthtest::MakeBf16Values(kHidden, [](int64_t i) {
             return 0.9f + 0.005f * static_cast<float>(i % 8);
         }));
    Push(t, Llm("norm.weight"), "BF16", {kHidden},
         synthtest::MakeBf16Values(kHidden, [](int64_t i) {
             return 0.9f + 0.005f * static_cast<float>(i % 8);
         }));
    for (int l = 0; l < kLayers; ++l) {
        auto lt = MakeLayerTensors(l);
        t.insert(t.end(), lt.begin(), lt.end());
    }
    // Ballast: vision (2), audio (1), MTP (2) — stripped and counted.
    Push(t, "model.visual.final_norm.weight", "BF16", {kHidden},
         synthtest::MakeBf16(kHidden));
    Push(t, "model.visual.layers.linear_0.weight", "BF16", {8, 8},
         synthtest::MakeBf16(64));
    Push(t, "model.audio.encoder.weight", "BF16", {16, kHidden},
         synthtest::MakeBf16(16 * kHidden));
    Push(t, "model.mtp.layers.0.input_proj.weight", "BF16", {kHidden, kHidden},
         synthtest::MakeBf16(kHidden * kHidden));
    Push(t, "model.mtp.layers.0.hidden_norm.weight", "BF16", {kHidden},
         synthtest::MakeBf16(kHidden));
    synthtest::WriteSafetensors(dir, Shard0(), t);
}

}  // namespace inklingbasetest