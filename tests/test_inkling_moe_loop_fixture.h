#pragma once

// Inkling-2-Mini synthetic-checkpoint fixture. Family-neutral primitives come
// from test_synth_checkpoint.h; this header only knows the inkling2_loop tensor
// names and a legal small geometry: 8 layers so the AttnRes boundary table
// (block 2) and the sliding/full split are exercised, loop core [2,6) with 2
// passes (bounds are block multiples, as the reference config validator
// requires), and a 6/2 NVFP4/MXFP8 expert split so both source codecs are
// covered.
//
// The real checkpoint's numbers live in src/inkling_moe_loop_constants.h; the
// converter relaxes its frozen cross-check under
// LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1 and reads the *parsed* geometry.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <absl/strings/str_cat.h>

#include "test_synth_checkpoint.h"

namespace inklinglooptest {

// ── synthetic geometry ─────────────────────────────────────────────────────
inline constexpr int kHidden = 64;
inline constexpr int kLayers = 8;
inline constexpr int kHeads = 4;
inline constexpr int kKvHeads = 2;
inline constexpr int kHeadDim = 16;
inline constexpr int kDRel = 4;
inline constexpr int kRelExtent = 8;
inline constexpr int kWindow = 8;
inline constexpr int kConvK = 4;
inline constexpr int kAttnResBlock = 2;   // boundaries 0, 2, 4, 6
inline constexpr int kExperts = 4;
inline constexpr int kMoeInter = 32;
inline constexpr int kVocab = 256;
inline constexpr int kUnpadded = 200;
// Layers [0, kNovfp4Layers) carry NVFP4 experts, the rest MXFP8.
inline constexpr int kNovfp4Layers = 6;
// Full-attention layers: everything outside `local_layer_ids` ((l+1)%6==0).
inline constexpr int kFullLayer = 5;
// The loop core: [2,6) x 2 passes -> 8 + 4 = 12 cache slots.
inline constexpr int kLoopStart = 2;
inline constexpr int kLoopEnd = 6;
inline constexpr int kLoopPasses = 2;
inline constexpr int kLoopGateRank = 8;
inline constexpr int kLoopLayers = kLoopEnd - kLoopStart;  // 4

inline constexpr int kQDim = kHeads * kHeadDim;      // 64
inline constexpr int kKvDim = kKvHeads * kHeadDim;   // 32
inline constexpr int kRelDim = kHeads * kDRel;       // 16
inline constexpr int kGateUp = 2 * kMoeInter;        // 64

inline bool IsFull(int layer) { return (layer + 1) % 6 == 0; }
inline bool IsLoop(int layer) { return layer >= kLoopStart && layer < kLoopEnd; }

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
    out.push_back(
        synthtest::FakeTensor{std::move(name), std::move(dtype),
                              std::move(shape), std::move(data)});
}

inline std::vector<uint8_t> Nibbles(int64_t bytes, uint8_t seed)
{
    std::vector<uint8_t> out(static_cast<size_t>(bytes));
    for (size_t i = 0; i < out.size(); ++i)
        out[i] = static_cast<uint8_t>((i * 37 + seed * 11) & 0xFF);
    return out;
}

inline std::vector<uint8_t> ScalesE4m3(int64_t bytes, uint8_t seed)
{
    // 0x38 = 2^0 .. 0x40 = 2^1: legal positive E4M3FN block scales.
    std::vector<uint8_t> out(static_cast<size_t>(bytes));
    for (size_t i = 0; i < out.size(); ++i)
        out[i] = static_cast<uint8_t>(0x38 + ((i + seed) % 8));
    return out;
}

// MXFP8 plane: E4M3 weights in a small band + ue8m0 scales (0x7F == 1.0), so
// expert magnitudes stay comparable across the two codecs (the mixed-codec
// tree must decode without the residual stream overflowing FP16).
inline std::vector<uint8_t> E4m3Bytes(int64_t bytes, uint8_t seed)
{
    std::vector<uint8_t> out(static_cast<size_t>(bytes));
    for (size_t i = 0; i < out.size(); ++i)
        out[i] = static_cast<uint8_t>(0x18 + ((i * 7 + seed) % 8));
    return out;
}

inline std::vector<uint8_t> Ue8m0Scales(int64_t bytes)
{
    return std::vector<uint8_t>(static_cast<size_t>(bytes), 0x7F);
}

struct ExpertPlane {
    std::vector<synthtest::FakeTensor> tensors;
    std::vector<float> amax13, amax2;
};

inline ExpertPlane MakeExpertPlane(int layer)
{
    ExpertPlane p;
    const int64_t E = kExperts;
    const bool nvfp4 = layer < kNovfp4Layers;
    if (nvfp4) {
        Push(p.tensors, Layer(layer, "mlp.experts.w13_weight"), "U8",
             {E, kGateUp, kHidden / 2}, Nibbles(E * kGateUp * (kHidden / 2), 3));
        Push(p.tensors, Layer(layer, "mlp.experts.w13_weight_scale"), "F8_E4M3",
             {E, kGateUp, kHidden / 16},
             ScalesE4m3(E * kGateUp * (kHidden / 16), 5));
        Push(p.tensors, Layer(layer, "mlp.experts.w2_weight"), "U8",
             {E, kHidden, kMoeInter / 2},
             Nibbles(E * kHidden * (kMoeInter / 2), 7));
        Push(p.tensors, Layer(layer, "mlp.experts.w2_weight_scale"), "F8_E4M3",
             {E, kHidden, kMoeInter / 16},
             ScalesE4m3(E * kHidden * (kMoeInter / 16), 9));
        p.amax13.assign(static_cast<size_t>(E), 0.08f);
        p.amax2.assign(static_cast<size_t>(E), 0.04f);
        Push(p.tensors, Layer(layer, "mlp.experts.w13_weight_amax"), "F32",
             {E}, synthtest::MakeF32Values(E, [](int64_t i) {
                 return 0.08f + 0.001f * static_cast<float>(i);
             }));
        Push(p.tensors, Layer(layer, "mlp.experts.w2_weight_amax"), "F32", {E},
             synthtest::MakeF32Values(
                 E, [](int64_t i) { return 0.04f + 0.001f * static_cast<float>(i); }));
        Push(p.tensors,
             Layer(layer, "mlp.experts.w13_weight_outer_scale_denominator"),
             "F32", {1}, synthtest::MakeF32Values(1, [](int64_t) { return 2688.0f; }));
        Push(p.tensors,
             Layer(layer, "mlp.experts.w2_weight_outer_scale_denominator"),
             "F32", {1}, synthtest::MakeF32Values(1, [](int64_t) { return 2688.0f; }));
        for (int e = 0; e < kExperts; ++e) {
            p.amax13[static_cast<size_t>(e)] =
                0.08f + 0.001f * static_cast<float>(e);
            p.amax2[static_cast<size_t>(e)] =
                0.04f + 0.001f * static_cast<float>(e);
        }
    } else {
        Push(p.tensors, Layer(layer, "mlp.experts.w13_weight"), "F8_E4M3",
             {E, kGateUp, kHidden}, E4m3Bytes(E * kGateUp * kHidden, 11));
        Push(p.tensors, Layer(layer, "mlp.experts.w13_weight_scale"), "U8",
             {E, kGateUp, kHidden / 32}, Ue8m0Scales(E * kGateUp * (kHidden / 32)));
        Push(p.tensors, Layer(layer, "mlp.experts.w2_weight"), "F8_E4M3",
             {E, kHidden, kMoeInter}, E4m3Bytes(E * kHidden * kMoeInter, 13));
        Push(p.tensors, Layer(layer, "mlp.experts.w2_weight_scale"), "U8",
             {E, kHidden, kMoeInter / 32},
             Ue8m0Scales(E * kHidden * (kMoeInter / 32)));
    }
    return p;
}

inline std::vector<synthtest::FakeTensor> MakeLayerTensors(int layer)
{
    std::vector<synthtest::FakeTensor> t;
    // Dense projections: BF16 in the release, kept BF16 in the artifact.
    Push(t, Layer(layer, "attn.wq_du.weight"), "BF16", {kQDim, kHidden},
         synthtest::MakeBf16(kQDim * kHidden));
    Push(t, Layer(layer, "attn.wk_dv.weight"), "BF16", {kKvDim, kHidden},
         synthtest::MakeBf16(kKvDim * kHidden));
    Push(t, Layer(layer, "attn.wv_dv.weight"), "BF16", {kKvDim, kHidden},
         synthtest::MakeBf16(kKvDim * kHidden));
    Push(t, Layer(layer, "attn.wo_ud.weight"), "BF16", {kHidden, kQDim},
         synthtest::MakeBf16(kHidden * kQDim));
    Push(t, Layer(layer, "attn.wr_du.weight"), "BF16", {kRelDim, kHidden},
         synthtest::MakeBf16(kRelDim * kHidden));
    Push(t, Layer(layer, "attn.gate_attn.weight"), "BF16", {kQDim, kHidden},
         synthtest::MakeBf16(kQDim * kHidden));
    // Raw (never quantized) attention side tensors.
    Push(t, Layer(layer, "attn.attn_sink_proj.weight"), "BF16", {kHeads, kHidden},
         synthtest::MakeBf16(kHeads * kHidden));
    Push(t, Layer(layer, "attn.rel_logits_proj.proj"), "BF16",
         {kDRel, kRelExtent}, synthtest::MakeBf16(kDRel * kRelExtent));
    Push(t, Layer(layer, "attn.q_l2_clip.scale"), "BF16", {kHeads, kHeadDim},
         synthtest::MakeBf16(kHeads * kHeadDim));
    Push(t, Layer(layer, "attn.rel_l2_clip.scale"), "BF16", {kHeads, kDRel},
         synthtest::MakeBf16(kHeads * kDRel));
    Push(t, Layer(layer, "attn_sconv.weight"), "BF16", {kHidden, 1, kConvK},
         synthtest::MakeBf16(kHidden * kConvK));
    Push(t, Layer(layer, "mlp_sconv.weight"), "BF16", {kHidden, 1, kConvK},
         synthtest::MakeBf16(kHidden * kConvK));
    Push(t, Layer(layer, "mlp_norm.weight"), "BF16", {kHidden},
         synthtest::MakeBf16(kHidden));
    Push(t, Layer(layer, "mlp.gate.weight"), "BF16", {kExperts, kHidden},
         synthtest::MakeBf16(kExperts * kHidden));
    Push(t, Layer(layer, "mlp.gate.bias"), "F32", {kExperts},
         synthtest::MakeF32Values(kExperts, [](int64_t i) {
             return 0.01f * static_cast<float>(i);
         }));
    if (IsFull(layer)) {
        Push(t, Layer(layer, "k_input_norm.weight"), "BF16", {kHidden},
             synthtest::MakeBf16(kHidden));
        Push(t, Layer(layer, "q_input_norm.weight"), "BF16", {kHidden},
             synthtest::MakeBf16(kHidden));
        Push(t, Layer(layer, "v_input_norm.weight"), "BF16", {kHidden},
             synthtest::MakeBf16(kHidden));
        Push(t, Layer(layer, "attn.k_sconv.static_w"), "BF16", {kConvK, kKvDim},
             synthtest::MakeBf16(kConvK * kKvDim));
        Push(t, Layer(layer, "attn.v_sconv.static_w"), "BF16", {kConvK, kKvDim},
             synthtest::MakeBf16(kConvK * kKvDim));
        Push(t, Layer(layer, "attn.k_sconv.weight_proj"), "BF16",
             {kKvHeads * kConvK, kHidden},
             synthtest::MakeBf16(kKvHeads * kConvK * kHidden));
        Push(t, Layer(layer, "attn.v_sconv.weight_proj"), "BF16",
             {kKvHeads * kConvK, kHidden},
             synthtest::MakeBf16(kKvHeads * kConvK * kHidden));
    } else {
        Push(t, Layer(layer, "attn_norm.weight"), "BF16", {kHidden},
             synthtest::MakeBf16(kHidden));
        Push(t, Layer(layer, "attn.k_sconv.weight"), "BF16",
             {kKvDim, 1, kConvK}, synthtest::MakeBf16(kKvDim * kConvK));
        Push(t, Layer(layer, "attn.v_sconv.weight"), "BF16",
             {kKvDim, 1, kConvK}, synthtest::MakeBf16(kKvDim * kConvK));
    }
    // Loop controllers: four MessageGates per loop layer (q/k/v/mlp inputs).
    // down is BF16 [rank, 2*hidden]; forget/gain are the gate projections
    // [hidden, rank] (FP32 in the reference's strict list).
    if (IsLoop(layer)) {
        for (const char *m : {"attn", "k", "v", "mlp"}) {
            Push(t, Layer(layer, absl::StrCat("controllers.", m, ".down_weight")),
                 "BF16", {kLoopGateRank, 2 * kHidden},
                 synthtest::MakeBf16(kLoopGateRank * 2 * kHidden));
            Push(t, Layer(layer, absl::StrCat("controllers.", m, ".forget_weight")),
                 "BF16", {kHidden, kLoopGateRank},
                 synthtest::MakeBf16(kHidden * kLoopGateRank));
            Push(t, Layer(layer, absl::StrCat("controllers.", m, ".gain_weight")),
                 "BF16", {kHidden, kLoopGateRank},
                 synthtest::MakeBf16(kHidden * kLoopGateRank));
        }
    }
    return t;
}

// Bounded, per-element-varying field for numeric tests (see the Small fixture
// for why the raw MakeBf16 pattern cannot drive a decode). The packed expert
// nibbles and the amax/denominator scales are left alone.
inline void RenderSaneValues(std::vector<synthtest::FakeTensor> &ts)
{
    for (auto &t : ts) {
        if (t.name.find("amax") != std::string::npos ||
            t.name.find("denominator") != std::string::npos)
            continue;
        const bool bf16 = (t.dtype == "BF16");
        const bool f32 = (t.dtype == "F32");
        if (!bf16 && !f32) continue;
        const size_t elem = bf16 ? 2 : 4;
        const size_t n = t.data.size() / elem;
        const float gain = (t.name.find("wo_ud") != std::string::npos) ? 0.025f : 1.0f;
        for (size_t i = 0; i < n; ++i) {
            const float v = gain *
                            (0.05f * std::sin(0.7f * static_cast<float>(i)) +
                             0.02f * std::cos(0.11f * static_cast<float>(i)));
            if (bf16) {
                const uint16_t b = synthtest::FloatToBf16(v);
                t.data[i * 2] = static_cast<uint8_t>(b & 0xff);
                t.data[i * 2 + 1] = static_cast<uint8_t>(b >> 8);
            } else {
                std::memcpy(t.data.data() + i * 4, &v, 4);
            }
        }
    }
}

inline void WriteConfig(const std::filesystem::path &dir)
{
    std::string local;
    for (int l = 0; l < kLayers; ++l) {
        if (IsFull(l)) continue;
        if (!local.empty()) local += ",";
        local += std::to_string(l);
    }
    const std::string text =
        absl::StrCat("{\n"
                     "  \"architectures\": [\"Inkling2LoopForConditionalGeneration\"],\n"
                     "  \"model_type\": \"inkling2_loop\",\n"
                     "  \"eos_token_id\": 200006,\n"
                     "  \"text_config\": {\n"
                     "    \"hidden_size\": ",
                     kHidden,
                     ",\n    \"num_hidden_layers\": ", kLayers,
                     ",\n    \"vocab_size\": ", kVocab,
                     ",\n    \"unpadded_vocab_size\": ", kUnpadded,
                     ",\n    \"num_attention_heads\": ", kHeads,
                     ",\n    \"num_key_value_heads\": ", kKvHeads,
                     ",\n    \"head_dim\": ", kHeadDim,
                     ",\n    \"d_rel\": ", kDRel,
                     ",\n    \"rel_extent\": ", kRelExtent,
                     ",\n    \"sliding_window_size\": ", kWindow,
                     ",\n    \"sconv_kernel_size\": ", kConvK,
                     ",\n    \"n_layers_per_attn_res_block\": ", kAttnResBlock,
                     ",\n    \"n_routed_experts\": ", kExperts,
                     ",\n    \"num_experts_per_tok\": 2",
                     ",\n    \"intermediate_size\": ", kMoeInter,
                     ",\n    \"n_shared_experts\": 0",
                     ",\n    \"route_scale\": 8.0",
                     ",\n    \"rms_norm_eps\": 1e-06",
                     ",\n    \"qk_l2_clip_threshold\": 22.627416997969522",
                     ",\n    \"rel_l2_clip_threshold\": 8.0",
                     ",\n    \"log_scaling_n_floor\": 8192",
                     ",\n    \"log_scaling_alpha\": 0.1",
                     ",\n    \"mlp_gate_softcap\": 4.0",
                     ",\n    \"mlp_up_softcap\": 25.0",
                     ",\n    \"final_logit_softcapping\": 30.0",
                     ",\n    \"local_layer_ids\": [", local, "],\n",
                     "    \"loop_config\": {\"core_start\": ", kLoopStart,
                     ", \"core_end\": ", kLoopEnd,
                     ", \"num_passes\": ", kLoopPasses,
                     ", \"gate_rank\": ", kLoopGateRank,
                     ", \"is_varloop\": false}\n",
                     "  }\n}\n");
    std::ofstream f(dir / "config.json");
    f << text;
}

// hf_quant_config.json with the same two expert codec groups as the release.
inline void WriteQuantConfig(const std::filesystem::path &dir)
{
    std::string nvfp4, mxfp8;
    for (int l = 0; l < kLayers; ++l) {
        const bool nv = l < kNovfp4Layers;
        for (const char *kind : {"w13", "w2"}) {
            std::string &dst = nv ? nvfp4 : mxfp8;
            if (!dst.empty()) dst += ",";
            dst += absl::StrCat("\"model.llm.layers.", l, ".mlp.experts.",
                                kind, "_weight\"");
        }
    }
    std::string text =
        absl::StrCat("{\"model\": {\"parameter_groups\": [",
                     "{\"compute_dtype\": \"nvfp4\", \"tensors\": [", nvfp4,
                     "]}, {\"compute_dtype\": \"mxfp8\", \"tensors\": [", mxfp8,
                     "]}]}}\n");
    std::ofstream f(dir / "hf_quant_config.json");
    f << text;
}

// Writes the whole synthetic checkpoint into `dir`. `sane` renders a bounded
// field (see RenderSaneValues); the artifact-layout tests keep the raw byte
// pattern so their exact pass-through bytes stay pinned.
inline void WriteCheckpoint(const std::filesystem::path &dir, bool sane = false)
{
    WriteConfig(dir);
    WriteQuantConfig(dir);

    std::vector<synthtest::FakeTensor> t;
    // globals
    Push(t, Llm("embed.weight"), "BF16", {kVocab, kHidden},
         synthtest::MakeBf16(kVocab * kHidden));
    Push(t, Llm("unembed.weight"), "BF16", {kVocab, kHidden},
         synthtest::MakeBf16(kVocab * kHidden));
    Push(t, Llm("embed_norm.weight"), "BF16", {kHidden},
         synthtest::MakeBf16(kHidden));
    Push(t, Llm("norm.weight"), "BF16", {kHidden}, synthtest::MakeBf16(kHidden));
    Push(t, Llm("attn_res.output_query"), "F32", {kHidden},
         synthtest::MakeF32Values(kHidden, [](int64_t i) {
             return 0.001f * static_cast<float>(i + 1);
         }));
    // mlp queries: one per layer; attn queries: layers 1..L-1 (source index =
    // layer-1, and full layers carry three rows).
    for (int l = 0; l < kLayers; ++l) {
        Push(t, Llm(absl::StrCat("attn_res.mlp_queries.", l)), "F32", {kHidden},
             synthtest::MakeF32Values(kHidden, [l](int64_t i) {
                 return 0.0005f * static_cast<float>(i + 1 + l);
             }));
        if (l == 0) continue;
        const std::vector<int64_t> shape =
            IsFull(l) ? std::vector<int64_t>{3, kHidden}
                      : std::vector<int64_t>{kHidden};
        const int64_t n = IsFull(l) ? 3 * kHidden : kHidden;
        Push(t, Llm(absl::StrCat("attn_res.attn_queries.", l - 1)), "F32", shape,
             synthtest::MakeF32Values(n, [l](int64_t i) {
                 return 0.0007f * static_cast<float>(i + 1 + l);
             }));
    }
    // Cross-pass message queries: two [2*loop_layers, hidden] FP32 tensors.
    // `queries` rows are [q, mlp] per loop layer; `kv_queries` rows are [k, v].
    // Distinct per-row values so the merge order is observable.
    Push(t, Llm("middle_cycle_cross_pass_queries"), "F32",
         {2 * kLoopLayers, kHidden},
         synthtest::MakeF32Values(2 * kLoopLayers * kHidden, [](int64_t i) {
             return 0.001f * static_cast<float>(i + 1);
         }));
    Push(t, Llm("middle_cycle_cross_pass_kv_queries"), "F32",
         {2 * kLoopLayers, kHidden},
         synthtest::MakeF32Values(2 * kLoopLayers * kHidden, [](int64_t i) {
             return -0.001f * static_cast<float>(i + 1);
         }));
    // ballast that must be stripped
    Push(t, "model.visual.backbone.ln_pre.weight", "BF16", {kHidden},
         synthtest::MakeBf16(kHidden));
    Push(t, "model.visual.proj_out.weight", "BF16", {kHidden, kHidden},
         synthtest::MakeBf16(kHidden * kHidden));

    for (int l = 0; l < kLayers; ++l) {
        auto layer_t = MakeLayerTensors(l);
        t.insert(t.end(), layer_t.begin(), layer_t.end());
        auto plane = MakeExpertPlane(l);
        t.insert(t.end(), plane.tensors.begin(), plane.tensors.end());
    }
    if (sane) RenderSaneValues(t);
    synthtest::WriteSafetensors(dir, Shard0(), t);
}

}  // namespace inklinglooptest