#pragma once

// Inkling-2-Small synthetic-checkpoint fixture. Family-neutral primitives come
// from test_synth_checkpoint.h; this header only knows Inkling's tensor names
// and the legal small geometry (6 layers so that the AttnRes boundary table and
// the sliding/full split are both exercised, with a 4/2 NVFP4/MXFP8 expert
// split so both source codecs are covered).
//
// The real checkpoint's numbers live in src/inkling_moe_constants.h; the
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

namespace inklingtest {

// ── synthetic geometry ─────────────────────────────────────────────────────
inline constexpr int kHidden = 64;
inline constexpr int kLayers = 6;
inline constexpr int kHeads = 4;
inline constexpr int kKvHeads = 2;
inline constexpr int kHeadDim = 16;
inline constexpr int kDRel = 4;
inline constexpr int kRelExtent = 8;
inline constexpr int kWindow = 8;
inline constexpr int kConvK = 4;
inline constexpr int kAttnResBlock = 2;   // boundaries 0, 2, 4
inline constexpr int kExperts = 4;
inline constexpr int kMoeInter = 32;
inline constexpr int kVocab = 256;
inline constexpr int kUnpadded = 200;
// Layers [0, kNovfp4Layers) carry NVFP4 experts, the rest MXFP8.
inline constexpr int kNovfp4Layers = 4;
// Full-attention layers: everything outside `local_layer_ids`.
inline constexpr int kFullLayer = 5;

inline constexpr int kQDim = kHeads * kHeadDim;      // 64
inline constexpr int kKvDim = kKvHeads * kHeadDim;   // 32
inline constexpr int kRelDim = kHeads * kDRel;       // 16
inline constexpr int kGateUp = 2 * kMoeInter;        // 64

inline bool IsFull(int layer) { return (layer + 1) % 6 == 0; }

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

// Deterministic fake NVFP4 expert plane: packed nibbles + per-16 scales + the
// per-expert amax the converter turns into a global scale.
struct ExpertPlane {
    std::vector<synthtest::FakeTensor> tensors;
    std::vector<float> amax13, amax2;
};

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

// MXFP8 plane: E4M3 weights + ue8m0 scales (0x7F == 1.0). The weights stay in
// the 2^-10..2^-7 band on purpose: the converted NVFP4 expert planes of the
// other layers carry a per-expert global scale of amax/2688 ~ 3e-5, and a
// fixture whose MXFP8 layers were ~1e4 times larger than the NVFP4 ones pushed
// the residual stream past the FP16 range within a few tokens (measured: the
// layer-5 residual reached 2.9e4 and the next add overflowed). Expert
// magnitudes have to be comparable across the two codecs for the mixed-codec
// tree to decode at all.
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
        // amax per expert and the outer-scale denominator (2688 as shipped).
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
    // NVFP4 dense projections.
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
    return t;
}

// The shared byte-pattern fixture (`MakeBf16`) writes `i % 65536` reinterpreted
// as BF16. Two properties of that make it unusable for a numeric test: the
// values span 1e-38..3.4e38 (after `rmsnorm * weight` the hidden state is ~1e38
// and every unembed dot saturates), and EACH ROW carries its own magnitude, so
// a row whose values are ~1e-40 quantizes to all-zero blocks and its vocabulary
// entry contributes nothing. Both were observed as a decode that emitted token 0
// with an all-zero logits row.
//
// A test that wants to watch the engine carry its input into the output needs a
// smooth, bounded, per-element-varying field. `RenderSaneValues` replaces every
// BF16 / F32 element with one (the packed expert nibbles and the per-expert
// amax / outer-scale-denominator are left alone: the first is a pass-through
// codec, the others are scales whose value the converter depends on).
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
        // The OUTPUT projections set the per-layer residual gain. An untrained
        // net with unit-ish gain diverges: the residual stream is FP16, and
        // `x += w_o * attn` with |w_o| ~ 0.05 compounds past 65504 within a few
        // tokens (observed: non-finite from the third position). Scaling the
        // attention output projection down by 40x keeps the fixture stable for
        // the handful of tokens a unit test decodes, which is all this fixture
        // is for -- it pins the plumbing, not the numerics.
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
                     "  \"architectures\": [\"Inkling2ForConditionalGeneration\"],\n"
                     "  \"model_type\": \"inkling2\",\n"
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
                     ",\n    \"log_scaling_n_floor\": 128000",
                     ",\n    \"log_scaling_alpha\": 0.1",
                     ",\n    \"mlp_gate_softcap\": 4.0",
                     ",\n    \"mlp_up_softcap\": 25.0",
                     ",\n    \"final_logit_softcapping\": 30.0",
                     ",\n    \"local_layer_ids\": [", local, "]\n",
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

// Writes the whole synthetic checkpoint into `dir`.
// `sane` renders a bounded field (see RenderSaneValues); the artifact-layout tests
// keep the raw byte pattern so their exact NVFP4 pass-through bytes stay pinned.
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

}  // namespace inklingtest
