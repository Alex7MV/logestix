#pragma once

// NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 synthetic checkpoint fixture.
// Family-owned (Model Ingestion Protocol): it re-exports the family-neutral
// primitives of test_synth_checkpoint.h under `nemotronv4test::` and builds a
// tiny PRE-QUANTIZED NemotronH checkpoint that reproduces the real vault's
// source layout byte for byte:
//
//   * mamba in_proj/out_proj  : F8_E4M3 + SCALAR F32 weight_scale + F32
//                               input_scale          (ModelOpt per-tensor FP8)
//   * routed + shared experts: U8 packed E2M1 + F8_E4M3 weight_scale [R, C/16]
//                               + F32 weight_scale_2 (W4A16_NVFP4, group 16)
//   * attention q/k/v/o_proj : BF16, plus F32[1] k_scale / v_scale (the FP8
//                               KV-cache scales - NOT weight scales)
//   * lm_head                : U8 NVFP4 + block scale + weight_scale_2
//   * embeddings / norms / conv / A_log / D / dt_bias : BF16
//   * gate.weight            : F32 (the vault's fp8 run left the router F32)
//   * mtp.*                  : BF16, must be cut and counted
//
// Never touches the real vault: config.json + safetensors are written into a
// temporary directory and the converter runs on those.

#include "test_synth_checkpoint.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace nemotronv4test {

using synthtest::FakeTensor;
using synthtest::FloatToBf16;
using synthtest::MakeBf16;
using synthtest::MakeF32Values;
using synthtest::WriteSafetensors;

inline std::filesystem::path MakeTempDir(const std::string &tag)
{
    static std::atomic<unsigned> counter{0};
    const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
    auto p = std::filesystem::temp_directory_path() /
             (tag + "_" + std::to_string(ticks) + "_" +
              std::to_string(counter.fetch_add(1)));
    std::filesystem::remove_all(p);
    std::filesystem::create_directories(p);
    return p;
}

// Tiny NemotronH geometry (frozen gates relaxed via
// LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1). Layer 0 = mamba, 1 = moe, 2 =
// attention - every block type of the real 52-layer mix, so the converter's
// per-block-type source dispatch is all exercised.
inline constexpr int kTH = 16;         // hidden
inline constexpr int kTVocab = 32;
inline constexpr int kTMambaIn = 42;   // d_inner 16 + conv_dim 24 + dt 2
inline constexpr int kTConvDim = 24;
inline constexpr int kTExperts = 4;
inline constexpr int kTI = 16;         // moe intermediate (multiple of 16)
inline constexpr int kTShared = 32;    // shared expert intermediate
inline constexpr int kTAttnOut = 16;   // heads 2 * head_dim 8

// ── Source-encoding emitters ────────────────────────────────────────────────
// Deterministic, small-magnitude values so the NVFP4 grid is actually hit
// (a fixture of huge magnitudes would quantize everything to the same code and
// hide codec defects).

// Pack one logical [R, C] row-major value matrix into the source's NVFP4
// encoding: U8 nibbles (lo nibble = even element, per PTX
// cvt.rn.satfinite.e2m1x2.f32) + full-E4M3 block scales per 16 + F32 global.
// `scale_bytes` receives the E4M3 bytes, `global` the weight_scale_2.
inline void PackSourceNvfp4(const std::vector<float> &v, int64_t R, int64_t C,
                            std::vector<uint8_t> &packed,
                            std::vector<uint8_t> &scale_bytes, float &global)
{
    constexpr float kE2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const int64_t nblk = (R * C + 15) / 16;
    std::vector<float> block_max(static_cast<size_t>(nblk), 0.0f);
    for (int64_t b = 0; b < nblk; ++b) {
        float m = 0.0f;
        for (int64_t k = 0; k < 16; ++k) {
            const int64_t i = b * 16 + k;
            if (i >= R * C) break;
            m = std::fmax(m, std::fabs(v[static_cast<size_t>(i)]));
        }
        block_max[static_cast<size_t>(b)] = m;
    }
    // The source stores a per-tensor F32 weight_scale_2 alongside per-16 block
    // scales. This fixture keeps the block scale in its natural place -
    // S_b = block_amax / 6 - and the global scale at 1.0, which is a valid
    // (if not optimal) ModelOpt encoding and lands every block scale in the
    // middle of the E4M3 normal range. What matters for the codec test is that
    // the E4M3 bytes carry MANTISSA bits, which is exactly what makes a byte
    // passthrough wrong.
    global = 1.0f;

    // Round-to-nearest-even onto the E2M1 grid.
    auto code_for = [&](float x) {
        const float a = std::fabs(x);
        int best = 0;
        float bestd = std::fabs(a - kE2M1[0]);
        for (int i = 1; i < 8; ++i) {
            const float d = std::fabs(a - kE2M1[i]);
            if (d < bestd) { bestd = d; best = i; }
        }
        if (x < 0.0f && best != 0) best |= 8;
        return static_cast<uint8_t>(best);
    };

    packed.assign(static_cast<size_t>((R * C + 1) / 2), 0);
    for (int64_t i = 0; i < R * C; ++i) {
        const int64_t b = i / 16;
        const float s = block_max[static_cast<size_t>(b)] / 6.0f / global;
        const uint8_t nib = code_for(v[static_cast<size_t>(i)] / s);
        uint8_t &byte = packed[static_cast<size_t>(i / 2)];
        if (i & 1) byte = static_cast<uint8_t>((byte & 0x0F) | (nib << 4));
        else       byte = static_cast<uint8_t>((byte & 0xF0) | nib);
    }

    // Full E4M3 encode of the block scale (round-to-nearest-even on 3 mantissa
    // bits) - deliberately NOT a power of two, which is exactly the property
    // that makes a naive byte passthrough wrong.
    scale_bytes.assign(static_cast<size_t>(nblk), 0);
    for (int64_t b = 0; b < nblk; ++b) {
        const float s = block_max[static_cast<size_t>(b)] / 6.0f / global;
        int e = 0;
        float m = 1.0f;
        if (s > 0.0f) {
            e = static_cast<int>(std::floor(std::log2(s)));
            if (std::pow(2.0f, static_cast<float>(e)) > s) --e;
            m = s / std::pow(2.0f, static_cast<float>(e));
        } else {
            e = -6; m = 0.0f;
        }
        int mant = static_cast<int>(std::lround((m - 1.0f) * 8.0f));
        if (mant < 0) mant = 0;
        if (mant > 7) mant = 7;
        int e4 = e + 7;
        if (e4 < 1) e4 = 1;
        if (e4 > 14) e4 = 14;   // 15 is the E4M3 NaN slot
        scale_bytes[static_cast<size_t>(b)] =
            static_cast<uint8_t>((e4 << 3) | mant);
    }
}

// F8_E4M3 encode (OCP E4M3FN: bias 7, no Inf, 0x7F/0xFF are NaN).
inline uint8_t FloatToE4m3(float f)
{
    if (f == 0.0f) return 0;
    const float a = std::fabs(f);
    int e = static_cast<int>(std::floor(std::log2(a)));
    const float p2 = std::pow(2.0f, static_cast<float>(e));
    if (p2 > a) --e;
    int mant = static_cast<int>(std::lround((a / p2 - 1.0f) * 8.0f));
    if (mant > 7) { mant = 0; ++e; }
    int e4 = e + 7;
    if (e4 <= 0) return static_cast<uint8_t>(std::lround(a * 512.0f));
    if (e4 >= 15) return 0x7E;
    return static_cast<uint8_t>((e4 << 3) | mant);
}

struct Nvfp4Source {
    FakeTensor weight;
    FakeTensor weight_scale;
    FakeTensor weight_scale_2;
};

// One NVFP4 source triple at logical shape [R, C].
// ModelOpt names every companion scale after the MODULE, not after the weight
// tensor (`lm_head.weight_scale`, `...down_proj.weight_scale_2`). `weight` is
// the full tensor name; the scale names drop its trailing ".weight".
inline Nvfp4Source MakeNvfp4(const std::string &weight, int64_t R, int64_t C,
                             int64_t seed)
{
    std::vector<float> v(static_cast<size_t>(R * C));
    for (int64_t i = 0; i < R * C; ++i) {
        // Deterministic, non-trivial magnitudes in a sane weight range.
        const float t = static_cast<float>((i * 37 + seed * 101) % 211) / 211.0f;
        // +/-1.0 range: block_amax/6 then lands near 0.17, comfortably inside
        // the E4M3 normal range [2^-6, 448] with a non-trivial mantissa.
        v[static_cast<size_t>(i)] = (t - 0.5f) * 2.0f;
    }
    Nvfp4Source s;
    float gs = 0.0f;
    PackSourceNvfp4(v, R, C, s.weight.data, s.weight_scale.data, gs);
    s.weight_scale_2.data.assign(4, 0);
    std::memcpy(s.weight_scale_2.data.data(), &gs, 4);
    const std::string module = weight.substr(0, weight.size() - 7);  // strip ".weight"
    s.weight.name = weight;
    s.weight.dtype = "U8";
    s.weight.shape = {R, C / 2};          // packed nibbles
    s.weight_scale.name = module + ".weight_scale";
    s.weight_scale.dtype = "F8_E4M3";
    s.weight_scale.shape = {R, C / 16};  // one E4M3 scale per 16
    s.weight_scale_2.name = module + ".weight_scale_2";
    s.weight_scale_2.dtype = "F32";
    s.weight_scale_2.shape = {};
    return s;
}

// F8_E4M3 per-tensor FP8 source triple at logical shape [R, C].
struct Fp8Source {
    FakeTensor weight;
    FakeTensor weight_scale;   // SCALAR (shape []) - ModelOpt per-tensor
    FakeTensor input_scale;    // activation scale, ignored by the converter
};

inline Fp8Source MakeFp8(const std::string &weight, int64_t R, int64_t C,
                         int64_t seed)
{
    Fp8Source s;
    float amax = 0.0f;
    std::vector<float> v(static_cast<size_t>(R * C));
    for (int64_t i = 0; i < R * C; ++i) {
        const float t = static_cast<float>((i * 29 + seed * 53) % 199) / 199.0f;
        v[static_cast<size_t>(i)] = (t - 0.5f) * 0.5f;  // FP8 source, per-tensor
        amax = std::fmax(amax, std::fabs(v[static_cast<size_t>(i)]));
    }
    // Per-tensor scale so that amax maps onto the E4M3 maximum (448).
    const float scale = amax > 0.0f ? amax / 448.0f : 1.0f;
    const std::string module = weight.substr(0, weight.size() - 7);  // strip ".weight"
    s.weight.name = weight;
    s.weight.dtype = "F8_E4M3";
    s.weight.shape = {R, C};
    s.weight.data.assign(static_cast<size_t>(R * C), 0);
    for (int64_t i = 0; i < R * C; ++i)
        s.weight.data[static_cast<size_t>(i)] = FloatToE4m3(v[static_cast<size_t>(i)] / scale);
    s.weight_scale.name = module + ".weight_scale";
    s.weight_scale.dtype = "F32";
    s.weight_scale.shape = {};             // scalar: per-tensor FP8
    s.weight_scale.data.assign(4, 0);
    std::memcpy(s.weight_scale.data.data(), &scale, 4);
    s.input_scale.name = module + ".input_scale";
    s.input_scale.dtype = "F32";
    s.input_scale.shape = {1};
    s.input_scale.data.assign(4, 0);
    const float isc = 0.25f;
    std::memcpy(s.input_scale.data.data(), &isc, 4);
    return s;
}

inline FakeTensor MakeBf16Tensor(const std::string &name,
                                 std::vector<int64_t> shape, int64_t seed)
{
    int64_t n = 1;
    for (auto d : shape) n *= d;
    std::vector<uint8_t> data(static_cast<size_t>(n) * 2);
    for (int64_t i = 0; i < n; ++i) {
        const float t = static_cast<float>((i * 17 + seed * 7) % 97) / 97.0f;
        const uint16_t b = FloatToBf16((t - 0.5f) * 0.3f);
        data[static_cast<size_t>(i) * 2] = static_cast<uint8_t>(b & 0xff);
        data[static_cast<size_t>(i) * 2 + 1] = static_cast<uint8_t>(b >> 8);
    }
    return FakeTensor{name, "BF16", std::move(shape), std::move(data)};
}

inline FakeTensor MakeF32Tensor(const std::string &name,
                                std::vector<int64_t> shape, int64_t seed,
                                float scale = 0.02f)
{
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return FakeTensor{name, "F32", std::move(shape),
                      MakeF32Values(n, [seed, scale](int64_t i) {
                          const float t =
                              static_cast<float>((i * 13 + seed * 3) % 89) / 89.0f;
                          return (t - 0.5f) * scale;
                      })};
}

// Writes config.json for the tiny 3-layer checkpoint. Field-for-field the same
// names the real vault's config.json uses, so ParseNemotronConfig exercises the
// real code path.
inline void WriteConfig(const std::filesystem::path &dir)
{
    std::ofstream f(dir / "config.json");
    f << R"({
  "architectures": ["NemotronHForCausalLM"],
  "model_type": "nemotron_h",
  "hidden_size": 16,
  "num_hidden_layers": 3,
  "vocab_size": 32,
  "mamba_num_heads": 2,
  "mamba_head_dim": 8,
  "ssm_state_size": 4,
  "n_groups": 1,
  "conv_kernel": 4,
  "num_attention_heads": 2,
  "num_key_value_heads": 1,
  "head_dim": 8,
  "rope_theta": 10000,
  "layer_norm_epsilon": 1e-5,
  "n_routed_experts": 4,
  "num_experts_per_tok": 2,
  "moe_intermediate_size": 16,
  "moe_shared_expert_intermediate_size": 32,
  "moe_hidden_act": "relu2",
  "routed_scaling_factor": 2.5,
  "norm_topk_prob": true,
  "sliding_window": null,
  "partial_rotary_factor": 1.0,
  "rescale_prenorm_residual": true,
  "tie_word_embeddings": false,
  "eos_token_id": 2,
  "bos_token_id": 1,
  "pad_token_id": 0,
  "num_nextn_predict_layers": 1,
  "mtp_layers_block_type": ["attention", "moe"],
  "layers_block_type": ["mamba", "moe", "attention"],
  "quantization_config": {"quant_algo": "MIXED_PRECISION", "producer": "modelopt"}
})";
}

// Writes the whole tiny PRE-QUANTIZED checkpoint (config + one safetensors
// shard). Every pipeline path of the converter is covered: globals, FP8 dense,
// NVFP4 routed + shared experts, NVFP4 lm_head, BF16 attention with FP8 KV
// scales, F32 router, and the MTP ballast that must be cut and counted.
inline void WriteNemotronNvfp4Checkpoint(const std::filesystem::path &dir)
{
    WriteConfig(dir);

    std::vector<FakeTensor> ts;
    int64_t seed = 1;

    // Globals: embeddings stay BF16 (excluded from quantization); lm_head is
    // NVFP4 in the vault.
    ts.push_back(MakeBf16Tensor("backbone.embeddings.weight", {kTVocab, kTH},
                                seed++));
    {
        auto lm = MakeNvfp4("lm_head.weight", kTVocab, kTH, seed++);
        ts.push_back(lm.weight);
        ts.push_back(lm.weight_scale);
        ts.push_back(lm.weight_scale_2);
    }
    ts.push_back(MakeBf16Tensor("backbone.norm_f.weight", {kTH}, seed++));

    // Layer 0: mamba. in_proj / out_proj are per-tensor FP8.
    ts.push_back(MakeBf16Tensor("backbone.layers.0.norm.weight", {kTH}, seed++));
    {
        auto ip = MakeFp8("backbone.layers.0.mixer.in_proj.weight", kTMambaIn,
                          kTH, seed++);
        ts.push_back(ip.weight); ts.push_back(ip.weight_scale);
        ts.push_back(ip.input_scale);
        auto op = MakeFp8("backbone.layers.0.mixer.out_proj.weight", kTH,
                          kTAttnOut, seed++);
        ts.push_back(op.weight); ts.push_back(op.weight_scale);
        ts.push_back(op.input_scale);
    }
    ts.push_back(MakeBf16Tensor("backbone.layers.0.mixer.conv1d.weight",
                                {kTConvDim, 1, 4}, seed++));
    ts.push_back(MakeBf16Tensor("backbone.layers.0.mixer.conv1d.bias",
                                {kTConvDim}, seed++));
    ts.push_back(MakeBf16Tensor("backbone.layers.0.mixer.A_log", {2}, seed++));
    ts.push_back(MakeBf16Tensor("backbone.layers.0.mixer.D", {2}, seed++));
    ts.push_back(MakeBf16Tensor("backbone.layers.0.mixer.dt_bias", {2}, seed++));
    ts.push_back(MakeBf16Tensor("backbone.layers.0.mixer.norm.weight",
                                {kTAttnOut}, seed++));

    // Layer 1: moe. Router F32, routed + shared experts NVFP4.
    ts.push_back(MakeBf16Tensor("backbone.layers.1.norm.weight", {kTH}, seed++));
    ts.push_back(MakeF32Tensor("backbone.layers.1.mixer.gate.weight",
                               {kTExperts, kTH}, seed++));
    ts.push_back(MakeF32Tensor(
        "backbone.layers.1.mixer.gate.e_score_correction_bias", {kTExperts},
        seed++));
    {
        auto su = MakeNvfp4("backbone.layers.1.mixer.shared_experts.up_proj.weight",
                            kTShared, kTH, seed++);
        ts.push_back(su.weight); ts.push_back(su.weight_scale);
        ts.push_back(su.weight_scale_2);
        auto sd = MakeNvfp4(
            "backbone.layers.1.mixer.shared_experts.down_proj.weight", kTH,
            kTShared, seed++);
        ts.push_back(sd.weight); ts.push_back(sd.weight_scale);
        ts.push_back(sd.weight_scale_2);
    }
    for (int e = 0; e < kTExperts; ++e) {
        auto up = MakeNvfp4("backbone.layers.1.mixer.experts." +
                                std::to_string(e) + ".up_proj.weight",
                            kTI, kTH, seed++);
        ts.push_back(up.weight); ts.push_back(up.weight_scale);
        ts.push_back(up.weight_scale_2);
        auto dn = MakeNvfp4("backbone.layers.1.mixer.experts." +
                                std::to_string(e) + ".down_proj.weight",
                            kTH, kTI, seed++);
        ts.push_back(dn.weight); ts.push_back(dn.weight_scale);
        ts.push_back(dn.weight_scale_2);
    }

    // Layer 2: attention. BF16 weights + the FP8 KV-cache scales that the
    // converter must classify and DROP (they are not weight dequant factors).
    ts.push_back(MakeBf16Tensor("backbone.layers.2.norm.weight", {kTH}, seed++));
    ts.push_back(MakeBf16Tensor("backbone.layers.2.mixer.q_proj.weight",
                                {kTAttnOut, kTH}, seed++));
    ts.push_back(MakeBf16Tensor("backbone.layers.2.mixer.k_proj.weight",
                                {kTAttnOut / 2, kTH}, seed++));
    ts.push_back(MakeBf16Tensor("backbone.layers.2.mixer.v_proj.weight",
                                {kTAttnOut / 2, kTH}, seed++));
    ts.push_back(MakeBf16Tensor("backbone.layers.2.mixer.o_proj.weight",
                                {kTH, kTAttnOut}, seed++));
    ts.push_back(MakeF32Tensor("backbone.layers.2.mixer.k_proj.k_scale", {1},
                               seed++, 0.031f));
    ts.push_back(MakeF32Tensor("backbone.layers.2.mixer.v_proj.v_scale", {1},
                               seed++, 0.003f));

    // MTP ballast: 2 blocks in the vault (attention + moe). Cut and counted.
    ts.push_back(MakeBf16Tensor("mtp.layers.0.mixer.q_proj.weight",
                                {kTAttnOut, kTH}, seed++));
    ts.push_back(MakeBf16Tensor("mtp.layers.0.mixer.experts.0.up_proj.weight",
                                {kTI, kTH}, seed++));
    ts.push_back(MakeBf16Tensor("mtp.layers.1.mixer.q_proj.weight",
                                {kTAttnOut, kTH}, seed++));

    WriteSafetensors(dir, "model.safetensors", ts);
}

// Real-frozen-geometry layer mix (attention at 5/12/19/26/33/42), identical to
// the BF16-distribution sibling - the two models differ only in how the SOURCE
// checkpoint was encoded.
inline std::vector<std::string> RealLayerTypes()
{
    // Taken verbatim from the vault's layers_block_type (52 entries, full
    // attention at 5/12/19/26/33/42, 23 mamba + 23 moe + 6 attention) and
    // asserted equal to the BF16 sibling's fixture list - the two
    // distributions describe the same layer mix.
    return {    "mamba", "moe", "mamba", "moe", "mamba", "attention", "moe",
    "mamba", "moe", "mamba", "moe", "mamba", "attention", "moe",
    "mamba", "moe", "mamba", "moe", "mamba", "attention", "moe",
    "mamba", "moe", "mamba", "moe", "mamba", "attention", "moe",
    "mamba", "moe", "mamba", "moe", "mamba", "attention", "moe",
    "mamba", "moe", "mamba", "moe", "mamba", "moe", "mamba",
    "attention", "moe", "mamba", "moe", "mamba", "moe", "mamba",
    "moe", "mamba", "moe",};
}

// Writes a real-geometry model.index.json (no weights) so the loader's frozen
// passport can be exercised without the synthetic-geometry bypass. `model_name`
// defaults to THIS variant; the loader-stage test uses the override to prove a
// sibling tree cannot be loaded under this model's kType.
inline void WriteNemotronNvfp4Index(
    const std::filesystem::path &model_root, int hidden_override = 2688,
    bool experts_no_gate = true,
    const std::string &format_tag = "logestix-nemotron-nvfp4",
    const std::string &model_name = "Nemotron-3.5-Lightning-30B-A3B-NVFP4",
    const std::string &eos_json = "[2, 11]")
{
    std::filesystem::create_directories(model_root / "logestix");
    const auto types = RealLayerTypes();
    std::ofstream f(model_root / "logestix" / "model.index.json");
    f << "{\n  \"format\": \"" << format_tag << "\",\n"
      << "  \"model\": \"" << model_name << "\",\n"
      << "  \"num_layers\": 52,\n  \"layer_types\": [";
    for (size_t i = 0; i < types.size(); ++i) {
        if (i) f << ",";
        f << "\"" << types[i] << "\"";
    }
    f << "],\n  \"attn_layer_idx\": [5,12,19,26,33,42],\n"
      << "  \"hidden_size\": " << hidden_override << ",\n"
      << "  \"vocab_size\": 131072,\n  \"mamba_num_heads\": 64,\n"
      << "  \"mamba_head_dim\": 64,\n  \"ssm_state_size\": 128,\n"
      << "  \"n_groups\": 8,\n  \"conv_kernel\": 4,\n"
      << "  \"num_attention_heads\": 32,\n  \"num_key_value_heads\": 2,\n"
      << "  \"head_dim\": 128,\n  \"rope_theta\": 10000,\n"
      << "  \"rms_norm_eps\": 1e-05,\n  \"num_experts\": 128,\n"
      << "  \"num_experts_per_tok\": 6,\n  \"moe_intermediate_size\": 1856,\n"
      << "  \"shared_expert_intermediate_size\": 3712,\n"
      << "  \"router_scaling_factor\": 2.5,\n"
      << "  \"experts_no_gate\": " << (experts_no_gate ? "true" : "false")
      << ",\n  \"bos_token_id\": 1,\n  \"eos_token_id\": " << eos_json
      << ",\n  \"pad_token_id\": 0,\n  \"fp8_kv_scales_dropped\": 12,\n"
      << "  \"deferred\": {\"mtp\": \"cut\", \"mtp_tensors\": 15},\n"
      << "  \"total_pinned_tensors\": 0\n}\n";
}

} // namespace nemotronv4test