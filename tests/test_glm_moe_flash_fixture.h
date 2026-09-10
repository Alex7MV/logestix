#pragma once

// Shared fixture builder for GLM-5.3-Flash loader/converter tests.
// Builds a tiny synthetic safetensors checkpoint (KDA layer 0 + DSA/MoE
// layer 3, dense MLP on layer 0) and drives ConvertGlm5NextModel over it,
// producing the full logestix layout (pinned globals/layers, NVFP4 batches,
// per-expert arrows, index) that LoadGlmFlashDenseWeights /
// LoadGlmFlashExpertStore strictly consume. Unified loader-test discipline:
// loader suites never load the model vault — they convert this fixture.
// Tensor-writing primitives come from the family-neutral
// test_synth_checkpoint.h (suite-autonomy invariant).

#include "test_synth_checkpoint.h"

#include <absl/strings/str_cat.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <lsxcommon/glm_moe_flash_convert.h>

namespace glmflash {

// Cross-platform env gate set/clear (etalon: test_qwen_moe_fixture.h).
inline void SetSyntheticGate()
{
#ifdef _WIN32
    _putenv_s("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1");
#else
    setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1", 1);
#endif
}

inline void ClearSyntheticGate()
{
#ifdef _WIN32
    _putenv_s("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "");
#else
    unsetenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY");
#endif
}

// Copies the shared synthetic BPE tokenizer into `dir` so the converter
// builds the tokenizer arrows the model's BeginLoad requires.
inline void WriteTokenizer(const std::filesystem::path &dir)
{
    std::filesystem::copy_file(
        TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
        dir / "tokenizer.json",
        std::filesystem::copy_options::overwrite_existing);
}

// Minimal text_config: ParseFlashConfig requires hidden/layers/vocab + MoE
// fields; everything else is defaulted and flows into the index. Layer count
// is the frozen main-stack depth so the loader's layer loop matches the
// frozen-range layout the converter always writes.
inline void WriteConfig(const std::filesystem::path &dir)
{
    std::ofstream cfg(dir / "config.json");
    cfg << R"json({"text_config":{
  "hidden_size":64,"num_hidden_layers":45,"vocab_size":256,
  "n_routed_experts":2,"num_experts_per_tok":1,
  "moe_intermediate_size":128,"intermediate_size":256,
  "hc_mult":4,"index_n_heads":8,"index_head_dim":16,
  "index_topk":8,"qk_head_dim":32,"q_lora_rank":32,
  "kv_lora_rank":32,"routed_scaling_factor":2.5,
  "eos_token_id":[100],"pad_token_id":100,
  "linear_attn_config":{"short_conv_kernel_size":4,
                        "num_heads":8,"head_dim":32}}})json";
}

namespace detail {

// Scale tile: 1x1 F32 grid covers a [128,128] fp8 tensor (ceil/128).
inline std::vector<uint8_t> Fp8Vals(int64_t n)
{
    std::vector<uint8_t> v(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        v[static_cast<size_t>(i)] = static_cast<uint8_t>(0x08 + (i % 0x70));
    return v;
}

inline std::vector<uint8_t> F32Scale(float s)
{
    std::vector<uint8_t> b(4);
    std::memcpy(b.data(), &s, 4);
    return b;
}

inline void AppendFp8(std::vector<synthtest::FakeTensor> &t,
                      const std::string &weight_name, int rows, int cols)
{
    t.push_back({weight_name, "F8_E4M3", {rows, cols},
                 Fp8Vals(static_cast<int64_t>(rows) * cols)});
    // Converter expects the scale tile named after the weight with the
    // ".weight" suffix replaced (…/gate_proj.weight_scale_inv).
    std::string scale = weight_name;
    scale.replace(scale.size() - strlen(".weight"), strlen(".weight"),
                  ".weight_scale_inv");
    t.push_back({scale, "F32", {1, 1}, F32Scale(1.5f)});
}

// Hyper-connection sextet shared by every layer (fn [24, hc*H] BF16 ->
// upconverted F32 by the loader; base/scale raw F32).
inline void AppendHc(std::vector<synthtest::FakeTensor> &t, int l)
{
    const std::string p =
        absl::StrCat("model.language_model.layers.", l, ".");
    t.push_back({p + "hc_attn_fn", "BF16", {24, 256}, synthtest::MakeBf16(24 * 256)});
    t.push_back({p + "hc_attn_base", "F32", {24}, synthtest::MakeF32(24)});
    t.push_back({p + "hc_attn_scale", "F32", {3}, synthtest::MakeF32(3)});
    t.push_back({p + "hc_ffn_fn", "BF16", {24, 256}, synthtest::MakeBf16(24 * 256)});
    t.push_back({p + "hc_ffn_base", "F32", {24}, synthtest::MakeF32(24)});
    t.push_back({p + "hc_ffn_scale", "F32", {3}, synthtest::MakeF32(3)});
}

inline void AppendLayernorms(std::vector<synthtest::FakeTensor> &t, int l)
{
    const std::string p =
        absl::StrCat("model.language_model.layers.", l, ".");
    t.push_back({p + "input_layernorm.weight", "BF16", {64},
                 synthtest::MakeBf16(64)});
    t.push_back({p + "post_attention_layernorm.weight", "BF16", {64},
                 synthtest::MakeBf16(64)});
}

} // namespace detail

inline constexpr int kSynthH = 64;    // hidden
inline constexpr int kSynthV = 256;   // vocab
inline constexpr int kSynthE = 2;     // routed experts
inline constexpr int kSynthD = 128;   // fp8 tile side (GPU: 128-multiples)

// Full synthetic checkpoint: strict loader coverage for the KDA layer 0
// (four NVFP4 projections + FP16 conv/b_proj/gating + F32 scalars + dense
// MLP) and the DSA/MoE layer 3 (FP8->NVFP4 requant, indexer records, router,
// shared expert, routed experts).
inline std::vector<synthtest::FakeTensor> BuildSyntheticFlashTensors()
{
    using synthtest::FakeTensor;
    using synthtest::MakeBf16;
    std::vector<FakeTensor> t;

    // Globals: raw BF16 pass.
    t.push_back({"model.language_model.embed_tokens.weight", "BF16",
                 {kSynthV, kSynthH}, MakeBf16(kSynthV * kSynthH)});
    t.push_back({"lm_head.weight", "BF16", {kSynthV, kSynthH},
                 MakeBf16(kSynthV * kSynthH)});
    t.push_back({"model.language_model.norm.weight", "BF16", {kSynthH},
                 MakeBf16(kSynthH)});

    // ── Layer 0: KDA + dense MLP. ──
    {
        const std::string p = "model.language_model.layers.0.";
        detail::AppendLayernorms(t, 0);
        const auto a = p + "self_attn.";
        t.push_back({a + "q_proj.weight", "BF16", {kSynthD, kSynthH},
                     MakeBf16(kSynthD * kSynthH)});
        t.push_back({a + "k_proj.weight", "BF16", {kSynthD, kSynthH},
                     MakeBf16(kSynthD * kSynthH)});
        t.push_back({a + "v_proj.weight", "BF16", {kSynthD, kSynthH},
                     MakeBf16(kSynthD * kSynthH)});
        t.push_back({a + "o_proj.weight", "BF16", {kSynthH, kSynthD},
                     MakeBf16(kSynthH * kSynthD)});
        for (const char *c : {"q_conv1d", "k_conv1d", "v_conv1d"})
            t.push_back({a + std::string(c) + ".weight", "BF16", {kSynthH, 4},
                         MakeBf16(kSynthH * 4)});
        t.push_back({a + "b_proj.weight", "BF16", {32, kSynthH},
                     MakeBf16(32 * kSynthH)});
        t.push_back({a + "f_a_proj.weight", "BF16", {16, kSynthH},
                     MakeBf16(16 * kSynthH)});
        t.push_back({a + "f_b_proj.weight", "BF16", {kSynthD, 16},
                     MakeBf16(kSynthD * 16)});
        t.push_back({a + "g_a_proj.weight", "BF16", {16, kSynthH},
                     MakeBf16(16 * kSynthH)});
        t.push_back({a + "g_b_proj.weight", "BF16", {kSynthD, 16},
                     MakeBf16(kSynthD * 16)});
        t.push_back({a + "o_norm.weight", "BF16", {16}, MakeBf16(16)});
        t.push_back({a + "A_log", "F32", {8}, synthtest::MakeF32(8)});
        t.push_back({a + "dt_bias", "F32", {kSynthH}, synthtest::MakeF32(kSynthH)});
        // Dense MLP (layers 0..2): NVFP4 gate/up/down.
        t.push_back({p + "mlp.gate_proj.weight", "BF16", {kSynthH, kSynthH},
                     MakeBf16(kSynthH * kSynthH)});
        t.push_back({p + "mlp.up_proj.weight", "BF16", {kSynthH, kSynthH},
                     MakeBf16(kSynthH * kSynthH)});
        t.push_back({p + "mlp.down_proj.weight", "BF16", {kSynthH, kSynthH},
                     MakeBf16(kSynthH * kSynthH)});
        detail::AppendHc(t, 0);
    }

    // ── Layer 3: DSA + first MoE layer. ──
    {
        const std::string p = "model.language_model.layers.3.";
        detail::AppendLayernorms(t, 3);
        const auto a = p + "self_attn.";
        detail::AppendFp8(t, a + "q_a_proj.weight", kSynthD, kSynthD);
        t.push_back({a + "q_b_proj.weight", "BF16", {kSynthD, 32},
                     MakeBf16(kSynthD * 32)});
        detail::AppendFp8(t, a + "kv_a_proj_with_mqa.weight", kSynthD, kSynthD);
        t.push_back({a + "o_proj.weight", "BF16", {kSynthH, 4 * kSynthD},
                     MakeBf16(kSynthH * 4 * kSynthD)});
        t.push_back({a + "q_a_layernorm.weight", "BF16", {32}, MakeBf16(32)});
        t.push_back({a + "kv_a_layernorm.weight", "BF16", {32}, MakeBf16(32)});
        t.push_back({a + "kv_b_proj.weight", "BF16", {512, 32},
                     MakeBf16(512 * 32)});
        // Indexer: raw BF16/F32 records (DSA kernel consumes them directly).
        t.push_back({a + "indexer.wq_b.weight", "BF16", {16, kSynthH},
                     MakeBf16(16 * kSynthH)});
        t.push_back({a + "indexer.wk.weight", "BF16", {16, kSynthH},
                     MakeBf16(16 * kSynthH)});
        t.push_back({a + "indexer.weights_proj.weight", "BF16", {16, kSynthH},
                     MakeBf16(16 * kSynthH)});
        t.push_back({a + "indexer.k_norm.weight", "BF16", {16}, MakeBf16(16)});
        t.push_back({a + "indexer.k_norm.bias", "F32", {16},
                     synthtest::MakeF32(16)});
        t.push_back({a + "indexer.index_kpool_compress_ape", "BF16", {8},
                     MakeBf16(8)});
        t.push_back({a + "indexer.index_kpool_compress_gate", "BF16", {8},
                     MakeBf16(8)});
        detail::AppendHc(t, 3);
        // Router.
        t.push_back({p + "mlp.gate.weight", "BF16", {kSynthE, kSynthH},
                     MakeBf16(kSynthE * kSynthH)});
        t.push_back({p + "mlp.gate.e_score_correction_bias", "F32", {kSynthE},
                     detail::F32Scale(0.25f)});
        // Shared expert (BF16 -> NVFP4).
        t.push_back({p + "mlp.shared_experts.gate_proj.weight", "BF16",
                     {kSynthH, kSynthH}, MakeBf16(kSynthH * kSynthH)});
        t.push_back({p + "mlp.shared_experts.up_proj.weight", "BF16",
                     {kSynthH, kSynthH}, MakeBf16(kSynthH * kSynthH)});
        t.push_back({p + "mlp.shared_experts.down_proj.weight", "BF16",
                     {kSynthH, kSynthH}, MakeBf16(kSynthH * kSynthH)});
        // Routed experts: FP8 -> NVFP4 requant with scale tiles.
        for (int e = 0; e < kSynthE; ++e) {
            const std::string ep =
                p + "mlp.experts." + std::to_string(e) + ".";
            detail::AppendFp8(t, ep + "gate_proj.weight", kSynthD, kSynthD);
            detail::AppendFp8(t, ep + "up_proj.weight", kSynthD, kSynthD);
            detail::AppendFp8(t, ep + "down_proj.weight", kSynthD, kSynthD);
        }
    }

    return t;
}

struct ConvertedFlashFixture
{
    std::filesystem::path in;
    std::filesystem::path out;

    // Builds <in> (config + safetensors) and converts it into <out> with the
    // synthetic-geometry gate held only for the conversion call (restored to
    // its prior state afterwards — the ctest ENVIRONMENT owns the load-side
    // gate).
    arrow::Status Convert()
    {
        const bool gate_was_set =
            std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") != nullptr;
        std::filesystem::create_directories(in);
        WriteConfig(in);
        WriteTokenizer(in);
        synthtest::WriteSafetensors(in, "model.safetensors",
                                    BuildSyntheticFlashTensors());
        SetSyntheticGate();
        auto st = lsxcommon::glm_moe_flash::ConvertGlm5NextModel(in.string(),
                                                                 out.string());
        if (!gate_was_set) ClearSyntheticGate();
        return st;
    }
};

} // namespace glmflash
