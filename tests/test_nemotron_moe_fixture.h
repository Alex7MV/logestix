#pragma once

// Nemotron-3.5 synthetic checkpoint fixture. Family-owned: it re-exports the
// family-neutral primitives of test_synth_checkpoint.h under `nemotrontest::`
// and builds a tiny NemotronH checkpoint (config.json + safetensors) that
// exercises every converter path without touching the real vault.

#include "test_synth_checkpoint.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace nemotrontest {

using synthtest::FakeTensor;
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

// Tiny NemotronH geometry. Frozen gates are bypassed in tests via
// LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1.
inline constexpr int kTH = 16;      // hidden
inline constexpr int kTVocab = 32;
inline constexpr int kTMambaIn = 42;   // d_inner 16 + conv_dim 24 + dt 2
inline constexpr int kTConvDim = 24;
inline constexpr int kTExperts = 4;
inline constexpr int kTI = 16;         // moe intermediate
inline constexpr int kTShared = 32;

inline void WriteNemotronCheckpoint(const std::filesystem::path &dir)
{
    // config.json: 3 layers (mamba, moe, attention), legacy block tags.
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
  "eos_token_id": 2,
  "bos_token_id": 1,
  "pad_token_id": 0,
  "num_nextn_predict_layers": 1,
  "layers_block_type": ["mamba", "moe", "attention"]
})";
    }

    std::vector<FakeTensor> ts;
    auto bf16 = [](const std::string &n, std::vector<int64_t> shape) {
        int64_t n_elem = 1;
        for (auto d : shape) n_elem *= d;
        return FakeTensor{n, "BF16", std::move(shape), MakeBf16(n_elem)};
    };

    // Globals.
    ts.push_back(bf16("backbone.embeddings.weight", {kTVocab, kTH}));
    ts.push_back(bf16("lm_head.weight", {kTVocab, kTH}));
    ts.push_back(bf16("backbone.norm_f.weight", {kTH}));

    // Layer 0: mamba.
    ts.push_back(bf16("backbone.layers.0.norm.weight", {kTH}));
    ts.push_back(bf16("backbone.layers.0.mixer.in_proj.weight", {kTMambaIn, kTH}));
    ts.push_back(bf16("backbone.layers.0.mixer.out_proj.weight", {kTH, 16}));
    ts.push_back(bf16("backbone.layers.0.mixer.conv1d.weight", {kTConvDim, 1, 4}));
    ts.push_back(bf16("backbone.layers.0.mixer.conv1d.bias", {kTConvDim}));
    ts.push_back(bf16("backbone.layers.0.mixer.A_log", {2}));
    ts.push_back(bf16("backbone.layers.0.mixer.D", {2}));
    ts.push_back(bf16("backbone.layers.0.mixer.dt_bias", {2}));
    ts.push_back(bf16("backbone.layers.0.mixer.norm.weight", {16}));

    // Layer 1: moe.
    ts.push_back(bf16("backbone.layers.1.norm.weight", {kTH}));
    ts.push_back(bf16("backbone.layers.1.mixer.gate.weight", {kTExperts, kTH}));
    ts.push_back(FakeTensor{"backbone.layers.1.mixer.gate.e_score_correction_bias",
                            "F32", {kTExperts}, MakeF32Values(kTExperts, [](int64_t) {
                                return 0.0f;
                            })});
    ts.push_back(bf16("backbone.layers.1.mixer.shared_experts.up_proj.weight",
                      {kTShared, kTH}));
    ts.push_back(bf16("backbone.layers.1.mixer.shared_experts.down_proj.weight",
                      {kTH, kTShared}));
    for (int e = 0; e < kTExperts; ++e) {
        ts.push_back(bf16("backbone.layers.1.mixer.experts." + std::to_string(e) +
                              ".up_proj.weight",
                          {kTI, kTH}));
        ts.push_back(bf16("backbone.layers.1.mixer.experts." + std::to_string(e) +
                              ".down_proj.weight",
                          {kTH, kTI}));
    }

    // Layer 2: attention.
    ts.push_back(bf16("backbone.layers.2.norm.weight", {kTH}));
    ts.push_back(bf16("backbone.layers.2.mixer.q_proj.weight", {16, kTH}));
    ts.push_back(bf16("backbone.layers.2.mixer.k_proj.weight", {8, kTH}));
    ts.push_back(bf16("backbone.layers.2.mixer.v_proj.weight", {8, kTH}));
    ts.push_back(bf16("backbone.layers.2.mixer.o_proj.weight", {kTH, 16}));

    // MTP head: must be cut and counted.
    ts.push_back(bf16("mtp.layers.0.mixer.q_proj.weight", {16, kTH}));

    WriteSafetensors(dir, "model.safetensors", ts);
}

// Opt-in: gives the synthetic checkpoint a real BPE tokenizer so the framing
// tests can encode/decode. Not written by WriteNemotronCheckpoint on purpose —
// the loader treats a missing tokenizer as optional, and the other nemotron
// tests must keep exercising that path.
// `tok_json` is passed in (not a TOKENIZER_TEST_DATA_DIR macro) so this header
// stays free of build-system globals: only the target that calls it needs the
// compile definition.
inline void WriteNemotronTokenizer(const std::filesystem::path &dir,
                                   const std::filesystem::path &tok_json)
{
    std::filesystem::copy_file(tok_json, dir / "tokenizer.json");
}

// Real frozen-geometry layer mix (attention at 5/12/19/26/33/42).
inline std::vector<std::string> RealLayerTypes()
{
    return {"mamba", "moe", "mamba", "moe", "mamba", "attention", "moe",
            "mamba", "moe",   "mamba", "moe",   "mamba",   "attention", "moe",
            "mamba", "moe",   "mamba", "moe",   "mamba",   "attention", "moe",
            "mamba", "moe",   "mamba", "moe",   "mamba",   "attention", "moe",
            "mamba", "moe",   "mamba", "moe",   "mamba",   "attention", "moe",
            "mamba", "moe",   "mamba", "moe",   "mamba",   "moe",       "mamba",
            "attention", "moe", "mamba", "moe", "mamba",   "moe", "mamba",
            "moe", "mamba", "moe"};
}

// Writes a real-geometry model.index.json (no weights) so the loader's frozen
// passport can be exercised without the synthetic-geometry bypass.
inline void WriteNemotronIndex(
    const std::filesystem::path &model_root, int hidden_override = 2688,
    bool experts_no_gate = true,
    const std::string &format_tag = "logestix-nemotron-nvfp4")
{
    std::filesystem::create_directories(model_root / "logestix");
    const auto types = RealLayerTypes();
    std::ofstream f(model_root / "logestix" / "model.index.json");
    f << "{\n  \"format\": \"" << format_tag << "\",\n"
      << "  \"model\": \"Nemotron-3.5-Lightning-30B-A3B\",\n"
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
      << ",\n  \"bos_token_id\": 1,\n  \"eos_token_id\": 2,\n"
      << "  \"pad_token_id\": 0,\n  \"deferred\": {\"mtp\": \"cut\", "
         "\"mtp_tensors\": 128},\n  \"total_pinned_tensors\": 0\n}\n";
}

} // namespace nemotrontest
