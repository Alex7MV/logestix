#pragma once

// Nemotron-3-Ultra-550B synthetic checkpoint fixture. Family-owned: it
// re-exports the family-neutral primitives of test_synth_checkpoint.h under
// `ultratest::` and builds a tiny NemotronH checkpoint whose geometry exercises
// every Ultra-only path without touching the real vault:
//
//   * grouped routing      n_group 4 over 8 experts, topk_group 1, top-2
//   * latent MoE           moe_latent_size 8, experts at the LATENT width
//                          (up 8->16, down 16->8), fc1/fc2 bridge 16->8->16
//   * mamba-2              2 heads x head_dim 8, 2 state groups
//   * NoPE attention       GQA 4:2 over head_dim 4
//   * MTP cut              3 `mtp.*` tensors, counted and dropped
//
// The 30B sibling's fixture is hidden 16 / 4 experts / no latent / n_group 1.
// No geometry value here may coincide with it: a copied fixture would let a
// cross-family copy pass both suites, which is the failure
// test_model_suite_guard exists to catch.

#include "test_synth_checkpoint.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace ultratest {

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

// ── Tiny Ultra geometry (LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1 bypasses the
// ── frozen-passport gate; the real values live in the passport header) ──────
inline constexpr int kTH = 16;      // hidden
inline constexpr int kTVocab = 32;
inline constexpr int kTAH = 4;      // attn heads
inline constexpr int kTAHD = 4;     // attn head_dim (kTAH*kTAHD == kTH)
inline constexpr int kTKVH = 2;     // kv heads (GQA 2:1)
inline constexpr int kTMH = 2;      // mamba heads
inline constexpr int kTMHD = 8;     // mamba head_dim => d_inner 16
inline constexpr int kTSsm = 4;     // ssm state size
inline constexpr int kTGroups = 2;  // ssm groups (state, not router)
inline constexpr int kTConvDim = kTMH * kTMHD + 2 * kTGroups * kTSsm;  // 32
inline constexpr int kTMambaIn = kTMH * kTMHD + kTConvDim + kTMH;      // 50
inline constexpr int kTExperts = 8;
inline constexpr int kTGroup = 4;      // n_group: 2 experts per group
inline constexpr int kTTopkGroup = 1;  // topk_group
inline constexpr int kTTopK = 2;
inline constexpr int kTI = 16;         // moe intermediate (at the latent width)
inline constexpr int kTLatent = 8;     // moe_latent_size
inline constexpr int kTShared = 32;
inline constexpr int kTMtpTensors = 3;  // exactly the `mtp.*` tensors written

// Layer mix of the synthetic model: one of each block type, in the order the
// converter and the pipeline both walk.
inline std::vector<std::string> LayerTypes() { return {"mamba", "moe", "attention"}; }

// Writes `<dir>/config.json` + `<dir>/model.safetensors` (one shard, BF16/F32
// payloads, tensor names byte-identical to the real checkpoint's).
inline void WriteUltraConfig(const std::filesystem::path &dir)
{
    std::ofstream f(dir / "config.json");
    f << R"({
  "architectures": ["NemotronHForCausalLM"],
  "model_type": "nemotron_h",
  "hidden_size": )" << kTH << R"(,
  "num_hidden_layers": 3,
  "vocab_size": )" << kTVocab << R"(,
  "mamba_num_heads": )" << kTMH << R"(,
  "mamba_head_dim": )" << kTMHD << R"(,
  "ssm_state_size": )" << kTSsm << R"(,
  "n_groups": )" << kTGroups << R"(,
  "conv_kernel": 4,
  "num_attention_heads": )" << kTAH << R"(,
  "num_key_value_heads": )" << kTKVH << R"(,
  "head_dim": )" << kTAHD << R"(,
  "rope_theta": 10000,
  "layer_norm_epsilon": 1e-5,
  "n_routed_experts": )" << kTExperts << R"(,
  "num_experts_per_tok": )" << kTTopK << R"(,
  "n_group": )" << kTGroup << R"(,
  "topk_group": )" << kTTopkGroup << R"(,
  "norm_topk_prob": true,
  "routed_scaling_factor": 5.0,
  "moe_intermediate_size": )" << kTI << R"(,
  "moe_latent_size": )" << kTLatent << R"(,
  "moe_shared_expert_intermediate_size": )" << kTShared << R"(,
  "mlp_hidden_act": "relu2",
  "mamba_hidden_act": "silu",
  "eos_token_id": 2,
  "bos_token_id": 1,
  "pad_token_id": 0,
  "num_nextn_predict_layers": 1,
  "layers_block_type": ["mamba", "moe", "attention"]
})";
    // The real checkpoint splits the stop set across the two files: config.json
    // carries a scalar, generation_config.json the authoritative list.
    std::ofstream g(dir / "generation_config.json");
    g << R"({
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": [2, 11],
  "pad_token_id": 0
})";
}

// The full tensor inventory the fixture writes. Exposed so a negative case can
// drop or mutate exactly one entry instead of hand-writing a shard.
inline std::vector<FakeTensor> MakeUltraTensors()
{
    std::vector<FakeTensor> ts;
    // Values, not raw bit patterns: `MakeBf16` writes i % 65536, which are BF16
    // subnormals that underflow to zero once uploaded as FP16 — fine for a
    // structural fixture, useless for a forward pass. This pattern stays inside
    // a plausible weight range and never lands on exact zero.
    auto bf16 = [](const std::string &n, std::vector<int64_t> shape) {
        int64_t n_elem = 1;
        for (auto d : shape) n_elem *= d;
        return FakeTensor{n, "BF16", std::move(shape),
                          synthtest::MakeBf16Values(n_elem, [](int64_t i) {
                              return 0.05f * static_cast<float>(i % 17) - 0.4f;
                          })};
    };

    // Globals (untied embeddings + head).
    ts.push_back(bf16("backbone.embeddings.weight", {kTVocab, kTH}));
    ts.push_back(bf16("lm_head.weight", {kTVocab, kTH}));
    ts.push_back(bf16("backbone.norm_f.weight", {kTH}));

    // Layer 0: mamba.
    ts.push_back(bf16("backbone.layers.0.norm.weight", {kTH}));
    ts.push_back(bf16("backbone.layers.0.mixer.in_proj.weight", {kTMambaIn, kTH}));
    ts.push_back(bf16("backbone.layers.0.mixer.out_proj.weight", {kTH, kTMH * kTMHD}));
    ts.push_back(bf16("backbone.layers.0.mixer.conv1d.weight", {kTConvDim, 1, 4}));
    ts.push_back(bf16("backbone.layers.0.mixer.conv1d.bias", {kTConvDim}));
    ts.push_back(bf16("backbone.layers.0.mixer.A_log", {kTMH}));
    ts.push_back(bf16("backbone.layers.0.mixer.D", {kTMH}));
    ts.push_back(bf16("backbone.layers.0.mixer.dt_bias", {kTMH}));
    ts.push_back(bf16("backbone.layers.0.mixer.norm.weight", {kTMH * kTMHD}));

    // Layer 1: moe with the latent bridge. The routed experts live at the
    // LATENT width (kTLatent), not at hidden - that is what Ultra adds over the
    // 30B sibling and what fc1/fc2 are for.
    ts.push_back(bf16("backbone.layers.1.norm.weight", {kTH}));
    ts.push_back(bf16("backbone.layers.1.mixer.gate.weight", {kTExperts, kTH}));
    ts.push_back(FakeTensor{
        "backbone.layers.1.mixer.gate.e_score_correction_bias", "F32", {kTExperts},
        MakeF32Values(kTExperts, [](int64_t i) {
            return 0.01f * static_cast<float>(i % 5);
        })});
    ts.push_back(bf16("backbone.layers.1.mixer.fc1_latent_proj.weight", {kTLatent, kTH}));
    ts.push_back(bf16("backbone.layers.1.mixer.fc2_latent_proj.weight", {kTH, kTLatent}));
    ts.push_back(
        bf16("backbone.layers.1.mixer.shared_experts.up_proj.weight", {kTShared, kTH}));
    ts.push_back(
        bf16("backbone.layers.1.mixer.shared_experts.down_proj.weight", {kTH, kTShared}));
    for (int e = 0; e < kTExperts; ++e) {
        ts.push_back(bf16("backbone.layers.1.mixer.experts." + std::to_string(e) +
                              ".up_proj.weight",
                          {kTI, kTLatent}));
        ts.push_back(bf16("backbone.layers.1.mixer.experts." + std::to_string(e) +
                              ".down_proj.weight",
                          {kTLatent, kTI}));
    }

    // Layer 2: full attention (NoPE - no rotary tensors exist for this family).
    ts.push_back(bf16("backbone.layers.2.norm.weight", {kTH}));
    ts.push_back(bf16("backbone.layers.2.mixer.q_proj.weight", {kTAH * kTAHD, kTH}));
    ts.push_back(bf16("backbone.layers.2.mixer.k_proj.weight", {kTKVH * kTAHD, kTH}));
    ts.push_back(bf16("backbone.layers.2.mixer.v_proj.weight", {kTKVH * kTAHD, kTH}));
    ts.push_back(bf16("backbone.layers.2.mixer.o_proj.weight", {kTH, kTAH * kTAHD}));

    // MTP head: the reference drops `mtp.*` on load; the converter must CUT and
    // count them (kTMtpTensors), never silently ignore an unknown tensor.
    ts.push_back(bf16("mtp.layers.0.norm.weight", {kTH}));
    ts.push_back(bf16("mtp.layers.0.mixer.q_proj.weight", {kTAH * kTAHD, kTH}));
    ts.push_back(bf16("mtp.layers.0.mixer.k_proj.weight", {kTKVH * kTAHD, kTH}));
    return ts;
}

inline void WriteUltraCheckpoint(const std::filesystem::path &dir)
{
    WriteUltraConfig(dir);
    WriteSafetensors(dir, "model.safetensors", MakeUltraTensors());
}

// Overrides for the converted index so the loader's negative cases can be
// written without re-running the converter for each one.
struct UltraIndexOptions {
    std::string format = "logestix-nemotron-ultra-nvfp4";
    std::string model = "NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16";
    int num_layers = 3;
    int hidden_size = kTH;
    int vocab_size = kTVocab;
    int mamba_num_heads = kTMH;
    int mamba_head_dim = kTMHD;
    int ssm_state_size = kTSsm;
    int n_groups = kTGroups;
    int conv_kernel = 4;
    int num_attention_heads = kTAH;
    int num_key_value_heads = kTKVH;
    int head_dim = kTAHD;
    int num_experts = kTExperts;
    int num_experts_per_tok = kTTopK;
    int moe_intermediate_size = kTI;
    int shared_expert_intermediate_size = kTShared;
    int moe_latent_size = kTLatent;
    int n_group = kTGroup;
    int topk_group = kTTopkGroup;
    float routed_scaling_factor = 5.0f;
    bool norm_topk_prob = true;
    int mtp_tensors = kTMtpTensors;
    bool partial = false;
    std::vector<int> layers_converted = {0, 1, 2};
    long long peak_expert_records = static_cast<long long>(kTExperts);
};

// Writes `<root>/logestix/model.index.json` only. The TENSORS are produced by
// the converter under test (tests run it on WriteUltraCheckpoint output) rather
// than hand-written here: a hand-written Arrow tree would make a converter
// schema bug invisible to both the writer and the reader.
inline void WriteUltraIndex(const std::filesystem::path &model_root,
                            const UltraIndexOptions &opts = {})
{
    std::filesystem::create_directories(model_root / "logestix");
    const auto types = LayerTypes();
    std::ofstream f(model_root / "logestix" / "model.index.json");
    f << "{\n  \"format\": \"" << opts.format << "\",\n"
      << "  \"model\": \"" << opts.model << "\",\n"
      << "  \"num_layers\": " << opts.num_layers << ",\n  \"layer_types\": [";
    for (std::size_t i = 0; i < types.size(); ++i) {
        if (i) f << ",";
        f << "\"" << types[i] << "\"";
    }
    f << "],\n  \"attn_layer_idx\": [2],\n"
      << "  \"hidden_size\": " << opts.hidden_size << ",\n"
      << "  \"vocab_size\": " << opts.vocab_size << ",\n"
      << "  \"mamba_num_heads\": " << opts.mamba_num_heads << ",\n"
      << "  \"mamba_head_dim\": " << opts.mamba_head_dim << ",\n"
      << "  \"ssm_state_size\": " << opts.ssm_state_size << ",\n"
      << "  \"n_groups\": " << opts.n_groups << ",\n"
      << "  \"conv_kernel\": " << opts.conv_kernel << ",\n"
      << "  \"num_attention_heads\": " << opts.num_attention_heads << ",\n"
      << "  \"num_key_value_heads\": " << opts.num_key_value_heads << ",\n"
      << "  \"head_dim\": " << opts.head_dim << ",\n"
      << "  \"rms_norm_eps\": 1e-05,\n"
      << "  \"num_experts\": " << opts.num_experts << ",\n"
      << "  \"num_experts_per_tok\": " << opts.num_experts_per_tok << ",\n"
      << "  \"moe_intermediate_size\": " << opts.moe_intermediate_size << ",\n"
      << "  \"shared_expert_intermediate_size\": "
      << opts.shared_expert_intermediate_size << ",\n"
      << "  \"moe_latent_size\": " << opts.moe_latent_size << ",\n"
      << "  \"n_group\": " << opts.n_group << ",\n"
      << "  \"topk_group\": " << opts.topk_group << ",\n"
      << "  \"routed_scaling_factor\": " << opts.routed_scaling_factor << ",\n"
      << "  \"norm_topk_prob\": " << (opts.norm_topk_prob ? "true" : "false")
      << ",\n  \"bos_token_id\": 1,\n  \"eos_token_id\": [2, 11],\n"
      << "  \"pad_token_id\": 0,\n"
      << "  \"mtp_tensors\": " << opts.mtp_tensors << ",\n"
      << "  \"partial\": " << (opts.partial ? "true" : "false") << ",\n"
      << "  \"layers_converted\": [";
    for (std::size_t i = 0; i < opts.layers_converted.size(); ++i) {
        if (i) f << ",";
        f << opts.layers_converted[i];
    }
    f << "],\n  \"peak_expert_records\": " << opts.peak_expert_records << ",\n"
      << "  \"total_pinned_tensors\": 0\n}\n";
}

// Copies a checkpoint directory (config + shards) so a negative case can mutate
// it without disturbing the positive fixture.
inline std::filesystem::path CopyUltraCheckpoint(const std::filesystem::path &src,
                                                const std::string &tag)
{
    auto dst = MakeTempDir(tag);
    for (const auto &e : std::filesystem::directory_iterator(src)) {
        if (!e.is_regular_file()) continue;
        std::filesystem::copy_file(e.path(), dst / e.path().filename());
    }
    return dst;
}

} // namespace ultratest
