#pragma once

// Hy-SimulMT-30B-A3B synthetic-checkpoint fixture: the Hy-Memory text
// backbone (same tensor names/shapes) plus a speech encoder shard that must be
// stripped and COUNTED, and a config that carries Hy-SimulMT's layout deltas
// (flat `rope_theta`, no `mlp_layer_types`, `moe_intermediate_size` equal to
// the real expert width). The frozen passport is relaxed with
// LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY.
#include "test_hy_moe_fixture.h"

#include <absl/strings/str_cat.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace hy_simulmt_test {

using Cfg = hy_test::Cfg;

inline const Cfg &TestCfg() { return hy_test::TestCfg(); }

// Hy-SimulMT's config.json: speech architecture tag, flat rope_theta, no
// `mlp_layer_types` (dispatch comes from `first_k_dense_replace`), and the
// real (non-vestigial) `moe_intermediate_size`.
inline void WriteConfig(const std::filesystem::path &dir)
{
    const Cfg &c = TestCfg();
    std::ofstream f(dir / "config.json");
    // The published generation_config declares TWO stop ids; the text
    // engine must stop on both.
    {
        std::ofstream g(dir / "generation_config.json");
        g << R"({"bos_token_id":3,"eos_token_id":[4,63],"max_new_tokens":8})"
          << "\n";
    }
    f << R"({"architectures":["HYV3SpeechForCausalLM"],"model_type":"hy_v3",)"
      << R"("attn_impl":"eager","bos_token_id":3,"eos_token_id":4,)"
      << R"("pad_token_id":5,"eod_token_id":6,"sep_token_id":7,)"
      << R"("audio_token_id":8,"turn_end_token_id":9,)"
      << "\"hidden_size\":" << c.hidden
      << ",\"intermediate_size\":" << c.dense_inter
      << ",\"num_hidden_layers\":" << c.layers
      << ",\"num_attention_heads\":" << c.heads
      << ",\"num_key_value_heads\":" << c.kv_heads
      << ",\"head_dim\":" << c.head_dim
      << ",\"vocab_size\":" << c.vocab
      << ",\"rms_norm_eps\":1e-05"
      << ",\"rope_theta\":11158840.0"
      << ",\"moe_intermediate_size\":" << c.moe_inter
      << ",\"expert_hidden_dim\":" << c.moe_inter
      << ",\"num_experts\":" << c.experts
      << ",\"num_experts_per_tok\":" << c.top_k
      << ",\"num_shared_experts\":1"
      << ",\"first_k_dense_replace\":" << c.first_moe_layer
      << ",\"router_scaling_factor\":" << c.router_scaling_factor
      << ",\"tie_word_embeddings\":false"
      << ",\"audio_config\":{\"d_model\":8,\"encoder_layers\":1}"
      << "}\n";
    f.close();
}

// The speech-encoder ballast: unrelated names/shapes that must never reach the
// converted graph, only the `skipped_visual_tensors` counter.
inline std::vector<synthtest::FakeTensor> AudioTensors()
{
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
                const int64_t k = (i * 3 + seed * 5) % 19;
                return static_cast<float>(k - 9) * 0.003f;
            });
        return f;
    };
    return {
        bf16("audio_tower.speech_encoder.conv2d1.weight", {8, 1, 3, 3}, 1),
        bf16("audio_tower.speech_encoder.conv2d1.bias", {8}, 2),
        bf16("audio_tower.speech_encoder.layers.0.fc1.weight", {16, 8}, 3),
        bf16("audio_tower.speech_adapter.layers.0.weight", {8, 8}, 4),
        bf16("audio_tower.speech_adapter.layers.0.bias", {8}, 5),
    };
}

inline constexpr int kAudioTensorCount = 5;

// Text shards + config from the Hy-Memory fixture, then the SimulMT config and
// a separate audio shard (the published tree keeps the tower in its own file).
inline int BuildSyntheticCheckpoint(
    const std::filesystem::path &dir, bool with_tokenizer = false,
    const std::filesystem::path &tok_fixture = {})
{
    const int n =
        hy_test::BuildSyntheticCheckpoint(dir, with_tokenizer, tok_fixture);
    WriteConfig(dir);
    synthtest::WriteSafetensors(dir, "model-audio-tower.safetensors",
                                AudioTensors());
    return n + kAudioTensorCount;
}

} // namespace hy_simulmt_test
