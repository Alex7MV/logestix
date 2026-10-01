#pragma once

// phi-4 (Microsoft) frozen geometry passport (Model Ingestion Protocol
// step 3). Phi3ForCausalLM is a dense decoder-only transformer: RMSNorm, GQA
// attention with full-dim RoPE, SwiGLU MLP. No MoE, no QK-norm, no biases,
// no config-level attention/residual multipliers. This header is a LEAF: no
// other model's passport may be included (model-independence invariant). It
// is device-safe (no abseil containers/streams).

#include <cstddef>
#include <cstdint>
#include <string>

#include "lsxcommon/vram_estimate.h"

namespace lsxcommon {
namespace phi {

// Loaded/converted topology of a phi checkpoint (geometry + special tokens +
// the mandatory conversion-time VRAM profile). Device-safe.
struct PhiTopology {
    int num_layers = 0;
    int hidden_size = 0;
    int intermediate_size = 0;
    int vocab_size = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int head_dim = 0;
    double rms_norm_eps = 1e-5;
    double rope_theta = 250000.0;
    // phi applies the textbook 1/sqrt(head_dim) attention score scale (no
    // config multiplier). Pinned at conversion time.
    double attention_scale = 0.08838834764831845;
    int bos_id = -1;
    int eos_id = -1;
    int pad_id = -1;
    int total_pinned_tensors = 0;
    VramProfile vram;        // conversion-time VRAM requirements (mandatory)
    std::string error_msg;   // non-empty when index.json is invalid
};

namespace frozen {

// ── Frozen compile-time geometry (config.json, 2026-10-01) ─────────────────
inline constexpr int kHiddenSize = 5120;
inline constexpr int kNumLayers = 40;
inline constexpr int kNumHeads = 40;
inline constexpr int kNumKVHeads = 10;
inline constexpr int kHeadDim = 128;
inline constexpr int kIntermediateSize = 17920;
inline constexpr int kVocabSize = 100352;

inline constexpr double kRmsNormEps = 1e-5;
inline constexpr double kRopeTheta = 250000.0;
// 1 / sqrt(kHeadDim).
inline constexpr float kAttentionScale = 0.08838834764831845f;

// Special tokens (config.json): bos <|endoftext|> / eos <|im_end|> /
// pad <|dummy_85|>.
inline constexpr int32_t kBosId = 100257;
inline constexpr int32_t kEosId = 100265;
inline constexpr int32_t kPadId = 100349;

// Compile-time geometry invariants.
static_assert(kHiddenSize == kNumHeads * kHeadDim,
              "hidden_size must equal num_attention_heads * head_dim");
static_assert(kNumKVHeads > 0 && kNumKVHeads <= kNumHeads,
              "num_key_value_heads must be in (0, num_attention_heads]");
static_assert(kHiddenSize % kNumKVHeads == 0,
              "hidden_size must be divisible by num_key_value_heads");
static_assert(kNumLayers > 0 && kIntermediateSize > 0 && kVocabSize > 0,
              "positive phi geometry required");

} // namespace frozen
} // namespace phi
} // namespace lsxcommon
