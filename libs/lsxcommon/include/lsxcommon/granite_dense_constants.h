#pragma once

// Granite-4.2-30B frozen geometry passport (Model Ingestion Protocol step 3).
//
// IBM Granite 4.2 30B is a dense decoder-only transformer (GraniteForCausalLM):
// RMSNorm, GQA attention with full RoPE, SwiGLU MLP. No MoE, no QK-norm, no
// biases, no linear attention. This header is a LEAF: it must not include any
// other model's passport (model-independence invariant). It is device-safe
// (no abseil containers/streams) so the .cu inference layer can include it.

#include <cstddef>
#include <cstdint>
#include <string>

#include "lsxcommon/vram_estimate.h"

namespace lsxcommon {
namespace granite {

// Loaded/converted topology of a Granite checkpoint (geometry + special
// tokens + the mandatory conversion-time VRAM profile). Device-safe (no
// arrow/absl) so the .cu inference layer can include this header.
struct GraniteTopology {
    int num_layers = 0;
    int hidden_size = 0;
    int intermediate_size = 0;
    int vocab_size = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int head_dim = 0;
    double rms_norm_eps = 1e-5;
    double rope_theta = 50000000.0;
    double attention_multiplier = 0.0078125;
    int bos_id = -1;
    int eos_id = -1;
    int pad_id = -1;
    int total_pinned_tensors = 0;
    VramProfile vram;        // conversion-time VRAM requirements (mandatory)
    std::string error_msg;   // non-empty when index.json is invalid
};

namespace frozen {

// ── Frozen compile-time geometry (config.json, 2026-09-22) ─────────────────
inline constexpr int kHiddenSize = 4096;
inline constexpr int kNumLayers = 64;
inline constexpr int kNumHeads = 32;
inline constexpr int kNumKVHeads = 8;
inline constexpr int kHeadDim = 128;
inline constexpr int kIntermediateSize = 32768;
inline constexpr int kVocabSize = 100352;

inline constexpr double kRmsNormEps = 1e-5;
inline constexpr double kRopeTheta = 50000000.0;

// Granite fixed attention score scale (config attention_multiplier). Applied
// to the attention logits INSTEAD of 1/sqrt(head_dim).
inline constexpr float kAttentionMultiplier = 0.0078125f;
// All 1.0 in this checkpoint; kept explicit so the runtime stays faithful.
inline constexpr float kEmbeddingMultiplier = 1.0f;
inline constexpr float kResidualMultiplier = 1.0f;
inline constexpr float kLogitsScaling = 1.0f;

// Special tokens (read from config.json at conversion time; these are the
// reference values the passport pins).
inline constexpr int32_t kBosId = 100283;
inline constexpr int32_t kEosId = 100257;
inline constexpr int32_t kPadId = 100257;

// Compile-time geometry invariants.
static_assert(kHiddenSize == kNumHeads * kHeadDim,
              "hidden_size must equal num_attention_heads * head_dim");
static_assert(kNumKVHeads > 0 && kNumKVHeads <= kNumHeads,
              "num_key_value_heads must be in (0, num_attention_heads]");
static_assert(kHiddenSize % kNumKVHeads == 0,
              "hidden_size must be divisible by num_key_value_heads");
static_assert(kNumLayers > 0 && kIntermediateSize > 0 && kVocabSize > 0,
              "positive Granite geometry required");

} // namespace frozen
} // namespace granite
} // namespace lsxcommon
