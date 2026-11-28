#pragma once

// Frozen Qwen3.6-35B-A3B geometry (compile-time passport). Model-independent
// from Qwen3.5-35B-A3B by directive: values coincide (same qwen3_5_moe
// architecture), but the passport is owned and versioned separately.
//
// Source of truth: <source root>/Qwen/Qwen3.6-35B-A3B/config.json
// (text_config, model_type qwen3_5_moe).

namespace lsxcommon::qwen_moe_36 {
namespace frozen {

inline constexpr int kHidden = 2048;
inline constexpr int kLayers = 40;
inline constexpr int kFullAttnInterval = 4;   // full attention at i%4==3
inline constexpr int kHeads = 16;
inline constexpr int kKVHeads = 2;
inline constexpr int kHeadDim = 256;
inline constexpr int kLinearNumKeyHeads = 16;
inline constexpr int kLinearNumValueHeads = 32;
inline constexpr int kLinearKeyHeadDim = 128;
inline constexpr int kLinearValueHeadDim = 128;
inline constexpr int kLinearConvKernelDim = 4;
inline constexpr int kVocab = 248320;
inline constexpr int kNumExperts = 256;
inline constexpr int kTopK = 8;
inline constexpr int kMoeIntermediate = 512;
inline constexpr int kSharedIntermediate = 512;
inline constexpr int kBosTokenId = 248044;    // explicit in the 3.6 config
inline constexpr double kRmsNormEps = 1e-6;

// Frozen-geometry passport invariants (MODEL INGEST INVARIANT step 3).
static_assert(kTopK <= kNumExperts, "top-k exceeds expert count");
static_assert(kLayers % kFullAttnInterval == 0,
              "hybrid interval pattern incomplete");
static_assert(kMoeIntermediate > 0 && kSharedIntermediate > 0,
              "bad MoE geometry");

} // namespace frozen
} // namespace lsxcommon::qwen_moe_36
