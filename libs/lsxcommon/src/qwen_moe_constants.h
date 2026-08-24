#pragma once

// Frozen Qwen3.5-35B-A3B geometry (compile-time passport). Shared between the
// converter (runtime cross-check against config.json / tensor shapes) and the
// model class (static_asserts + ValidateGeometry).
//
// Source of truth: /mnt/llm_models/models/Qwen3.5-35B-A3B/config.json
// (text_config, model_type qwen3_5_moe).

namespace lsxcommon::qwen_moe {
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
inline constexpr double kRmsNormEps = 1e-6;

} // namespace frozen
} // namespace lsxcommon::qwen_moe
