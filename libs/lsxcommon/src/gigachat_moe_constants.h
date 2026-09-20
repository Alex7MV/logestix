#pragma once

// Frozen GigaChat3.5-432B-A28B-Reasoning geometry (compile-time passport).
// Source of truth: /mnt/llm_models/models/GigaChat3.5-432B-A28B-Reasoning/
// config.json + safetensors inventory audit (2026-09-20), and the reference
// modeling_gigachat3_5.py (read-only reference, not vendored). 40 layers + 3
// MTP; MLA full-attention at {3,7,...,39}; GatedDeltaNet elsewhere; MoE 3..39;
// ZeroCenteredGatedNorm on every norm.

#include <cstdint>

namespace lsxcommon::gigachat {
namespace frozen {

inline constexpr const char *kType = "gigachat3.5-432b-a28b-reasoning";

inline constexpr int kHidden = 7168;
inline constexpr int kLayers = 40;
inline constexpr int kVocab = 128256;
inline constexpr int kFirstKDense = 3;
inline constexpr float kRmsEps = 1e-6f;
inline constexpr float kNormGatingWeight = 2.0f;
inline constexpr int kGateRank = 16;

inline constexpr int kFullAttnLayers[10] = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39};
inline constexpr int kNumFullAttn = 10;
inline constexpr int kMtpLayers[3] = {40, 41, 42};

// MLA.
inline constexpr int kHeads = 64;
inline constexpr int kKVHeads = 64;
inline constexpr int kQkNopeHeadDim = 128;
inline constexpr int kQkRopeHeadDim = 64;
inline constexpr int kQkHeadDim = kQkNopeHeadDim + kQkRopeHeadDim;  // 192
inline constexpr int kVHeadDim = 128;
inline constexpr int kQLoraRank = 1536;
inline constexpr int kKvLoraRank = 512;
inline constexpr float kRopeTheta = 100000.0f;
inline constexpr double kYarnFactor = 8.0;
inline constexpr double kYarnOrig = 32768.0;
inline constexpr double kYarnBetaFast = 32.0;
inline constexpr double kYarnBetaSlow = 1.0;
inline constexpr double kYarnMscale = 1.0;
inline constexpr double kYarnMscaleAllDim = 1.0;

// GatedDeltaNet.
inline constexpr int kLinearNumKeyHeads = 32;
inline constexpr int kLinearNumValueHeads = 64;
inline constexpr int kLinearKvRepeat = kLinearNumValueHeads / kLinearNumKeyHeads;  // 2
inline constexpr int kLinearKeyHeadDim = 128;
inline constexpr int kLinearValueHeadDim = 128;
inline constexpr int kLinearConvKernel = 4;
inline constexpr int kLinearKeyDim = kLinearNumKeyHeads * kLinearKeyHeadDim;       // 4096
inline constexpr int kLinearValueDim = kLinearNumValueHeads * kLinearValueHeadDim; // 8192
inline constexpr int kLinearConvDim = 2 * kLinearKeyDim + kLinearValueDim;         // 16384
inline constexpr int kLinearQkvzDim = 2 * kLinearKeyDim + 2 * kLinearValueDim;     // 24576
inline constexpr int kLinearBaDim = 2 * kLinearNumValueHeads;                      // 128
inline constexpr float kLinearSigmoidGateScale = 2.0f;
inline constexpr float kLinearONormEps = 1e-6f;

// MoE / MLP.
inline constexpr int kNumRouted = 256;
inline constexpr int kTopK = 8;
inline constexpr int kMoeIntermediate = 2048;
inline constexpr int kSharedExperts = 1;
inline constexpr int kSharedIntermediate = kMoeIntermediate * kSharedExperts;
inline constexpr int kDenseIntermediate = 18432;
inline constexpr float kRoutedScale = 2.5f;
inline constexpr float kSwigluLimit = 10.0f;

// Tokens.
inline constexpr int kBosId = 1;
inline constexpr int kEosId = 2;
inline constexpr int kPadId = 2;
inline constexpr int kRoleSepId = 128000;
inline constexpr int kMessageSepId = 128001;
inline constexpr int kThinkId = 128014;
inline constexpr int kThinkEndId = 128015;

// Frozen-geometry invariants (compile-time).
static_assert(kQkHeadDim == 192, "MLA qk head dim");
static_assert(kLinearKvRepeat * kLinearNumKeyHeads == kLinearNumValueHeads,
              "GDN head repeat");
static_assert(kLinearQkvzDim == 2 * kLinearKeyDim + 2 * kLinearValueDim,
              "fused qkvz width");
static_assert(kSharedIntermediate == 2048, "shared expert width");
static_assert(kNumFullAttn == 10, "full-attention layer count");
static_assert(kTopK <= kNumRouted, "top-k exceeds expert count");
static_assert(kHidden % 128 == 0, "hidden must tile by 128");

} // namespace frozen
} // namespace lsxcommon::gigachat
