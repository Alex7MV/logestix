#pragma once

// Frozen GigaChat4-37B-A4B geometry (compile-time passport).
//
// Source of truth (read-only references, not vendored):
//   /mnt/fast_storage/lsx_model/ai-sage/GigaChat4-37B-A4B/config.json
//   .../model.index.json (36 700 tensors, 3 shards, 39 GB FP8/BF16)
//   .../modeling_gigachat4.py, .../configuration_gigachat4.py
//
// Architecture: 32 decoder layers — 1 dense FFN (layer 0) + 31 MoE — of which
// 8 are gated MLA full-attention layers carrying a DSA lightning indexer
// (3, 7, …, 31) and 24 are GatedDeltaNet linear-attention layers. Every norm is
// a GatedNorm (plain RMSNorm scale `w`, NOT the zero-centred `1+w` of
// GigaChat3.5), the GDN output norm is `gated_rmsnorm_sigmoid`, and the logits
// pass a tanh soft cap. One MTP layer (32) is stored and never applied.
// This header is a leaf: it must never include another model's passport.

#include <cstdint>

namespace lsxcommon::gigachat4 {
namespace frozen {

inline constexpr const char *kType = "gigachat4-37b-a4b";
inline constexpr const char *kFormat = "logestix-gigachat4-nvfp4";
inline constexpr const char *kModelName = "GigaChat4-37B-A4B";

inline constexpr int kHidden = 2560;
inline constexpr int kLayers = 32;
inline constexpr int kVocab = 128256;
inline constexpr int kFirstKDense = 1;
inline constexpr float kRmsEps = 1e-6f;
inline constexpr float kNormGatingWeight = 2.0f;
inline constexpr int kGateRank = 16;
// config.final_logit_softcapping — applied as cap * tanh(x / cap) in F32.
inline constexpr float kLogitSoftcap = 30.0f;

// Layer map. The checkpoint lists the full-attention layers explicitly;
// everything else is GatedDeltaNet.
inline constexpr int kFullAttnLayers[8] = {3, 7, 11, 15, 19, 23, 27, 31};
inline constexpr int kNumFullAttn = 8;
// Single MTP prediction layer (config.num_nextn_predict_layers = 1).
inline constexpr int kMtpLayer = 32;

// ── MLA (full-attention layers) ─────────────────────────────────────────────
inline constexpr int kHeads = 32;
inline constexpr int kKVHeads = 32;
inline constexpr int kQkNopeHeadDim = 128;
inline constexpr int kQkRopeHeadDim = 64;
inline constexpr int kQkHeadDim = kQkNopeHeadDim + kQkRopeHeadDim;  // 192
inline constexpr int kVHeadDim = 128;
inline constexpr int kQLoraRank = 1536;
inline constexpr int kKvLoraRank = 512;
inline constexpr float kRopeTheta = 100000.0f;
// YaRN: factor 32 over the 32768-token pretrained window, beta 32/1.
inline constexpr double kYarnFactor = 32.0;
inline constexpr double kYarnOrig = 32768.0;
inline constexpr double kYarnBetaFast = 32.0;
inline constexpr double kYarnBetaSlow = 1.0;
inline constexpr double kYarnMscale = 1.0;
inline constexpr double kYarnMscaleAllDim = 1.0;
// `attention_factor` (the cos/sin multiplier) is
// get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim) = 1.0 for
// mscale == mscale_all_dim, so the rope tables are unscaled. The attention
// scale carries the YaRN mscale squared:
//   (0.1*ln(32) + 1)^2 / sqrt(192) = 1.813259…/ 13.856406… = 0.130860799…
inline constexpr float kAttnScale = 0.13086079996295008f;
// alpha_q = sqrt(hidden / q_lora), alpha_kv = sqrt(hidden / kv_lora): the
// latents are scaled BEFORE q_b_proj / kv_b_proj, and the alpha_q-scaled q
// latent also feeds the indexer's wq_b.
inline constexpr float kAlphaQ = 1.2909944487358056f;   // sqrt(2560/1536)
inline constexpr float kAlphaKv = 2.2360679774997896f;  // sqrt(2560/512)

// ── GatedDeltaNet (linear-attention layers) ─────────────────────────────────
inline constexpr int kLinearNumKeyHeads = 16;
inline constexpr int kLinearNumValueHeads = 32;
inline constexpr int kLinearKvRepeat = kLinearNumValueHeads / kLinearNumKeyHeads;  // 2
inline constexpr int kLinearKeyHeadDim = 128;
inline constexpr int kLinearValueHeadDim = 128;
inline constexpr int kLinearConvKernel = 4;
inline constexpr int kLinearKeyDim = kLinearNumKeyHeads * kLinearKeyHeadDim;       // 2048
inline constexpr int kLinearValueDim = kLinearNumValueHeads * kLinearValueHeadDim; // 4096
inline constexpr int kLinearConvDim = 2 * kLinearKeyDim + kLinearValueDim;         // 8192
inline constexpr int kLinearQkvzDim = 2 * kLinearKeyDim + 2 * kLinearValueDim;     // 12288
inline constexpr int kLinearBaDim = 2 * kLinearNumValueHeads;                      // 64
inline constexpr float kLinearSigmoidGateScale = 2.0f;
inline constexpr float kLinearONormEps = 1e-6f;

// ── MoE / MLP ───────────────────────────────────────────────────────────────
inline constexpr int kNumRouted = 192;
inline constexpr int kTopK = 8;
inline constexpr int kMoeIntermediate = 768;
inline constexpr int kSharedExperts = 1;
inline constexpr int kSharedIntermediate = kMoeIntermediate * kSharedExperts;
inline constexpr int kDenseIntermediate = 8192;
inline constexpr float kRoutedScale = 2.5f;
inline constexpr float kSwigluLimit = 10.0f;

// ── DSA lightning indexer (the 8 full-attention layers) ─────────────────────
inline constexpr int kIndexHeads = 64;
inline constexpr int kIndexHeadDim = 128;
inline constexpr int kIndexTopk = 2048;
inline constexpr int kIndexRope = 64;  // == kQkRopeHeadDim, non-interleaved here
// weights_proj is pre-scaled by n_heads^-0.5; the score dot by head_dim^-0.5.
inline constexpr float kIndexWeightScale = 0.125f;                // 64^-0.5
inline constexpr float kIndexScoreScale = 0.08838834764831845f;   // 128^-0.5

// ── Tokens ──────────────────────────────────────────────────────────────────
inline constexpr int kBosId = 1;
inline constexpr int kEosId = 2;
inline constexpr int kPadId = 2;
inline constexpr int kRoleSepId = 128000;
inline constexpr int kMessageSepId = 128001;
inline constexpr int kThinkId = 128014;
inline constexpr int kThinkEndId = 128015;

// Frozen-geometry invariants (compile-time).
static_assert(kQkHeadDim == 192, "MLA qk head dim");
static_assert(kQkHeadDim == kQkNopeHeadDim + kQkRopeHeadDim, "MLA qk split");
static_assert(kLinearKvRepeat * kLinearNumKeyHeads == kLinearNumValueHeads,
              "GDN head repeat");
static_assert(kLinearQkvzDim == 12288, "fused qkvz width");
static_assert(kLinearConvDim == 8192, "GDN conv width");
static_assert(kLinearBaDim == 64, "GDN beta/alpha width");
static_assert(kSharedIntermediate == 768, "shared expert width");
static_assert(kNumFullAttn == 8, "full-attention layer count");
static_assert(kFullAttnLayers[7] == 31 && kFullAttnLayers[0] == 3,
              "full-attention layer map");
static_assert(kTopK <= kNumRouted, "top-k exceeds expert count");
static_assert(kHidden % 128 == 0, "hidden must tile by 128");
static_assert(kIndexHeads * kIndexHeadDim == 8192, "indexer q width");
static_assert(kIndexRope <= kIndexHeadDim, "indexer rope fits the head");
static_assert(kQLoraRank == 1536 && kKvLoraRank == 512, "MLA lora ranks");

} // namespace frozen
} // namespace lsxcommon::gigachat4
