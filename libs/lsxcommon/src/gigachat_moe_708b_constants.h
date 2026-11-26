#pragma once

// Frozen GigaChat4-708B-A41B geometry (compile-time passport).
//
// Source of truth (read-only references, not vendored):
//   /mnt/fast_storage/huggingface/GigaChat4-708B-A41B/config.json
//   .../model.safetensors.index.json (95 616 tensors, 45 shards, 720 GB FP8/BF16)
//   .../modeling_gigachat4.py, .../configuration_gigachat4.py
//
// Architecture: the SAME code as the 37B sibling (modeling_gigachat4.py,
// configuration_gigachat4.py, tokenizer.json and chat_template.jinja are
// byte-identical), at 19x the parameters. 64 decoder layers — 3 dense FFN
// (0..2) + 61 MoE — of which 16 are gated MLA full-attention layers carrying a
// DSA lightning indexer (3, 7, …, 63) and 48 are GatedDeltaNet linear-attention
// layers. Every norm is a GatedNorm (plain RMSNorm scale `w`, NOT the
// zero-centred `1+w` of GigaChat3.5) whose rank-16 gate is computed from the
// state ALREADY scaled by the norm weight, the GDN output norm is
// `gated_rmsnorm_sigmoid`, and the logits pass a tanh soft cap. One MTP layer
// (64) is stored and never applied.
// This header is a leaf: it must never include another model's passport.

#include <cstdint>

namespace lsxcommon::gigachat4_708b {
namespace frozen {

inline constexpr const char *kType = "gigachat4-708b-a41b";
inline constexpr const char *kFormat = "logestix-gigachat4-708b-nvfp4";
inline constexpr const char *kModelName = "GigaChat4-708B-A41B";

inline constexpr int kHidden = 7168;
inline constexpr int kLayers = 64;
inline constexpr int kVocab = 128256;
inline constexpr int kFirstKDense = 3;
inline constexpr float kRmsEps = 1e-6f;
inline constexpr float kNormGatingWeight = 2.0f;
inline constexpr int kGateRank = 16;
// config.final_logit_softcapping — applied as cap * tanh(x / cap) in F32.
inline constexpr float kLogitSoftcap = 30.0f;

// Layer map. The checkpoint lists the full-attention layers explicitly;
// everything else is GatedDeltaNet.
inline constexpr int kFullAttnLayers[16] = {3,  7,  11, 15, 19, 23, 27, 31,
                                            35, 39, 43, 47, 51, 55, 59, 63};
inline constexpr int kNumFullAttn = 16;
// config.num_nextn_predict_layers = 1: the single MTP prediction layer.
inline constexpr int kMtpLayer = 64;

// ── MLA (full-attention layers) ─────────────────────────────────────────────
inline constexpr int kHeads = 64;
inline constexpr int kKVHeads = 64;
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
//   (0.1*ln(32) + 1)^2 / sqrt(192) = 1.813259… / 13.856406… = 0.130860799…
inline constexpr float kAttnScale = 0.13086079996295008f;
// alpha_q = sqrt(hidden / q_lora), alpha_kv = sqrt(hidden / kv_lora): the
// latents are scaled BEFORE q_b_proj / kv_b_proj, and the alpha_q-scaled q
// latent also feeds the indexer's wq_b. These are the two constants that
// differ from the 37B row (1.2909944 / 2.2360680).
inline constexpr float kAlphaQ = 2.1602468994692870f;   // sqrt(7168/1536)
inline constexpr float kAlphaKv = 3.7416573867739413f;  // sqrt(7168/512)

// ── GatedDeltaNet (linear-attention layers) ─────────────────────────────────
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

// ── MoE / MLP ───────────────────────────────────────────────────────────────
inline constexpr int kNumRouted = 256;
inline constexpr int kTopK = 8;
inline constexpr int kMoeIntermediate = 2048;
inline constexpr int kSharedExperts = 1;
inline constexpr int kSharedIntermediate = kMoeIntermediate * kSharedExperts;
inline constexpr int kDenseIntermediate = 18432;
inline constexpr float kRoutedScale = 2.5f;
inline constexpr float kSwigluLimit = 10.0f;
// MoE layers: 3 dense FFN layers then 61 routed-expert layers.
inline constexpr int kMoeLayers = kLayers - kFirstKDense;  // 61
inline constexpr int kRoutedExperts = kMoeLayers * kNumRouted;  // 15 616

// ── DSA lightning indexer (the 16 full-attention layers) ────────────────────
inline constexpr int kIndexHeads = 64;
inline constexpr int kIndexHeadDim = 128;
inline constexpr int kIndexTopk = 2048;
inline constexpr int kIndexRope = 64;  // == kQkRopeHeadDim, non-interleaved here
// weights_proj is pre-scaled by n_heads^-0.5; the score dot by head_dim^-0.5.
inline constexpr float kIndexWeightScale = 0.125f;                // 64^-0.5
inline constexpr float kIndexScoreScale = 0.08838834764831845f;   // 128^-0.5

// ── Tokens (identical to the 37B row: same tokenizer) ───────────────────────
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
static_assert(kLinearQkvzDim == 24576, "fused qkvz width");
static_assert(kLinearConvDim == 16384, "GDN conv width");
static_assert(kLinearBaDim == 128, "GDN beta/alpha width");
static_assert(kSharedIntermediate == 2048, "shared expert width");
static_assert(kNumFullAttn == 16, "full-attention layer count");
static_assert(kFullAttnLayers[0] == 3 && kFullAttnLayers[15] == 63,
              "full-attention layer map");
static_assert(kLayers == 64, "decoder depth");
static_assert(kMoeLayers == 61, "MoE layer count");
static_assert(kRoutedExperts == 15616, "routed expert count");
static_assert(kMtpLayer == kLayers, "the MTP layer follows the decoder stack");
static_assert(kTopK <= kNumRouted, "top-k exceeds expert count");
static_assert(kHidden % 128 == 0, "hidden must tile by 128");
static_assert(kMoeIntermediate % 128 == 0, "moe width must tile by 128");
static_assert(kIndexHeads * kIndexHeadDim == 8192, "indexer q width");
static_assert(kIndexRope <= kIndexHeadDim, "indexer rope fits the head");
static_assert(kQLoraRank == 1536 && kKvLoraRank == 512, "MLA lora ranks");
static_assert(kHeads == kKVHeads, "MLA shares q/kv head count");

} // namespace frozen
} // namespace lsxcommon::gigachat4_708b
