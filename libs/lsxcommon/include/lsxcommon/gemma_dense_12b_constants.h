#pragma once

// Gemma-4-12B-it frozen geometry passport (Model Ingestion Protocol step 3).
//
// The SMALL dense member of the Gemma-4 family: `Gemma4UnifiedForConditional-
// Generation` (model_type `gemma4_unified`), whose TEXT tower is the sibling of
// gemma-4-31B-it's — 5:1 sliding/global attention, per-head QK-norm with K tied
// to V on the global layers, four RMSNorms + a per-layer scalar per layer,
// GeGLU, tied embed/lm_head, softcapped logits — with a different width, depth
// and head layout. Its "unified" wrapper adds a patch-embedding tower and one
// audio projection; both are stripped at conversion and counted.
//
// ONE converter, ONE loader and ONE kernel set serve both members (the
// `granite-switch-4.1` shape). What keeps them apart is this passport: each
// class validates the converted tree against `frozen::kFrozen`, so the 12B
// cannot decode a 31B tree (or the reverse) on a shared geometry constant.
//
// This header is a LEAF: it includes only the family's shared TYPES header
// (`GemmaDenseTopology` / `GemmaDenseFrozen` / the geometry gate), never another
// model's passport, and it defines no value that belongs to a sibling. It is
// device-safe (no abseil, no arrow) so the .cu layer can include it.

#include <cstddef>
#include <cstdint>

#include "lsxcommon/gemma_dense_constants.h"

namespace lsxcommon {
namespace gemma_dense_12b {

namespace frozen {

// ── Frozen compile-time geometry (config.json text_config, 2026-11-28) ──────
inline constexpr int kHiddenSize = 3840;
inline constexpr int kNumLayers = 48;
inline constexpr int kNumHeads = 16;
inline constexpr int kNumKVHeads = 8;          // sliding layers
inline constexpr int kNumGlobalKVHeads = 1;    // full-attention layers (K is V)
inline constexpr int kHeadDim = 256;           // sliding layers
inline constexpr int kGlobalHeadDim = 512;     // full-attention layers
inline constexpr int kIntermediateSize = 15360;
inline constexpr int kVocabSize = 262144;
inline constexpr int kSlidingWindow = 1024;

// The 5:1 sliding/full pattern: layer l is global iff (l + 1) % 6 == 0
// (5, 11, …, 47 — verified against config.json `layer_types`).
inline constexpr int kFullLayerPeriod = 6;
inline constexpr int kNumFullLayers = kNumLayers / kFullLayerPeriod;  // 8
inline constexpr int kNumSlidingLayers = kNumLayers - kNumFullLayers;

inline constexpr double kRmsNormEps = 1e-6;
inline constexpr double kRopeThetaSliding = 10000.0;
inline constexpr double kRopeThetaFull = 1000000.0;

// "proportional" RoPE of the global layers (rope_type = "proportional",
// partial_rotary_factor = 0.25 over the whole 512-wide head): the first
// kFullRotaryAngles rotate_half pairs carry a frequency, the rest are identity.
inline constexpr double kPartialRotaryFactor = 0.25;
inline constexpr int kFullRotaryAngles =
    static_cast<int>(kPartialRotaryFactor * kGlobalHeadDim / 2);  // 64

// Gemma-4 attention has NO score scale: q and k are per-head RMS-normalized
// before the product (Gemma4TextAttention.scaling = 1.0).
inline constexpr float kAttentionScale = 1.0f;

// Gemma4TextScaledWordEmbedding multiplies the embedding by sqrt(hidden_size)
// cast to bf16 first: sqrt(3840) = 61.9677… -> 62.0.
inline constexpr float kEmbedScale = 62.0f;

// logits = tanh(l / cap) * cap (cap 30).
inline constexpr float kFinalLogitSoftcap = 30.0f;

// Special tokens: bos/pad/eos live in text_config; the top-level eos array is
// [1, 106] and generation_config.json adds 50 (<end_of_turn>).
inline constexpr int32_t kBosId = 2;
inline constexpr int32_t kPadId = 0;
inline constexpr int32_t kEosIdPrimary = 1;
inline constexpr int32_t kEosIdTurn = 106;
inline constexpr int32_t kEosIdEndTurn = 50;

// Geometry invariants.
static_assert(kHiddenSize > 0 && kIntermediateSize > 0 && kVocabSize > 0,
              "positive Gemma-4 dense geometry required");
static_assert(kNumHeads % kNumKVHeads == 0 && kNumKVHeads > 0,
              "num_attention_heads must be a multiple of num_key_value_heads");
static_assert(kNumHeads % kNumGlobalKVHeads == 0 && kNumGlobalKVHeads > 0,
              "num_attention_heads must be a multiple of the global kv heads");
static_assert(kHeadDim * kNumKVHeads != kGlobalHeadDim * kNumGlobalKVHeads,
              "sliding and global KV row counts must differ (they gate the "
              "per-layer cache geometry)");
static_assert(kNumFullLayers * kFullLayerPeriod == kNumLayers,
              "the 5:1 pattern must divide the depth exactly");
static_assert(kFullRotaryAngles ==
                  static_cast<int>(kPartialRotaryFactor * kGlobalHeadDim / 2),
              "kFullRotaryAngles must be partial_rotary_factor*head_dim/2");
static_assert(kFullRotaryAngles * 2 <= kGlobalHeadDim / 2,
              "the rotated pairs must fit inside the first rotate_half half");

// ── This checkpoint's frozen expectations, as the shared runtime sees them ──
inline constexpr gemma_dense::GemmaDenseFrozen kFrozen = {
    "gemma-4-12b-it",
    kNumLayers,
    kHiddenSize,
    kIntermediateSize,
    kVocabSize,
    kNumHeads,
    kNumKVHeads,
    kNumGlobalKVHeads,
    kHeadDim,
    kGlobalHeadDim,
    kSlidingWindow,
    kRmsNormEps,
    kRopeThetaSliding,
    kRopeThetaFull,
    kPartialRotaryFactor,
    static_cast<double>(kFinalLogitSoftcap),
    kAttentionScale,
    kFullLayerPeriod,
};

} // namespace frozen
} // namespace gemma_dense_12b
} // namespace lsxcommon
