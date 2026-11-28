#pragma once

// Gemma-4-31B-it frozen geometry passport (Model Ingestion Protocol step 3).
//
// The DENSE member of the Gemma-4 family: `Gemma4ForConditionalGeneration`
// with `text_config.enable_moe_block = false`, i.e. the same decoder as
// gemma-4-26B-A4B-it (5:1 sliding/full-attention pattern, per-head QK-norm with
// K tied to V on the global layers, four norms per layer, GeGLU, tied
// embed/lm_head, softcapped logits) but with NO routed experts, twice the depth
// and ~2x the width.
//
// The decoder BODY is shared with the 12B sibling (gemma_dense_constants.h holds
// the family's types); this header holds ONLY this checkpoint's numbers, and
// `GemmaDenseGeometryMismatch` is what makes a converted tree acceptable to one
// sibling and not the other. It includes the family's shared TYPES header and no
// other model's passport (model-independence invariant), and it is device-safe
// (no abseil, no arrow) so the .cu inference layer can include it.

#include <cstddef>
#include <cstdint>

#include "lsxcommon/gemma_dense_constants.h"

namespace lsxcommon {
namespace gemma_dense {

namespace frozen {

// ── Frozen compile-time geometry (config.json + safetensors, 2026-10-09) ───
inline constexpr int kHiddenSize = 5376;
inline constexpr int kNumLayers = 60;
inline constexpr int kNumHeads = 32;
inline constexpr int kNumKVHeads = 16;          // sliding layers
inline constexpr int kNumGlobalKVHeads = 4;     // full-attention layers
inline constexpr int kHeadDim = 256;            // sliding layers
inline constexpr int kGlobalHeadDim = 512;      // full-attention layers
inline constexpr int kIntermediateSize = 21504;
inline constexpr int kVocabSize = 262144;
inline constexpr int kSlidingWindow = 1024;

// The 5:1 sliding/full pattern: layer l is global iff (l + 1) % 6 == 0
// (5, 11, …, 59 — verified against config.json `layer_types`).
inline constexpr int kFullLayerPeriod = 6;
inline constexpr int kNumFullLayers = 10;
inline constexpr int kNumSlidingLayers = kNumLayers - kNumFullLayers;

inline constexpr double kRmsNormEps = 1e-6;
inline constexpr double kRopeThetaSliding = 10000.0;
inline constexpr double kRopeThetaFull = 1000000.0;

// "proportional" RoPE of the global layers (rope_type = "proportional",
// partial_rotary_factor = 0.25, transformers/models/gemma4 +
// modeling_rope_utils._compute_proportional_rope_parameters): the encoding
// spans the WHOLE head dim and rotate_half pairs (i, i + head_dim/2); the first
// `kFullRotaryAngles` frequency pairs are theta^(-2i / head_dim), the remaining
// pairs are zero (cos=1, sin=0, i.e. the middle of the head is left untouched).
inline constexpr double kPartialRotaryFactor = 0.25;
inline constexpr int kFullRotaryAngles =
    static_cast<int>(kPartialRotaryFactor * kGlobalHeadDim / 2);  // 64

// Gemma-4 attention has NO score scale: q and k are per-head RMS-normalized
// before the product (Gemma4TextAttention.scaling = 1.0), so 1/sqrt(head_dim)
// must NOT be applied.
inline constexpr float kAttentionScale = 1.0f;

// Gemma4TextScaledWordEmbedding multiplies the embedding by sqrt(hidden_size);
// upstream casts that constant to the weight dtype first (bf16), so 73.3212…
// rounds to 73.5 and the gather multiplies by THAT, not by the exact root. The
// runtime derives the value from the loaded topology (GemmaDenseEmbedScale);
// this constant is the passport's statement of it.
inline constexpr float kEmbedScale = 73.5f;

// Final logits soft cap: logits = tanh(l / cap) * cap (cap 30).
inline constexpr float kFinalLogitSoftcap = 30.0f;

// Special tokens (config.json / generation_config.json).
inline constexpr int32_t kBosId = 2;
inline constexpr int32_t kPadId = 0;
// config.json carries [1, 106]; generation_config.json adds 50 (<end_of_turn>).
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
// One converter, loader, kernel set and lifecycle body serve every dense
// Gemma-4 size; the numbers travel as data so a tree is only ever decoded by the
// passport it describes.
inline constexpr GemmaDenseFrozen kFrozen = {
    "gemma-4-31b-it",
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
} // namespace gemma_dense
} // namespace lsxcommon
