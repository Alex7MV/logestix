#pragma once

// Gemma-4-31B-it frozen geometry passport (Model Ingestion Protocol step 3).
//
// The DENSE member of the Gemma-4 family: `Gemma4ForConditionalGeneration`
// with `text_config.enable_moe_block = false`, i.e. the same decoder as
// gemma-4-26B-A4B-it (5:1 sliding/full-attention pattern, per-head QK-norm with
// K tied to V on the global layers, four norms per layer, GeGLU, tied
// embed/lm_head, softcapped logits) but with NO routed experts, twice the depth
// and ~2x the width. A different geometry is a NEW family in this repo: it owns
// this passport, its own converter/loader/kernels and its own namespace, and it
// borrows no constant from the 26B MoE model.
//
// This header is a LEAF: it includes no other model's passport
// (model-independence invariant). It is device-safe (no abseil, no arrow) so the
// .cu inference layer can include it.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/vram_estimate.h"

namespace lsxcommon {
namespace gemma_dense {

// Loaded/converted topology of a Gemma-4 dense checkpoint. Device-safe so the
// .cu layer can include it; populated by the converter from config.json and
// re-read from <root>/logestix/model.index.json by the loader.
struct GemmaDenseTopology {
    int num_layers = 60;
    // "sliding_attention" | "full_attention", one per layer.
    std::vector<std::string> layer_types;
    int hidden_size = 5376;
    int intermediate_size = 21504;
    int vocab_size = 262144;
    int num_attention_heads = 32;
    int num_key_value_heads = 16;          // sliding layers
    int num_global_key_value_heads = 4;    // full-attention layers (K is also V)
    int head_dim = 256;                    // sliding layers
    int global_head_dim = 512;             // full-attention layers
    int sliding_window = 1024;
    float rms_norm_eps = 1e-6f;
    float rope_theta_sliding = 10000.0f;
    float rope_theta_full = 1000000.0f;
    float partial_rotary_factor = 0.25f;   // full-attention layers only
    float final_logit_softcapping = 30.0f;
    int bos_id = 2;
    std::vector<int> eos_ids;
    int pad_id = 0;
    int total_pinned_tensors = 0;
    VramProfile vram;         // conversion-time VRAM requirements (mandatory)
    std::string error_msg;    // non-empty when index.json is invalid

    // Per-layer attention geometry. Full-attention ("global") layers use the
    // global head dim and the global KV-head count; the sliding ones use the
    // plain pair. window <= 0 means "attend to the whole causal context".
    bool IsFull(int layer) const
    {
        return layer >= 0 && static_cast<size_t>(layer) < layer_types.size() &&
               layer_types[static_cast<size_t>(layer)] == "full_attention";
    }
    int HeadDim(int layer) const
    {
        return IsFull(layer) ? global_head_dim : head_dim;
    }
    int KvHeads(int layer) const
    {
        return IsFull(layer) ? num_global_key_value_heads : num_key_value_heads;
    }
    int QRows(int layer) const
    {
        return num_attention_heads * HeadDim(layer);
    }
    int KvRows(int layer) const
    {
        return KvHeads(layer) * HeadDim(layer);
    }
    double RopeTheta(int layer) const
    {
        return IsFull(layer) ? static_cast<double>(rope_theta_full)
                             : static_cast<double>(rope_theta_sliding);
    }
    int Window(int layer) const { return IsFull(layer) ? 0 : sliding_window; }
};

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
// rounds to 73.5 and the gather multiplies by THAT, not by the exact root.
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

} // namespace frozen
} // namespace gemma_dense
} // namespace lsxcommon
