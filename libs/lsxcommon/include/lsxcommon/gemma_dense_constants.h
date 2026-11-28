#pragma once

// Gemma-4 DENSE family — shared TYPES, not a passport (Model Ingestion
// Protocol step 3).
//
// The Gemma-4 dense decoder is ingested at several sizes (gemma-4-31B-it and
// its 12B sibling); ONE converter, ONE loader, ONE kernel set and ONE lifecycle
// body serve them all. This header holds what they SHARE: the loaded topology,
// the frozen-expectations struct, the geometry gate and the embedding-scale
// derivation. It deliberately holds NO checkpoint's numbers — those live in
// each model's own passport:
//
//   gemma_dense_31b_constants.h  -> lsxcommon::gemma_dense::frozen
//   gemma_dense_12b_constants.h  -> lsxcommon::gemma_dense_12b::frozen
//
// That split is the model-independence invariant: a shared copy of the geometry
// is how one sibling ends up decoding a plausible-but-wrong model.
//
// This header is a LEAF (vram_estimate.h only) and is device-safe (no abseil,
// no arrow), so the .cu inference layer can include it.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/vram_estimate.h"

namespace lsxcommon {
namespace gemma_dense {

// Loaded/converted topology of a Gemma-4 dense checkpoint. Device-safe so the
// .cu layer can include it; populated by the converter from config.json and
// re-read from <root>/model.index.json by the loader.
//
// The shape fields default to ZERO, never to a size: several checkpoints share
// this struct, so a default would be one sibling's geometry, and a config (or
// index) that omits a field would then convert to a tree that spreads out at the
// wrong width. `ValidateGemmaDenseTopology` refuses the parse instead, which is
// the only honest answer when we cannot tell what the checkpoint is.
struct GemmaDenseTopology {
    int num_layers = 0;
    // "sliding_attention" | "full_attention", one per layer.
    std::vector<std::string> layer_types;
    int hidden_size = 0;
    int intermediate_size = 0;
    int vocab_size = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;          // sliding layers
    int num_global_key_value_heads = 0;   // full-attention layers (K is also V)
    int head_dim = 0;                     // sliding layers
    int global_head_dim = 0;              // full-attention layers
    int sliding_window = 0;
    float rms_norm_eps = 1e-6f;
    float rope_theta_sliding = 10000.0f;
    float rope_theta_full = 1000000.0f;
    float partial_rotary_factor = 0.25f;   // full-attention layers only
    float final_logit_softcapping = 30.0f;
    int bos_id = 2;
    std::vector<int> eos_ids;
    int pad_id = 0;
    int total_pinned_tensors = 0;
    // Multimodal tensors the converter recognised and stripped (counted, never
    // applied). Written to model.index.json so the stripped mass is auditable
    // after the checkpoint is gone.
    int skipped_visual_tensors = 0;
    // The checkpoint this tree was converted from (the input directory's leaf).
    // NOT a hardcoded model card: an artifact that still says "gemma-4-31B-it"
    // cannot be told apart from the 31B tree by a reader.
    std::string model_name;
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

// ── The frozen expectations, as DATA ───────────────────────────────────────
//
// One converter, one loader, one kernel set and one lifecycle body serve every
// dense Gemma-4 member (the `granite-switch-4.1` shape). What keeps two
// checkpoints apart is the passport each class validates the converted tree
// against, so the numbers travel as this struct instead of being read from a
// namespace: `granite_dense_constants.h` established the precedent after a
// shared copy of the numbers made a sibling decode a plausible-but-wrong model.
//
// Each passport header defines its own `frozen::kFrozen` instance; nothing here
// holds a value that belongs to one checkpoint.
struct GemmaDenseFrozen {
    const char *model_type = "";
    int num_layers = 0;
    int hidden_size = 0;
    int intermediate_size = 0;
    int vocab_size = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;         // sliding layers
    int num_global_key_value_heads = 0;  // full-attention layers (K is also V)
    int head_dim = 0;                    // sliding layers
    int global_head_dim = 0;             // full-attention layers
    int sliding_window = 0;
    double rms_norm_eps = 0.0;
    double rope_theta_sliding = 0.0;
    double rope_theta_full = 0.0;
    double partial_rotary_factor = 0.0;
    double final_logit_softcapping = 0.0;
    // Gemma-4 has NO attention score scale (q/k are per-head RMS-normalised),
    // so this is 1.0 for every member; it is data anyway, because a family that
    // silently applies 1/sqrt(head_dim) decodes garbage and the passport is
    // where "what this checkpoint says" belongs.
    float attention_scale = 1.0f;
    // 5:1 pattern: layer l is full_attention iff (l + 1) % period == 0.
    int full_layer_period = 0;
};

// Non-empty when a REQUIRED geometry field is absent/zero. Called by both
// parsers (the checkpoint's config.json and the converted model.index.json):
// with several checkpoints behind one struct there is no safe default, so a
// missing field must fail the parse instead of being filled from a sibling.
// `sliding_window`/`max_position_embeddings` are deliberately NOT required —
// a full-attention-only tree has no window, and the context length is not a
// decode parameter.
std::string ValidateGemmaDenseTopology(const GemmaDenseTopology &top);

// Empty when `top` is the tree `f` describes; otherwise every difference on its
// own line (a re-converted tree that is wrong twice must not need two runs to
// discover). Declared here — the family's shared types header — rather than in
// a model TU so a host-only unit test can exercise it with no GPU and no
// checkpoint.
std::string GemmaDenseGeometryMismatch(const GemmaDenseFrozen &f,
                                       const GemmaDenseTopology &top);

// The embedding multiplier upstream actually applies: sqrt(hidden_size) cast to
// the weight dtype (bf16) FIRST, so 3840 -> 62.0 and 5376 -> 73.5, not the
// exact roots. Derived from the loaded topology, never from a sibling's
// constant: using the 31B's 73.5 on a 12B scales every embedding by 1.19.
float GemmaDenseEmbedScale(int hidden_size);

} // namespace gemma_dense
} // namespace lsxcommon
