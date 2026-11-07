#pragma once

// Granite-Switch 4.1 8B frozen geometry passport
// (Model Ingestion Protocol step 3).
//
// The small sibling of granite-switch-4.1-30b: same `GraniteSwitchForCausalLM`
// composition (dense decoder + two embedded aLoRA adapters + control-token
// switch + token exchange), same tokenizer bytes, same chat template, same
// tensor names. Four fields differ, and three of them are arithmetic rather
// than shape:
//
//   decoder layers        40      (config 42 = 40 + 2 switch cache slots)
//   intermediate_size     12800   (30B: 32768)
//   rope_theta            1e7     (30B: 5e7)
//   residual_multiplier   0.22    (30B: 0.175)
//
// The others are identical and are pinned here anyway, because a passport is
// what `ValidateGeometry` compares a converted tree against: an omitted field is
// a field the engine will accept from any checkpoint.
//
// This header is a LEAF: it must not include another model's passport
// (model-independence invariant). It is device-safe (no abseil
// containers/streams) so the .cu inference layer can include it.

#include <cstdint>

#include "lsxcommon/granite_dense_switch_constants.h"

namespace lsxcommon {
namespace granite_switch_8b {

namespace frozen {

// ── Frozen compile-time geometry (config.json, 2026-11-06) ─────────────────
inline constexpr int kHiddenSize = 4096;
inline constexpr int kNumLayers = 40;
inline constexpr int kSwitchCacheLayers = 2;
inline constexpr int kNumHeads = 32;
inline constexpr int kNumKVHeads = 8;
inline constexpr int kHeadDim = 128;
inline constexpr int kIntermediateSize = 12800;
inline constexpr int kVocabSize = 100355;

inline constexpr double kRmsNormEps = 1e-5;
inline constexpr double kRopeTheta = 10000000.0;

// Granite fixed attention score scale (config attention_multiplier), applied to
// the attention logits INSTEAD of 1/sqrt(head_dim).
inline constexpr float kAttentionMultiplier = 0.0078125f;
inline constexpr float kEmbeddingMultiplier = 12.0f;
inline constexpr float kResidualMultiplier = 0.22f;
inline constexpr float kLogitsScaling = 16.0f;

// Adapter metadata (config.json / adapter_index.json).
inline constexpr int kNumAdapters = 2;
inline constexpr int kMaxLoraRank = 64;
inline constexpr int kAdapterRanks[kNumAdapters] = {16, 64};
inline constexpr int32_t kControlTokenIds[kNumAdapters] = {100352, 100353};
inline constexpr int32_t kSubstituteTokenIds[kNumAdapters] = {27, 14711};

// Special tokens (read from config.json; these are the reference values).
inline constexpr int32_t kBosId = 100257;
inline constexpr int32_t kEosId = 100257;
inline constexpr int32_t kPadId = 100256;

// Compile-time geometry invariants.
static_assert(kHiddenSize == kNumHeads * kHeadDim,
              "hidden_size must equal num_attention_heads * head_dim");
static_assert(kNumKVHeads > 0 && kNumKVHeads <= kNumHeads,
              "num_key_value_heads must be in (0, num_attention_heads]");
static_assert(kHiddenSize % kNumKVHeads == 0,
              "hidden_size must be divisible by num_key_value_heads");
static_assert(kNumLayers > 0 && kIntermediateSize > 0 && kVocabSize > 0,
              "positive Granite-Switch geometry required");
static_assert(kNumAdapters == 2, "this passport pins exactly two adapters");
static_assert(sizeof(kAdapterRanks) / sizeof(kAdapterRanks[0]) == kNumAdapters,
              "adapter rank table must cover every adapter");
// The 8B is NOT the 30B with fewer layers: three arithmetic fields differ, so a
// sibling that reuses the parent's wrapper without its own passport is caught
// here at compile time rather than at the first wrong token.
static_assert(kIntermediateSize != 32768 && kRopeTheta != 50000000.0,
              "the 8B geometry must not alias the 30B passport");

} // namespace frozen
// The expectations a converted 8B tree must satisfy. Consumed by
// GraniteSwitchGeometryMismatch() and by the model class; the shape is shared
// with the 30B passport so the checker has ONE implementation.
inline granite_switch::GraniteSwitchFrozen Frozen()
{
    granite_switch::GraniteSwitchFrozen f{};
    f.model_type = "granite-switch-4.1-8b";
    f.num_layers = frozen::kNumLayers;
    f.hidden_size = frozen::kHiddenSize;
    f.num_attention_heads = frozen::kNumHeads;
    f.num_key_value_heads = frozen::kNumKVHeads;
    f.head_dim = frozen::kHeadDim;
    f.intermediate_size = frozen::kIntermediateSize;
    f.vocab_size = frozen::kVocabSize;
    f.num_adapters = frozen::kNumAdapters;
    f.rms_norm_eps = frozen::kRmsNormEps;
    f.rope_theta = frozen::kRopeTheta;
    f.attention_multiplier = frozen::kAttentionMultiplier;
    f.embedding_multiplier = frozen::kEmbeddingMultiplier;
    f.residual_multiplier = frozen::kResidualMultiplier;
    f.logits_scaling = frozen::kLogitsScaling;
    f.adapter_ranks = frozen::kAdapterRanks;
    f.control_token_ids = frozen::kControlTokenIds;
    f.substitute_token_ids = frozen::kSubstituteTokenIds;
    return f;
}



} // namespace granite_switch_8b
} // namespace lsxcommon
