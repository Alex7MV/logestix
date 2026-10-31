#pragma once

// Granite-Switch 4.1 30B frozen geometry passport
// (Model Ingestion Protocol step 3).
//
// Granite-Switch composes the Granite 4.1 30B dense decoder with two embedded
// aLoRA adapters, control-token routing and a token-exchange LUT. The switch
// itself is a coded-memory attention over control tokens; its per-token
// adapter index is exactly the most recent control token's adapter (verified
// against granite_switch.hf.switch.multi on random sequences), so the runtime
// only needs a prefix scan.
//
// This header is a LEAF: it must not include another model's passport
// (model-independence invariant). It is device-safe (no abseil
// containers/streams) so the .cu inference layer can include it.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/vram_estimate.h"

namespace lsxcommon {
namespace granite_switch {

// Loaded/converted topology of a Granite-Switch checkpoint (geometry +
// special tokens + adapter metadata + the mandatory VRAM profile).
struct GraniteSwitchTopology {
    int num_layers = 0;          // decoder layers (64; config says 66 = +2 switch)
    int switch_cache_layers = 2; // counting + memory cache slots
    int hidden_size = 0;
    int intermediate_size = 0;
    int vocab_size = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int head_dim = 0;
    double rms_norm_eps = 1e-5;
    double rope_theta = 50000000.0;
    double attention_multiplier = 0.0078125;
    double embedding_multiplier = 12.0;
    double residual_multiplier = 0.175;
    double logits_scaling = 16.0;
    bool tie_word_embeddings = true;
    int bos_id = -1;
    int eos_id = -1;
    int pad_id = -1;

    int num_adapters = 0;
    std::vector<int> adapter_ranks;
    std::vector<int> adapter_token_ids;
    std::vector<int> adapter_substitute_token_ids;
    std::vector<std::string> adapter_names;

    int total_pinned_tensors = 0;
    VramProfile vram;      // conversion-time VRAM requirements (mandatory)
    std::string error_msg; // non-empty when index.json is invalid
};

namespace frozen {

// ── Frozen compile-time geometry (config.json, 2026-10-31) ─────────────────
inline constexpr int kHiddenSize = 4096;
inline constexpr int kNumLayers = 64;
inline constexpr int kSwitchCacheLayers = 2;
inline constexpr int kNumHeads = 32;
inline constexpr int kNumKVHeads = 8;
inline constexpr int kHeadDim = 128;
inline constexpr int kIntermediateSize = 32768;
inline constexpr int kVocabSize = 100355;

inline constexpr double kRmsNormEps = 1e-5;
inline constexpr double kRopeTheta = 50000000.0;

// Granite fixed attention score scale (config attention_multiplier), applied
// to the attention logits INSTEAD of 1/sqrt(head_dim).
inline constexpr float kAttentionMultiplier = 0.0078125f;
inline constexpr float kEmbeddingMultiplier = 12.0f;
inline constexpr float kResidualMultiplier = 0.175f;
inline constexpr float kLogitsScaling = 16.0f;

// Adapter metadata (config.json / BUILD.md).
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

} // namespace frozen
} // namespace granite_switch
} // namespace lsxcommon
