#pragma once

// NVIDIA-Nemotron-Labs-Dory1.0-4B frozen geometry passport (Model Ingestion
// Protocol stage 3).
//
// Dory is a *looped (latent-recurrent) normalized-GPT* (nGPT) decoder, not a
// transformer with pre-norms:
//   * every physical "layer" is ONE mixer (attention XOR mlp) with its own
//     nGPT residual; there is no RMSNorm and no final norm anywhere;
//   * the residual stream is L2-normalized (`justnorm`) after every mixer, with
//     a spherical interpolation `justnorm(a) + |alpha*alpha_scale| *
//     (justnorm(b) - justnorm(a))`, renormalized;
//   * the middle `kRecurrentLayers` mixers are applied `kRecurrentLoops` times,
//     re-injecting the input block's output each loop.
//
// Reference implementation: vllm-project/vllm#60149
// (vllm/model_executor/models/dory.py) — transcribed and verified against the
// real checkpoint in tools/dory_oracle.py. Every constant below is pinned from
// that file's config.json.
//
// This header is a LEAF: no other model's passport may be included
// (model-independence invariant). Device-safe: no abseil containers/streams.

#include <cstddef>
#include <cstdint>
#include <string>

#include "lsxcommon/vram_estimate.h"

namespace lsxcommon {
namespace dory {

// Loaded/converted topology of a Dory checkpoint (geometry + special tokens +
// the mandatory conversion-time VRAM profile). Device-safe.
struct DoryTopology {
    int64_t layout = 0;  // schema layout generation (kIndexLayout)
    int num_layers = 0;             // physical mixers (config num_hidden_layers)
    int hidden_size = 0;
    int intermediate_size = 0;
    int vocab_size = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int head_dim = 0;
    int input_layers = 0;
    int recurrent_layers = 0;
    int output_layers = 0;
    int recurrent_loops = 1;
    int sliding_window = 0;
    int num_attention_layers = 0;   // counted from the frozen layer pattern
    double rope_theta_sliding = 10000.0;
    double rope_theta_full = 10000000.0;
    // nGPT scale reference: every learned nGPT vector is normalized by this
    // value at inference time (see the factors below).
    double initializer_range = 0.0177;
    double ngpt_alpha_init = 0.05;
    // Textbook 1/sqrt(head_dim); the checkpoint folds nothing into the weights.
    double attention_scale = 0.0625;
    int bos_id = -1;
    int eos_id = -1;
    int pad_id = -1;
    int total_pinned_tensors = 0;
    VramProfile vram;        // conversion-time VRAM requirements (mandatory)
    std::string error_msg;   // non-empty when index.json is invalid
};

namespace frozen {

// ── Frozen compile-time geometry (config.json, 2026-11-10) ─────────────────
inline constexpr int kHiddenSize = 2560;
inline constexpr int kNumLayers = 72;          // single-mixer layers
inline constexpr int kNumHeads = 16;
inline constexpr int kNumKVHeads = 4;
inline constexpr int kHeadDim = 256;
inline constexpr int kQSize = kNumHeads * kHeadDim;      // 4096
inline constexpr int kKVSize = kNumKVHeads * kHeadDim;   // 1024
inline constexpr int kIntermediateSize = 7168;
inline constexpr int kVocabSize = 131072;

// Layer partition (config n_input_layers / n_recurrent_layers /
// n_output_layers / n_recurrent_loops) and the per-layer mixer pattern:
// attention at even indices, mlp at odd. `full_attention` iff layer % 8 == 6,
// `sliding_attention` otherwise (verified against config.layer_types for all
// 72 entries).
inline constexpr int kInputLayers = 16;
inline constexpr int kRecurrentLayers = 40;
inline constexpr int kOutputLayers = 16;
inline constexpr int kRecurrentLoops = 2;
inline constexpr int kNumAttentionLayers = kNumLayers / 2;   // 36
inline constexpr int kSlidingWindow = 1024;

inline constexpr double kRopeThetaSliding = 10000.0;
inline constexpr double kRopeThetaFull = 10000000.0;
inline constexpr double kJustNormEps = 1e-6;

// nGPT scale reference + the three derived factors the reference multiplies the
// learned vectors with before use. Kept as exact double literals (the factor is
// a function of constants; a change here is a re-ingestion).
inline constexpr double kInitializerRange = 0.0177;
inline constexpr double kNgptAlphaInit = 0.05;
inline constexpr double kSqkFactor = 903.954802259887;      // sqrt(hd)/std
inline constexpr double kSGateFactor = 401.87105311614073;  // sqrt(sqrt(H))/std
inline constexpr double kSuvFactor = 50.59644256269407;     // sqrt(H)
inline constexpr double kAlphaScale = 2.824858757062147;    // alpha_init/std
inline constexpr float kAttentionScale = 0.0625f;           // 1/sqrt(hd)

// Special tokens: bos <s> / eos+pad <|im_end|> (generation_config.json).
inline constexpr int32_t kBosId = 1;
inline constexpr int32_t kEosId = 11;
inline constexpr int32_t kPadId = 11;

// Compile-time invariants.
static_assert(kInputLayers + kRecurrentLayers + kOutputLayers == kNumLayers,
              "the Dory layer partition must cover every physical layer");
static_assert(kNumLayers % 2 == 0 && kNumAttentionLayers == kNumLayers / 2,
              "mixers alternate attention/mlp over an even layer count");
static_assert(kInputLayers % 2 == 0 && kRecurrentLayers % 2 == 0 &&
                  kOutputLayers % 2 == 0,
              "the partition must not split an attention/mlp pair");
static_assert(kRecurrentLoops >= 1, "at least one recurrent pass is required");
static_assert(kNumHeads % kNumKVHeads == 0, "GQA group must divide the heads");
static_assert(kQSize != kHiddenSize,
              "Dory's attention is rectangular (hidden 2560 != 16*256); the "
              "square-geometry dense kernels cannot serve it");
static_assert(kVocabSize > 0 && kIntermediateSize > 0 && kHeadDim > 0,
              "positive Dory geometry required");

// Physical layer role (compile-time mirror of config.layer_types).
constexpr bool LayerIsAttention(int layer)
{
    return (layer & 1) == 0;
}
constexpr bool LayerIsFullAttention(int layer)
{
    return layer % 8 == 6;
}
constexpr bool LayerIsRecurrent(int layer)
{
    return layer >= kInputLayers && layer < kInputLayers + kRecurrentLayers;
}
// True when the layer is a *recurrent attention* layer, i.e. the only layer
// kind that needs one KV cache per loop.
constexpr bool LayerNeedsPerLoopCache(int layer)
{
    return LayerIsAttention(layer) && LayerIsRecurrent(layer);
}

constexpr int CountAttentionLayers()
{
    int n = 0;
    for (int l = 0; l < kNumLayers; ++l)
        if (LayerIsAttention(l)) ++n;
    return n;
}
static_assert(CountAttentionLayers() == kNumAttentionLayers,
              "attention-layer count must match the frozen pattern");

// KV cache slot count: one per attention layer, times kRecurrentLoops for the
// recurrent ones (`recurrent_kv_cache_mode: per_loop`).
constexpr int CountKvSlots()
{
    int n = 0;
    for (int l = 0; l < kNumLayers; ++l) {
        if (!LayerIsAttention(l)) continue;
        n += LayerNeedsPerLoopCache(l) ? kRecurrentLoops : 1;
    }
    return n;
}
inline constexpr int kKvSlots = 56;   // 8 input + 20*2 recurrent + 8 output
static_assert(CountKvSlots() == kKvSlots,
              "KV slot count must match the frozen pattern");

} // namespace frozen
} // namespace dory
} // namespace lsxcommon
