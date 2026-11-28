// Mistral-Large-4-1T-A52B frozen geometry passport (Model Ingestion Protocol
// stage 3). Leaf header: never includes another model's passport. Every number
// is compile-time; MistralLarge4MoeModel::Load and the converter cross-check the
// parsed source against this block (a mismatch must fail). Tests relax it only
// through the runtime gate SyntheticGeometryAllowed()
// (LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY).
//
// Source of truth (read-only reference, not vendored):
//   <source root>/mistralai/Mistral-Large-4-1T-A52B/config.json
//   .../consolidated.safetensors.index.json  (56563 tensors, 988 GB)
//   reference: transformers/models/mistral_large4/{configuration,modeling}_mistral_large4.py
//
// Architecture: DeepSeek-style MLA without Q-LoRA (direct wq, 64 heads,
// kv_lora 512, qk_nope 64 + qk_rope 64, v 128), hybrid RoPE/NoPE (every 4th
// layer is NoPE, and its 64 rotary dims are plain key dims), interleaved-pair
// YaRN (theta 1e4, factor 2048 over a 512-token window) scaled per frequency by
// the learned LeRoPE alpha tensor, per-head position-dependent query scaling
// (ssmax), DSA Lightning Indexer (32 heads x 128, top-2048, 44 full layers,
// 28 shared-mask layers), cross-layer latent-KV sharing (layers 39,43,...,71
// reuse layer 35), 128 routed experts top-4 + one gated shared expert, a dense
// layer 0, and final logit softcapping at 30.
#pragma once

#include <array>
#include <cstdint>

namespace lsxcommon {
namespace mistral_large4 {
namespace frozen {

inline constexpr const char *kType = "mistral-large-4-1t";
inline constexpr const char *kFormat = "logestix-mistral-large4-nvfp4";
inline constexpr const char *kModelName = "Mistral-Large-4-1T";

inline constexpr int   kHidden            = 6144;
inline constexpr int   kLayers            = 72;
inline constexpr int   kVocab             = 204800;
inline constexpr float kRmsEps            = 1e-5f;
inline constexpr float kLogitSoftcap      = 30.0f;

// MLA. num_key_value_heads == num_attention_heads (64), so the cache has no
// head repetition; there is no Q-LoRA path.
inline constexpr int   kHeads             = 64;
inline constexpr int   kKvHeads           = 64;
inline constexpr int   kKvLora            = 512;
inline constexpr int   kQkNope            = 64;
inline constexpr int   kQkRope            = 64;
inline constexpr int   kQkHead            = kQkNope + kQkRope;  // 128
inline constexpr int   kVHead             = 128;
inline constexpr int   kQSize             = kHeads * kQkHead;   // 8192
inline constexpr int   kKvAWidthRope      = kKvLora + kQkRope;  // 576
inline constexpr int   kKvAWidthNope      = kKvLora;            // 512
inline constexpr int   kKvBRowsRope       = kHeads * (kQkNope + kVHead);  // 12288
inline constexpr int   kKvBRowsNope       = kHeads * (kQkHead + kVHead);  // 16384
inline constexpr int   kOSize             = kHeads * kVHead;    // 8192

// RoPE / LeRoPE. `alpha` multiplies the inverse frequencies by exp(alpha).
inline constexpr double kRopeTheta        = 10000.0;
inline constexpr double kYarnFactor       = 2048.0;
inline constexpr double kYarnOrig         = 512.0;
inline constexpr double kYarnBetaFast     = 32.0;
inline constexpr double kYarnBetaSlow     = 1.0;
inline constexpr double kYarnAttnFactor   = 1.0;
inline constexpr int    kRopeDim          = kQkRope;   // 64
inline constexpr int    kLeRopeDims       = kRopeDim / 2;  // 32
inline constexpr bool   kRopeInterleaved  = true;

inline constexpr float  kAttnScale        = 0.08838834764831845f;  // 1/sqrt(128)

// DSA Lightning Indexer.
inline constexpr int    kIndexHeads       = 32;
inline constexpr int    kIndexHeadDim     = 128;
inline constexpr int    kIndexTopk        = 2048;
inline constexpr float  kIndexScoreScale  = 0.08838834764831845f;  // 1/sqrt(128)
inline constexpr float  kIndexWeightScale = 0.17677669529663687f;  // 1/sqrt(32)

// MoE / MLP.
inline constexpr int    kFirstKDense       = 1;
inline constexpr int    kMoeFirstLayer     = kFirstKDense;
inline constexpr int    kDenseIntermediate = 24576;
inline constexpr int    kMoeIntermediate   = 6144;
inline constexpr int    kNumRouted         = 128;
inline constexpr int    kTopK              = 4;
inline constexpr int    kSharedExperts     = 1;
inline constexpr int    kSharedIntermediate = kMoeIntermediate * kSharedExperts;
inline constexpr int    kNGroups           = 1;
inline constexpr int    kTopkGroup         = 1;
inline constexpr float  kRoutedScale       = 1.0f;

// Special tokens (tokenizer.json added_tokens; all special:true, ids < 1000).
inline constexpr int32_t kUnkId              = 0;
inline constexpr int32_t kBosId              = 1;
inline constexpr int32_t kEosId              = 2;
inline constexpr int32_t kInstId             = 3;
inline constexpr int32_t kInstEndId          = 4;
inline constexpr int32_t kAvailableToolsId   = 5;
inline constexpr int32_t kAvailableToolsEndId = 6;
inline constexpr int32_t kToolResultsId      = 7;
inline constexpr int32_t kToolResultsEndId   = 8;
inline constexpr int32_t kToolCallsId        = 9;
inline constexpr int32_t kImageId            = 10;
inline constexpr int32_t kPadId              = 11;
inline constexpr int32_t kArgsId             = 32;
inline constexpr int32_t kCallIdId           = 33;
inline constexpr int32_t kThinkId            = 34;
inline constexpr int32_t kThinkEndId         = 35;
inline constexpr int32_t kModelSettingsId    = 36;
inline constexpr int32_t kModelSettingsEndId = 37;

// Per-layer tables, verbatim from config.json. `kRopeLayer[i] == 1` means the
// layer uses rotary embeddings (no_rope_layers[i] == 1); 0 is a NoPE layer.
// `kKvSource[i]` is the layer whose latent KV cache layer i reads
// (kv_source_layers). `kIndexerFull[i] == 1` means the layer runs its own
// indexer (indexer_types "full"); 0 reuses the previous full layer's top-k.
// `kMoeLayer[i] == 1` means the layer is sparse (mlp_layer_types).
inline constexpr std::array<std::uint8_t, kLayers> kRopeLayer = {
    1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,1,0,
    1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,1,0,
    1,1,1,0, 1,1,1,0};

inline constexpr std::array<std::uint8_t, kLayers> kKvSource = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,
   24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,35,40,41,42,35,44,45,46,35,
   48,49,50,35,52,53,54,35,56,57,58,35,60,61,62,35,64,65,66,35,68,69,70,35};

inline constexpr std::array<std::uint8_t, kLayers> kIndexerFull = {
    1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,0,0,1, 1,0,0,1, 1,0,0,1, 1,0,0,1,
    1,0,0,1, 1,0,0,1, 1,0,0,1, 1,0,0,1, 1,0,0,1, 1,0,0,1, 1,0,0,1, 1,0,0,1,
    1,0,0,1, 1,0,0,1};

inline constexpr std::array<std::uint8_t, kLayers> kMoeLayer = {
    0,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1,
    1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1,
    1,1,1,1, 1,1,1,1};

inline constexpr int kNumRopeLayers    = 54;
inline constexpr int kNumNopeLayers    = 18;
inline constexpr int kNumKvShared      = 9;
inline constexpr int kNumFullIndexers  = 44;
inline constexpr int kKvSourceLayer    = 35;  // the only sharing source

// Frozen-geometry invariants (compile-time).
static_assert(kQkHead == 128, "MLA qk head dim");
static_assert(kQSize == 8192, "wq output rows");
static_assert(kKvAWidthRope == 576, "wkv_a (RoPE layer) = 512 latent + 64 k_pe");
static_assert(kKvAWidthNope == 512, "wkv_a (NoPE layer) = 512 latent");
static_assert(kKvBRowsRope == 12288, "wkv_b (RoPE) = 64 x (64 + 128)");
static_assert(kKvBRowsNope == 16384, "wkv_b (NoPE) = 64 x (128 + 128)");
static_assert(kOSize == 8192, "wo input columns");
static_assert(kSharedIntermediate == 6144, "shared expert width");
static_assert(kIndexHeads * kIndexHeadDim == 4096, "indexer wq_b rows");
static_assert(kTopK <= kNumRouted, "top-k exceeds expert count");
static_assert(kHidden % 128 == 0, "hidden must tile by 128");
static_assert(kMoeIntermediate % 128 == 0, "expert width must tile by 128");
static_assert(kRopeDim % 2 == 0, "rope dim must be even");

static_assert(kRopeLayer.size() == static_cast<size_t>(kLayers));
static_assert(kKvSource.size() == static_cast<size_t>(kLayers));
static_assert(kIndexerFull.size() == static_cast<size_t>(kLayers));
static_assert(kMoeLayer.size() == static_cast<size_t>(kLayers));

} // namespace frozen
} // namespace mistral_large4
} // namespace lsxcommon
