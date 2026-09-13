#pragma once

#include <cstdint>

namespace lsxcommon {
namespace deepseek_v41 {

// Registry id (detect_model_type normalization of "DeepSeek-V4.1-Flash").
inline constexpr char kType[] = "deepseek-v4.1-flash";

// ── Frozen text-backbone geometry (source: config.json text_config +
//    inference/config.json). Locked compile-time per MODEL INGEST step 3.
inline constexpr int kHidden = 5120;
inline constexpr int kLayers = 40;
inline constexpr int kMoeIntermediate = 2304;
inline constexpr int kSharedIntermediate = 2304;
inline constexpr int kNumRouted = 384;
inline constexpr int kTopK = 6;
inline constexpr int kHeads = 64;
inline constexpr int kHeadDim = 512;
inline constexpr int kRopeHeadDim = 64;
inline constexpr int kQLoraRank = 1280;
inline constexpr int kOLoraRank = 1024;
inline constexpr int kOGroups = 8;
inline constexpr int kIndexNHeads = 32;
inline constexpr int kIndexHeadDim = 128;
inline constexpr int kIndexTopk = 512;
inline constexpr int kWindow = 128;              // sliding-window ring size
inline constexpr int kVocab = 129280;
inline constexpr int kHcMult = 4;
inline constexpr int kHcMix = (2 + kHcMult) * kHcMult;  // 24
inline constexpr int kHcDim = kHcMult * kHidden;        // 20480
inline constexpr int kSinkhornIters = 20;
inline constexpr float kRmsEps = 1e-20f;
inline constexpr float kHcEps = 1e-6f;
inline constexpr float kRouteScale = 1.5f;
inline constexpr float kGateTemp = 1.0f;
inline constexpr float kSwigluLimit = 10.0f;
inline constexpr float kCompressRopeTheta = 160000.0f;  // ratio > 0 layers
inline constexpr float kRopeTheta = 10000.0f;           // pure-SWA layers
inline constexpr int kRopeFactor = 16;
inline constexpr int kOriginalSeqLen = 65536;
inline constexpr int kBetaFast = 32;
inline constexpr int kBetaSlow = 1;
inline constexpr int kCandidateBlockSize = 8;
inline constexpr int kCandidateTopkBlocks = 2048;
inline constexpr int kMtpLayers = 3;   // stored-not-applied

// ── CED classification (inference/config.json compress_ratios + source sets).
// Layers 0-1: pure sliding window. 2..19: compressed, ratio 2 (softmax
// pooling). 20..39: compressed, ratio 1 (plain projection). The three MTP
// layers (indices 40..42) are ratio 0 but never run in the backbone.
constexpr int CompressRatio(int layer)
{
    return layer < 2 ? 0 : (layer < 20 ? 2 : 1);
}

constexpr bool IsKvSource(int layer)
{
    return layer == 2 || layer == 8 || layer == 14 || layer == 20;
}

constexpr bool IsIndexSource(int layer)
{
    return layer == 2 || layer == 8 || layer == 14 || layer == 20 ||
           layer == 24 || layer == 28 || layer == 32 || layer == 36;
}

inline constexpr int kCandidateSourceLayer = 20;

constexpr bool IsCandidateSource(int layer)
{
    return layer == kCandidateSourceLayer;
}

struct LayerMeta {
    int compress_ratio;
    bool is_kv_source;
    bool is_index_source;
    bool is_candidate_source;
};

constexpr LayerMeta LayerMetaAt(int layer)
{
    return LayerMeta{CompressRatio(layer), IsKvSource(layer),
                     IsIndexSource(layer), IsCandidateSource(layer)};
}

// ── Compile-time relations (MODEL INGEST INVARIANT step 3).
static_assert(kLayers == 40, "V4.1 backbone is 40 layers");
static_assert(kHidden == 5120, "V4.1 hidden is 5120");
static_assert(kHcDim == kHcMult * kHidden, "hc flattened width");
static_assert(kHcMix == (2 + kHcMult) * kHcMult, "hc mix rows");
static_assert(kHidden % 256 == 0, "hidden tiles cleanly on Blackwell");
static_assert(kMoeIntermediate % 256 == 0, "moe intermediate tiles cleanly");
static_assert(kTopK > 0 && kTopK <= kNumRouted, "bad top-k");
static_assert(kHeadDim > kRopeHeadDim, "rope head must be a sub-vector");
static_assert(kHeads * kHeadDim % kOGroups == 0, "grouped wo_a geometry");
static_assert(kIndexNHeads * kIndexHeadDim == 32 * 128,
              "indexer head geometry");
static_assert(CompressRatio(0) == 0 && CompressRatio(1) == 0,
              "layers 0-1 are pure SWA");
static_assert(CompressRatio(2) == 2 && CompressRatio(19) == 2,
              "layers 2-19 compress at ratio 2");
static_assert(CompressRatio(20) == 1 && CompressRatio(39) == 1,
              "layers 20-39 compress at ratio 1");
static_assert(IsKvSource(2) && IsKvSource(20) && !IsKvSource(21),
              "kv_source set");
static_assert(IsIndexSource(2) && IsIndexSource(36) && !IsIndexSource(35),
              "index_source set");

constexpr bool KvSourceIsIndexSource()
{
    for (int l = 0; l < kLayers; ++l)
        if (IsKvSource(l) && !IsIndexSource(l)) return false;
    return true;
}
static_assert(KvSourceIsIndexSource(), "kv sources must run an indexer");

} // namespace deepseek_v41
} // namespace lsxcommon
