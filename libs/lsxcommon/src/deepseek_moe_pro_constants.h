#pragma once

// DeepSeek-V4-Pro-0813 (checkpoint "DeepSeek-V4-Pro-0813") frozen passport.
//
// Stage 0 triage: the architecture is DeepseekV4ForCausalLM (model_type
// "deepseek_v4") — the SAME MLA + Hyper-Connections + compressed-KV/indexer
// + hash-router + sqrtsoftplus MoE contour as DeepSeek-V4-Flash — but with a
// different geometry (71.68e2 hidden, 61 layers, 128 heads, 384x3072 experts,
// q_lora 1536, o_groups 16) and a different per-layer compression grid
// (layers 0-1 ratio 128, then 4/128 alternating). Geometry drift is a new
// family (Model Ingestion Protocol stage 0.1), so these constants are locked
// here and cross-checked by the loader/converter.
//
// Verified against the source checkpoint (inference/config.json + the
// safetensors inventory) on 2026-10-26:
//   routed experts  : 61 x 384 x 3, E2M1 nibbles packed 2/byte (I8) with
//                     F8_E8M0 per-32 scales ("expert_dtype": "fp4")
//   dense           : F8_E4M3 with F8_E8M0 per-128x128 block scales
//   embed/head/norm : BF16 (ties: false)
//   hash router     : layers 0..2 (`ffn.gate.tid2eid`), gate.bias afterwards
//   indexer         : every compress_ratio==4 layer (2,4,...,60)
//   ballast         : mtp.0..2 (DSpark speculative stage) stored+counted,
//                     never applied
#include <cstdint>

namespace lsxcommon {
namespace deepseek_pro {

// Registry id (detect_model_type normalization of "DeepSeek-V4-Pro-0813").
inline constexpr char kType[] = "deepseek-v4-pro-0813";

// ── Frozen backbone geometry ────────────────────────────────────────────────
struct Passport {
    // V4-Pro keeps the Hyper-Connections residual in FP32: the HC sum exceeds
    // FP16's 65504 range (the source runs BF16/FP8 with a 3e38 range) and is
    // stored in FP32 so it does not become inf before the scale-invariant norm.
    using ResidualT = float;

    static constexpr int kHidden = 7168;
    static constexpr int kLayers = 61;
    static constexpr int kHeads = 128;
    static constexpr int kHeadDim = 512;
    static constexpr int kRopeHeadDim = 64;
    static constexpr int kQLoraRank = 1536;
    static constexpr int kOLoraRank = 1024;
    static constexpr int kOGroups = 16;
    static constexpr int kMoeIntermediate = 3072;
    static constexpr int kSharedIntermediate = 3072;
    static constexpr int kNumRouted = 384;
    static constexpr int kTopK = 6;
    static constexpr int kIndexNHeads = 64;
    static constexpr int kIndexHeadDim = 128;
    static constexpr int kIndexTopk = 1024;
    static constexpr int kWindow = 128;      // sliding-window ring size
    static constexpr int kVocab = 129280;
    static constexpr int kHashLayers = 3;    // tid2eid router layers 0..2

    // Hyper-Connections (hc_mult copies of the residual stream).
    static constexpr int kHcMult = 4;
    static constexpr int kHcMix = (2 + kHcMult) * kHcMult;  // 24
    static constexpr int kHcDim = kHcMult * kHidden;        // 28672
    static constexpr int kSinkhornIters = 20;

    // Numeric constants.
    static constexpr float kRmsEps = 1e-6f;
    static constexpr float kHcEps = 1e-6f;
    static constexpr float kRouteScale = 2.5f;
    static constexpr float kGateTemp = 1.0f;
    static constexpr float kSwigluLimit = 10.0f;
    static constexpr float kCompressRopeTheta = 160000.0f;
    static constexpr float kRopeTheta = 10000.0f;
    static constexpr int kRopeFactor = 16;
    static constexpr int kOriginalSeqLen = 65536;
    static constexpr int kBetaFast = 32;
    static constexpr int kBetaSlow = 1;
};  // struct Passport

// Legacy aliases: the templated driver refers to T::k*, the existing call
// sites keep the bare names until the pro copies are deleted (task 7).
inline constexpr int kHidden = Passport::kHidden;
inline constexpr int kLayers = Passport::kLayers;
inline constexpr int kHeads = Passport::kHeads;
inline constexpr int kHeadDim = Passport::kHeadDim;
inline constexpr int kRopeHeadDim = Passport::kRopeHeadDim;
inline constexpr int kQLoraRank = Passport::kQLoraRank;
inline constexpr int kOLoraRank = Passport::kOLoraRank;
inline constexpr int kOGroups = Passport::kOGroups;
inline constexpr int kMoeIntermediate = Passport::kMoeIntermediate;
inline constexpr int kSharedIntermediate = Passport::kSharedIntermediate;
inline constexpr int kNumRouted = Passport::kNumRouted;
inline constexpr int kTopK = Passport::kTopK;
inline constexpr int kIndexNHeads = Passport::kIndexNHeads;
inline constexpr int kIndexHeadDim = Passport::kIndexHeadDim;
inline constexpr int kIndexTopk = Passport::kIndexTopk;
inline constexpr int kWindow = Passport::kWindow;
inline constexpr int kVocab = Passport::kVocab;
inline constexpr int kHashLayers = Passport::kHashLayers;
inline constexpr int kHcMult = Passport::kHcMult;
inline constexpr int kHcMix = Passport::kHcMix;
inline constexpr int kHcDim = Passport::kHcDim;
inline constexpr int kSinkhornIters = Passport::kSinkhornIters;
inline constexpr float kRmsEps = Passport::kRmsEps;
inline constexpr float kHcEps = Passport::kHcEps;
inline constexpr float kRouteScale = Passport::kRouteScale;
inline constexpr float kGateTemp = Passport::kGateTemp;
inline constexpr float kSwigluLimit = Passport::kSwigluLimit;
inline constexpr float kCompressRopeTheta = Passport::kCompressRopeTheta;
inline constexpr float kRopeTheta = Passport::kRopeTheta;
inline constexpr int kRopeFactor = Passport::kRopeFactor;
inline constexpr int kOriginalSeqLen = Passport::kOriginalSeqLen;
inline constexpr int kBetaFast = Passport::kBetaFast;
inline constexpr int kBetaSlow = Passport::kBetaSlow;


// DSpark / MTP speculative stage: stored, counted, never applied.
inline constexpr int kMtpLayers = 3;
inline constexpr int kDsparkBlockSize = 5;
inline constexpr int kDsparkNoiseTokenId = 128799;
inline constexpr int kDsparkMarkovRank = 512;

// ── Per-layer compression grid (config.json compress_ratios[0..60]) ─────────
// 0-1: ratio 128 (non-overlapping), 2..60: even ratio 4 (overlap/coff=2),
// odd ratio 128. Entries 61..63 are the MTP layers (ratio 0, not run).
constexpr int CompressRatio(int layer)
{
    if (layer < 0 || layer >= kLayers) return 0;
    if (layer < 2) return 128;
    return (layer % 2 == 0) ? 4 : 128;
}

// Only ratio-4 layers instantiate the indexer (model.py: Indexer created
// when compress_ratio == 4).
constexpr bool HasIndexer(int layer)
{
    return CompressRatio(layer) == 4;
}

constexpr bool IsHashLayer(int layer)
{
    return layer >= 0 && layer < kHashLayers;
}

// CSA overlap: ratio 4 pools two consecutive blocks (coff = 1 + overlap).
constexpr int CompressorCoff(int layer)
{
    return CompressRatio(layer) == 4 ? 2 : 1;
}

// ── Compile-time relations (MODEL INGEST INVARIANT step 3) ──────────────────
static_assert(kLayers == 61, "V4-Pro backbone is 61 layers");
static_assert(kHidden == 7168, "V4-Pro hidden is 7168");
static_assert(kHcDim == kHcMult * kHidden, "hc flattened width");
static_assert(kHcMix == (2 + kHcMult) * kHcMult, "hc mix rows");
static_assert(kHidden % 256 == 0, "hidden tiles cleanly on Blackwell");
static_assert(kMoeIntermediate % 256 == 0, "moe intermediate tiles cleanly");
static_assert(kSharedIntermediate == kMoeIntermediate, "shared expert width");
static_assert(kTopK > 0 && kTopK <= kNumRouted, "bad top-k");
static_assert(kHeadDim > kRopeHeadDim, "rope head must be a sub-vector");
static_assert(kHeads * kHeadDim % kOGroups == 0, "grouped wo_a geometry");
static_assert(kHeads % kOGroups == 0, "heads must split across o_groups");
static_assert(kIndexNHeads * kIndexHeadDim == 64 * 128,
              "indexer head geometry");
static_assert(kHashLayers == 3, "V4-Pro has three hash-routing layers");
static_assert(CompressRatio(0) == 128 && CompressRatio(1) == 128,
              "layers 0-1 compress at ratio 128");
static_assert(CompressRatio(2) == 4 && CompressRatio(3) == 128,
              "ratio grid alternates 4/128 from layer 2");
static_assert(CompressRatio(60) == 4, "last backbone layer is ratio 4");
static_assert(CompressRatio(61) == 0, "MTP layers are outside the grid");
static_assert(HasIndexer(2) && HasIndexer(60) && !HasIndexer(3),
              "indexer only on ratio-4 layers");
static_assert(CompressorCoff(2) == 2 && CompressorCoff(3) == 1,
              "coff follows the overlap bit");
static_assert(IsHashLayer(0) && IsHashLayer(2) && !IsHashLayer(3),
              "hash router covers layers 0..2");

} // namespace deepseek_pro
} // namespace lsxcommon
