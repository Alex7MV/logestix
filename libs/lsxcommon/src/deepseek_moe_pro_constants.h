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
inline constexpr int kHidden = 7168;
inline constexpr int kLayers = 61;
inline constexpr int kHeads = 128;
inline constexpr int kHeadDim = 512;
inline constexpr int kRopeHeadDim = 64;
inline constexpr int kQLoraRank = 1536;
inline constexpr int kOLoraRank = 1024;
inline constexpr int kOGroups = 16;
inline constexpr int kMoeIntermediate = 3072;
inline constexpr int kSharedIntermediate = 3072;
inline constexpr int kNumRouted = 384;
inline constexpr int kTopK = 6;
inline constexpr int kIndexNHeads = 64;
inline constexpr int kIndexHeadDim = 128;
inline constexpr int kIndexTopk = 1024;
inline constexpr int kWindow = 128;      // sliding-window ring size
inline constexpr int kVocab = 129280;
inline constexpr int kHashLayers = 3;    // tid2eid router layers 0..2

// Hyper-Connections (hc_mult copies of the residual stream).
inline constexpr int kHcMult = 4;
inline constexpr int kHcMix = (2 + kHcMult) * kHcMult;  // 24
inline constexpr int kHcDim = kHcMult * kHidden;        // 28672
inline constexpr int kSinkhornIters = 20;

// Numeric constants.
inline constexpr float kRmsEps = 1e-6f;
inline constexpr float kHcEps = 1e-6f;
inline constexpr float kRouteScale = 2.5f;
inline constexpr float kGateTemp = 1.0f;
inline constexpr float kSwigluLimit = 10.0f;
inline constexpr float kCompressRopeTheta = 160000.0f;
inline constexpr float kRopeTheta = 10000.0f;
inline constexpr int kRopeFactor = 16;
inline constexpr int kOriginalSeqLen = 65536;
inline constexpr int kBetaFast = 32;
inline constexpr int kBetaSlow = 1;

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
