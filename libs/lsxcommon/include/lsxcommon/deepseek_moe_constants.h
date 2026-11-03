#pragma once

// DeepSeek-V4-Flash frozen passport.
//
// Stage 0 triage froze the architecture: DeepseekV4ForCausalLM (model_type
// "deepseek_v4") — MLA + Hyper-Connections + compressed-KV/indexer + hash
// router + sqrtsoftplus MoE — shared with V4-Pro-0813 and V4.1-Flash. The
// driver is one templated implementation (<class T> with T = a passport); this
// header is Flash's, and it changes only the values and the residual type.
//
// Values verified against the source checkpoint (inference/config.json) on
// 2026-11-03; geometry drift is a new family (Model Ingestion Protocol
// stage 0.1), locked by the static_asserts at the bottom.

#include <cuda_fp16.h>

#include <cstdint>

namespace lsxcommon {
namespace deepseek {

struct Passport {
    // Hyper-Connections keep the residual stream in FP16 on Flash; V4-Pro
    // needs FP32 because its HC sum exceeds the FP16 range (bf16/fp8 source).
    using ResidualT = __half;

    // ── Backbone geometry (config.json) ────────────────────────────────────
    static constexpr int kHidden = 4096;              // dim
    static constexpr int kLayers = 43;                // n_layers
    static constexpr int kHeads = 64;                 // n_heads
    static constexpr int kHeadDim = 512;              // head_dim
    static constexpr int kRopeHeadDim = 64;           // rope_head_dim
    static constexpr int kQLoraRank = 1024;           // q_lora_rank
    static constexpr int kOLoraRank = 1024;           // o_lora_rank
    static constexpr int kOGroups = 8;                // o_groups
    static constexpr int kMoeIntermediate = 2048;     // moe_inter_dim
    static constexpr int kSharedIntermediate = 2048;  // shared expert width
    static constexpr int kNumRouted = 256;            // n_routed_experts
    static constexpr int kTopK = 6;                   // n_activated_experts
    static constexpr int kWindow = 128;               // window_size
    static constexpr int kVocab = 129280;             // vocab_size
    static constexpr int kIndexNHeads = 64;           // index_n_heads
    static constexpr int kIndexHeadDim = 128;         // index_head_dim
    static constexpr int kIndexTopk = 512;            // index_topk
    static constexpr int kHashLayers = 3;             // n_hash_layers

    // Hyper-Connections (hc_mult copies of the residual stream).
    static constexpr int kHcMult = 4;
    static constexpr int kHcMix = (2 + kHcMult) * kHcMult;  // 24
    static constexpr int kHcDim = kHcMult * kHidden;        // 16384
    static constexpr int kSinkhornIters = 20;              // hc_sinkhorn_iters

    // ── Numeric constants ─────────────────────────────────────────────────
    static constexpr float kRmsEps = 1e-6f;
    static constexpr float kHcEps = 1e-6f;
    static constexpr float kRouteScale = 1.5f;
    static constexpr float kGateTemp = 1.0f;
    static constexpr float kSwigluLimit = 10.0f;
    static constexpr float kCompressRopeTheta = 160000.0f;
    static constexpr float kRopeTheta = 10000.0f;
    static constexpr int kRopeFactor = 16;
    static constexpr int kOriginalSeqLen = 65536;
    static constexpr int kBetaFast = 32;
    static constexpr int kBetaSlow = 1;

    // ── Per-layer compression grid (config.json compress_ratios[0..42]) ─────
    // Flash: layers 0-1 are pure MLA (ratio 0), 2..42 alternate 4/128 with
    // the ratio-4 layers carrying the indexer. The grid is also delivered at
    // runtime by the loader (`pw.layers[i].compress_ratio`); this table is the
    // compile-time contract the loader is checked against.
    static constexpr int CompressRatio(int layer)
    {
        if (layer < 0 || layer >= kLayers) return 0;
        if (layer < 2) return 0;
        return (layer % 2 == 0) ? 4 : 128;
    }

    static constexpr bool HasIndexer(int layer)
    {
        return CompressRatio(layer) == 4;
    }

    static constexpr bool IsHashLayer(int layer)
    {
        return layer >= 0 && layer < kHashLayers;
    }

    // CSA overlap: ratio 4 pools two consecutive blocks (coff = 1 + overlap).
    static constexpr int CompressorCoff(int layer)
    {
        return CompressRatio(layer) == 4 ? 2 : 1;
    }
};

// Registry id (detect_model_type normalization of "DeepSeek-V4-Flash").
inline constexpr char kType[] = "deepseek-v4-flash";

// Sole supported vault index format: routed experts carry pow2-E4M3 per-16
// scales + per-tensor global_scale.
inline constexpr char kIndexFormatNvfp4[] = "logestix-v1-nvfp4";

// ── Compile-time relations (MODEL INGEST INVARIANT step 3) ──────────────────
static_assert(Passport::kLayers == 43, "V4-Flash backbone is 43 layers");
static_assert(Passport::kHidden == 4096, "V4-Flash hidden is 4096");
static_assert(Passport::kHcDim == Passport::kHcMult * Passport::kHidden,
              "hc flattened width");
static_assert(Passport::kHcMix == (2 + Passport::kHcMult) * Passport::kHcMult,
              "hc mix rows");
static_assert(Passport::kHidden % 256 == 0,
              "hidden tiles cleanly on Blackwell");
static_assert(Passport::kMoeIntermediate % 256 == 0,
              "moe intermediate tiles cleanly");
static_assert(Passport::kSharedIntermediate == Passport::kMoeIntermediate,
              "shared expert width");
static_assert(Passport::kTopK > 0 && Passport::kTopK <= Passport::kNumRouted,
              "bad top-k");
static_assert(Passport::kHeadDim > Passport::kRopeHeadDim,
              "rope head must be a sub-vector");
static_assert(Passport::kHeads * Passport::kHeadDim % Passport::kOGroups == 0,
              "grouped wo_a geometry");
static_assert(Passport::kHeads % Passport::kOGroups == 0,
              "heads must split across o_groups");
static_assert(Passport::kIndexNHeads * Passport::kIndexHeadDim == 64 * 128,
              "indexer head geometry");
static_assert(Passport::kHashLayers == 3, "V4-Flash has three hash layers");
static_assert(Passport::CompressRatio(0) == 0 &&
                  Passport::CompressRatio(1) == 0,
              "layers 0-1 are pure MLA");
static_assert(Passport::CompressRatio(2) == 4 &&
                  Passport::CompressRatio(3) == 128,
              "ratio grid alternates 4/128 from layer 2");
static_assert(Passport::CompressRatio(42) == 4,
              "last backbone layer is ratio 4");
static_assert(Passport::HasIndexer(2) && Passport::HasIndexer(42) &&
                  !Passport::HasIndexer(3),
              "indexer only on ratio-4 layers");
static_assert(Passport::CompressorCoff(2) == 2 &&
                  Passport::CompressorCoff(3) == 1,
              "coff follows the overlap bit");
static_assert(Passport::IsHashLayer(0) && Passport::IsHashLayer(2) &&
                  !Passport::IsHashLayer(3),
              "hash router covers layers 0..2");

} // namespace deepseek
} // namespace lsxcommon
