#pragma once

// Frozen Kimi-K3 geometry (compile-time passport).
// Source of truth: <source root>/moonshotai/Kimi-K3 (config.json +
// modeling_kimi_linear.py + configuration_kimi_k3.py + safetensors headers,
// audited 2026-11-03):
// 93 layers (layer 0 dense MLP, 1..92 Stable LatentMoE), hidden 7168,
// 69 KDA linear-attention layers (96 heads x 128, conv-4 silu, full-rank gate,
// gate_lower_bound -5, channel-wise A_log) + 24 gated MLA layers
// (q_lora 1536, kv_lora 512, qk_nope 128, qk_rope 64, v 128, NoPE, output
// gate, scale 1/sqrt(192)) at the 0-based layers {3,7,...,91,92},
// AttnRes depth mixer with block 12, 896 routed experts top-16 with a 3584
// latent (RMSNorm) and 2 shared experts (I=6144), sigmoid router with a
// correction bias + renormalisation, SiTU-GLU (beta 4, linear beta 25),
// weight-form RMSNorm eps 1e-5, vocab 163840 (163584 base + 256 reserved),
// bos 163584, stop 163586 ([EOS] 163585 also stops), pad 163839.
// Source precision: MXFP4 (E2M1 + E8M0/32) routed experts; everything else
// BF16, KDA conv/A_log/dt_bias/o_norm F32. Vision tower + mm_projector are
// stripped and counted. No MTP layer.

#include <array>

namespace lsxcommon::kimi_k3 {
namespace frozen {

inline constexpr char kModelType[] = "kimi-k3";
inline constexpr char kFormatTag[] = "logestix-kimi-k3-nvfp4";
inline constexpr int kTokenizerKind = 2;  // tiktoken BPE (Kimi pattern)

// ── text tower ────────────────────────────────────────────────────────────
inline constexpr int kHidden = 7168;
inline constexpr int kLayers = 93;
inline constexpr int kDenseLayers = 1;  // layer 0: dense SiTU MLP
inline constexpr int kAttnResBlock = 12;
inline constexpr int kFullAttnInterval = 4;  // 3 KDA -> 1 MLA, repeating
inline constexpr double kRmsNormEps = 1e-5;

// ── hybrid attention tables (0-based) ─────────────────────────────────────
inline constexpr int kMlaLayerCount = 24;
inline constexpr int kKdaLayerCount = 69;
inline constexpr std::array<int, kMlaLayerCount> kMlaLayers = {
    3,  7,  11, 15, 19, 23, 27, 31, 35, 39, 43, 47,
    51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 92};

constexpr bool IsMlaLayer(int layer)
{
    for (int l : kMlaLayers)
        if (l == layer) return true;
    return false;
}

constexpr bool IsKdaLayer(int layer) { return !IsMlaLayer(layer); }

// ── gated MLA (NoPE) ──────────────────────────────────────────────────────
inline constexpr int kMlaHeads = 96;
inline constexpr int kMlaKVHeads = 96;  // no GQA repeat
inline constexpr int kQLoraRank = 1536;
inline constexpr int kKvLoraRank = 512;
inline constexpr int kQkNopeHeadDim = 128;
inline constexpr int kQkRopeHeadDim = 64;
inline constexpr int kVHeadDim = 128;
inline constexpr int kQHeadDim = kQkNopeHeadDim + kQkRopeHeadDim;
inline constexpr int kMlaLatentWidth = kKvLoraRank + kQkRopeHeadDim;
inline constexpr bool kMlaOutputGate = true;
inline constexpr bool kMlaUsesRotary = false;  // mla_use_nope
inline constexpr int kMlaScaleDenom = kQHeadDim;
inline constexpr int kMlaQProjRows = kMlaHeads * kQHeadDim;
inline constexpr int kMlaKvProjRows = kMlaHeads * (kQkNopeHeadDim + kVHeadDim);
inline constexpr int kMlaOutRows = kMlaHeads * kVHeadDim;

// ── KDA (Kimi Delta Attention) ────────────────────────────────────────────
inline constexpr int kKdaHeads = 96;
inline constexpr int kKdaHeadDim = 128;
inline constexpr int kKdaConvKernel = 4;
inline constexpr double kKdaGateLowerBound = -5.0;
inline constexpr bool kKdaFullRankGate = true;
// The released checkpoint stores A_log with head_dim (128) entries, not one
// per head, so the decay is channel-wise and shared across heads.
inline constexpr bool kKdaA_logPerChannel = true;
inline constexpr int kKdaGateRank = 128;  // f_a_proj rows
inline constexpr int kKdaProjRows = kKdaHeads * kKdaHeadDim;

// ── Stable LatentMoE ──────────────────────────────────────────────────────
inline constexpr int kExperts = 896;
inline constexpr int kTopK = 16;
inline constexpr int kMoeIntermediate = 3072;
inline constexpr int kLatentDim = 3584;
inline constexpr bool kLatentMoeNorm = true;
inline constexpr int kSharedExperts = 2;
inline constexpr int kSharedIntermediate = kMoeIntermediate * kSharedExperts;
inline constexpr bool kMoeRenormalize = true;
inline constexpr double kRouteScale = 1.0;
inline constexpr int kExpertGroup = 1;  // use_grouped_topk with a single group
inline constexpr int kTopkGroup = 1;

// ── SiTU-GLU ──────────────────────────────────────────────────────────────
inline constexpr double kSituBeta = 4.0;
inline constexpr double kSituLinearBeta = 25.0;

// ── vocabulary / special ids ──────────────────────────────────────────────
inline constexpr int kVocab = 163840;
inline constexpr int kBaseVocab = 163584;
inline constexpr int kBosId = 163584;      // [BOS]
inline constexpr int kEosId = 163586;      // <|end_of_msg|>
inline constexpr int kStopIdAlt = 163585;  // [EOS]
inline constexpr int kPadId = 163839;

// ── chat / tool markers (XTML) ────────────────────────────────────────────
inline constexpr int kMarkerOpen = 163587;   // <|open|>  (special:false)
inline constexpr int kMarkerClose = 163588;  // <|close|> (special:false)
inline constexpr int kMarkerSep = 163589;    // <|sep|>   (special:false)

// ── frozen-geometry invariant checks ──────────────────────────────────────
static_assert(kLayers == 93, "Kimi-K3 has 93 decoder layers");
static_assert(kMlaLayerCount + kKdaLayerCount == kLayers,
              "hybrid layer tables must partition the stack");
static_assert(kMlaLayerCount == 24 && kKdaLayerCount == 69,
              "24 gated-MLA + 69 KDA layers");
static_assert(kMlaLayers.back() == kLayers - 1,
              "the final layer is full attention");
static_assert(kHidden % kKdaHeadDim == 0, "hidden/head-dim mismatch");
static_assert(kKdaHeads * kKdaHeadDim == 12288, "KDA projection width");
static_assert(kQHeadDim == 192, "MLA q head dim");
static_assert(kMlaLatentWidth == 576, "MLA kv_a width");
static_assert(kMlaHeads == kMlaKVHeads, "Kimi-K3 MLA has no GQA repeat");
static_assert(kAttnResBlock % kFullAttnInterval == 0,
              "AttnRes block must span whole attention groups");
static_assert(kLayers > kAttnResBlock, "not enough layers for one block");
static_assert(kTopK > 0 && kTopK <= kExperts, "bad top-k");
static_assert(kLatentDim > 0 && kMoeIntermediate > 0, "bad latent MoE widths");
static_assert((kMoeIntermediate * kSharedExperts) == kSharedIntermediate,
              "shared expert width is 2 x moe intermediate");
static_assert(kBaseVocab + 256 == kVocab, "256 reserved special slots");
static_assert(kBosId >= kBaseVocab && kEosId >= kBaseVocab,
              "special ids live in the reserved band");
static_assert(sizeof(float) == 4, "FP32 assumptions");

}  // namespace frozen
}  // namespace lsxcommon::kimi_k3
