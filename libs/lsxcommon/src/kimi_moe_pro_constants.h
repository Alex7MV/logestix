#pragma once

// Frozen Kimi-K3.1-Pro geometry (compile-time passport).
// Source of truth: /mnt/fast_storage/lsx_model/moonshotai/Kimi-K3.1-Pro (config.json +
// configuration_kimi_k3.py + safetensors headers, audited 2026-11-05).
//
// Kimi-K3.1-Pro is a SIBLING VARIANT of the `kimi` family, not a new family:
// the text tower is the Kimi-K3 KimiLinear contour (93 layers, hidden 7168,
// 69 KDA linear-attention + 24 gated-MLA layers at the 0-based layers
// {3,7,...,91,92}, AttnRes block 12, 896 routed experts top-16 with a 3584
// latent and 2 shared experts, SiTU-GLU, weight-form RMSNorm eps 1e-5,
// vocab 163840, XTML framing) with ONE delta: the 24 full-attention layers are
// Gated DSA -- the same gated NoPE MLA plus a DeepSeek-style Lightning Indexer
// that selects `kIndexTopK` keys per query.
//
// The checkpoint ships NO modeling code (`modeling_kimi_k31.py` is absent), so
// the indexer arithmetic is reconstructed from the tensor shapes, `dsa_config`
// and the Lightning-Indexer prior art already in this repository
// (`deepseek_moe_pro` CSA, `mistral_moe_large4` DSA). That residual risk is
// recorded in `docs/OPEN_TASKS.md` §R17, not claimed as a parity result.
//
// Measured delta against Kimi-K3/model.safetensors.index.json: 497 220 of
// 497 340 tensor names are identical; +120 = 24 DSA layers x 5 indexer tensors
// (all BF16, never quantized -- they decide *which* keys are visible):
//   self_attn.indexer.wq_b.weight       [4096, 1536]  (32 heads x 128 from q-lora)
//   self_attn.indexer.wk.weight         [128, 7168]   (index_head_dim from hidden)
//   self_attn.indexer.k_norm.weight     [128]
//   self_attn.indexer.k_norm.bias       [128]         (LayerNorm, not RMSNorm)
//   self_attn.indexer.weights_proj.weight [32, 7168]
//
// Self-contained BY DESIGN: the independence guard forbids a passport header
// including another passport header, so the K3 geometry is restated here and
// `tests/test_kimi_pro_constants.cpp` pins every restated value.

#include <array>

namespace lsxcommon::kimi_pro {
namespace frozen {

inline constexpr char kModelType[] = "kimi-k3.1-pro";
inline constexpr char kFormatTag[] = "logestix-kimi-k31-pro-nvfp4";
inline constexpr char kSourceModelType[] = "kimi_k31";  // config.text_config.model_type
inline constexpr int kTokenizerKind = 2;  // tiktoken BPE (Kimi pattern)

// ── text tower ────────────────────────────────────────────────────────────
inline constexpr int kHidden = 7168;
inline constexpr int kLayers = 93;
inline constexpr int kDenseLayers = 1;  // layer 0: dense SiTU MLP
inline constexpr int kAttnResBlock = 12;
inline constexpr int kFullAttnInterval = 4;  // 3 KDA -> 1 full attention
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

// ── Gated DSA: every full-attention layer carries the indexer ─────────────
// config `dsa_config.dsa_layers` is 1-based {4,8,...,92,93}; the 0-based set is
// exactly the MLA table, i.e. Kimi-K3.1-Pro indexes EVERY full-attention layer.
inline constexpr int kDsaLayerCount = kMlaLayerCount;
inline constexpr std::array<int, kDsaLayerCount> kDsaLayers = kMlaLayers;

constexpr bool IsDsaLayer(int layer) { return IsMlaLayer(layer); }

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
// The released checkpoint stores A_log with head_dim (128) entries, not one per
// head, so the decay is channel-wise and shared across heads.
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

// ── Lightning Indexer (Gated DSA) ─────────────────────────────────────────
// dsa_config: index_topk 2048, index_head_dim 128, index_num_heads 32,
// index_rope_dim 0, index_k_norm_type "layernorm", index_topk_freq 1.
inline constexpr int kIndexHeads = 32;
inline constexpr int kIndexHeadDim = 128;
inline constexpr int kIndexTopK = 2048;
inline constexpr int kIndexRopeDim = 0;      // no rotary on the indexer q/k
inline constexpr int kIndexQLora = kQLoraRank;    // wq_b reads the normalised q-lora
inline constexpr int kIndexKvLen = kHidden;       // wk reads the attention input
inline constexpr int kIndexWidth = kIndexHeads * kIndexHeadDim;  // wq_b rows
inline constexpr int kIndexWeightsRows = kIndexHeads;            // weights_proj rows
inline constexpr bool kIndexNormHasBias = true;   // LayerNorm, weight + bias
inline constexpr double kIndexNormEps = 1e-5;
inline constexpr bool kIndexPerPositionKeys = true;  // the key is cached per position
// Derived scales of the Lightning Indexer: the query is pre-scaled by
// index_head_dim^-0.5 and the per-head score weight by index_num_heads^-0.5.
inline constexpr float kIndexQScale = 0.08838834764831845f;       // 128^-0.5
inline constexpr float kIndexWeightScale = 0.17677669529663687f;  // 32^-0.5

// ── frozen-geometry invariant checks ──────────────────────────────────────
static_assert(kLayers == 93, "Kimi-K3.1-Pro has 93 decoder layers");
static_assert(kMlaLayerCount + kKdaLayerCount == kLayers,
              "hybrid layer tables must partition the stack");
static_assert(kMlaLayerCount == 24 && kKdaLayerCount == 69,
              "24 gated-MLA + 69 KDA layers");
static_assert(kDsaLayerCount == kMlaLayerCount,
              "Kimi-K3.1-Pro indexes every full-attention layer");
static_assert(kDsaLayers.back() == kLayers - 1,
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
static_assert(kIndexHeads * kIndexHeadDim == 4096, "wq_b row count");
static_assert(kIndexTopK == 2048, "dsa_config.index_topk");
static_assert(kIndexRopeDim == 0, "the indexer has no rotary");
static_assert(kIndexQLora == kQLoraRank, "wq_b reads the q-lora vector");
static_assert(kIndexKvLen == kHidden, "wk reads the attention input");
static_assert(kIndexTopK % 128 == 0, "the top-k radix pass assumes a block size");
static_assert(sizeof(float) == 4, "FP32 assumptions");

}  // namespace frozen
}  // namespace lsxcommon::kimi_pro
