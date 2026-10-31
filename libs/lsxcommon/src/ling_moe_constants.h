#pragma once

// Frozen Ling-3.1-flash geometry (compile-time passport).
// Source of truth: /mnt/fast_storage/huggingface/Ling-3.1-flash/config.json
// (BailingMoeV3ForCausalLM, model_type bailing_hybrid) + the safetensors
// inventory (2026-10-30): 54 shards, 535 GB, BF16 edges and F8_E4M3
// block-128 linears with F32 weight_scale_inv.
// 72 main layers (63 KDA + 9 gated MLA) + MTP layer 72 (store-not-apply),
// 68 MoE layers (4 dense-MLP prefix), 512 experts top-8, one shared expert.

#include <cstdint>
#include <string>

namespace lsxcommon::ling {
namespace frozen {

inline constexpr const char *kType = "ling";
inline constexpr const char *kModelName = "Ling-3.1-flash";
inline constexpr const char *kFormatTag = "logestix-ling-nvfp4";

inline constexpr int kHidden = 4096;
inline constexpr int kLayers = 72;    // main layers 0..71
inline constexpr int kMtpLayer = 72;  // nextn: stored, not applied
inline constexpr int kVocab = 157184;
inline constexpr int kHeads = 64;
inline constexpr int kKvHeads = 64;
inline constexpr int kHeadDim = 128;  // KDA head dim / MLA v head dim
inline constexpr int kQkHeadDim = 192;
inline constexpr int kQkNope = 128;
inline constexpr int kQkRope = 64;
inline constexpr int kVHead = 128;
inline constexpr int kQLora = 1536;
inline constexpr int kKvLora = 512;
inline constexpr int kConvKernel = 4;
inline constexpr int kNumExperts = 512;
inline constexpr int kTopK = 8;
inline constexpr int kMoeIntermediate = 1280;
inline constexpr int kSharedIntermediate = 1280;
inline constexpr int kDenseIntermediate = 10240;
inline constexpr int kFirstKDense = 4;
inline constexpr int kMoeFirstLayer = 4;
inline constexpr int kGroupCount = 8;
inline constexpr int kTopkGroup = 4;
inline constexpr int kLayerGroupSize = 8;

inline constexpr double kRmsEps = 1e-6;
inline constexpr double kRopeTheta = 6000000.0;

// routed_scaling_factor 2.5 — exact rational to avoid fp parsing drift.
inline constexpr int kRoutedScalingNum = 25;
inline constexpr int kRoutedScalingDen = 10;

inline constexpr int kBosId = -1;
inline constexpr int kEosId = 156895;
inline constexpr int kPadId = 156892;

// (layer + 1) % 8 == 0 → gated MLA; every other main layer is KDA.
inline constexpr int kMlaLayerIds[9] = {7, 15, 23, 31, 39, 47, 55, 63, 71};
inline constexpr int kMlaLayerCount = 9;
inline constexpr int kKdaLayerCount = 63;
inline constexpr int kMoeLayerCount = 68;

static_assert(kMlaLayerCount + kKdaLayerCount == kLayers,
              "MLA + KDA layer tables must cover all main layers");
static_assert(kMlaLayerCount ==
                  sizeof(kMlaLayerIds) / sizeof(kMlaLayerIds[0]),
              "MLA layer table length mismatch");
static_assert(kMlaLayerCount * kLayerGroupSize == kLayers,
              "layer_group_size must divide num_hidden_layers");
static_assert(kMtpLayer == kLayers, "MTP layer follows the last main layer");
static_assert(kMoeFirstLayer == kFirstKDense,
              "MoE starts where the dense-MLP prefix ends");
static_assert(kMoeLayerCount == kLayers - kFirstKDense,
              "MoE layer count mismatch");
static_assert(kTopK <= kNumExperts, "top-k exceeds expert count");
static_assert(kGroupCount > 0 && kTopkGroup > 0 &&
                  kTopkGroup <= kGroupCount &&
                  kNumExperts % kGroupCount == 0,
              "group-limited top-k geometry");
static_assert(kQkNope + kQkRope == kQkHeadDim, "qk head dim split");
static_assert(kHidden % 128 == 0, "hidden must tile by 128");
static_assert(kMoeIntermediate % 64 == 0 && kSharedIntermediate % 64 == 0,
              "expert intermediate must tile by 64");
static_assert(kRoutedScalingDen != 0, "routed scaling denominator");

inline constexpr bool IsMlaLayer(int l)
{
    for (int i = 0; i < kMlaLayerCount; ++i)
        if (kMlaLayerIds[i] == l) return true;
    return false;
}

inline constexpr bool IsKdaLayer(int l)
{
    return l >= 0 && l < kLayers && !IsMlaLayer(l);
}

inline constexpr bool IsMoeLayer(int l)
{
    return l >= kMoeFirstLayer && l <= kMtpLayer;
}

} // namespace frozen

// Tensor kinds of the Ling/BailingMoeV3 checkpoint.
enum class LingTensorKind {
    Unknown,
    Embed,      // model.word_embeddings.weight
    LmHead,     // lm_head.weight
    FinalNorm,  // model.norm.weight
    InputNorm,
    PostNorm,
    // KDA (linear_attention) layer tensors.
    KdaQ,
    KdaK,
    KdaV,
    KdaO,
    KdaF,        // attention.f_proj (direct, no_kda_lora)
    KdaG,        // attention.g_proj (output gate)
    KdaConv,     // attention.{q,k,v}_conv1d.weight
    KdaALog,
    KdaDtBias,
    KdaB,        // attention.b_proj.weight
    KdaONorm,    // attention.o_norm.weight
    // Gated MLA (full attention) layer tensors.
    MlaQa,
    MlaQb,
    MlaKvA,      // attention.kv_a_proj_with_mqa.weight
    MlaKvB,
    MlaDense,    // attention.dense.weight (o_proj)
    MlaG,        // attention.g_proj.weight (head-wise gate)
    MlaNorm,     // q_a_layernorm / kv_a_layernorm
    // MoE.
    RouterGate,  // mlp.gate.weight BF16
    RouterBias,  // mlp.gate.expert_bias F32
    DenseMlpGate,
    DenseMlpUp,
    DenseMlpDown,
    SharedGate,
    SharedUp,
    SharedDown,
    RoutedGate,
    RoutedDown,
    RoutedUp,
    // Stored-not-applied MTP layer and unknown ballast.
    Mtp,
    Visual,
};

struct ParsedLingTensor {
    LingTensorKind kind = LingTensorKind::Unknown;
    int layer = -1;
    int expert = -1;
    int proj_id = 0;  // {gate,down,up}_proj -> {1,2,3}
};

// Classifies a BailingMoeV3 tensor name ("model." prefix required, as in the
// source checkpoint). Layers >= frozen::kLayers are the MTP block.
ParsedLingTensor classify_ling_tensor(const std::string &name);

// True when the kind is NVFP4-quantized (FP8 source matrix).
bool ling_tensor_is_nvfp4(LingTensorKind kind);

bool ling_tensor_is_routed_expert(LingTensorKind kind);

} // namespace lsxcommon::ling
