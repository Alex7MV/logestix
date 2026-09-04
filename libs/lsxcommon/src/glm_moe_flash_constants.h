#pragma once

// Frozen GLM-5.3-Flash geometry (compile-time passport).
// Source of truth: /mnt/llm_models/models/GLM-5.3-Flash/config.json
// (text_config, model_type glm5_next) + safetensors inventory audit
// (2026-09-04): 62 shards, 306 GB, FP8_E4M3 blockwise 128x128 with F32
// weight_scale_inv. 45 main layers (34 KDA linear_attention + 11 DSA
// deepseek_sparse_attention) + MTP nextn layer 45 (store-not-apply).
// MoE 288 experts top-8 sigmoid noaux_tc, hc_mult=4, first-3-dense MLP.

#include <cstdint>
#include <string>

namespace lsxcommon::glm_moe_flash {
namespace frozen {

inline constexpr int kHidden = 4096;
inline constexpr int kLayers = 45;      // main layers 0..44
inline constexpr int kMtpLayer = 45;    // nextn: stored, not applied
inline constexpr int kVocab = 154880;
inline constexpr int kHeads = 64;
inline constexpr int kHeadDim = 128;    // KDA head dim
inline constexpr int kQkHeadDim = 256;  // DSA MLA qk/v head dim
inline constexpr int kQLora = 1536;
inline constexpr int kKvLora = 512;
inline constexpr int kConvKernel = 4;
inline constexpr int kGateLowrank = 128;
inline constexpr int kFirstKDense = 3;  // dense MLP layers 0..2
inline constexpr int kNumExperts = 288;
inline constexpr int kTopK = 8;
inline constexpr int kMoeIntermediate = 2048;
inline constexpr int kDenseIntermediate = 12288;
inline constexpr int kHcCount = 4;
inline constexpr int kHcFnRows = 24;    // hc_attn_fn/ffn_fn rows [24, hc*H]
inline constexpr int kIndexerHeads = 32;
inline constexpr int kIndexerHeadDim = 128;
inline constexpr int kIndexerTopk = 2048;
inline constexpr int kIndexerKpool = 4;
inline constexpr int kMoeFirstLayer = 3;

// routed_scaling_factor 2.5 — exact rational to avoid fp parsing drift.
inline constexpr int kRoutedScalingNum = 25;
inline constexpr int kRoutedScalingDen = 10;

// Layer-type tables (config layer_types / linear_attn_config).
inline constexpr int kKdaLayerIds[34] = {
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24,
    25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38, 40, 41, 42, 44};
inline constexpr int kDsaLayerIds[11] = {
    3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43};

inline constexpr int kKdaLayerCount = 34;
inline constexpr int kDsaLayerCount = 11;

inline constexpr bool IsKdaLayer(int l)
{
    for (int i = 0; i < kKdaLayerCount; ++i)
        if (kKdaLayerIds[i] == l) return true;
    return false;
}

inline constexpr bool IsDsaLayer(int l)
{
    for (int i = 0; i < kDsaLayerCount; ++i)
        if (kDsaLayerIds[i] == l) return true;
    return false;
}

inline constexpr bool IsMoeLayer(int l)
{
    return l >= kMoeFirstLayer && l <= kMtpLayer;
}

} // namespace frozen

// Tensor kinds of the GLM-5.3-Flash (glm5_next) checkpoint.
enum class FlashTensorKind {
    Unknown,
    Embed,        // model.language_model.embed_tokens.weight
    LmHead,       // lm_head.weight
    FinalNorm,    // model.language_model.norm.weight
    // KDA (linear_attention) layer tensors — high precision.
    KdaQ,         // self_attn.q_proj.weight [8192, H]
    KdaK,         // self_attn.k_proj.weight
    KdaV,         // self_attn.v_proj.weight
    KdaO,         // self_attn.o_proj.weight [H, 8192]
    KdaConv,      // self_attn.{q,k,v}_conv1d.weight [8192,1,4]
    KdaALog,      // self_attn.A_log F32[64]
    KdaDtBias,    // self_attn.dt_bias F32[8192]
    KdaB,         // self_attn.b_proj.weight [64, H]
    KdaFa,        // self_attn.f_a_proj.weight [128, H]
    KdaFb,        // self_attn.f_b_proj.weight [8192, 128]
    KdaGa,        // self_attn.g_a_proj.weight [128, H]
    KdaGb,        // self_attn.g_b_proj.weight [8192, 128]
    KdaONorm,     // self_attn.o_norm.weight [128]
    // DSA (deepseek_sparse_attention) layer tensors.
    DsaQa,        // self_attn.q_a_proj.weight (FP8)
    DsaQb,        // self_attn.q_b_proj.weight (FP8)
    DsaKvA,       // self_attn.kv_a_proj_with_mqa.weight (FP8)
    DsaKvB,       // self_attn.kv_b_proj.weight BF16 (fused, not quantized)
    DsaOBig,      // self_attn.o_proj.weight (FP8, [H, heads*qk])
    Indexer,      // self_attn.indexer.* (BF16)
    DsaNorms,     // q_a_layernorm / kv_a_layernorm / input/post norms
    // Hyper-connections per layer.
    HcAttnBase,   // hc_attn_base F32[24]
    HcAttnFn,     // hc_attn_fn BF16[24, 16384]
    HcAttnScale,  // hc_attn_scale F32[3]
    HcFfnBase,
    HcFfnFn,
    HcFfnScale,
    // MoE.
    RoutedExpert,     // mlp.experts.E.{gate,up,down}_proj.weight (FP8)
    SharedExpert,     // mlp.shared_experts.* (FP8)
    RouterGate,       // mlp.gate.weight BF16[288, H]
    RouterBias,       // mlp.gate.e_score_correction_bias F32[288]
    DenseMlp,         // mlp.{gate,up,down}_proj.weight layers 0..2 (FP8)
    // MTP (store-not-apply) + visual.
    MtpEhProj,        // layers.45.eh_proj / enorm / hnorm / shared_head.norm
    Visual,           // model.visual.* / visual.* — skipped + counted
};

struct ParsedFlashTensor {
    FlashTensorKind kind = FlashTensorKind::Unknown;
    int layer = -1;
    int expert = -1;
    int proj_id = 0;  // {gate,down,up}_proj -> {1,2,3}
};

// Classifies an HF glm5_next tensor name (model.language_model prefix
// optional; both "model.layers.N.*" and "model.language_model.layers.N.*"
// accepted).
ParsedFlashTensor classify_flash_tensor(const std::string &name);

// True when the kind is NVFP4-quantized by the converter (FP8 source
// matrices). Norms, conv1d, A_log/dt_bias, hc weights, router gate,
// kv_b_proj (fused BF16), indexer, embed, lm_head stay high precision.
bool flash_tensor_is_nvfp4(FlashTensorKind kind);

} // namespace lsxcommon::glm_moe_flash
