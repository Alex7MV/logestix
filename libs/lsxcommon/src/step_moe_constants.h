#pragma once

// Step-5-Preview (checkpoint "stepfun-ai/Step-5-Preview") frozen passport.
//
// Stage 0 triage: `model_type = "step5_vl"`, text backbone `Step5ForCausalLM`
// (config.json `text_config`, model_type "step5") — a 600B-total / 27B-active
// sparse-MoE decoder:
//
//   attention   GQA, 64 Q heads / 4 KV heads / head_dim 192, per-head QK
//               ZERO-CENTERED RMSNorm ((1+w), F32 weights), NEOX-paired
//               PARTIAL RoPE, per-head
//               sigmoid output gate (self_attn.g_proj). Hybrid schedule from
//               `layer_types`: 3 sliding layers (window 512, full 192-dim RoPE,
//               theta 1e4) then 1 full-attention layer (64-dim RoPE,
//               theta 1e7) — every 4th layer is full. The full-attention
//               layers are the DSA layers (`sparse_config.apply_to_layer_types`
//               == ["full_attention"]): a Lightning-Indexer top-512 region
//               selection over FP8 region summaries (region 8, proxy 256).
//   FFN         layers 0..2 dense SwiGLU (I=13824); layers 3..91 routed MoE
//               (352 experts, top-8, I=1536) + one gated shared expert
//               (I=1536). SwiGLU clamps: min(silu(gate), 7) * clamp(up,-7,7).
//   router      F32 sigmoid logits from a BF16 gate table + a selection-only
//               bias: select top-k of (sigmoid + bias), coefficient =
//               (score - bias) renormalized to sum 1, times 3.0.
//   residual    FP32 stream (`fp32_residual_connection`), zero-centered norms
//               ((1+w), F32),
//               matmuls in BF16; the final RMSNorm narrows the FP32 residual
//               to BF16 first (part of the reference arithmetic).
//
// Geometry drift from every existing family (92 layers, 4096 hidden, 64/4 GQA
// at head_dim 192, 352 fused experts at I=1536, hybrid sliding+full, DSA
// indexer, FP32 residual) makes this a NEW family (Model Ingestion Protocol
// stage 0.1), so the numbers are locked here.
//
// Verified against the source checkpoint on 2026-10-31 (config.json +
// model.safetensors.index.json + 24 shards + the reference inference/model.py):
//   routed experts : 88 layers x 352 experts, FUSED on disk as
//                    moe.{gate,up}_proj.weight [352,1536,4096] and
//                    moe.down_proj.weight [352,4096,1536], F8_E4M3 with a
//                    moe.*.weight_scale_inv [352,ceil(N/128),ceil(K/128)] F32
//                    block grid (128x128). weight = e4m3(w) * scale.
//   shared expert  : 88 layers x {gate,up,down} at [1536,4096]/[4096,1536],
//                    BF16 (not FP8).
//   dense prefix   : layers 0..2 x {gate,up,down} at 13824, BF16.
//   attention      : q_proj [12288,4096], k/v [768,4096], o_proj [4096,12288],
//                    g_proj [64,4096], q_norm/k_norm [192] — all BF16, norms
//                    F32. The F32 `norm_dtype` covers input/post layernorm and
//                    the per-head QK norms.
//   indexer        : full-attention layers only, `sparse_indexer_{q,k,z,w}`
//                    (q [4096,4096], k/z [256,4096], w [16,4096] F32), q_norm
//                    [256], k_norm [256] + bias, ssmax_s [64] F32.
//   router         : moe.gate.weight [352,4096] BF16, moe.router_bias [352] F32.
//   embed / lm_head: BF16 [128896,4096], NOT tied (different shards/bytes).
//   ballast        : vision tower (`vision_model.*`, `vit_large_projector.*`,
//                    `vit.safetensors`) and 3 MTP layers (indices 92..94:
//                    enorm/hnorm/eh_proj/transformer.shared_head.*) — stored
//                    and COUNTED, never applied. This engine implements the
//                    92-layer backbone only.
//
// Semantics pinned by this passport (each was a candidate for a silently wrong
// token):
//   - attention scale 1/sqrt(192), applied at QK only
//   - partial rotary: span = round(192 * partial_rotary_factors[layer]); the
//     span is HALVED for the table (pair (d, d+span/2), NEOX). The
//     `rope_scaling` llama3 block (factor 1.0) is the identity, so it is not
//     applied (documented in the model report).
//   - the indexer does not affect attention values; at <= topk*region_size
//     (4096) tokens every region is selected and the sparse path is the dense
//     causal path. This engine runs the dense path and stores the indexer
//     weights for the >4096 DSA stage (OPEN_TASKS §R9).
//   - routed coefficients are renormalized from (sigmoid+bias-bias) and scaled
//     by 3.0; the gated shared expert is added in FP32.
//   - no final logit soft cap and no logit temperature at the model level.
#include <cstdint>

namespace lsxcommon {
namespace step {
namespace frozen {

// Registry id (LsxModelFactory::Register / model.index.json "model_type").
inline constexpr char kType[] = "step-5-preview";
inline constexpr char kFormatTag[] = "logestix-step5-nvfp4";
inline constexpr char kModelName[] = "Step-5-Preview";

// ── Frozen backbone geometry ────────────────────────────────────────────────
inline constexpr int kHidden = 4096;
inline constexpr int kLayers = 92;
inline constexpr int kHeads = 64;
inline constexpr int kKvHeads = 4;
inline constexpr int kHeadDim = 192;
inline constexpr int kVocab = 128896;

inline constexpr int kDenseIntermediate = 13824;
inline constexpr int kMoeIntermediate = 1536;
inline constexpr int kSharedIntermediate = 1536;
inline constexpr int kNumExperts = 352;
inline constexpr int kTopK = 8;
// `moe_layer_list` is 3..90: layers 0..2 form the dense prefix and layer 91 is
// a DENSE-MLP full-attention layer (the MoE range is not "the tail").
inline constexpr int kFirstMoeLayer = 3;
inline constexpr int kMoeLayerEnd = 91;
inline constexpr int kNumMoeLayers = kMoeLayerEnd - kFirstMoeLayer;  // 88
// The neutral ExpertStore only knows a contiguous dense PREFIX; the flat
// expert index is layer*experts+e, so bounding it at the last MoE layer + 1
// keeps the arithmetic intact and avoids probing layer 91's missing files.
inline constexpr int kExpertStoreLayers = kMoeLayerEnd;

// Hybrid schedule: layers 3, 7, 11, ... 91 are full attention (23 of them),
// every other layer is sliding-window attention. The DSA indexer runs on the
// full-attention set.
inline constexpr int kFullAttentionPeriod = 4;
inline constexpr int kFullAttentionPhase = 3;
inline constexpr int kSlidingWindow = 512;
inline constexpr int kNumFullAttnLayers = 23;

// Rotary: sliding layers rotate the full head (192) at theta 1e4; full layers
// rotate a third of it (64) at theta 1e7.
inline constexpr int kRotarySpanSliding = kHeadDim;   // 192
inline constexpr int kRotarySpanFull = 64;            // 192 * 1/3, even
inline constexpr float kRopeThetaSliding = 10000.0f;
inline constexpr float kRopeThetaFull = 10000000.0f;

// DSA (stored-not-applied until the >4096-token stage).
inline constexpr int kIndexProxyDim = 256;
inline constexpr int kIndexHeads = 16;
inline constexpr int kIndexTopk = 512;
inline constexpr int kIndexRegion = 8;
inline constexpr int kIndexRopeDim = 32;
// Every region is selected at or below topk*region; above it the dense path
// this engine implements stops being the reference path.
inline constexpr int kMaxContextTokens = kIndexTopk * kIndexRegion;  // 4096

// MTP layers (92..94) are stored-not-applied.
inline constexpr int kMtpLayers = 3;
inline constexpr int kSourceLayersWithMtp = kLayers + kMtpLayers;

// Tokenizer specials (tokenizer.json added_tokens; config.json text_config).
inline constexpr int kBosId = 0;        // <｜begin▁of▁sentence｜>
inline constexpr int kEosId = 1;        // <｜end▁of▁sentence｜>
inline constexpr int kEosId2 = 2;       // <｜▁pad▁｜> (the config lists [1, 2])
inline constexpr int kPadId = 2;
inline constexpr int kImStartId = 128006;      // <|im_start|>  (special)
inline constexpr int kImEndId = 128007;        // <|im_end|>    (special)
inline constexpr int kImStartRawId = 128000;   // <im_start>    (special)
inline constexpr int kImEndRawId = 128002;     // <im_end>      (special)
inline constexpr int kImPatchId = 128001;      // <im_patch>    (special)
inline constexpr int kToolCallId = 128008;     // <tool_call>      (special:false)
inline constexpr int kToolCallEndId = 128009;  // </tool_call>     (special:false)
inline constexpr int kToolResponseId = 128010;  // <tool_response> (special:false)
inline constexpr int kToolResponseEndId = 128011;  // </tool_response>
inline constexpr int kToolCallsId = 128012;      // <tool_calls>   (special)
inline constexpr int kToolCallsEndId = 128013;   // </tool_calls>  (special)

// Numeric constants.
inline constexpr float kRmsEps = 1e-5f;
inline constexpr float kRoutedScale = 3.0f;
inline constexpr float kSwigluLimit = 7.0f;
inline constexpr float kAttnScale = 0.07216878364870322f;  // 1/sqrt(192)

inline constexpr int kIndexerLayers = kNumFullAttnLayers;

// ── Compile-time relations (MODEL INGEST INVARIANT step 3) ─────────────────
static_assert(kLayers == 92, "Step-5-Preview backbone is 92 layers");
static_assert(kHidden == 4096, "Step-5-Preview hidden is 4096");
static_assert(kHidden % 256 == 0, "hidden tiles cleanly on Blackwell");
static_assert(kMoeIntermediate % 128 == 0, "moe intermediate tiles cleanly");
static_assert(kDenseIntermediate % 128 == 0, "dense intermediate tiles cleanly");
static_assert(kSharedIntermediate % 128 == 0, "shared expert tiles cleanly");
static_assert(kHeads * kHeadDim == 12288, "q/o projection width");
static_assert(kKvHeads * kHeadDim == 768, "k/v projection width");
static_assert(kHeads % kKvHeads == 0, "GQA groups");
static_assert(kHeads * kHeadDim % 16 == 0, "NVFP4 activation blocks");
static_assert(kHeadDim % 2 == 0, "half-split rotary pairing");
static_assert(kRotarySpanSliding == kHeadDim, "sliding layers rotate the head");
static_assert(kRotarySpanFull % 2 == 0, "full layers rotate an even span");
static_assert(kTopK > 0 && kTopK <= kNumExperts, "bad top-k");
static_assert(kFirstMoeLayer > 0 && kFirstMoeLayer < kLayers, "dense prefix");
static_assert(kMoeLayerEnd < kLayers, "layer 91 is dense");
static_assert(kNumMoeLayers == 88, "88 routed-MoE layers");
static_assert(kMaxContextTokens == 4096, "dense-equivalence bound");
static_assert(kBosId < kVocab && kEosId < kVocab && kEosId2 < kVocab,
              "tokenizer specials inside the vocabulary");

constexpr bool IsFullAttention(int layer)
{
    return layer >= kFullAttentionPhase &&
           (layer - kFullAttentionPhase) % kFullAttentionPeriod == 0;
}

constexpr bool IsSlidingAttention(int layer)
{
    return layer >= 0 && layer < kLayers && !IsFullAttention(layer);
}

constexpr bool IsMoeLayer(int layer)
{
    return layer >= kFirstMoeLayer && layer < kMoeLayerEnd;
}

constexpr int RotarySpan(int layer)
{
    return IsFullAttention(layer) ? kRotarySpanFull : kRotarySpanSliding;
}

constexpr float RopeTheta(int layer)
{
    return IsFullAttention(layer) ? kRopeThetaFull : kRopeThetaSliding;
}

constexpr int CountFullAttentionLayers()
{
    int n = 0;
    for (int l = 0; l < kLayers; ++l)
        if (IsFullAttention(l)) ++n;
    return n;
}

static_assert(CountFullAttentionLayers() == kNumFullAttnLayers,
              "full-attention layer count");
static_assert(IsFullAttention(3) && IsFullAttention(91) && !IsFullAttention(4),
              "every fourth layer from 3");
static_assert(!IsMoeLayer(kMoeLayerEnd), "the expert store bound excludes 91");
static_assert(IsMoeLayer(kMoeLayerEnd - 1), "last MoE layer is 90");
static_assert(IsMoeLayer(3) && !IsMoeLayer(2), "first MoE layer is 3");

} // namespace frozen
} // namespace step
} // namespace lsxcommon
