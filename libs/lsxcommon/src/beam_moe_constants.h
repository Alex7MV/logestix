#pragma once

// Beam-1.0-501B-A23B (Reflection) frozen passport.
//
// `BeamForCausalLM` is a decoder-only sparse MoE built from a Qwen3-style GQA
// contour with five deltas that exist in no ingested family:
//
//   1. ATTENTION OUTPUT GATE: `self_attn.gate_proj` [NH*HD,H] plus one
//      `self_attn.gate_bias` [NH]; `attn_out *= sigmoid(gate + bias[head])`
//      BEFORE o_proj.
//   2. SANDWICH NORMS + DEPTH-SCALED RESIDUAL: four norms per layer
//      (input / post-attention / pre-ffn / post-ffn) and the combine
//      `branch/sqrt(idx) + residual*sqrt((idx-1)/idx)`, idx = 2*(layer+1)
//      for attention and idx+1 for the MLP.
//   3. HYBRID ATTENTION: `layer_types` is sliding_attention except every
//      sixth layer and the last one; sliding layers use rotate-half RoPE
//      (theta 10000, full head_dim), FULL layers run NoPE with an SSMax
//      query scale (1 + (pos+1)/2048)^0.1.
//   4. GROUP-LIMITED SIGMOID ROUTER: FP32 logits, sigmoid scores,
//      `e_score_correction_bias` for CHOICE, 8 groups -> top-4 by the sum of
//      their two best corrected scores, top-8 by corrected score, weights are
//      the PLAIN sigmoid scores renormalized, and routed_scaling_factor 2.5 is
//      applied AFTER expert aggregation.
//   5. FINAL LOGIT SOFTCAP 30 (`logits/30 -> tanh -> *30`, FP32).
//
// Verified against the source checkpoint on 2026-10-31 (config.json + 188-shard
// index, 38 366 tensors, 934 GB BF16):
//   attention      : q/o [6144,5120], k/v [1024,5120], gate_proj [6144,5120],
//                    gate_bias [48], q_norm/k_norm [128]
//   dense MLP (0-2): gate/up [20480,5120], down [5120,20480]
//   router         : mlp.gate.weight [256,5120] F32, correction bias [256] F32
//   routed experts : 49 layers x 256 experts x {gate,up [2560,5120],
//                    down [5120,2560]}, BF16
//   embed / lm_head: [151936,5120], NOT tied
//   ballast        : none (no vision tower, no MTP, no shared expert)
//
// Semantics (candidate silently-wrong-token sources, each pinned by tests):
//   - attention scale 1/sqrt(128) at QK; gate applies per (head, dim)
//   - q_norm / k_norm are per-head plain RMSNorm (weight form), before RoPE
//   - sliding window 512 INCLUDING the current token
//   - full-attention layers carry NO rotary, only the SSMax query scale
//   - SwiGLU activation/product evaluated in FP32 then rounded
//   - residual stream FP16 with FP32 norm accumulation
#include <cstdint>

namespace lsxcommon {
namespace beam {
namespace frozen {

inline constexpr char kType[] = "beam";
inline constexpr char kFormatTag[] = "logestix-beam-nvfp4";
inline constexpr char kModelName[] = "Beam-1.0-501B-A23B";

// ── Frozen backbone geometry ────────────────────────────────────────────────
inline constexpr int kHidden = 5120;
inline constexpr int kLayers = 52;
inline constexpr int kHeads = 48;
inline constexpr int kKvHeads = 8;
inline constexpr int kHeadDim = 128;
inline constexpr int kVocab = 151936;

inline constexpr int kFirstMoeLayer = 3;        // mlp_layer_types 0..2 = dense
inline constexpr int kDenseIntermediate = 20480;
inline constexpr int kMoeIntermediate = 2560;
inline constexpr int kNumExperts = 256;
inline constexpr int kTopK = 8;
inline constexpr int kNGroups = 8;
inline constexpr int kTopkGroup = 4;

inline constexpr int kWindow = 512;             // current + 511 previous

inline constexpr int kNumFullAttnLayers = 9;
inline constexpr int kFullAttnLayers[kNumFullAttnLayers] = {
    5, 11, 17, 23, 29, 35, 41, 47, 51};

// ── Tokenizer / Harmony protocol ids (tokenizer.json added_tokens) ──────────
inline constexpr int kPadId = 151643;           // <|endoftext|>
inline constexpr int kBosId = -1;               // bos_token is null
inline constexpr int kEosId = 151674;           // <|return|>
inline constexpr int kEosAltId = 151675;        // <|call|>
inline constexpr int kStartId = 151669;         // <|start|>      (plain)
inline constexpr int kEndId = 151670;           // <|end|>        (plain)
inline constexpr int kMessageId = 151671;       // <|message|>    (plain)
inline constexpr int kChannelId = 151672;       // <|channel|>    (plain)
inline constexpr int kConstrainId = 151673;     // <|constrain|>  (plain)
inline constexpr int kParamOpenId = 151679;     // <rfl:parameter name=" (plain)
inline constexpr int kParamCloseId = 151680;    // \n</rfl:parameter> (plain)

// Numeric constants.
inline constexpr float kRmsEps = 1e-6f;
inline constexpr float kRopeTheta = 10000.0f;
inline constexpr float kAttnScale = 0.08838834764831845f;  // 1/sqrt(128)
inline constexpr float kRoutedScale = 2.5f;
inline constexpr float kSoftcap = 30.0f;
inline constexpr float kSsmAlpha = 0.1f;
inline constexpr float kSsmBase = 2048.0f;

constexpr bool IsFullAttn(int layer)
{
    for (int i = 0; i < kNumFullAttnLayers; ++i)
        if (kFullAttnLayers[i] == layer) return true;
    return false;
}

// Full-attention layers are the NoPE layers (the config maps full_attention
// to a null rope_parameters entry).
constexpr bool SkipsRope(int layer) { return IsFullAttn(layer); }

constexpr bool IsMoeLayer(int layer)
{
    return layer >= kFirstMoeLayer && layer < kLayers;
}

constexpr bool IsDenseLayer(int layer)
{
    return layer >= 0 && layer < kFirstMoeLayer;
}

constexpr bool IsEosId(int id) { return id == kEosId || id == kEosAltId; }

// ── Compile-time relations (MODEL INGEST INVARIANT step 3) ──────────────────
static_assert(kLayers == 52, "beam backbone is 52 layers");
static_assert(kHidden == 5120, "beam hidden is 5120");
static_assert(kHidden % 256 == 0, "hidden tiles cleanly on Blackwell");
static_assert(kMoeIntermediate % 256 == 0, "moe intermediate tiles cleanly");
static_assert(kDenseIntermediate % 256 == 0, "dense intermediate tiles cleanly");
static_assert(kHeads * kHeadDim == 6144, "q/o/gate projection width");
static_assert(kKvHeads * kHeadDim == 1024, "k/v projection width");
static_assert(kHeads % kKvHeads == 0, "GQA groups");
static_assert(kHeadDim % 2 == 0, "half-split rotary pairing");
static_assert(kTopK > 0 && kTopK <= kNumExperts, "bad top-k");
static_assert(kTopkGroup > 0 && kTopkGroup <= kNGroups, "bad topk_group");
static_assert(kNumExperts % kNGroups == 0, "expert groups are even");
static_assert(kFirstMoeLayer == 3, "dense prefix is layers 0..2");
static_assert(kWindow > 1, "sliding window includes the current token");
static_assert(kNumFullAttnLayers > 0 && kNumFullAttnLayers < kLayers,
              "hybrid attention needs both layer kinds");
static_assert(kVocab == 151936, "beam vocabulary");
static_assert(kPadId < kVocab && kEosId < kVocab, "specials inside the vocab");

} // namespace frozen
} // namespace beam
} // namespace lsxcommon
