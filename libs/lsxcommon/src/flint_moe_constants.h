#pragma once

// RecursiveSI/flint (39.8B total, ~2.5B active) frozen passport.
//
// Stage 0 triage: `FlintForCausalLM` — a decoder-only MoE transformer built from
// the Qwen3-MoE contour (GQA + per-head QK RMSNorm + SwiGLU) plus five deltas
// that exist in no ingested family:
//
//   1. LEARNED ATTENTION SINKS: `self_attn.softmax_offset` [heads], a per-head
//      logit added to the softmax DENOMINATOR with no value contribution.
//   2. HYBRID attention from `config.layer_types`: 35 sliding-window layers
//      (window 513 = current token + 512 previous) and 14 full-attention layers.
//   3. NO POSITIONAL ENCODING on `config.no_rope_layer_indices` — which is
//      exactly the full-attention set. RoPE runs on SWA layers only.
//   4. A dense SwiGLU MLP on `config.mlp_only_layers` (0,1); sparse MoE
//      elsewhere with a sigmoid router (top-6, weights renormalized from the
//      plain sigmoid scores) and ONE ungated shared expert.
//   5. lm_head is separate (tie_word_embeddings false) and served in FP32 by
//      the vendor (`--enable-fp32-lm-head`).
//
// Verified against the source checkpoint on 2026-10-30 (config.json + 15 shard
// inventory, 36 734 tensors, 79.57 GB BF16):
//   attention      : q [4096,2048], k/v [512,2048], o [2048,4096],
//                    q_norm/k_norm [128], softmax_offset [32]
//   dense MLP (0,1): gate/up [6144,2048], down [2048,6144]
//   router         : mlp.gate.weight [256,2048] BF16 (no bias tensor exists)
//   routed experts : 47 layers x 256 experts x {gate,up [512,2048],
//                    down [2048,512]}, BF16
//   shared expert  : 47 layers x {gate,up [1024,2048], down [2048,1024]}
//   embed / lm_head: [150144,2048], NOT tied
//   ballast        : none (no vision tower, no MTP)
//
// Semantics pinned here (each was a candidate for a silently wrong token):
//   - attention scale 1/sqrt(128) at QK only, no value scale
//   - the sink is an extra softmax column: numerator untouched, denominator
//     gains exp(sink - max)
//   - full-attention layers carry NO rotary; SWA layers use rotate-half RoPE
//     over all 128 dims, theta 10000
//   - q_norm / k_norm are per-head plain RMSNorm (weight form), eps 1e-6
//   - routed weights: sigmoid in F32, top-6 by score, renormalize the selected
//     scores (norm_topk_prob true); no routed scale factor
//   - the shared expert is UNGATED and added to the routed sum
//   - residual stream FP16; router + MoE combination F32; 4 stop ids
#include <cstdint>

namespace lsxcommon {
namespace flint {
namespace frozen {

inline constexpr char kType[] = "flint";
inline constexpr char kFormatTag[] = "logestix-flint-nvfp4";
inline constexpr char kModelName[] = "Recursive Flint";

// ── Frozen backbone geometry ────────────────────────────────────────────────
inline constexpr int kHidden = 2048;
inline constexpr int kLayers = 49;
inline constexpr int kHeads = 32;
inline constexpr int kKvHeads = 4;
inline constexpr int kHeadDim = 128;
inline constexpr int kVocab = 150144;

inline constexpr int kFirstMoeLayer = 2;          // mlp_only_layers = 0,1
inline constexpr int kDenseIntermediate = 6144;
inline constexpr int kMoeIntermediate = 512;
inline constexpr int kSharedIntermediate = 1024;
inline constexpr int kNumExperts = 256;
inline constexpr int kTopK = 6;

inline constexpr int kWindow = 513;               // current + 512 previous
inline constexpr int kNumFullAttnLayers = 14;

// The full-attention layer ids, and (separately named because the checkpoint
// carries two lists) the no-RoPE ids. They are identical in this checkpoint;
// the converter rejects any tree where they diverge, because a full layer with
// rotary or an SWA layer without it decodes silently wrong text.
inline constexpr int kFullAttnLayers[kNumFullAttnLayers] = {
    0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48};
inline constexpr int kNoRopeLayers[kNumFullAttnLayers] = {
    0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48};

// Tokenizer / chat protocol ids (tokenizer.json added_tokens).
inline constexpr int kBosId = 150000;             // <|begin_of_text|>
inline constexpr int kPadId = 150001;             // <|end_of_text|> (pad)
inline constexpr int kBeginOfTextId = 150000;
inline constexpr int kEndOfTextId = 150001;
inline constexpr int kSystemTokenId = 150002;
inline constexpr int kUserTokenId = 150003;
inline constexpr int kAssistantTokenId = 150004;
inline constexpr int kObservationTokenId = 150005;
inline constexpr int kThinkBeginId = 150006;      // <think>       (plain)
inline constexpr int kThinkEndId = 150007;        // </think>      (plain)
inline constexpr int kToolsBeginId = 150008;      // <tools>       (plain)
inline constexpr int kToolsEndId = 150009;        // </tools>      (plain)
inline constexpr int kToolCallBeginId = 150010;   // <tool_call>   (plain)
inline constexpr int kToolCallEndId = 150011;     // </tool_call>  (plain)
inline constexpr int kArgKeyBeginId = 150012;     // <arg_key>     (plain)
inline constexpr int kArgKeyEndId = 150013;       // </arg_key>    (plain)
inline constexpr int kArgValueBeginId = 150014;   // <arg_value>   (plain)
inline constexpr int kArgValueEndId = 150015;     // </arg_value>  (plain)
inline constexpr int kToolResponseBeginId = 150016;
inline constexpr int kToolResponseEndId = 150017;
inline constexpr int kDeveloperTokenId = 150018;

// config.json eos_token_id is a LIST; any of them ends the turn.
inline constexpr int kEosIds[4] = {150001, 150003, 150005, 150018};
inline constexpr int kNumEosIds = 4;

// Numeric constants.
inline constexpr float kRmsEps = 1e-6f;
inline constexpr float kRopeTheta = 10000.0f;
inline constexpr float kAttnScale = 0.08838834764831845f;  // 1/sqrt(128)

constexpr bool IsFullAttn(int layer)
{
    for (int i = 0; i < kNumFullAttnLayers; ++i)
        if (kFullAttnLayers[i] == layer) return true;
    return false;
}

constexpr bool SkipsRope(int layer)
{
    for (int i = 0; i < kNumFullAttnLayers; ++i)
        if (kNoRopeLayers[i] == layer) return true;
    return false;
}

constexpr bool IsMoeLayer(int layer)
{
    return layer >= kFirstMoeLayer && layer < kLayers;
}

constexpr bool IsEosId(int id)
{
    for (int i = 0; i < kNumEosIds; ++i)
        if (kEosIds[i] == id) return true;
    return false;
}

// ── Compile-time relations (MODEL INGEST INVARIANT step 3) ──────────────────
static_assert(kLayers == 49, "flint backbone is 49 layers");
static_assert(kHidden == 2048, "flint hidden is 2048");
static_assert(kHidden % 256 == 0, "hidden tiles cleanly on Blackwell");
static_assert(kMoeIntermediate % 256 == 0, "moe intermediate tiles cleanly");
static_assert(kDenseIntermediate % 256 == 0, "dense intermediate tiles cleanly");
static_assert(kSharedIntermediate % 256 == 0, "shared expert tiles cleanly");
static_assert(kHeads * kHeadDim == 4096, "q/o projection width");
static_assert(kKvHeads * kHeadDim == 512, "k/v projection width");
static_assert(kHeads % kKvHeads == 0, "GQA groups");
static_assert(kHeadDim % 2 == 0, "half-split rotary pairing");
static_assert(kTopK > 0 && kTopK <= kNumExperts, "bad top-k");
static_assert(kFirstMoeLayer > 0 && kFirstMoeLayer < kLayers, "dense prefix");
static_assert(kWindow > 1, "sliding window includes the current token");
static_assert(kNumFullAttnLayers > 0 && kNumFullAttnLayers < kLayers,
              "hybrid attention needs both layer kinds");
static_assert(kBosId < kVocab && kEndOfTextId < kVocab && kVocab == 150144,
              "tokenizer specials inside the vocabulary");
static_assert(kNumEosIds == 4, "config.json eos_token_id has four ids");
constexpr bool FullAndNoRopeAgree()
{
    for (int i = 0; i < kNumFullAttnLayers; ++i)
        if (kFullAttnLayers[i] != kNoRopeLayers[i]) return false;
    return true;
}
static_assert(FullAndNoRopeAgree(),
              "the checkpoint's full-attention set IS its no-RoPE set");

} // namespace frozen
} // namespace flint
} // namespace lsxcommon
