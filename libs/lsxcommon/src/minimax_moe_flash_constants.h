#pragma once

// MiniMax-M3.1-Flash (checkpoint "MiniMaxAI/MiniMax-M3.1-Flash") frozen
// passport — a SIBLING of MiniMax-M3, not a new architecture.
//
// Stage 0 triage (2026-10-31), verified against the source tree:
//   text side   : `MiniMaxM3SparseForCausalLM` with the M3 backbone contour
//                 (Gemma (1+w) RMSNorm, per-head QK norm, partial RoPE on the
//                 first 64 of 128 head dims, SwiGLU-OAI, sigmoid router with
//                 e_score_correction_bias top-4 + renormalised scores ×2.0 +
//                 one UNGATED 3072 shared expert), 60 layers / 6144H /
//                 64Q·4KV·128HD / vocab 200064 / 128 experts.
//   deltas      : RoPE theta 1e7 (M3: 5e6); MiniMax Sparse Attention (MSA) on
//                 EVERY layer (`sparse_attention_freq` = all ones,
//                 `sparse_disable_index_value` = all ones), i.e. the layer-0..2
//                 dense-attention prefix is gone while their dense MLP stays;
//                 the source is already quantized — MXFP8 dense (F8_E4M3 +
//                 per-32 E8M0 `weight_scale_inv`) and ModelOpt NVFP4 routed
//                 experts (`weight_packed` / `weight_scale` / F32 divisor
//                 `weight_global_scale` + `input_global_scale`), quantized-
//                 aware-trained, so the converter DEQUANTIZES the dense side
//                 to the repo's NVFP4 and passes the experts through.
//   ballast     : vision_tower.* + multi_modal_projector.* + patch_merge_mlp.*
//                 (counted, never applied) and the separate `dspark/` draft
//                 head directory (DSpark speculative decoding, not part of this
//                 text checkpoint's weight map — stored, never applied).
//   tokenizer   : tokenizer.json / vocab.json / merges.txt / added_tokens.json
//                 are BYTE-IDENTICAL to MiniMax-M3's (same 200064 vocab,
//                 `]~b]`/`[e~[` framing); `chat_template.jinja` differs (next
//                 paragraph), so the tool bridge is a sibling policy, not a
//                 copy.
//   chat        : system prompt names MiniMax-M3.1 and appends
//                 `<model_id>8bb8149fe</model_id>`; thinking mode is fixed to
//                 ADAPTIVE (no `thinking_mode` variable) and the generation
//                 prompt therefore carries NO think-END prefix (M3 emits it);
//                 mid-conversation system/developer turns render as `developer`
//                 blocks; consecutive tool results share ONE `]~b]tool` block.
//
// Geometry is identical to M3's, but the numbers are re-declared here so this
// model owns its own passport (model independence: one frozen namespace per
// REGISTERED model — same reason the nemotron NVFP4 sibling re-declares).
// The loader, kernels and generation pipeline are the PARENT's
// (`lsxcommon::minimax`); this family never duplicates them. The emitted
// artifact layout is identical, so the format tag is shared — only the index's
// `model` field distinguishes the trees (`detect_model_type` normalizes it to
// exactly kType).
//
// Semantics pinned (each was a candidate for a silently wrong token):
//   - attention scale 1/sqrt(128), applied at QK only
//   - partial rotary: dims [0,64) rotated half-split, dims [64,128) passthrough,
//     theta 1e7, indexer RoPE uses the same theta
//   - the indexer scores are UNSCALED FP32 dot products; the M3 block
//     selection (top-16 of 128-key blocks + 1 local block) now runs on all 60
//     layers
//   - shared expert is ungated and added after scaling the routed sum
//   - no final logit soft cap and no logit temperature at the model level
#include <cstdint>

namespace lsxcommon {
namespace minimax_flash {
namespace frozen {

// Registry id (LsxModelFactory::Register / model.index.json "model").
inline constexpr char kType[] = "minimax-m3.1-flash";
inline constexpr char kModelName[] = "MiniMax-M3.1-Flash";
// Structurally identical artifact: the parent loader accepts both trees.
inline constexpr char kFormatTag[] = "logestix-minimax-nvfp4";

// ── Frozen backbone geometry (identical to MiniMax-M3) ──────────────────────
inline constexpr int kHidden = 6144;
inline constexpr int kLayers = 60;
inline constexpr int kHeads = 64;
inline constexpr int kKvHeads = 4;
inline constexpr int kHeadDim = 128;
inline constexpr int kRotaryDim = 64;      // partial_rotary_factor 0.5
inline constexpr int kVocab = 200064;

inline constexpr int kFirstMoeLayer = 3;   // dense SwiGLU MLP on layers 0-2
inline constexpr int kDenseIntermediate = 12288;
inline constexpr int kMoeIntermediate = 3072;
inline constexpr int kSharedIntermediate = 3072;
inline constexpr int kNumExperts = 128;
inline constexpr int kTopK = 4;

// MSA is active on every layer in this checkpoint.
inline constexpr int kFirstSparseLayer = 0;
inline constexpr int kIndexNHeads = 4;
inline constexpr int kIndexHeadDim = 128;
inline constexpr int kIndexBlockSize = 128;
inline constexpr int kIndexTopkBlocks = 16;
inline constexpr int kIndexLocalBlocks = 1;

// Tokenizer specials (identical tokenizer.json to MiniMax-M3).
inline constexpr int kBosId = 200019;   // ]~b]
inline constexpr int kEosId = 200020;   // [e~[
inline constexpr int kPadId = 200000;   // ]!p~[
inline constexpr int kThinkBeginId = 200059;  // <mm:think>
inline constexpr int kThinkEndId = 200060;    // </mm:think>

// Numeric constants.
inline constexpr float kRmsEps = 1e-6f;
inline constexpr float kRopeTheta = 10000000.0f;
inline constexpr float kSwiGLUAlpha = 1.702f;
inline constexpr float kSwiGLULimit = 7.0f;
inline constexpr float kRoutedScale = 2.0f;
inline constexpr float kAttnScale = 0.08838834764831845f;  // 1/sqrt(128)

// Chat template identity (chat_template.jinja).
inline constexpr char kModelId[] = "8bb8149fe";

// ── Compile-time relations (MODEL INGEST INVARIANT step 3) ──────────────────
static_assert(kLayers == 60, "M3.1 backbone is 60 layers");
static_assert(kHidden == 6144, "M3.1 hidden is 6144");
static_assert(kHidden % 256 == 0, "hidden tiles cleanly on Blackwell");
static_assert(kMoeIntermediate % 256 == 0, "moe intermediate tiles cleanly");
static_assert(kDenseIntermediate % 256 == 0, "dense intermediate tiles cleanly");
static_assert(kSharedIntermediate % 256 == 0, "shared expert tiles cleanly");
static_assert(kHeads * kHeadDim == 8192, "q/o projection width");
static_assert(kHeads % kKvHeads == 0, "GQA groups");
static_assert(kIndexNHeads == kKvHeads, "one indexer head per KV group");
static_assert(kIndexHeadDim == kHeadDim, "indexer head shares the qk-norm width");
static_assert(kHeadDim % 2 == 0 && kRotaryDim > 0 && kRotaryDim < kHeadDim,
              "partial rotary must be a proper half-split of the head");
static_assert(kTopK > 0 && kTopK <= kNumExperts, "bad top-k");
static_assert(kFirstMoeLayer > 0 && kFirstMoeLayer < kLayers, "dense prefix");
static_assert(kFirstSparseLayer == 0, "MSA runs on every layer of M3.1");
static_assert(kIndexTopkBlocks * kIndexBlockSize >= 2048,
              "the canonical 2048-token protocol stays dense-equivalent");
static_assert(kBosId < kVocab && kEosId < kVocab && kPadId < kVocab,
              "tokenizer specials inside the vocabulary");

constexpr bool IsMoeLayer(int layer)
{
    return layer >= kFirstMoeLayer && layer < kLayers;
}

constexpr bool IsSparseLayer(int layer)
{
    return layer >= kFirstSparseLayer && layer < kLayers;
}

} // namespace frozen
} // namespace minimax_flash
} // namespace lsxcommon
