#pragma once

// MiniMax-M3 (checkpoint "MiniMaxAI/MiniMax-M3") frozen passport.
//
// Stage 0 triage: the text backbone is `MiniMaxM3SparseForCausalLM` — a GQA
// decoder with Gemma-style (1+w) RMSNorm, per-head QK norm, partial RoPE over
// the first 64 of 128 head dims, GPT-OSS-style SwiGLU-OAI
// (`(up+1) * gate*sigmoid(alpha*gate)`, gate/up clamped to +-swiglu_limit), a
// sigmoid router with an e_score_correction_bias (top-k by
// `sigmoid(logit)+bias`, weights renormalized from the plain sigmoid scores,
// routed sum scaled by routed_scaling_factor and then added to an UNGATED
// shared expert), and MiniMax Sparse Attention (MSA): a Lightning Indexer
// (4 heads x 128 dims, block=128 keys, top-16 blocks + 1 local block, max-pooled
// per block, block selection per KV/GQA group) that masks the main attention on
// layers >= kFirstSparseLayer.
//
// Geometry drift from every existing family (6144 hidden, 60 layers, 64 Q / 4
// KV heads, 128 experts top-4 with a 3072 shared expert, first 3 layers dense
// MLP + dense attention) makes this a NEW family (Model Ingestion Protocol
// stage 0.1), so the numbers are locked here and cross-checked by the
// converter (config.json) and the loader (model.index.json).
//
// Verified against the source checkpoint on 2026-10-27 (config.json
// text_config + safetensors inventory):
//   routed experts : 57 layers x 128 experts x {w1 gate, w2 down, w3 up},
//                    each [3072, 6144] / [6144, 3072], BF16 on disk
//   dense          : BF16 (attention + dense MLP + shared expert)
//   router         : F32 gate.weight [128, 6144] + F32 e_score_correction_bias
//   norms          : BF16, Gemma offset form (scale = 1 + w)
//   ballast        : vision_tower.* + multi_modal_projector.* + patch_merge_mlp.*
//                    (768 tensors) skipped+counted; no mtp tensors in the tree
//                    (`num_mtp_modules` is declared but unpopulated)
//
// Semantics pinned by this passport (each was a candidate for a silently wrong
// token):
//   - attention scale 1/sqrt(128), applied at QK only (nothing folded into w)
//   - partial rotary: dims [0,64) rotated half-split (rotate_half pairing
//     j <-> j+32), dims [64,128) passed through, theta 5e6
//   - q_norm / k_norm / index_q_norm / index_k_norm are Gemma-form per-head
//     RMSNorm (indexer norms are per head too, dim 128)
//   - the indexer scores are UNSCALED dot products in FP32
//   - shared expert is ungated and added AFTER scaling the routed sum
//   - no final logit soft cap and no logit temperature at the model level
#include <cstdint>

namespace lsxcommon {
namespace minimax {
namespace frozen {

// Registry id (LsxModelFactory::Register / model.index.json "model_type").
inline constexpr char kType[] = "minimax-m3";
inline constexpr char kFormatTag[] = "logestix-minimax-nvfp4";
inline constexpr char kModelName[] = "MiniMax-M3";

// ── Frozen backbone geometry ────────────────────────────────────────────────
inline constexpr int kHidden = 6144;
inline constexpr int kLayers = 60;
inline constexpr int kHeads = 64;
inline constexpr int kKvHeads = 4;
inline constexpr int kHeadDim = 128;
inline constexpr int kRotaryDim = 64;      // partial_rotary_factor 0.5
inline constexpr int kVocab = 200064;

// MLP dispatch: layers [0, kFirstMoeLayer) use the dense SwiGLU MLP.
inline constexpr int kFirstMoeLayer = 3;
inline constexpr int kDenseIntermediate = 12288;
inline constexpr int kMoeIntermediate = 3072;
inline constexpr int kSharedIntermediate = 3072;
inline constexpr int kNumExperts = 128;
inline constexpr int kTopK = 4;

// MiniMax Sparse Attention (sparse_attention_config; active on every layer
// whose sparse_attention_freq entry is 1, i.e. layers >= kFirstSparseLayer).
inline constexpr int kFirstSparseLayer = 3;
inline constexpr int kIndexNHeads = 4;      // == kKvHeads: one per GQA group
inline constexpr int kIndexHeadDim = 128;
inline constexpr int kIndexBlockSize = 128;
inline constexpr int kIndexTopkBlocks = 16;
inline constexpr int kIndexLocalBlocks = 1;

// Tokenizer specials (tokenizer_config.json / generation_config.json).
inline constexpr int kBosId = 200019;   // ]~b]
inline constexpr int kEosId = 200020;   // [e~[
inline constexpr int kPadId = 200000;   // ]!p~[
inline constexpr int kThinkBeginId = 200059;  // <mm:think>
inline constexpr int kThinkEndId = 200060;    // </mm:think>

// Numeric constants.
inline constexpr float kRmsEps = 1e-6f;
inline constexpr float kRopeTheta = 5000000.0f;
inline constexpr float kSwiGLUAlpha = 1.702f;
inline constexpr float kSwiGLULimit = 7.0f;
inline constexpr float kRoutedScale = 2.0f;
inline constexpr float kAttnScale = 0.08838834764831845f;  // 1/sqrt(128)

// ── Compile-time relations (MODEL INGEST INVARIANT step 3) ──────────────────
static_assert(kLayers == 60, "M3 backbone is 60 layers");
static_assert(kHidden == 6144, "M3 hidden is 6144");
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
static_assert(kFirstSparseLayer == kFirstMoeLayer,
              "config: layers 0-2 are dense MLP AND dense attention");
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
} // namespace minimax
} // namespace lsxcommon
