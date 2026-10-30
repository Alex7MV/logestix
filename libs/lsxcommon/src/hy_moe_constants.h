#pragma once

// Hy-Memory-30B-A3B (checkpoint "tencent/Hy-Memory-30B-A3B") frozen passport.
//
// Stage 0 triage: the text backbone is `HYV3ForCausalLM` (config
// model_type "hy_v3") — a GQA decoder with plain RMSNorm (weight form, NOT
// Gemma's 1+w), full-dimension RoPE (rotate_half over all 128 head dims,
// theta 11158840, no scaling), per-head QK RMSNorm, a dense SwiGLU MLP on
// layer 0 and a sigmoid router with an expert bias on layers 1..47:
//
//   logits_e = W[e] . x                  (F32 accumulation)
//   score_e  = sigmoid(logits_e)
//   select   = top-K of (score_e + bias_e)
//   coeff_k  = score_e / sum(selected) * router_scaling_factor
//
// plus ONE gated shared expert (SwiGLU, intermediate 768) added to the routed
// sum in FP32 (`enable_moe_fp32_combine: true`).
//
// Geometry drift from every existing family (2048 hidden, 48 layers, 32 Q / 4
// KV heads, 128 experts top-8 at intermediate 768, shared expert 768) makes
// this a NEW family (Model Ingestion Protocol stage 0.1), so the numbers are
// locked here and cross-checked by the converter (config.json) and the loader
// (model.index.json).
//
// Verified against the source checkpoint on 2026-10-30 (config.json +
// 51-shard safetensors inventory, 18673 tensors):
//   routed experts : 47 layers x 128 experts x {gate, up, down},
//                    gate/up [768, 2048], down [2048, 768], BF16 on disk.
//                    NOTE: config `moe_intermediate_size` is 1536 while the
//                    tensors are 768 — the real width is `expert_hidden_dim`
//                    (768). The tensor shapes are authoritative; the passport
//                    pins 768 and the converter rejects a 1536-wide tree.
//   shared expert  : 47 layers x {gate, up, down} at [768, 2048]/[2048, 768]
//   dense layer 0  : gate/up [6912, 2048], down [2048, 6912]
//   router         : BF16 gate.weight [128, 2048] + F32 expert_bias [128]
//   norms          : BF16, plain form (scale = w)
//   embed / lm_head: BF16 [120832, 2048], NOT tied (different bytes)
//   ballast        : none — no vision tower, no MTP (`num_nextn_predict_layers`
//                    is 0 and the tree carries no mtp.* tensors)
//
// Semantics pinned by this passport (each was a candidate for a silently wrong
// token):
//   - attention scale 1/sqrt(128), applied at QK only
//   - FULL rotary on all 128 dims, rotate_half pairing j <-> j + 64
//   - q_norm / k_norm are per-head plain RMSNorm (weight form), dim 128
//   - the residual stream is FP16 in this engine; the router and the routed +
//     shared combination accumulate in FP32
//   - no final logit soft cap and no logit temperature at the model level
#include <cstdint>

namespace lsxcommon {
namespace hy {
namespace frozen {

// Registry id (LsxModelFactory::Register / model.index.json "model_type").
inline constexpr char kType[] = "hy-memory-30b-a3b";
inline constexpr char kFormatTag[] = "logestix-hy-nvfp4";
inline constexpr char kModelName[] = "Hy-Memory-30B-A3B";

// ── Frozen backbone geometry ────────────────────────────────────────────────
inline constexpr int kHidden = 2048;
inline constexpr int kLayers = 48;
inline constexpr int kHeads = 32;
inline constexpr int kKvHeads = 4;
inline constexpr int kHeadDim = 128;
inline constexpr int kRotaryDim = 128;   // full rotary (partial_rotary_factor absent)
inline constexpr int kVocab = 120832;

// MLP dispatch: layers [0, kFirstMoeLayer) use the dense SwiGLU MLP.
inline constexpr int kFirstMoeLayer = 1;
inline constexpr int kDenseIntermediate = 6912;
inline constexpr int kMoeIntermediate = 768;     // config expert_hidden_dim
inline constexpr int kSharedIntermediate = 768;  // moe_intermediate_size * 1 shared
inline constexpr int kNumExperts = 128;
inline constexpr int kTopK = 8;

// Tokenizer specials (config.json / tokenizer_config.json / tokenizer.json).
inline constexpr int kBosId = 120000;   // <｜hy_begin▁of▁sentence｜>
inline constexpr int kEosId = 120025;   // <eos:6124c78e>
inline constexpr int kPadId = 120002;   // <｜hy_▁pad▁｜>
inline constexpr int kUserTokenId = 120006;       // <｜hy_User｜>
inline constexpr int kAssistantTokenId = 120007;  // <｜hy_Assistant｜>
inline constexpr int kThinkBeginId = 120029;      // <think>   (special:false)
inline constexpr int kThinkEndId = 120030;        // </think>  (special:false)

// Numeric constants.
inline constexpr float kRmsEps = 1e-5f;
inline constexpr float kRopeTheta = 11158840.0f;
inline constexpr float kRoutedScale = 2.826f;
inline constexpr float kAttnScale = 0.08838834764831845f;  // 1/sqrt(128)

// ── Compile-time relations (MODEL INGEST INVARIANT step 3) ──────────────────
static_assert(kLayers == 48, "Hy-Memory-30B-A3B backbone is 48 layers");
static_assert(kHidden == 2048, "Hy-Memory-30B-A3B hidden is 2048");
static_assert(kHidden % 256 == 0, "hidden tiles cleanly on Blackwell");
static_assert(kMoeIntermediate % 256 == 0, "moe intermediate tiles cleanly");
static_assert(kDenseIntermediate % 256 == 0, "dense intermediate tiles cleanly");
static_assert(kSharedIntermediate % 256 == 0, "shared expert tiles cleanly");
static_assert(kHeads * kHeadDim == 4096, "q/o projection width");
static_assert(kKvHeads * kHeadDim == 512, "k/v projection width");
static_assert(kHeads % kKvHeads == 0, "GQA groups");
static_assert(kRotaryDim == kHeadDim, "full rotary over the whole head");
static_assert(kHeadDim % 2 == 0, "half-split rotary pairing");
static_assert(kTopK > 0 && kTopK <= kNumExperts, "bad top-k");
static_assert(kFirstMoeLayer > 0 && kFirstMoeLayer < kLayers, "dense prefix");
static_assert(kBosId < kVocab && kEosId < kVocab && kPadId < kVocab,
              "tokenizer specials inside the vocabulary");

constexpr bool IsMoeLayer(int layer)
{
    return layer >= kFirstMoeLayer && layer < kLayers;
}

} // namespace frozen
} // namespace hy
} // namespace lsxcommon
