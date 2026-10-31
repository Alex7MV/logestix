#pragma once

// Hy-SimulMT-30B-A3B (checkpoint "tencent/Hy-SimulMT-30B-A3B") frozen
// passport — a SIBLING of Hy-Memory-30B-A3B, not a new family.
//
// Stage 0 triage (2026-10-31): the checkpoint is `HYV3SpeechForCausalLM`
// (model_type "hy_v3") with the Hy-Memory text backbone plus a 24-layer speech
// encoder. Verified against the source tree:
//   text side : 18673 tensors, byte-identical names/shapes to Hy-Memory
//               (`model.*` + `lm_head.weight`; per-expert
//               `mlp.experts.E.{gate,up,down}_proj`, `mlp.shared_mlp.*`,
//               `mlp.router.gate.weight` + `mlp.expert_bias`)
//   ballast   : 403 `audio_tower.*` tensors (660 MB) — stripped and counted,
//               never applied (speech input is out of scope for this engine)
//   tokenizer : tokenizer.json is byte-identical to Hy-Memory's
//               (sha256 ad96ef9f641b79bb…, 120000 BPE / 818 added tokens)
//
// Geometry is therefore the same as Hy-Memory's, and the numbers are locked
// here so this model owns its own passport (model independence: one frozen
// namespace per model). The loader, kernels and generation pipeline are the
// PARENT's — this family never duplicates them.
//
// Config deltas the shared parser absorbs:
//   - `moe_intermediate_size` is 768 (no vestigial 1536);
//   - no `mlp_layer_types` (dispatch comes from `first_k_dense_replace: 1`);
//   - flat `rope_theta` (no `rope_parameters` object).
//
// Semantics pinned (same as Hy-Memory, re-verified on this checkpoint):
//   - attention scale 1/sqrt(128) at QK only; full rotary on all 128 dims
//     (rotate_half pairing j <-> j + 64), theta 11158840, no rope scaling;
//   - q_norm / k_norm are per-head plain RMSNorm (weight form), dim 128;
//   - residual stream FP16 in this engine; router + routed/shared combination
//     accumulate in FP32; no final logit softcap / temperature.
#include <cstdint>

namespace lsxcommon {
namespace hy_simulmt {
namespace frozen {

// Registry id (LsxModelFactory::Register / model.index.json "model_type").
inline constexpr char kType[] = "hy-simulmt-30b-a3b";
inline constexpr char kFormatTag[] = "logestix-hy-simulmt-nvfp4";
inline constexpr char kModelName[] = "Hy-SimulMT-30B-A3B";

// ── Frozen backbone geometry ────────────────────────────────────────────────
inline constexpr int kHidden = 2048;
inline constexpr int kLayers = 48;
inline constexpr int kHeads = 32;
inline constexpr int kKvHeads = 4;
inline constexpr int kHeadDim = 128;
inline constexpr int kRotaryDim = 128;
inline constexpr int kVocab = 120832;

// MLP dispatch: layers [0, kFirstMoeLayer) use the dense SwiGLU MLP.
inline constexpr int kFirstMoeLayer = 1;
inline constexpr int kDenseIntermediate = 6912;
inline constexpr int kMoeIntermediate = 768;     // config expert_hidden_dim
inline constexpr int kSharedIntermediate = 768;
inline constexpr int kNumExperts = 128;
inline constexpr int kTopK = 8;

// Tokenizer specials (identical tokenizer.json to Hy-Memory).
inline constexpr int kBosId = 120000;
inline constexpr int kEosId = 120025;
inline constexpr int kPadId = 120002;
inline constexpr int kUserTokenId = 120006;
inline constexpr int kAssistantTokenId = 120007;
inline constexpr int kThinkBeginId = 120029;      // <think>   (special:false)
inline constexpr int kThinkEndId = 120030;        // </think>  (special:false)

// ── Serving contract ────────────────────────────────────────────────────────
// The checkpoint is a simultaneous translator: the official client
// (SimulInfer/simulinfer/translation.py `INSTRUCTIONS`) sends the direction
// directive as the first text part of the user turn, followed by the source
// text/speech. Without it the decoder only transcribes (echoes) the source.
// The text engine auto-selects the direction from the source script: Han ->
// zh-en, otherwise en-zh.
inline constexpr char kZhEnInstruction[] =
    "请完成下面的同声传译任务，将陆续收到的语音翻译成英文。请结合上下文边听"
    "边译，输出能够确定的译文，不必等待整句结束。";
inline constexpr char kEnZhInstruction[] =
    "请完成下面的同声传译任务，将陆续收到的语音翻译成中文。请结合上下文边听"
    "边译，输出能够确定的译文，不必等待整句结束。";

// Numeric constants.
inline constexpr float kRmsEps = 1e-5f;
inline constexpr float kRopeTheta = 11158840.0f;
inline constexpr float kRoutedScale = 2.826f;
inline constexpr float kAttnScale = 0.08838834764831845f;  // 1/sqrt(128)

// ── Compile-time relations (MODEL INGEST INVARIANT step 3) ──────────────────
static_assert(kLayers == 48, "Hy-SimulMT backbone is 48 layers");
static_assert(kHidden == 2048, "Hy-SimulMT hidden is 2048");
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
} // namespace hy_simulmt
} // namespace lsxcommon
