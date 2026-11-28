// Xing4.0-29B-A4B frozen geometry passport (Model Ingestion Protocol step 3).
// Leaf header: NEVER includes another model's passport. All geometry is
// compile-time; XingMoeModel::Load cross-checks the parsed index against this
// block (a mismatch must fail Load). Tests relax it only through the runtime
// gate SyntheticGeometryAllowed() (LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY).
//
// Source of truth (read-only reference, not vendored):
//   <source root>/XingChen-AGI/Xing4.0-29B-A4B/config.json
//   .../modeling_xing4_0.py  (China Telecom AI, TeleChat lineage)
//   .../model.safetensors.index.json  (8307 tensors, 58.1 GiB bf16)
//
// Architecture: mHC (manifold-constrained Hyper-Connections, hc_mult=4
// parallel hidden streams with a Sinkhorn-normalised 4x4 mixing matrix) +
// MLA (q/kv latent compression, no GQA) + DeepSeek-V3-style sigmoid noaux_tc
// router + MTP. Unlike every other family in the repo the residual stream is
// FOUR-WIDE, so each layer starts by collapsing the 4 streams with a learned
// per-stream gate before it can feed the norms.
#pragma once

#include <array>
#include <cstdint>

namespace lsxcommon {
namespace xing {
namespace frozen {

inline constexpr const char *kType = "xing4.0-29b-a4b";

inline constexpr int   kHidden            = 3584;
inline constexpr int   kLayers            = 40;
inline constexpr int   kVocab             = 131072;
inline constexpr float kRmsEps            = 1e-6f;

// mHC — manifold-constrained Hyper-Connections (Xing4_0HyperConnection).
inline constexpr int   kHcMult            = 4;      // parallel hidden streams
inline constexpr int   kHcMix             = (2 + kHcMult) * kHcMult;  // 24
inline constexpr int   kHcFlat            = kHcMult * kHidden;         // 14336
inline constexpr int   kHcSinkhornIters   = 20;
inline constexpr float kHcEps             = 1e-6f;
inline constexpr float kHcClampMin        = -30.0f;
inline constexpr float kHcClampMax        = 30.0f;
inline constexpr float kHcPostGain        = 2.0f;   // post = 2*sigmoid(...)

// MLA (Xing4_0Attention). MHA over the latent: num_key_value_heads ==
// num_attention_heads, so no head repetition in the cache.
inline constexpr int kHeads        = 32;
inline constexpr int kKvHeads      = 32;
inline constexpr int kQLoraRank    = 768;
inline constexpr int kKvLoraRank   = 512;
inline constexpr int kQkNopeHeadDim = 128;
inline constexpr int kQkRopeHeadDim = 64;
inline constexpr int kQkHeadDim    = kQkNopeHeadDim + kQkRopeHeadDim;  // 192
inline constexpr int kVHeadDim     = 128;

inline constexpr int kQSize    = kHeads * kQkHeadDim;                     // 6144
inline constexpr int kKvAWidth = kKvLoraRank + kQkRopeHeadDim;            // 576
inline constexpr int kKvBWidth = kHeads * (kQkNopeHeadDim + kVHeadDim);   // 8192
inline constexpr int kOSize    = kHeads * kVHeadDim;                      // 4096

// YaRN (NTK-by-parts) with theta 10000, factor 64 over a 4096-token original
// window, interleaved (x0,x1) rotary pairs — rope_interleave defaults to True
// and config.json does not override it.
inline constexpr double kRopeTheta      = 10000.0;
inline constexpr double kYarnFactor     = 64.0;
inline constexpr double kYarnOrig       = 4096.0;
inline constexpr double kYarnBetaFast   = 32.0;
inline constexpr double kYarnBetaSlow   = 1.0;
inline constexpr int    kRopeDim        = kQkRopeHeadDim;   // 64
inline constexpr bool   kRopeInterleave = true;

// YaRN cos/sin scaling. transformers 5.15's `_compute_yarn_parameters` returns
// attention_factor = 1.0 for this checkpoint (rope_scaling.mscale_all_dim=1.0
// does NOT reach the returned factor), and Xing4_0RotaryEmbedding multiplies
// cos/sin by it - verified against modeling_xing4_0.py, whose tables satisfy
// cos^2 + sin^2 == 1 exactly. The mscale = 0.1*ln(factor)+1 = 1.4158883 shows up
// ONLY in the softmax scale, where the attention module folds mscale*mscale in.
inline constexpr double kYarnMscale     = 1.0;
// mscale^2 / sqrt(192).
inline constexpr float kAttnScale = 0.14467962580268925f;

// MoE / dense MLP.
inline constexpr int   kFirstKDense        = 2;   // layers 0..1 dense
inline constexpr int   kMoeFirstLayer      = kFirstKDense;
inline constexpr int   kDenseIntermediate  = 9216;
inline constexpr int   kMoeIntermediate    = 1024;
inline constexpr int   kNumRouted          = 64;
inline constexpr int   kTopK               = 4;
inline constexpr int   kSharedExperts      = 1;
inline constexpr int   kSharedIntermediate = kMoeIntermediate * kSharedExperts;
inline constexpr float kRoutedScale        = 2.0f;  // routed_scaling_factor
inline constexpr bool  kNormTopkProb       = true;
inline constexpr bool  kSigmoidScore       = true;  // scoring_func = sigmoid
inline constexpr bool  kNoauxTc            = true;  // topk_method = noaux_tc

// MTP: model.layers.40 is stored + counted, never applied at decode
// (Xing4_0PreTrainedModel._keys_to_ignore_on_load_unexpected drops it).
inline constexpr std::array<int, 1> kMtpLayers = {40};

// Special tokens (tokenizer_config.json added_tokens_decoder; all special:true).
inline constexpr int32_t kBosId            = 1;   // <_start>
inline constexpr int32_t kEosId            = 2;   // <_end>
inline constexpr int32_t kPadId            = 3;   // <_pad>
inline constexpr int32_t kUserId           = 4;   // <_user>
inline constexpr int32_t kBotId            = 5;   // <_bot>
inline constexpr int32_t kSystemId        = 6;   // <_system>
inline constexpr int32_t kThinkId          = 9;   // <think>
inline constexpr int32_t kThinkEndId       = 10;  // </think>
inline constexpr int32_t kToolCallId       = 11;  // <tool_call>
inline constexpr int32_t kToolCallEndId    = 12;  // </tool_call>
inline constexpr int32_t kToolResponseId   = 13;  // <tool_response>
inline constexpr int32_t kToolResponseEndId = 14; // </tool_response>

inline constexpr bool IsMoeLayer(int i)
{
    return i >= kMoeFirstLayer && i < kLayers;
}

static_assert(kHcMix == 24, "hc_fn rows: pre(4) + post(4) + comb(4x4)");
static_assert(kHcFlat == 14336, "hc input_norm covers all 4 streams");
static_assert(kQSize == 6144, "q_b_proj output rows");
static_assert(kKvAWidth == 576, "kv_a_proj_with_mqa = 512 latent + 64 rope");
static_assert(kKvBWidth == 8192, "kv_b_proj output rows");
static_assert(kOSize == 4096, "o_proj input columns");
static_assert(kRopeDim % 2 == 0, "rope dim must be even");
static_assert(kMoeFirstLayer == kFirstKDense, "dense prefix length");
static_assert(kMoeIntermediate * kSharedExperts == kSharedIntermediate,
              "shared expert width");

} // namespace frozen
} // namespace xing
} // namespace lsxcommon