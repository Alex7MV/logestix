// Mistral-Small-4-119B-2603 frozen geometry passport (Model Ingestion Protocol
// step 3). Leaf header: NEVER includes another model's passport. All geometry is
// compile-time; MistralSmall4MoeModel::Load cross-checks the parsed index against
// this block (a mismatch must fail Load). Tests relax it only through the runtime
// gate SyntheticGeometryAllowed() (LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY).
//
// Source of truth (read-only reference, not vendored):
//   /mnt/llm_models/models/Mistral-Small-4-119B-2603-NVFP4/params.json
//   .../consolidated.safetensors.index.json  (56313 tensors, 66 GiB)
//   model shipped with the Mistral "consolidated" layout (params.json, no
//   config.json), already post-training quantized to NVFP4 by llm-compressor
//   (compressed-tensors). Text decoder only: the vision tower / patch merger /
//   vision_language_adapter are stripped ballast.
//
// Architecture: DeepSeek-V2/V3-style MLA (q/kv latent compression, 32 heads,
// qk nope 64 + rope 64 = 128, v 128) + a 128-expert top-4 MoE with 1 shared
// expert on every layer (first_k_dense_replace = 0) + YaRN (factor 128) with
// Llama-4 logit temperature tuning. Plain RMSNorm everywhere (no gating).
#pragma once

#include <array>
#include <cstdint>

namespace lsxcommon {
namespace mistral_moe {
namespace frozen {

inline constexpr const char *kType = "mistral-small-4-119b";

inline constexpr int   kHidden            = 4096;
inline constexpr int   kLayers            = 36;
inline constexpr int   kVocab             = 131072;
inline constexpr float kRmsEps            = 1e-6f;

// MLA (MistralSmall4Attention). MHA over the latent: num_key_value_heads ==
// num_attention_heads (32), so there is no head repetition in the cache.
inline constexpr int   kHeads             = 32;
inline constexpr int   kKvHeads           = 32;
inline constexpr int   kQLoraRank         = 1024;
inline constexpr int   kKvLoraRank        = 256;
inline constexpr int   kQkNopeHeadDim     = 64;
inline constexpr int   kQkRopeHeadDim     = 64;
inline constexpr int   kQkHeadDim         = kQkNopeHeadDim + kQkRopeHeadDim;  // 128
inline constexpr int   kVHeadDim          = 128;

inline constexpr int   kQSize    = kHeads * kQkHeadDim;                     // 4096
inline constexpr int   kKvAWidth = kKvLoraRank + kQkRopeHeadDim;            // 320
inline constexpr int   kKvBWidth = kHeads * (kQkNopeHeadDim + kVHeadDim);   // 6144
inline constexpr int   kOSize    = kHeads * kVHeadDim;                      // 4096

// YaRN (NTK-by-parts), theta 10000, factor 128 over an 8192-token original
// window. params.json sets yarn.apply_scale = false, so the cos/sin tables are
// NOT multiplied by the mscale attention factor; the MLA softmax scale is the
// plain 1/sqrt(qk_head_dim).
inline constexpr double kRopeTheta      = 10000.0;
inline constexpr double kYarnFactor     = 128.0;
inline constexpr double kYarnOrig       = 8192.0;
inline constexpr double kYarnBetaFast   = 32.0;
inline constexpr double kYarnBetaSlow   = 1.0;
inline constexpr int    kRopeDim        = kQkRopeHeadDim;   // 64
inline constexpr bool   kRopeInterleave = false;            // split-half RoPE

// Llama-4 long-context logit temperature (params.json llama_4_scaling):
//   factor(pos) = 1 + beta * log(floor(pos / orig) + 1)
// applied multiplicatively to the attention logits for pos >= orig.
inline constexpr double kLlama4Beta      = 0.1;
inline constexpr double kLlama4Orig      = 8192.0;

inline constexpr float  kAttnScale = 0.08838834764831845f;  // 1/sqrt(128)

// MoE / no dense prefix.
inline constexpr int   kFirstKDense        = 0;   // all layers MoE
inline constexpr int   kMoeFirstLayer      = kFirstKDense;
inline constexpr int   kMoeIntermediate    = 2048;
inline constexpr int   kNumRouted          = 128;
inline constexpr int   kTopK               = 4;
inline constexpr int   kSharedExperts      = 1;
inline constexpr int   kSharedIntermediate = kMoeIntermediate * kSharedExperts;
inline constexpr float kRoutedScale        = 1.0f;

// Special tokens (tokenizer.json added_tokens; all special:true).
inline constexpr int32_t kUnkId              = 0;   // <unk>
inline constexpr int32_t kBosId              = 1;   // <s>
inline constexpr int32_t kEosId              = 2;   // </s>
inline constexpr int32_t kInstId             = 3;   // [INST]
inline constexpr int32_t kInstEndId          = 4;   // [/INST]
inline constexpr int32_t kAvailableToolsId   = 5;   // [AVAILABLE_TOOLS]
inline constexpr int32_t kAvailableToolsEndId = 6;  // [/AVAILABLE_TOOLS]
inline constexpr int32_t kToolResultsId      = 7;   // [TOOL_RESULTS]
inline constexpr int32_t kToolResultsEndId   = 8;   // [/TOOL_RESULTS]
inline constexpr int32_t kToolCallsId        = 9;   // [TOOL_CALLS]
inline constexpr int32_t kPadId              = 11;  // <pad>
inline constexpr int32_t kSystemPromptId     = 17;  // [SYSTEM_PROMPT]
inline constexpr int32_t kSystemPromptEndId  = 18;  // [/SYSTEM_PROMPT]
inline constexpr int32_t kArgsId             = 32;  // [ARGS]
inline constexpr int32_t kCallIdId           = 33;  // [CALL_ID]
inline constexpr int32_t kThinkId            = 34;  // [THINK]
inline constexpr int32_t kThinkEndId         = 35;  // [/THINK]
inline constexpr int32_t kModelSettingsId    = 36;  // [MODEL_SETTINGS]
inline constexpr int32_t kModelSettingsEndId = 37;  // [/MODEL_SETTINGS]

// Frozen-geometry invariants (compile-time).
static_assert(kQkHeadDim == 128, "MLA qk head dim");
static_assert(kQSize == 4096, "q_b projection output rows");
static_assert(kKvAWidth == 320, "kv_a_with_mqa = 256 latent + 64 rope");
static_assert(kKvBWidth == 6144, "kv_b projection output rows");
static_assert(kOSize == 4096, "o_proj input columns");
static_assert(kRopeDim % 2 == 0, "rope dim must be even");
static_assert(kSharedIntermediate == 2048, "shared expert width");
static_assert(kTopK <= kNumRouted, "top-k exceeds expert count");
static_assert(kHidden % 128 == 0, "hidden must tile by 128");

} // namespace frozen
} // namespace mistral_moe
} // namespace lsxcommon
