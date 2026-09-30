// MiMo-V2.6-Flash-RL frozen geometry passport (Model Ingestion Protocol
// step 3). Leaf header: NEVER includes another model's passport. All geometry
// is compile-time; MiMoV26FlashModel::Load cross-checks the parsed index
// against this block (a mismatch must fail Load). Tests relax it only through
// the runtime gate SyntheticGeometryAllowed() (LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY).
#pragma once

#include <array>
#include <cstdint>

namespace lsxcommon {
namespace mimo {
namespace frozen {

inline constexpr const char *kType = "mimo-v2.6-flash-rl";

inline constexpr int   kHidden            = 4096;
inline constexpr int   kLayers            = 48;
inline constexpr int   kVocab             = 152576;
inline constexpr int   kDenseIntermediate = 16384;
inline constexpr int   kMoeIntermediate   = 2048;
inline constexpr int   kNumRouted         = 256;
inline constexpr int   kTopK              = 8;
inline constexpr float kRmsEps            = 1e-6f;

// Layer 0 is the sole dense FFN; layers 1..47 are MoE.
inline constexpr int kMoeFirstLayer = 1;

// MTP speculative layers (model.mtp.layers.N): stored + counted, never applied.
inline constexpr std::array<int, 3> kMtpLayers = {0, 1, 2};

// Attention (GA = full/global, SWA = local sliding-window).
inline constexpr int kHeads      = 64;
inline constexpr int kKvHeadsGa  = 4;
inline constexpr int kKvHeadsSwa = 8;
inline constexpr int kQkHeadDim  = 192;
inline constexpr int kVHeadDim   = 128;
inline constexpr int kRopeDim    = 64;   // int(192 * partial_rotary_factor=0.334)

inline constexpr double kRopeThetaGa  = 10000000.0;
inline constexpr double kRopeThetaSwa = 10000.0;
inline constexpr int   kSlidingWindow = 128;
inline constexpr float kValueScale    = 0.707f;

inline constexpr int kQSize   = kHeads * kQkHeadDim;        // 12288
inline constexpr int kKSizeGa = kKvHeadsGa  * kQkHeadDim;   // 768
inline constexpr int kVSizeGa = kKvHeadsGa  * kVHeadDim;    // 512
inline constexpr int kKSizeSwa = kKvHeadsSwa * kQkHeadDim;  // 1536
inline constexpr int kVSizeSwa = kKvHeadsSwa * kVHeadDim;   // 1024
inline constexpr int kOSize    = kHeads * kVHeadDim;        // 8192

// 192^-0.5.
inline constexpr float kAttnScale = 0.07216878364870322f;

// hybrid_layer_pattern == 0 (global/full attention); 39 SWA layers elsewhere.
inline constexpr std::array<int, 9> kFullAttnLayers =
    {0, 5, 11, 17, 23, 29, 35, 41, 47};

// Router / MoE knobs.
inline constexpr int   kNGroups       = 1;      // noaux_tc group step = identity
inline constexpr int   kTopkGroup     = 1;
inline constexpr bool  kNormTopkProb  = true;
inline constexpr float kRoutedScale   = 1.0f;   // routed_scaling_factor is null

// Special tokens (read from the checkpoint config/tokenizer; kept here for
// reference — kernels take them as parameters, never hardcode).
inline constexpr int32_t kEosId     = 151645;  // <|im_end|>
inline constexpr int32_t kPadId     = 151643;  // <|endoftext|>
inline constexpr int32_t kBosId     = -1;      // no BOS
inline constexpr int32_t kImStartId = 151644;  // <|im_start|>

inline constexpr int kSinkHeads = kHeads;      // SWA attention_sink_bias [64]

inline constexpr bool IsFullAttnLayer(int i)
{
    for (int l : kFullAttnLayers) if (l == i) return true;
    return false;
}

inline constexpr bool IsMoeLayer(int i)
{
    return i >= kMoeFirstLayer && i < kLayers;
}

static_assert(kQSize == 12288, "Q projection geometry");
static_assert(kQSize + kKSizeGa + kVSizeGa == 13568, "GA fused qkv rows");
static_assert(kQSize + kKSizeSwa + kVSizeSwa == 14848, "SWA fused qkv rows");
static_assert(kOSize == 8192, "o_proj input geometry");
static_assert(kRopeDim % 2 == 0, "rope dim must be even");
static_assert(kFullAttnLayers.size() == 9, "9 full-attention layers");

} // namespace frozen
} // namespace mimo
} // namespace lsxcommon
