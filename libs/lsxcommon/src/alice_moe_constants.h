#pragma once

// Frozen AliceAI-Foundation-80B-A3B-Base geometry (compile-time passport).
// Source of truth: <source root>/yandex/AliceAI-Foundation-80B-A3B-Base
// (config.json + modeling_alice_ai.py + safetensors audit 2026-10-01):
// 48 layers, 12 x (3 x (KDA->MoE) -> 1 x (gated attention->MoE)), hidden 2048,
// KDA 32/32 heads hd 128 conv 4, gated attention 16Q/2KV hd 256 partial RoPE
// 64, MoE 512 experts top-10 I=512 + shared I=512, block AttnRes 4, vocab
// 129024, bos 1 eos 2, MTP 1 layer (never applied), all BF16.

namespace lsxcommon::alice {
namespace frozen {

inline constexpr int kHidden = 2048;
inline constexpr int kLayers = 48;
inline constexpr int kFullAttnInterval = 4;  // full attention at i % 4 == 3
inline constexpr int kBlockAttnRes = 4;

inline constexpr int kHeads = 16;
inline constexpr int kKVHeads = 2;
inline constexpr int kHeadDim = 256;
inline constexpr int kRotaryDim = 64;  // 256 * partial_rotary_factor 0.25
inline constexpr double kRopeTheta = 1000000.0;
inline constexpr double kRmsNormEps = 1e-6;

inline constexpr int kLinearKeyHeads = 32;
inline constexpr int kLinearValueHeads = 32;
inline constexpr int kLinearKeyHeadDim = 128;
inline constexpr int kLinearValueHeadDim = 128;
inline constexpr int kLinearConvKernel = 4;
inline constexpr int kLinearKvRepeat = kLinearValueHeads / kLinearKeyHeads;

inline constexpr int kNumExperts = 512;
inline constexpr int kTopK = 10;
inline constexpr int kMoeIntermediate = 512;
inline constexpr int kSharedIntermediate = 512;

inline constexpr int kVocab = 129024;
inline constexpr int kBosId = 1;
inline constexpr int kEosId = 2;
inline constexpr int kPadId = -1;

inline constexpr int kMtpLayers = 1;

static_assert(kLayers % kFullAttnInterval == 0, "hybrid pattern incomplete");
static_assert(kLayers % kBlockAttnRes == 0, "AttnRes blocking incomplete");
static_assert(kFullAttnInterval == kBlockAttnRes,
              "AttnRes block != full-attn interval");
static_assert(kHeads % kKVHeads == 0, "GQA repeat mismatch");
static_assert(kLinearValueHeads % kLinearKeyHeads == 0, "KDA repeat mismatch");
static_assert(kRotaryDim > 0 && kRotaryDim % 2 == 0, "rotary dim must be even");
static_assert(kRotaryDim <= kHeadDim, "rotary dim exceeds head dim");
static_assert(kTopK > 0 && kTopK <= kNumExperts, "bad top-k");

} // namespace frozen
} // namespace lsxcommon::alice
