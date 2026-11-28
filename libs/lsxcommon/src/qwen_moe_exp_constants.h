#pragma once

// Frozen Qwen3.8-Flash-Next geometry (compile-time passport).
// Source of truth: <source root>/Qwen/Qwen3.8-Flash-Next/config.json
// (text_config, model_type qwen4_exp_text) + safetensors inventory audit
// (2026-09-02): 48 layers, GDN 16x128 QK / 48x128 V, QSA 24Q/2KV HD256,
// indexer 4Q/1KV MQA HD128 budget 2048 compress 4, MoE 512 experts top-10
// I=640, shared I=640, hc_count=4 lowrank=320, PLE layer_idx=1 (layer 2),
// ngram_size=3, heads_per_ngram=8, split 128 shards [2500012,160],
// vocab 248320, eos [<|im_end|>=248046, <|endoftext|>=248044], pad 248044.

#include <cstdint>

namespace lsxcommon::qwen_moe_exp {
namespace frozen {

inline constexpr int kHidden = 2560;
inline constexpr int kLayers = 48;
inline constexpr int kFullAttnInterval = 4;  // full attention at i%4==3
inline constexpr int kHeads = 24;
inline constexpr int kKVHeads = 2;
inline constexpr int kHeadDim = 256;
inline constexpr int kRotaryDim = 64;  // 256 * partial_rotary_factor 0.25

// Gated DeltaNet (linear attention).
inline constexpr int kLinearNumKeyHeads = 16;
inline constexpr int kLinearNumValueHeads = 48;
inline constexpr int kLinearKeyHeadDim = 128;
inline constexpr int kLinearValueHeadDim = 128;
inline constexpr int kLinearConvKernelDim = 4;
inline constexpr int kLinearKvRepeat = 3;  // 48/16 repeat_interleave

// QSA indexer (micro-block selection).
inline constexpr int kIndexerNHeads = 4;
inline constexpr int kIndexerKVHeads = 1;
inline constexpr int kIndexerHeadDim = 128;
inline constexpr int kIndexerBudget = 2048;    // tokens
inline constexpr int kIndexerCompress = 4;     // tokens per block
inline constexpr int kIndexerBlockTopk = 512;  // 2048 / 4

// Hyper-connections (GatedResidual).
inline constexpr int kHcCount = 4;
inline constexpr int kHcLowrank = 320;
inline constexpr int kHcHidden = kHidden * kHcCount;  // 10240

// MoE.
inline constexpr int kNumExperts = 512;
inline constexpr int kTopK = 10;
inline constexpr int kMoeIntermediate = 640;
inline constexpr int kSharedIntermediate = 640;
inline constexpr int kVocab = 248320;

// PLE (n-gram embedding, layer 2 / HF layer_idx 1).
inline constexpr int kPleLayerIdx = 1;  // HF layer_idx; tensor prefix layers.1.ple
inline constexpr int kPleNgramSize = 3;
inline constexpr int kPleHeadsPerNgram = 8;
inline constexpr int kPleHeads = (kPleNgramSize - 1) * kPleHeadsPerNgram;  // 16
inline constexpr int kPleEmbedDim = 2560;                                  // == kHidden
inline constexpr int kPleHeadDim = kPleEmbedDim / kPleHeads;               // 160
inline constexpr int kPleSplitParts = 128;
inline constexpr int64_t kPleNgramVocabBase = 20000000;
inline constexpr int64_t kPleSeed = 1234;
inline constexpr int kPleConvKernel = 4;
inline constexpr int kPleConvDilation = kPleNgramSize;                           // 3
inline constexpr int kPleConvStateLen = (kPleConvKernel - 1) * kPleConvDilation; // 9

// Special tokens.
inline constexpr int kEosPrimary = 248046;    // <|im_end|>
inline constexpr int kEosSecondary = 248044;  // <|endoftext|>
inline constexpr int kPadId = 248044;
inline constexpr int kImStart = 248045;
inline constexpr int kThink = 248068;
inline constexpr int kThinkEnd = 248069;

inline constexpr double kRmsNormEps = 1e-6;
inline constexpr double kPartialRotaryFactor = 0.25;
inline constexpr double kRopeTheta = 10000000.0;

// Frozen-geometry passport invariants (MODEL INGEST INVARIANT step 3).
static_assert(kTopK <= kNumExperts, "top-k exceeds expert count");
static_assert(kLayers % kFullAttnInterval == 0, "hybrid pattern incomplete");
static_assert(kLinearNumValueHeads % kLinearNumKeyHeads == 0, "GDN repeat mismatch");
static_assert(kPleHeads * kPleHeadDim == kPleEmbedDim, "PLE head geometry");
static_assert(kHcCount * kHidden == kHcHidden, "hc layout");
static_assert(kIndexerBudget % kIndexerCompress == 0, "indexer block geometry");
static_assert(kPleHeadDim == 160, "PLE head dim fixed by checkpoint shards");

} // namespace frozen
} // namespace lsxcommon::qwen_moe_exp
