#pragma once

// Frozen Inkling-2-Mini geometry (compile-time passport).
// Source of truth: /mnt/fast_storage/lsx_model/thinkingmachines/Inkling-2-Mini (config.json +
// hf_quant_config.json + safetensors headers, audited 2026-11-21) and the
// reference semantics in transformers 5.24.0
// `models/inkling2_loop/modeling_inkling2_loop.py`.
//
// Text tower: 24 PHYSICAL MoE decoder layers, hidden 2304, 24 query heads / 8
// kv heads at head_dim 128, no RoPE anywhere (q/k are L2-clipped, not rotated).
// Attention is a hybrid: 20 sliding-window layers (1024) and 4 full-attention
// layers at the (layer+1)%6==0 ids {5,11,17,23}. A learned relative-position
// bias bank of d_rel 16 profiles over 1024 backward distances replaces RoPE,
// and the full layers additionally log-scale q and the relative states from
// position 8192 (Small's floor is 128000 — do NOT copy it).
//
// THE LOOP: layers [6,18) run THREE passes with shared weights — 48 layer
// applications per token from 24 layers of weights. Pass 0 runs inline; passes
// 1/2 restart from the pre-loop hidden + block states and receive per-layer
// "messages" retrieved by a single-head attention (message_input.query,
// [12][4][2304] FP32) over the PREVIOUS pass's block stack + final hidden.
// Each loop layer carries four rank-64 MessageGate controllers (q/k/v/mlp):
//   gate = down(concat(RMSnorm(h), RMSnorm(msg)))   # down [64, 4608]
//   h'   = (1 - tanh(forget(gate))) * h + tanh(gain(gate)) * msg
// applied AFTER the AttnRes mix and BEFORE the input norm (the mlp gate after
// the mlp_input mix). Every pass owns its OWN KV + conv cache slots:
// 24 + 2*12 = 48 cache layers (CacheSlot below).
//
// Depth mixing ("AttnRes"): every layer's attention input (layers > 0) and MLP
// input is a single-head scaled-dot-product mixture over a stack of block
// prefix sums; a block is 6 layers and its boundary (layer%6==0) appends the
// incoming hidden state to the stack instead of keeping a local residual. The
// 4 full layers ask for THREE mixtures (separate q/k/v inputs); sliding layers
// and every MLP input ask for one; the final norm input uses `output_query`.
// `attn_res.attn_queries.i` belongs to layer i+1; `mlp_queries.i` to layer i.
//
// MoE: 128 routed experts, top-8, no shared experts, sigmoid scoring with a
// selection-only FP32 correction bias, weights renormalised to 1 then scaled
// by 8.0. Experts are softcapped SwiGLU: tanh(gate/4)*4 * sigmoid(gate) *
// tanh(up/25)*25, accumulated in FP32.
//
// Source precision is MIXED and undocumented in config.json: routed experts of
// layers 0..19 are NVFP4 (E2M1 nibbles + per-16 E4M3 scales + a per-expert
// global scale amax/2688), layers 20..23 are MXFP8 (E4M3 + ue8m0 scale per 32).
// Everything else is BF16 except the AttnRes queries, the cross-pass queries
// and the router bias, which are FP32. The vision tower (422 tensors) and the
// 3-layer MTP head (mtp.safetensors, NOT in the index) are stripped and
// counted, never applied.

#include <array>

namespace lsxcommon::inkling::loop {
namespace frozen {

inline constexpr char kModelType[] = "inkling-2-mini";
inline constexpr char kFormatTag[] = "logestix-inkling-2-mini";
inline constexpr char kSourceModelType[] = "inkling2_loop";
// tiktoken BPE (o200k Split pre-tokenizer over a `tiktoken/tokenizer.model`
// mergeable-ranks file), byte-identical to Inkling-2-Small's.
inline constexpr int kTokenizerKind = 2;

// ── text tower ────────────────────────────────────────────────────────────
inline constexpr int kHidden = 2304;
inline constexpr int kLayers = 24;  // PHYSICAL layers; the loop adds applications
inline constexpr int kHeads = 24;
inline constexpr int kKvHeads = 8;
inline constexpr int kHeadDim = 128;
inline constexpr int kGqaGroups = kHeads / kKvHeads;
inline constexpr double kRmsNormEps = 1e-6;
inline constexpr double kAttnScaleDenom = 128.0;  // scaling = head_dim**-0.5
inline constexpr int kConvKernel = 4;

// ── hybrid attention tables (0-based) ─────────────────────────────────────
inline constexpr int kFullLayerCount = 4;
inline constexpr int kSlidingLayerCount = kLayers - kFullLayerCount;
inline constexpr std::array<int, kFullLayerCount> kFullLayers = {
    5, 11, 17, 23};
inline constexpr int kSlidingWindow = 1024;
inline constexpr int kRelExtent = 1024;  // full layers; sliding uses the window
inline constexpr int kDRel = 16;
inline constexpr double kQkL2Clip = 22.627416997969522;  // 16*sqrt(2)
inline constexpr double kRelL2Clip = 8.0;
inline constexpr int kLogScaleFloor = 8192;
inline constexpr double kLogScaleAlpha = 0.1;

constexpr bool IsFullLayer(int layer)
{
    for (int l : kFullLayers)
        if (l == layer) return true;
    return false;
}

constexpr bool IsSlidingLayer(int layer) { return !IsFullLayer(layer); }

// ── the loop core ─────────────────────────────────────────────────────────
inline constexpr int kLoopStart = 6;   // loop_config.core_start
inline constexpr int kLoopEnd = 18;    // loop_config.core_end (exclusive)
inline constexpr int kLoopPasses = 3;  // loop_config.num_passes
inline constexpr int kLoopGateRank = 64;  // loop_config.gate_rank
inline constexpr int kLoopLayers = kLoopEnd - kLoopStart;  // 12
inline constexpr int kLayerApplications =
    kLayers + (kLoopPasses - 1) * kLoopLayers;  // 48
// Each pass owns its own KV + conv state slots, appended after the physical
// stack (the reference appends deepcopy cache layers per pass).
inline constexpr int kCacheLayers = kLayerApplications;  // 48
// Four messages per loop layer: q, k, v, mlp inputs.
inline constexpr int kLoopMessages = 4;

constexpr bool IsLoopLayer(int layer)
{
    return layer >= kLoopStart && layer < kLoopEnd;
}

// Cache slot of (physical layer, pass). Pass 0 uses the physical slot; pass p
// (1-based extra passes) uses kLayers + (p-1)*kLoopLayers + (layer-kLoopStart).
// Non-loop layers only ever run in pass 0.
constexpr int CacheSlot(int layer, int pass)
{
    return pass == 0 ? layer
                     : kLayers + (pass - 1) * kLoopLayers + (layer - kLoopStart);
}

// ── AttnRes depth mixer ───────────────────────────────────────────────────
inline constexpr int kAttnResBlock = 6;
inline constexpr int kAttnResBlocks = kLayers / kAttnResBlock;  // 4
// Block boundaries are layer%6==0; the stack never holds more than
// boundaries-seen + 1 (the live hidden state).
inline constexpr int kAttnResMaxSources = kAttnResBlocks + 1;
inline constexpr int kFullLayerMixQueries = 3;  // q, k, v inputs
inline constexpr int kSingleMixQueries = 1;

constexpr bool IsBlockBoundary(int layer) { return layer % kAttnResBlock == 0; }

// ── routed MoE ────────────────────────────────────────────────────────────
inline constexpr int kExperts = 128;
inline constexpr int kTopK = 8;
inline constexpr int kMoeIntermediate = 1152;
inline constexpr int kSharedExperts = 0;
inline constexpr double kRouteScale = 8.0;
inline constexpr double kMlpGateSoftcap = 4.0;
inline constexpr double kMlpUpSoftcap = 25.0;
inline constexpr double kFinalLogitSoftcap = 30.0;
inline constexpr int kExpertGateUpRows = 2 * kMoeIntermediate;  // 2304

// ── source quantization ───────────────────────────────────────────────────
inline constexpr int kQuantBlock = 16;                 // NVFP4 scale period
inline constexpr double kOuterScaleDenominator = 2688.0;  // 6 * 448 (E2M1 max * E4M3 max)
// Layers 0..19 store the routed experts as NVFP4; 20..23 as MXFP8.
inline constexpr int kNovfp4ExpertLayerCount = 20;
inline constexpr int kMxfp8ExpertLayerCount = kLayers - kNovfp4ExpertLayerCount;
inline constexpr int kMxfp8Block = 32;

constexpr bool ExpertIsNovfp4(int layer) { return layer < kNovfp4ExpertLayerCount; }

// ── vocabulary / special ids ──────────────────────────────────────────────
inline constexpr int kVocab = 201024;        // padded embed/unembed rows
inline constexpr int kBaseVocab = 200058;    // real tokens; logits are sliced
inline constexpr int kAddedTokenBase = 199998;
inline constexpr int kBosId = -1;            // the checkpoint has no BOS
// config.json `eos_token_id` is the end-of-sampling marker, not <|endoftext|>.
inline constexpr int kEosId = 200006;        // <|content_model_end_sampling|>
inline constexpr int kStopIdAlt = 199999;    // <|endoftext|> also terminates
inline constexpr int kPadId = 199999;        // generation_config.json pad_token_id

// ── chat / tool markers (Inkling2Loop chat_template.jinja) ────────────────
inline constexpr int kMarkerUnused = 199998;    // <|unused|>
inline constexpr int kMarkerUser = 200000;      // <|message_user|>
inline constexpr int kMarkerModel = 200001;     // <|message_model|>
inline constexpr int kMarkerSystem = 200002;    // <|message_system|>
inline constexpr int kMarkerTool = 200003;      // <|message_tool|>
inline constexpr int kMarkerText = 200004;      // <|content_text|>
inline constexpr int kMarkerImage = 200005;     // <|content_image|>
inline constexpr int kMarkerEndSampling = 200006;  // <|content_model_end_sampling|>
inline constexpr int kMarkerThinking = 200008;  // <|content_thinking|>
inline constexpr int kMarkerEndMessage = 200010;  // <|end_message|>
inline constexpr int kMarkerToolError = 200022;   // <|content_tool_error|>
inline constexpr int kMarkerXml = 200024;         // <|content_xml|>
inline constexpr int kMarkerBeginOfText = 200028;  // <|begin_of_text|>
inline constexpr int kMarkerInvokeJson = 200049;   // <|content_invoke_tool_json|>
inline constexpr int kMarkerInvokeText = 200057;   // <|content_invoke_tool_text|>
inline constexpr int kImagePlaceholder = 200054;   // processor_config image_token

// ── frozen-geometry invariant checks ──────────────────────────────────────
static_assert(kHidden == 2304, "Inkling-2-Mini has hidden 2304");
static_assert(kLayers == 24, "Inkling-2-Mini has 24 physical decoder layers");
static_assert(kHeads * kHeadDim == 3072, "query projection width");
static_assert(kKvHeads * kHeadDim == 1024, "key/value projection width");
static_assert(kGqaGroups == 3, "one kv head serves three query heads");
static_assert(kFullLayerCount + kSlidingLayerCount == kLayers,
              "the hybrid tables must partition the stack");
static_assert(kFullLayers.back() == kLayers - 1,
              "the final layer is full attention");
static_assert(kAttnResBlock * kAttnResBlocks == kLayers,
              "AttnRes blocks must tile the stack");
static_assert(kLoopStart > 0 && kLoopStart % kAttnResBlock == 0 &&
                  kLoopEnd % kAttnResBlock == 0,
              "the loop core must start/end on AttnRes block boundaries");
static_assert(kLoopStart < kLoopEnd && kLoopEnd <= kLayers,
              "the loop core must sit inside the stack");
static_assert(kLoopPasses >= 1, "at least one pass");
static_assert(kCacheLayers == 48,
              "24 physical + 2 extra passes x 12 loop layers = 48 cache slots");
static_assert(kSlidingWindow == kRelExtent,
              "sliding layers gather the relative bias over the window");
static_assert(kTopK > 0 && kTopK <= kExperts, "bad top-k");
static_assert(kMoeIntermediate * 2 == kExpertGateUpRows, "gate/up concat width");
static_assert(kNovfp4ExpertLayerCount + kMxfp8ExpertLayerCount == kLayers,
              "the expert codec tables must cover every layer");
static_assert(kBaseVocab + 1024 >= kVocab, "padded vocabulary sanity");
static_assert(kBosId < 0, "the checkpoint carries no BOS");
static_assert(kMarkerUser == 200000 && kMarkerInvokeText == 200057,
              "marker ids come from the checkpoint added_tokens band");
static_assert(kEosId == kMarkerEndSampling,
              "the configured eos is the end-of-sampling marker");
static_assert(kStopIdAlt == 199999, "<|endoftext|> id from added_tokens");
static_assert(sizeof(float) == 4, "FP32 assumptions");

}  // namespace frozen
}  // namespace lsxcommon::inkling::loop