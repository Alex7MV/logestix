#pragma once

// Frozen TML-d24-varloop-research passport (compile-time geometry).
//
// Source of truth: /mnt/fast_storage/huggingface/TML-d24-varloop-research
// (config.json + hf_quant_config.json + safetensors index + tokenizer.json
// added_tokens, audited 2026-11-22) and the reference semantics in
// transformers 5.24.0 `models/inkling2_loop/modeling_inkling2_loop.py`.
//
// Stage-0 classification (2026-11-22): a SIBLING of Inkling-2-Mini, not a new
// family. `architectures: ["Inkling2LoopForConditionalGeneration"]`,
// `model_type: inkling2_loop`, and the ONLY config delta against
// /mnt/fast_storage/huggingface/Inkling-2-Mini is
// `text_config.loop_config.is_varloop` (true here, false there) — every other
// field, the 1244-tensor index (total_size 19 525 870 752), the tokenizer, the
// chat template and hf_quant_config.json are byte-identical. The weights are a
// different fine-tune, so the model answers differently and owns its own
// goldens.
//
// This header therefore repeats Mini's verified numbers instead of including
// its passport (model independence: one frozen namespace per model), and adds
// the numbers only this row has — the legal pass range and the cache-slab size
// that range requires.
//
// ── the architecture in one paragraph ───────────────────────────────────────
// 24 physical MoE decoder layers, hidden 2304, 24 query heads / 8 kv heads at
// head_dim 128, no RoPE anywhere (q/k are L2-clipped, not rotated). Attention
// is a hybrid: 20 sliding-window layers (1024) and 4 full-attention layers at
// the (layer+1)%6==0 ids {5,11,17,23}. A learned relative-position bias bank of
// d_rel 16 profiles over 1024 backward distances replaces RoPE, and the full
// layers additionally log-scale q and the relative states from position 8192
// (Small's floor is 128000 — do NOT copy it). Depth mixing ("AttnRes"): every
// layer's attention input and MLP input is a single-head mixture over a stack
// of block prefix sums, one block every 6 layers.
//
// ── the loop, and what VARLOOP adds ─────────────────────────────────────────
// Layers [6,18) run `passes` times with shared weights. Pass 0 runs inline;
// every later pass restarts from the pre-loop hidden + block states and
// receives per-layer "messages" retrieved by a single-head attention over the
// PREVIOUS pass's block stack + final hidden. Each loop layer carries four
// rank-64 MessageGate controllers (q/k/v/mlp).
//
// `is_varloop: true` is a capability declaration, not a second code path: the
// release reads `loop_num_passes` from `loop_config.num_passes` (3 here) and
// appends `passes-1` deep copies of the loop layers' cache state, whatever that
// number is. The model card states the weights were trained with 1..5 passes,
// sampled 1:2:6:2:1, and post-trained at 3. So:
//   * DEFAULT = 3, the checkpoint's own declared value;
//   * LEGAL RANGE = [1, 5]; the engine clamps instead of running an untrained
//     count (5+ degrades, and 6+ is outside anything the weights saw);
//   * the KV/conv SLAB is sized for the maximum (kCacheLayersMax = 72 slots), so
//     choosing a smaller count uses a prefix of the slab and never re-allocates.
//
// Passes share the weights: only the KV + conv STATE is per pass, which is why
// CacheSlot(layer, pass) — the release's own cache offset — is the whole
// difference between a 1-pass and a 5-pass run.
//
// ── precision ───────────────────────────────────────────────────────────────
// Routed experts of layers 0..19 are NVFP4 (E2M1 nibbles + per-16 E4M3 scales
// + a per-expert global scale amax/2688), layers 20..23 are MXFP8 (E4M3 +
// ue8m0 scale per 32). Everything else is BF16 except the AttnRes queries, the
// cross-pass queries and the router bias, which are FP32. The vision tower
// (422 tensors) and mtp.safetensors are stripped and counted, never applied.
//
// MoE: 128 routed experts, top-8, no shared experts, sigmoid scoring with a
// selection-only FP32 correction bias, weights renormalised to 1 then scaled by
// 8.0. Experts are softcapped SwiGLU: tanh(gate/4)*4 * sigmoid(gate) *
// tanh(up/25)*25, accumulated in FP32.

#include <array>

namespace lsxcommon::inkling::varloop {
namespace frozen {

inline constexpr char kModelType[] = "tml-d24-varloop-research";
// The CARD name the converter writes into model.index.json: path
// auto-detection normalizes it (lower-case) into kModelType, and the vault
// directory is this name.
inline constexpr char kModelCard[] = "TML-d24-varloop-research";
inline constexpr char kFormatTag[] = "logestix-inkling-2-varloop";
inline constexpr char kSourceModelType[] = "inkling2_loop";
// tiktoken BPE (o200k Split pre-tokenizer over a `tiktoken/tokenizer.model`
// mergeable-ranks file), byte-identical to Inkling-2-Mini's.
inline constexpr int kTokenizerKind = 2;
// The capability this model exists for: a runtime-chosen pass count.
inline constexpr bool kVarLoop = true;

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
inline constexpr std::array<int, kFullLayerCount> kFullLayers = {5, 11, 17, 23};
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
inline constexpr int kLoopGateRank = 64;  // loop_config.gate_rank
inline constexpr int kLoopLayers = kLoopEnd - kLoopStart;  // 12
// loop_config.num_passes: what the checkpoint declares and was post-trained at.
inline constexpr int kLoopPassesDefault = 3;
// The trained range (model card §3): "trained with 1 to 5 passes ... 1:2:6:2:1
// ratio", so 1 and 5 are both inside what the weights saw and 6 is not.
inline constexpr int kLoopPassesMin = 1;
inline constexpr int kLoopPassesMax = 5;
inline constexpr int kLayerApplicationsDefault =
    kLayers + (kLoopPassesDefault - 1) * kLoopLayers;  // 48
// The KV + conv slab is sized for the MAXIMUM count: 24 + 4*12 = 72 slots.
inline constexpr int kCacheLayersMax =
    kLayers + (kLoopPassesMax - 1) * kLoopLayers;
// Four messages per loop layer: q, k, v, mlp inputs.
inline constexpr int kLoopMessages = 4;

constexpr bool IsLoopLayer(int layer)
{
    return layer >= kLoopStart && layer < kLoopEnd;
}

// Layer APPLICATIONS per token for a given pass count.
constexpr int LayerApplications(int passes)
{
    return kLayers + (passes - 1) * kLoopLayers;
}

// Slot of (physical layer, pass) in the cache slab. Identical to the release's
// `num_hidden_layers + (loop_idx-1)*loop_layers - loop_start` cache offset:
// pass 0 uses the physical slot, pass p a block appended after the stack.
constexpr int CacheSlot(int layer, int pass)
{
    return pass == 0 ? layer
                     : kLayers + (pass - 1) * kLoopLayers + (layer - kLoopStart);
}

// Requested pass count -> the one the engine will run. Absent or non-positive
// means the checkpoint's own default; out-of-range values clamp into the
// trained range rather than running a count the weights never saw.
constexpr int ResolvePasses(int requested)
{
    if (requested <= 0) return kLoopPassesDefault;
    if (requested < kLoopPassesMin) return kLoopPassesMin;
    if (requested > kLoopPassesMax) return kLoopPassesMax;
    return requested;
}

// ── AttnRes depth mixer ───────────────────────────────────────────────────
inline constexpr int kAttnResBlock = 6;
inline constexpr int kAttnResBlocks = kLayers / kAttnResBlock;  // 4
// Block boundaries are layer%6==0; the stack never holds more than
// boundaries-seen + 1 (the live hidden state). Independent of the pass count:
// each pass restarts from the snapshot the loop core was entered with.
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
inline constexpr int kQuantBlock = 16;                    // NVFP4 scale period
inline constexpr double kOuterScaleDenominator = 2688.0;  // 6 * 448
// Layers 0..19 store the routed experts as NVFP4; 20..23 as MXFP8.
inline constexpr int kNovfp4ExpertLayerCount = 20;
inline constexpr int kMxfp8ExpertLayerCount =
    kLayers - kNovfp4ExpertLayerCount;
inline constexpr int kMxfp8Block = 32;

constexpr bool ExpertIsNovfp4(int layer) { return layer < kNovfp4ExpertLayerCount; }

// ── vocabulary / special ids ──────────────────────────────────────────────
inline constexpr int kVocab = 201024;      // padded embed/unembed rows
inline constexpr int kBaseVocab = 200058;  // real tokens; logits are sliced
inline constexpr int kAddedTokenBase = 199998;
inline constexpr int kBosId = -1;  // the checkpoint has no BOS
// config.json `eos_token_id` is the end-of-sampling marker, and
// special_tokens_map.json names it as eos too.
inline constexpr int kEosId = 200006;       // <|content_model_end_sampling|>
inline constexpr int kStopIdAlt = 199999;   // <|endoftext|> also terminates
inline constexpr int kPadId = 199999;       // generation_config pad_token_id

// ── chat / tool markers (the byte-identical inkling2_loop template) ───────
inline constexpr int kMarkerUnused = 199998;        // <|unused|>
inline constexpr int kMarkerUser = 200000;          // <|message_user|>
inline constexpr int kMarkerModel = 200001;         // <|message_model|>
inline constexpr int kMarkerSystem = 200002;        // <|message_system|>
inline constexpr int kMarkerTool = 200003;          // <|message_tool|>
inline constexpr int kMarkerText = 200004;          // <|content_text|>
inline constexpr int kMarkerImage = 200005;         // <|content_image|>
inline constexpr int kMarkerEndSampling = 200006;   // <|content_model_end_sampling|>
inline constexpr int kMarkerThinking = 200008;      // <|content_thinking|>
inline constexpr int kMarkerEndMessage = 200010;    // <|end_message|>
inline constexpr int kMarkerToolError = 200022;     // <|content_tool_error|>
inline constexpr int kMarkerXml = 200024;           // <|content_xml|>
inline constexpr int kMarkerBeginOfText = 200028;   // <|begin_of_text|>
inline constexpr int kMarkerInvokeJson = 200049;    // <|content_invoke_tool_json|>
inline constexpr int kMarkerInvokeText = 200057;    // <|content_invoke_tool_text|>
inline constexpr int kImagePlaceholder = 200054;    // processor_config image_token

// ── frozen-geometry invariant checks ──────────────────────────────────────
static_assert(kHidden == 2304, "this checkpoint has hidden 2304");
static_assert(kLayers == 24, "24 physical decoder layers");
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
static_assert(kLoopPassesMin >= 1, "at least one pass");
static_assert(kLoopPassesDefault >= kLoopPassesMin &&
                  kLoopPassesDefault <= kLoopPassesMax,
              "the checkpoint's declared count lies inside the trained range");
static_assert(kLayerApplicationsDefault == 48,
              "24 physical + 2 extra passes x 12 loop layers = 48 applications");
static_assert(kCacheLayersMax == 72,
              "the slab sizes for the maximum: 24 + 4 x 12 = 72 slots");
static_assert(kSlidingWindow == kRelExtent,
              "sliding layers gather the relative bias over the window");
static_assert(kTopK > 0 && kTopK <= kExperts, "bad top-k");
static_assert(kMoeIntermediate * 2 == kExpertGateUpRows, "gate/up concat width");
static_assert(kNovfp4ExpertLayerCount + kMxfp8ExpertLayerCount == kLayers,
              "the expert codec tables must cover every layer");
static_assert(kBaseVocab + 1024 >= kVocab, "padded vocabulary sanity");
static_assert(kBosId < 0, "the checkpoint carries no BOS");
static_assert(kEosId == kMarkerEndSampling,
              "the configured eos is the end-of-sampling marker");
static_assert(sizeof(float) == 4, "FP32 assumptions");

}  // namespace frozen
}  // namespace lsxcommon::inkling::varloop
