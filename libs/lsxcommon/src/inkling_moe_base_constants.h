#pragma once

// Frozen Inkling (base) geometry (compile-time passport).
// Source of truth: /mnt/fast_storage/lsx_model/thinkingmachines/Inkling (config.json,
// tokenizer_config.json and all 108 safetensors headers, audited 2026-11-17)
// and the reference semantics in transformers 5.15.1
// `models/inkling/modeling_inkling.py` + `conversion_mapping.py`
// ("inkling_mm_model").
//
// This is the ORIGINAL Inkling architecture (`inkling`), NOT inkling2
// (Inkling-2-Small, a sibling family member with its own passport). The
// differences that break a copied decoder:
//   * attention scale is 1/head_dim (q/k are per-head RMS-normed with learned
//     [128] weights), not head_dim**-0.5;
//   * no L2 clips, no learned attention sinks, no output gate;
//   * no AttnRes depth mixing — a plain residual stream;
//   * four STATIC depthwise causal convolutions per layer, `conv(x)+x` with
//     NO 1/sqrt(k+1) normalizer and no dynamic taps;
//   * expert SwiGLU is plain silu(g)*u — no softcaps;
//   * the router renormalises jointly over the top-6 routed + 2 shared logits
//     (softmax of logsigmoid), x route_scale 8.0 x gate.global_scale; the
//     selection bias steers only WHICH 6 of the 256 routed experts run;
//   * the head divides hidden by logits_mup_width_multiplier 24.0 BEFORE
//     lm_head and applies NO logit softcap;
//   * layers 0..1 carry a dense MLP (intermediate 24576, x mlp.global_scale),
//     layers 2..65 are sparse.
//
// Text tower: 66 decoder layers, hidden 6144, 64 query heads at head_dim 128.
// Hybrid attention: 55 sliding-window layers (window 512, 16 kv heads) and 11
// full-attention layers (8 kv heads) at the (layer+1)%6==0 ids
// {5,11,...,65}. No RoPE anywhere: a learned relative-position bank of d_rel 16
// profiles over rel_extent 1024 backward distances (the sliding layers gather
// over their 512 window); the full layers additionally log-scale q and the
// relative bias from position 128000 (alpha 0.1).
//
// Source precision is pure BF16 (no hf_quant_config.json): the converter
// quantizes every dense linear matrix and every routed expert to NVFP4
// (E2M1 nibbles + per-16 E4M3 scales + a per-tensor global scale); norms,
// biases, the relative banks, the conv taps, the router and the embedding
// tables stay high precision. The vision tower (hmlp, 8 tensors), the audio
// tower (2) and the 8-layer MTP head (160 + mtp.safetensors) are stripped and
// counted, never applied.

#include <array>

namespace lsxcommon::inkling::base {
namespace frozen {

inline constexpr char kModelType[] = "inkling";
inline constexpr char kFormatTag[] = "logestix-inkling-nvfp4";
inline constexpr char kSourceModelType[] = "inkling_mm_model";
// tiktoken BPE (o200k Split pre-tokenizer over a `tiktoken/tokenizer.model`
// mergeable-ranks file), the Kimi / Inkling-2 pattern. The ranks and the
// added_tokens DIFFER from Inkling-2-Small's — never share the arrows.
inline constexpr int kTokenizerKind = 2;

// ── text tower ────────────────────────────────────────────────────────────
inline constexpr int kHidden = 6144;
inline constexpr int kLayers = 66;
inline constexpr int kHeads = 64;
inline constexpr int kHeadDim = 128;
inline constexpr int kKvHeadsFull = 8;
inline constexpr int kKvHeadsSliding = 16;
inline constexpr double kRmsNormEps = 1e-6;
// q/k are per-head RMS-normalized, hence 1/d rather than 1/sqrt(d).
inline constexpr double kAttnScaleDenom = 128.0;
inline constexpr int kConvKernel = 4;

// q_dim (8192) intentionally differs from hidden (6144): 64 heads x 128.
static_assert(kHeads * kHeadDim == 8192);
static_assert(kHeads % kKvHeadsFull == 0);
static_assert(kHeads % kKvHeadsSliding == 0);

// ── hybrid attention tables (0-based) ─────────────────────────────────────
inline constexpr int kFullLayerCount = 11;
inline constexpr int kSlidingLayerCount = kLayers - kFullLayerCount;  // 55
inline constexpr std::array<int, kFullLayerCount> kFullLayers = {
    5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65};
inline constexpr int kSlidingWindow = 512;
inline constexpr int kRelExtent = 1024;  // full layers; sliding uses the window
inline constexpr int kDRel = 16;
inline constexpr int kLogScaleFloor = 128000;
inline constexpr double kLogScaleAlpha = 0.1;

constexpr bool IsFullLayer(int layer)
{
    for (int l : kFullLayers)
        if (l == layer) return true;
    return false;
}

constexpr bool IsSlidingLayer(int layer) { return !IsFullLayer(layer); }

// ── dense MLP prefix ──────────────────────────────────────────────────────
// config.json `dense_mlp_idx` is 2: layers [0, 2) are dense SwiGLU with
// intermediate 24576 and a learned per-layer output global_scale.
inline constexpr int kDenseMlpLayers = 2;
inline constexpr int kDenseIntermediate = 24576;
static_assert(kDenseIntermediate == 4 * kHidden);  // 24576

constexpr bool IsDenseMlpLayer(int layer) { return layer < kDenseMlpLayers; }

// ── routed MoE (layers 2..65) ─────────────────────────────────────────────
inline constexpr int kExperts = 256;
inline constexpr int kTopK = 6;
inline constexpr int kMoeIntermediate = 3072;
inline constexpr int kSharedExperts = 2;
inline constexpr int kRouterRows = kExperts + kSharedExperts;  // 258
inline constexpr double kRouteScale = 8.0;
static_assert(kMoeIntermediate % 16 == 0);
static_assert(kLayers == 66);

// ── head ──────────────────────────────────────────────────────────────────
// hidden / logits_mup_width_multiplier before lm_head; no logit softcap
// (final_logit_softcapping is null in the checkpoint).
inline constexpr double kLogitsMupDiv = 24.0;

// ── quantization ──────────────────────────────────────────────────────────
inline constexpr int kQuantBlock = 16;  // NVFP4 scale period

// ── vocabulary / special ids ──────────────────────────────────────────────
inline constexpr int kVocab = 201024;      // padded embed/unembed rows
inline constexpr int kBaseVocab = 200058;  // real tokens; logits are sliced
inline constexpr int kAddedTokenBase = 199998;
inline constexpr int kBosId = -1;          // the checkpoint has no BOS
inline constexpr int kPadId = -1;
// config.json `eos_token_id` is the end-of-sampling marker, not <|endoftext|>.
inline constexpr int kEosId = 200006;      // <|content_model_end_sampling|>
inline constexpr int kStopIdAlt = 199999;  // <|endoftext|> also terminates

// Chat markers (tokenizer_config.json added_tokens_decoder, all special:true;
// verified identical ids to Inkling-2-Small on 2026-11-17 — the RANKS differ).
inline constexpr int kMarkerUser = 200000;         // <|message_user|>
inline constexpr int kMarkerModel = 200001;        // <|message_model|>
inline constexpr int kMarkerSystem = 200002;       // <|message_system|>
inline constexpr int kMarkerTool = 200003;         // <|message_tool|>
inline constexpr int kMarkerText = 200004;         // <|content_text|>
inline constexpr int kMarkerEndSampling = 200006;  // <|content_model_end_sampling|>
inline constexpr int kMarkerThinking = 200008;     // <|content_thinking|>
inline constexpr int kMarkerEndMessage = 200010;   // <|end_message|>
inline constexpr int kMarkerXml = 200024;          // <|content_xml|>
inline constexpr int kMarkerInvokeJson = 200049;   // <|content_invoke_tool_json|>

}  // namespace frozen
}  // namespace lsxcommon::inkling::base