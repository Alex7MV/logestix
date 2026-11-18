#pragma once

// Muse-Glimmer-30B frozen geometry passport (Model Ingestion Protocol step 3).
//
// A NEW dense family: `MuseGlimmerForConditionalGeneration` with text tower
// `muse_glimmer_text` (transformers 5.15.1 models/muse_glimmer). 52 layers of
// gated GQA attention ([sliding, sliding, sliding, full] x 13; the full layers
// are NoPE), four CENTERED norms per layer (x * (1 + w)), a weightless RMSNorm
// over the gathered embeddings (no sqrt(hidden) scale), SwiGLU (silu) MLP and
// softcapped logits with an output multiplier. The vision tower + adapter are
// ballast: counted, never applied. A different geometry is a NEW family in
// this repo: it owns this passport, its own converter/loader/kernels and its
// own namespace, and it borrows no constant from any other model.
//
// This header is a LEAF: it includes no other model's passport
// (model-independence invariant). It is device-safe (no abseil, no arrow) so
// the .cu inference layer can include it.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/vram_estimate.h"

namespace lsxcommon {
namespace muse_dense {

// Loaded/converted topology of a Muse-Glimmer checkpoint. Device-safe so the
// .cu layer can include it; populated by the converter from config.json and
// re-read from <root>/logestix/model.index.json by the loader.
struct MuseDenseTopology {
    int num_layers = 52;
    // "sliding_attention" | "full_attention", one per layer.
    std::vector<std::string> layer_types;
    // Per-layer RoPE base; 0 disables rotary (NoPE) for that layer.
    std::vector<double> layer_rope_theta;
    int hidden_size = 6656;
    int intermediate_size = 19968;
    int vocab_size = 202048;
    int num_attention_heads = 32;
    int num_key_value_heads = 2;
    int head_dim = 128;
    int sliding_window = 2048;
    float rms_norm_eps = 1e-5f;
    float post_norm_eps = 1e-8f;
    float rope_theta = 500000.0f;
    float qk_scale_factor = 3.87f;
    float output_multiplier = 0.19611613513818404f;
    float final_logit_softcapping = 20.0f;
    int bos_id = 200000;
    std::vector<int> eos_ids;
    int pad_id = 200018;
    int total_pinned_tensors = 0;
    int skipped_visual_tensors = 0;
    VramProfile vram;         // conversion-time VRAM requirements (mandatory)
    std::string error_msg;    // non-empty when config/index.json is invalid

    // Per-layer attention geometry. window <= 0 means "attend to the whole
    // causal context"; theta 0 means NoPE (no rotary at all on that layer).
    bool IsFull(int layer) const
    {
        return layer >= 0 && static_cast<size_t>(layer) < layer_types.size() &&
               layer_types[static_cast<size_t>(layer)] == "full_attention";
    }
    int Window(int layer) const { return IsFull(layer) ? 0 : sliding_window; }
    double RopeTheta(int layer) const
    {
        if (static_cast<size_t>(layer) < layer_rope_theta.size())
            return layer_rope_theta[static_cast<size_t>(layer)];
        return IsFull(layer) ? 0.0 : static_cast<double>(rope_theta);
    }
    int QRows() const { return num_attention_heads * head_dim; }
    int KvRows() const { return num_key_value_heads * head_dim; }
};

namespace frozen {

// ── Identity ────────────────────────────────────────────────────────────────
inline constexpr const char *kModelType = "muse-glimmer-30b";
inline constexpr const char *kFormatTag = "logestix-muse-nvfp4";
inline constexpr const char *kSourceModelType = "muse_glimmer";
inline constexpr int kTokenizerKind = 3;   // o200k/Llama-3 byte-level BPE

// ── Frozen compile-time geometry (config.json + safetensors, 2026-11-18) ───
inline constexpr int kHiddenSize = 6656;
inline constexpr int kNumLayers = 52;
inline constexpr int kIntermediateSize = 19968;
inline constexpr int kVocabSize = 202048;
inline constexpr int kBaseVocabSize = 200000;   // BPE table; rest are added
inline constexpr int kNumHeads = 32;
inline constexpr int kNumKVHeads = 2;
inline constexpr int kHeadDim = 128;
inline constexpr int kQRows = kNumHeads * kHeadDim;      // 4096 (NOT hidden)
inline constexpr int kKvRows = kNumKVHeads * kHeadDim;   // 256
inline constexpr int kSlidingWindow = 2048;
inline constexpr int kMaxPositionEmbeddings = 131072;

// The [sliding, sliding, sliding, full] pattern: layer l is full iff
// l % 4 == 3 (3, 7, …, 51 — verified against config.json `layer_types`).
inline constexpr int kFullLayerPeriod = 4;
inline constexpr int kNumFullLayers = kNumLayers / kFullLayerPeriod;   // 13
inline constexpr int kNumSlidingLayers = kNumLayers - kNumFullLayers;  // 39

constexpr bool IsFullLayer(int layer) { return layer % kFullLayerPeriod == 3; }

// RoPE theta 500000 on the sliding layers ONLY; the full-attention layers
// are NoPE (config `layer_rope_theta[i] == 0` -> position_embeddings=None).
inline constexpr double kRopeTheta = 500000.0;
constexpr double RopeThetaForLayer(int layer)
{
    return IsFullLayer(layer) ? 0.0 : kRopeTheta;
}
constexpr int WindowForLayer(int layer)
{
    return IsFullLayer(layer) ? 0 : kSlidingWindow;
}

// ── Attention semantics ─────────────────────────────────────────────────────
// Weightless per-head RMSNorm on q and k (Gemma4RMSNorm with_scale=False,
// eps rms_norm_eps), then Q *= qk_scale_factor, then the STANDARD
// 1/sqrt(head_dim) score scale (AfmoeAttention.scaling). The attention output
// is multiplied elementwise by sigmoid(self_attn.gate_proj(x)) BEFORE o_proj.
inline constexpr float kQkScaleFactor = 3.87f;
inline constexpr float kAttentionScale = 0.08838834764831845f;  // 1/sqrt(128)

// ── Norms ───────────────────────────────────────────────────────────────────
// All four per-layer norms are CENTERED (Gemma2RMSNorm: out = rms(x)*(1+w));
// input/pre-ffn use rms_norm_eps, post-attn/post-ffn use post_norm_eps.
// The final norm is Gemma4RMSNorm WITH scale (plain rms(x)*w, rms_norm_eps).
// The embedding norm is weightless (rms(x), rms_norm_eps), NO sqrt(hidden).
inline constexpr double kRmsNormEps = 1e-5;
inline constexpr double kPostNormEps = 1e-8;

// ── Logits ──────────────────────────────────────────────────────────────────
// logits = cap * tanh(raw * multiplier / cap). The multiplier equals
// 1/sqrt(hidden/256); dropping it ships ~5.1x hot logits that greedy argmax
// hides (a positive scale preserves the ordering).
inline constexpr float kFinalLogitSoftcap = 20.0f;
inline constexpr float kOutputMultiplier = 0.19611613513818404f;

// ── Special tokens (tokenizer.json added_tokens / generation_config.json) ──
inline constexpr int32_t kBosId = 200000;         // <|begin_of_text|>
inline constexpr int32_t kEosIdText = 200001;     // <|end_of_text|>
inline constexpr int32_t kEosIdTurn = 200008;     // <|eot|>
inline constexpr int32_t kPadId = 200018;         // <|finetune_right_pad|>
inline constexpr int32_t kMarkerEom = 200007;     // <|eom|>
inline constexpr int32_t kMarkerStart = 200022;   // <|start|>
inline constexpr int32_t kMarkerMessage = 200023; // <|message|>
inline constexpr int32_t kMarkerImage = 200090;   // <|image|>
inline constexpr int32_t kMarkerVideo = 200091;   // <|video|>
inline constexpr int32_t kMarkerPatch = 200092;   // <|patch|>

// ── Geometry invariants ─────────────────────────────────────────────────────
static_assert(kHiddenSize > 0 && kIntermediateSize > 0 && kVocabSize > 0,
              "positive Muse-Glimmer geometry required");
static_assert(kNumHeads % kNumKVHeads == 0 && kNumKVHeads > 0,
              "num_attention_heads must be a multiple of num_key_value_heads");
static_assert(kQRows != kHiddenSize,
              "q/o rows are heads*head_dim, not hidden_size — the o_proj is "
              "[hidden, q_rows], not square");
static_assert(kNumFullLayers * kFullLayerPeriod == kNumLayers,
              "the 3:1 sliding/full pattern must divide the depth exactly");
static_assert(kBaseVocabSize < kVocabSize,
              "added special tokens live above the BPE table");
static_assert(kBosId >= kBaseVocabSize && kMarkerPatch < kVocabSize,
              "special ids must sit in the added-token range");

} // namespace frozen
} // namespace muse_dense
} // namespace lsxcommon