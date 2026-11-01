#pragma once

// Frozen NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 geometry (compile-time
// passport).
//
// Source of truth:
// /mnt/llm_models/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4/config.json
// (model_type nemotron_h, NemotronHForCausalLM) + the safetensors inventory
// audit of the 52 shards.
//
// This is the SIBLING VARIANT of `nemotron-3.5-lightning-30b-a3b`, which is
// ingested from the BF16 distribution of the same weights. The architecture and
// the geometry are IDENTICAL - every field below was compared field-by-field
// against the BF16 sibling's passport and matches - and the converted layout is
// the same `logestix-nemotron-nvfp4`, which is what lets the sibling's loader,
// expert cache, kernels, model pipeline and tool codec be reused verbatim.
//
// The difference is the SOURCE ENCODING, and it is recorded here too because it
// is a property of this model that a reader of this header would otherwise have
// to go and rediscover:
//
//   routed + shared experts, lm_head : W4A16_NVFP4, group 16
//       U8 packed E2M1 + F8_E4M3 block scale per 16 [R, C/16] + F32
//       weight_scale_2. The block scale is a COMPLETE OCP E4M3 - all three
//       mantissa bits are in play (measured: mantissa {0..7} all present), so
//       the bytes CANNOT be passed through to the repo's pow2-only runtime
//       decoder (29.9% relative L2). The converter dequantizes with the full
//       E4M3 decode and requantizes through `quantize_tensor_gpu`.
//   mamba in_proj / out_proj : per-tensor FP8 (F8_E4M3 + SCALAR F32
//       weight_scale, shape [], plus an F32 input_scale for the activations,
//       which this repo does not quantize)
//   attention q/k/v/o_proj : BF16 (excluded from quantization by ModelOpt) with
//       F32[1] k_scale / v_scale that are the FP8 KV-cache scales - NOT weight
//       dequant factors. Counted and dropped by the converter.
//   router gate.weight : F32 (ModelOpt left the router high precision)
//
// The independence guard requires one namespace per REGISTERED model, which is
// why these constants are re-declared rather than shared: the sibling's
// `nemotron_moe::frozen` namespace belongs to the other model. They are pinned
// against each other by
// `tests/test_nemotron_moe_nvfp4_loader.cpp::"nvfp4 passport is
// self-consistent"`.

#include <cstdint>
#include <string>
#include <vector>

namespace lsxcommon {
namespace nemotron_nvfp4 {
namespace frozen {

// The sibling's kType plus the distribution marker; `detect_model_type`
// normalizes the index's "model" field into exactly this string.
inline constexpr const char *kType = "nemotron-3.5-lightning-30b-a3b-nvfp4";
inline constexpr const char *kModelName = "Nemotron-3.5-Lightning-30B-A3B-NVFP4";

// Emitted layout. Deliberately the SAME tag the BF16 sibling emits: the
// converted trees are structurally identical, so the sibling's loader reads
// both and only the `model` field distinguishes them.
inline constexpr const char *kFormatTag = "logestix-nemotron-nvfp4";

// ── Dense backbone (identical to the sibling) ───────────────────────────────
inline constexpr int kHidden = 2688;
inline constexpr int kLayers = 52;
inline constexpr int kVocab = 131072;
inline constexpr int kMambaLayers = 23;
inline constexpr int kMoeLayers = 23;
inline constexpr int kAttnLayers = 6;
inline constexpr int kAttnLayerIdx[kAttnLayers] = {5, 12, 19, 26, 33, 42};
inline constexpr double kRmsEps = 1e-5;

// Mamba-2 (NemotronHMamba2Mixer).
inline constexpr int kMambaHeads = 64;
inline constexpr int kMambaHeadDim = 64;
inline constexpr int kMambaDInner = kMambaHeads * kMambaHeadDim;      // 4096
inline constexpr int kSsmStateSize = 128;
inline constexpr int kSsmGroups = 8;
inline constexpr int kMambaConvKernel = 4;
inline constexpr int kMambaConvDim =
    kMambaDInner + 2 * kSsmGroups * kSsmStateSize;                     // 6144
inline constexpr int kMambaInProj =
    kMambaDInner + kMambaConvDim + kMambaHeads;                        // 10304
inline constexpr int kMambaGroupSize = kMambaDInner / kSsmGroups;      // 512
inline constexpr int kMambaChunk = 128;
inline constexpr float kDtMin = 0.001f;

// Full attention (NemotronHAttention). These layers are **NoPE** - the
// checkpoint's reference class has no rotary module (identical to the sibling).
// `rope_theta` / `partial_rotary_factor` are vestigial config fields; see the
// 2026-11-01 measurement in nemotron_moe_inference_pipeline.cu.
inline constexpr int kHeads = 32;
inline constexpr int kKVHeads = 2;
inline constexpr int kHeadDim = 128;
inline constexpr double kRopeTheta = 10000.0;

// MoE (NemotronHMoE / NemotronHExperts / NemotronHTopkRouter). Experts are
// non-gated relu2 MLPs.
inline constexpr int kNumRouted = 128;
inline constexpr int kTopK = 6;
inline constexpr int kMoeIntermediate = 1856;
inline constexpr int kSharedIntermediate = 3712;
inline constexpr float kRoutedScale = 2.5f;

// Tokenizer ids. bos/eos/pad come from config.json; generation_config.json adds
// a SECOND stop (id 11), so the stop set is a list and is part of the passport -
// running past it is the silent failure the list exists to prevent.
inline constexpr int kBosTokenId = 1;
inline constexpr int kPadTokenId = 0;
// A plain array, not a std::vector: the count has to be usable in a
// static_assert (the passport's whole job is to be compile-time).
inline constexpr int kEosTokenIds[] = {2, 11};
inline constexpr int kNumEosTokenIds =
    static_cast<int>(sizeof(kEosTokenIds) / sizeof(kEosTokenIds[0]));

// Source-encoding facts (see the header comment). These are NOT geometry, but
// they are frozen properties of this distribution, so they live in the passport
// rather than in a comment nobody re-reads.
inline constexpr int kSourceNvfp4GroupSize = 16;   // one block scale per 16
inline constexpr int kFp8KvScaleTensorsPerAttnLayer = 2;  // k_scale + v_scale

// Passport invariants (MODEL INGEST INVARIANT step 3).
static_assert(kMambaLayers + kMoeLayers + kAttnLayers == kLayers,
              "layer mix must sum to kLayers");
static_assert(kMambaHeads * kMambaHeadDim == kMambaDInner,
              "mamba d_inner geometry");
static_assert(kMambaDInner + kMambaConvDim + kMambaHeads == kMambaInProj,
              "in_proj geometry");
static_assert(kMambaDInner % kSsmGroups == 0, "gated-norm group geometry");
static_assert(kTopK <= kNumRouted, "top-k exceeds expert count");
static_assert(kHeadDim * kHeads == 4096 && kHeadDim * kKVHeads == 256,
              "attention projection geometry");
static_assert(kSourceNvfp4GroupSize == 16,
              "NVFP4 block scales are one per 16 elements");
static_assert(kNumEosTokenIds >= 2,
              "the vault declares a second stop token (id 11)");
static_assert(kEosTokenIds[0] == 2 && kEosTokenIds[1] == 11,
              "the vault's stop set is [2, 11]");

} // namespace frozen
} // namespace nemotron_nvfp4
} // namespace lsxcommon