#pragma once

// Frozen Nemotron-3.5-Lightning-30B-A3B geometry (compile-time passport).
// Source of truth: /mnt/llm_models/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/config.json
// (model_type nemotron_h) + safetensors inventory audit (2026-09-16): 52 layers
// (23 mamba + 23 moe + 6 full attention at 5/12/19/26/33/42), hidden 2688,
// mamba2 d_inner 4096 (64x64), n_groups 8, state 128, conv 4, in_proj 10304,
// MoE 128 experts top-6 relu2 I=1856 + shared I=3712, vocab 131072, eps 1e-5.

#include <cstdint>

namespace lsxcommon::nemotron_moe {
namespace frozen {

inline constexpr const char *kType = "nemotron-3.5-lightning-30b-a3b";

// The `model` string this distribution writes into model.index.json. Both
// siblings emit the `logestix-nemotron-nvfp4` format tag, so this is the only
// thing that tells their trees apart - and they carry separate parity goldens,
// so a mixed-up tree must fail Load rather than verify against the wrong set
// (the NVFP4 leaf has enforced this since ingestion; the BF16 leaf did not
// until 2026-11-13).
inline constexpr const char *kModelName = "Nemotron-3.5-Lightning-30B-A3B";

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
inline constexpr int kMambaConvDim = kMambaDInner + 2 * kSsmGroups * kSsmStateSize; // 6144
inline constexpr int kMambaInProj = kMambaDInner + kMambaConvDim + kMambaHeads;     // 10304
inline constexpr int kMambaGroupSize = kMambaDInner / kSsmGroups;     // 512
inline constexpr int kMambaChunk = 128;                               // reference only (milestone 1)
inline constexpr float kDtMin = 0.001f;

// Full attention (NemotronHAttention). These layers are **NoPE**: the
// checkpoint's own reference class has no rotary module and ignores
// `position_ids`, so q/k carry no positional rotation (position comes from the
// Mamba-2 layers). `kRopeTheta` is kept because `rope_theta` /
// `partial_rotary_factor` are checkpoint config fields, NOT because they are
// applied - see the 2026-11-01 measurement in the pipeline TU.
inline constexpr int kHeads = 32;
inline constexpr int kKVHeads = 2;
inline constexpr int kHeadDim = 128;
inline constexpr double kRopeTheta = 10000.0;

// MoE (NemotronHMoE / NemotronHExperts / NemotronHTopkRouter).
inline constexpr int kNumRouted = 128;
inline constexpr int kTopK = 6;
inline constexpr int kMoeIntermediate = 1856;
inline constexpr int kSharedIntermediate = 3712;
inline constexpr float kRoutedScale = 2.5f;

// Passport invariants (MODEL INGEST INVARIANT step 3).
static_assert(kMambaLayers + kMoeLayers + kAttnLayers == kLayers, "layer mix must sum to kLayers");
static_assert(kMambaHeads * kMambaHeadDim == kMambaDInner, "mamba d_inner geometry");
static_assert(kMambaDInner + kMambaConvDim + kMambaHeads == kMambaInProj, "in_proj geometry");
static_assert(kMambaDInner % kSsmGroups == 0, "gated-norm group geometry");
static_assert(kTopK <= kNumRouted, "top-k exceeds expert count");
static_assert(kHeadDim * kHeads == 4096 && kHeadDim * kKVHeads == 256, "attention projection geometry");

} // namespace frozen
} // namespace lsxcommon::nemotron_moe
