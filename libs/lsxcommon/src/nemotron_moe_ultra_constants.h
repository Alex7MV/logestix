#pragma once

// Frozen NVIDIA-Nemotron-3-Ultra-550B-A55B geometry (compile-time passport).
// Source of truth: /mnt/fast_storage/huggingface/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16
// config.json (model_type nemotron_h) + safetensors inventory audited 2026-11-10:
// 108 layers (48 mamba + 48 moe + 12 full attention at 7/14/23/32/39/48/57/64/
// 73/82/89/98), hidden 8192, mamba2 d_inner 16384 (256x64), n_groups 8, state
// 128, conv 4, in_proj 35072, MoE 512 experts top-22 routed at latent 2048
// (n_group 8, topk_group 1) + shared expert I=10240, vocab 131072, eps 1e-5.
//
// This model is a SEPARATE LINEAGE from nemotron-3.5-lightning-30b-a3b: no
// width, layer count or expert geometry is shared, so the two never read each
// other's converted tree (the loader validates a tree against THIS passport and
// fails by field name on a mismatch).
//
// The full-attention layers are **NoPE**: the reference `NemotronHAttention`
// (transformers 5.15.1) has no rotary module, never builds cos/sin, and ignores
// the `position_ids` it is handed. Position in this hybrid comes from the Mamba-2
// layers. `rope_theta` in the checkpoint config is vestigial - it is NOT applied
// here, and applying it is the exact defect the 30B sibling shipped and fixed on
// 2026-11-01 (per-layer relL2 0.09 -> 1.20 by position 284, tool prompts
// degenerating into tag soup).

#include <cstdint>

namespace lsxcommon::nemotron_ultra {
namespace frozen {

inline constexpr const char *kType = "nemotron-ultra-550b-a55b";

// ── Backbone ───────────────────────────────────────────────────────────────
inline constexpr int kHidden = 8192;
inline constexpr int kLayers = 108;
inline constexpr int kVocab = 131072;
inline constexpr int kMambaLayers = 48;
inline constexpr int kMoeLayers = 48;
inline constexpr int kAttnLayers = 12;
inline constexpr int kAttnLayerIdx[kAttnLayers] = {7,  14, 23, 32, 39, 48,
                                                  57, 64, 73, 82, 89, 98};
inline constexpr double kRmsEps = 1e-5;

// ── Mamba-2 (NemotronHMamba2Mixer) ─────────────────────────────────────────
inline constexpr int kMambaHeads = 256;
inline constexpr int kMambaHeadDim = 64;
inline constexpr int kMambaDInner = kMambaHeads * kMambaHeadDim;  // 16384
inline constexpr int kSsmStateSize = 128;
inline constexpr int kSsmGroups = 8;
inline constexpr int kMambaConvKernel = 4;
inline constexpr int kMambaConvDim =
    kMambaDInner + 2 * kSsmGroups * kSsmStateSize;                 // 18432
inline constexpr int kMambaInProj =
    kMambaDInner + kMambaConvDim + kMambaHeads;                    // 35072
inline constexpr int kMambaGroupSize = kMambaDInner / kSsmGroups;  // 2048
// `time_step_min`; lower clamp on dt. Upper bound is +inf, so no dt max.
inline constexpr float kDtMin = 0.001f;

// ── Full attention (NemotronHAttention) — NoPE, see the header note ────────
inline constexpr int kHeads = 64;
inline constexpr int kKVHeads = 2;
inline constexpr int kHeadDim = 128;
inline constexpr float kAttnScale = 0.08838834764831845f;  // 1/sqrt(128)

// ── MoE (NemotronHMoE / NemotronHExperts / NemotronHTopkRouter) ────────────
inline constexpr int kNumRouted = 512;
inline constexpr int kTopK = 22;
inline constexpr int kMoeIntermediate = 5120;  // per-expert I, at the latent width
inline constexpr int kMoeLatent = 2048;        // moe_latent_size (fc1/fc2 bridge)
inline constexpr int kSharedIntermediate = 10240;
// Routing groups. NOTE the two "n_group" spellings in this checkpoint's config:
// `n_group` (= 1 here) groups the EXPERTS for the router, while `n_groups`
// (= 8, below) groups the Mamba-2 SSM state. With n_group == 1 the reference's
// group stage is the identity over all 512 experts - the same shape the 30B
// sibling has - and `kNumRouted / kRouterGroups == kNumRouted`.
inline constexpr int kRouterGroups = 1;     // n_group (routing, NOT n_groups)
inline constexpr int kRouterTopkGroup = 1;  // topk_group
inline constexpr float kRoutedScale = 5.0f;

// ── Passport invariants (MODEL INGEST INVARIANT step 3) ────────────────────
static_assert(kMambaLayers + kMoeLayers + kAttnLayers == kLayers,
              "layer mix must sum to kLayers");
static_assert(kMambaHeads * kMambaHeadDim == kMambaDInner, "mamba d_inner geometry");
static_assert(kMambaDInner + kMambaConvDim + kMambaHeads == kMambaInProj,
              "in_proj geometry");
static_assert(kMambaDInner % kSsmGroups == 0, "gated-norm group geometry");
static_assert(kNumRouted % kRouterGroups == 0,
              "expert groups must divide the expert count");
static_assert(kRouterTopkGroup <= kRouterGroups, "topk_group exceeds n_group");
static_assert(kTopK <= (kNumRouted / kRouterGroups) * kRouterTopkGroup,
              "top-k must be reachable from the surviving groups");
static_assert(kHeadDim * kHeads == 8192 && kHeadDim * kKVHeads == 256,
              "attention projection geometry");
static_assert(kMoeLatent > 0 && kMoeLatent < kHidden, "latent MoE must compress");
static_assert(kSharedIntermediate > 0 && kMoeIntermediate > 0, "bad expert widths");

} // namespace frozen
} // namespace lsxcommon::nemotron_ultra
