#pragma once

#include <cstdint>

// Hy4-flash (tencent/Hy4-flash) text-backbone passport — MODEL INGEST step 3.
//
// Frozen geometry. Every number below is transcribed from the checkpoint's own
// config.json `text_config` and is cross-checked against it at conversion time
// and by tests/test_hy_moe_flash_constants.cpp; a runtime config read is not the
// contract, a mismatch at Load is a failure.
//
// The architecture is the DeepSeek-V4 family's, with four departures that each
// change what the model computes and therefore live here rather than in prose:
//
//   1. HYBRID attention at a 1:3 ratio. 12 layers run DeepSeek sparse attention
//      ("IndexPool": pooled-group selection of up to 2048 tokens) with 32 heads
//      and rope theta 1e7; 30 layers run sliding-window attention with 64 heads,
//      a 512-token causal window and rope theta 1e4. A uniform attention
//      assumption decodes garbage.
//   2. iHC ("identity Hyper-Connections") instead of DeepSeek's hyper-connections
//      WITH the comb matrix: 4 residual streams, a per-stream pre/post gain
//      computed from an [8, 4*hidden] matrix, magnitude 2.0, no Sinkhorn.
//   3. LatentMoE: the routed experts read and write a 1536-wide latent space
//      (`routed_expert_down_proj` / `routed_expert_up_proj`), so the expert
//      matrices are [2*1536, 1536] and [1536, 1536], not [2*I, hidden].
//   4. Sandwich norms on the SUBLAYER OUTPUT (`post_proj_norm`,
//      `post_fc2_norm`, `combine_output_norm`), and rms eps 1e-5 (DeepSeek-V4.1
//      uses 1e-20, the parent hy_v4 1e-6).
//
// Ballast is NOT part of this passport: Engram (4 layers of n-gram embedding
// tables, 126.7 GB BF16), the ViT tower and the 5-layer DFly2 speculative
// drafter are cut at conversion and counted in the index `deferred` block.

namespace lsxcommon {
namespace hy4 {

// The artifact format tag. One identity per architecture: a tree written by a
// different recipe must not load here (no backward compatibility, AGENTS.md).
inline constexpr char kFormatTag[] = "logestix-hy4-nvfp4";

// Registry id (the fold `gpu::detect_model_type` applies to the card name).
inline constexpr char kType[] = "hy4-flash";

// The CARD name, i.e. the vault directory name.
inline constexpr char kModelName[] = "Hy4-flash";

// ── Text-backbone geometry ─────────────────────────────────────────────────
inline constexpr int kHidden = 3072;
inline constexpr int kLayers = 42;
inline constexpr int kVocab = 120832;

// Feed-forward: layer 0 is dense at 13824, the other 41 are MoE.
inline constexpr int kDenseIntermediate = 13824;
inline constexpr int kMoeIntermediate = 1536;   // per-expert width
inline constexpr int kLatent = 1536;            // LatentMoE working width
inline constexpr int kSharedIntermediate = 1536;
inline constexpr int kNumRouted = 512;
inline constexpr int kTopK = 16;

// MLA. qk_head_dim = qk_nope + qk_rope = 256; `head_dim` in the config is the
// ROPE SLICE (64), which is what the cache stores alongside the 512-wide kv
// latent. Keys are never grouped (one KV latent per position serves all heads).
inline constexpr int kQLora = 1024;
inline constexpr int kKvLora = 512;
inline constexpr int kQkNope = 192;
inline constexpr int kQkRope = 64;
inline constexpr int kVHead = 256;
inline constexpr int kQkHead = kQkNope + kQkRope;   // 256
inline constexpr int kHeadsDsa = 32;
inline constexpr int kHeadsSwa = 64;

// Sliding-window layers: 511 preceding + the current token.
inline constexpr int kSwaWindow = 512;

// Sparse indexer (IndexPool) — DSA layers only.
inline constexpr int kIndexNHeads = 16;
inline constexpr int kIndexHeadDim = 128;
inline constexpr int kPoolRatio = 4;        // compressed_indexer_compress_ratio
inline constexpr int kGroupTopk = 512;      // groups kept, <= kIndexTopk tokens
inline constexpr int kIndexTopk = 2048;     // max selected tokens

// iHC.
inline constexpr int kHcMult = 4;                  // residual streams
inline constexpr int kHcRows = 2 * kHcMult;        // 8 = [pre(4) | post(4)]
inline constexpr int kHcDim = kHcMult * kHidden;   // 12288

// Numbers that change what the model computes.
inline constexpr float kRmsEps = 1e-5f;
inline constexpr float kHcEps = 1e-6f;
inline constexpr float kHcMagnitude = 2.0f;
inline constexpr float kRouteScale = 3.998f;
inline constexpr float kSwigluLimit = 10.0f;
inline constexpr float kRopeThetaDsa = 10000.0f * 1000.0f;  // 1e7
inline constexpr float kRopeThetaSwa = 10000.0f;
// Interleaved MRoPE sections [12, 10, 10] over the 32 rope pairs. Text-only
// input puts all three sections on the same position, which degenerates to
// plain rotate-half RoPE; the sections are recorded so the oracle and any
// future vision path cannot silently disagree.
inline constexpr int kMropeSectionT = 12;
inline constexpr int kMropeSectionH = 10;
inline constexpr int kMropeSectionW = 10;
inline constexpr int kRopePairs = kQkRope / 2;      // 32

// Special ids (tokenizer_config / text_config).
inline constexpr int kBosId = 120000;
inline constexpr int kEosId = 120025;
inline constexpr int kPadId = 120002;

// Context the engine plans for. The checkpoint declares 1048576; the artifact
// is sized for the canonical 2048-token protocol and grows on demand.
inline constexpr int64_t kMaxPosition = 1048576;

// ── Layer classification ───────────────────────────────────────────────────
//
// layer_types[l]: 0 and 1, then every layer with l % 4 == 1, are
// "deepseek_sparse_attention"; the rest are "sliding_attention". That is the
// README's "2 x Sparse -> 10 x (3 x SWA -> Sparse)" pattern.
constexpr bool IsDsa(int l) { return l >= 0 && l < kLayers && (l <= 1 || l % 4 == 1); }
constexpr bool IsSwa(int l) { return l >= 0 && l < kLayers && !IsDsa(l); }
constexpr int Heads(int l) { return IsDsa(l) ? kHeadsDsa : kHeadsSwa; }

// mlp_layer_types[l]: only layer 0 is dense.
constexpr bool IsMoeLayer(int l) { return l > 0 && l < kLayers; }

// text_config.engram_layer_ids = [1, 11, 21, 31] — the four layers whose
// engram tables are cut at conversion (counted, never loaded).
constexpr bool IsEngramCut(int l)
{
    return l == 1 || l == 11 || l == 21 || l == 31;
}
constexpr bool IsIndexerAt(int l) { return IsDsa(l); }

// KV footprint of one token across the whole backbone: every layer stores the
// 512-wide kv latent plus the 64-wide rope key, one copy (MLA), FP16.
constexpr int64_t KvBytesPerToken()
{
    return static_cast<int64_t>(kLayers) * (kKvLora + kQkRope) * 2;
}

// ── Compile-time relations ─────────────────────────────────────────────────
static_assert(kLayers == 42, "Hy4-flash backbone is 42 layers");
static_assert(kHidden == 3072, "Hy4-flash hidden is 3072");
static_assert(kVocab == 120832, "Hy4-flash vocab is 120832");
static_assert(kHeadsSwa == 2 * kHeadsDsa, "SWA layers carry twice the heads");
static_assert(kQkNope + kQkRope == kQkHead, "qk head = nope + rope");
static_assert(kQkHead == kVHead, "flash keeps qk and v head widths equal");
static_assert(kLatent == kMoeIntermediate, "LatentMoE width == expert width");
static_assert(kHcRows == 2 * kHcMult, "iHC emits pre and post rows");
static_assert(kHcDim == kHcMult * kHidden, "iHC flattened width");
static_assert(kRopePairs == kMropeSectionT + kMropeSectionH + kMropeSectionW,
              "MRoPE sections cover the rope pairs");
static_assert(kSwaWindow == 512, "SWA window is 512 (511 + current)");
static_assert(kTopK > 0 && kTopK <= kNumRouted, "bad top-k");
static_assert(kGroupTopk * kPoolRatio >= kIndexTopk,
              "group top-k must be able to supply the token budget");
static_assert(kHidden % 256 == 0, "hidden tiles cleanly on Blackwell");
static_assert(kMoeIntermediate % 128 == 0, "expert width tiles cleanly");

// The hybrid pattern, pinned layer by layer: 12 DSA + 30 SWA.
static_assert(IsDsa(0) && IsDsa(1) && IsDsa(5) && IsDsa(9) && IsDsa(13) &&
                  IsDsa(17) && IsDsa(21) && IsDsa(25) && IsDsa(29) &&
                  IsDsa(33) && IsDsa(37) && IsDsa(41),
              "DSA layer set");
static_assert(!IsDsa(2) && !IsDsa(4) && !IsDsa(6) && !IsDsa(40),
              "SWA layer set");
static_assert(IsSwa(2) && IsSwa(40), "SWA complement");
static_assert(kHeadsDsa == 32 && kHeadsSwa == 64, "per-mode head counts");

// One dense layer, 41 MoE layers.
static_assert(!IsMoeLayer(0) && IsMoeLayer(1) && IsMoeLayer(41),
              "dense layer 0, MoE elsewhere");

// Engram cuts.
static_assert(IsEngramCut(1) && IsEngramCut(11) && IsEngramCut(21) &&
                  IsEngramCut(31),
              "engram layer set");
static_assert(!IsEngramCut(0) && !IsEngramCut(12) && !IsEngramCut(32),
              "engram complement");

constexpr int DsaLayerCount()
{
    int n = 0;
    for (int l = 0; l < kLayers; ++l)
        if (IsDsa(l)) ++n;
    return n;
}
constexpr int MoeLayerCount()
{
    int n = 0;
    for (int l = 0; l < kLayers; ++l)
        if (IsMoeLayer(l)) ++n;
    return n;
}
static_assert(DsaLayerCount() == 12, "12 sparse layers");
static_assert(MoeLayerCount() == 41, "41 MoE layers");

} // namespace hy4
} // namespace lsxcommon
