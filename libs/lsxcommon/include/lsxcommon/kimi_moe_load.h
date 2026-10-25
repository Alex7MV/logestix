#pragma once

#include "lsxcommon/schema.h"
#include "lsxcommon/expert_store.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lsxcommon {
namespace kimi {

// Kimi-K2.7-Code geometry (config.json text_config, verified against the
// vault checkpoint). Compile-time locked per MODEL INGEST INVARIANT; config
// values are cross-checked at Load() and any mismatch is a hard error.
constexpr int kKimiHidden = 7168;
constexpr int kKimiNumLayers = 61;

// Frozen-geometry passport (MODEL INGEST INVARIANT step 3).
static_assert(kKimiHidden == 7168 && kKimiNumLayers == 61,
              "Kimi-K2.7 frozen passport drifted");
constexpr int kKimiNumHeads = 64;
constexpr int kKimiKvLoraRank = 512;   // compressed KV latent
constexpr int kKimiQLoraRank = 1536;   // q_lora_rank
constexpr int kKimiQkNope = 128;       // per-head non-rope q/k dim
constexpr int kKimiQkRope = 64;        // shared rope dim
constexpr int kKimiHeadV = 128;        // v_head_dim
constexpr int kKimiMoeIntermediate = 2048;
constexpr int kKimiDenseIntermediate = 18432;
constexpr int kKimiNumRouted = 384;
constexpr int kKimiNumShared = 1;
constexpr int kKimiTopK = 8;
constexpr int kKimiVocab = 163840;
constexpr int kKimiFirstKDenseReplace = 1;
constexpr float kKimiRoutedScalingFactor = 2.827f;
constexpr float kKimiRmsEps = 1e-5f;
constexpr float kKimiRopeTheta = 50000.0f;
constexpr float kKimiYarnFactor = 64.0f;
constexpr int kKimiYarnOriginalMaxPos = 4096;

// YaRN attention scale: the reference multiplies q_head_dim**-0.5 by
// yarn_get_mscale(factor, mscale_all_dim)**2 when mscale_all_dim is truthy.
// For this config (factor 64, mscale = mscale_all_dim = 1.0):
//   mscale = 0.1*ln(64) + 1 = 1.4158883083359672
// The cos/sin path is unscaled because yarn_get_mscale(64, 1)/itself == 1.
constexpr float kKimiYarnMscale = 1.4158883083359672f;

// One adjacent-pair RoPE rotation: element pair (2p, 2p+1) by theta.
// Single source of truth for the kernel and the parity test; the golden is
// captured from the checkpoint's own modeling_deepseek.py.
//
// Two overloads (float for the host parity test, bf16 for the device path)
// with identical pairing; the bf16 path is additionally pinned end-to-end by
// test_kimi_moe_attention.
__device__ __host__ inline void KimiRopePair(int p, float theta, float *r)
{
    // The reference de-interleaves with view(d/2, 2).transpose().reshape()
    // before rotate_half, i.e. element pairs (2p, 2p+1) share inv_freq[p].
    const int a = 2 * p;
    const int b = 2 * p + 1;
    const float cs = cosf(theta), sn = sinf(theta);
    const float x = r[a], y = r[b];
    r[a] = x * cs - y * sn;
    r[b] = x * sn + y * cs;
}

__device__ __host__ inline void KimiRopePair(int p, float theta,
                                             __nv_bfloat16 *r)
{
    const int a = 2 * p;
    const int b = 2 * p + 1;
    const float cs = cosf(theta), sn = sinf(theta);
    const float x = __bfloat162float(r[a]), y = __bfloat162float(r[b]);
    r[a] = __float2bfloat16_rn(x * cs - y * sn);
    r[b] = __float2bfloat16_rn(x * sn + y * cs);
}

// YaRN interpolation over the pair index (HF DeepseekV3YarnRotaryEmbedding).
std::vector<float> KimiYarnInvFreq();

// Attention temperature used by the MLA kernels. Header-inline because the
// decode kernel calls it from device code and the parity test calls it from
// the host. (Task 3 adds the YaRN mscale**2 factor; this commit keeps the
// current value so the parity test fails on the number, not on the build.)
__device__ __host__ inline float KimiAttnScale()
{
    // HF DeepseekV3Attention: softmax_scale = q_head_dim**-0.5
    //   * yarn_get_mscale(factor, mscale_all_dim)**2   (mscale_all_dim truthy)
    return (1.0f / sqrtf(static_cast<float>(kKimiQkNope + kKimiQkRope))) *
           (kKimiYarnMscale * kKimiYarnMscale);
}

// Derived MLA projection row counts (rows of the checkpoint weight matrices).
constexpr int kKimiQaRows = kKimiQLoraRank;                               // 1536
constexpr int kKimiQbRows = kKimiNumHeads * (kKimiQkNope + kKimiQkRope);  // 12288
constexpr int kKimiKvaRows = kKimiKvLoraRank + kKimiQkRope;               // 576
constexpr int kKimiKvbRows = kKimiNumHeads * (kKimiQkNope + kKimiHeadV);  // 16384
constexpr int kKimiORows = kKimiHidden;                                   // 7168

// Per-layer pinned weights for one Kimi layer, uploaded as FP16 (host-side
// BF16→FP16 via the pinned staging ring). Layer 0 fills the mlp_* fields,
// MoE layers (1..60) fill the router fields. Pointers stay null when the
// layer file lacks the tensor.
struct KimiLayerWeights {
    // Layer norms
    __nv_bfloat16 *d_input_norm = nullptr;
    __nv_bfloat16 *d_post_attn_norm = nullptr;

    // MLA attention projections: NVFP4 packed E2M1 + per-16 E4M3 scales +
    // global scale (round-2 r2 format). Norms stay BF16 below.
    uint8_t *d_q_a_packed = nullptr;
    uint8_t *d_q_a_scale = nullptr;
    float q_a_gs = 1.0f;
    int64_t q_a_packed_bytes = 0, q_a_scale_bytes = 0;
    int64_t q_a_rows = 0, q_a_cols = 0;

    uint8_t *d_q_b_packed = nullptr;
    uint8_t *d_q_b_scale = nullptr;
    float q_b_gs = 1.0f;
    int64_t q_b_packed_bytes = 0, q_b_scale_bytes = 0;
    int64_t q_b_rows = 0, q_b_cols = 0;

    uint8_t *d_kv_a_packed = nullptr;
    uint8_t *d_kv_a_scale = nullptr;
    float kv_a_gs = 1.0f;
    int64_t kv_a_packed_bytes = 0, kv_a_scale_bytes = 0;
    int64_t kv_a_rows = 0, kv_a_cols = 0;

    uint8_t *d_kv_b_packed = nullptr;
    uint8_t *d_kv_b_scale = nullptr;
    float kv_b_gs = 1.0f;
    int64_t kv_b_packed_bytes = 0, kv_b_scale_bytes = 0;
    int64_t kv_b_rows = 0, kv_b_cols = 0;

    uint8_t *d_o_packed = nullptr;
    uint8_t *d_o_scale = nullptr;
    float o_gs = 1.0f;
    int64_t o_packed_bytes = 0, o_scale_bytes = 0;
    int64_t o_rows = 0, o_cols = 0;

    __nv_bfloat16 *d_q_a_norm = nullptr;
    __nv_bfloat16 *d_kv_a_norm = nullptr;

    // Router (MoE layers 1..60): mlp.gate.weight + e_score_correction_bias
    __nv_bfloat16 *d_router_gate = nullptr;
    float *h_gate_bias = nullptr;
    int gate_bias_n = 0;

    // kv_b dequantized to BF16 for the absorbed-attention kernel, decoded
    // lazily on first use (per layer — a single static memo slot used to
    // thrash across layers, re-dequantizing 60x per token). Weights are
    // read-only for the process lifetime.
    __nv_bfloat16 *d_kv_b_bf16 = nullptr;

    // Dense MLP (layer 0 only)
    __nv_bfloat16 *d_mlp_gate = nullptr;
    __nv_bfloat16 *d_mlp_down = nullptr;
    __nv_bfloat16 *d_mlp_up = nullptr;

    // Non-null when the layer was uploaded arena-backed: every d_* pointer
    // above is a view into this single allocation and Free() releases the
    // arena instead of the individual pointers.
    void *d_arena = nullptr;
    size_t arena_bytes = 0;

    void Free();
};

struct KimiPinnedWeights {
    __nv_bfloat16 *d_embed = nullptr;
    __nv_bfloat16 *d_head = nullptr;
    __nv_bfloat16 *d_final_norm = nullptr;
    int vocab_size = 0;
    int hidden = 0;
    std::vector<KimiLayerWeights> layers;   // indexed by layer 0..60

    // Shared experts: one NVFP4-packed shared expert per MoE layer from
    // pinned/shared_experts.arrow. Zero-copy mmap views, host-side only.
    std::vector<PackedExpertView> shared_host;
    std::shared_ptr<void> shared_keepalive;

    // Peak device-resident weight set uploaded by LoadKimiPinnedWeights.
    std::uint64_t resident_bytes = 0;

    void Free();
};

// Reads <model_path>/logestix/pinned/{global,layer_<n>}.arrow and uploads
// every pinned tensor to device memory as FP16. Missing layer files are a
// hard error (uniform Kimi grid: layer 0 dense, layers 1..60 MoE).
cudaError_t LoadKimiPinnedWeights(const std::string &model_path,
                                  KimiPinnedWeights &pw,
                                  cudaStream_t stream);

// ── On-demand routed-expert index ──────────────────────────────────────────
// Memory-mapped view of one routed expert: NVFP4 packed E2M1 nibble payloads
// + per-block E4M3 scales + decoded global scales, zero-copy into the Arrow
// IPC buffer. Dequant invariant matches GlmExpertEntry:
//   w[i] = E2M1(nibble_i) * 2^(((scales[i>>4]>>3)&0xF) - 7) * global_scale
struct KimiExpertEntry {
    const uint8_t *host_w1 = nullptr;
    const uint8_t *host_w2 = nullptr;
    const uint8_t *host_w3 = nullptr;
    const uint8_t *host_w1_scales = nullptr;
    const uint8_t *host_w2_scales = nullptr;
    const uint8_t *host_w3_scales = nullptr;
    float w1_global_scale = 1.0f;
    float w2_global_scale = 1.0f;
    float w3_global_scale = 1.0f;
    int64_t w1_bytes = 0;
    int64_t w2_bytes = 0;
    int64_t w3_bytes = 0;
    int64_t w1_scales_bytes = 0;
    int64_t w2_scales_bytes = 0;
    int64_t w3_scales_bytes = 0;
    // Pinned-store view (DeepSeek round-8 parity): when the whole store is
    // pinned into locked host memory, each expert's blobs live contiguously
    // in one block laid out as [w1|w3][s1|s3][w2][s2] (w3 at AlignUp256(w1),
    // s3 at AlignUp256(s1)) so the slot upload can stream the gate/up pair
    // first, fence it, and overlap the down projection with the w13 GEMM.
    // host_* pointers are re-pointed into the chunks and pinned_blob marks
    // the expert block start.
    const uint8_t *pinned_blob = nullptr;
    int64_t blob_a_bytes = 0;   // [w1|w3] packed
    int64_t blob_b_bytes = 0;   // [s1|s3] scales
    int64_t blob_c_bytes = 0;   // [w2] packed
    int64_t blob_d_bytes = 0;   // [s2] scales
};

using KimiExpertDb = lsxcommon::gpu::expert_store::ExpertStore<KimiExpertEntry>;

// Builds the flat file-path index of every routed expert
// (<model_path>/logestix/experts/layer_<l>/exp_<e>.arrow with fallback to
// <model_path>/experts/...) and warms the LRU up to max_open entries.
KimiExpertDb load_kimi_expert_index(const std::string &model_path,
                                    int num_layers, int experts_per_layer);

// Ensures the mmap for (layer, expert) is LRU-resident, synchronizing the
// stream before any eviction (in-flight DMA reads must finish first).
void ensure_kimi_expert_resident(KimiExpertDb &db, int layer, int expert,
                                 cudaStream_t stream);

// Releases all expert mmaps.
void free_kimi_expert_index(KimiExpertDb &db);

// ── MLA attention state ────────────────────────────────────────────────────
// Latent KV cache: compressed KV latents (512 FP16) + shared yarn-roped keys
// (64 FP16) per slot. One allocation, 16-byte aligned arrays.
constexpr int kKimiMaxSlots = 1024;

struct alignas(16) KimiMlaState {
    __nv_bfloat16 kv[kKimiMaxSlots][kKimiKvLoraRank];    // cached latents
    __nv_bfloat16 k_rope[kKimiMaxSlots][kKimiQkRope];    // shared roped keys
    int n_used = 0;        // occupied slot count
    int turn_id = -1;      // last appended turn
};

static_assert(offsetof(KimiMlaState, kv) % 16 == 0 &&
                  offsetof(KimiMlaState, k_rope) % 16 == 0,
              "MLA cache arrays must stay 16-byte aligned");

// Async zero-reset of the state (n_used=0 etc.) before first use.
cudaError_t KimiMlaReset(KimiMlaState *d_state, cudaStream_t stream);

// One-layer MLA decode/prefill attention (DeepSeek-V3 math, absorbed form):
//   q = q_b(RMSNorm(q_a(x)))            split per head: 128 nope + 64 rope
//   c_kv = RMSNorm(kv_a[0:512](x))      latent, appended to cache
//   k_rope = yarn_rope(kv_a[512:576](x))
//   score[h,s] = dot(W_k(h)·q_nope_h, c_s)/sqrt(192)*mscale
//              + dot(q_rope_h, krope_s)/sqrt(192)*mscale
//   out_h = W_v(h)·Σ_s p[h,s]·c_s       (absorbed value path)
//   d_out = d_residual + o_proj(concat heads)
// Whole pass async on the caller's stream (no host sync inside).
// base_slot: first cache slot owned by this pass (chronological order);
// pos_offset: absolute RoPE position of the first row of this pass.
cudaError_t RunKimiMlaAttention(cudaStream_t stream, int turn,
                                const KimiLayerWeights &lw, int seq,
                                const __nv_bfloat16 *d_hidden,
                                const __nv_bfloat16 *d_residual, __nv_bfloat16 *d_out,
                                KimiMlaState *d_state,
                                int base_slot,
                                int pos_offset);

} // namespace kimi
} // namespace lsxcommon
