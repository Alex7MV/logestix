#pragma once

// Kimi-K3 loader: the pinned attention/dense graph, the mmap-streamed
// Stable-LatentMoE plane (latent down/up + two shared experts) and the neutral
// routed-expert store.
//
// Two residency classes on purpose:
//   * pinned   — attention (KDA + gated NoPE MLA), layer-0 MLP, embed/lm_head,
//                routers and every norm; ~23.7 GiB of NVFP4 on a 93-layer
//                model, which is what has to be device-resident every step.
//   * streamed — the per-layer latent projections and shared experts
//                (~103 MiB/layer, 9.5 GiB total). Pinning them needs ~34 GiB
//                and does not fit a 32 GiB card, so they stay as Arrow mmap
//                views and are copied into one device staging buffer per layer
//                per step. There is deliberately no "pin it all" knob: on this
//                card it cannot succeed, and a knob that only works on unseen
//                hardware is untested code.
//   * routed   — the neutral `ExpertStore<KimiK3ExpertEntry>` (mmap + LRU); the
//                payload is copied into a staging buffer per step.
//

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>
#include <memory>
#include <string>
#include <vector>

#include "lsxcommon/expert_store.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/schema.h"

namespace lsxcommon::kimi_k3 {

// ── runtime topology (cross-checked against the frozen passport) ────────────
struct KimiK3Topology {
    std::string format;
    std::string model;
    int num_layers = 0;
    int dense_layers = 1;
    int hidden = 0;
    int vocab = 0;
    int attn_res_block = 0;
    int dense_intermediate = 0;
    // gated MLA
    int mla_heads = 0;
    int kv_heads = 0;
    int q_lora = 0;
    int kv_lora = 0;
    int qk_nope = 0;
    int qk_rope = 0;
    int v_head = 0;
    bool mla_nope = true;
    bool mla_gate = true;
    // KDA
    int kda_heads = 0;
    int kda_head_dim = 0;
    int conv_kernel = 4;
    bool kda_full_rank_gate = true;
    double kda_gate_lower_bound = -5.0;
    // Stable LatentMoE
    int num_experts = 0;
    int top_k = 0;
    int moe_intermediate = 0;
    int latent = 0;
    int shared_intermediate = 0;
    int shared_experts = 0;
    bool latent_norm = true;
    bool moe_renormalize = true;
    double route_scale = 1.0;
    // numerics / ids
    double rms_eps = 1e-5;
    double situ_beta = 1.0;
    double situ_linear_beta = 0.0;
    int bos_id = -1;
    int eos_id = -1;
    int pad_id = -1;
    int tokenizer_kind = 0;
    int skipped_visual = 0;
    int skipped_projector = 0;
    std::vector<int> mla_layers;
    std::vector<int> kda_layers;
    std::string error_msg;

    bool IsMla(int layer) const
    {
        for (int l : mla_layers)
            if (l == layer) return true;
        return false;
    }
};

// ── streamed plane (host mmap views, alive for the session) ─────────────────
struct KimiK3HostNvfp4 {
    const uint8_t *packed = nullptr;
    const uint8_t *scales = nullptr;
    float global_scale = 1.0f;
    int rows = 0;
    int cols = 0;
    int64_t packed_bytes = 0;
    int64_t scale_bytes = 0;
};

struct KimiK3StreamedLayer {
    std::shared_ptr<void> keepalive;  // the mmap + batch views
    KimiK3HostNvfp4 latent_down;      // [latent, hidden]
    KimiK3HostNvfp4 latent_up;        // [hidden, latent]
    KimiK3HostNvfp4 shared_gate;      // [shared_inter, hidden]
    KimiK3HostNvfp4 shared_up;        // [shared_inter, hidden]
    KimiK3HostNvfp4 shared_down;      // [hidden, shared_inter]
    bool loaded = false;
};

// Device-side staging for one layer's streamed plane. The five weights are laid
// out back to back (packed then scales per weight) in one allocation.
struct KimiK3StreamedDevice {
    uint8_t *buffer = nullptr;
    int64_t bytes = 0;
    lsx_nvfp4::LsxNvfp4Weight latent_down;
    lsx_nvfp4::LsxNvfp4Weight latent_up;
    lsx_nvfp4::LsxNvfp4Weight shared_gate;
    lsx_nvfp4::LsxNvfp4Weight shared_up;
    lsx_nvfp4::LsxNvfp4Weight shared_down;
};

// ── per-layer pinned weights ────────────────────────────────────────────────
struct KimiK3LayerWeights {
    bool mla = false;
    __half *input_norm = nullptr;
    __half *post_norm = nullptr;
    __half *attn_res_proj = nullptr;   // [hidden] (the [1, hidden] row)
    __half *attn_res_norm = nullptr;
    __half *mlp_res_proj = nullptr;
    __half *mlp_res_norm = nullptr;

    // gated NoPE MLA
    lsx_nvfp4::LsxNvfp4Weight mla_qa;   // [q_lora, hidden]
    lsx_nvfp4::LsxNvfp4Weight mla_qb;   // [heads*(nope+rope), q_lora]
    lsx_nvfp4::LsxNvfp4Weight mla_kva;  // [kv_lora+rope, hidden]
    lsx_nvfp4::LsxNvfp4Weight mla_kvb;  // [heads*(nope+v), kv_lora]
    lsx_nvfp4::LsxNvfp4Weight mla_o;    // [hidden, heads*v]
    lsx_nvfp4::LsxNvfp4Weight mla_g;    // [heads*v, hidden]
    __half *mla_qanorm = nullptr;
    __half *mla_kvanorm = nullptr;

    // KDA
    lsx_nvfp4::LsxNvfp4Weight kda_q, kda_k, kda_v, kda_o, kda_g;
    __half *kda_fa = nullptr;   // [gate_rank, hidden]
    __half *kda_fb = nullptr;   // [heads*head_dim, gate_rank]
    __half *kda_b = nullptr;    // [heads, hidden]
    float *kda_conv_q = nullptr;
    float *kda_conv_k = nullptr;
    float *kda_conv_v = nullptr;
    float *kda_alog = nullptr;  // [head_dim]
    float *kda_dt = nullptr;    // [heads*head_dim]
    float *kda_onorm = nullptr; // [head_dim]

    // dense MLP (layer 0)
    lsx_nvfp4::LsxNvfp4Weight dense_gate, dense_up, dense_down;

    // router (pinned) + the latent norm (tiny even though the plane streams)
    __half *router_gate = nullptr;  // [experts, hidden]
    float *router_bias = nullptr;   // [experts]
    __half *latent_norm = nullptr;  // [latent]

    bool has_moe = false;
};

struct KimiK3PinnedWeights {
    KimiK3Topology top;
    lsx_nvfp4::LsxNvfp4Weight embed;  // [vocab, hidden]
    lsx_nvfp4::LsxNvfp4Weight head;   // [vocab, hidden]
    __half *final_norm = nullptr;
    __half *out_res_proj = nullptr;
    __half *out_res_norm = nullptr;
    std::vector<KimiK3LayerWeights> layers;
    std::vector<KimiK3StreamedLayer> streamed;  // [L]
    KimiK3StreamedDevice stream_dev;            // staging (unused when pinned)
    std::uint64_t resident_bytes = 0;
};

// ── routed-expert store ─────────────────────────────────────────────────────
// Field order matches gpu::expert_store::ExpertStore<T>::DecodeEntry().
struct KimiK3ExpertEntry {
    const uint8_t *host_w1 = nullptr;  // [moe_inter, latent]
    const uint8_t *host_w2 = nullptr;  // [latent, moe_inter]
    const uint8_t *host_w3 = nullptr;  // [moe_inter, latent]
    const uint8_t *host_w1_scales = nullptr;  // per-16 E4M3 block scales
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
};
static_assert(std::is_trivially_copyable_v<KimiK3ExpertEntry>,
              "ExpertStore entries must stay trivially copyable views");
static_assert(sizeof(KimiK3ExpertEntry) <= 128,
              "ExpertStore entry must stay a compact view");

using KimiK3ExpertDb = gpu::expert_store::ExpertStore<KimiK3ExpertEntry>;

// ── API ─────────────────────────────────────────────────────────────────────
arrow::Status LoadKimiK3Topology(const std::string &model_path,
                                 KimiK3Topology &out);

// Cross-checks the loaded index against the frozen passport.
bool KimiK3TopologyMatchesFrozen(const KimiK3Topology &top);

// Loads the pinned graph and the streamed-plane views. Returns false on any
// missing/short tensor (a partially loaded model is never left behind).
bool LoadKimiK3PinnedWeights(const std::string &root, KimiK3PinnedWeights &pw,
                             cudaStream_t stream);

void FreeKimiK3PinnedWeights(KimiK3PinnedWeights &pw);

// Copies one layer's streamed plane into the device staging buffer and points
// `pw.stream_dev` at it.
cudaError_t StageKimiK3Layer(KimiK3PinnedWeights &pw, int layer,
                             cudaStream_t stream);

KimiK3ExpertDb LoadKimiK3ExpertIndex(const std::string &model_path,
                                     int num_layers, int experts_per_layer,
                                     int first_moe_layer, size_t max_open);
void EnsureKimiK3ExpertResident(KimiK3ExpertDb &db, int layer, int expert,
                                cudaStream_t stream);
void FreeKimiK3ExpertIndex(KimiK3ExpertDb &db);

}  // namespace lsxcommon::kimi_k3
