#pragma once

// Inkling (base) loader (Model Ingestion Protocol step 3). The converted tree
// is a uniform NVFP4 artifact (the source release is pure BF16):
//   * `pinned/global.arrow`         embed/unembed + embed/final norms (raw BF16,
//                                   uploaded as FP16 tables / FP16 norms)
//   * `pinned/layer_<l>.arrow`      attn/mlp norms, per-head q/k norm weights,
//                                   rel bank, 4 sconv tap tables, router
//                                   weight/bias/global_scale, dense mlp scale
//   * `pinned/layer_<l>_nvfp4.arrow` q/k/v/o/r projections; dense gate/up/down
//                                   (layers < dense_mlp_layers); the 2 shared
//                                   experts' gate/up/down (MoE layers)
//   * `experts/layer_<l>/exp_<e>.arrow`  routed experts (neutral ExpertStore,
//                                   mmap views, layers >= dense_mlp_layers)
//
// The dense plane is device-resident (~9 GiB NVFP4 + 4.6 GiB FP16 tables on the
// real geometry); the 928B-parameter routed set is only ever an mmap view.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include "lsxcommon/expert_store.h"
#include "lsxcommon/lsx_inference_nvfp4.h"

namespace lsxcommon::inkling::base {

// ── runtime topology (cross-checked against the frozen passport) ───────────
struct Topology {
    // Schema layout generation declared by model.index.json
    // (lsxcommon::kIndexLayout). 0 = absent.
    int64_t layout = 0;
    std::string source_model_type;
    int num_layers = 0;
    int hidden = 0;
    int vocab = 0;
    int unpadded_vocab = 0;
    int num_heads = 0;
    int num_kv_heads = 0;      // full layers
    int swa_kv_heads = 0;      // sliding layers
    int head_dim = 0;
    int d_rel = 0;
    int rel_extent = 0;
    int sliding_window = 0;
    int conv_kernel = 0;
    int dense_mlp_layers = 0;
    int dense_intermediate = 0;
    int num_experts = 0;
    int top_k = 0;
    int moe_intermediate = 0;
    int shared_experts = 0;
    double route_scale = 1.0;
    double attn_scale_denom = 128.0;
    double rms_eps = 1e-6;
    int log_scaling_n_floor = 0;
    double log_scaling_alpha = 0.0;
    double logits_mup_div = 1.0;
    int bos_id = -1;
    int eos_id = -1;
    int stop_id_alt = -1;
    int pad_id = -1;
    int tokenizer_kind = 0;
    int skipped_visual = 0;
    int skipped_audio = 0;
    int skipped_mtp = 0;
    std::vector<int> full_layers;
    std::vector<int> sliding_layers;
    std::string error_msg;

    bool IsFull(int layer) const
    {
        for (int l : full_layers)
            if (l == layer) return true;
        return false;
    }
    bool IsDenseMlp(int layer) const
    {
        return layer >= 0 && layer < dense_mlp_layers;
    }
    int KvHeadsOf(int layer) const
    {
        return IsFull(layer) ? num_kv_heads : swa_kv_heads;
    }
    int KvDimOf(int layer) const { return KvHeadsOf(layer) * head_dim; }
    int RelExtentOf(int layer) const
    {
        return IsFull(layer) ? rel_extent : sliding_window;
    }
};

// ── routed-expert store entry (field order matches ExpertStore::DecodeEntry) ─
struct ExpertEntry {
    const uint8_t *host_w1 = nullptr;  // gate [I, H] packed
    const uint8_t *host_w2 = nullptr;  // down [H, I] packed
    const uint8_t *host_w3 = nullptr;  // up   [I, H] packed
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
};
static_assert(std::is_trivially_copyable_v<ExpertEntry>,
              "ExpertStore entries must stay trivially copyable views");
static_assert(sizeof(ExpertEntry) <= 128,
              "ExpertStore entry must stay a compact view");

using ExpertDb = gpu::expert_store::ExpertStore<ExpertEntry>;

// ── pinned weights (device-resident for the whole session) ──────────────────
// One dense NVFP4 projection: [rows, cols] row-major through the neutral layer.
using DenseNvfp4 = lsx_nvfp4::LsxNvfp4Weight;

struct SharedExpertWeights {
    DenseNvfp4 gate;  // [I, H]
    DenseNvfp4 up;    // [I, H]
    DenseNvfp4 down;  // [H, I]
};

struct LayerWeights {
    bool full = false;
    DenseNvfp4 wq, wk, wv, wo, wr;
    // Dense MLP prefix (layers < dense_mlp_layers).
    DenseNvfp4 dense_gate, dense_up, dense_down;
    float *mlp_gs = nullptr;  // [1] dense output global scale
    // Shared-sink experts (MoE layers); sized by top.shared_experts.
    std::vector<SharedExpertWeights> shared;
    // Norms (FP16 weights, never quantized).
    __half *attn_norm = nullptr;  // [hidden]
    __half *mlp_norm = nullptr;   // [hidden]
    __half *q_norm = nullptr;     // [head_dim] per-head RMSNorm weight
    __half *k_norm = nullptr;     // [head_dim]
    // FP32 side tensors (the reference computes these at full precision).
    float *rel_proj = nullptr;    // [d_rel, extent]
    float *kconv_w = nullptr;     // [kv_dim, k] conv1d taps
    float *vconv_w = nullptr;
    float *attn_conv_w = nullptr;  // [hidden, k]
    float *mlp_conv_w = nullptr;
    float *router_w = nullptr;    // [E+S, hidden] (MoE layers)
    float *router_bias = nullptr;  // [E] selection-only bias
    float *router_gs = nullptr;   // [1] gate global scale
    // Recurrent conv state, FOUR convolutions per layer (k/v + post-attn +
    // post-MLP), laid out back to back: [k][kv_dim] x2 then [k][hidden] x2.
    // PER-SEQUENCE state: every request starts with it zeroed.
    float *conv_state = nullptr;
    int64_t conv_state_bytes = 0;
};

struct PinnedWeights {
    Topology top;
    // embed/unembed are FP16 tables, NOT NVFP4: they carry the residual stream
    // at full width and 4-bit rounding there displaces every token vector.
    __half *embed = nullptr;  // [vocab, hidden]
    __half *head = nullptr;   // [vocab, hidden] (unembed)
    int embed_rows = 0;
    int embed_cols = 0;
    __half *embed_norm = nullptr;
    __half *final_norm = nullptr;
    std::vector<LayerWeights> layers;
    std::uint64_t resident_bytes = 0;
};

// ── API ─────────────────────────────────────────────────────────────────────
arrow::Status LoadTopology(const std::string &model_path, Topology &out);

// Cross-checks the loaded index against the frozen passport (relaxed under
// LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY).
bool TopologyMatchesFrozen(const Topology &top);

// Uploads the pinned plane. Returns false on any missing/short tensor (a
// partially loaded model is never left behind).
bool LoadPinnedWeights(const std::string &root, PinnedWeights &pw,
                       cudaStream_t stream);

// Releases EVERY allocation LoadPinnedWeights added.
void FreePinnedWeights(PinnedWeights &pw);

// The routed-expert window floor: one token touches top_k experts in every
// MoE layer, walked in ascending order once per token.
size_t ExpertWindowFloor(const Topology &top);
size_t ResolveExpertWindow(const Topology &top, long requested);

ExpertDb LoadExpertIndex(const std::string &model_path, int num_layers,
                         int first_moe_layer, int experts_per_layer,
                         size_t max_open);
void FreeExpertIndex(ExpertDb &db);

}  // namespace lsxcommon::inkling::base