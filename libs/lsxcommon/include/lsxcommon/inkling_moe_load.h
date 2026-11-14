#pragma once

// Inkling-2-Small loader. The converted tree is a uniform NVFP4 artifact:
//   * `pinned/global.arrow`        embed/final norms, AttnRes queries (F32)
//   * `pinned/global_nvfp4.arrow`  embed + unembed
//   * `pinned/layer_<l>.arrow`     norms, L2-clip scales, rel bank, sinks,
//                                  conv taps, router weight/bias
//   * `pinned/layer_<l>_nvfp4.arrow` q/k/v/o/r/gate projections (+ the
//                                  full-layer dynamic-conv projections)
//   * `experts/layer_<l>/exp_<e>.arrow`  routed experts (neutral ExpertStore)
//
// The dense plane (~4.7 GiB of NVFP4 plus ~170 MiB of FP16 router weights) is
// device-resident; the 594B-parameter routed set is only ever an mmap view.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include "lsxcommon/expert_store.h"
#include "lsxcommon/lsx_inference_nvfp4.h"

namespace lsxcommon::inkling {

// ── runtime topology (cross-checked against the frozen passport) ───────────
struct InklingTopology {
    std::string format;
    std::string source_model_type;
    int num_layers = 0;
    int hidden = 0;
    int vocab = 0;
    int unpadded_vocab = 0;
    int num_heads = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    int d_rel = 0;
    int rel_extent = 0;
    int sliding_window = 0;
    int conv_kernel = 0;
    int attn_res_block = 0;
    int num_experts = 0;
    int top_k = 0;
    int moe_intermediate = 0;
    int shared_experts = 0;
    bool moe_renormalize = true;
    double route_scale = 1.0;
    double attn_scale_denom = 128.0;
    double rms_eps = 1e-6;
    double qk_l2_clip = 0.0;
    double rel_l2_clip = 0.0;
    int log_scaling_n_floor = 0;
    double log_scaling_alpha = 0.0;
    double mlp_gate_softcap = 0.0;
    double mlp_up_softcap = 0.0;
    double final_logit_softcap = 0.0;
    int bos_id = -1;
    int eos_id = -1;
    int stop_id_alt = -1;
    int pad_id = -1;
    int tokenizer_kind = 0;
    int skipped_visual = 0;
    std::vector<int> full_layers;
    std::vector<int> sliding_layers;
    std::vector<bool> expert_nvfp4;
    // TRUE where the routed experts carry the release's MXFP8 plane: E4M3 bytes
    // plus a ue8m0 exponent per 16 (the source's per-32 byte, duplicated).
    std::vector<bool> expert_fp8;
    std::string error_msg;

    bool IsFull(int layer) const
    {
        for (int l : full_layers)
            if (l == layer) return true;
        return false;
    }
    bool ExpertIsFp8(int layer) const
    {
        return layer >= 0 && layer < static_cast<int>(expert_fp8.size()) &&
               expert_fp8[static_cast<size_t>(layer)];
    }
    bool IsBlockBoundary(int layer) const
    {
        return attn_res_block > 0 && layer % attn_res_block == 0;
    }
    int FullLayerIndex(int layer) const
    {
        for (size_t i = 0; i < full_layers.size(); ++i)
            if (full_layers[i] == layer) return static_cast<int>(i);
        return -1;
    }
};

// ── routed-expert store entry (field order matches ExpertStore::DecodeEntry) ─
struct InklingExpertEntry {
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
static_assert(std::is_trivially_copyable_v<InklingExpertEntry>,
              "ExpertStore entries must stay trivially copyable views");
static_assert(sizeof(InklingExpertEntry) <= 128,
              "ExpertStore entry must stay a compact view");

using InklingExpertDb = gpu::expert_store::ExpertStore<InklingExpertEntry>;

// ── pinned weights (device-resident for the whole session) ──────────────────
// The dense plane is ~4.7 GiB of NVFP4 plus the FP32 routers (~340 MiB) and the
// FP32 side tensors; the 594B-parameter routed set is only ever an mmap view.
// One FP16 dense projection: [rows, cols], row-major.
struct InklingDenseF16 {
    __half *d = nullptr;
    int rows = 0;
    int cols = 0;
};

struct InklingLayerWeights {
    bool full = false;
    // BF16 in the checkpoint, FP16 on device (lossless for these magnitudes).
    InklingDenseF16 wq, wk, wv, wo, wr, attn_gate;
    InklingDenseF16 ksconv_proj, vsconv_proj;  // full layers only
    // Norms (FP16 weights, never quantized).
    __half *attn_norm = nullptr;    // sliding layers
    __half *q_in_norm = nullptr;    // full layers
    __half *k_in_norm = nullptr;
    __half *v_in_norm = nullptr;
    __half *mlp_norm = nullptr;
    // FP32 side tensors: the checkpoint itself keeps these at full precision.
    float *sink_w = nullptr;        // [heads, hidden] (BF16 -> FP16 on upload)
    __half *sink_proj = nullptr;    // [heads, hidden] FP16 (matvec)
    float *q_clip = nullptr;        // [heads, head_dim] learned scale
    float *rel_clip = nullptr;      // [heads, d_rel]   learned scale
    float *rel_proj = nullptr;      // [d_rel, rel_extent]
    float *kconv_w = nullptr;       // [kv_dim, k] conv1d taps (sliding)
    float *vconv_w = nullptr;
    float *kconv_static = nullptr;  // [kv_dim, k] lag-0-first taps (full)
    float *vconv_static = nullptr;
    __half *kconv_proj_w = nullptr;  // dynamic tap projection [kv_heads*k, hidden]
    __half *vconv_proj_w = nullptr;
    float *attn_conv_w = nullptr;   // [hidden, k] conv1d taps
    float *mlp_conv_w = nullptr;
    float *router_w = nullptr;      // [experts, hidden] (FP32 in the checkpoint)
    float *router_bias = nullptr;   // [experts] FP32 selection bias
    // Recurrent conv state, FOUR convolutions per layer (k/v + post-attn +
    // post-MLP), laid out back to back: [k][kv_dim] x2 then [k][hidden] x2.
    // It is PER-SEQUENCE state: every request starts with it zeroed, or a second
    // request inherits the first one's history (measured: two identical requests
    // produced {101,3,110,110} and then {16,110,126,110}).
    float *conv_state = nullptr;
    int64_t conv_state_bytes = 0;
};

struct InklingPinnedWeights {
    InklingTopology top;
    // `embed`/`unembed` are FP16, NOT NVFP4 (see InklingTensorIsPinnedNvfp4 in
    // the converter): they carry the residual stream at full width, and 4-bit
    // rounding there displaces every token vector at the input.
    __half *embed = nullptr;           // [vocab, hidden] FP16
    __half *head = nullptr;            // [vocab, hidden] FP16 (unembed)
    int embed_rows = 0;
    int embed_cols = 0;
    __half *embed_norm = nullptr;
    __half *final_norm = nullptr;
    // AttnRes query banks, FP32: [layers][nq][hidden] with nq = 3 on the full
    // layers and 1 elsewhere, so the stride is LAYER-dependent.
    float *attn_queries = nullptr;     // [sum(nq_l)][hidden]
    int *attn_query_offset = nullptr;  // [layers] host array of row offsets
    float *mlp_queries = nullptr;      // [layers][hidden]
    float *out_query = nullptr;        // [hidden]
    std::vector<InklingLayerWeights> layers;
    std::uint64_t resident_bytes = 0;
};

// ── API ────────────────────────────────────────────────────────────────────
arrow::Status LoadInklingTopology(const std::string &model_path,
                                  InklingTopology &out);

// Cross-checks the loaded index against the frozen passport (relaxed under
// LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY).
bool InklingTopologyMatchesFrozen(const InklingTopology &top);

// Uploads the pinned plane. Returns false on any missing/short tensor (a
// partially loaded model is never left behind).
bool LoadInklingPinnedWeights(const std::string &root, InklingPinnedWeights &pw,
                              cudaStream_t stream);

void FreeInklingPinnedWeights(InklingPinnedWeights &pw);

// The neutral routed-expert store (mmap + LRU, no host copy).
InklingExpertDb LoadInklingExpertIndex(const std::string &model_path,
                                       int num_layers, int experts_per_layer,
                                       size_t max_open,
                                       bool warm_on_load = false);
// One decode token's working set: top_k draws from every layer.
size_t InklingExpertWindowFloor(const InklingTopology &top);
size_t InklingResolveExpertWindow(const InklingTopology &top, long requested);
void EnsureInklingExpertResident(InklingExpertDb &db, int layer, int expert,
                                 cudaStream_t stream);
void FreeInklingExpertIndex(InklingExpertDb &db);

}  // namespace lsxcommon::inkling
