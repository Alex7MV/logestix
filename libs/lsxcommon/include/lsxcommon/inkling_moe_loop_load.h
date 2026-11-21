#pragma once

// Inkling-2-Mini loader. The converted tree keeps the release's precision:
//   * `pinned/global.arrow`        embed/unembed (BF16->FP16 on upload), norms,
//                                  AttnRes queries (F32), merged message_queries
//                                  [loop_layers, 4, hidden] F32
//   * `pinned/layer_<l>.arrow`     dense projections (BF16->FP16), norms,
//                                  L2-clip scales, rel bank, sinks, conv taps,
//                                  router weight/bias, and (loop layers only)
//                                  the four MessageGate controller triples
//   * `experts/layer_<l>/exp_<e>.arrow`  routed experts (neutral ExpertStore;
//                                  NVFP4 layers 0..19, MXFP8 20..23)
//
// The dense plane is device-resident; the routed set is only ever an mmap view.
// The loop adds no weight planes: passes share the loop layers' weights, and
// only the KV/conv STATE is per-pass (owned by the session, not the loader).
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
struct InklingLoopTopology {
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
    // the loop core
    int loop_start = 0;
    int loop_end = 0;
    int loop_passes = 1;
    int loop_gate_rank = 0;
    int bos_id = -1;
    int eos_id = -1;
    int stop_id_alt = -1;
    int pad_id = -1;
    int tokenizer_kind = 0;
    int skipped_visual = 0;
    std::vector<int> full_layers;
    std::vector<int> sliding_layers;
    std::vector<bool> expert_nvfp4;
    std::vector<bool> expert_fp8;
    std::string error_msg;

    bool IsFull(int layer) const
    {
        for (int l : full_layers)
            if (l == layer) return true;
        return false;
    }
    bool IsLoop(int layer) const
    {
        return layer >= loop_start && layer < loop_end;
    }
    int LoopLayers() const { return loop_end - loop_start; }
    // 24 physical + (passes-1) * loop_layers cache slots.
    int CacheLayers() const
    {
        return num_layers + (loop_passes - 1) * LoopLayers();
    }
    // Cache slot of (physical layer, pass); pass 0 uses the physical slot.
    int CacheSlot(int layer, int pass) const
    {
        return pass == 0 ? layer
                         : num_layers + (pass - 1) * LoopLayers() +
                               (layer - loop_start);
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
};

// ── routed-expert store entry (field order matches ExpertStore::DecodeEntry) ─
struct InklingLoopExpertEntry {
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
static_assert(std::is_trivially_copyable_v<InklingLoopExpertEntry>,
              "ExpertStore entries must stay trivially copyable views");
static_assert(sizeof(InklingLoopExpertEntry) <= 128,
              "ExpertStore entry must stay a compact view");

using InklingLoopExpertDb = gpu::expert_store::ExpertStore<InklingLoopExpertEntry>;

// ── pinned weights (device-resident for the whole session) ──────────────────
// One FP16 dense projection: [rows, cols], row-major.
struct InklingLoopDenseF16 {
    __half *d = nullptr;
    int rows = 0;
    int cols = 0;
};

// One MessageGate controller: down [rank, 2*hidden] FP16, forget/gain
// [hidden, rank] FP32 (the reference computes the gates in FP32).
struct InklingLoopMessageGate {
    InklingLoopDenseF16 down;
    float *forget = nullptr;
    float *gain = nullptr;
};

struct InklingLoopLayerWeights {
    bool full = false;
    bool loop = false;
    // BF16 in the checkpoint, FP16 on device.
    InklingLoopDenseF16 wq, wk, wv, wo, wr, attn_gate;
    InklingLoopDenseF16 ksconv_proj, vsconv_proj;  // full layers only
    // Norms (FP16 weights, never quantized).
    __half *attn_norm = nullptr;    // sliding layers
    __half *q_in_norm = nullptr;    // full layers
    __half *k_in_norm = nullptr;
    __half *v_in_norm = nullptr;
    __half *mlp_norm = nullptr;
    // FP32 side tensors.
    __half *sink_proj = nullptr;    // [heads, hidden] FP16 (matvec)
    float *q_clip = nullptr;        // [heads, head_dim] learned scale
    float *rel_clip = nullptr;      // [heads, d_rel]   learned scale
    float *rel_proj = nullptr;      // [d_rel, rel_extent]
    float *kconv_w = nullptr;       // [kv_dim, k] conv1d taps (sliding)
    float *vconv_w = nullptr;
    float *kconv_static = nullptr;  // [k, kv_dim] lag-0-first taps (full)
    float *vconv_static = nullptr;
    float *attn_conv_w = nullptr;   // [hidden, k] conv1d taps
    float *mlp_conv_w = nullptr;
    float *router_w = nullptr;      // [experts, hidden] FP32
    float *router_bias = nullptr;   // [experts] FP32 selection bias
    // Loop controllers: q, k, v, mlp message gates (loop layers only).
    InklingLoopMessageGate msg[4];
    // Recurrent conv state, FOUR convolutions per layer (k/v + post-attn +
    // post-MLP): [k][kv_dim] x2 then [k][hidden] x2. PER-SEQUENCE state; the
    // session owns one copy per cache slot (the loader's copy is the pass-0
    // template and is zeroed at every request start).
    float *conv_state = nullptr;
    int64_t conv_state_bytes = 0;
};

struct InklingLoopPinnedWeights {
    InklingLoopTopology top;
    // embed/unembed are FP16 (the residual stream at full width).
    __half *embed = nullptr;           // [vocab, hidden] FP16
    __half *head = nullptr;            // [vocab, hidden] FP16 (unembed)
    int embed_rows = 0;
    int embed_cols = 0;
    __half *embed_norm = nullptr;
    __half *final_norm = nullptr;
    // AttnRes query banks, FP32: [layers][nq][hidden], nq = 3 on full layers.
    float *attn_queries = nullptr;     // [sum(nq_l)][hidden]
    int *attn_query_offset = nullptr;  // [layers] host array of row offsets
    float *mlp_queries = nullptr;      // [layers][hidden]
    float *out_query = nullptr;        // [hidden]
    // Merged cross-pass message queries, FP32 [loop_layers][4][hidden],
    // order [q, k, v, mlp] per loop layer.
    float *message_queries = nullptr;
    std::vector<InklingLoopLayerWeights> layers;
    std::uint64_t resident_bytes = 0;
};

// ── API ────────────────────────────────────────────────────────────────────
arrow::Status LoadInklingLoopTopology(const std::string &model_path,
                                      InklingLoopTopology &out);

// Cross-checks the loaded index against the frozen passport (relaxed under
// LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY).
bool InklingLoopTopologyMatchesFrozen(const InklingLoopTopology &top);

// Uploads the pinned plane. Returns false on any missing/short tensor (a
// partially loaded model is never left behind).
bool LoadInklingLoopPinnedWeights(const std::string &root,
                                  InklingLoopPinnedWeights &pw,
                                  cudaStream_t stream);

void FreeInklingLoopPinnedWeights(InklingLoopPinnedWeights &pw);

// The neutral routed-expert store (mmap + LRU, no host copy).
InklingLoopExpertDb LoadInklingLoopExpertIndex(const std::string &model_path,
                                               int num_layers,
                                               int experts_per_layer,
                                               size_t max_open,
                                               bool warm_on_load = false);
// One decode token's working set: top_k draws from every layer APPLICATION
// (the loop layers are visited once per pass).
size_t InklingLoopExpertWindowFloor(const InklingLoopTopology &top);
size_t InklingLoopResolveExpertWindow(const InklingLoopTopology &top,
                                      long requested);
void EnsureInklingLoopExpertResident(InklingLoopExpertDb &db, int layer,
                                     int expert, cudaStream_t stream);
void FreeInklingLoopExpertIndex(InklingLoopExpertDb &db);

}  // namespace lsxcommon::inkling