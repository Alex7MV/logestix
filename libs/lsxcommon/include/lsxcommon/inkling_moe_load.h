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
    std::string error_msg;

    bool IsFull(int layer) const
    {
        for (int l : full_layers)
            if (l == layer) return true;
        return false;
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

// ── API ────────────────────────────────────────────────────────────────────
arrow::Status LoadInklingTopology(const std::string &model_path,
                                  InklingTopology &out);

// Cross-checks the loaded index against the frozen passport (relaxed under
// LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY).
bool InklingTopologyMatchesFrozen(const InklingTopology &top);

}  // namespace lsxcommon::inkling
