#pragma once

#include <arrow/api.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/schema.h"

namespace lsxcommon::alice {

// Topology parsed from a converted logestix-alice-nvfp4 model.index.json.
struct AliceMoeTopology {
    std::string format;
    std::string model;
    int num_layers = 0;
    int hidden_size = 0;
    int vocab_size = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int head_dim = 0;
    int linear_num_key_heads = 0;
    int linear_num_value_heads = 0;
    int linear_key_head_dim = 0;
    int linear_value_head_dim = 0;
    int linear_conv_kernel_dim = 4;
    int num_experts = 0;
    int num_experts_per_tok = 0;
    int moe_intermediate_size = 0;
    int shared_expert_intermediate_size = 0;
    int block_attn_res_block_size = 4;
    int mtp_num_hidden_layers = 0;
    double rms_norm_eps = 1e-6;
    double partial_rotary_factor = 0.25;
    double rope_theta = 1e6;
    int bos_id = 1;
    int eos_id = 2;
    int pad_id = -1;
    std::vector<std::string> layer_types;
    std::string error_msg;
};

// Reads <model_path>/logestix/model.index.json. Enforces the format tag. Fills
// out.error_msg and returns a non-OK status on any failure.
arrow::Status LoadAliceTopology(const std::string &model_path,
                                AliceMoeTopology &out);

// Strict comparison against the frozen passport (libs/lsxcommon/src/
// alice_moe_constants.h): every geometry field and the layer-type pattern.
bool AliceTopologyMatchesFrozen(const AliceMoeTopology &top);

// ── Pinned device weight graph ──────────────────────────────────────────────

// Per-layer resident weights. Norms/projections/gates are FP16 device buffers;
// large linear matrices are neutral NVFP4 weights.
struct AliceLayerWeights {
    bool loaded = false;
    bool full = false;

    __half *input_norm = nullptr;
    __half *post_norm = nullptr;
    __half *attn_res_norm = nullptr;  // null on layer 0
    __half *mlp_res_norm = nullptr;
    __half *attn_res_proj = nullptr;  // null on layer 0
    __half *mlp_res_proj = nullptr;

    // KDA (linear_attention)
    __half *kda_q = nullptr;  // FP16 (precision policy)
    __half *kda_k = nullptr;
    __half *kda_v = nullptr;
    __half *kda_o = nullptr;
    __half *kda_fa = nullptr;
    __half *kda_fb = nullptr;  // gate decay projection, kept FP16 (sensitive)
    __half *kda_ga = nullptr;
    __half *kda_gb = nullptr;  // output gate projection, kept FP16
    __half *kda_b = nullptr;
    __half *kda_qconv = nullptr;
    __half *kda_kconv = nullptr;
    __half *kda_vconv = nullptr;
    __half *kda_alog = nullptr;
    __half *kda_dt = nullptr;
    __half *kda_onorm = nullptr;

    // Gated full attention
    __half *fa_q = nullptr;  // FP16 (precision policy)
    __half *fa_k = nullptr;
    __half *fa_v = nullptr;
    __half *fa_o = nullptr;
    __half *fa_qnorm = nullptr;
    __half *fa_knorm = nullptr;

    // MoE
    __half *router_gate = nullptr;  // [E, hidden]
    __half *router_bias = nullptr;  // [E]
    __half *shared_gate = nullptr;  // [hidden]
    lsx_nvfp4::LsxNvfp4Weight shared_gate_proj, shared_up_proj, shared_down_proj;
};

struct AlicePinnedWeights {
    AliceMoeTopology top;
    __half *embed = nullptr;              // [vocab, hidden]
    __half *head = nullptr;               // [vocab, hidden]
    __half *final_norm = nullptr;         // [hidden]
    __half *attnres_final_proj = nullptr; // [hidden]
    __half *attnres_final_norm = nullptr; // [hidden]
    std::vector<AliceLayerWeights> layers;
    std::uint64_t resident_bytes = 0;
};

// Loads all pinned weights from <root>/logestix/pinned or <root>/pinned.
bool LoadAlicePinnedWeights(const std::string &root, AlicePinnedWeights &pw,
                            cudaStream_t stream);
void FreeAlicePinnedWeights(AlicePinnedWeights &pw);

// Host-resident routed-expert store with lazy per-expert arrow loading.
class AliceExpertHost {
public:
    explicit AliceExpertHost(std::string root) : root_(std::move(root)) {}
    // Returns nullptr on missing/corrupt file.
    const PackedExpert *Get(int layer, int expert);

private:
    std::string root_;
    std::unordered_map<std::int64_t, PackedExpert> cache_;
};

} // namespace lsxcommon::alice
