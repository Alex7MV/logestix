#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/vram_estimate.h"

namespace lsxcommon::nemotron_moe {

// Parsed `logestix/model.index.json` for the Nemotron-3.5 hybrid. This is the
// loader-stage passport: it is validated against the compile-time frozen
// geometry (src/nemotron_moe_constants.h) before any weight is touched.
struct NemotronIndex {
    std::string format;
    std::string model;
    int num_layers = 0;
    std::vector<std::string> layer_types;  // "mamba" | "moe" | "attention"
    std::vector<int> attn_layer_idx;
    int hidden_size = 0;
    int vocab_size = 0;
    int mamba_num_heads = 0;
    int mamba_head_dim = 0;
    int ssm_state_size = 0;
    int n_groups = 0;
    int conv_kernel = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int head_dim = 0;
    double rope_theta = 0.0;
    double rms_norm_eps = 0.0;
    int num_experts = 0;
    int num_experts_per_tok = 0;
    int moe_intermediate_size = 0;
    int shared_expert_intermediate_size = 0;
    bool experts_no_gate = false;
    int bos_token_id = -1;
    int eos_token_id = -1;
    int pad_token_id = -1;
    int mtp_tensors = 0;
    int total_pinned_tensors = 0;
    VramProfile vram;
};

// Reads <model_path>/logestix/model.index.json. Returns false and fills
// `error` on a missing/unparseable index or an unknown format tag.
bool LoadNemotronIndex(const std::string &model_path, NemotronIndex &out,
                       std::string &error);

// Cross-checks `idx` against the frozen passport. Honors
// LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1 for synthetic fixtures. Returns false and
// fills `error` on the first mismatch.
bool ValidateNemotronGeometry(const NemotronIndex &idx, std::string &error);

// ── Device weight graph ─────────────────────────────────────────────────────

// Per-layer device weights. Untouched fields stay null by layer type.
struct NmLayerWeights {
    __half *d_norm = nullptr;                       // [H] FP16
    // Mamba-2 mixer.
    lsx_nvfp4::LsxNvfp4Weight in_proj;               // [in_proj, H]
    lsx_nvfp4::LsxNvfp4Weight out_proj;              // [H, d_inner]
    float *d_conv_w = nullptr;                       // [conv_dim * K] F32
    float *d_conv_b = nullptr;                       // [conv_dim] F32
    float *d_A_log = nullptr;                        // [heads] F32
    float *d_D = nullptr;                            // [heads] F32
    float *d_dt_bias = nullptr;                      // [heads] F32
    float *d_mamba_norm = nullptr;                   // [d_inner] F32
    // Full attention.
    lsx_nvfp4::LsxNvfp4Weight q, k, v, o;
    // MoE.
    __half *d_router = nullptr;                      // [E, H] FP16
    float *d_router_bias = nullptr;                  // [E] F32
    lsx_nvfp4::LsxNvfp4Weight shared_up;             // [SI, H]
    lsx_nvfp4::LsxNvfp4Weight shared_down;           // [H, SI]
    void Free();
};

// One routed expert (non-gated: up/down only).
struct NmExpertWeight {
    lsx_nvfp4::LsxNvfp4Weight up;    // [I, H]
    lsx_nvfp4::LsxNvfp4Weight down;  // [H, I]
    void Free() { up.Free(); down.Free(); }
};

struct NmWeights {
    __half *d_embed = nullptr;       // [vocab, H] FP16
    __half *d_head = nullptr;        // [vocab, H] FP16
    __half *d_final_norm = nullptr;  // [H] FP16
    std::vector<NmLayerWeights> layers;                 // [num_layers]
    std::vector<std::vector<NmExpertWeight>> experts;   // [num_layers][E]
    std::uint64_t resident_bytes = 0;
    void Free();
};

// Uploads the converted weights under <model_path>/logestix/ to the device.
// Routed experts are uploaded resident (per-expert NVFP4). Strict: returns
// false and fills `error` on any missing/malformed tensor.
bool LoadNemotronWeights(const std::string &model_path, const NemotronIndex &idx,
                         NmWeights &w, cudaStream_t stream, std::string &error);

} // namespace lsxcommon::nemotron_moe
