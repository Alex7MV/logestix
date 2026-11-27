#pragma once

// NVIDIA-Nemotron-3-Ultra-550B-A55B loader: index parse, frozen-geometry gate
// and the resident device weight graph.
//
// The index is the loader-stage passport: it is validated against the
// compile-time frozen geometry (src/nemotron_moe_ultra_constants.h) before a
// single weight is touched, so a tree converted for another Nemotron width
// (the 30B Lightning sibling) fails by FIELD NAME instead of uploading weights
// of the wrong shape.
//
// Routed experts are NOT part of the resident graph: 24 576 of them at ~11.8 MB
// is ~290 GB, so they stream through the host store + device slot LRU
// (nemotron_moe_ultra_expert_cache.h). `resident_bytes` therefore covers the
// dense backbone only (embed/head/norms/mixer projections/shared+latent).

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/vram_estimate.h"

namespace lsxcommon::nemotron_ultra {

struct UltraIndex {
    // Schema layout generation declared by model.index.json
    // (lsxcommon::kIndexLayout). 0 = absent.
    int64_t layout = 0;
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
    // Latent MoE bridge. > 0 means the routed experts run at this width and the
    // fc1/fc2 projections are mandatory; 0 means the identity latent.
    int moe_latent_size = 0;
    // Grouped routing: scores are split into `n_group` equal groups, the best
    // `topk_group` survive, and top-k is taken inside them.
    int n_group = 0;
    int topk_group = 0;
    float routed_scaling_factor = 0.0f;
    bool norm_topk_prob = false;
    bool experts_no_gate = false;
    int bos_token_id = -1;
    // `eos_token_id` is the PRIMARY stop; `eos_token_ids` is the full set. This
    // checkpoint declares [2, 11], and a reader that keeps only the first runs
    // past the model's own turn stop.
    int eos_token_id = -1;
    std::vector<int> eos_token_ids;
    int pad_token_id = -1;
    int mtp_tensors = 0;
    // A windowed conversion leaves a partial tree: every layer file the run did
    // not write is absent, so the loader must refuse it and name the first
    // missing layer instead of decoding a model with holes.
    bool partial = false;
    std::vector<int> layers_converted;
    int total_pinned_tensors = 0;
    int total_experts = 0;
    VramProfile vram;
};

// Reads <model_path>/logestix/model.index.json. Returns false and fills `error`
// on a missing/unparseable index or a foreign format tag.
bool LoadUltraIndex(const std::string &model_path, UltraIndex &out,
                    std::string &error);

// Cross-checks `idx` against the frozen passport, in a fixed field order, and
// rejects a partial tree naming the first layer it does not hold. Honors
// LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1 for synthetic fixtures (format and
// completeness are still checked).
bool ValidateUltraGeometry(const UltraIndex &idx, std::string &error);

// ── Device weight graph ─────────────────────────────────────────────────────

// Per-layer device weights. Fields stay null by layer type.
struct UltraLayerWeights {
    __half *d_norm = nullptr;                       // [H] FP16
    // Mamba-2 mixer.
    lsx_nvfp4::LsxNvfp4Weight in_proj;              // [in_proj, H]
    lsx_nvfp4::LsxNvfp4Weight out_proj;             // [H, d_inner]
    float *d_conv_w = nullptr;                      // [conv_dim * K] F32
    float *d_conv_b = nullptr;                      // [conv_dim] F32
    float *d_A_log = nullptr;                       // [heads] F32
    float *d_D = nullptr;                           // [heads] F32
    float *d_dt_bias = nullptr;                     // [heads] F32
    float *d_mamba_norm = nullptr;                  // [d_inner] F32
    // Full attention (NoPE).
    lsx_nvfp4::LsxNvfp4Weight q, k, v, o;
    // MoE.
    __half *d_router = nullptr;                     // [E, H] FP16
    float *d_router_bias = nullptr;                 // [E] F32
    lsx_nvfp4::LsxNvfp4Weight latent_up;            // [L, H]   null when no latent
    lsx_nvfp4::LsxNvfp4Weight latent_down;          // [H, L]   null when no latent
    lsx_nvfp4::LsxNvfp4Weight shared_up;            // [SI, H]
    lsx_nvfp4::LsxNvfp4Weight shared_down;          // [H, SI]
    void Free();
};

struct UltraWeights {
    __half *d_embed = nullptr;       // [vocab, H] FP16
    __half *d_head = nullptr;        // [vocab, H] FP16
    __half *d_final_norm = nullptr;  // [H] FP16
    std::vector<UltraLayerWeights> layers;  // [num_layers]
    std::uint64_t resident_bytes = 0;       // dense only
    void Free();
};

// Uploads the converted dense weights under <model_path>/logestix/ to the
// device. Routed experts are handled by LoadUltraExpertStore /
// AllocateUltraExpertCache. Strict: returns false and fills `error` on any
// missing/malformed tensor, naming it.
bool LoadUltraWeights(const std::string &model_path, const UltraIndex &idx,
                      UltraWeights &w, cudaStream_t stream, std::string &error);

} // namespace lsxcommon::nemotron_ultra
