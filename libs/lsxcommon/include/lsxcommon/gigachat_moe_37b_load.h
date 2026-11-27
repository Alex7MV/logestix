#pragma once

// Loader-stage passport for GigaChat4-37B-A4B. Parses
// <model_path>/logestix/model.index.json and validates it against the
// compile-time frozen geometry (src/gigachat_moe_37b_constants.h) before any
// weight is touched.
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "lsxcommon/gigachat_moe_load.h"  // GcGatedNorm, GcExpertEntry
#include "lsxcommon/gigachat_moe_37b_convert.h"  // Gc4FrozenGeometry
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/vram_estimate.h"

namespace lsxcommon::gigachat4 {

struct Gc4Index {
    std::string format;
    std::string model;
    int num_layers = 0;
    std::vector<std::string> layer_types;  // "full_attention" | "linear_attention"
    std::vector<int> full_attention_layers;
    int hidden_size = 0;
    int vocab_size = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int num_experts = 0;
    int num_experts_per_tok = 0;
    int moe_intermediate_size = 0;
    int dense_intermediate_size = 0;
    int first_k_dense_replace = 0;
    int q_lora_rank = 0;
    int kv_lora_rank = 0;
    int qk_nope_head_dim = 0;
    int qk_rope_head_dim = 0;
    int v_head_dim = 0;
    double rms_norm_eps = 0.0;
    double router_scaling_factor = 0.0;
    double swiglu_limit = 0.0;
    double logit_softcap = 0.0;
    double attn_scale = 0.0;
    int index_heads = 0;
    int index_head_dim = 0;
    int index_topk = 0;
    int index_rope = 0;
    int bos_token_id = -1;
    int eos_token_id = -1;
    int pad_token_id = -1;
    int mtp_tensors = 0;
    int total_pinned_tensors = 0;
    VramProfile vram;
};

// ── Device weight graph ─────────────────────────────────────────────────────
// One graph type for both GigaChat checkpoints (GcLayerWeights carries the DSA
// indexer planes as an optional member); the GC4 loader fills it in place.
using Gc4Weights = gigachat::GcWeights;

// One member's frozen expectation: the format tag it must see and the geometry
// its passport fixes. Supplied by the model TU, never read from another model.
struct Gc4FrozenExpect {
    const char *format = "";
    Gc4FrozenGeometry geo;
    std::vector<int> full_attention_layers;  // empty = do not check the map
    int gate_rank = 16;
};

// The GigaChat4-37B-A4B passport's expectation.
Gc4FrozenExpect Gc4DefaultExpect();

bool LoadGigaChat4IndexWithFormat(const std::string &model_path,
                                 const std::string &expected_format,
                                 Gc4Index &out, std::string &error);
bool LoadGigaChat4Index(const std::string &model_path, Gc4Index &out,
                        std::string &error);

// Frozen-geometry gate (bypassed only under LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY).
bool ValidateGigaChat4GeometryAgainst(const Gc4Index &idx,
                                     const Gc4FrozenExpect &expect,
                                     std::string &error);
bool ValidateGigaChat4Geometry(const Gc4Index &idx, std::string &error);

bool LoadGigaChat4WeightsWithExpect(const std::string &model_path,
                                   const Gc4Index &idx, Gc4Weights &w,
                                   cudaStream_t stream,
                                   const Gc4FrozenExpect &expect,
                                   std::string &error);
bool LoadGigaChat4Weights(const std::string &model_path, const Gc4Index &idx,
                          Gc4Weights &w, cudaStream_t stream,
                          std::string &error);

} // namespace lsxcommon::gigachat4
