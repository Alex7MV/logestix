#pragma once

// Loader-stage passport for GigaChat3.5-432B-A28B-Reasoning. Parses
// <model_path>/logestix/model.index.json and validates it against the
// compile-time frozen geometry (src/gigachat_moe_constants.h) before any
// weight is touched. Host-only: no CUDA dependency in Plan 1.
#include <string>
#include <vector>

#include "lsxcommon/vram_estimate.h"

namespace lsxcommon::gigachat {

struct GigaChatIndex {
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
    int first_k_dense_replace = 0;
    double rms_norm_eps = 0.0;
    double router_scaling_factor = 0.0;
    double swiglu_limit = 0.0;
    int bos_token_id = -1;
    int eos_token_id = -1;
    int pad_token_id = -1;
    int mtp_tensors = 0;
    int total_pinned_tensors = 0;
    VramProfile vram;
};

// Reads <model_path>/logestix/model.index.json. Returns false and fills
// `error` on a missing/unparseable index or an unknown format tag.
bool LoadGigaChatIndex(const std::string &model_path, GigaChatIndex &out,
                       std::string &error);

// Cross-checks `idx` against the frozen passport. Honors
// LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1 for synthetic fixtures. Returns false and
// fills `error` on the first mismatch.
bool ValidateGigaChatGeometry(const GigaChatIndex &idx, std::string &error);

} // namespace lsxcommon::gigachat
