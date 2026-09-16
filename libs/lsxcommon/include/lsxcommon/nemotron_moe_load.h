#pragma once

#include <string>
#include <vector>

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

} // namespace lsxcommon::nemotron_moe
