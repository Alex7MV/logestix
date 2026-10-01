#pragma once

#include <arrow/api.h>

#include <string>
#include <vector>

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

} // namespace lsxcommon::alice
