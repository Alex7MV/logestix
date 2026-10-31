#pragma once

// Ling-3.1-flash (BailingMoeV3) converter: config parse, tensor classifier,
// FP8 block-128 -> NVFP4 requant and the conversion pipeline.
#include <arrow/status.h>

#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/vram_estimate.h"

namespace lsxcommon::ling {

struct LingTopology {
    int hidden = 0;
    int layers = 0;
    int vocab = 0;
    int heads = 0;
    int kv_heads = 0;
    int head_dim = 0;
    int qk_head_dim = 0;
    int qk_nope = 0;
    int qk_rope = 0;
    int v_head = 0;
    int q_lora = 0;
    int kv_lora = 0;
    int conv_kernel = 0;
    int num_experts = 0;
    int top_k = 0;
    int n_group = 0;
    int topk_group = 0;
    int moe_intermediate = 0;
    int shared_intermediate = 0;
    int dense_intermediate = 0;
    int first_k_dense = 0;
    int layer_group_size = 0;
    double rms_eps = 0.0;
    double rope_theta = 0.0;
    double routed_scale = 0.0;
    int bos_id = -1;
    int eos_id = -1;
    int pad_id = -1;
    std::vector<float> expert_swiglu_limit;
    std::vector<float> shared_swiglu_limit;
    std::vector<int> mla_layers;
    std::vector<int> kda_layers;
    std::vector<int> moe_layers;
    std::vector<int> dense_layers;
    std::string error_msg;  // non-empty when config is invalid
};

LingTopology ParseLingConfig(const std::string &config_json_path);

// GPU-dequantizes an F8_E4M3 tensor with per-128x128 F32 scale_inv tiles into
// a BF16 byte buffer (row-major [rows, cols]); partial last tiles clamped.
arrow::Status LingDequantFp8Block128ToBf16Gpu(
    const std::vector<uint8_t> &fp8, const std::vector<int64_t> &shape,
    const std::vector<uint8_t> &scale_f32,
    const std::vector<int64_t> &scale_shape, std::vector<uint8_t> &bf16);

float LingFp8E4m3Decode(uint8_t bits);

arrow::Status ConvertLingModel(const std::string &input_path,
                               const std::string &output_path);

} // namespace lsxcommon::ling
