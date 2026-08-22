#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_set.h"
#include "arrow/api.h"
#include "lsxcommon/vram_estimate.h"

namespace lsxcommon {
namespace gemma {

enum class TensorKind {
    Unknown,
    Visual,
    Embed,
    FinalNorm,
    AttnQ,
    AttnK,
    AttnV,
    AttnO,
    AttnQNorm,
    AttnKNorm,
    MlpGate,
    MlpUp,
    MlpDown,
    ExpertsGateUp,
    ExpertsDown,
    RouterProj,
    RouterScale,
    RouterPerExpertScale,
    InputNorm,
    PostNorm,
    PreFFN,
    PostFFN,
    PreFFN2,
    PostFFN2,
    LayerScalar,
};

struct ParsedTensor {
    TensorKind kind = TensorKind::Unknown;
    int layer = -1;
};

ParsedTensor classify_gemma_tensor(const std::string &name);
bool gemma_tensor_is_nvfp4(TensorKind kind);

struct GemmaTopology {
    int num_layers = 30;
    std::vector<std::string> layer_types; // "sliding_attention" | "full_attention"
    int hidden_size = 2816;
    int num_attention_heads = 16;
    int num_key_value_heads = 8;
    int num_global_key_value_heads = 2;
    int head_dim = 256;
    int global_head_dim = 512;
    int sliding_window = 1024;
    int intermediate_size = 2112; // dense MLP
    int moe_intermediate_size = 704;
    int num_experts = 128;
    int top_k_experts = 8;
    float rms_norm_eps = 1e-6f;
    float rope_theta_full = 1000000.0f;
    float partial_rotary_factor = 0.25f;
    float rope_theta_sliding = 10000.0f;
    int vocab_size = 262144;
    float final_logit_softcapping = 30.0f;
    int bos_id = 2;
    int eos_id = 1;
    int pad_id = 0;
    int total_pinned_tensors = 0;
    VramProfile vram;
    std::string error_msg;
};

// Parses text_config out of <input>/config.json. Mandatory file; hard error_msg.
GemmaTopology load_gemma_topology(const std::string &config_path);

// Writes <output_root>/logestix/model.index.json ("format": "logestix-gemma").
arrow::Status WriteGemmaModelIndex(const std::string &output_root,
                                   const GemmaTopology &top);

// Full conversion pipeline (protocol stage 1): safetensors -> GLM-style
// layout (pinned/global.arrow + pinned/layer_N.arrow raw BF16/F32; routed
// experts NVFP4-packed one-file-per-expert under logestix/experts).
arrow::Status ConvertGemmaModel(const std::string &input_path,
                                const std::string &output_root);

} // namespace gemma
} // namespace lsxcommon
