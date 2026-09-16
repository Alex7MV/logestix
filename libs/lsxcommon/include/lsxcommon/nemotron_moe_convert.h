#pragma once

#include <arrow/api.h>
#include <string>
#include <vector>

#include "lsxcommon/vram_estimate.h"

namespace lsxcommon::nemotron_moe {

// Tensor kinds of the NVIDIA-Nemotron-3.5-Lightning-30B-A3B checkpoint
// (model_type nemotron_h, NemotronHForCausalLM).
enum class NemotronTensorKind {
    Unknown,
    Embed,          // backbone.embeddings.weight            BF16 (raw)
    LmHead,         // lm_head.weight                        BF16 (raw)
    FinalNorm,      // backbone.norm_f.weight                BF16 (raw)
    LayerNorm,      // backbone.layers.N.norm.weight         BF16 (raw)
    MambaInProj,    // layers.N.mixer.in_proj.weight         NVFP4
    MambaOutProj,   // layers.N.mixer.out_proj.weight        NVFP4
    MambaConv,      // layers.N.mixer.conv1d.weight          BF16 (raw)
    MambaConvBias,  // layers.N.mixer.conv1d.bias            BF16 (raw)
    MambaALog,      // layers.N.mixer.A_log                  F32  (raw)
    MambaD,         // layers.N.mixer.D                      BF16 (raw)
    MambaDtBias,    // layers.N.mixer.dt_bias                BF16 (raw)
    MambaNorm,      // layers.N.mixer.norm.weight            BF16 (raw)
    AttnQ,          // layers.N.mixer.q_proj.weight          NVFP4
    AttnK,          // layers.N.mixer.k_proj.weight          NVFP4
    AttnV,          // layers.N.mixer.v_proj.weight          NVFP4
    AttnO,          // layers.N.mixer.o_proj.weight          NVFP4
    RouterGate,     // layers.N.mixer.gate.weight            BF16 (raw)
    RouterBias,     // layers.N.mixer.gate.e_score_correction_bias  F32 (raw)
    SharedUpProj,   // layers.N.mixer.shared_experts.up_proj.weight   NVFP4
    SharedDownProj, // layers.N.mixer.shared_experts.down_proj.weight NVFP4
    ExpertUpProj,   // layers.N.mixer.experts.E.up_proj.weight   NVFP4 (per expert)
    ExpertDownProj, // layers.N.mixer.experts.E.down_proj.weight NVFP4 (per expert)
    Mtp,            // mtp.*                                 cut + counted
};

struct ParsedNemotronTensor {
    NemotronTensorKind kind = NemotronTensorKind::Unknown;
    int layer = -1;
    int expert = -1;  // only for ExpertUpProj / ExpertDownProj
};

// Classifies an HF NemotronH tensor name.
ParsedNemotronTensor classify_nemotron_tensor(const std::string &name);

struct NemotronTopology {
    int num_layers = 0;
    std::vector<std::string> layer_types;  // "mamba" | "moe" | "attention"
    int hidden_size = 0;
    int vocab_size = 0;
    int mamba_num_heads = 0, mamba_head_dim = 0, ssm_state_size = 0;
    int n_groups = 0, conv_kernel = 0;
    int num_attention_heads = 0, num_key_value_heads = 0, head_dim = 0;
    double rope_theta = 10000.0, rms_norm_eps = 1e-5;
    int num_experts = 0, num_experts_per_tok = 0;
    int moe_intermediate_size = 0, shared_expert_intermediate_size = 0;
    int eos_token_id = -1, bos_token_id = -1, pad_token_id = -1;
    int total_pinned_tensors = 0;
    VramProfile vram;      // conversion-time VRAM requirements
    std::string error_msg; // non-empty when config/index is invalid
};

// Parses <input_dir>/config.json (flat root; model_type nemotron_h).
NemotronTopology ParseNemotronConfig(const std::string &input_dir);

// Converts a Nemotron-3.5-Lightning-30B-A3B directory (safetensors shards +
// config.json) into the unified <output_root>/logestix arrow layout
// (`logestix-nemotron-nvfp4`).
arrow::Status ConvertNemotronModel(const std::string &input_dir,
                                   const std::string &output_root);

} // namespace lsxcommon::nemotron_moe
