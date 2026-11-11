#pragma once

// NVIDIA-Nemotron-3-Ultra-550B-A55B converter (Model Ingestion Protocol step 1).
//
// Source layout is the HF `NemotronHForCausalLM` one - `backbone.layers.N.mixer.*`
// - but the geometry is its own lineage: 108 layers, hidden 8192, 512 experts
// top-22 routed at a 2048-wide latent (fc1/fc2 bridge), grouped routing
// (n_group 8 / topk_group 1), routed_scaling_factor 5.0.
//
// Output is the Ultra schema `logestix-nemotron-ultra-nvfp4`: the sibling's
// pinned/experts arrow shapes plus the two latent projections, and an index that
// carries the router's group fields rather than a frozen constant. A tree of
// either format is rejected by the other's loader by name, never by shape.

#include <arrow/api.h>

#include <string>
#include <vector>

#include "lsxcommon/vram_estimate.h"

namespace lsxcommon::nemotron_ultra {

// Tensor kinds of the NVIDIA-Nemotron-3-Ultra-550B checkpoint.
enum class UltraTensorKind {
    Unknown,
    Embed,           // backbone.embeddings.weight                BF16 (raw)
    LmHead,          // lm_head.weight                            BF16 (raw)
    FinalNorm,       // backbone.norm_f.weight                    BF16 (raw)
    LayerNorm,       // backbone.layers.N.norm.weight              BF16 (raw)
    MambaInProj,     // layers.N.mixer.in_proj.weight             NVFP4
    MambaOutProj,    // layers.N.mixer.out_proj.weight            NVFP4
    MambaConv,       // layers.N.mixer.conv1d.weight              BF16 (raw)
    MambaConvBias,   // layers.N.mixer.conv1d.bias                BF16 (raw)
    MambaALog,       // layers.N.mixer.A_log                      BF16 (raw)
    MambaD,          // layers.N.mixer.D                          BF16 (raw)
    MambaDtBias,     // layers.N.mixer.dt_bias                    BF16 (raw)
    MambaNorm,       // layers.N.mixer.norm.weight                BF16 (raw)
    AttnQ,           // layers.N.mixer.q_proj.weight               NVFP4
    AttnK,           // layers.N.mixer.k_proj.weight               NVFP4
    AttnV,           // layers.N.mixer.v_proj.weight               NVFP4
    AttnO,           // layers.N.mixer.o_proj.weight               NVFP4
    RouterGate,      // layers.N.mixer.gate.weight                 BF16 (raw)
    RouterBias,      // layers.N.mixer.gate.e_score_correction_bias F32 (raw)
    MoeLatentUp,     // layers.N.mixer.fc1_latent_proj.weight     NVFP4 [latent, hidden]
    MoeLatentDown,   // layers.N.mixer.fc2_latent_proj.weight     NVFP4 [hidden, latent]
    SharedUpProj,    // layers.N.mixer.shared_experts.up_proj.weight   NVFP4
    SharedDownProj,  // layers.N.mixer.shared_experts.down_proj.weight NVFP4
    ExpertUpProj,    // layers.N.mixer.experts.E.up_proj.weight   NVFP4 [I, latent]
    ExpertDownProj,  // layers.N.mixer.experts.E.down_proj.weight NVFP4 [latent, I]
    Mtp,             // mtp.*                                     cut + counted
};

struct ParsedUltraTensor {
    UltraTensorKind kind = UltraTensorKind::Unknown;
    int layer = -1;
    int expert = -1;  // only for ExpertUpProj / ExpertDownProj
};

// Classifies an HF NemotronH tensor name. Never throws: a malformed index or an
// unknown suffix yields Unknown, and the converter turns that into an error.
ParsedUltraTensor classify_ultra_tensor(const std::string &name);

// True for the kinds stored NVFP4; false for the ones kept raw (norms, biases,
// the router, embeddings/head, conv1d, A_log/D/dt_bias).
bool ultra_tensor_is_nvfp4(UltraTensorKind kind);

struct UltraTopology {
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
    int moe_latent_size = 0;  // 0 => the identity latent (no fc1/fc2)
    int n_group = 0, topk_group = 0;
    float routed_scaling_factor = 0.0f;
    bool norm_topk_prob = false;
    // config.json declares a bare int; generation_config.json carries the
    // real stop set ([2, 11] for this checkpoint). Both are read, the list
    // wins, and the scalar is its first element.
    int eos_token_id = -1, bos_token_id = -1, pad_token_id = -1;
    std::vector<int> eos_token_ids;
    int total_pinned_tensors = 0;
    VramProfile vram;       // conversion-time VRAM requirements
    std::string error_msg;  // non-empty when config is invalid
};

// Parses <input_dir>/config.json (flat root; model_type nemotron_h).
UltraTopology ParseUltraConfig(const std::string &input_dir);

// Layer window for a slice conversion. `last_layer < 0` means "to the end".
// A window that does not cover every layer writes `"partial": true` and the
// covered list, so the loader refuses to run a partially converted tree.
struct ConversionWindow {
    int first_layer = 0;
    int last_layer = -1;
};

// Converts a checkpoint directory (safetensors shards + config.json) into
// <output_root>/logestix/ (`logestix-nemotron-ultra-nvfp4`).
//
// Routed experts are converted PER LAYER and freed before the next one: the
// model owns 24 576 of them (290 GB converted), so holding them all would
// exhaust any host. Peak host residency is one layer's expert records.
//
// Strict: `mtp.*` is the only cut (counted in the index); any other
// unclassified tensor is an error. Rerunning with a later window adds layers
// without rewriting the ones already converted.
arrow::Status ConvertNemotronUltraModel(const std::string &input_dir,
                                        const std::string &output_root,
                                        const ConversionWindow &window = {});

} // namespace lsxcommon::nemotron_ultra
