#pragma once

#include <arrow/api.h>

#include <string>
#include <vector>

#include "lsxcommon/granite_dense_switch_constants.h"

namespace lsxcommon {
namespace granite_switch {

// Structural classification of an HF Granite-Switch tensor name.
enum class GraniteSwitchTensorKind {
    Unknown,
    Embed,            // model.embed_tokens.weight (tied LM head)
    FinalNorm,        // model.norm.weight
    InputNorm,        // model.layers.N.input_layernorm.weight
    PostNorm,         // model.layers.N.post_attention_layernorm.weight
    AttnQkvBase,      // ...self_attn.qkv_proj.base_layer.weight
    AttnQkvLoraA,     // ...self_attn.qkv_proj.lora_A_slices.{0,1,2}
    AttnQkvLoraB,     // ...self_attn.qkv_proj.lora_B_slices.{0,1,2}
    AttnOBase,        // ...self_attn.o_proj.base_layer.weight
    AttnOLoraA,       // ...self_attn.o_proj.lora_A
    AttnOLoraB,       // ...self_attn.o_proj.lora_B
    MlpGateUpBase,    // ...shared_mlp.input_linear.base_layer.weight [2I, H]
    MlpGateUpLoraA,   // ...shared_mlp.input_linear.lora_A_slices.{0,1}
    MlpGateUpLoraB,   // ...shared_mlp.input_linear.lora_B_slices.{0,1}
    MlpDownBase,      // ...shared_mlp.output_linear.base_layer.weight
    MlpDownLoraA,     // ...shared_mlp.output_linear.lora_A
    MlpDownLoraB,     // ...shared_mlp.output_linear.lora_B
    SwitchCodebook,   // model.switch.codebook (F32 [capacity, dim])
    SwitchLut,        // model.switch.control_to_substitute_lut (I64 [vocab])
};

struct ParsedGraniteSwitchTensor {
    GraniteSwitchTensorKind kind = GraniteSwitchTensorKind::Unknown;
    int layer = -1;
    int slice = -1;
};

// Classifies an HF Granite-Switch tensor name. Unknown kinds are an error in
// the converter (no silent skipping).
ParsedGraniteSwitchTensor classify_granite_switch_tensor(
    const std::string &name);

// True when `kind` is the NVFP4 base-weight plane. Adapter planes, norms and
// the switch codebook stay raw (FP16/BF16/F32).
bool granite_switch_tensor_is_nvfp4(GraniteSwitchTensorKind kind);

// Reads <output_root>/logestix/model.index.json.
GraniteSwitchTopology load_granite_switch_topology(
    const std::string &index_path);

// Writes <output_root>/logestix/model.index.json (format
// "logestix-granite-switch").
arrow::Status WriteGraniteSwitchModelIndex(
    const std::string &output_root, const GraniteSwitchTopology &top);

// Converts a Granite-Switch source directory (config.json + safetensors) into
// the logestix-granite-switch Arrow layout under output_root.
arrow::Status ConvertGraniteSwitchModel(const std::string &input_path,
                                        const std::string &output_root);

} // namespace granite_switch
} // namespace lsxcommon
