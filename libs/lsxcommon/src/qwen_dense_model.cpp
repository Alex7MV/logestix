#include "qwen_dense_model.h"
#include "lsxcommon/log.h"

#include <absl/strings/str_cat.h>

namespace lsxcommon {

const std::string Qwen3_5_9BModel::kType = "qwen3.5-9b";

Qwen3_5_9BModel::Qwen3_5_9BModel(ModelInitConfig cfg)
    : QwenBaseModel(std::move(cfg))
{
    type_ = kType;
}

bool Qwen3_5_9BModel::ValidateGeometry(const qwen::QwenTopology &top) const
{
    if (SyntheticGeometryAllowed()) return true;
    const GeoFieldCheck fields[] = {
        {"hidden_size", kHiddenSize, top.hidden_size},
        {"num_layers", kNumLayers, top.num_layers},
        {"num_attention_heads", kNumHeads, top.num_attention_heads},
        {"num_key_value_heads", kNumKVHeads, top.num_key_value_heads},
        {"head_dim", kHeadDim, top.head_dim},
        {"intermediate_size", kIntermediateSize, top.intermediate_size},
        {"linear_num_value_heads", kLinearNumValueHeads,
         top.linear_num_value_heads},
    };
    return CheckGeometryFields("qwen3.5-9b", fields, 7);
}

} // namespace lsxcommon