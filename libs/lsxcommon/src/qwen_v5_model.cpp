#include "qwen_v5_model.h"
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
    auto check = [&](const char *name, int expected, int actual) {
        if (expected != actual) {
            lsxcommon::log::error(absl::StrCat(
                "[qwen3.5-9b] geometry mismatch: ", name, " expected=",
                expected, " got=", actual));
            return false;
        }
        return true;
    };
    return check("hidden_size", kHiddenSize, top.hidden_size) &&
           check("num_layers", kNumLayers, top.num_layers) &&
           check("num_attention_heads", kNumHeads, top.num_attention_heads) &&
           check("num_key_value_heads", kNumKVHeads, top.num_key_value_heads) &&
           check("head_dim", kHeadDim, top.head_dim) &&
           check("intermediate_size", kIntermediateSize, top.intermediate_size) &&
           check("linear_num_value_heads", kLinearNumValueHeads,
                 top.linear_num_value_heads);
}

} // namespace lsxcommon