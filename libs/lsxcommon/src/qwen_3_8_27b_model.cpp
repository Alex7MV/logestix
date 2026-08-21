#include "qwen_3_8_27b_model.h"
#include <spdlog/spdlog.h>

namespace lsxcommon {

const std::string Qwen3_8_27BModel::kType = "qwen3.8-27b";

Qwen3_8_27BModel::Qwen3_8_27BModel(ModelInitConfig cfg)
    : QwenBaseModel(std::move(cfg))
{
    type_ = kType;
}

bool Qwen3_8_27BModel::ValidateGeometry(const qwen::QwenTopology &top) const
{
    auto check = [&](const char *name, int expected, int actual) {
        if (expected != actual) {
            spdlog::error("[qwen3.8-27b] geometry mismatch: {} expected={} got={}",
                          name, expected, actual);
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