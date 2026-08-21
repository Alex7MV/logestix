#pragma once

#include "lsxcommon/qwen_base_model.h"

namespace lsxcommon {

class Qwen3_5_9BModel final : public QwenBaseModel {
public:
    static constexpr int kHiddenSize = 4096;
    static constexpr int kNumLayers = 32;
    static constexpr int kNumHeads = 16;
    static constexpr int kNumKVHeads = 4;
    static constexpr int kHeadDim = 256;
    static constexpr int kIntermediateSize = 12288;
    static constexpr int kLinearNumValueHeads = 32;

    // Compile-time geometry invariants (Model Ingestion Protocol step 3).
    static_assert(kHiddenSize == kNumHeads * kHeadDim,
                  "hidden_size must equal num_attention_heads * head_dim");
    static_assert(kNumKVHeads > 0 && kNumKVHeads <= kNumHeads,
                  "num_key_value_heads must be in (0, num_attention_heads]");
    static_assert(kNumLayers > 0, "num_layers must be positive");

    explicit Qwen3_5_9BModel(ModelInitConfig cfg);

    const std::string &Type() const override { return kType; }
    static const std::string kType;

protected:
    bool ValidateGeometry(const qwen::QwenTopology &top) const override;
};

} // namespace lsxcommon