#pragma once

#include "lsxcommon/qwen_base_model.h"

namespace lsxcommon {

class Qwen3_8_27BModel final : public QwenBaseModel {
public:
    static constexpr int kHiddenSize = 5120;
    static constexpr int kNumLayers = 64;
    static constexpr int kNumHeads = 24;
    static constexpr int kNumKVHeads = 4;
    static constexpr int kHeadDim = 256;
    static constexpr int kIntermediateSize = 17408;
    static constexpr int kLinearNumValueHeads = 48;

    // Compile-time geometry invariants (Model Ingestion Protocol step 3).
    // NOTE: unlike the 9B, the 27B topology does not satisfy
    // hidden == heads * head_dim; only relational invariants are asserted.
    static_assert(kNumKVHeads > 0 && kNumKVHeads <= kNumHeads,
                  "num_key_value_heads must be in (0, num_attention_heads]");
    static_assert(kNumLayers > 0, "num_layers must be positive");

    explicit Qwen3_8_27BModel(ModelInitConfig cfg);

    const std::string &Type() const override { return kType; }
    static const std::string kType;

protected:
    bool ValidateGeometry(const qwen::QwenTopology &top) const override;
};

} // namespace lsxcommon