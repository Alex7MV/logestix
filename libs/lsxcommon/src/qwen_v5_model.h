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

    explicit Qwen3_5_9BModel(ModelInitConfig cfg);

    const std::string &Type() const override { return kType; }
    static const std::string kType;

protected:
    void ValidateGeometry(const qwen::QwenTopology &top) const override;
};

} // namespace lsxcommon