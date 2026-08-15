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

    explicit Qwen3_8_27BModel(ModelInitConfig cfg);

    const std::string &Type() const override { return kType; }
    static const std::string kType;

protected:
    void ValidateGeometry(const qwen::QwenTopology &top) const override;
};

} // namespace lsxcommon