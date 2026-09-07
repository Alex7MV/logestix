#pragma once

// Qwen3.6-35B-A3B MoE model (kType "qwen3.6-35b-a3b"). Geometry-identical
// sibling of Qwen35MoeModel: shares the qwen_moe runtime graph
// (QwenMoeModelBase in qwen_moe_model.h) but owns its frozen-geometry
// passport (src/qwen_moe_36_constants.h) and its own Infer fast-math scope
// (model independence invariant).

#include "qwen_moe_model.h"

namespace lsxcommon {

class Qwen36MoeModel final : public QwenMoeModelBase {
public:
    static const std::string kType;  // "qwen3.6-35b-a3b"

    explicit Qwen36MoeModel(ModelInitConfig cfg);
    ~Qwen36MoeModel() override;

    gpu::InferenceResult Infer(const ModelRequest &req) override;

    const std::string &Type() const override { return kType; }

private:
    bool ValidateGeometry(const qwen_moe::QwenMoeTopology &top) const override;
};

} // namespace lsxcommon
