#pragma once

// GigaChat3.5-432B-A28B-Reasoning model (kType
// "gigachat3.5-432b-a28b-reasoning"). Hybrid MLA + GatedDeltaNet + MoE,
// standalone ModelBase subclass. The frozen geometry passport is compile-time
// locked (src/gigachat_moe_constants.h); Load cross-checks the converted index
// against it and fails on mismatch.
//
// Plan 1 milestone: loader/model contour. CUDA inference kernels and the
// decode pipeline land in Plan 2; Infer opens its own LsxFastMathScope
// (model-independence invariant) and returns a not-implemented sentinel until
// then.

#include <string>

#include "lsxcommon/gigachat_moe_load.h"
#include "lsxcommon/model_base.h"

namespace lsxcommon {

class GigaChat35Model final : public ModelBase {
public:
    static const std::string kType;  // "gigachat3.5-432b-a28b-reasoning"

    explicit GigaChat35Model(ModelInitConfig cfg);
    ~GigaChat35Model() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    const std::string &Type() const override { return kType; }

    // Test/diagnostics access to the parsed passport.
    const gigachat::GigaChatIndex &IndexForTest() const { return idx_; }

private:
    void TeardownWeights() override;

    gigachat::GigaChatIndex idx_;
};

} // namespace lsxcommon
