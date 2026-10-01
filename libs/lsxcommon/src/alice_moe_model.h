#pragma once

#include <memory>
#include <string>
#include <vector>

#include "lsxcommon/alice_moe_load.h"
#include "lsxcommon/model_base.h"

namespace lsxcommon {

// AliceAI-Foundation-80B-A3B-Base (kType "alice-foundation-80b-a3b"): hybrid
// KDA + gated-attention sparse MoE with block attention-residual mixing.
// Frozen geometry lives in src/alice_moe_constants.h (passport,
// model-independence invariant); Load cross-checks the converted index.
class AliceFoundation80BModel final : public ModelBase {
public:
    static const std::string kType;  // "alice-foundation-80b-a3b"

    explicit AliceFoundation80BModel(ModelInitConfig cfg);
    ~AliceFoundation80BModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    const std::string &Type() const override { return kType; }

protected:
    void TeardownWeights() override;

    alice::AliceMoeTopology top_;
};

} // namespace lsxcommon
