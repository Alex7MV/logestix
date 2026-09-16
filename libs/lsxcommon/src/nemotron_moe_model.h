#pragma once

// Nemotron-3.5-Lightning-30B-A3B model (kType "nemotron-3.5-lightning-30b-a3b").
// Hybrid Mamba-2 + MoE + full-attention, standalone ModelBase subclass
// (GlmV5Model / QwenMoeModelBase pattern). The frozen geometry passport is
// compile-time locked (src/nemotron_moe_constants.h); Load cross-checks the
// converted index against it and fails on mismatch.
//
// Milestone: loader/model contour. Device kernels (Mamba-2 / attention / MoE)
// and the layer pipeline land in the same protocol sequence; Infer opens its
// own LsxFastMathScope (model-independence invariant) and is wired to the
// pipeline driver once it exists.

#include <string>

#include "lsxcommon/model_base.h"
#include "lsxcommon/nemotron_moe_inference.h"
#include "lsxcommon/nemotron_moe_load.h"

namespace lsxcommon {

class NemotronHModel final : public ModelBase {
public:
    static const std::string kType;  // "nemotron-3.5-lightning-30b-a3b"

    explicit NemotronHModel(ModelInitConfig cfg);
    ~NemotronHModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    bool BuildPromptInputIds(const ArrowTokenizerDb &tok,
                             std::string_view raw_prompt,
                             std::vector<int32_t> &out_ids) const override;

    const std::string &Type() const override { return kType; }

    // Test/diagnostics access to the parsed passport.
    const nemotron_moe::NemotronIndex &IndexForTest() const { return idx_; }

private:
    void TeardownWeights() override;
    bool ValidateGeometry(const nemotron_moe::NemotronIndex &idx) const;

    nemotron_moe::NemotronIndex idx_;
    nemotron_moe::NmWeights weights_;
    nemotron_moe::NmSession session_;
};

} // namespace lsxcommon
