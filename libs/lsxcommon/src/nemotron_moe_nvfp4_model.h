#pragma once

// NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 model
// (kType "nemotron-3.5-lightning-30b-a3b-nvfp4"), the pre-quantized sibling of
// the BF16-distribution `NemotronHModel`.
//
// It shares the `NemotronModelBase` lifecycle/generation contour, the routed
// expert cache, the kernels and the tool codec (both distributions emit the
// same `logestix-nemotron-nvfp4` converted layout). What this leaf owns:
//   * its own frozen passport (`nemotron_moe_nvfp4_constants.h`);
//   * its own index loader, which additionally requires the index's `model`
//     field to name THIS distribution - both emit the same format tag, so
//     without that check the sibling's tree would load here and then verify
//     against the wrong FSI/CODE/DET goldens;
//   * an Infer that opens LsxFastMathScope with its own config.

#include "nemotron_moe_model.h"

namespace lsxcommon {

class NemotronNvfp4Model final : public NemotronModelBase {
public:
    static const std::string kType;  // "nemotron-3.5-lightning-30b-a3b-nvfp4"

    explicit NemotronNvfp4Model(ModelInitConfig cfg);
    ~NemotronNvfp4Model() override;

    gpu::InferenceResult Infer(const ModelRequest &req) override;

    const std::string &Type() const override { return kType; }

protected:
    bool LoadAndValidateIndex(std::string &error) override;
};

} // namespace lsxcommon
