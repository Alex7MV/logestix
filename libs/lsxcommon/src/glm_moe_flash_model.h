#pragma once

// GLM-5.3-Flash model (kType "glm-5.3-flash"). Standalone ModelBase
// subclass (Kimi/Qwen38 pattern). Frozen geometry passport in
// src/glm_moe_flash_constants.h; loader validates at load.
#include <memory>
#include <vector>

#include "lsxcommon/model_base.h"
#include "lsxcommon/qwen_inference.h"
#include "lsxcommon/glm_moe_flash_inference.h"
#include "lsxcommon/glm_moe_flash_load.h"

namespace lsxcommon {

class Glm5NextFlashModel final : public ModelBase {
public:
    static const std::string kType;  // "glm-5.3-flash"

    explicit Glm5NextFlashModel(ModelInitConfig cfg);
    ~Glm5NextFlashModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    const std::string &Type() const override { return kType; }

private:
    void TeardownWeights() override;

    glm_moe_flash::FlashPinnedWeights pw_;
    qwen_moe::QwenMoeExpertStore experts_;
    gpu::ExpertDeviceCache cache_{};
};

} // namespace lsxcommon
