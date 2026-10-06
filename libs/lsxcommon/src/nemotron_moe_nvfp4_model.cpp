#include "nemotron_moe_nvfp4_model.h"

#include "nemotron_moe_nvfp4_constants.h"
#include "lsxcommon/log.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/nemotron_moe_nvfp4_load.h"

#include <absl/strings/str_format.h>

namespace lsxcommon {

const std::string NemotronNvfp4Model::kType = nemotron_nvfp4::frozen::kType;

NemotronNvfp4Model::NemotronNvfp4Model(ModelInitConfig cfg)
    : NemotronModelBase(std::move(cfg))
{
}

NemotronNvfp4Model::~NemotronNvfp4Model() { Unload(); }

gpu::InferenceResult NemotronNvfp4Model::Infer(const ModelRequest &req)
{
    // Model independence: this model's own kernel-routing policy for the
    // duration of the call (see LsxFastMathScope in lsx_inference_nvfp4.h).
    lsx_nvfp4::LsxFastMathScope scope(EffectiveFastMathMode(cfg_));
    return RunInferShared(req);
}

bool NemotronNvfp4Model::LoadAndValidateIndex(std::string &error)
{
    if (!nemotron_nvfp4::LoadNemotronNvfp4Index(cfg_.model_path, idx_, error))
        return false;
    if (!nemotron_nvfp4::ValidateNemotronNvfp4Geometry(idx_, error)) {
        log::error(absl::StrFormat(
            "[nemotron-nvfp4] frozen geometry mismatch: %s", error));
        return false;
    }
    return true;
}

} // namespace lsxcommon
