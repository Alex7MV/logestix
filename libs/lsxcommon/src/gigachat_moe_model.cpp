#include "lsxcommon/gigachat_moe_model.h"

#include "gigachat_moe_constants.h"

#include "lsxcommon/log.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/model_runtime.h"

#include <absl/strings/str_format.h>

namespace lsxcommon {

const std::string GigaChat35Model::kType = gigachat::frozen::kType;

GigaChat35Model::GigaChat35Model(ModelInitConfig cfg)
    : ModelBase(std::move(cfg))
{
    // Synthetic fixtures may omit the tokenizer arrows.
    tokenizer_optional_ = true;
}

GigaChat35Model::~GigaChat35Model() { Unload(); }

void GigaChat35Model::TeardownWeights() {}

bool GigaChat35Model::Load()
{
    if (loaded_) return true;
    if (!BeginLoad(cfg_.model_path)) return AbortLoad();

    std::string err;
    if (!gigachat::LoadGigaChatIndex(cfg_.model_path, idx_, err))
    {
        log::error(absl::StrFormat("[gigachat] index error: %s", err));
        return AbortLoad();
    }
    if (!gigachat::ValidateGigaChatGeometry(idx_, err))
    {
        log::error(absl::StrFormat("[gigachat] frozen geometry mismatch: %s",
                                   err));
        return AbortLoad();
    }
    vram_ = idx_.vram;

    // Device weights, expert streaming and the decode pipeline land in Plan 2.
    FinishLoad();
    return true;
}

gpu::InferenceResult GigaChat35Model::Infer(const ModelRequest &req)
{
    // Model-independence invariant: each model asserts its OWN fast-math
    // policy per call (the routing toggles are process-global).
    lsx_nvfp4::LsxFastMathScope scope(EffectiveFastMathMode(cfg_));
    (void)req;
    gpu::InferenceResult r;
    r.ok = false;
    r.error_msg =
        "gigachat inference pipeline not implemented yet (Plan 2)";
    return r;
}

gpu::InferenceResult GigaChat35Model::SelfCheck(const ModelRequest &req)
{
    return runtime::RunParisSelfCheck(*this, req);
}

} // namespace lsxcommon
