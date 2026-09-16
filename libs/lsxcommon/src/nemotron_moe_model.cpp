#include "nemotron_moe_model.h"
#include "nemotron_moe_constants.h"

#include "lsxcommon/log.h"
#include "lsxcommon/lsx_inference_nvfp4.h"

#include <absl/strings/str_format.h>

namespace lsxcommon {

const std::string NemotronHModel::kType = nemotron_moe::frozen::kType;

NemotronHModel::NemotronHModel(ModelInitConfig cfg) : ModelBase(std::move(cfg))
{
    // Synthetic fixtures may omit the tokenizer arrows.
    tokenizer_optional_ = true;
}

NemotronHModel::~NemotronHModel() { Unload(); }

bool NemotronHModel::ValidateGeometry(
    const nemotron_moe::NemotronIndex &idx) const
{
    std::string why;
    if (!nemotron_moe::ValidateNemotronGeometry(idx, why)) {
        log::error(absl::StrFormat(
            "[nemotron] frozen geometry mismatch: %s", why));
        return false;
    }
    return true;
}

bool NemotronHModel::Load()
{
    if (loaded_) return true;
    if (!BeginLoad(cfg_.model_path)) return AbortLoad();

    std::string err;
    if (!nemotron_moe::LoadNemotronIndex(cfg_.model_path, idx_, err)) {
        log::error(absl::StrFormat("[nemotron] index error: %s", err));
        return AbortLoad();
    }
    if (!ValidateGeometry(idx_)) return AbortLoad();
    vram_ = idx_.vram;

    if (!EnsureContext()) return AbortLoad();
    FinishLoad();
    log::info(absl::StrFormat(
        "[nemotron] loaded %s: %d layers, %d experts, vocab %d",
        idx_.model.c_str(), idx_.num_layers, idx_.num_experts, idx_.vocab_size));
    return true;
}

void NemotronHModel::TeardownWeights()
{
    // Device weight graph is torn down here once the kernel pipeline lands.
}

gpu::InferenceResult NemotronHModel::Infer(const ModelRequest &req)
{
    lsx_nvfp4::LsxFastMathScope scope(EffectiveFastMathMode(cfg_));
    return RunUnderResidencyLock([&] {
        gpu::InferenceResult r;
        if (!ValidateInferRequest(req, r)) return r;
        r.error_msg = "nemotron inference pipeline not wired yet";
        return r;
    });
}

gpu::InferenceResult NemotronHModel::SelfCheck(const ModelRequest &req)
{
    gpu::InferenceResult r = Infer(req);
    r.selfcheck_passed = false;
    return r;
}

} // namespace lsxcommon
