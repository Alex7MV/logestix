#include "inkling_moe_model.h"

#include "lsxcommon/inkling_moe_load.h"
#include "lsxcommon/log.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/model_runtime.h"

#include <absl/strings/str_format.h>

#include <utility>

namespace lsxcommon {

const std::string InklingMoeModel::kType = "inkling-2-small";

InklingMoeModel::InklingMoeModel(ModelInitConfig cfg)
    : ModelBase(std::move(cfg))
{
}

InklingMoeModel::~InklingMoeModel() { Unload(); }

bool InklingMoeModel::Load()
{
    if (loaded_) return true;
    tokenizer_optional_ = SyntheticGeometryAllowed();
    if (!BeginLoad(cfg_.model_path)) return AbortLoad();

    inkling::InklingTopology top;
    auto st = inkling::LoadInklingTopology(cfg_.model_path, top);
    if (!st.ok()) {
        log::error(absl::StrFormat("[inkling] topology load failed: %s",
                                   st.ToString()));
        return AbortLoad();
    }
    if (!inkling::InklingTopologyMatchesFrozen(top)) {
        log::error("[inkling] frozen geometry mismatch");
        return AbortLoad();
    }
    if (!EnsureContext()) {
        log::error("[inkling] cuda context init failed");
        return AbortLoad();
    }
    // The decode pipeline (stage 5) is not wired yet: refuse the load rather
    // than hand a half-loaded model to the caller.
    log::error(
        "[inkling] the decode pipeline is not implemented in this tree yet");
    return AbortLoad();
}

gpu::InferenceResult InklingMoeModel::Infer(const ModelRequest &req)
{
    gpu::InferenceResult r;
    if (!ValidateInferRequest(req, r)) return r;
    r.error_msg = "inkling-2-small decode pipeline not implemented";
    return r;
}

gpu::InferenceResult InklingMoeModel::SelfCheck(const ModelRequest &req)
{
    runtime::ParisSelfCheckOptions opts;
    opts.frame_prompt = false;
    opts.min_tokens = 2;
    opts.answer_window = 8;
    return runtime::RunParisSelfCheck(*this, req, opts);
}

void InklingMoeModel::TeardownWeights()
{
    if (weights_loaded_) weights_loaded_ = false;
    TeardownContext();
}

}  // namespace lsxcommon
