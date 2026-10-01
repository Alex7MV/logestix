#include "alice_moe_model.h"

#include "lsxcommon/log.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/synthetic_geometry.h"

#include <absl/strings/str_format.h>
#include <utility>

namespace lsxcommon {

const std::string AliceFoundation80BModel::kType = "alice-foundation-80b-a3b";

AliceFoundation80BModel::AliceFoundation80BModel(ModelInitConfig cfg)
    : ModelBase(std::move(cfg))
{
}

AliceFoundation80BModel::~AliceFoundation80BModel()
{
    Unload();
}

bool AliceFoundation80BModel::Load()
{
    if (loaded_) return true;
    if (!BeginLoad(cfg_.model_path)) return AbortLoad();

    auto st = alice::LoadAliceTopology(cfg_.model_path, top_);
    if (!st.ok()) {
        log::error(absl::StrFormat("[alice] topology load failed: %s",
                                   st.ToString()));
        return AbortLoad();
    }
    if (!SyntheticGeometryAllowed() &&
        !alice::AliceTopologyMatchesFrozen(top_)) {
        log::error("[alice] frozen geometry mismatch");
        return AbortLoad();
    }

    // Weight runtime (pinned graph + expert cache) lands with the CUDA
    // pipeline; until then Load validates the topology and tokenizer only.
    FinishLoad();
    return true;
}

gpu::InferenceResult AliceFoundation80BModel::Infer(const ModelRequest &req)
{
    gpu::InferenceResult r;
    if (!ValidateInferRequest(req, r)) return r;

    // Model independence: this model's own kernel-routing policy for the
    // duration of the call.
    const lsx_nvfp4::LsxFastMathScope fm_scope(EffectiveFastMathMode(cfg_));

    r.ok = false;
    r.error_msg = "alice CUDA runtime not yet wired";
    return r;
}

gpu::InferenceResult AliceFoundation80BModel::SelfCheck(const ModelRequest &req)
{
    gpu::InferenceResult r = Infer(req);
    r.selfcheck_passed =
        r.ok && r.first_gen_tok_id >= 0 && !r.gen_tokens.empty();
    return r;
}

void AliceFoundation80BModel::TeardownWeights() {}

} // namespace lsxcommon
