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

    {
        std::string err;
        if (!nemotron_moe::LoadNemotronWeights(cfg_.model_path, idx_, weights_,
                                               stream(), err)) {
            log::error(absl::StrFormat("[nemotron] weight load error: %s", err));
            weights_.Free();
            return AbortLoad();
        }
        const int kMaxSeq = 4096;
        if (!nemotron_moe::AllocNemotronSession(idx_, kMaxSeq, session_, err)) {
            log::error(absl::StrFormat("[nemotron] session alloc error: %s",
                                       err));
            weights_.Free();
            return AbortLoad();
        }
        resident_bytes_ = weights_.resident_bytes;
    }

    FinishLoad();
    log::info(absl::StrFormat(
        "[nemotron] loaded %s: %d layers, %d experts, vocab %d",
        idx_.model.c_str(), idx_.num_layers, idx_.num_experts, idx_.vocab_size));
    return true;
}

void NemotronHModel::TeardownWeights()
{
    session_.Free();
    weights_.Free();
}

namespace {

int ArgMaxHost(const std::vector<float> &v)
{
    int best = 0;
    for (size_t i = 1; i < v.size(); ++i)
        if (v[i] > v[best]) best = static_cast<int>(i);
    return best;
}

} // namespace

gpu::InferenceResult NemotronHModel::Infer(const ModelRequest &req)
{
    lsx_nvfp4::LsxFastMathScope scope(EffectiveFastMathMode(cfg_));
    return RunUnderResidencyLock([&] {
        gpu::InferenceResult r;
        if (!ValidateInferRequest(req, r)) return r;

        nemotron_moe::ResetNemotronSession(session_, stream());
        std::vector<float> logits(static_cast<size_t>(idx_.vocab_size));

        int pos = 0;
        for (int32_t id : req.input_ids) {
            cudaError_t e = nemotron_moe::NemotronForwardToken(
                idx_, weights_, session_, id, pos, stream());
            if (e != cudaSuccess) {
                r.error_msg = std::string("forward failed: ") +
                              cudaGetErrorString(e);
                return r;
            }
            ++pos;
        }

        const int eos = idx_.eos_token_id;
        int generated = 0;
        for (int step = 0; step < req.max_gen_tokens; ++step) {
            if (cudaMemcpyAsync(logits.data(), session_.head_logits,
                                logits.size() * sizeof(float),
                                cudaMemcpyDeviceToHost, stream()) !=
                    cudaSuccess ||
                cudaStreamSynchronize(stream()) != cudaSuccess) {
                r.error_msg = "logits readback failed";
                return r;
            }
            const int next = ArgMaxHost(logits);
            r.gen_tokens.push_back(next);
            ++generated;
            if (step == 0) {
                r.first_gen_tok_id = next;
                r.first_gen_maxv = logits[static_cast<size_t>(next)];
            }
            // Keep the last step's logits row (protocol: determinism check).
            r.final_logits = logits;
            if (!req.ignore_eos && next == eos) {
                r.eos_terminated = true;
                break;
            }
            if (step + 1 < req.max_gen_tokens) {
                cudaError_t e = nemotron_moe::NemotronForwardToken(
                    idx_, weights_, session_, next, pos, stream());
                if (e != cudaSuccess) {
                    r.error_msg = std::string("forward failed: ") +
                                  cudaGetErrorString(e);
                    return r;
                }
                ++pos;
            }
            if (req.token_callback) {
                TokenChunk c;
                c.token_id = static_cast<uint32_t>(next);
                c.step_t = static_cast<uint32_t>(step);
                c.logit_maxv = r.first_gen_maxv;
                req.token_callback(c);
            }
            if (req.abort_flag && req.abort_flag->load()) break;
        }
        r.num_tokens = generated;
        r.num_layers = idx_.num_layers;
        r.weight_resident_bytes = resident_bytes_;
        r.ok = true;
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
