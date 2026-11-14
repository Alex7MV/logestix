#include "inkling_moe_model.h"

#include "lsxcommon/inkling_moe_inference_pipeline.h"
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

    auto st = inkling::LoadInklingTopology(cfg_.model_path, top_);
    if (!st.ok()) {
        log::error(absl::StrFormat("[inkling] topology load failed: %s",
                                   st.ToString()));
        return AbortLoad();
    }
    if (!inkling::InklingTopologyMatchesFrozen(top_)) {
        log::error("[inkling] frozen geometry mismatch");
        return AbortLoad();
    }
    if (!EnsureContext()) {
        log::error("[inkling] cuda context init failed");
        return AbortLoad();
    }
    // The weights, the expert store and the KV session are all in place, but the
    // per-layer forward pass (stage 5) is not written yet. Refuse the load: a
    // model that loads and then cannot decode is worse than one that says so,
    // because `lsxfabric --serve` would advertise a session it can never
    // answer. `InklingMoeModel::Infer` reports the same error.
    log::error(
        "[inkling] the decode pipeline is not implemented yet (stage 5); "
        "see docs/open-tasks/2026-11-14-inkling-2-small-ingest.md");
    return AbortLoad();
}

gpu::InferenceResult InklingMoeModel::Infer(const ModelRequest &req)
{
    gpu::InferenceResult r;
    if (!ValidateInferRequest(req, r)) return r;
    if (!weights_loaded_) {
        r.error_msg = "inkling-2-small weights not loaded";
        return r;
    }

    // Model independence: this model's own kernel-routing policy.
    const lsx_nvfp4::LsxFastMathScope fm_scope(EffectiveFastMathMode(cfg_));

    if (!EnsureContext()) {
        r.error_msg = "cuda init failed";
        return r;
    }
    const int eos_id =
        (req.ignore_eos || !tok_db_) ? kIgnoreEosTokenId : tok_db_->eos_id;

    // Streaming + cooperative cancellation: the whole decode runs inside the
    // residency lock, and the generation stream carries the per-token callback
    // and the abort flag (enforced by test_model_independence_guard).
    lsxcommon::GenerationStream gs;
    gs.on_token = req.token_callback ? &req.token_callback : nullptr;
    gs.abort_flag = req.abort_flag;

    sample::ISampler &sampler = sample::SamplerFor(req.sampling);
    inkling::InklingGenerationResult gen;
    const auto ec = this->RunUnderResidencyLock([&]() {
        return inkling::RunInklingGeneration(
            stream(), pw_, *experts_, session_, scratch_, req.input_ids,
            req.max_gen_tokens, std::vector<int32_t>{eos_id},
            static_cast<float>(top_.final_logit_softcap), top_.unpadded_vocab,
            sampler, gen, &gs);
    });
    if (ec != cudaSuccess || !gen.ok) {
        r.error_msg = gen.error_msg.empty() ? "inkling-2-small cuda error"
                                            : gen.error_msg;
        return r;
    }
    r.ok = true;
    r.num_layers = top_.num_layers;
    r.gen_tokens = gen.tokens;
    r.num_tokens = static_cast<int>(gen.tokens.size());
    r.first_gen_tok_id = gen.first_token;
    r.weight_resident_bytes = resident_bytes_;
    r.final_logits = gen.final_logits;
    r.eos_terminated = gen.eos_terminated;
    r.ttft_ms = gen.ttft_ms;
    r.total_time_ms = gen.total_ms;
    if (gen.total_ms > 0.0)
        r.tokens_per_sec =
            static_cast<double>(gen.tokens.size()) / (gen.total_ms / 1000.0);
    if (tok_db_ && !gen.tokens.empty()) {
        std::size_t cap = gen.tokens.size() * 16 + 64;
        std::string text(cap, '\0');
        if (arrow_tokenizer_decode(*tok_db_, gen.tokens.data(),
                                   gen.tokens.size(), text.data(), &cap) == 0)
            r.output_text.assign(text.data(), cap);
    }
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
    scratch_.Free();
    session_.Free();
    if (experts_) {
        inkling::FreeInklingExpertIndex(*experts_);
        experts_.reset();
    }
    if (weights_loaded_) {
        inkling::FreeInklingPinnedWeights(pw_);
        weights_loaded_ = false;
    }
    TeardownContext();
}

}  // namespace lsxcommon
