#include "alice_moe_model.h"

#include "lsxcommon/alice_tool_format.h"
#include "lsxcommon/model_runtime.h"

#include "lsxcommon/log.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/sampling.h"
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
    tokenizer_optional_ = SyntheticGeometryAllowed();
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
    tokenizer_optional_ = SyntheticGeometryAllowed();

    if (!EnsureContext()) {
        log::error("[alice] cuda context init failed");
        return AbortLoad();
    }
    cudaStreamSynchronize(stream());
    if (!alice::LoadAlicePinnedWeights(cfg_.model_path, pw_, stream())) {
        log::error("[alice] pinned weight load failed");
        return AbortLoad();
    }
    cudaStreamSynchronize(stream());
    experts_ = std::make_unique<alice::AliceExpertDb>(
        alice::load_alice_expert_index(cfg_.model_path, top_.num_layers,
                                       top_.num_experts));
    weights_loaded_ = true;
    resident_bytes_ = pw_.resident_bytes;
    // Governor accounting (lifecycle-unification II): the pinned dense set is
    // the permanent lease (released by the canonical Unload/AbortLoad). The
    // routed-expert store is a host mmap and streams through the per-call
    // device cache, so it carries no permanent lease.
    if (!AcquirePermanentWeights(resident_bytes_)) return AbortLoad();
    FinishLoad();
    return true;
}

gpu::InferenceResult AliceFoundation80BModel::Infer(const ModelRequest &req)
{
    gpu::InferenceResult r;
    if (!ValidateInferRequest(req, r)) return r;
    if (!weights_loaded_) {
        r.error_msg = "alice weights not loaded";
        return r;
    }

    // Model independence: this model's own kernel-routing policy for the
    // duration of the call.
    const lsx_nvfp4::LsxFastMathScope fm_scope(EffectiveFastMathMode(cfg_));

    if (!EnsureContext()) {
        r.error_msg = "cuda init failed";
        return r;
    }
    const int eos_id = (req.ignore_eos || !tok_db_)
                           ? kIgnoreEosTokenId
                           : tok_db_->eos_id;
    // Streaming + cooperative cancellation: lsxfabric --serve and TaskMosaic
    // publish tokens exclusively through req.token_callback.
    lsxcommon::GenerationStream gs;
    gs.on_token = req.token_callback ? &req.token_callback : nullptr;
    gs.abort_flag = req.abort_flag;

    alice::AliceGenerationResult gen;
    auto ec = RunUnderResidencyLock([&]() {
        return alice::RunAliceGeneration(stream(), pw_, *experts_, req.input_ids,
                                         req.max_gen_tokens, eos_id, gen,
                                         sample::SamplerFor(req.sampling), &gs);
    });
    if (ec != cudaSuccess || !gen.ok) {
        r.error_msg = gen.error_msg.empty() ? "alice cuda error" : gen.error_msg;
        return r;
    }
    r.ok = true;
    r.num_layers = pw_.top.num_layers;
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

gpu::InferenceResult AliceFoundation80BModel::SelfCheck(const ModelRequest &req)
{
    // Base model: it answers the control prompt with a short MCQ-style prefix,
    // so accept "Paris" anywhere in a 32-token window (substring contract).
    runtime::ParisSelfCheckOptions opts;
    opts.frame_prompt = true;
    opts.min_tokens = 32;
    opts.answer_window = 32;
    opts.substring_match = true;
    return runtime::RunParisSelfCheck(*this, req, opts);
}

bool AliceFoundation80BModel::BuildConversationInputIds(
    const ArrowTokenizerDb &tok, const ChatConversation &conv,
    std::vector<int32_t> &out_ids) const
{
    return alice::RenderConversation(tok, conv, /*add_generation_prompt=*/true,
                                     out_ids);
}

bool AliceFoundation80BModel::ParseToolCalls(
    const ArrowTokenizerDb &tok, const std::vector<int32_t> &gen_tokens,
    std::vector<ToolCall> &out_calls) const
{
    return alice::ParseToolCallsAlice(tok, gen_tokens, out_calls);
}

void AliceFoundation80BModel::TeardownWeights()
{
    if (experts_) {
        alice::free_alice_expert_index(*experts_);
        experts_.reset();
    }
    if (weights_loaded_) {
        alice::FreeAlicePinnedWeights(pw_);
        weights_loaded_ = false;
    }
    TeardownContext();
}

} // namespace lsxcommon
