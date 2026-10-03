// Xing4.0-29B-A4B model class (Model Ingestion Protocol stages 3 + 8).
//
// Owns the whole lifecycle for this family: load (all experts resident),
// geometry gate against the frozen passport, per-request decode session under
// the residency lock, streaming/abort through GenerationStream, Paris
// SelfCheck, and the native <_system>/<_user>/<_bot> tool bridge.

#include "xing_moe_model.h"

#include "xing_moe_constants.h"
#include "lsxcommon/xing_moe_tool_format.h"

#include "lsxcommon/log.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/qwen_inference.h"   // DecideSession (shared, model-neutral)
#include "lsxcommon/sampling.h"
#include "lsxcommon/synthetic_geometry.h"

#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>

#include <chrono>
#include <cmath>

namespace lsxcommon {

const std::string XingMoeModel::kType = "xing4.0-29b-a4b";

namespace {

using namespace lsxcommon::xing::frozen;

int MsSince(std::chrono::steady_clock::time_point t0)
{
    return static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0)
            .count());
}

}  // namespace

XingMoeModel::XingMoeModel(ModelInitConfig cfg) : ModelBase(std::move(cfg))
{
}

void XingMoeModel::TeardownWeights()
{
    session_.reset();
    if (d_rc_) cudaFree(d_rc_);
    if (d_rs_) cudaFree(d_rs_);
    d_rc_ = nullptr;
    d_rs_ = nullptr;
    rc_.clear();
    rs_.clear();
    host_logits_.clear();
    yarn_capacity_ = 0;
    session_tail_.clear();
    xing::FreeXingWeights(pw_);
    TeardownContext();
}

bool XingMoeModel::ValidateGeometry(const xing::XingTopology &top) const
{
    if (SyntheticGeometryAllowed()) return true;
    const bool ok = top.num_layers == kLayers &&
                    top.experts_per_layer == kNumRouted &&
                    top.hc_mult == kHcMult &&
                    top.moe_layers.size() == kLayers - kFirstKDense &&
                    top.dense_layers.size() == kFirstKDense &&
                    top.mtp_layers.size() == kMtpLayers.size();
    if (!ok)
    {
        log::error(absl::StrCat(
            "[xing-moe-model] frozen geometry mismatch: layers=",
            top.num_layers, " experts=", top.experts_per_layer,
            " hc_mult=", top.hc_mult, " moe=", top.moe_layers.size(),
            " dense=", top.dense_layers.size()));
        return false;
    }
    for (size_t i = 0; i < top.moe_layers.size(); ++i)
        if (top.moe_layers[i] != kFirstKDense + static_cast<long long>(i))
        {
            log::error("[xing-moe-model] MoE layer list is not the frozen prefix");
            return false;
        }
    return true;
}

bool XingMoeModel::Load()
{
    Unload();  // idempotent re-load support
    if (!BeginLoad(cfg_.model_path)) return false;
    if (!EnsureContext())
    {
        log::error("[xing-moe-model] stream init failed");
        return AbortLoad();
    }

    std::string err;
    const auto t0 = std::chrono::steady_clock::now();
    const cudaError_t e =
        xing::LoadXingWeights(cfg_.model_path, pw_, stream(), err);
    if (e != cudaSuccess)
    {
        log::error("[xing-moe-model] load failed: " + err);
        return AbortLoad();
    }
    log::info(absl::StrCat("[xing-moe-model] weights loaded in ", MsSince(t0),
                           " ms"));

    const xing::XingTopology top =
        xing::xing_load_topology((std::filesystem::path(cfg_.model_path) /
                                  "logestix" / "model.index.json")
                                     .string());
    if (!ValidateGeometry(top)) return AbortLoad();

    dims_ = xing::XingDimsFromWeights(pw_);

    // Everything is permanently resident (this family has no expert cache), so
    // the permanent-weights lease covers the routed experts too.
    if (!AcquirePermanentWeights(pw_.resident_bytes)) return AbortLoad();
    log::info(absl::StrCat("[xing-moe-model] resident ", pw_.resident_bytes,
                           " B, layers=", dims_.layers, " experts=",
                           dims_.experts));
    FinishLoad();
    return true;
}

gpu::InferenceResult XingMoeModel::Infer(const ModelRequest &req)
{
    gpu::InferenceResult r;
    if (!ValidateInferRequest(req, r)) return r;
    if (!loaded_)
    {
        r.error_msg = "model not loaded";
        return r;
    }
    if (!ContextReady())
    {
        r.error_msg = "execution context missing";
        return r;
    }
    if (req.input_ids.empty())
    {
        r.error_msg = "empty input";
        return r;
    }

    const int eos_id =
        (req.ignore_eos || !tok_db_) ? kIgnoreEosTokenId : tok_db_->eos_id;
    const int capacity =
        static_cast<int>(req.input_ids.size()) +
        std::max(1, req.max_gen_tokens);

    std::vector<int32_t> tokens;
    std::string gen_error;
    bool eos_terminated = false;
    double ttft_ms = 0.0, total_ms = 0.0;
    int kv_slots = 0;
    const auto t_start = std::chrono::steady_clock::now();

    lsxcommon::GenerationStream gs;
    gs.on_token = req.token_callback ? &req.token_callback : nullptr;
    gs.abort_flag = req.abort_flag;

    const cudaError_t run = RunUnderResidencyLock([&] {
        // Session state is created, decoded through and published INSIDE the
        // lock, so a concurrent Infer cannot free the cache this one reads.
        const qwen::SessionDecision session = qwen::DecideSession(
            session_ && session_->length > 0,
            session_ ? session_->token_id : -1, session_tail_, req.input_ids);
        if (!session.is_continuation)
        {
            session_.reset();
            session_tail_.clear();
            if (yarn_capacity_ < capacity)
            {
                if (!xing::BuildXingYarnTables(capacity, rc_, rs_))
                    return cudaErrorInvalidValue;
                if (d_rc_) cudaFree(d_rc_);
                if (d_rs_) cudaFree(d_rs_);
                d_rc_ = nullptr;
                d_rs_ = nullptr;
                if (cudaMalloc(reinterpret_cast<void **>(&d_rc_),
                               rc_.size() * sizeof(float)) != cudaSuccess ||
                    cudaMalloc(reinterpret_cast<void **>(&d_rs_),
                               rs_.size() * sizeof(float)) != cudaSuccess)
                {
                    return cudaErrorMemoryAllocation;
                }
                const cudaError_t up = cudaMemcpy(
                    d_rc_, rc_.data(), rc_.size() * sizeof(float),
                    cudaMemcpyHostToDevice);
                if (up != cudaSuccess) return up;
                const cudaError_t us = cudaMemcpy(
                    d_rs_, rs_.data(), rs_.size() * sizeof(float),
                    cudaMemcpyHostToDevice);
                if (us != cudaSuccess) return us;
                yarn_capacity_ = capacity;
            }
            session_ = std::make_unique<xing::XingSession>();
            const cudaError_t ae = xing::AllocateXingSession(
                *session_, dims_, std::max(4, yarn_capacity_), stream());
            if (ae != cudaSuccess) return ae;
        }

        // Prefill: one token per step. Xing has no incremental prefill path
        // yet, so a prompt is replayed token by token — correct-first.
        std::vector<int32_t> feed = session.is_continuation
                                        ? session.gen_input
                                        : req.input_ids;
        int pos = session.is_continuation ? session_->length : 0;
        float maxv = 0.0f;
        for (size_t i = 0; i < feed.size(); ++i)
        {
            if (gs.Aborted()) break;
            const cudaError_t se = xing::XingDecodeStep(
                pw_, dims_, *session_, feed[i], d_rc_, d_rs_, pos, stream());
            if (se != cudaSuccess) return se;
            ++pos;
            if (cudaStreamSynchronize(stream()) != cudaSuccess)
                return cudaErrorUnknown;
        }
        if (gs.Aborted())
        {
            session_tail_ = feed;
            return cudaSuccess;
        }

        auto &sampler = sample::DefaultSampler();
        for (int step = 0; step < std::max(0, req.max_gen_tokens); ++step)
        {
            if (gs.Aborted()) break;
            host_logits_.assign(dims_.vocab, 0.0f);
            xing::XingReadLogits(*session_, host_logits_.data(), stream());
            const cudaError_t ce = cudaStreamSynchronize(stream());
            if (ce != cudaSuccess) return ce;
            int32_t d_token = 0;
            float d_maxv = 0.0f;
            int32_t *d_tok = nullptr;
            float *d_mv = nullptr;
            if (cudaMalloc(reinterpret_cast<void **>(&d_tok),
                           sizeof(int32_t)) != cudaSuccess ||
                cudaMalloc(reinterpret_cast<void **>(&d_mv),
                           sizeof(float)) != cudaSuccess)
                return cudaErrorMemoryAllocation;
            float *d_logits_dev = nullptr;
            if (cudaMalloc(reinterpret_cast<void **>(&d_logits_dev),
                           static_cast<size_t>(dims_.vocab) * sizeof(float)) !=
                cudaSuccess)
            {
                cudaFree(d_tok);
                cudaFree(d_mv);
                return cudaErrorMemoryAllocation;
            }
            cudaMemcpy(d_logits_dev, host_logits_.data(),
                       static_cast<size_t>(dims_.vocab) * sizeof(float),
                       cudaMemcpyHostToDevice);
            const cudaError_t sce = sampler.Sample(
                stream(), d_logits_dev, dims_.vocab, sample::LogitType::kF32,
                d_tok, d_mv);
            int32_t next = 0;
            if (sce == cudaSuccess)
            {
                cudaMemcpy(&next, d_tok, sizeof(int32_t), cudaMemcpyDeviceToHost);
                cudaMemcpy(&maxv, d_mv, sizeof(float), cudaMemcpyDeviceToHost);
            }
            cudaFree(d_tok);
            cudaFree(d_mv);
            cudaFree(d_logits_dev);
            if (sce != cudaSuccess) return sce;
            tokens.push_back(next);
            gs.Emit(next, pos, maxv);
            if (tokens.size() == 1)
                ttft_ms = static_cast<double>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t_start)
                        .count());
            if (next == eos_id && !req.ignore_eos)
            {
                eos_terminated = true;
                break;
            }
            const cudaError_t se = xing::XingDecodeStep(
                pw_, dims_, *session_, next, d_rc_, d_rs_, pos, stream());
            if (se != cudaSuccess) return se;
            ++pos;
        }
        kv_slots = session_ ? session_->length : pos;
        session_tail_ = feed;
        session_tail_.insert(session_tail_.end(), tokens.begin(), tokens.end());
        return cudaSuccess;
    });

    total_ms = static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_start)
            .count());

    if (run != cudaSuccess)
    {
        r.error_msg = runtime::CombineGenError(run, gen_error);
        r.ok = false;
        return r;
    }

    std::vector<float> final_logits(
        host_logits_.begin(),
        host_logits_.end());
    runtime::FillInferenceResult(
        r,
        {.tokens = &tokens,
         .final_logits = &final_logits,
         .ttft_ms = ttft_ms,
         .total_ms = total_ms,
         .error_msg = &gen_error,
         .eos_terminated = eos_terminated,
         .kv_slots_used = kv_slots,
         .first_maxv = 0.f,
         .resident_bytes = pw_.resident_bytes},
        dims_.layers);
    if (tok_db_ && !r.gen_tokens.empty())
        r.output_text = runtime::DecodeGeneratedText(*tok_db_, r.gen_tokens);
    return r;
}

gpu::InferenceResult XingMoeModel::SelfCheck(const ModelRequest &req)
{
    return runtime::RunParisSelfCheck(*this, req);
}

bool XingMoeModel::BuildConversationInputIds(
    const ArrowTokenizerDb &tok, const ChatConversation &conv,
    std::vector<int32_t> &out_ids) const
{
    return xing::tool_format::Render(tok, conv, out_ids, true);
}

bool XingMoeModel::ParseToolCalls(const ArrowTokenizerDb &tok,
                                  const std::vector<int32_t> &gen_tokens,
                                  std::vector<ToolCall> &out_calls) const
{
    return xing::tool_format::Parse(tok, gen_tokens, out_calls);
}

}  // namespace lsxcommon