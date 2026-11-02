// Mistral-Large-4-1T-A52B model class (Model Ingestion Protocol stages 3 + 8).
// Owns the whole lifecycle: load (NVFP4 dense graph + BF16 edges + streamed
// routed experts), the frozen-geometry gate, per-request decode under the
// residency lock, streaming/abort through GenerationStream, the Paris
// SelfCheck and the native [INST]/[TOOL_CALLS] tool bridge with the Large-4
// [MODEL_SETTINGS] block.

#include "mistral_moe_large4_model.h"

#include "mistral_moe_large4_constants.h"
#include "lsxcommon/mistral_moe_tool_format.h"

#include "lsxcommon/gpu_perf.h"
#include "lsxcommon/log.h"
#include "lsxcommon/lsx_attn_inference.h"  // DecideSession (shared, neutral)
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/resource_governor.h"
#include "lsxcommon/sampling.h"
#include "lsxcommon/synthetic_geometry.h"

#include <absl/strings/str_cat.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define LARGE4_MODEL_CK(call)                  \
    do {                                       \
        const cudaError_t e_ = (call);         \
        if (e_ != cudaSuccess) return e_;      \
    } while (0)

namespace lsxcommon {

const std::string MistralLarge4MoeModel::kType = "mistral-large-4-1t";

namespace {
using namespace lsxcommon::mistral_large4;

int MsSince(std::chrono::steady_clock::time_point t0)
{
    return static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0)
            .count());
}
} // namespace

MistralLarge4MoeModel::MistralLarge4MoeModel(ModelInitConfig cfg)
    : ModelBase(std::move(cfg))
{
    kv_cache_layout_ = ModelBase::cfg_.kv_cache_layout;
    tokenizer_optional_ = lsxcommon::SyntheticGeometryAllowed();
}

void MistralLarge4MoeModel::TeardownWeights()
{
    session_.reset();
    if (d_next_token_) cudaFree(d_next_token_);
    if (d_next_maxv_) cudaFree(d_next_maxv_);
    d_next_token_ = nullptr;
    d_next_maxv_ = nullptr;
    rc_.clear();
    rs_.clear();
    host_logits_.clear();
    rope_capacity_ = 0;
    session_tail_.clear();
    mistral_large4::FreeMistralLarge4Weights(pw_);
    TeardownContext();
}

bool MistralLarge4MoeModel::ValidateGeometry() const
{
    if (SyntheticGeometryAllowed()) return true;
    const auto &d = pw_.dims;
    const bool ok = d.layers == frozen::kLayers &&
                    d.hidden == frozen::kHidden && d.vocab == frozen::kVocab &&
                    d.heads == frozen::kHeads && d.kv_lora == frozen::kKvLora &&
                    d.qk_nope == frozen::kQkNope &&
                    d.qk_rope == frozen::kQkRope && d.v_head == frozen::kVHead &&
                    d.experts == frozen::kNumRouted &&
                    d.top_k == frozen::kTopK &&
                    d.moe_inter == frozen::kMoeIntermediate &&
                    d.dense_inter == frozen::kDenseIntermediate &&
                    d.index_heads == frozen::kIndexHeads &&
                    d.index_head_dim == frozen::kIndexHeadDim &&
                    d.index_topk == frozen::kIndexTopk;
    if (!ok)
        log::error("[mistral-large4-model] frozen geometry mismatch after load");
    return ok;
}

bool MistralLarge4MoeModel::PrepareRopeTables(int capacity)
{
    if (rope_capacity_ >= capacity) return true;
    if (!mistral_large4::BuildLarge4RopeTables(
            capacity, pw_.lerope_alpha_host, pw_.dims.qk_rope,
            frozen::kRopeTheta, frozen::kYarnFactor, frozen::kYarnOrig,
            frozen::kYarnBetaFast, frozen::kYarnBetaSlow, rc_, rs_))
    {
        log::error("[mistral-large4-model] rope table build failed");
        return false;
    }
    rope_capacity_ = capacity;
    return true;
}

bool MistralLarge4MoeModel::Load()
{
    Unload();
    if (!BeginLoad(cfg_.model_path)) return false;
    if (!EnsureContext())
    {
        log::error("[mistral-large4-model] stream init failed");
        return AbortLoad();
    }

    std::string err;
    const auto t0 = std::chrono::steady_clock::now();
    const cudaError_t e =
        mistral_large4::LoadMistralLarge4Weights(cfg_.model_path, pw_,
                                                 stream(), err);
    if (e != cudaSuccess)
    {
        log::error("[mistral-large4-model] load failed: " + err);
        return AbortLoad();
    }
    log::info(absl::StrCat("[mistral-large4-model] weights loaded in ",
                           MsSince(t0), " ms"));
    if (!ValidateGeometry()) return AbortLoad();
    if (!PrepareRopeTables(4096)) return AbortLoad();
    if (!AcquirePermanentWeights(pw_.resident_bytes)) return AbortLoad();

    const int total_experts = pw_.expert_store.num_layers *
                              pw_.expert_store.num_experts;
    size_t max_pack = 0, max_scale = 0;
    for (const auto &recs : pw_.expert_store.records)
        for (const auto &r : recs)
        {
            max_pack = std::max(max_pack, static_cast<size_t>(
                                              r.w1_bytes + r.w2_bytes +
                                              r.w3_bytes));
            max_scale = std::max(max_scale, static_cast<size_t>(
                                                r.s1_bytes + r.s2_bytes +
                                                r.s3_bytes));
        }
    const auto align_up = [](size_t v) { return (v + 255) & ~static_cast<size_t>(255); };
    const size_t per_expert = align_up(max_pack) + align_up(max_scale);
    int slots = total_experts;
    if (const char *env = std::getenv("LSX_MISTRAL_LARGE4_EXPERT_CACHE_SLOTS"))
    {
        const int v = std::atoi(env);
        if (v > 0) slots = std::min(v, total_experts);
    }
    else
    {
        size_t free_bytes = 0, total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess)
            slots = lifecycle::ExpertCacheSlots(
                free_bytes, per_expert, total_experts,
                perf::CurrentKnobs().expert_slots_pct);
    }
    if (slots <= 0) slots = 4;
    if (slots > total_experts) slots = total_experts;
    AcquireExpertCacheLease(static_cast<std::uint64_t>(slots) * per_expert,
                            per_expert,
                            static_cast<std::uint64_t>(total_experts) *
                                per_expert);
    expert_slots_ = slots;
    log::info(absl::StrCat("[mistral-large4-model] resident ",
                           pw_.resident_bytes, " B, layers=", pw_.dims.layers,
                           " experts=", pw_.dims.experts, " cache_slots=",
                           slots, "/", total_experts));
    FinishLoad();
    return true;
}

gpu::InferenceResult MistralLarge4MoeModel::Infer(const ModelRequest &req)
{
    const lsx_nvfp4::LsxFastMathScope fm_scope(EffectiveFastMathMode(cfg_));

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

    const bool has_tok = tok_db_ && tok_db_->vocab_size > 0;
    const int eos_id =
        (req.ignore_eos || !has_tok) ? kIgnoreEosTokenId : tok_db_->eos_id;
    const int capacity = static_cast<int>(req.input_ids.size()) +
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
        const qwen::SessionDecision session = qwen::DecideSession(
            session_ && session_->length > 0,
            session_ ? session_->last_tok : -1, session_tail_, req.input_ids);
        bool use_cont = session.is_continuation;
        if (use_cont && session_ &&
            session_->length + 1 + std::max(0, req.max_gen_tokens) >
                session_->capacity)
            use_cont = false;
        if (!use_cont)
        {
            session_.reset();
            session_tail_.clear();
            if (!PrepareRopeTables(capacity)) return cudaErrorInvalidValue;
            session_ = std::make_unique<mistral_large4::Large4Session>();
            const cudaError_t ae = mistral_large4::AllocateLarge4Session(
                *session_, pw_.dims, pw_.layers, std::max(4, capacity),
                expert_slots_, rc_, rs_, stream());
            if (ae != cudaSuccess) return ae;
            std::string cache_err;
            if (!moe::AllocateMoEExpertCache(pw_.expert_store, expert_slots_,
                                             session_->cache, cache_err))
                return cudaErrorMemoryAllocation;
        }

        std::vector<int32_t> feed = use_cont ? session.gen_input
                                             : req.input_ids;
        int pos = use_cont ? session_->length : 0;
        for (size_t i = 0; i < feed.size(); ++i)
        {
            if (gs.Aborted()) break;
            const cudaError_t se = mistral_large4::Large4DecodeStep(
                pw_, *session_, feed[i], pos, stream());
            if (se != cudaSuccess) return se;
            ++pos;
        }
        if (gs.Aborted())
        {
            session_tail_ = feed;
            return cudaSuccess;
        }

        auto &sampler = sample::SamplerFor(req.sampling);
        if (d_next_token_ == nullptr &&
            (cudaMalloc(reinterpret_cast<void **>(&d_next_token_),
                        sizeof(int32_t)) != cudaSuccess ||
             cudaMalloc(reinterpret_cast<void **>(&d_next_maxv_),
                        sizeof(float)) != cudaSuccess))
            return cudaErrorMemoryAllocation;

        for (int step = 0; step < std::max(0, req.max_gen_tokens); ++step)
        {
            if (gs.Aborted()) break;
            const cudaError_t sce = sampler.Sample(
                stream(), session_->d_logits, pw_.dims.vocab,
                sample::LogitType::kF32, d_next_token_, d_next_maxv_);
            if (sce != cudaSuccess) return sce;
            int32_t next = 0;
            float maxv = 0.0f;
            cudaMemcpyAsync(&next, d_next_token_, sizeof(int32_t),
                            cudaMemcpyDeviceToHost, stream());
            cudaMemcpyAsync(&maxv, d_next_maxv_, sizeof(float),
                            cudaMemcpyDeviceToHost, stream());
            if (cudaStreamSynchronize(stream()) != cudaSuccess)
                return cudaErrorUnknown;
            if (next == eos_id || step + 1 >= std::max(0, req.max_gen_tokens))
            {
                host_logits_.assign(static_cast<size_t>(pw_.dims.vocab), 0.0f);
                mistral_large4::Large4ReadLogits(*session_, host_logits_.data(),
                                                 stream());
                cudaStreamSynchronize(stream());
            }
            tokens.push_back(next);
            session_->last_tok = next;
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
            if (step + 1 >= std::max(0, req.max_gen_tokens)) break;
            const cudaError_t se = mistral_large4::Large4DecodeStep(
                pw_, *session_, next, pos, stream());
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
        log::error(absl::StrCat("[mistral-large4-model] Infer run=",
                                static_cast<int>(run), " err='", r.error_msg,
                                "'"));
        r.ok = false;
        return r;
    }

    std::vector<float> final_logits(host_logits_.begin(), host_logits_.end());
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
        pw_.dims.layers);
    if (has_tok && !r.gen_tokens.empty())
        r.output_text = runtime::DecodeGeneratedText(*tok_db_, r.gen_tokens);
    return r;
}

gpu::InferenceResult MistralLarge4MoeModel::SelfCheck(const ModelRequest &req)
{
    return runtime::RunParisSelfCheck(*this, req);
}

bool MistralLarge4MoeModel::BuildConversationInputIds(
    const ArrowTokenizerDb &tok, const ChatConversation &conv,
    std::vector<int32_t> &out_ids) const
{
    mistral_moe::tool_format::RenderOptions opts;
    opts.format = mistral_moe::tool_format::MistralFormat::kLarge4;
    opts.reasoning_effort = "high";
    return mistral_moe::tool_format::Render(tok, conv, out_ids, true, opts);
}

bool MistralLarge4MoeModel::ParseToolCalls(
    const ArrowTokenizerDb &tok, const std::vector<int32_t> &gen_tokens,
    std::vector<ToolCall> &out_calls) const
{
    return mistral_moe::tool_format::Parse(tok, gen_tokens, out_calls);
}

} // namespace lsxcommon
