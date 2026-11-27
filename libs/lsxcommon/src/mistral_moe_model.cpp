// Mistral-Small-4-119B model class (Model Ingestion Protocol stages 3 + 8).
// Owns the whole lifecycle: load (BF16 dense graph + streamed routed experts),
// geometry gate against the frozen passport, per-request decode under the
// residency lock, streaming/abort through GenerationStream, Paris SelfCheck and
// the native [INST]/[TOOL_CALLS] tool bridge.

#include "mistral_moe_model.h"

#include "mistral_moe_constants.h"
#include "lsxcommon/mistral_moe_tool_format.h"

#include "lsxcommon/gpu_perf.h"
#include "lsxcommon/log.h"
#include "lsxcommon/lsx_attn_inference.h"   // DecideSession (shared, model-neutral)
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

#define MISTRAL_MODEL_CK(call)                  \
    do {                                         \
        const cudaError_t e_ = (call);           \
        if (e_ != cudaSuccess) return e_;        \
    } while (0)

namespace lsxcommon {

const std::string MistralSmall4MoeModel::kType = "mistral-small-4-119b";

namespace {
using namespace lsxcommon::mistral_moe::frozen;

int MsSince(std::chrono::steady_clock::time_point t0)
{
    return static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0)
            .count());
}
} // namespace

MistralSmall4MoeModel::MistralSmall4MoeModel(ModelInitConfig cfg)
    : ModelBase(std::move(cfg))
{
    kv_cache_layout_ = ModelBase::cfg_.kv_cache_layout;
    // A synthetic fixture ships no tokenizer arrows; tolerate their absence.
    tokenizer_optional_ = lsxcommon::SyntheticGeometryAllowed();
}

void MistralSmall4MoeModel::TeardownWeights()
{
    session_.reset();
    if (d_rc_) cudaFree(d_rc_);
    if (d_rs_) cudaFree(d_rs_);
    if (d_next_token_) cudaFree(d_next_token_);
    if (d_next_maxv_) cudaFree(d_next_maxv_);
    d_rc_ = nullptr;
    d_rs_ = nullptr;
    d_next_token_ = nullptr;
    d_next_maxv_ = nullptr;
    rc_.clear();
    rs_.clear();
    host_logits_.clear();
    yarn_capacity_ = 0;
    session_tail_.clear();
    mistral_moe::FreeMistralWeights(pw_);
    TeardownContext();
}

bool MistralSmall4MoeModel::ValidateGeometry(
    const mistral_moe::MistralTopology &top) const
{
    if (SyntheticGeometryAllowed()) return true;
    const bool ok = top.num_layers == kLayers &&
                    top.experts_per_layer == kNumRouted &&
                    top.moe_layers.size() == static_cast<size_t>(kLayers) &&
                    top.dense_layers.empty();
    if (!ok)
    {
        log::error(absl::StrCat(
            "[mistral-model] frozen geometry mismatch: layers=", top.num_layers,
            " experts=", top.experts_per_layer, " moe=", top.moe_layers.size()));
        return false;
    }
    return true;
}

bool MistralSmall4MoeModel::Load()
{
    Unload();
    if (!BeginLoad(cfg_.model_path)) return false;
    if (!EnsureContext())
    {
        log::error("[mistral-model] stream init failed");
        return AbortLoad();
    }

    std::string err;
    const auto t0 = std::chrono::steady_clock::now();
    const cudaError_t e =
        mistral_moe::LoadMistralWeights(cfg_.model_path, pw_, stream(), err);
    if (e != cudaSuccess)
    {
        log::error("[mistral-model] load failed: " + err);
        return AbortLoad();
    }
    log::info(absl::StrCat("[mistral-model] weights loaded in ", MsSince(t0),
                           " ms"));

    const mistral_moe::MistralTopology top =
        mistral_moe::mistral_load_topology(
            (std::filesystem::path(cfg_.model_path) /
             "model.index.json")
                .string());
    if (!ValidateGeometry(top)) return AbortLoad();
    vram_ = top.vram;

    dims_ = mistral_moe::MistralDimsFromWeights(pw_);
    if (!AcquirePermanentWeights(pw_.resident_bytes)) return AbortLoad();

    // Expert-cache slots: env override, else as many as free VRAM allows with a
    // 2 GiB reserve (dense weights are already uploaded at this point).
    const int total_experts = pw_.expert_store.num_layers *
                              pw_.expert_store.num_experts;
    size_t max_pack = 0, max_scale = 0;
    for (int l = 0; l < pw_.expert_store.num_layers; ++l)
        for (const auto &r : pw_.expert_store.records[static_cast<size_t>(l)])
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
    if (const char *env = std::getenv("LSX_MISTRAL_EXPERT_CACHE_SLOTS"))
    {
        const int v = std::atoi(env);
        if (v > 0) slots = std::min(v, total_experts);
    }
    else
    {
        size_t free_bytes = 0, total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess)
        {
            // Shared estimator (8 GiB calibrated reserve + the resolved
            // expert_slots_pct knob). A local budget with a smaller reserve
            // over-commits VRAM and starves the KV/session arena at decode
            // time (the 2048-token bench OOMs with 1838 slots).
            slots = lifecycle::ExpertCacheSlots(
                free_bytes, per_expert, total_experts,
                perf::CurrentKnobs().expert_slots_pct);
        }
    }
    if (slots <= 0) slots = 4;
    if (slots > total_experts) slots = total_experts;
    AcquireExpertCacheLease(static_cast<std::uint64_t>(slots) * per_expert,
                            per_expert,
                            static_cast<std::uint64_t>(total_experts) *
                                per_expert);
    expert_slots_ = slots;
    log::info(absl::StrCat("[mistral-model] resident ", pw_.resident_bytes,
                           " B, layers=", dims_.layers, " experts=",
                           dims_.experts, " cache_slots=", slots, "/",
                           total_experts));
    FinishLoad();
    return true;
}

gpu::InferenceResult MistralSmall4MoeModel::Infer(const ModelRequest &req)
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
        // A continuation is only safe while the existing cache can hold the
        // token we are about to append plus the whole generation budget
        // (audit X1: the old path wrote past capacity). If it cannot, fall
        // back to a fresh prefill of the full context instead of growing the
        // multi-buffer MLA session in place.
        bool use_cont = session.is_continuation;
        if (use_cont && session_ &&
            session_->length + 1 + std::max(0, req.max_gen_tokens) >
                session_->capacity)
            use_cont = false;
        if (!use_cont)
        {
            session_.reset();
            session_tail_.clear();
            if (yarn_capacity_ < capacity)
            {
                if (!mistral_moe::BuildMistralYarnTables(capacity, rc_, rs_))
                    return cudaErrorInvalidValue;
                if (d_rc_) cudaFree(d_rc_);
                if (d_rs_) cudaFree(d_rs_);
                d_rc_ = nullptr;
                d_rs_ = nullptr;
                if (mistral_moe::UploadMistralYarnTables(
                        stream(), capacity, rc_, rs_, &d_rc_, &d_rs_) !=
                    cudaSuccess)
                    return cudaErrorMemoryAllocation;
                yarn_capacity_ = capacity;
            }
            session_ = std::make_unique<mistral_moe::MistralSession>();
            const cudaError_t ae = mistral_moe::AllocateMistralSession(
                *session_, dims_, std::max(4, capacity), expert_slots_,
                stream());
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
            const cudaError_t se = mistral_moe::MistralDecodeStep(
                pw_, dims_, *session_, feed[i], d_rc_, d_rs_, pos, stream());
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
                stream(), session_->d_logits, dims_.vocab,
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
                host_logits_.assign(static_cast<size_t>(dims_.vocab), 0.0f);
                mistral_moe::MistralReadLogits(*session_, host_logits_.data(),
                                               stream());
                cudaStreamSynchronize(stream());
            }
            tokens.push_back(next);
            // `next` is the last SAMPLED token. It is decoded only if another
            // step follows; otherwise it stays out of the cache and is the
            // continuation seed (audit X2).
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
            const cudaError_t se = mistral_moe::MistralDecodeStep(
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
        log::error(absl::StrCat("[mistral-model] Infer run=",
                                static_cast<int>(run), " err='", r.error_msg,
                                "' gen_error='", gen_error, "'"));
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
        dims_.layers);
    if (has_tok && !r.gen_tokens.empty())
        r.output_text = runtime::DecodeGeneratedText(*tok_db_, r.gen_tokens);
    return r;
}

gpu::InferenceResult MistralSmall4MoeModel::SelfCheck(const ModelRequest &req)
{
    return runtime::RunParisSelfCheck(*this, req);
}

bool MistralSmall4MoeModel::BuildConversationInputIds(
    const ArrowTokenizerDb &tok, const ChatConversation &conv,
    std::vector<int32_t> &out_ids) const
{
    return mistral_moe::tool_format::Render(tok, conv, out_ids, true);
}

bool MistralSmall4MoeModel::ParseToolCalls(
    const ArrowTokenizerDb &tok, const std::vector<int32_t> &gen_tokens,
    std::vector<ToolCall> &out_calls) const
{
    return mistral_moe::tool_format::Parse(tok, gen_tokens, out_calls);
}

} // namespace lsxcommon
