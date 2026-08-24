#include "glm_v5_model.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <filesystem>
#include <shared_mutex>

#include "lsxcommon/glm_convert.h"
#include "lsxcommon/glm_inference.h"
#include "lsxcommon/log.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/resource_governor.h"

namespace lsxcommon {

const std::string GlmV5Model::kType = "glm-5.2";

// GLM-5.2 chat framing — model-owned (prompt_framing.h declares the symbol).
namespace glm {
const prompt::FramingDescriptor kFraming = {
    {"[gMASK]", "<sop>", "<|user|>"},
    {"<|assistant|>", "<think>", "</think>"},
    3, 3,
    /*leading_bos=*/false,
};
bool BuildPromptInputIds(const ArrowTokenizerDb &tok,
                         std::string_view raw_prompt,
                         std::vector<int32_t> &out_ids)
{
    return prompt::ApplyFraming(kFraming, tok, raw_prompt, out_ids);
}
} // namespace glm

bool GlmV5Model::BuildPromptInputIds(const ArrowTokenizerDb &tok,
                                     std::string_view raw_prompt,
                                     std::vector<int32_t> &out_ids) const
{
    return glm::BuildPromptInputIds(tok, raw_prompt, out_ids);
}

GlmV5Model::GlmV5Model(ModelInitConfig cfg)
    : ModelBase{std::move(cfg)} {}

GlmV5Model::~GlmV5Model() { Unload(); }

bool GlmV5Model::EnsureExecutionContext()
{
    if (!stream_ && cudaStreamCreateWithFlags(&stream_,
                                              cudaStreamNonBlocking) !=
                        cudaSuccess) {
        stream_ = nullptr;
        return false;
    }
    if (!cublas_ && cublasCreate(&cublas_) != CUBLAS_STATUS_SUCCESS) {
        cublas_ = nullptr;
        return false;
    }
    cublasSetStream(cublas_, stream_);
    return true;
}

std::uint64_t GlmV5Model::EstimateExpertMeanBytes() const
{
    // NVFP4 routed expert: three [kGlmMoeIntermediate x kGlmHidden] matrices,
    // each packed E2M1 nibbles (elems/2 B) + E4M3 per-block scales (elems/16 B).
    constexpr std::int64_t elems =
        static_cast<std::int64_t>(glm::kGlmMoeIntermediate) * glm::kGlmHidden;
    constexpr std::int64_t per_matrix = elems / 2 + elems / 16;
    return static_cast<std::uint64_t>(per_matrix) * 3;
}

void GlmV5Model::TeardownWeights()
{
    tuner_.reset();          // releases the adaptive expert-cache lease
    permanent_lease_ = {};   // RAII release
    cache_lease_ = {};
    glm::free_glm_expert_index(experts_);
    pw_.Free();
    if (cublas_) { cublasDestroy(cublas_); cublas_ = nullptr; }
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
    resident_bytes_ = 0;
}

bool GlmV5Model::Load()
{
    Unload();  // idempotent re-load support
    if (!BeginLoad(cfg_.model_path)) return false;

    namespace fs = std::filesystem;
    fs::path idx = fs::path(cfg_.model_path) / "logestix" / "model.index.json";
    auto topo = glm::load_topology(idx.string());
    if (!topo.error_msg.empty()) {
        lsxcommon::log::error("[glm-model] topology error in "
                              + idx.string() + ": " + topo.error_msg);
        tok_db_.reset();
        return false;
    }
    vram_ = topo.vram;

    auto &gov = ResourceGovernor::Instance();
    if (!gov.IsInitialized()) {
        log::warn("[glm-model] governor not initialized; standalone init");
        GovernorOptions gopts;
        if (!gov.Initialize(gopts)) {
            log::error("[glm-model] governor init failed");
            tok_db_.reset();
            return false;
        }
    }

    if (!EnsureExecutionContext()) {
        log::error("[glm-model] cublas/stream init failed");
        TeardownWeights();
        tok_db_.reset();
        return false;
    }

    const cudaError_t e = glm::LoadGlmPinnedWeights(cfg_.model_path, pw_, stream_);
    if (e != cudaSuccess ||
        pw_.hidden != glm::kGlmHidden ||
        pw_.layers.size() != static_cast<size_t>(glm::kGlmNumLayers) ||
        !pw_.d_embed || !pw_.d_head) {
        log::error(std::string("[glm-model] pinned-weight load/validation failed: ")
                   + (e != cudaSuccess ? cudaGetErrorString(e)
                                       : "geometry mismatch"));
        TeardownWeights();
        tok_db_.reset();
        return false;
    }

    experts_ = glm::load_glm_expert_index(cfg_.model_path, glm::kGlmNumLayers,
                                          glm::kGlmNumRouted);
    if (experts_.num_layers != glm::kGlmNumLayers ||
        experts_.experts_per_layer != glm::kGlmNumRouted) {
        log::error("[glm-model] expert index incomplete");
        TeardownWeights();
        tok_db_.reset();
        return false;
    }

    lifecycle::LifecycleInput in;
    in.model_type = kType;
    in.archetype = lifecycle::Archetype::MoE;
    in.vram = topo.vram;
    in.expert_mean_bytes = EstimateExpertMeanBytes();
    const auto &plan = lifecycle::GetOrBuildPlan(in);

    permanent_lease_ = gov.Acquire(LeaseKind::kPermanentWeights,
                                   pw_.resident_bytes, LeasePolicy{});
    if (!permanent_lease_.Valid() && !gov.options().relax_memory_refusal) {
        log::error("[glm-model] permanent-weight lease refused ("
                   + std::to_string(pw_.resident_bytes) + " bytes)");
        TeardownWeights();
        tok_db_.reset();
        return false;
    }

    LeasePolicy adaptive{LeasePolicy::Mode::kAdaptive,
                         plan.tuner_bounds.expert_cache_slots_min *
                             in.expert_mean_bytes,
                         plan.expert_cache_max_bytes};
    cache_lease_ = gov.Acquire(LeaseKind::kExpertCache,
                               plan.expert_cache_slots * in.expert_mean_bytes,
                               adaptive);
    tuner_ = std::make_unique<lifecycle::RuntimeTuner>(
        &gov, std::move(cache_lease_), in.expert_mean_bytes, plan.tuner_bounds);
    glm::SetGlmExpertCacheSlotsOverride(static_cast<int>(tuner_->Slots()));
    tuner_last_hits_ = tuner_last_misses_ = 0;

    resident_bytes_ = pw_.resident_bytes;
    log::info("[glm-model] loaded once: resident="
              + std::to_string(pw_.resident_bytes)
              + " bytes, plan: cache slots="
              + std::to_string(plan.expert_cache_slots)
              + " streams=" + std::to_string(plan.copy_streams)
              + " staging=" + std::to_string(plan.staging_ring_bytes));
    FinishLoad();
    return true;
}

gpu::InferenceResult GlmV5Model::Infer(const ModelRequest &req)
{
    gpu::InferenceResult r;
    if (!ValidateInferRequest(req, r)) return r;
    if (!loaded_ || !pw_.d_embed) {
        r.error_msg = "model not loaded (call Load(); weights are no longer "
                      "reloaded per Infer call)";
        r.ok = false;
        return r;
    }

    glm::GlmGenerationResult gen;
    {
        std::shared_lock<std::shared_mutex> residency(residency_mutex_);
        const cudaError_t e = glm::RunGlmRealGeneration(
            cublas_, stream_, pw_, experts_, req.input_ids,
            req.max_gen_tokens, gen);
        if (e != cudaSuccess || !gen.ok) {
            r.error_msg = gen.ok ? cudaGetErrorString(e) : gen.error_msg;
            r.ok = false;
            return r;
        }
    }

    if (tuner_) {  // between requests: adapt the expert cache (spec 4)
        tuner_->Report(gen.cache_hits - tuner_last_hits_,
                       gen.cache_misses - tuner_last_misses_);
        tuner_last_hits_ = gen.cache_hits;
        tuner_last_misses_ = gen.cache_misses;
        glm::SetGlmExpertCacheSlotsOverride(static_cast<int>(tuner_->Slots()));
        ResourceGovernor::Instance().ReconcileVram();
    }

    runtime::FillInferenceResult(
        r, {.tokens = &gen.tokens,
            .final_logits = &gen.final_logits,
            .ttft_ms = gen.ttft_ms,
            .total_ms = gen.total_ms,
            .error_msg = &gen.error_msg,
            .eos_terminated = gen.eos_terminated,
            .kv_slots_used = gen.kv_slots_used,
            .first_maxv = gen.first_maxv,
            .resident_bytes = gen.resident_bytes},
        glm::kGlmNumLayers);

    // Decode the generated token stream into output_text (gen.tokens carries
    // only generated tokens, never the prompt). A decode failure leaves
    // output_text empty but keeps r.ok true — generation succeeded, decode is
    // best-effort.
    if (!r.gen_tokens.empty())
        r.output_text = runtime::DecodeGeneratedText(*tok_db_, r.gen_tokens);
    return r;
}

void GlmV5Model::Unload()
{
    TeardownWeights();
    tok_db_.reset();
    loaded_ = false;
}

gpu::InferenceResult GlmV5Model::SelfCheck(const ModelRequest &req)
{
    return runtime::RunParisSelfCheck(*this, req);
}

gpu::BenchmarkResult GlmV5Model::Benchmark(int num_tokens)
{
    gpu::BenchmarkResult r;
    if (!loaded_ || !pw_.d_embed) { r.error_msg = "model not loaded"; return r; }
    // Synthetic prompt: repeat token 0 (a stable vocab entry) 4 times.
    const std::vector<int32_t> prompt = {0, 0, 0, 0};

    glm::GlmGenerationResult gr;
    {
        std::shared_lock<std::shared_mutex> residency(residency_mutex_);
        const cudaError_t e = glm::RunGlmRealGeneration(
            cublas_, stream_, pw_, experts_, prompt, num_tokens, gr);
        if (e != cudaSuccess || !gr.ok) {
            r.error_msg = gr.ok ? cudaGetErrorString(e) : gr.error_msg;
            return r;
        }
    }
    runtime::FillBenchmarkResult(r, gr.total_ms,
                                 static_cast<int>(gr.tokens.size()),
                                 glm::kGlmNumLayers, gr.resident_bytes);
    return r;
}

} // namespace lsxcommon
