#include "kimi_moe_model.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <filesystem>
#include <shared_mutex>

#include "kimi_moe_kernels.h"
#include "lsxcommon/kimi_moe_convert.h"
#include "lsxcommon/log.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/resource_governor.h"

namespace lsxcommon {

const std::string KimiK27CodeModel::kType = "kimi-k2.7-code";

KimiK27CodeModel::KimiK27CodeModel(ModelInitConfig cfg)
    : ModelBase{std::move(cfg)} {}

KimiK27CodeModel::~KimiK27CodeModel() { Unload(); }

void KimiK27CodeModel::TeardownWeights()
{
    kimi::FreeKimiMoeExpertDeviceCache(expert_cache_.device);
    permanent_lease_ = {};
    kimi::free_kimi_expert_index(experts_);
    pw_.Free();
    TeardownContext();
}

bool KimiK27CodeModel::Load()
{
    Unload();  // idempotent re-load support
    if (!BeginLoad(cfg_.model_path)) return false;

    namespace fs = std::filesystem;

    if (!EnsureContext(/*with_cublas=*/true)) {
        log::error("[kimi-model] cublas/stream init failed");
        return AbortLoad();
    }

    const cudaError_t e =
        kimi::LoadKimiPinnedWeights(cfg_.model_path, pw_, stream());
    if (e != cudaSuccess ||
        pw_.hidden != kimi::kKimiHidden ||
        pw_.layers.size() != static_cast<size_t>(kimi::kKimiNumLayers) ||
        !pw_.d_embed || !pw_.d_head) {
        log::error(std::string(
                       "[kimi-model] pinned-weight load/validation failed: ") +
                   (e != cudaSuccess ? cudaGetErrorString(e)
                                     : "geometry mismatch"));
        return AbortLoad();
    }

    experts_ = kimi::load_kimi_expert_index(cfg_.model_path,
                                            kimi::kKimiNumLayers,
                                            kimi::kKimiNumRouted);
    expert_cache_.plan.Init(0, kimi::kKimiNumLayers * kimi::kKimiNumRouted);
    if (experts_.num_layers != kimi::kKimiNumLayers ||
        experts_.experts_per_layer != kimi::kKimiNumRouted) {
        log::error("[kimi-model] expert index incomplete");
        return AbortLoad();
    }

    // Governor parity with GLM (lifecycle-unification 2026-08): standalone
    // init when the engine never initialized it, then the permanent-weight
    // lease.
    auto &gov = ResourceGovernor::Instance();
    if (!gov.IsInitialized()) {
        log::warn("[kimi-model] governor not initialized; standalone init");
        GovernorOptions gopts;
        if (!gov.Initialize(gopts)) {
            log::error("[kimi-model] governor init failed");
            return AbortLoad();
        }
    }
    if (!permanent_lease_.Valid()) {
        permanent_lease_ = gov.Acquire(LeaseKind::kPermanentWeights,
                                       pw_.resident_bytes, LeasePolicy{});
        if (!permanent_lease_.Valid() &&
            !gov.options().relax_memory_refusal) {
            log::error("[kimi-model] permanent-weight lease refused");
            return AbortLoad();
        }
    }

    resident_bytes_ = pw_.resident_bytes;
    log::info("[kimi-model] loaded once: resident="
              + std::to_string(pw_.resident_bytes) + " bytes");
    FinishLoad();
    return true;
}

gpu::InferenceResult KimiK27CodeModel::Infer(const ModelRequest &req)
{
    gpu::InferenceResult r;
    if (!ValidateInferRequest(req, r)) return r;
    if (!loaded_ || !pw_.d_embed) {
        r.error_msg = "model not loaded (call Load(); weights are no longer "
                      "reloaded per Infer call)";
        r.ok = false;
        return r;
    }

    kimi::KimiGenerationResult gen;
    const cudaError_t err = RunUnderResidencyLock([&] {
        return kimi::RunKimiGeneration(cublas(), stream(), pw_, experts_,
                                       req.input_ids, req.max_gen_tokens,
                                       gen, &expert_cache_);
    });
    if (err != cudaSuccess || !gen.ok) {
        r.error_msg = runtime::CombineGenError(err, gen.error_msg);
        r.ok = false;
        return r;
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
            .resident_bytes = resident_bytes_},
        kimi::kKimiNumLayers);

    if (!r.gen_tokens.empty())
        r.output_text = runtime::DecodeGeneratedText(*tok_db_, r.gen_tokens);
    return r;
}

// KimiK27CodeModel::Unload — canonical ModelBase template method
// (TeardownWeights override above); no per-model override.

gpu::InferenceResult KimiK27CodeModel::SelfCheck(const ModelRequest &req)
{
    return runtime::RunParisSelfCheck(*this, req);
}

gpu::BenchmarkResult KimiK27CodeModel::Benchmark(int num_tokens)
{
    gpu::BenchmarkResult r;
    if (!RequireLoaded(r)) return r;
    if (!pw_.d_embed) {
        r.error_msg = "model not loaded";
        return r;
    }
    // Synthetic prompt: repeat token 0 (a stable vocab entry) 4 times
    // (GLM parity).
    const std::vector<int32_t> prompt = {0, 0, 0, 0};

    kimi::KimiGenerationResult gr;
    const cudaError_t berr = RunUnderResidencyLock([&] {
        return kimi::RunKimiGeneration(cublas(), stream(), pw_, experts_,
                                       prompt, num_tokens, gr,
                                       &expert_cache_);
    });
    if (berr != cudaSuccess || !gr.ok) {
        r.error_msg = runtime::CombineGenError(berr, gr.error_msg);
        return r;
    }
    runtime::FillBenchmarkResult(r, gr.total_ms,
                                 static_cast<int>(gr.tokens.size()),
                                 kimi::kKimiNumLayers, resident_bytes_);
    return r;
}

} // namespace lsxcommon
