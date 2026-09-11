#include "kimi_moe_model.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <filesystem>
#include <shared_mutex>

#include "kimi_moe_kernels.h"
#include "lsxcommon/kimi_moe_convert.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
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

    // Governor parity (lifecycle-unification II): standalone init + the
    // permanent-weight lease live in ModelBase.
    if (!AcquirePermanentWeights(pw_.resident_bytes)) return AbortLoad();

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
        // Model-independence contract: this model asserts its OWN fast-math
        // policy for the duration of the generation (LsxFastMathScope).
        const lsx_nvfp4::LsxFastMathScope fm_scope(
            EffectiveFastMathMode(cfg_));
        return kimi::RunKimiGeneration(cublas(), stream(), pw_, experts_,
                                       req.input_ids, req.max_gen_tokens,
                                       gen, &expert_cache_,
                                       Tokenizer().eos_id, cfg_.fast_math);
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

} // namespace lsxcommon
