// Glm5NextFlashModel implementation (Kimi/Qwen38 ModelBase pattern).
#include "glm_moe_flash_model.h"

#include "glm_moe_flash_constants.h"

#include "lsxcommon/log.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/prompt_framing.h"

namespace lsxcommon {

const std::string Glm5NextFlashModel::kType = "glm-5.3-flash";

Glm5NextFlashModel::Glm5NextFlashModel(ModelInitConfig cfg)
    : ModelBase(std::move(cfg))
{
}

Glm5NextFlashModel::~Glm5NextFlashModel() { Unload(); }

bool Glm5NextFlashModel::Load()
{
    Unload();  // idempotent re-load support
    if (!BeginLoad(cfg_.model_path)) return false;

    if (!EnsureContext(/*with_cublas=*/false)) {
        log::error("[glm-flash-model] stream init failed");
        return AbortLoad();
    }

    std::string error;
    if (!glm_moe_flash::LoadGlmFlashDenseWeights(cfg_.model_path, pw_,
                                                 stream(), error)) {
        log::error("[glm-flash-model] dense load failed: " + error);
        return AbortLoad();
    }
    if (!glm_moe_flash::LoadGlmFlashExpertStore(cfg_.model_path, experts_,
                                                error)) {
        log::error("[glm-flash-model] expert store load failed: " + error);
        return AbortLoad();
    }
    // LRU device cache sized to the free-VRAM budget; slots bounded by the
    // top-k resident working set per layer.
    {
        const std::string cache_err;
        const int slots = 64;  // 8 slots/layer working set headroom
        if (!qwen_moe::AllocateQwenMoeExpertCache(experts_, slots, cache_,
                                                  error)) {
            log::error("[glm-flash-model] expert cache alloc failed: "
                       + error);
            return AbortLoad();
        }
        (void) cache_err;
    }

    if (!AcquirePermanentWeights(pw_.resident_bytes)) return AbortLoad();

    resident_bytes_ = pw_.resident_bytes;
    log::info("[glm-flash-model] loaded: resident=" +
              std::to_string(resident_bytes_) + " layers=" +
              std::to_string(pw_.layers.size()));
    FinishLoad();
    return true;
}

bool Glm5NextFlashModel::BuildPromptInputIds(const ArrowTokenizerDb &tok,
                                             std::string_view raw_prompt,
                                             std::vector<int32_t> &out_ids) const
{
    return glm::BuildPromptInputIds(tok, raw_prompt, out_ids);
}

gpu::InferenceResult Glm5NextFlashModel::Infer(const ModelRequest &req)
{
    gpu::InferenceResult r;
    if (!ValidateInferRequest(req, r)) return r;
    if (!loaded_ || pw_.layers.empty()) {
        r.error_msg = "model not loaded (call Load(); weights are no longer "
                      "reloaded per Infer call)";
        r.ok = false;
        return r;
    }

    qwen::QwenGenerationResult gen;
    sample::GreedySampler greedy;
    const cudaError_t err = RunUnderResidencyLock([&] {
        return glm_moe_flash::RunGlmFlashGeneration(
            stream(), pw_, experts_, cache_, req.input_ids, req.max_gen_tokens,
            pw_.top.eos_ids, gen, greedy);
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
            .kv_slots_used = -1,
            .first_maxv = 0.0f,
            .resident_bytes = resident_bytes_},
        glm_moe_flash::frozen::kLayers);
    return r;
}

gpu::InferenceResult Glm5NextFlashModel::SelfCheck(const ModelRequest &req)
{
    return runtime::RunParisSelfCheck(*this, req);
}

gpu::BenchmarkResult Glm5NextFlashModel::Benchmark(int num_tokens)
{
    gpu::BenchmarkResult r;
    if (!RequireLoaded(r)) return r;
    r.error_msg = "benchmark not implemented for glm-5.3-flash";
    r.ok = false;
    return r;
}

void Glm5NextFlashModel::TeardownWeights()
{
    pw_.Free();
    experts_.Free();
    qwen_moe::FreeQwenMoeExpertCache(cache_);
}

} // namespace lsxcommon
