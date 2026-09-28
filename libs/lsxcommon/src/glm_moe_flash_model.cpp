// Glm5NextFlashModel implementation (Kimi/Qwen38 ModelBase pattern).
#include "lsxcommon/gpu_perf.h"
#include "glm_moe_flash_model.h"

#include "glm_moe_flash_constants.h"

#include "lsxcommon/log.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/qwen_inference.h"
#include "lsxcommon/lsx_inference_nvfp4.h"

#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>
#include <algorithm>
#include <cstdlib>

#include <cuda_runtime.h>

namespace lsxcommon {

const std::string Glm5NextFlashModel::kType = "glm-5.3-flash";

namespace {

// Device memory reserved for generation-time allocations on top of the
// expert cache: fp16 embed table (V*H*2B ~ 0.63 GB), head/logits, KDA/DSA
// per-step arenas, and CUDA context overhead.
constexpr size_t kFlashGenReserveBytes = size_t{3} << 30;

int FlashExpertSlotBytes(const qwen_moe::QwenMoeExpertStore &store)
{
    size_t max_pack = 0, max_scale = 0;
    for (int l = 0; l < store.num_layers; ++l) {
        for (const auto &r : store.records[static_cast<size_t>(l)]) {
            max_pack = std::max(max_pack,
                                static_cast<size_t>(r.w1_bytes + r.w2_bytes +
                                                    r.w3_bytes));
            max_scale = std::max(max_scale,
                                 static_cast<size_t>(r.s1_bytes + r.s2_bytes +
                                                     r.s3_bytes));
        }
    }
    constexpr size_t kAlign = 256;
    const size_t pack = (max_pack + kAlign - 1) / kAlign * kAlign;
    const size_t scale = (max_scale + kAlign - 1) / kAlign * kAlign;
    return static_cast<int>(pack + scale);
}

int ResolveExpertSlots(const qwen_moe::QwenMoeExpertStore &store)
{
    const int max_slots = store.num_layers * store.num_experts;
    int slots = 64;  // legacy floor: top-k per-layer working set headroom
    if (const char *env = std::getenv("LSX_FLASH_SLOTS")) {
        const int parsed = std::atoi(env);
        if (parsed > 0) slots = parsed;
    } else {
        size_t free_bytes = 0, total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
            size_t budget =
                free_bytes > kFlashGenReserveBytes
                    ? free_bytes - kFlashGenReserveBytes
                    : 0;
            // Perf policy: scale the VRAM budget by the resolved slot
            // percentage (default 100 = unchanged behavior).
            const int pct = perf::CurrentKnobs().expert_slots_pct;
            if (pct < 100) budget = budget / 100u * static_cast<unsigned>(pct);
            const int slot_bytes = FlashExpertSlotBytes(store);
            if (slot_bytes > 0) {
                const int by_vram =
                    static_cast<int>(budget / static_cast<size_t>(slot_bytes));
                slots = std::max(slots, std::min(by_vram, max_slots));
            }
        }
    }
    return std::min(slots, max_slots);
}

} // namespace

Glm5NextFlashModel::Glm5NextFlashModel(ModelInitConfig cfg)
    : ModelBase(std::move(cfg))
{
    // Both toggles are process-wide (see Qwen35MoeModel): without them every
    // NVFP4 GEMM runs the shared-memory tile kernel even at seq==1 decode.
    const FastMathMode fm = EffectiveFastMathMode(cfg_);
    if (fm != FastMathMode::kAuto) {
        lsxcommon::lsx_nvfp4::SetLsxTcFastMath(fm == FastMathMode::kOn);
        lsxcommon::lsx_nvfp4::SetLsxSeq1Gemv(fm == FastMathMode::kOn);
    }
}

Glm5NextFlashModel::~Glm5NextFlashModel() { Unload(); }

bool Glm5NextFlashModel::Load()
{
    Unload();  // idempotent re-load support
    if (!BeginLoad(cfg_.model_path)) return false;

    if (!EnsureContext()) {
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
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const int slots = ResolveExpertSlots(experts_);
        log::info(absl::StrCat(
            "[glm-flash-model] expert cache slots=", slots, " (~",
            static_cast<long long>(static_cast<int64_t>(slots) *
                                   FlashExpertSlotBytes(experts_) >> 20),
            " MiB); pre-cache free=", static_cast<long long>(free_b >> 20),
            " MiB total=", static_cast<long long>(total_b >> 20), " MiB"));
        if (!qwen_moe::AllocateQwenMoeExpertCache(experts_, slots, cache_,
                                                  error)) {
            log::error("[glm-flash-model] expert cache alloc failed: "
                       + error);
            return AbortLoad();
        }
    }

    if (!AcquirePermanentWeights(pw_.resident_bytes)) return AbortLoad();

    resident_bytes_ = pw_.resident_bytes;
    log::info("[glm-flash-model] loaded: resident=" +
              std::to_string(resident_bytes_) + " layers=" +
              std::to_string(pw_.layers.size()));
    FinishLoad();
    return true;
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

    // Model independence: assert THIS model's kernel-routing policy for the
    // duration of the call (process-global flags would otherwise leak the
    // last-constructed model's fast_math across the registry).
    const lsx_nvfp4::LsxFastMathScope fm_scope(EffectiveFastMathMode(cfg_));

    qwen::QwenGenerationResult gen;
    sample::GreedySampler greedy;
    lsxcommon::GenerationStream gs;
    gs.on_token = req.token_callback ? &req.token_callback : nullptr;
    gs.abort_flag = req.abort_flag;
    const cudaError_t err = RunUnderResidencyLock([&] {
        std::vector<int> eos_ids =
            req.ignore_eos ? std::vector<int>{lsxcommon::kIgnoreEosTokenId} : pw_.top.eos_ids;
        return glm_moe_flash::RunGlmFlashGeneration(
            stream(), pw_, experts_, cache_, req.input_ids, req.max_gen_tokens,
            eos_ids, gen, greedy, &gs);
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
    log::info(absl::StrCat(
        "[glm-flash-model] expert cache: hits=", cache_.hits,
        " misses=", cache_.misses,
        " hit_rate=",
        (cache_.hits + cache_.misses) > 0
            ? absl::StrFormat("%.1f%%",
                              100.0 * static_cast<double>(cache_.hits) /
                                  static_cast<double>(cache_.hits +
                                                      cache_.misses))
            : std::string("n/a")));
    if (tok_db_ && !r.gen_tokens.empty())
        r.output_text = runtime::DecodeGeneratedText(*tok_db_, r.gen_tokens);
    return r;
}

gpu::InferenceResult Glm5NextFlashModel::SelfCheck(const ModelRequest &req)
{
    return runtime::RunParisSelfCheck(*this, req);
}

void Glm5NextFlashModel::TeardownWeights()
{
    pw_.Free();
    experts_.Free();
    qwen_moe::FreeQwenMoeExpertCache(cache_);
}

} // namespace lsxcommon
