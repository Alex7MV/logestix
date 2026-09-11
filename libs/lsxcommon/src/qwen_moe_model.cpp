#include "qwen_moe_model.h"
#include "qwen_moe_constants.h"

#include <absl/strings/str_cat.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>

#include "lsxcommon/log.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/resource_governor.h"
#include "lsxcommon/gpu_perf.h"

namespace lsxcommon {

namespace {
double MsSince(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0)
        .count();
}
} // namespace

const std::string Qwen35MoeModel::kType = "qwen3.5-35b-a3b";

QwenMoeModelBase::QwenMoeModelBase(ModelInitConfig cfg)
    : ModelBase{std::move(cfg)}
{
    // The TC W4A4 toggle is process-wide; the dense Qwen models flip it in
    // QwenBaseModel, but this MoE family derives from ModelBase directly and
    // would silently run every GEMM on the scalar path without this.
    const FastMathMode fm = EffectiveFastMathMode(cfg_);
    if (fm != FastMathMode::kAuto) {
        qwen::SetQwenTcFastMath(fm == FastMathMode::kOn);
        qwen::SetQwenSeq1Gemv(fm == FastMathMode::kOn);
    } else {
        // Engine auto policy: mirror the per-GPU knob set resolved at
        // InitializeEngine (legacy zeros when no engine init — preserves
        // the bit-exact default for direct-factory users).
        const perf::LsxPerfKnobs knobs = perf::CurrentKnobs();
        qwen::SetQwenTcFastMath(knobs.tc_fast_math_default);
        qwen::SetQwenSeq1Gemv(knobs.seq1_gemv_default);
    }
}

Qwen35MoeModel::Qwen35MoeModel(ModelInitConfig cfg)
    : QwenMoeModelBase{std::move(cfg)}
{
}

Qwen35MoeModel::~Qwen35MoeModel() { Unload(); }

void QwenMoeModelBase::TeardownWeights()
{
    qwen_moe::FreeQwenMoeGenScratch(scratch_);
    qwen_moe::FreeQwenMoeExpertCache(cache_);
    experts_.Free();
    qwen_moe::FreeQwenMoePinnedWeights(pw_);
    kv_.reset();
    session_tail_.clear();
    TeardownContext();
}

bool Qwen35MoeModel::ValidateGeometry(
    const qwen_moe::QwenMoeTopology &top) const
{
    using namespace qwen_moe::frozen;
    if (SyntheticGeometryAllowed()) return true;

    const GeoFieldCheck fields[] = {
        {"hidden_size", kHidden, top.hidden_size},
        {"num_layers", kLayers, top.num_layers},
        {"num_attention_heads", kHeads, top.num_attention_heads},
        {"num_key_value_heads", kKVHeads, top.num_key_value_heads},
        {"head_dim", kHeadDim, top.head_dim},
        {"vocab_size", kVocab, top.vocab_size},
        {"num_experts", kNumExperts, top.num_experts},
        {"num_experts_per_tok", kTopK, top.num_experts_per_tok},
        {"moe_intermediate_size", kMoeIntermediate,
         top.moe_intermediate_size},
        {"shared_expert_intermediate_size", kSharedIntermediate,
         top.shared_expert_intermediate_size},
    };
    if (!CheckGeometryFields("qwen-moe-model", fields, 10)) return false;
    for (int l = 0; l < top.num_layers; ++l) {
        const std::string want =
            (l % kFullAttnInterval == kFullAttnInterval - 1) ? "full_attention"
                                                             : "linear_attention";
        if (top.layer_types[static_cast<size_t>(l)] != want) {
            log::error(absl::StrCat("[qwen-moe-model] layer_types[", l,
                                    "] != ", want));
            return false;
        }
    }
    return true;
}

bool QwenMoeModelBase::Load()
{
    Unload();  // idempotent re-load support
    if (!BeginLoad(cfg_.model_path)) return false;

    if (!EnsureContext(/*with_cublas=*/false)) {
        log::error("[qwen-moe-model] stream init failed");
        return AbortLoad();
    }

    std::string err;
    const auto t_dense = std::chrono::steady_clock::now();
    if (!qwen_moe::LoadQwenMoeDenseWeights(cfg_.model_path, pw_, stream(), err)) {
        log::error("[qwen-moe-model] dense load failed: " + err);
        return AbortLoad();
    }
    log::info(absl::StrCat("[qwen-moe-model] dense load ", MsSince(t_dense),
                           " ms"));
    if (!ValidateGeometry(pw_.moe_top)) {
        return AbortLoad();
    }
    const auto t_experts = std::chrono::steady_clock::now();
    if (!qwen_moe::LoadQwenMoeExpertStore(cfg_.model_path, experts_, err)) {
        log::error("[qwen-moe-model] expert store load failed: " + err);
        return AbortLoad();
    }
    log::info(absl::StrCat("[qwen-moe-model] expert store ",
                           MsSince(t_experts), " ms"));
    // Governor accounting (lifecycle-unification II): the dense/pinned
    // weight set is permanent residency.
    if (!AcquirePermanentWeights(pw_.resident_bytes)) return AbortLoad();
    // Cache residency: default to keeping EVERY routed expert device-
    // resident. Full residency removes all per-token H2D staging stalls
    // (measured 42 -> 66 tok/s on RTX 5090 / 35B-A3B); the LRU machinery
    // remains as the graceful-degradation path for tight VRAM.
    // Bound by free VRAM (dense weights are already uploaded at this
    // point), reserving headroom for KV cache, activations and the TC
    // workspace. LSX_QWEN_MOE_EXPERT_CACHE_SLOTS overrides explicitly
    // (parity with LSX_DEEPSEEK_EXPERT_CACHE_SLOTS).
    const int total_experts =
        experts_.num_layers * experts_.num_experts;
    int slots = total_experts;
    bool explicit_slots = false;
    if (const char *env_slots = std::getenv("LSX_QWEN_MOE_EXPERT_CACHE_SLOTS")) {
        const int v = std::atoi(env_slots);
        if (v > 0) {
            slots = std::min(v, total_experts);
            explicit_slots = true;
        }
    }
    // Worst-case per-expert device footprint mirrors the allocator's
    // stride math (pack + scale, arena-aligned).
    size_t max_pack = 0, max_scale = 0;
    for (int l = 0; l < experts_.num_layers; ++l) {
        for (const auto &r :
             experts_.records[static_cast<size_t>(l)]) {
            max_pack = std::max(
                max_pack,
                static_cast<size_t>(r.w1_bytes + r.w2_bytes +
                                    r.w3_bytes));
            max_scale =
                std::max(max_scale,
                         static_cast<size_t>(r.s1_bytes + r.s2_bytes +
                                             r.s3_bytes));
        }
    }
    constexpr size_t kArenaAlign = 256;
    const auto align_up = [](size_t v) {
        return (v + kArenaAlign - 1) & ~(kArenaAlign - 1);
    };
    const size_t per_expert =
        align_up(max_pack) + align_up(max_scale);
    if (!explicit_slots) {
        size_t free_bytes = 0, total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
            // Unified launch-strategy policy (resource_governor.h): maximize
            // expert-cache VRAM utilization with a small central reserve.
            slots = lifecycle::ExpertCacheSlots(
                free_bytes, per_expert, total_experts,
                perf::CurrentKnobs().expert_slots_pct);
            if (slots > 0 && slots < total_experts)
                log::info(absl::StrCat(
                    "[qwen-moe-model] free=", free_bytes >> 20, "MiB, ",
                    slots, "/", total_experts,
                    " experts resident; LRU covers the rest"));
        }
    }
    if (slots <= 0) slots = 4;
    // Adaptive expert-cache lease (soft): accounts the residency budget;
    // a governor refusal is non-fatal — the shrink-retry allocation below
    // is the graceful-degradation path.
    AcquireExpertCacheLease(static_cast<std::uint64_t>(slots) * per_expert,
                            per_expert,
                            static_cast<std::uint64_t>(total_experts) *
                                per_expert);
    // Shrink-retry so an optimistic estimate degrades instead of failing
    // the lifecycle.
    while (slots >= 4 &&
           !qwen_moe::AllocateQwenMoeExpertCache(experts_, slots, cache_,
                                                 err)) {
        slots /= 2;
        cache_ = {};
    }
    if (!cache_.d_base) {
        log::error("[qwen-moe-model] expert cache alloc failed: " + err);
        return AbortLoad();
    }
    // Full residency: prestage EVERY expert + complete the no-handshake
    // mirror. Natural routing alone cannot touch all 40x256 experts within
    // a generation (coupon-collector bound), so the device-driven TC-batch
    // decode path would never engage without this one-time pass.
    if (slots >= total_experts) {
        const auto t_pre = std::chrono::steady_clock::now();
        for (int l = 0; l < experts_.num_layers; ++l) {
            for (int e = 0; e < experts_.num_experts; ++e) {
                if (qwen_moe::EnsureQwenMoeExpertResident(
                        experts_, l, e, cache_, stream(), err) < 0) {
                    log::error("[qwen-moe-model] prestage failed: " + err);
                    return AbortLoad();
                }
            }
        }
        if (stream()) cudaStreamSynchronize(stream());
        qwen_moe::QwenMoeGemvNohandshake(experts_);
        log::info(absl::StrCat("[qwen-moe-model] expert prestage ",
                               MsSince(t_pre), " ms"));
    }
    if (expert_cache_lease_.Valid())
        expert_cache_lease_.Resize(static_cast<std::uint64_t>(slots) *
                                   per_expert);

    // First-request warmup: run a 1-token dummy generation so the lazily
    // built TC fused shared-expert weights, the embed dequant and every
    // kernel's first-launch overhead land on Load instead of the measured
    // TTFT (one-time ~40-80 ms otherwise paid inside the first Infer).
    {
        auto kv = std::make_unique<qwen::QwenKvCache>();
        qwen::QwenGenerationResult warm;
        lsxcommon::sample::ISampler &warm_sampler =
            lsxcommon::sample::DefaultSampler();
        const std::vector<int32_t> warm_prompt = {0};
        const int eos_id = tok_db_ ? tok_db_->eos_id : -1;
        const cudaError_t we = RunUnderResidencyLock([&] {
            return qwen_moe::RunQwenMoeGeneration(
                stream(), pw_, experts_, cache_, scratch_, warm_prompt, 1,
                eos_id, warm, warm_sampler, kv.get());
        });
        if (we != cudaSuccess || !warm.ok)
            log::warn("[qwen-moe-model] warmup skipped: " +
                      (we != cudaSuccess ? cudaGetErrorString(we)
                                         : warm.error_msg));
    }

    resident_bytes_ = pw_.resident_bytes;
    FinishLoad();
    log::info(absl::StrCat("[qwen-moe-model] loaded: resident=",
                           resident_bytes_, " bytes"));
    return true;
}

gpu::InferenceResult QwenMoeModelBase::RunInferShared(const ModelRequest &req)
{
    gpu::InferenceResult r;
    if (!ValidateInferRequest(req, r)) return r;
    if (!loaded_) {
        r.error_msg = "model not loaded";
        return r;
    }
    if (!ContextReady()) {
        r.error_msg = "execution context missing";
        return r;
    }

    // Incremental session policy (mirrors QwenBaseModel).
    const qwen::SessionDecision session = qwen::DecideSession(
        kv_ && kv_->allocated(),
        kv_ ? kv_->last_tok : -1, session_tail_, req.input_ids);
    if (!session.is_continuation) {
        kv_.reset();
        kv_ = std::make_unique<qwen::QwenKvCache>();
        session_tail_.clear();
    }
    const std::vector<int32_t> &gen_input = session.gen_input;

    qwen::QwenGenerationResult gen;
    const int eos_id = tok_db_ ? tok_db_->eos_id : -1;
    cudaError_t e = RunUnderResidencyLock([&] {
        return qwen_moe::RunQwenMoeGeneration(
            stream(), pw_, experts_, cache_, scratch_, gen_input,
            req.max_gen_tokens, eos_id, gen,
            lsxcommon::sample::DefaultSampler(), kv_.get());
    });
    // Uniform failure contract (GLM/Kimi parity): both a CUDA error and a
    // family-level generation failure abort the request and drop the session.
    if (e != cudaSuccess || !gen.ok) {
        r.error_msg = runtime::CombineGenError(e, gen.error_msg);
        r.ok = false;
        kv_.reset();
        session_tail_.clear();
        return r;
    }

    const int kv_slots =
        (kv_ && kv_->allocated())
            ? kv_->kv_rows + 1
            : static_cast<int>(gen.tokens.size()) +
                  static_cast<int>(req.input_ids.size());
    runtime::FillInferenceResult(
        r, {.tokens = &gen.tokens,
            .final_logits = &gen.final_logits,
            .ttft_ms = gen.ttft_ms,
            .total_ms = gen.total_ms,
            .error_msg = &gen.error_msg,
            .eos_terminated = gen.eos_terminated,
            .kv_slots_used = kv_slots,
            .first_maxv = 0.f,
            .resident_bytes = gen.resident_bytes},
        pw_.top.num_layers);
    session_tail_ = r.gen_tokens;
    if (tok_db_ && !r.gen_tokens.empty())
        r.output_text = runtime::DecodeGeneratedText(*tok_db_, r.gen_tokens);
    return r;
}

// Qwen35MoeModel::Unload — canonical ModelBase template method
// (TeardownWeights override above); no per-model override. This also releases
// the tokenizer db like every other family (divergence fixed 2026-08).

gpu::InferenceResult Qwen35MoeModel::Infer(const ModelRequest &req)
{
    // Model independence: this model's own kernel-routing policy for the
    // duration of the call (see LsxFastMathScope in lsx_inference_nvfp4.h).
    const lsx_nvfp4::LsxFastMathScope fm_scope(EffectiveFastMathMode(cfg_));
    return RunInferShared(req);
}

gpu::InferenceResult QwenMoeModelBase::SelfCheck(const ModelRequest &req)
{
    return runtime::RunParisSelfCheck(*this, req);
}

} // namespace lsxcommon
