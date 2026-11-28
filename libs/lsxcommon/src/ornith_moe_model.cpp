#include "ornith_moe_model.h"
#include "ornith_moe_constants.h"

#include <absl/strings/str_cat.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>

#include "lsxcommon/log.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/prompt_framing.h"
#include "lsxcommon/resource_governor.h"
#include "lsxcommon/gpu_perf.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/qwen_tool_format.h"

namespace lsxcommon {

namespace {
double MsSince(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0)
        .count();
}
} // namespace

const std::string OrnithMoeModel::kType = "ornith-1.5-35b-a3b";

OrnithMoeModelBase::OrnithMoeModelBase(ModelInitConfig cfg)
    : ModelBase{std::move(cfg)}
{
    // The TC W4A4 toggle is process-wide; the dense Qwen models flip it in
    // QwenBaseModel, but this MoE family derives from ModelBase directly and
    // would silently run every GEMM on the scalar path without this.
    const FastMathMode fm = EffectiveFastMathMode(cfg_);
    if (fm != FastMathMode::kAuto) {
        lsx_nvfp4::SetLsxTcFastMath(fm == FastMathMode::kOn);
        lsx_nvfp4::SetLsxSeq1Gemv(fm == FastMathMode::kOn);
    } else {
        // Engine auto policy: mirror the per-GPU knob set resolved at
        // InitializeEngine (legacy zeros when no engine init — preserves
        // the bit-exact default for direct-factory users).
        const perf::LsxPerfKnobs knobs = perf::CurrentKnobs();
        lsx_nvfp4::SetLsxTcFastMath(knobs.tc_fast_math_default);
        lsx_nvfp4::SetLsxSeq1Gemv(knobs.seq1_gemv_default);
    }
}

OrnithMoeModel::OrnithMoeModel(ModelInitConfig cfg)
    : OrnithMoeModelBase{std::move(cfg)}
{
}

OrnithMoeModel::~OrnithMoeModel() { Unload(); }

void OrnithMoeModelBase::TeardownWeights()
{
    ornith::FreeOrnithMoeGenScratch(scratch_);
    moe::FreeMoEExpertCache(cache_);
    experts_.Free();
    ornith::FreeOrnithMoePinnedWeights(pw_);
    kv_.reset();
    session_tail_.clear();
    TeardownContext();
}

bool OrnithMoeModel::ValidateGeometry(
    const ornith::OrnithMoeTopology &top) const
{
    using namespace ornith::frozen;
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
    if (!CheckGeometryFields("ornith-moe-model", fields, 10)) return false;
    for (int l = 0; l < top.num_layers; ++l) {
        const std::string want =
            (l % kFullAttnInterval == kFullAttnInterval - 1) ? "full_attention"
                                                             : "linear_attention";
        if (top.layer_types[static_cast<size_t>(l)] != want) {
            log::error(absl::StrCat("[ornith-moe-model] layer_types[", l,
                                    "] != ", want));
            return false;
        }
    }
    return true;
}

bool OrnithMoeModelBase::Load()
{
    Unload();  // idempotent re-load support
    if (!BeginLoad(cfg_.model_path)) return false;

    if (!EnsureContext()) {
        log::error("[ornith-moe-model] stream init failed");
        return AbortLoad();
    }

    std::string err;
    const auto t_dense = std::chrono::steady_clock::now();
    if (!ornith::LoadOrnithMoeDenseWeights(cfg_.model_path, pw_, stream(), err)) {
        log::error("[ornith-moe-model] dense load failed: " + err);
        return AbortLoad();
    }
    log::info(absl::StrCat("[ornith-moe-model] dense load ", MsSince(t_dense),
                           " ms"));
    if (!ValidateGeometry(pw_.moe_top)) {
        return AbortLoad();
    }
    const auto t_experts = std::chrono::steady_clock::now();
    if (!moe::LoadMoEExpertStore(cfg_.model_path, experts_, err)) {
        log::error("[ornith-moe-model] expert store load failed: " + err);
        return AbortLoad();
    }
    log::info(absl::StrCat("[ornith-moe-model] expert store ",
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
    // workspace. LSX_ORNITH_MOE_EXPERT_CACHE_SLOTS overrides explicitly
    // (parity with LSX_DEEPSEEK_EXPERT_CACHE_SLOTS).
    const int total_experts =
        experts_.num_layers * experts_.num_experts;
    int slots = total_experts;
    bool explicit_slots = false;
    if (const char *env_slots = std::getenv("LSX_ORNITH_MOE_EXPERT_CACHE_SLOTS")) {
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
                perf::CurrentKnobs().expert_slots_pct,
                lifecycle::ExpertCacheReserveBytes(
                    ResourceGovernor::Instance().hardware()));
            if (slots > 0 && slots < total_experts)
                log::info(absl::StrCat(
                    "[ornith-moe-model] free=", free_bytes >> 20, "MiB, ",
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
           !moe::AllocateMoEExpertCache(experts_, slots, cache_,
                                                 err)) {
        slots /= 2;
        cache_ = {};
    }
    if (!cache_.d_base) {
        log::error("[ornith-moe-model] expert cache alloc failed: " + err);
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
                if (moe::EnsureMoEExpertResident(
                        experts_, l, e, cache_, stream(), err) < 0) {
                    log::error("[ornith-moe-model] prestage failed: " + err);
                    return AbortLoad();
                }
            }
        }
        if (stream()) cudaStreamSynchronize(stream());
        moe::MoEGemvNohandshake(experts_);
        log::info(absl::StrCat("[ornith-moe-model] expert prestage ",
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
        auto kv = std::make_unique<qwen::KvCache>();
        qwen::GenerationResult warm;
        lsxcommon::sample::ISampler &warm_sampler =
            lsxcommon::sample::DefaultSampler();
        const std::vector<int32_t> warm_prompt = {0};
        const int eos_id = tok_db_ ? tok_db_->eos_id : -1;
        const cudaError_t we = RunUnderResidencyLock([&] {
            return ornith::RunOrnithMoeGeneration(
                stream(), pw_, experts_, cache_, scratch_, warm_prompt, 1,
                eos_id, warm, warm_sampler, kv.get());
        });
        if (we != cudaSuccess || !warm.ok)
            log::warn("[ornith-moe-model] warmup skipped: " +
                      (we != cudaSuccess ? cudaGetErrorString(we)
                                         : warm.error_msg));
    }

    resident_bytes_ = pw_.resident_bytes;
    FinishLoad();
    log::info(absl::StrCat("[ornith-moe-model] loaded: resident=",
                           resident_bytes_, " bytes"));
    return true;
}

gpu::InferenceResult OrnithMoeModelBase::RunInferShared(const ModelRequest &req)
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

    lsxcommon::GenerationStream gs;
    gs.on_token = req.token_callback ? &req.token_callback : nullptr;
    gs.abort_flag = req.abort_flag;
    const int eos_id = (req.ignore_eos || !tok_db_) ? lsxcommon::kIgnoreEosTokenId : tok_db_->eos_id;
    // The whole request runs inside ONE shared-lock scope. Incremental
    // session policy (mirrors QwenBaseModel): a request whose input_ids
    // match the previous call's generated tail continues the KV cache;
    // anything else starts fresh. Creating, decoding through and publishing
    // that state under the lock. The lock is shared, so it orders this request against Unload()'s unique
    // lock and does NOT serialise Infer against Infer - two concurrent
    // Infer calls on one instance both hold it. One Infer per instance at
    // a time is a precondition (OPEN_TASKS E1b), and that is what keeps
    // them from freeing each other's cache.
    return RunUnderResidencyLock([&]() -> gpu::InferenceResult {
        const qwen::SessionDecision session = qwen::DecideSession(
            kv_ && kv_->allocated(),
            kv_ ? kv_->last_tok : -1, session_tail_, req.input_ids);
        if (!session.is_continuation) {
            kv_.reset();
            kv_ = std::make_unique<qwen::KvCache>();
            session_tail_.clear();
        }

        qwen::GenerationResult gen;
        const cudaError_t e = ornith::RunOrnithMoeGeneration(
            stream(), pw_, experts_, cache_, scratch_, session.gen_input,
            req.max_gen_tokens, eos_id, gen,
            lsxcommon::sample::SamplerFor(req.sampling), kv_.get(), &gs);
        // Uniform failure contract (GLM/Kimi parity): both a CUDA error and a
        // family-level generation failure abort the request and drop the
        // session.
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
    });
}

// OrnithMoeModel::Unload — canonical ModelBase template method
// (TeardownWeights override above); no per-model override. This also releases
// the tokenizer db like every other family (divergence fixed 2026-08).

gpu::InferenceResult OrnithMoeModel::Infer(const ModelRequest &req)
{
    // Model independence: this model's own kernel-routing policy for the
    // duration of the call (see LsxFastMathScope in lsx_inference_nvfp4.h).
    const lsx_nvfp4::LsxFastMathScope fm_scope(EffectiveFastMathMode(cfg_));
    return RunInferShared(req);
}

gpu::InferenceResult OrnithMoeModelBase::SelfCheck(const ModelRequest &req)
{
    return runtime::RunParisSelfCheck(*this, req);
}

bool OrnithMoeModelBase::BuildConversationInputIds(
    const ArrowTokenizerDb &tok, const ChatConversation &conv,
    std::vector<int32_t> &out_ids) const
{
    // The checkpoint requires a leading system turn (see BuildPromptInputIds);
    // inject a default one when the caller omitted it.
    ChatConversation c = conv;
    if (c.messages.empty() || c.messages.front().role != "system")
        c.messages.insert(c.messages.begin(),
                          {"system", "You are a helpful assistant.", "", {}});
    std::string text;
    // Ornith is a reasoning fine-tune: let it think before the tool call.
    if (!RenderQwenConversation(c, text, /*enable_thinking=*/true))
        return false;
    if (!AppendQwenChatIds(tok, text, out_ids))
        return BuildPromptInputIds(tok, text, out_ids);
    return true;
}

bool OrnithMoeModelBase::ParseToolCalls(
    const ArrowTokenizerDb &tok, const std::vector<int32_t> &gen_tokens,
    std::vector<ToolCall> &out_calls) const
{
    return ParseQwenTokenCalls(tok, gen_tokens, out_calls);
}

// ── Ornith chat framing (reasoning model) ─────────────────────────────────

bool OrnithMoeModel::BuildPromptInputIds(const ArrowTokenizerDb &tok,
                                         std::string_view raw_prompt,
                                         std::vector<int32_t> &out_ids) const
{
    // Render the single turn through the checkpoint's Qwen chat template, like
    // NemotronModelBase — NOT a hand-rolled role-word frame. This reasoning
    // fine-tune degenerates on a frame that opens at the user turn (repetition
    // collapse), so the leading system turn its RL harness always injects is
    // required. A prefilled empty <think> block selects the
    // enable_thinking=false control path so the model answers directly.
    ChatConversation conv;
    conv.messages.push_back(
        {"system", "You are a helpful assistant.", "", {}});
    conv.messages.push_back({"user", std::string(raw_prompt), "", {}});
    std::string frame;
    if (!RenderQwenConversation(conv, frame, /*enable_thinking=*/false))
        return false;
    if (AppendQwenChatIds(tok, frame, out_ids)) return true;
    return prompt::EncodeRawPrompt(tok, frame, out_ids);
}

gpu::InferenceResult OrnithMoeModel::SelfCheck(const ModelRequest &req)
{
    // Ornith is a reasoning fine-tune: the raw control prompt does not answer
    // directly. Frame with the checkpoint's own ChatML template (prefilled
    // empty think block) and accept "Paris" within a short answer window
    // (Model Ingestion Protocol step 8 deviation, recorded in the design).
    const runtime::ParisSelfCheckOptions opts{
        .frame_prompt = true,
        .min_tokens = 16,
        .answer_window = 16,
        .substring_match = true,
    };
    return runtime::RunParisSelfCheck(*this, req, opts);
}

} // namespace lsxcommon
