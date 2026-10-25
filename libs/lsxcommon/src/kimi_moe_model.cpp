#include "kimi_moe_model.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <filesystem>
#include <shared_mutex>

#include "kimi_moe_kernels.h"
#include "lsxcommon/kimi_moe_convert.h"
#include "lsxcommon/kimi_moe_tool_format.h"
#include "lsxcommon/tokenizer.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/fast_math_mode.h"
#include "lsxcommon/gpu_perf.h"
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
    kimi::FreeKimiSharedExpertArena(expert_cache_);
    kimi::free_kimi_expert_index(experts_);
    pw_.Free();
    TeardownContext();
}

bool KimiK27CodeModel::Load()
{
    Unload();  // idempotent re-load support
    if (!BeginLoad(cfg_.model_path)) return false;

    namespace fs = std::filesystem;

    if (!EnsureContext()) {
        log::error("[kimi-model] stream init failed");
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
        // Legacy bool (cfg_.fast_math) is never set by the CLI/engine — the
        // tri-state below is the routing source. kAuto follows the per-GPU
        // perf policy resolved at engine init (tc_fast_math_default).
        const FastMathMode eff = EffectiveFastMathMode(cfg_);
        lsxcommon::GenerationStream gs;
        gs.on_token = req.token_callback ? &req.token_callback : nullptr;
        gs.abort_flag = req.abort_flag;
        const bool fast_math =
            eff == FastMathMode::kOn
                || (eff == FastMathMode::kAuto
                    && perf::CurrentKnobs().tc_fast_math_default);
        return kimi::RunKimiGeneration(stream(), pw_, experts_,
                                       req.input_ids, req.max_gen_tokens,
                                       gen, &expert_cache_,
                                       req.ignore_eos ? lsxcommon::kIgnoreEosTokenId : Tokenizer().eos_id, fast_math,
                                       lsxcommon::sample::SamplerFor(req.sampling),
                                       &gs);
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

bool KimiK27CodeModel::BuildConversationInputIds(
    const ArrowTokenizerDb &tok, const ChatConversation &conv,
    std::vector<int32_t> &out_ids) const
{
    std::string text;
    if (!RenderKimiConversation(conv, text)) return false;
    // A tokenizer without the Kimi sentinels falls back to the generic
    // transcript (keeps the bridge usable on any fixture).
    if (arrow_tokenizer_lookup(tok, "<|im_user|>", 11) < 0)
        return BuildPromptInputIds(tok, text, out_ids);
    return AppendKimiIds(tok, text, out_ids);
}

bool KimiK27CodeModel::ParseToolCalls(
    const ArrowTokenizerDb &tok, const std::vector<int32_t> &gen_tokens,
    std::vector<ToolCall> &out_calls) const
{
    out_calls.clear();
    if (gen_tokens.empty()) return true;
    std::string text(4096, '\0');
    size_t cap = text.size();
    for (int attempt = 0; attempt < 8; ++attempt) {
        const int rc = arrow_tokenizer_decode(
            tok, gen_tokens.data(), gen_tokens.size(), text.data(), &cap);
        if (rc == 0) {
            text.resize(cap);
            out_calls = ParseKimiToolCalls(text);
            return true;
        }
        if (rc == -2 && cap > text.size() && cap <= (1u << 20)) {
            text.resize(cap);
            continue;
        }
        break;
    }
    return false;
}

// KimiK27CodeModel::Unload — canonical ModelBase template method
// (TeardownWeights override above); no per-model override.

gpu::InferenceResult KimiK27CodeModel::SelfCheck(const ModelRequest &req)
{
    return runtime::RunParisSelfCheck(*this, req);
}

} // namespace lsxcommon
