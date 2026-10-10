#include "lsxcommon/qwen_base_model.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "lsxcommon/log.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/qwen_dense_inference_pipeline.h"
#include "lsxcommon/qwen_inference.h"
#include "lsxcommon/qwen_inference_load.h"
#include "fast_load_internal.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/tokenizer.h"
#include "lsxcommon/qwen_tool_format.h"

#include <absl/strings/str_cat.h>
#include <cstdlib>

namespace lsxcommon {

QwenBaseModel::QwenBaseModel(ModelInitConfig cfg)
    : ModelBase(std::move(cfg)) {}

QwenBaseModel::~QwenBaseModel() { Unload(); }

void QwenBaseModel::TeardownWeights()
{
    pw_.Free();
    weights_loaded_ = false;
    kv_.reset();
    session_tail_.clear();
    TeardownContext();
}

bool QwenBaseModel::Load()
{
    Unload();  // idempotent re-load support (uniform lifecycle policy)
    tokenizer_optional_ = true;
    // Blackwell tensor-core fast path (audit Phase 3): opt-in via
    // ModelInitConfig::fast_math. Default off = bit-exact scalar path.
    const FastMathMode fm = EffectiveFastMathMode(cfg_);
    if (fm != FastMathMode::kAuto) {
        qwen::SetQwenTcFastMath(fm == FastMathMode::kOn);
        qwen::SetQwenSeq1Gemv(fm == FastMathMode::kOn);
    }
    if (!BeginLoad(cfg_.model_path)) return false;
    if (!EnsureContext()) {
        lsxcommon::log::error("[qwen-model] stream create failed");
        return AbortLoad();
    }
    const bool ok = qwen::LoadQwenPinnedWeights(cfg_.model_path, pw_,
                                                stream());
    fastload::StreamDrain{stream()};
    weights_loaded_ = ok;
    if (!ok) {
        lsxcommon::log::error(
            "[qwen-model] LoadQwenPinnedWeights failed");
        return AbortLoad();
    }
    // Dense residency (spec 3.3): the entire model is resident after
    // Load(), so ResidentWeightBytes() reports the full persistent set.
    resident_bytes_ = pw_.resident_bytes;
    vram_ = pw_.top.vram;
    // Governor accounting (lifecycle-unification II): dense models lease
    // their full permanent residency too.
    if (!AcquirePermanentWeights(pw_.resident_bytes)) return AbortLoad();
    lsxcommon::log::info(absl::StrCat(
        "[qwen-model] pinned weights loaded: ", pw_.layers.size(),
        " layers (", pw_.resident_bytes, " bytes resident)"));
    // Geometry gate BEFORE FinishLoad (lifecycle-unification 2026-08): a
    // failed Load must leave loaded_ == false, otherwise Infer() could run
    // on an unvalidated topology.
    if (!ValidateGeometry(pw_.top)) {
        lsxcommon::log::error(absl::StrCat(
            "[qwen-model] geometry validation failed for ", cfg_.model_path));
        return AbortLoad();
    }
    FinishLoad();
    return true;
}

gpu::InferenceResult QwenBaseModel::Infer(const ModelRequest &req)
{
    gpu::InferenceResult r;
    if (!ValidateInferRequest(req, r)) return r;

    // Model independence: this model's own kernel-routing policy for the
    // duration of the call (see LsxFastMathScope in lsx_inference_nvfp4.h).
    const lsx_nvfp4::LsxFastMathScope fm_scope(EffectiveFastMathMode(cfg_));

    // Persistent execution context: created once in Load(), reused for
    // every request (no temp-per-call stream churn).
    if (!EnsureContext()) {
        r.error_msg = "cuda init failed";
        return r;
    }
    cudaStream_t st = stream();

    lsxcommon::GenerationStream gs;
    gs.on_token = req.token_callback ? &req.token_callback : nullptr;
    gs.abort_flag = req.abort_flag;

    // The whole request runs inside ONE shared-lock scope. Incremental
    // session policy (spec item 15): a request whose input_ids exactly match
    // the previous call's generated tail continues the KV cache from
    // {kv_->last_tok}; anything else starts a fresh one. Creating, decoding
    // through and publishing that state all happen under the lock, so two
    // concurrent Infer calls can never free the cache the other one reads.
    return RunUnderResidencyLock([&]() -> gpu::InferenceResult {
        const qwen::SessionDecision session = qwen::DecideSession(
            kv_ && kv_->allocated(),
            kv_ ? kv_->last_tok : -1, session_tail_, req.input_ids);
        if (!session.is_continuation) {
            kv_.reset();
            kv_ = std::make_unique<qwen::QwenKvCache>();
            session_tail_.clear();
        }

        qwen::QwenGenerationResult gen;
        const int eos_id = (req.ignore_eos || !tok_db_)
                               ? lsxcommon::kIgnoreEosTokenId
                               : tok_db_->eos_id;
        const cudaError_t e = qwen::RunQwenGeneration(
            st, pw_, session.gen_input, req.max_gen_tokens, eos_id, gen,
            lsxcommon::sample::DefaultSampler(), kv_.get(), &gs);
        if (e != cudaSuccess) {
            r.error_msg = runtime::CombineGenError(e, gen.error_msg);
            r.ok = false;
            // A failed continuation may have advanced kv_rows mid-stream; drop
            // the session so the next call starts fresh.
            kv_.reset();
            session_tail_.clear();
            return r;
        }

        // Occupancy (spec item 15): the incremental cache holds prompt +
        // generated rows once the pending last-token row is counted
        // (kv_rows + 1). The fallback (non-incremental) mirrors the same
        // prompt + gen semantic.
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

// QwenBaseModel::Unload — canonical ModelBase template method
// (TeardownWeights override above); no per-model override.

gpu::InferenceResult QwenBaseModel::SelfCheck(const ModelRequest &req)
{
    return runtime::RunParisSelfCheck(*this, req);
}

bool QwenBaseModel::ValidateGeometry(const qwen::QwenTopology &) const { return true; }

bool QwenBaseModel::BuildConversationInputIds(const ArrowTokenizerDb &tok,
                                              const ChatConversation &conv,
                                              std::vector<int32_t> &out_ids) const
{
    std::string text;
    if (!RenderQwenConversation(conv, text)) return false;
    if (!AppendQwenChatIds(tok, text, out_ids))
        return BuildPromptInputIds(tok, text, out_ids);
    return true;
}

bool QwenBaseModel::ParseToolCalls(const ArrowTokenizerDb &tok,
                                   const std::vector<int32_t> &gen_tokens,
                                   std::vector<ToolCall> &out_calls) const
{
    return ParseQwenTokenCalls(tok, gen_tokens, out_calls);
}

} // namespace lsxcommon
