#include "gemma_base_model.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <string>

#include <absl/strings/str_format.h>

#include "lsxcommon/log.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/prompt_framing.h"
#include "lsxcommon/gemma_tool_format.h"
#include "fast_load_internal.h"

namespace lsxcommon {

// Gemma-4 canonical chat template (chat_template.jinja, generation prompt;
// enable_thinking defaults to false -> the EMPTY thinking channel is part of
// the generation prompt):
//   {{ bos_token }}<|turn>user\n{prompt}<turn|>\n<|turn>model\n
//   <|channel>thought\n<channel|>
// Sentinel strings verified against the real checkpoint tokenizer.json:
// <|turn>=105, <turn|>=106, "\n"=107, <|channel>=100, thought=45518,
// <channel|>=101; "user"/"model" are single vocab keys.
namespace gemma {
const prompt::FramingDescriptor kFraming = {
    {"<|turn>", "user", "\n"},
    {"<turn|>", "\n", "<|turn>", "model", "\n",
     "<|channel>", "thought", "\n", "<channel|>"},
    3, 9,
    /*leading_bos=*/true,
};
bool BuildPromptInputIds(const ArrowTokenizerDb &tok,
                         std::string_view raw_prompt,
                         std::vector<int32_t> &out_ids)
{
    return prompt::ApplyFraming(kFraming, tok, raw_prompt, out_ids);
}
} // namespace gemma

// fast-math policy: per-call fast-math policy (LsxFastMathScope, model-independence invariant).
// seq1_gemv is always on (the weight-streaming GEMV is gemma's
// default decode core); tc_fast_math follows the instance config and routes
// dispatcher dense GEMMs to the tensor-core mxf4nvf4 path when set.

// Base default: no frozen geometry. Concrete models override with the
// compile-time constants gate.
bool GemmaBaseModel::ValidateGeometry(const gemma::GemmaTopology &) const
{
    return true;
}

GemmaBaseModel::GemmaBaseModel(ModelInitConfig cfg)
    : ModelBase(std::move(cfg)) {}

GemmaBaseModel::~GemmaBaseModel() { Unload(); }

void GemmaBaseModel::TeardownWeights()
{
    pw_.Free();
    weights_loaded_ = false;
    kv_.reset();
    session_tail_.clear();
    free_gemma_expert_index(expert_db_);
    expert_db_ready_ = false;
    gemma::DestroyGemmaGenContext(gen_ctx_);
    gen_ctx_ = nullptr;
    TeardownContext();
}

bool GemmaBaseModel::Load()
{
    Unload();  // idempotent re-load support (uniform lifecycle policy)
    tokenizer_optional_ = true;
    if (!BeginLoad(cfg_.model_path)) return false;
    double pinned_ms = 0.0;
    const auto t_pinned = std::chrono::steady_clock::now();
    if (!EnsureContext()) {
        lsxcommon::log::error("[gemma-model] stream create failed");
        return AbortLoad();
    }
    cudaError_t lrc = gemma::LoadGemmaPinnedWeights(
        cfg_.model_path, pw_, stream());
    fastload::StreamDrain{stream()};
    const bool ok = (lrc == cudaSuccess);
    weights_loaded_ = ok;
    if (!ok) {
        lsxcommon::log::error(
            "[gemma-model] LoadGemmaPinnedWeights failed: rc=" +
            std::to_string(static_cast<int>(lrc)) + " (" +
            cudaGetErrorString(lrc) + ")");
        return AbortLoad();
    }
    resident_bytes_ = pw_.resident_bytes;
    vram_ = pw_.top.vram;
    // Governor accounting (lifecycle-unification II): the pinned weight set
    // is permanent residency. The routed-expert device cache is allocated
    // lazily during generation and stays budgeted by the unified slot
    // policy (ComputeGemmaResidentSlots / lifecycle::ExpertCacheSlots).
    if (!AcquirePermanentWeights(pw_.resident_bytes)) return AbortLoad();
    pinned_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t_pinned)
                    .count();
    lsxcommon::log::info(
        "[gemma-model] pinned weights loaded: " +
        std::to_string(pw_.layers.size()) + " layers (" +
        std::to_string(pw_.resident_bytes) + " bytes resident)");
    // Geometry gate BEFORE FinishLoad (lifecycle-unification 2026-08): a
    // failed Load must leave loaded_ == false, otherwise Infer() could run
    // on an unvalidated topology.
    if (!ValidateGeometry(pw_.top)) {
        lsxcommon::log::error("[gemma-model] geometry validation failed for " +
                              cfg_.model_path);
        return AbortLoad();
    }
    FinishLoad();
    if (!expert_db_ready_) {
        const auto t_index = std::chrono::steady_clock::now();
        expert_db_ = gemma::load_gemma_expert_index(
            cfg_.model_path, static_cast<int>(pw_.layers.size()),
            pw_.top.error_msg.empty() ? pw_.top.num_experts : 1);
        expert_db_ready_ = true;
        lsxcommon::log::info(
            "[gemma-model] pinned load " +
            absl::StrFormat("%.1f", pinned_ms) + " ms, expert index " +
            absl::StrFormat("%.1f", std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_index)
                .count()) + " ms (" +
            std::to_string(expert_db_.parse_ok) + "/" +
            std::to_string(expert_db_.num_layers *
                           expert_db_.experts_per_layer) +
            " files parsed)");
    }
    return true;
}

gpu::InferenceResult GemmaBaseModel::Infer(const ModelRequest &req)
{
    gpu::InferenceResult r;
    if (!ValidateInferRequest(req, r)) return r;

    // Model-independence invariant: per-call fast-math policy. seq1_gemv is
    // always on for gemma (the weight-streaming GEMV is its default decode
    // path, bit-exact class); tc_fast_math follows the tri-state config and
    // routes the dispatcher's dense GEMMs to the tensor-core mxf4nvf4 path
    // when forced on.
    const FastMathMode fm = EffectiveFastMathMode(cfg_);
    const lsx_nvfp4::LsxFastMathScope fm_scope(
        fm == FastMathMode::kAuto ? true : fm == FastMathMode::kOn,
        /*seq1_gemv=*/true);

    if (!EnsureContext()) {
        r.error_msg = "cuda stream create failed";
        return r;
    }
    cudaStream_t st = stream();

    // Session policy: every request runs a fresh prefill+decode (the
    // generation loop always prefills its input); session_tail_ tracks the
    // last generated tail for orchestrator introspection. The whole request
    // runs inside ONE shared-lock scope — creating, allocating, decoding
    // through and publishing the KV state under the lock — so two concurrent
    // Infer calls can never free the cache the other one reads.
    gemma::GemmaExpertDb &expert_db = expert_db_;
    gemma::GemmaGenerationResult gen;
    if (!gen_ctx_) gen_ctx_ = gemma::CreateGemmaGenContext();
    lsxcommon::GenerationStream gs;
    gs.on_token = req.token_callback ? &req.token_callback : nullptr;
    gs.abort_flag = req.abort_flag;
    return RunUnderResidencyLock([&]() -> gpu::InferenceResult {
        kv_.reset();
        kv_ = std::make_unique<gemma::GemmaKvCache>();
        session_tail_.clear();
        const std::vector<int32_t> &gen_input = req.input_ids;

        // Capacity: prompt + generation + slack.
        const int need = static_cast<int>(gen_input.size()) +
                         std::max(1, req.max_gen_tokens) + 8;
        const cudaError_t ae = kv_->Allocate(pw_, need);
        if (ae != cudaSuccess) {
            r.error_msg = "kv alloc failed";
            return r;
        }

        std::vector<int> eos_ids;
        if (req.ignore_eos || !tok_db_)
            eos_ids = {lsxcommon::kIgnoreEosTokenId};
        else if (!pw_.top.eos_ids.empty())
            eos_ids = pw_.top.eos_ids;
        else
            eos_ids = {tok_db_->eos_id};
        const cudaError_t ge = gemma::RunGemmaGeneration(
            st, pw_, expert_db, gen_input, req.max_gen_tokens, eos_ids,
            pw_.top.error_msg.empty() ? pw_.top.final_logit_softcapping : 0.f,
            gen, *kv_, gen_ctx_, sample::SamplerFor(req.sampling), &gs);
        if (ge != cudaSuccess)
            lsxcommon::log::error(
                "[gemma-model] RunGemmaGeneration rc=" +
                std::to_string(static_cast<int>(ge)) + " (" +
                cudaGetErrorString(ge) + ") gen_err='" + gen.error_msg + "'");
        if (ge != cudaSuccess || !gen.ok) {
            r.error_msg = runtime::CombineGenError(ge, gen.error_msg);
            r.ok = false;
            kv_.reset();
            session_tail_.clear();
            return r;
        }

        runtime::FillInferenceResult(
            r, {.tokens = &gen.tokens,
                .final_logits = &gen.final_logits,
                .ttft_ms = gen.ttft_ms,
                .total_ms = gen.total_ms,
                .error_msg = &gen.error_msg,
                .eos_terminated = gen.eos_terminated,
                .kv_slots_used = static_cast<int>(gen_input.size()) +
                                 static_cast<int>(gen.tokens.size()),
                .first_maxv = 0.f,
                .resident_bytes = resident_bytes_},
            static_cast<int>(pw_.layers.size()));
        session_tail_ = r.gen_tokens;
        if (tok_db_ && !r.gen_tokens.empty())
            r.output_text = runtime::DecodeGeneratedText(*tok_db_, r.gen_tokens);
        return r;
    });
}

// GemmaBaseModel::Unload — canonical ModelBase template method
// (TeardownWeights override above); no per-model override.

gpu::InferenceResult GemmaBaseModel::SelfCheck(const ModelRequest &req)
{
    // Gemma semantics expressed through the unified harness options:
    // chat framing + leading <bos>, forced >= 8 generated tokens, and a
    // 12-token window in which "Paris" must appear (this checkpoint greedily
    // begins its answer with "The capital of France is **Paris**", verified
    // against HF transformers greedy decode with identical framing).
    runtime::ParisSelfCheckOptions opts;
    opts.frame_prompt = true;
    opts.min_tokens = 8;
    opts.answer_window = 12;
    opts.substring_match = true;
    return runtime::RunParisSelfCheck(*this, req, opts);
}

bool GemmaBaseModel::BuildConversationInputIds(const ArrowTokenizerDb &tok,
                                               const ChatConversation &conv,
                                               std::vector<int32_t> &out_ids) const
{
    std::string text;
    if (!RenderGemmaConversation(conv, text)) return false;
    if (!AppendGemmaChatIds(tok, text, out_ids))
        return BuildPromptInputIds(tok, text, out_ids);
    return true;
}

bool GemmaBaseModel::ParseToolCalls(const ArrowTokenizerDb &tok,
                                    const std::vector<int32_t> &gen_tokens,
                                    std::vector<ToolCall> &out_calls) const
{
    return ParseGemmaTokenCalls(tok, gen_tokens, out_calls);
}

} // namespace lsxcommon
