#include "gemma_dense_model.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <string>

#include <absl/strings/str_format.h>

#include "fast_load_internal.h"
#include "lsxcommon/gemma_tool_format.h"
#include "lsxcommon/log.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/prompt_framing.h"

namespace lsxcommon {

const std::string Gemma4_31BDenseModel::kType = "gemma-4-31b-it";

// fast-math policy: per-call fast-math policy (LsxFastMathScope, model-independence
// invariant). seq1_gemv follows the instance config and routes the neutral
// dispatcher's dense GEMVs to the streaming kernel; tc_fast_math follows the
// tri-state config and routes eligible dense GEMMs to the TC mxf4nvf4 path.

Gemma4_31BDenseModel::Gemma4_31BDenseModel(ModelInitConfig cfg)
    : ModelBase(std::move(cfg)) {}

Gemma4_31BDenseModel::~Gemma4_31BDenseModel() { Unload(); }

void Gemma4_31BDenseModel::TeardownWeights()
{
    pw_.Free();
    weights_loaded_ = false;
    kv_.reset();
    session_tail_.clear();
    gemma_dense::DestroyGemmaDenseGenContext(gen_ctx_);
    gen_ctx_ = nullptr;
    TeardownContext();
}

bool Gemma4_31BDenseModel::ValidateGeometry(
    const gemma_dense::GemmaDenseTopology &top) const
{
    if (!top.error_msg.empty()) return false;
    if (SyntheticGeometryAllowed()) return true;
    const GeoFieldCheck fields[] = {
        {"hidden_size", gemma_dense::frozen::kHiddenSize, top.hidden_size},
        {"num_layers", gemma_dense::frozen::kNumLayers, top.num_layers},
        {"num_attention_heads", gemma_dense::frozen::kNumHeads,
         top.num_attention_heads},
        {"num_key_value_heads", gemma_dense::frozen::kNumKVHeads,
         top.num_key_value_heads},
        {"num_global_key_value_heads", gemma_dense::frozen::kNumGlobalKVHeads,
         top.num_global_key_value_heads},
        {"head_dim", gemma_dense::frozen::kHeadDim, top.head_dim},
        {"global_head_dim", gemma_dense::frozen::kGlobalHeadDim,
         top.global_head_dim},
        {"intermediate_size", gemma_dense::frozen::kIntermediateSize,
         top.intermediate_size},
        {"vocab_size", gemma_dense::frozen::kVocabSize, top.vocab_size},
    };
    return CheckGeometryFields("gemma-4-31b-it", fields, 9);
}

bool Gemma4_31BDenseModel::Load()
{
    Unload();  // idempotent re-load support
    tokenizer_optional_ = true;
    if (!BeginLoad(cfg_.model_path)) return false;
    const auto t_pinned = std::chrono::steady_clock::now();
    if (!EnsureContext()) {
        lsxcommon::log::error("[gemma-dense-model] stream create failed");
        return AbortLoad();
    }
    const cudaError_t lrc =
        gemma_dense::LoadGemmaDensePinnedWeights(cfg_.model_path, pw_, stream());
    fastload::StreamDrain{stream()};
    const bool ok = (lrc == cudaSuccess);
    weights_loaded_ = ok;
    if (!ok) {
        lsxcommon::log::error(
            "[gemma-dense-model] LoadGemmaDensePinnedWeights failed: rc=" +
            std::to_string(static_cast<int>(lrc)) + " (" +
            cudaGetErrorString(lrc) + ")");
        return AbortLoad();
    }
    resident_bytes_ = pw_.resident_bytes;
    vram_ = pw_.top.vram;
    if (!AcquirePermanentWeights(pw_.resident_bytes)) return AbortLoad();
    lsxcommon::log::info(
        "[gemma-dense-model] pinned weights loaded: " +
        std::to_string(pw_.layers.size()) + " layers (" +
        std::to_string(pw_.resident_bytes) + " bytes resident)");
    // Geometry gate BEFORE FinishLoad: a failed Load must leave loaded_ false.
    if (!ValidateGeometry(pw_.top)) {
        lsxcommon::log::error(
            "[gemma-dense-model] geometry validation failed for " +
            cfg_.model_path);
        return AbortLoad();
    }
    FinishLoad();
    lsxcommon::log::info(
        "[gemma-dense-model] pinned load " +
        absl::StrFormat("%.1f", std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - t_pinned)
                                    .count()) +
        " ms");
    return true;
}

gpu::InferenceResult Gemma4_31BDenseModel::Infer(const ModelRequest &req)
{
    gpu::InferenceResult r;
    if (!ValidateInferRequest(req, r)) return r;

    // Model-independence invariant: this model's OWN kernel-routing policy for
    // the duration of the call.
    const lsx_nvfp4::LsxFastMathScope fm_scope(EffectiveFastMathMode(cfg_));

    if (!EnsureContext()) {
        r.error_msg = "cuda stream create failed";
        return r;
    }
    cudaStream_t st = stream();

    // Streaming + cancellation handle (one Emit per decoded token; the abort
    // flag is honoured between decode steps).
    lsxcommon::GenerationStream gs;
    gs.on_token = req.token_callback ? &req.token_callback : nullptr;
    gs.abort_flag = req.abort_flag;

    // The whole request runs inside ONE shared-lock scope: creating,
    // allocating, decoding through and publishing the KV state under the lock.
    // The lock is shared, so it orders this request against Unload()'s unique
    // lock and does NOT serialise Infer against Infer - two concurrent
    // Infer calls on one instance both hold it. One Infer per instance at
    // a time is a precondition (OPEN_TASKS E1b), and that is what keeps
    // them from freeing each other's cache.
    return RunUnderResidencyLock([&]() -> gpu::InferenceResult {
        kv_.reset();
        kv_ = std::make_unique<gemma_dense::GemmaDenseKvCache>();
        session_tail_.clear();
        if (!gen_ctx_) gen_ctx_ = gemma_dense::CreateGemmaDenseGenContext();

        const int need = static_cast<int>(req.input_ids.size()) +
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

        gemma_dense::GemmaDenseGenerationResult gen;
        const cudaError_t ge = gemma_dense::RunGemmaDenseGeneration(
            st, pw_, req.input_ids, req.max_gen_tokens, eos_ids, gen, *kv_,
            gen_ctx_, &gs);
        if (ge != cudaSuccess || !gen.ok) {
            lsxcommon::log::error(
                "[gemma-dense-model] RunGemmaDenseGeneration rc=" +
                std::to_string(static_cast<int>(ge)) + " (" +
                cudaGetErrorString(ge) + ") gen_err='" + gen.error_msg + "'");
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
                .kv_slots_used = static_cast<int>(req.input_ids.size()) +
                                 static_cast<int>(gen.tokens.size()),
                .first_maxv = 0.f,
                .resident_bytes = gen.resident_bytes},
            static_cast<int>(pw_.layers.size()));
        session_tail_ = r.gen_tokens;
        if (tok_db_ && !r.gen_tokens.empty())
            r.output_text = runtime::DecodeGeneratedText(*tok_db_, r.gen_tokens);
        return r;
    });
}

gpu::InferenceResult Gemma4_31BDenseModel::SelfCheck(const ModelRequest &req)
{
    // Gemma semantics: chat framing + leading <bos>, forced >= 8 generated
    // tokens, and a 12-token window in which "Paris" must appear.
    runtime::ParisSelfCheckOptions opts;
    opts.frame_prompt = true;
    opts.min_tokens = 8;
    opts.answer_window = 12;
    opts.substring_match = true;
    return runtime::RunParisSelfCheck(*this, req, opts);
}

bool Gemma4_31BDenseModel::BuildConversationInputIds(
    const ArrowTokenizerDb &tok, const ChatConversation &conv,
    std::vector<int32_t> &out_ids) const
{
    std::string text;
    if (!RenderGemmaConversation(conv, text)) return false;
    if (!AppendGemmaChatIds(tok, text, out_ids))
        return BuildPromptInputIds(tok, text, out_ids);
    return true;
}

bool Gemma4_31BDenseModel::ParseToolCalls(
    const ArrowTokenizerDb &tok, const std::vector<int32_t> &gen_tokens,
    std::vector<ToolCall> &out_calls) const
{
    return ParseGemmaTokenCalls(tok, gen_tokens, out_calls);
}

} // namespace lsxcommon
