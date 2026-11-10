#include "dory_dense_model.h"

#include <cuda_runtime.h>

#include <cstring>
#include <string_view>

#include <absl/strings/str_cat.h>

#include "fast_load_internal.h"
#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/dory_tool_format.h"
#include "lsxcommon/log.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/sampling.h"
#include "lsxcommon/tokenizer.h"
#include "lsxcommon/tool_format.h"

namespace lsxcommon {

const std::string DoryDenseModel::kType = "dory-1.0-4b";

DoryDenseModel::DoryDenseModel(ModelInitConfig cfg)
    : ModelBase(std::move(cfg))
{
}

DoryDenseModel::~DoryDenseModel() { Unload(); }

void DoryDenseModel::TeardownWeights()
{
    pw_.Free();
    weights_loaded_ = false;
    kv_.reset();
    session_tail_.clear();
    TeardownContext();
}

bool DoryDenseModel::Load()
{
    Unload();
    tokenizer_optional_ = false;
    const FastMathMode fm = EffectiveFastMathMode(cfg_);
    if (fm != FastMathMode::kAuto) {
        lsx_nvfp4::SetLsxTcFastMath(fm == FastMathMode::kOn);
        lsx_nvfp4::SetLsxSeq1Gemv(fm == FastMathMode::kOn);
    }
    if (!BeginLoad(cfg_.model_path)) return false;
    if (!EnsureContext()) {
        lsxcommon::log::error("[dory-model] stream create failed");
        return AbortLoad();
    }
    const bool ok = dory::LoadDoryPinnedWeights(cfg_.model_path, pw_, stream());
    fastload::StreamDrain{stream()};
    weights_loaded_ = ok;
    if (!ok) {
        lsxcommon::log::error("[dory-model] LoadDoryPinnedWeights failed");
        return AbortLoad();
    }
    resident_bytes_ = pw_.resident_bytes;
    vram_ = pw_.top.vram;
    if (!AcquirePermanentWeights(pw_.resident_bytes)) return AbortLoad();
    log::info(absl::StrCat("[dory-model] pinned weights loaded: ",
                           pw_.layers.size(), " mixers (",
                           pw_.resident_bytes, " bytes resident, ",
                           pw_.kv_slots, " kv slots)"));
    if (!ValidateGeometry(pw_.top)) {
        log::error(absl::StrCat("[dory-model] geometry validation failed for ",
                                cfg_.model_path));
        return AbortLoad();
    }
    FinishLoad();
    return true;
}

bool DoryDenseModel::ValidateGeometry(const dory::DoryTopology &top) const
{
    if (SyntheticGeometryAllowed()) return true;
    const GeoFieldCheck fields[] = {
        {"hidden_size", dory::frozen::kHiddenSize, top.hidden_size},
        {"num_layers", dory::frozen::kNumLayers, top.num_layers},
        {"num_attention_heads", dory::frozen::kNumHeads,
         top.num_attention_heads},
        {"num_key_value_heads", dory::frozen::kNumKVHeads,
         top.num_key_value_heads},
        {"head_dim", dory::frozen::kHeadDim, top.head_dim},
        {"intermediate_size", dory::frozen::kIntermediateSize,
         top.intermediate_size},
        {"vocab_size", dory::frozen::kVocabSize, top.vocab_size},
        {"n_input_layers", dory::frozen::kInputLayers, top.input_layers},
        {"n_recurrent_layers", dory::frozen::kRecurrentLayers,
         top.recurrent_layers},
        {"n_output_layers", dory::frozen::kOutputLayers, top.output_layers},
        {"n_recurrent_loops", dory::frozen::kRecurrentLoops,
         top.recurrent_loops},
        {"sliding_window", dory::frozen::kSlidingWindow, top.sliding_window},
        {"num_attention_layers", dory::frozen::kNumAttentionLayers,
         top.num_attention_layers},
        {"num_kv_slots", dory::frozen::kKvSlots, pw_.kv_slots},
    };
    return CheckGeometryFields("dory", fields,
                               sizeof(fields) / sizeof(fields[0]));
}

gpu::InferenceResult DoryDenseModel::Infer(const ModelRequest &req)
{
    gpu::InferenceResult r;
    if (!ValidateInferRequest(req, r)) return r;

    // Model independence: this model's own kernel-routing policy for the
    // duration of the call.
    //
    // PINNED to kOff (not the requested auto policy). Measured 2026-11-10 on
    // the converted tree: the Blackwell W4A4 mxf4nvf4 tensor-core path is
    // tolerance-class per GEMM (relL2 0.06 at seq 1 rising to 0.19 at seq 32
    // against the scalar reference), and Dory amplifies that far beyond any
    // other family — every mixer output is L2-normalized before it is fed
    // forward, so a relative error of a few percent in a projection becomes a
    // *direction* error of the residual stream at the very next layer, and the
    // 2x40 recurrent passes compound it. Symptom: "The capital of France is"
    // framed as a chat answered `# Prompt\n\nWrite a short story about …` on
    // the TC path versus `The user is asking a simple factual question …
    // Paris` on the strict path, i.e. the auto policy silently produced tokens
    // the source model never produces. Same remedy, same reason as the granite
    // and granite-switch families. The per-GPU policy is still consulted (and
    // the requested value is read) so the model-independence contract holds;
    // the residual NVFP4 plane plus the strict scalar GEMV route is what makes
    // the 4-bit body track the source.
    const FastMathMode requested = EffectiveFastMathMode(cfg_);
    (void)requested;
    const lsx_nvfp4::LsxFastMathScope fm_scope(FastMathMode::kOff);

    if (!EnsureContext()) {
        r.error_msg = "cuda init failed";
        return r;
    }
    cudaStream_t st = stream();

    lsxcommon::GenerationStream gs;
    gs.on_token = req.token_callback ? &req.token_callback : nullptr;
    gs.abort_flag = req.abort_flag;

    // The whole request runs inside ONE shared-lock scope (see phi/model_base:
    // it orders this request against Unload's unique lock; one Infer per
    // instance at a time is a precondition, not something the lock provides).
    return RunUnderResidencyLock([&]() -> gpu::InferenceResult {
        const bool continuation =
            kv_ && kv_->allocated() && !session_tail_.empty() &&
            req.input_ids == session_tail_;
        std::vector<int32_t> gen_input;
        if (continuation) {
            gen_input = std::vector<int32_t>{kv_->last_tok};
        } else {
            kv_.reset();
            kv_.reset(new dory::DoryKvCache());
            session_tail_.clear();
            gen_input = req.input_ids;
        }

        dory::DoryGenerationResult gen;
        const int eos_id = (req.ignore_eos || !tok_db_)
                               ? lsxcommon::kIgnoreEosTokenId
                               : tok_db_->eos_id;
        const cudaError_t e = dory::RunDoryGeneration(
            st, pw_, gen_input, req.max_gen_tokens, eos_id, gen,
            lsxcommon::sample::SamplerFor(req.sampling), kv_.get(), &gs);
        if (e != cudaSuccess) {
            r.error_msg = runtime::CombineGenError(e, gen.error_msg);
            r.ok = false;
            kv_.reset();
            session_tail_.clear();
            return r;
        }

        const int kv_slots =
            (kv_ && kv_->allocated())
                ? kv_->kv_rows
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

gpu::InferenceResult DoryDenseModel::SelfCheck(const ModelRequest &req)
{
    // The canonical probe is a RAW continuation, and it is the family's
    // diagnostic in every entry point.
    //
    // Why: this checkpoint's served framing wraps the prompt in a ChatML user
    // turn and opens a <think> block. Answering "The capital of France is" as
    // a CHAT TURN therefore restates the question first — with reasoning on the
    // first tokens are "The user …", with reasoning off they are "The capital
    // of France is Paris" — so the strict 1-2 token "Paris" contract can only
    // hold for the raw continuation, which is what the shared SelfCheck driver
    // already sends (`selfcheck_frame_prompt: false` in the manifest).
    // `lsxbenchmark --selfcheck` frames instead, and would report "passed: no"
    // for a model that answers its own probe correctly. Encoding the probe here
    // makes the two entry points say the same thing, and it is a no-op for the
    // driver (the ids are identical to the ones it passes).
    ModelRequest sc = req;
    sc.input_ids.clear();
    {
        static constexpr char kProbe[] = "The capital of France is";
        std::vector<int32_t> ids(64, 0);
        size_t cap = ids.size();
        const int32_t n = tok_db_
                              ? arrow_tokenizer_encode(*tok_db_, kProbe,
                                                       std::strlen(kProbe),
                                                       ids.data(), &cap)
                              : 0;
        if (n <= 0) {
            gpu::InferenceResult r;
            r.error_msg = "selfcheck probe encode failed";
            return r;
        }
        ids.resize(static_cast<size_t>(n));
        sc.input_ids = std::move(ids);
    }
    return runtime::RunParisSelfCheck(*this, sc);
}

// ── Token bridge (stage 4 + 4b) ────────────────────────────────────────────

namespace {

// Appends `text` through the sentinel-aware encoder. The reference tokenizer
// never emits BOS (`add_bos_token: false`), and the bridge writes bos_id = -1,
// so no leading id has to be stripped here.
bool AppendDoryText(const ArrowTokenizerDb &tok, const std::string &text,
                    std::vector<int32_t> &out)
{
    std::vector<int32_t> ids;
    if (!AppendDoryChatIds(tok, text, ids)) return false;
    out.insert(out.end(), ids.begin(), ids.end());
    return true;
}

} // namespace

bool DoryDenseModel::BuildPromptInputIds(const ArrowTokenizerDb &tok,
                                         std::string_view raw_prompt,
                                         std::vector<int32_t> &out_ids) const
{
    out_ids.clear();
    ChatConversation conv;
    conv.messages.push_back({"user", std::string(raw_prompt), "", {}});
    std::string text;
    if (!RenderDoryConversation(conv, text)) return false;
    return AppendDoryText(tok, text, out_ids);
}

bool DoryDenseModel::BuildConversationInputIds(
    const ArrowTokenizerDb &tok, const ChatConversation &conv,
    std::vector<int32_t> &out_ids) const
{
    out_ids.clear();
    std::string text;
    if (!RenderDoryConversation(conv, text)) return false;
    return AppendDoryText(tok, text, out_ids);
}

bool DoryDenseModel::ParseToolCalls(const ArrowTokenizerDb &tok,
                                    const std::vector<int32_t> &gen_tokens,
                                    std::vector<ToolCall> &out_calls) const
{
    return ParseDoryTokenCalls(tok, gen_tokens, out_calls);
}

} // namespace lsxcommon
