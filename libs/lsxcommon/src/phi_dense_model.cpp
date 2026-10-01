#include "phi_dense_model.h"

#include <cuda_runtime.h>

#include "fast_load_internal.h"
#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/log.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/sampling.h"
#include "lsxcommon/tokenizer.h"
#include "lsxcommon/tool_format.h"

#include <absl/strings/str_cat.h>

#include <cstring>
#include <string_view>

namespace lsxcommon {

const std::string Phi4Model::kType = "phi-4";

Phi4Model::Phi4Model(ModelInitConfig cfg)
    : ModelBase(std::move(cfg)) {}

Phi4Model::~Phi4Model() { Unload(); }

void Phi4Model::TeardownWeights()
{
    pw_.Free();
    weights_loaded_ = false;
    kv_.reset();
    session_tail_.clear();
    TeardownContext();
}

bool Phi4Model::Load()
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
        lsxcommon::log::error("[phi-model] stream create failed");
        return AbortLoad();
    }
    const bool ok =
        phi::LoadPhiPinnedWeights(cfg_.model_path, pw_, stream());
    fastload::StreamDrain{stream()};
    weights_loaded_ = ok;
    if (!ok) {
        lsxcommon::log::error("[phi-model] LoadPhiPinnedWeights failed");
        return AbortLoad();
    }
    resident_bytes_ = pw_.resident_bytes;
    vram_ = pw_.top.vram;
    if (!AcquirePermanentWeights(pw_.resident_bytes)) return AbortLoad();
    lsxcommon::log::info(absl::StrCat(
        "[phi-model] pinned weights loaded: ", pw_.layers.size(),
        " layers (", pw_.resident_bytes, " bytes resident)"));
    if (!ValidateGeometry(pw_.top)) {
        lsxcommon::log::error(absl::StrCat(
            "[phi-model] geometry validation failed for ",
            cfg_.model_path));
        return AbortLoad();
    }
    FinishLoad();
    return true;
}

bool Phi4Model::ValidateGeometry(
    const phi::PhiTopology &top) const
{
    if (SyntheticGeometryAllowed()) return true;
    const GeoFieldCheck fields[] = {
        {"hidden_size", kHiddenSize, top.hidden_size},
        {"num_layers", kNumLayers, top.num_layers},
        {"num_attention_heads", kNumHeads, top.num_attention_heads},
        {"num_key_value_heads", kNumKVHeads, top.num_key_value_heads},
        {"head_dim", kHeadDim, top.head_dim},
        {"intermediate_size", kIntermediateSize, top.intermediate_size},
        {"vocab_size", kVocabSize, top.vocab_size},
    };
    return CheckGeometryFields("phi-4", fields, 7);
}

gpu::InferenceResult Phi4Model::Infer(const ModelRequest &req)
{
    gpu::InferenceResult r;
    if (!ValidateInferRequest(req, r)) return r;

    // Model independence: this model's own kernel-routing policy for the
    // duration of the call.
    const lsx_nvfp4::LsxFastMathScope fm_scope(EffectiveFastMathMode(cfg_));

    if (!EnsureContext()) {
        r.error_msg = "cuda init failed";
        return r;
    }
    cudaStream_t st = stream();

    // Incremental session policy: a request matching the previous call's
    // generated tail continues from {kv_->last_tok}; anything else restarts.
    const bool continuation =
        kv_ && kv_->allocated() && !session_tail_.empty() &&
        req.input_ids == session_tail_;
    std::vector<int32_t> gen_input;
    if (continuation)
        gen_input = std::vector<int32_t>{kv_->last_tok};
    else {
        kv_.reset();
        kv_.reset(new phi::PhiKvCache());
        session_tail_.clear();
        gen_input = req.input_ids;
    }

    phi::PhiGenerationResult gen;
    lsxcommon::GenerationStream gs;
    gs.on_token = req.token_callback ? &req.token_callback : nullptr;
    gs.abort_flag = req.abort_flag;
    auto run = [&]() -> cudaError_t {
        const int eos_id = (req.ignore_eos || !tok_db_)
                               ? lsxcommon::kIgnoreEosTokenId
                               : tok_db_->eos_id;
        return phi::RunPhiGeneration(
            st, pw_, gen_input, req.max_gen_tokens, eos_id, gen,
            lsxcommon::sample::DefaultSampler(), kv_.get(), &gs);
    };

    cudaError_t e = RunUnderResidencyLock(run);
    if (e != cudaSuccess) {
        r.error_msg = runtime::CombineGenError(e, gen.error_msg);
        r.ok = false;
        kv_.reset();
        session_tail_.clear();
        return r;
    }

    const int kv_slots = (kv_ && kv_->allocated())
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
}

gpu::InferenceResult Phi4Model::SelfCheck(const ModelRequest &req)
{
    return runtime::RunParisSelfCheck(*this, req);
}

// ── phi ChatML token bridge (Model Ingestion Protocol step 4b) ──────────────

namespace {

bool EncodePhiText(const ArrowTokenizerDb &tok, std::string_view text,
                   std::vector<int32_t> &out)
{
    if (text.empty()) return true;
    std::vector<int32_t> ids(text.size() + 16, 0);
    size_t cap = ids.size();
    const int32_t n = arrow_tokenizer_encode(tok, text.data(), text.size(),
                                             ids.data(), &cap);
    if (n <= 0) return false;
    out.insert(out.end(), ids.begin(), ids.begin() + n);
    return true;
}

void AppendPhiMarker(const ArrowTokenizerDb &tok, const char *marker,
                     std::vector<int32_t> &out)
{
    const int32_t id = arrow_tokenizer_lookup(
        tok, marker, static_cast<size_t>(std::strlen(marker)));
    if (id >= 0) out.push_back(id);
}

// Appends one phi ChatML turn. Control markers are special added_tokens and
// are injected as ids; only the content is BPE-encoded. Untrusted content is
// neutralized so it can never forge a control marker.
bool AppendPhiTurn(const ArrowTokenizerDb &tok, const char *role,
                   std::string_view content, std::vector<int32_t> &out)
{
    AppendPhiMarker(tok, "<|im_start|>", out);
    if (!EncodePhiText(tok, role, out)) return false;
    AppendPhiMarker(tok, "<|im_sep|>", out);
    const std::string safe = tools::NeutralizeControlMarkers(
        content, {"<|im_start|>", "<|im_sep|>", "<|im_end|>"});
    if (!EncodePhiText(tok, safe, out)) return false;
    AppendPhiMarker(tok, "<|im_end|>", out);
    return true;
}

// Open assistant generation prompt: `<|im_start|>assistant<|im_sep|>`.
void AppendPhiAssistantPrompt(const ArrowTokenizerDb &tok,
                              std::vector<int32_t> &out)
{
    AppendPhiMarker(tok, "<|im_start|>", out);
    EncodePhiText(tok, "assistant", out);
    AppendPhiMarker(tok, "<|im_sep|>", out);
}

} // namespace

bool Phi4Model::BuildPromptInputIds(const ArrowTokenizerDb &tok,
                                    std::string_view raw_prompt,
                                    std::vector<int32_t> &out_ids) const
{
    out_ids.clear();
    if (!AppendPhiTurn(tok, "user", raw_prompt, out_ids)) return false;
    AppendPhiAssistantPrompt(tok, out_ids);
    return true;
}

bool Phi4Model::BuildConversationInputIds(const ArrowTokenizerDb &tok,
                                          const ChatConversation &conv,
                                          std::vector<int32_t> &out_ids) const
{
    out_ids.clear();
    // phi has no native tool syntax: render the schemas as a system text
    // block (Hermes JSON) so the transcript stays self-consistent.
    if (!conv.tools.empty()) {
        const std::string block = tools::RenderToolsBlock(conv.tools);
        if (!AppendPhiTurn(tok, "system", block, out_ids)) return false;
    }
    for (const auto &m : conv.messages) {
        std::string content = m.content;
        if (!m.tool_calls.empty()) {
            if (!content.empty()) content += '\n';
            content += "<tool_calls>";
            for (const auto &c : m.tool_calls)
                content += "\n" + c.name + " " + c.arguments_json;
        }
        // phi's template has no `tool` role; fold tool results into a user turn.
        const char *role = (m.role == "tool") ? "user" : m.role.c_str();
        if (!AppendPhiTurn(tok, role, content, out_ids)) return false;
    }
    AppendPhiAssistantPrompt(tok, out_ids);
    return true;
}

} // namespace lsxcommon
