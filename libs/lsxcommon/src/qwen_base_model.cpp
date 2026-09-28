#include "lsxcommon/qwen_base_model.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "lsxcommon/log.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/qwen_inference.h"
#include "lsxcommon/qwen_inference_load.h"
#include "fast_load_internal.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/tokenizer.h"
#include "lsxcommon/tool_format.h"

#include <absl/strings/str_cat.h>
#include <simdjson.h>

#include <cstdlib>
#include <cstring>

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

        // Incremental session policy (spec item 15): a request whose input_ids
    // exactly match the previous call's generated tail continues the session
    // KV cache from {kv_->last_tok}; anything else starts a fresh session.
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
    lsxcommon::GenerationStream gs;
    gs.on_token = req.token_callback ? &req.token_callback : nullptr;
    gs.abort_flag = req.abort_flag;
    auto run = [&]() -> cudaError_t {
        const int eos_id = (req.ignore_eos || !tok_db_) ? lsxcommon::kIgnoreEosTokenId : tok_db_->eos_id;
        return qwen::RunQwenGeneration(st, pw_, gen_input,
                                       req.max_gen_tokens, eos_id, gen,
                                       lsxcommon::sample::DefaultSampler(),
                                       kv_.get(), &gs);
    };

    cudaError_t e = RunUnderResidencyLock(run);
    if (e != cudaSuccess) {
        r.error_msg = runtime::CombineGenError(e, gen.error_msg);
        r.ok = false;
        // A failed continuation may have advanced kv_rows mid-stream; drop the
        // session so the next call starts fresh.
        kv_.reset();
        session_tail_.clear();
        return r;
    }

    // Occupancy (spec item 15): the incremental cache holds prompt + generated
    // rows once the pending last-token row is counted (kv_rows + 1). The
    // fallback (non-incremental) mirrors the same prompt + gen semantic.
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

// QwenBaseModel::Unload — canonical ModelBase template method
// (TeardownWeights override above); no per-model override.

gpu::InferenceResult QwenBaseModel::SelfCheck(const ModelRequest &req)
{
    return runtime::RunParisSelfCheck(*this, req);
}

bool QwenBaseModel::ValidateGeometry(const qwen::QwenTopology &) const { return true; }

// ── Qwen ChatML + native tool format ────────────────────────────────────────
// Verified against the checkpoint chat_template.jinja (Qwen3.5/3.6):
//   <|im_start|>system\n# Tools\n\n…<tools>…</tools>…<IMPORTANT>…</IMPORTANT>
//   <|im_start|>user\n…<|im_end|>\n
//   <|im_start|>assistant\n{content}<tool_call><function=NAME><parameter=K>V…
//   <|im_start|>user\n<tool_response>…</tool_response><|im_end|>\n
//   <|im_start|>assistant\n<think>\n\n</think>\n\n   (enable_thinking=false)

namespace {

// Appends the BPE encoding of `text` to `ids`. arrow_tokenizer_encode
// auto-prepends bos_id when the tokenizer defines one; ChatML frames turns with
// <|im_start|> instead, so the prepended BOS is dropped.
bool AppendQwenEncoded(const ArrowTokenizerDb &tok, std::string_view text,
                       std::vector<int32_t> &ids)
{
    for (size_t cap = 256; cap <= 65536; cap *= 2) {
        std::vector<int32_t> buf(cap, 0);
        size_t enc_cap = cap;
        const int32_t n = arrow_tokenizer_encode(tok, text.data(), text.size(),
                                                 buf.data(), &enc_cap);
        if (n == -2) continue;
        if (n < 0) return false;
        const size_t begin = (tok.bos_id >= 0 && n > 0) ? 1 : 0;
        ids.insert(ids.end(), buf.begin() + begin, buf.begin() + n);
        return true;
    }
    return false;
}

// Control sentinels that must be injected as their own vocab ids:
// arrow_tokenizer_encode is pure BPE and never resolves added tokens.
constexpr const char *kQwenSentinels[] = {
    "<|im_start|>",   "<|im_end|>",     "<tool_call>", "</tool_call>",
    "<tool_response>", "</tool_response>", "<think>",   "</think>",
};

bool AppendQwenIds(const ArrowTokenizerDb &tok, const std::string &text,
                   std::vector<int32_t> &ids)
{
    size_t i = 0;
    size_t run = 0;
    auto flush = [&](size_t end) -> bool {
        if (end > run &&
            !AppendQwenEncoded(tok,
                               std::string_view(text).substr(run, end - run),
                               ids))
            return false;
        return true;
    };
    while (i < text.size()) {
        bool matched = false;
        for (const char *s : kQwenSentinels) {
            const size_t n = std::strlen(s);
            if (text.compare(i, n, s) == 0) {
                if (!flush(i)) return false;
                const int32_t id = arrow_tokenizer_lookup(tok, s, n);
                if (id < 0) return false;
                ids.push_back(id);
                i += n;
                run = i;
                matched = true;
                break;
            }
        }
        if (!matched) ++i;
    }
    return flush(text.size());
}

std::string_view TrimAscii(std::string_view s)
{
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t' ||
                          s.front() == '\r' || s.front() == '\n'))
        s.remove_prefix(1);
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t' ||
                          s.back() == '\r' || s.back() == '\n'))
        s.remove_suffix(1);
    return s;
}

std::string QwenJsonEscape(std::string_view in)
{
    std::string out;
    out.reserve(in.size() + 8);
    static const char *kHex = "0123456789abcdef";
    for (char c : in) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    out += "\\u00";
                    out += kHex[(static_cast<unsigned char>(c) >> 4) & 0xF];
                    out += kHex[static_cast<unsigned char>(c) & 0xF];
                } else {
                    out += c;
                }
        }
    }
    return out;
}

std::string QwenSerialize(simdjson::dom::element e);

std::string QwenSerialize(simdjson::dom::element e)
{
    using simdjson::dom::element_type;
    switch (e.type()) {
        case element_type::OBJECT: {
            std::string out = "{";
            bool first = true;
            for (auto [k, v] : e.get_object()) {
                if (!first) out += ",";
                first = false;
                out += "\"" + QwenJsonEscape(k) + "\":" + QwenSerialize(v);
            }
            out += "}";
            return out;
        }
        case element_type::ARRAY: {
            std::string out = "[";
            bool first = true;
            for (auto v : e.get_array()) {
                if (!first) out += ",";
                first = false;
                out += QwenSerialize(v);
            }
            out += "]";
            return out;
        }
        case element_type::STRING:
            return "\"" + QwenJsonEscape(e.get_string().value_unsafe()) + "\"";
        case element_type::INT64:
            return std::to_string(e.get_int64().value_unsafe());
        case element_type::UINT64:
            return std::to_string(e.get_uint64().value_unsafe());
        case element_type::DOUBLE: {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.17g",
                          e.get_double().value_unsafe());
            return std::string(buf);
        }
        case element_type::BOOL:
            return e.get_bool().value_unsafe() ? "true" : "false";
        default:
            return "null";
    }
}

} // namespace

bool QwenBaseModel::RenderQwenConversation(const ChatConversation &conv,
                                           std::string &out)
{
    out.clear();
    const bool has_tools = !conv.tools.empty();
    const bool first_system =
        !conv.messages.empty() && conv.messages[0].role == "system";

    if (has_tools) {
        out += "<|im_start|>system\n";
        out +=
            "# Tools\n\nYou have access to the following functions:\n\n"
            "<tools>";
        for (const ToolSpec &t : conv.tools) {
            out += "\n{\"type\":\"function\",\"function\":{\"name\":\"";
            out += QwenJsonEscape(t.name);
            out += "\",\"description\":\"";
            out += QwenJsonEscape(t.description);
            out += "\",\"parameters\":";
            out += t.parameters_json.empty() ? std::string("{}")
                                             : t.parameters_json;
            out += "}}";
        }
        out +=
            "\n</tools>\n\nIf you choose to call a function ONLY reply in the "
            "following format with NO suffix:\n\n"
            "<tool_call>\n<function=example_function_name>\n"
            "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
            "<parameter=example_parameter_2>\nThis is the value for the second "
            "parameter\nthat can span\nmultiple lines\n</parameter>\n"
            "</function>\n</tool_call>\n\n"
            "<IMPORTANT>\nReminder:\n"
            "- Function calls MUST follow the specified format: an inner "
            "<function=...></function> block must be nested within "
            "<tool_call></tool_call> XML tags\n"
            "- Required parameters MUST be specified\n"
            "- You may provide optional reasoning for your function call in "
            "natural language BEFORE the function call, but NOT after\n"
            "- If there is no function call available, answer the question "
            "like normal with your current knowledge and do not tell the user "
            "about function calls\n</IMPORTANT>";
        if (first_system && !conv.messages[0].content.empty())
            out += "\n\n" + conv.messages[0].content;
        out += "<|im_end|>\n";
    } else if (first_system) {
        out += "<|im_start|>system\n" + conv.messages[0].content +
               "<|im_end|>\n";
    }

    for (size_t mi = 0; mi < conv.messages.size(); ++mi) {
        const ChatMessage &m = conv.messages[mi];
        if (m.role == "system") continue;  // emitted in the preamble
        if (m.role == "user") {
            out += "<|im_start|>user\n" + m.content + "<|im_end|>\n";
        } else if (m.role == "assistant") {
            out += "<|im_start|>assistant\n" + m.content;
            bool first = true;
            for (const ToolCall &c : m.tool_calls) {
                if (first) {
                    out += m.content.empty()
                               ? "<tool_call>\n<function="
                               : "\n\n<tool_call>\n<function=";
                    first = false;
                } else {
                    out += "\n<tool_call>\n<function=";
                }
                out += c.name + ">\n";
                if (!c.arguments_json.empty()) {
                    simdjson::dom::parser parser;
                    simdjson::dom::element args;
                    simdjson::padded_string padded(c.arguments_json);
                    if (parser.parse(padded).get(args) == simdjson::SUCCESS &&
                        args.type() ==
                            simdjson::dom::element_type::OBJECT) {
                        for (auto [k, v] : args.get_object()) {
                            out += "<parameter=";
                            out += std::string(k);
                            out += ">\n";
                            if (v.type() ==
                                simdjson::dom::element_type::STRING)
                                out += std::string(
                                    v.get_string().value_unsafe());
                            else
                                out += QwenSerialize(v);
                            out += "\n</parameter>\n";
                        }
                    }
                }
                out += "</function>\n</tool_call>";
            }
            out += "<|im_end|>\n";
        } else if (m.role == "tool") {
            out += "<|im_start|>user\n<tool_response>\n" + m.content +
                   "\n</tool_response><|im_end|>\n";
        } else {
            return false;
        }
    }

    out += "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    return true;
}

std::vector<ToolCall> QwenBaseModel::ParseQwenToolCallText(std::string_view text)
{
    std::vector<ToolCall> calls;
    constexpr std::string_view kOpen = "<tool_call>";
    constexpr std::string_view kClose = "</tool_call>";
    constexpr std::string_view kFn = "<function=";
    constexpr std::string_view kFnEnd = "</function>";
    constexpr std::string_view kParam = "<parameter=";
    constexpr std::string_view kParamEnd = "</parameter>";
    size_t pos = 0;
    while (true) {
        const size_t b = text.find(kOpen, pos);
        if (b == std::string_view::npos) break;
        const size_t e = text.find(kClose, b + kOpen.size());
        if (e == std::string_view::npos) break;
        const std::string_view body =
            text.substr(b + kOpen.size(), e - (b + kOpen.size()));
        pos = e + kClose.size();

        const size_t fb = body.find(kFn);
        if (fb == std::string_view::npos) continue;
        const size_t name_start = fb + kFn.size();
        const size_t name_end = body.find('>', name_start);
        if (name_end == std::string_view::npos) continue;
        const std::string name(
            TrimAscii(body.substr(name_start, name_end - name_start)));
        if (name.empty()) continue;
        const size_t fe = body.find(kFnEnd, name_end);
        const size_t args_end = (fe == std::string_view::npos) ? body.size() : fe;
        const std::string_view region =
            body.substr(name_end + 1,
                        args_end > name_end + 1 ? args_end - (name_end + 1) : 0);

        std::string args = "{";
        bool first = true;
        size_t ap = 0;
        while (true) {
            const size_t pb = region.find(kParam, ap);
            if (pb == std::string_view::npos) break;
            const size_t kstart = pb + kParam.size();
            const size_t kend = region.find('>', kstart);
            if (kend == std::string_view::npos) break;
            const std::string key(
                TrimAscii(region.substr(kstart, kend - kstart)));
            const size_t vstart = kend + 1;
            const size_t vend = region.find(kParamEnd, vstart);
            if (vend == std::string_view::npos) break;
            const std::string_view raw = TrimAscii(
                region.substr(vstart, vend - vstart));
            ap = vend + kParamEnd.size();
            if (!first) args += ",";
            first = false;
            args += "\"" + QwenJsonEscape(key) + "\":";
            // Preserve JSON types (numbers/objects); otherwise a string.
            simdjson::dom::parser parser;
            simdjson::dom::element ev;
            simdjson::padded_string padded(raw);
            if (parser.parse(padded).get(ev) == simdjson::SUCCESS)
                args += QwenSerialize(ev);
            else
                args += "\"" + QwenJsonEscape(raw) + "\"";
        }
        args += "}";
        ToolCall call;
        call.name = name;
        call.arguments_json = std::move(args);
        call.id = "call_" + std::to_string(calls.size());
        calls.push_back(std::move(call));
    }
    return calls;
}

bool QwenBaseModel::BuildConversationInputIds(const ArrowTokenizerDb &tok,
                                              const ChatConversation &conv,
                                              std::vector<int32_t> &out_ids) const
{
    std::string text;
    if (!RenderQwenConversation(conv, text)) return false;
    out_ids.clear();
    // Missing ChatML sentinels → fall back to the single-prompt framing.
    if (arrow_tokenizer_lookup(tok, "<|im_start|>", 12) < 0 ||
        arrow_tokenizer_lookup(tok, "<|im_end|>", 10) < 0)
        return BuildPromptInputIds(tok, text, out_ids);
    return AppendQwenIds(tok, text, out_ids);
}

bool QwenBaseModel::ParseToolCalls(const ArrowTokenizerDb &tok,
                                   const std::vector<int32_t> &gen_tokens,
                                   std::vector<ToolCall> &out_calls) const
{
    out_calls.clear();
    if (gen_tokens.empty()) return true;
    std::size_t cap = 4096;
    std::string text(cap, '\0');
    bool decoded = false;
    for (int attempt = 0; attempt < 8; ++attempt) {
        const int rc = arrow_tokenizer_decode(tok, gen_tokens.data(),
                                              gen_tokens.size(), text.data(),
                                              &cap);
        if (rc == 0) {
            text.resize(cap);
            decoded = true;
            break;
        }
        if (rc == -2 && cap > text.size() && cap <= (1u << 20)) {
            text.resize(cap);
            continue;
        }
        break;
    }
    if (!decoded) return false;
    out_calls = ParseQwenToolCallText(text);
    if (out_calls.empty()) out_calls = tools::ParseToolCallsText(text);
    return true;
}

} // namespace lsxcommon
