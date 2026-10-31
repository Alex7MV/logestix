#include "lsxcommon/step_moe_tool_format.h"

#include "step_moe_constants.h"

#include "lsxcommon/model_runtime.h"
#include "lsxcommon/tool_format.h"

#include <lsxnetwork/json_escape.h>

#include <simdjson.h>

#include <cstdio>
#include <cstring>
#include <string_view>

namespace lsxcommon::step {
namespace {

using lsxnetwork::json::JsonEscape;

constexpr const char *kBos = "<｜begin▁of▁sentence｜>";
constexpr const char *kImStart = "<|im_start|>";
constexpr const char *kImEnd = "<|im_end|>";
constexpr const char *kThinkBegin = "<think>";
constexpr const char *kThinkEnd = "</think>";
constexpr const char *kToolCallBegin = "<tool_call>";
constexpr const char *kToolCallEnd = "</tool_call>";
constexpr const char *kToolResponseBegin = "<tool_response>";
constexpr const char *kToolResponseEnd = "</tool_response>";

// Only markers that are real vocabulary pieces are injected as their own id;
// `<function=`, `<parameter=` and `<tools>` are plain text and go through BPE.
constexpr const char *kSentinels[] = {
    kImStart,        kImEnd,          kThinkBegin,       kThinkEnd,
    kToolCallBegin,  kToolCallEnd,    kToolResponseBegin, kToolResponseEnd,
};

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

bool AppendEncoded(const ArrowTokenizerDb &tok, std::string_view text,
                   std::vector<int32_t> &ids)
{
    if (text.empty()) return true;
    for (size_t cap = 256; cap <= (1u << 20); cap *= 2) {
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

bool AppendTextWithSentinels(const ArrowTokenizerDb &tok, const std::string &text,
                             std::vector<int32_t> &ids)
{
    size_t i = 0;
    size_t run = 0;
    auto flush = [&](size_t end) -> bool {
        if (end > run)
            return AppendEncoded(tok, std::string_view(text).substr(run, end - run),
                                 ids);
        return true;
    };
    while (i < text.size()) {
        bool matched = false;
        for (const char *s : kSentinels) {
            const size_t n = std::strlen(s);
            if (text.compare(i, n, s) != 0) continue;
            if (!flush(i)) return false;
            const int32_t id = arrow_tokenizer_lookup(tok, s, n);
            if (id < 0) return false;
            ids.push_back(id);
            i += n;
            run = i;
            matched = true;
            break;
        }
        if (!matched) ++i;
    }
    return flush(text.size());
}

// One tool definition in the template's `tool | tojson` shape.
void AppendToolSpec(const ToolSpec &t, std::string &out)
{
    out += "{\"type\": \"function\", \"function\": {\"name\": \"";
    out += JsonEscape(t.name);
    out += "\", \"description\": \"";
    out += JsonEscape(t.description);
    out += "\", \"parameters\": ";
    out += t.parameters_json.empty() ? "{}" : t.parameters_json;
    out += "}}";
}

void AppendToolCall(const ToolCall &call, std::string &out)
{
    out += kToolCallBegin;
    out += "\n<function=";
    out += call.name;
    out += ">\n";
    simdjson::dom::parser parser;
    simdjson::dom::element args;
    if (!call.arguments_json.empty() &&
        parser.parse(call.arguments_json).get(args) == simdjson::SUCCESS &&
        args.is_object()) {
        for (auto [k, v] : args.get_object()) {
            out += "<parameter=";
            out += std::string(k);
            out += ">\n";
            if (v.is_string()) {
                std::string_view s;
                if (v.get_string().get(s) != simdjson::SUCCESS) s = {};
                out += std::string(s);
            } else {
                simdjson::dom::element e = v;
                // Round-trip through minified JSON for non-string scalars.
                out += simdjson::to_string(e);
            }
            out += "\n</parameter>\n";
        }
    }
    out += "</function>\n</tool_call>";
}

} // namespace

bool StepChatSentinelsPresent(const ArrowTokenizerDb &tok)
{
    return arrow_tokenizer_lookup(tok, kImStart, std::strlen(kImStart)) >= 0 &&
           arrow_tokenizer_lookup(tok, kImEnd, std::strlen(kImEnd)) >= 0 &&
           arrow_tokenizer_lookup(tok, kThinkBegin, std::strlen(kThinkBegin)) >=
               0 &&
           arrow_tokenizer_lookup(tok, kToolCallBegin,
                                  std::strlen(kToolCallBegin)) >= 0;
}

bool RenderStepConversation(const ArrowTokenizerDb &tok,
                            const ChatConversation &conv,
                            std::vector<int32_t> &out_ids)
{
    out_ids.clear();
    if (!StepChatSentinelsPresent(tok)) return false;
    if (conv.messages.empty()) return false;

    std::string text;
    text += kBos;

    const bool has_tools = !conv.tools.empty();
    const bool first_is_system = conv.messages[0].role == "system";
    if (has_tools) {
        text += kImStart;
        text += "system\n";
        if (first_is_system) {
            text += conv.messages[0].content;
            text += "\n\n";
        }
        text += "# Tools\n\nYou have access to the following functions in "
                "JSONSchema format:\n\n<tools>";
        for (const auto &t : conv.tools) {
            text += "\n";
            AppendToolSpec(t, text);
        }
        text += "\n</tools>\n\nIf you choose to call a function ONLY reply in "
                "the following format with NO suffix:\n\n";
        text += kToolCallBegin;
        text += "\n<function=example_function_name>\n<parameter="
                "example_parameter_1>\nvalue_1\n</parameter>\n<parameter="
                "example_parameter_2>\nThis is the value for the second "
                "parameter\nthat can span\nmultiple lines\n</parameter>\n"
                "</function>\n";
        text += kToolCallEnd;
        text += "\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the "
                "specified format: an inner <function=...>\n...\n</function> "
                "block must be nested within ";
        text += kToolCallBegin;
        text += "\n...\n";
        text += kToolCallEnd;
        text += " XML tags\n- Required parameters MUST be specified\n"
                "</IMPORTANT>";
        text += kImEnd;
        text += "\n";
    } else if (first_is_system) {
        text += kImStart;
        text += "system\n";
        text += conv.messages[0].content;
        text += kImEnd;
        text += "\n";
    }

    for (size_t i = 0; i < conv.messages.size(); ++i) {
        const ChatMessage &m = conv.messages[i];
        if (i == 0 && m.role == "system") continue;  // emitted by the header
        if (m.role == "user" || m.role == "system") {
            text += kImStart;
            text += m.role;
            text += "\n";
            text += m.content;
            text += kImEnd;
            text += "\n";
        } else if (m.role == "assistant") {
            // The template always opens the assistant turn with <think> and
            // splits inline reasoning at </think>.
            std::string reasoning;
            std::string content = m.content;
            const size_t close = content.find(kThinkEnd);
            if (close != std::string::npos) {
                reasoning = content.substr(0, close);
                const size_t open = reasoning.find(kThinkBegin);
                if (open != std::string::npos)
                    reasoning = reasoning.substr(open + std::strlen(kThinkBegin));
                content = content.substr(close + std::strlen(kThinkEnd));
                const size_t skip = content.find_first_not_of("\n");
                if (skip != std::string::npos) content = content.substr(skip);
                else content.clear();
            }
            text += kImStart;
            text += "assistant\n";
            text += kThinkBegin;
            text += reasoning;
            text += kThinkEnd;
            text += content;
            for (const auto &call : m.tool_calls) AppendToolCall(call, text);
            text += kImEnd;
            text += "\n";
        } else if (m.role == "tool") {
            if (i == 0 || conv.messages[i - 1].role != "tool") {
                text += kImStart;
                text += "tool_response\n";
            }
            text += kToolResponseBegin;
            text += m.content;
            text += kToolResponseEnd;
            if (i + 1 == conv.messages.size() ||
                conv.messages[i + 1].role != "tool") {
                text += kImEnd;
                text += "\n";
            }
        }
    }

    // add_generation_prompt: thinking is enabled by default (the template has
    // no disable switch other than preserve_thinking).
    text += kImStart;
    text += "assistant\n";
    text += kThinkBegin;

    if (!AppendTextWithSentinels(tok, text, out_ids)) {
        out_ids.clear();
        return false;
    }
    return true;
}

std::vector<ToolCall> ParseStepToolCalls(const ArrowTokenizerDb &tok,
                                         const std::vector<int32_t> &gen_tokens)
{
    std::vector<ToolCall> calls;
    std::string text;
    if (!gen_tokens.empty())
        text = runtime::DecodeGeneratedText(tok, gen_tokens.data(),
                                            gen_tokens.size());
    size_t pos = 0;
    while (true) {
        const size_t at = text.find(kToolCallBegin, pos);
        if (at == std::string::npos) break;
        const size_t close = text.find(kToolCallEnd, at);
        if (close == std::string::npos) break;
        const size_t fb = text.find("<function=", at);
        if (fb == std::string::npos || fb >= close) {
            pos = close + std::strlen(kToolCallEnd);
            continue;
        }
        const size_t fname_end = text.find('>', fb);
        if (fname_end == std::string::npos || fname_end >= close) break;
        ToolCall call;
        call.name = std::string(TrimAscii(
            std::string_view(text).substr(fb + std::strlen("<function="),
                                          fname_end - fb -
                                              std::strlen("<function="))));
        std::string args = "{";
        bool first = true;
        size_t p = fname_end;
        while (true) {
            const size_t pb = text.find("<parameter=", p);
            if (pb == std::string::npos || pb >= close) break;
            const size_t pe = text.find('>', pb);
            if (pe == std::string::npos || pe >= close) break;
            const size_t vb = text.find('\n', pe);
            if (vb == std::string::npos || vb >= close) break;
            const size_t ve = text.find("</parameter>", vb);
            if (ve == std::string::npos || ve > close) break;
            const std::string key = std::string(TrimAscii(
                std::string_view(text).substr(pb + std::strlen("<parameter="),
                                              pe - pb -
                                                  std::strlen("<parameter="))));
            std::string_view raw(text.data() + vb + 1, ve - vb - 1);
            if (!raw.empty() && raw.back() == '\n') raw.remove_suffix(1);
            if (!first) args += ",";
            first = false;
            args += "\"" + JsonEscape(key) + "\":";
            const std::string_view t = TrimAscii(raw);
            simdjson::dom::parser vp;
            simdjson::dom::element doc;
            const bool json_ok = !t.empty() && t.front() != '"' &&
                                 vp.parse(t).get(doc) == simdjson::SUCCESS &&
                                 !doc.is_string();
            if (json_ok)
                args += std::string(t);
            else if (!raw.empty() && raw.front() == '"')
                args += std::string(raw);
            else
                args += "\"" + JsonEscape(std::string(raw)) + "\"";
            p = ve + std::strlen("</parameter>");
        }
        args += "}";
        call.arguments_json = std::move(args);
        calls.push_back(std::move(call));
        pos = close + std::strlen(kToolCallEnd);
    }
    if (calls.empty()) {
        calls = tools::ParseToolCallsText(text);
        if (calls.empty()) calls = tools::ParseBareToolCallJson(text);
    }
    for (size_t i = 0; i < calls.size(); ++i)
        if (calls[i].id.empty()) calls[i].id = "call_" + std::to_string(i);
    return calls;
}

} // namespace lsxcommon::step
