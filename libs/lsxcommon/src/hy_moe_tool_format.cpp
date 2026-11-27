#include "lsxcommon/hy_moe_tool_format.h"

#include "lsxcommon/tool_format.h"

#include <lsxnetwork/json_escape.h>

#include <simdjson.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>

namespace lsxcommon::hy {
namespace {
// Dedup Ф4: BPE append (cap 1<<20, empty fast path) is the shared tools body.
using lsxcommon::tools::AppendEncodedLarge;
using lsxcommon::tools::AppendEncodedLarge;
inline bool AppendEncoded(const ArrowTokenizerDb &tok, std::string_view text,
                          std::vector<int32_t> &ids)
{
    return AppendEncodedLarge(tok, text, ids);
}

// Dedup Ф4: the Python-tojson helpers are the shared lsxcommon::tools
// implementations (byte-identical move); local aliases keep call sites stable.
using lsxcommon::tools::PyJsonString;
using lsxcommon::tools::PyJsonValue;
using lsxcommon::tools::PyJsonDumps;

using lsxnetwork::json::JsonEscape;

// Control markers (chat_template.jinja literals).
constexpr const char *kBos = "<｜hy_begin▁of▁sentence｜>";
constexpr const char *kEos = "<eos:6124c78e>";
constexpr const char *kUser = "<｜hy_User｜>";
constexpr const char *kAssistant = "<｜hy_Assistant｜>";
constexpr const char *kThinkBegin = "<think>";
constexpr const char *kThinkEnd = "</think>";
constexpr const char *kReasoningMode = "<｜reasoning_mode｜>";
constexpr const char *kToolCallsBegin = "<tool_calls>";
constexpr const char *kToolCallsEnd = "</tool_calls>";
constexpr const char *kToolCallBegin = "<tool_call>";
constexpr const char *kToolCallEnd = "</tool_call>";
constexpr const char *kToolSep = "<tool_sep>";
constexpr const char *kArgKeyBegin = "<arg_key>";
constexpr const char *kArgKeyEnd = "</arg_key>";
constexpr const char *kArgValueBegin = "<arg_value>";
constexpr const char *kArgValueEnd = "</arg_value>";
constexpr const char *kToolResponsesBegin = "<tool_responses>";
constexpr const char *kToolResponsesEnd = "</tool_responses>";
constexpr const char *kToolResponseBegin = "<tool_response>";
constexpr const char *kToolResponseEnd = "</tool_response>";

// The template's default when the caller does not ask for thinking.
constexpr const char *kDefaultEffort = "no_think";

const std::vector<std::string_view> kSentinelVec = {
    kBos, kEos, kUser, kAssistant, kThinkBegin, kThinkEnd, kReasoningMode, kToolCallsBegin, kToolCallsEnd, kToolCallBegin, kToolCallEnd, kToolSep, kArgKeyBegin, kArgKeyEnd, kArgValueBegin, kArgValueEnd, kToolResponsesBegin, kToolResponsesEnd, kToolResponseBegin, kToolResponseEnd};

bool AppendTextWithSentinels(const ArrowTokenizerDb &tok,
                             const std::string &text,
                             std::vector<int32_t> &ids)
{
    return lsxcommon::tools::AppendTextWithSentinels(tok, text, kSentinelVec,
                                                     ids);
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

// One `<tool_call>` block: name<tool_sep>\n<arg_key>..</arg_key>\n...
void AppendToolCallText(const ToolCall &call, std::string &out)
{
    out += kToolCallBegin;
    out += call.name;
    out += kToolSep;
    out += '\n';
    simdjson::dom::parser parser;
    simdjson::dom::element args;
    if (!call.arguments_json.empty() &&
        parser.parse(call.arguments_json).get(args) == simdjson::SUCCESS &&
        args.is_object()) {
        for (auto [k, v] : args.get_object()) {
            out += kArgKeyBegin;
            out += std::string(k);
            out += kArgKeyEnd;
            out += '\n';
            out += kArgValueBegin;
            if (v.is_string()) {
                // The template emits string values raw and everything else as
                // JSON.
                std::string_view s;
                if (v.get_string().get(s) != simdjson::SUCCESS) s = {};
                out += std::string(s);
            } else {
                std::string js;
                PyJsonValue(v, js);
                // Jinja's tojson html-escapes; values here are plain scalars.
                out += js;
            }
            out += kArgValueEnd;
            out += '\n';
        }
    }
    out += kToolCallEnd;
    out += '\n';
}

} // namespace

bool HyChatSentinelsPresent(const ArrowTokenizerDb &tok)
{
    return arrow_tokenizer_lookup(tok, kBos, std::strlen(kBos)) >= 0 &&
           arrow_tokenizer_lookup(tok, kUser, std::strlen(kUser)) >= 0 &&
           arrow_tokenizer_lookup(tok, kAssistant, std::strlen(kAssistant)) >= 0 &&
           arrow_tokenizer_lookup(tok, kThinkEnd, std::strlen(kThinkEnd)) >= 0;
}

std::string RenderHyToolCallText(const ToolCall &call)
{
    std::string out;
    AppendToolCallText(call, out);
    return out;
}

std::vector<ToolCall> ParseHyToolCallText(const std::string &text)
{
    std::vector<ToolCall> calls;
    size_t pos = 0;
    while (true) {
        const size_t at = text.find(kToolCallBegin, pos);
        if (at == std::string::npos) break;
        const size_t body = at + std::strlen(kToolCallBegin);
        const size_t sep = text.find(kToolSep, body);
        if (sep == std::string::npos) break;
        const size_t close = text.find(kToolCallEnd, sep);
        if (close == std::string::npos) break;
        ToolCall call;
        call.name = std::string(TrimAscii(
            std::string_view(text).substr(body, sep - body)));
        // Arguments: repeated <arg_key>k</arg_key>\n<arg_value>v</arg_value>.
        std::string args = "{";
        bool first = true;
        size_t p = sep + std::strlen(kToolSep);
        while (true) {
            const size_t kb = text.find(kArgKeyBegin, p);
            if (kb == std::string::npos || kb >= close) break;
            const size_t ke = text.find(kArgKeyEnd, kb);
            if (ke == std::string::npos || ke >= close) break;
            const size_t vb = text.find(kArgValueBegin, ke);
            if (vb == std::string::npos || vb >= close) break;
            const size_t ve = text.find(kArgValueEnd, vb);
            if (ve == std::string::npos || ve >= close) break;
            const std::string key = std::string(TrimAscii(
                std::string_view(text).substr(kb + std::strlen(kArgKeyBegin),
                                              ke - kb -
                                                  std::strlen(kArgKeyBegin))));
            const std::string_view raw(text.data() + vb + std::strlen(kArgValueBegin),
                                       ve - vb - std::strlen(kArgValueBegin));
            if (!first) args += ",";
            first = false;
            args += "\"" + JsonEscape(key) + "\":";
            // A value that is already valid JSON (number/bool/null/object)
            // keeps its type; anything else is a string.
            const std::string_view t = TrimAscii(raw);
            simdjson::dom::parser vp;
            simdjson::dom::element ve_doc;
            const bool json_ok =
                !t.empty() && t.front() != '"' &&
                vp.parse(t).get(ve_doc) == simdjson::SUCCESS &&
                !ve_doc.is_string();
            if (json_ok)
                args += std::string(t);
            else if (!raw.empty() && raw.front() == '"')
                args += std::string(raw);
            else
                args += "\"" + JsonEscape(std::string(raw)) + "\"";
            p = ve + std::strlen(kArgValueEnd);
        }
        args += "}";
        call.arguments_json = std::move(args);
        calls.push_back(std::move(call));
        pos = close + std::strlen(kToolCallEnd);
    }
    for (size_t i = 0; i < calls.size(); ++i)
        if (calls[i].id.empty()) calls[i].id = "call_" + std::to_string(i);
    return calls;
}

bool RenderHyConversation(const ArrowTokenizerDb &tok, const ChatConversation &conv,
                          std::vector<int32_t> &out_ids)
{
    return RenderHyConversation(tok, conv, out_ids, HyRenderPolicy{});
}

bool RenderHyConversation(const ArrowTokenizerDb &tok, const ChatConversation &conv,
                          std::vector<int32_t> &out_ids,
                          const HyRenderPolicy &policy)
{
    out_ids.clear();
    if (!HyChatSentinelsPresent(tok)) return false;

    // System prompt: ALL system messages concatenated (the template folds them
    // into one string before the first marker).
    std::string system;
    size_t last_user = SIZE_MAX;
    for (size_t i = 0; i < conv.messages.size(); ++i) {
        if (conv.messages[i].role == "system") system += conv.messages[i].content;
        if (conv.messages[i].role == "user") last_user = i;
    }

    const bool has_tools = !conv.tools.empty();
    const std::string effort = kDefaultEffort;

    std::string text;
    text += kBos;
    text += system;
    if (has_tools) {
        text += system.empty()
                    ? "# Tools\n\nYou may call one or more functions to assist "
                      "with the user query."
                    : "\n\n# Tools\n\nYou may call one or more functions to "
                      "assist with the user query.";
        text += "\n\nYou are provided with function signatures within "
                "<tools></tools> XML tags:\n<tools>\n";
        for (size_t i = 0; i < conv.tools.size(); ++i) {
            if (i) text += '\n';
            const auto &t = conv.tools[i];
            std::string params = PyJsonDumps(t.parameters_json);
            if (params.empty()) params = "{}";
            // {"type": "function", "function": {"name": ..., "description":
            //  ..., "parameters": <schema>}} in Python's tojson style.
            text += "{\"type\": \"function\", \"function\": {\"name\": ";
            PyJsonString(t.name, text);
            text += ", \"description\": ";
            PyJsonString(t.description, text);
            text += ", \"parameters\": ";
            text += params;
            text += "}}";
        }
        text += "\n</tools>\n\n";
        text += "For function call returns, you should first print ";
        text += kToolCallsBegin;
        text += "\nFor each function call, you should return object like:\n";
        text += kToolCallBegin;
        text += "{function-name}";
        text += kToolSep;
        text += "\n";
        text += kArgKeyBegin;
        text += "{arg-key-1}";
        text += kArgKeyEnd;
        text += "\n";
        text += kArgValueBegin;
        text += "{arg-value-1}";
        text += kArgValueEnd;
        text += "\n";
        text += kArgKeyBegin;
        text += "{arg-key-2}";
        text += kArgKeyEnd;
        text += "\n";
        text += kArgValueBegin;
        text += "{arg-value-2}";
        text += kArgValueEnd;
        text += "\n...\n";
        text += policy.example_call_close_is_calls_end ? kToolCallsEnd
                                                       : kToolCallEnd;
        text += "\n";
        text += "At the end of function call returns, you should print ";
        text += kToolCallsEnd;
        text += kReasoningMode;
        text += "reasoning_effort:";
        text += effort;
    } else {
        text += kReasoningMode;
        text += "reasoning_effort:";
        text += effort;
    }

    bool prev_is_tool = false;
    bool prev_tool_first = true;
    bool last_is_assistant = false;
    for (size_t mi = 0; mi < conv.messages.size(); ++mi) {
        const ChatMessage &m = conv.messages[mi];
        const bool is_last = (mi + 1 == conv.messages.size());
        if (m.role == "system") continue;
        if (m.role == "user") {
            if (prev_is_tool) text += kToolResponsesEnd;
            text += kUser;
            text += m.content;
            prev_is_tool = false;
        }
        if (m.role == "assistant") {
            const std::string content =
                std::string(kThinkBegin) + kThinkEnd + m.content;
            if (prev_is_tool) text += kToolResponsesEnd;
            text += kAssistant;
            if (!m.tool_calls.empty()) {
                prev_tool_first = true;
                text += content;
                text += kToolCallsBegin;
                text += '\n';
                for (const auto &call : m.tool_calls) AppendToolCallText(call, text);
                text += kToolCallsEnd;
                text += kEos;
            } else {
                text += content;
                if (!is_last) {
                    text += kEos;
                }
            }
            prev_is_tool = false;
        }
        if (m.role == "tool") {
            prev_is_tool = true;
            if (prev_tool_first) {
                text += kToolResponsesBegin;
                text += '\n';
                prev_tool_first = false;
            }
            text += kToolResponseBegin;
            text += '\n';
            text += m.content;
            text += '\n';
            text += kToolResponseEnd;
            text += '\n';
        }
        if (is_last && m.role == "assistant") last_is_assistant = true;
    }
    if (prev_is_tool) text += kToolResponsesEnd;
    if (!last_is_assistant) {
        // Template: no_think -> assistant + think begin + think end.
        text += kAssistant;
        text += kThinkBegin;
        text += kThinkEnd;
    }
    (void)last_user;
    return AppendTextWithSentinels(tok, text, out_ids);
}

bool ParseHyToolCalls(const ArrowTokenizerDb &tok,
                      const std::vector<int32_t> &gen_tokens,
                      std::vector<ToolCall> &out_calls)
{
    out_calls.clear();
    std::string decoded;
    {
        std::vector<char> buf(static_cast<size_t>(gen_tokens.size()) * 8 + 64);
        size_t cap = buf.size();
        if (arrow_tokenizer_decode(tok, gen_tokens.data(), gen_tokens.size(),
                                   buf.data(), &cap) != 0)
            return false;
        decoded.assign(buf.data(), cap);
    }
    auto calls = ParseHyToolCallText(decoded);
    if (!calls.empty()) {
        out_calls = std::move(calls);
        return true;
    }
    out_calls = tools::ParseToolCallsText(decoded);
    if (!out_calls.empty()) return true;
    out_calls = tools::ParseBareToolCallJson(decoded);
    return !out_calls.empty();
}

} // namespace lsxcommon::hy
