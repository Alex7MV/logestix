#include "lsxcommon/flint_moe_tool_format.h"

#include "flint_moe_constants.h"

#include "lsxcommon/tool_format.h"

#include <lsxnetwork/json_escape.h>

#include <simdjson.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>

namespace lsxcommon::flint {
namespace {

using lsxnetwork::json::JsonEscape;

// Control markers (chat_template.jinja literals).
constexpr const char *kSystem = "<|system|>";
constexpr const char *kDeveloper = "<|developer|>";
constexpr const char *kUser = "<|user|>";
constexpr const char *kAssistant = "<|assistant|>";
constexpr const char *kObservation = "<|observation|>";
constexpr const char *kThinkBegin = "<think>";
constexpr const char *kThinkEnd = "</think>";
constexpr const char *kToolsBegin = "<tools>";
constexpr const char *kToolsEnd = "</tools>";
constexpr const char *kToolCallBegin = "<tool_call>";
constexpr const char *kToolCallEnd = "</tool_call>";
constexpr const char *kArgKeyBegin = "<arg_key>";
constexpr const char *kArgKeyEnd = "</arg_key>";
constexpr const char *kArgValueBegin = "<arg_value>";
constexpr const char *kArgValueEnd = "</arg_value>";
constexpr const char *kToolResponseBegin = "<tool_response>";
constexpr const char *kToolResponseEnd = "</tool_response>";
constexpr const char *kBeginOfText = "<|begin_of_text|>";
constexpr const char *kEndOfText = "<|end_of_text|>";

constexpr const char *kSentinels[] = {
    kBeginOfText,     kEndOfText,   kSystem,        kDeveloper,
    kUser,            kAssistant,   kObservation,   kThinkBegin,
    kThinkEnd,        kToolsBegin,  kToolsEnd,      kToolCallBegin,
    kToolCallEnd,     kArgKeyBegin, kArgKeyEnd,     kArgValueBegin,
    kArgValueEnd,     kToolResponseBegin, kToolResponseEnd,
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

void ReplaceAll(std::string &s, std::string_view from, std::string_view to)
{
    if (from.empty()) return;
    size_t at = 0;
    while ((at = s.find(from.data(), at, from.size())) != std::string::npos) {
        s.replace(at, from.size(), to.data(), to.size());
        at += to.size();
    }
}

// chat_template.jinja's `flint_text` macro: drop NUL, strip literal think tags,
// HTML-escape the role markers so content can never forge a control token.
std::string FlintText(std::string text)
{
    ReplaceAll(text, std::string_view("\0", 1), "");
    ReplaceAll(text, kThinkBegin, "");
    ReplaceAll(text, kThinkEnd, "");
    for (const char *m : {kBeginOfText, kEndOfText, kSystem, kDeveloper, kUser,
                          kAssistant, kObservation}) {
        const std::string esc = "&lt;" + std::string(m + 1, std::strlen(m) - 2) +
                                "&gt;";
        ReplaceAll(text, m, esc);
    }
    return text;
}

// Python `json.dumps(..., ensure_ascii=False)` string form: the quoting and
// escapes the template's `tojson` filter produces, with non-ASCII kept raw.
void PyJsonString(std::string_view s, std::string &out)
{
    out += '"';
    for (char ch : s) {
        const unsigned char c = static_cast<unsigned char>(ch);
        switch (c) {
        case '"': out += "\\\""; continue;
        case '\\': out += "\\\\"; continue;
        case '\n': out += "\\n"; continue;
        case '\r': out += "\\r"; continue;
        case '\t': out += "\\t"; continue;
        case '\b': out += "\\b"; continue;
        case '\f': out += "\\f"; continue;
        default: break;
        }
        if (c < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\u%04x", c);
            out += buf;
        } else {
            out += static_cast<char>(c);
        }
    }
    out += '"';
}

// Python-style JSON value (", " / ": " separators, insertion order, non-ASCII
// raw) matching `| tojson(ensure_ascii=False)`.
void PyJsonValue(const simdjson::dom::element &v, std::string &out)
{
    switch (v.type()) {
    case simdjson::dom::element_type::OBJECT: {
        out += '{';
        bool first = true;
        for (auto [k, val] : v.get_object()) {
            if (!first) out += ", ";
            first = false;
            PyJsonString(k, out);
            out += ": ";
            PyJsonValue(val, out);
        }
        out += '}';
        break;
    }
    case simdjson::dom::element_type::ARRAY: {
        out += '[';
        bool first = true;
        for (auto e : v.get_array()) {
            if (!first) out += ", ";
            first = false;
            PyJsonValue(e, out);
        }
        out += ']';
        break;
    }
    case simdjson::dom::element_type::STRING: {
        std::string_view s;
        if (v.get_string().get(s) != simdjson::SUCCESS) s = {};
        PyJsonString(s, out);
        break;
    }
    case simdjson::dom::element_type::INT64: {
        int64_t x = 0;
        if (v.get_int64().get(x) != simdjson::SUCCESS) x = 0;
        out += std::to_string(x);
        break;
    }
    case simdjson::dom::element_type::UINT64: {
        uint64_t x = 0;
        if (v.get_uint64().get(x) != simdjson::SUCCESS) x = 0;
        out += std::to_string(x);
        break;
    }
    case simdjson::dom::element_type::DOUBLE: {
        double d = 0.0;
        if (v.get_double().get(d) != simdjson::SUCCESS) d = 0.0;
        char buf[40];
        std::snprintf(buf, sizeof(buf), "%g", d);
        out += buf;
        break;
    }
    case simdjson::dom::element_type::BOOL: {
        bool b = false;
        if (v.get_bool().get(b) != simdjson::SUCCESS) b = false;
        out += b ? "true" : "false";
        break;
    }
    case simdjson::dom::element_type::NULL_VALUE:
        out += "null";
        break;
    }
}

std::string PyJsonDumps(std::string_view json)
{
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(json).get(doc) != simdjson::SUCCESS) return {};
    std::string out;
    PyJsonValue(doc, out);
    return out;
}

// The template's own tools turn: its own `<|system|>` marker, the fixed
// instructions, then one `fn | tojson` per tool with no separator.
std::string ToolPreamble(const std::vector<ToolSpec> &tools)
{
    std::string text;
    text += kSystem;
    text += "\n# Tools\n\nYou may call one or more functions to assist with the "
            "user query.\n\nYou are provided with function signatures within "
            "<tools></tools> XML tags:\n";
    text += kToolsBegin;
    for (const auto &t : tools) {
        std::string params = PyJsonDumps(t.parameters_json);
        if (params.empty()) params = "{}";
        text += "{\"name\": ";
        PyJsonString(t.name, text);
        text += ", \"description\": ";
        PyJsonString(t.description, text);
        text += ", \"parameters\": ";
        text += params;
        text += '}';
    }
    // HF's Jinja environment runs with trim_blocks: the newline after the
    // `{%- endfor %}` tag is removed, so `</tools>` follows the last tool
    // object with no separator.
    text += kToolsEnd;
    text += "\n\nFor each function call, output the function name and arguments "
            "within the following XML format:\n";
    text += kToolCallBegin;
    text += "{function-name}";
    text += kArgKeyBegin;
    text += "{arg-key-1}";
    text += kArgKeyEnd;
    text += kArgValueBegin;
    text += "{arg-value-1}";
    text += kArgValueEnd;
    text += "...";
    text += kToolCallEnd;
    text += '\n';
    return text;
}

// One `<tool_call>` block: name then <arg_key>/<arg_value> pairs, no separator.
void AppendToolCallText(const ToolCall &call, std::string &out)
{
    out += kToolCallBegin;
    out += call.name;
    simdjson::dom::parser parser;
    simdjson::dom::element args;
    if (!call.arguments_json.empty() &&
        parser.parse(call.arguments_json).get(args) == simdjson::SUCCESS &&
        args.is_object()) {
        for (auto [k, v] : args.get_object()) {
            out += kArgKeyBegin;
            out += std::string(k);
            out += kArgKeyEnd;
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
                out += js;
            }
            out += kArgValueEnd;
        }
    }
    out += kToolCallEnd;
}

bool AppendEncoded(const ArrowTokenizerDb &tok, std::string_view text,
                   std::vector<int32_t> &ids)
{
    if (text.empty()) return true;
    for (size_t cap = 256; cap <= (1u << 22); cap *= 2) {
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

// Splits `text` on the known markers and injects each marker as its own id.
bool AppendTextWithSentinels(const ArrowTokenizerDb &tok,
                             const std::string &text,
                             std::vector<int32_t> &ids)
{
    size_t i = 0;
    size_t run = 0;
    auto flush = [&](size_t end) -> bool {
        if (end > run)
            return AppendEncoded(
                tok, std::string_view(text).substr(run, end - run), ids);
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

} // namespace

std::string RenderFlintConversationText(const ChatConversation &conv,
                                        bool add_generation_prompt)
{
    const bool has_tools = !conv.tools.empty();
    std::string text;
    bool tools_pending = has_tools;
    for (const auto &m : conv.messages) {
        if (tools_pending && m.role != "system") {
            text += ToolPreamble(conv.tools);
            tools_pending = false;
        }
        if (m.role == "system") {
            text += kSystem;
            text += FlintText(m.content);
        } else if (m.role == "developer") {
            text += kDeveloper;
            text += FlintText(m.content);
        } else if (m.role == "user") {
            text += kUser;
            text += FlintText(m.content);
        } else if (m.role == "assistant") {
            text += kAssistant;
            // reasoning_content is not carried in ChatMessage: the assistant
            // body is the template's `content` slot. A caller that replays
            // reasoning prefixes it with <think>…</think> in `content`.
            if (!m.content.empty()) text += m.content;
            for (const auto &call : m.tool_calls) AppendToolCallText(call, text);
        } else if (m.role == "tool") {
            text += kObservation;
            text += kToolResponseBegin;
            text += FlintText(m.content);
            text += kToolResponseEnd;
        }
    }
    if (tools_pending) text += ToolPreamble(conv.tools);
    if (add_generation_prompt) text += kAssistant;
    return text;
}

bool FlintChatSentinelsPresent(const ArrowTokenizerDb &tok)
{
    return arrow_tokenizer_lookup(tok, kSystem, std::strlen(kSystem)) >= 0 &&
           arrow_tokenizer_lookup(tok, kUser, std::strlen(kUser)) >= 0 &&
           arrow_tokenizer_lookup(tok, kAssistant, std::strlen(kAssistant)) >= 0 &&
           arrow_tokenizer_lookup(tok, kObservation, std::strlen(kObservation)) >= 0;
}

bool RenderFlintConversation(const ArrowTokenizerDb &tok,
                             const ChatConversation &conv,
                             std::vector<int32_t> &out_ids)
{
    out_ids.clear();
    if (!FlintChatSentinelsPresent(tok)) return false;
    const std::string text = RenderFlintConversationText(conv, true);
    return AppendTextWithSentinels(tok, text, out_ids);
}

std::vector<ToolCall> ParseFlintToolCallText(const std::string &text)
{
    std::vector<ToolCall> calls;
    size_t pos = 0;
    while (true) {
        const size_t at = text.find(kToolCallBegin, pos);
        if (at == std::string::npos) break;
        const size_t body = at + std::strlen(kToolCallBegin);
        const size_t close = text.find(kToolCallEnd, body);
        if (close == std::string::npos) break;

        // The name is everything before the first <arg_key> (or before
        // </tool_call> when the call has no arguments).
        size_t name_end = text.find(kArgKeyBegin, body);
        if (name_end == std::string::npos || name_end > close) name_end = close;
        ToolCall call;
        call.name = std::string(TrimAscii(
            std::string_view(text).substr(body, name_end - body)));
        if (call.name.empty()) {
            pos = close + std::strlen(kToolCallEnd);
            continue;
        }

        std::string args = "{";
        bool first = true;
        size_t p = name_end;
        while (true) {
            const size_t kb = text.find(kArgKeyBegin, p);
            if (kb == std::string::npos || kb >= close) break;
            const size_t ke = text.find(kArgKeyEnd, kb);
            if (ke == std::string::npos || ke >= close) break;
            const size_t vb = text.find(kArgValueBegin, ke);
            if (vb == std::string::npos || vb >= close) break;
            const size_t ve = text.find(kArgValueEnd, vb);
            if (ve == std::string::npos || ve >= close) break;
            const std::string key = std::string(TrimAscii(std::string_view(text)
                .substr(kb + std::strlen(kArgKeyBegin),
                        ke - kb - std::strlen(kArgKeyBegin))));
            const std::string_view raw(
                text.data() + vb + std::strlen(kArgValueBegin),
                ve - vb - std::strlen(kArgValueBegin));
            if (!first) args += ",";
            first = false;
            args += "\"" + JsonEscape(key) + "\":";
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

bool ParseFlintToolCalls(const ArrowTokenizerDb &tok,
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
    auto calls = ParseFlintToolCallText(decoded);
    if (!calls.empty()) {
        out_calls = std::move(calls);
        return true;
    }
    out_calls = tools::ParseToolCallsText(decoded);
    if (!out_calls.empty()) return true;
    out_calls = tools::ParseBareToolCallJson(decoded);
    return !out_calls.empty();
}

} // namespace lsxcommon::flint
