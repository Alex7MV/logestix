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

constexpr const char *kSentinels[] = {
    kBos,       kEos,           kUser,             kAssistant,
    kThinkBegin, kThinkEnd,     kReasoningMode,    kToolCallsBegin,
    kToolCallsEnd, kToolCallBegin, kToolCallEnd,   kToolSep,
    kArgKeyBegin, kArgKeyEnd,   kArgValueBegin,    kArgValueEnd,
    kToolResponsesBegin, kToolResponsesEnd, kToolResponseBegin,
    kToolResponseEnd,
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

// Python json.dumps string escaping with ensure_ascii=True (what Jinja's
// `tojson` produces for the tool blobs; `<`, `>`, `&`, `'` are additionally
// escaped by Jinja's html-safe wrapper).
void PyJsonString(std::string_view s, std::string &out)
{
    out += '"';
    for (size_t i = 0; i < s.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        switch (c) {
        case '"': out += "\\\""; continue;
        case '\\': out += "\\\\"; continue;
        case '\n': out += "\\n"; continue;
        case '\r': out += "\\r"; continue;
        case '\t': out += "\\t"; continue;
        case '\b': out += "\\b"; continue;
        case '\f': out += "\\f"; continue;
        case '<': out += "\\u003c"; continue;
        case '>': out += "\\u003e"; continue;
        case '&': out += "\\u0026"; continue;
        case '\'': out += "\\u0027"; continue;
        default: break;
        }
        if (c < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\u%04x", c);
            out += buf;
        } else if (c < 0x80) {
            out += static_cast<char>(c);
        } else {
            // Decode one UTF-8 codepoint and emit \uXXXX (surrogate pair above
            // the BMP), matching json.dumps(ensure_ascii=True).
            int len = 1;
            unsigned cp = c;
            if ((c & 0xE0) == 0xC0) { len = 2; cp = c & 0x1F; }
            else if ((c & 0xF0) == 0xE0) { len = 3; cp = c & 0x0F; }
            else if ((c & 0xF8) == 0xF0) { len = 4; cp = c & 0x07; }
            for (int k = 1; k < len && i + k < s.size(); ++k)
                cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
            i += static_cast<size_t>(len - 1);
            char buf[16];
            if (cp >= 0x10000) {
                const unsigned v = cp - 0x10000;
                std::snprintf(buf, sizeof(buf), "\\u%04x\\u%04x",
                              static_cast<unsigned>(0xD800 + (v >> 10)),
                              static_cast<unsigned>(0xDC00 + (v & 0x3FF)));
            } else {
                std::snprintf(buf, sizeof(buf), "\\u%04x", cp);
            }
            out += buf;
        }
    }
    out += '"';
}

// Re-emits a JSON value in Python's json.dumps style (", " / ": " separators,
// insertion order preserved) so the rendered tool block is byte-identical to
// the template's `tool | tojson`.
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
    case simdjson::dom::element_type::BIGINT:
        out += simdjson::to_string(v);
        break;
    case simdjson::dom::element_type::NULL_VALUE:
        out += "null";
        break;
    }
}

// Appends the BPE encoding of `text` (never a marker).
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
        // Strip the auto-prepended BOS only when it is actually there: the
        // single-stage tokenizer kinds (7..10) do not prepend one, and
        // an unconditional strip would eat the first real token.
        const size_t begin =
            (tok.bos_id >= 0 && n > 0 && buf[0] == tok.bos_id) ? 1 : 0;
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

std::string PyJsonDumps(std::string_view json)
{
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(json).get(doc) != simdjson::SUCCESS) return {};
    std::string out;
    PyJsonValue(doc, out);
    return out;
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
