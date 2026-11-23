#include "lsxcommon/hy_moe_flash_tool_format.h"

#include "lsxcommon/tool_format.h"

#include <lsxnetwork/json_escape.h>

#include <simdjson.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>

namespace lsxcommon {
namespace hy4 {
namespace {

using lsxnetwork::json::JsonEscape;

// chat_template.jinja literals, with `HYTK = ':opensource'`.
constexpr const char *kStart = "<｜hy_start:opensource｜>";
constexpr const char *kMiddle = "<｜hy_middle:opensource｜>";
constexpr const char *kEnd = "<｜hy_end:opensource｜>";
constexpr const char *kThinkBegin = "<think:opensource>";
constexpr const char *kThinkEnd = "</think:opensource>";
constexpr const char *kToolsBegin = "<tools>";
constexpr const char *kToolsEnd = "</tools>";
constexpr const char *kToolCallsBegin = "<tool_calls:opensource>";
constexpr const char *kToolCallsEnd = "</tool_calls:opensource>";
constexpr const char *kToolCallBegin = "<tool_call:opensource>";
constexpr const char *kToolCallEnd = "</tool_call:opensource>";
constexpr const char *kArgKeyBegin = "<arg_key:opensource>";
constexpr const char *kArgKeyEnd = "</arg_key:opensource>";
constexpr const char *kArgValueBegin = "<arg_value:opensource>";
constexpr const char *kArgValueEnd = "</arg_value:opensource>";
constexpr const char *kToolResponseBegin = "<tool_response:opensource>";
constexpr const char *kToolResponseEnd = "</tool_response:opensource>";
constexpr const char *kReasoningMode = "<｜reasoning_mode:opensource｜>";

// Every marker that must exist in the vocabulary as its own id.
constexpr const char *kSentinels[] = {
    kStart,        kMiddle,        kEnd,           kThinkBegin,
    kThinkEnd,     kToolCallsBegin, kToolCallsEnd, kToolCallBegin,
    kToolCallEnd,  kArgKeyBegin,   kArgKeyEnd,     kArgValueBegin,
    kArgValueEnd,  kToolResponseBegin, kToolResponseEnd, kReasoningMode,
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

// Content can never forge a control marker: the template's own role markers are
// escaped and stray think tags are dropped.
std::string HunyuanText(std::string text)
{
    ReplaceAll(text, std::string_view("\0", 1), "");
    ReplaceAll(text, kThinkBegin, "");
    ReplaceAll(text, kThinkEnd, "");
    // A marker is `<` + U+FF5C + name + U+FF5C + `>`. The delimiters are 4 bytes
    // on each side (1 ASCII + 3 UTF-8), so a one-byte strip leaves malformed
    // UTF-8 and a 3-byte strip keeps a bar.
    constexpr size_t kEdge = 4;
    for (const char *m : {kStart, kMiddle, kEnd, kReasoningMode}) {
        const size_t len = std::strlen(m);
        if (len <= 2 * kEdge) continue;
        const std::string inner(m + kEdge, len - 2 * kEdge);
        ReplaceAll(text, m, "&lt;" + inner + "&gt;");
    }
    return text;
}

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

std::string PyJsonDumps(std::string_view json)
{
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(json).get(doc) != simdjson::SUCCESS) return {};
    std::string out;
    PyJsonValue(doc, out);
    return out;
}

// chat_template.jinja's `render_tools_prompt`, byte for byte.
void AppendToolsPrompt(const std::vector<ToolSpec> &tools, std::string &out)
{
    out += "# Tools\n\nYou may call one or more functions to assist with the "
           "user query.";
    out += "\n\nYou are provided with function signatures within "
           "<tools></tools> XML tags:";
    out += "\n<tools>\n";
    bool first = true;
    for (const auto &t : tools) {
        if (!first) out += '\n';
        first = false;
        std::string params = PyJsonDumps(t.parameters_json);
        if (params.empty()) params = "{}";
        out += "{\"name\": ";
        PyJsonString(t.name, out);
        out += ", \"description\": ";
        PyJsonString(t.description, out);
        out += ", \"parameters\": ";
        out += params;
        out += '}';
    }
    out += "\n</tools>\n\n";
    out += "For function call returns, you should first print ";
    out += kToolCallsBegin;
    out += "\nFor each function call, you should return object like:\n";
    out += kToolCallBegin;
    out += "{function-name}";
    out += kArgKeyBegin;
    out += "{arg-key-1}";
    out += kArgKeyEnd;
    out += kArgValueBegin;
    out += "{arg-value-1}";
    out += kArgValueEnd;
    out += kArgKeyBegin;
    out += "{arg-key-2}";
    out += kArgKeyEnd;
    out += kArgValueBegin;
    out += "{arg-value-2}";
    out += kArgValueEnd;
    out += "...";
    out += kToolCallEnd;
    out += "\nAt the end of function call returns, you should print ";
    out += kToolCallsEnd;
}

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
        const size_t begin =
            (tok.bos_id >= 0 && n > 0 && buf[0] == tok.bos_id) ? 1 : 0;
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

std::string RenderHy4ConversationText(const ChatConversation &conv,
                                      bool add_generation_prompt,
                                      std::string_view reasoning_effort)
{
    const bool no_think = reasoning_effort == "no_think";
    const bool has_tools = !conv.tools.empty();
    std::string text;

    // The template synthesizes a leading system turn when the caller supplies
    // none, carrying the tools block and the reasoning-mode marker.
    bool has_leading_system = false;
    if (!conv.messages.empty() && conv.messages.front().role == "system")
        has_leading_system = true;
    if (!has_leading_system) {
        text += kStart;
        text += "system";
        text += kMiddle;
        if (has_tools) AppendToolsPrompt(conv.tools, text);
        text += kReasoningMode;
        text += "reasoning_effort:";
        text += no_think ? "no_think" : "high";
        text += kEnd;
    }

    // The template attaches the tools block to the LEADING system turn. When the
    // caller supplied none, the synthesized turn above already carries it, so
    // nothing is pending here.
    bool tools_pending = has_tools && has_leading_system;
    for (const auto &m : conv.messages) {
        if (m.role == "system") {
            text += kStart;
            text += "system";
            text += kMiddle;
            if (tools_pending) {
                AppendToolsPrompt(conv.tools, text);
                tools_pending = false;
            }
            text += HunyuanText(m.content);
            if (has_tools && !m.content.empty()) text += "\n\n";
            text += kReasoningMode;
            text += "reasoning_effort:";
            text += no_think ? "no_think" : "high";
            text += kEnd;
        } else if (m.role == "user") {
            text += kStart;
            text += "user";
            text += kMiddle;
            if (tools_pending) {
                AppendToolsPrompt(conv.tools, text);
                tools_pending = false;
            }
            text += HunyuanText(m.content);
            text += kEnd;
        } else if (m.role == "assistant") {
            text += kStart;
            text += "assistant";
            text += kMiddle;
            // History turns carry an empty think block unless the caller
            // replayed reasoning inside `content`.
            if (no_think || m.content.rfind(kThinkBegin, 0) != 0) {
                text += kThinkBegin;
                text += kThinkEnd;
            }
            text += m.content;
            if (!m.tool_calls.empty()) {
                text += kToolCallsBegin;
                for (const auto &call : m.tool_calls)
                    AppendToolCallText(call, text);
                text += kToolCallsEnd;
            }
            text += kEnd;
        } else if (m.role == "tool") {
            text += kStart;
            text += "tool";
            text += kMiddle;
            text += kToolResponseBegin;
            text += HunyuanText(m.content);
            text += kToolResponseEnd;
            text += kEnd;
        } else {
            text += kStart;
            text += m.role;
            text += kMiddle;
            text += HunyuanText(m.content);
            text += kEnd;
        }
    }

    if (add_generation_prompt) {
        text += kStart;
        text += "assistant";
        text += kMiddle;
        text += kThinkBegin;
        if (no_think) text += kThinkEnd;
    }
    return text;
}

bool Hy4ChatSentinelsPresent(const ArrowTokenizerDb &tok)
{
    for (const char *s : {kStart, kMiddle, kEnd, kThinkBegin, kThinkEnd,
                          kToolCallBegin, kToolCallEnd}) {
        if (arrow_tokenizer_lookup(tok, s, std::strlen(s)) < 0) return false;
    }
    return true;
}

bool RenderHy4Conversation(const ArrowTokenizerDb &tok,
                           const ChatConversation &conv,
                           std::vector<int32_t> &out_ids)
{
    out_ids.clear();
    if (!Hy4ChatSentinelsPresent(tok)) return false;
    // Serving default: the checkpoint's own template defaults to 'high'; a
    // direct answer is what the gates and the OpenAI surface expect unless the
    // client asks to reason (ChatConversation has no field for it yet, so the
    // knob is the LSX_HY4_REASONING_EFFORT env var, recorded in the report).
    const char *env = std::getenv("LSX_HY4_REASONING_EFFORT");
    const std::string effort = (env && std::strcmp(env, "high") == 0)
                                   ? std::string("high")
                                   : std::string("no_think");
    const std::string text = RenderHy4ConversationText(conv, true, effort);
    return AppendTextWithSentinels(tok, text, out_ids);
}

std::vector<ToolCall> ParseHy4ToolCallText(const std::string &text)
{
    std::vector<ToolCall> calls;
    size_t pos = 0;
    while (true) {
        const size_t at = text.find(kToolCallBegin, pos);
        if (at == std::string::npos) break;
        const size_t body = at + std::strlen(kToolCallBegin);
        const size_t close = text.find(kToolCallEnd, body);
        if (close == std::string::npos) break;

        size_t name_end = text.find(kArgKeyBegin, body);
        if (name_end == std::string::npos || name_end > close) name_end = close;
        ToolCall call;
        call.name = std::string(
            TrimAscii(std::string_view(text).substr(body, name_end - body)));
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
            const std::string key(TrimAscii(std::string_view(text).substr(
                kb + std::strlen(kArgKeyBegin),
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

bool ParseHy4ToolCalls(const ArrowTokenizerDb &tok,
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
    auto calls = ParseHy4ToolCallText(decoded);
    if (!calls.empty()) {
        out_calls = std::move(calls);
        return true;
    }
    out_calls = tools::ParseToolCallsText(decoded);
    if (!out_calls.empty()) return true;
    out_calls = tools::ParseBareToolCallJson(decoded);
    return !out_calls.empty();
}

} // namespace hy4
} // namespace lsxcommon
