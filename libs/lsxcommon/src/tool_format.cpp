#include "lsxcommon/tool_format.h"

#include <lsxnetwork/json_escape.h>
#include <simdjson.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace lsxcommon {
namespace tools {
namespace {

using lsxnetwork::json::JsonEscape;

// Minimal recursive serializer for a simdjson DOM element (compact JSON).
std::string Serialize(simdjson::dom::element e);

std::string Serialize(simdjson::dom::element e) {
    using simdjson::dom::element_type;
    switch (e.type()) {
        case element_type::OBJECT: {
            std::string out = "{";
            bool first = true;
            for (auto [k, v] : e.get_object()) {
                if (!first) out += ",";
                first = false;
                out += "\"" + JsonEscape(std::string(k)) + "\":" + Serialize(v);
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
                out += Serialize(v);
            }
            out += "]";
            return out;
        }
        case element_type::STRING:
            return "\"" + JsonEscape(std::string(e.get_string().value_unsafe())) + "\"";
        case element_type::INT64:
            return std::to_string(e.get_int64().value_unsafe());
        case element_type::UINT64:
            return std::to_string(e.get_uint64().value_unsafe());
        case element_type::DOUBLE: {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.17g", e.get_double().value_unsafe());
            return std::string(buf);
        }
        case element_type::BOOL:
            return e.get_bool().value_unsafe() ? "true" : "false";
        default:
            return "null";
    }
}

}  // namespace

std::string RenderToolsBlock(const std::vector<ToolSpec>& tools) {
    if (tools.empty()) return {};
    std::string out = "<tools>\n[";
    for (std::size_t i = 0; i < tools.size(); ++i) {
        const ToolSpec& t = tools[i];
        if (i) out += ",";
        out += "{\"type\":\"function\",\"function\":{\"name\":\"" +
               JsonEscape(t.name) + "\",\"description\":\"" +
               JsonEscape(t.description) + "\",\"parameters\":" +
               (t.parameters_json.empty() ? std::string("{}")
                                          : t.parameters_json) +
               "}}";
    }
    out += "]\n</tools>";
    return out;
}

std::string RenderConversation(const ChatConversation& conv, bool include_tools) {
    std::string out;
    const std::string tools_block =
        include_tools ? RenderToolsBlock(conv.tools) : std::string{};
    if (!tools_block.empty()) out += tools_block + "\n";
    for (const ChatMessage& m : conv.messages) {
        if (!out.empty()) out += "\n";
        out += m.role + ": " + m.content;
        for (const ToolCall& c : m.tool_calls)
            out += "\n<tool_call>{\"name\":\"" + JsonEscape(c.name) +
                   "\",\"arguments\":" +
                   (c.arguments_json.empty() ? std::string("{}")
                                             : c.arguments_json) +
                   "}</tool_call>";
    }
    return out;
}

std::vector<ToolCall> ParseToolCallsText(std::string_view text) {
    std::vector<ToolCall> calls;
    constexpr std::string_view kOpen = "<tool_call>";
    constexpr std::string_view kClose = "</tool_call>";
    std::size_t pos = 0;
    while (true) {
        const std::size_t b = text.find(kOpen, pos);
        if (b == std::string_view::npos) break;
        const std::size_t e = text.find(kClose, b + kOpen.size());
        if (e == std::string_view::npos) break;
        std::string_view inner =
            text.substr(b + kOpen.size(), e - (b + kOpen.size()));
        pos = e + kClose.size();
        while (!inner.empty() &&
               (inner.front() == '\n' || inner.front() == ' ' ||
                inner.front() == '\t' || inner.front() == '\r'))
            inner.remove_prefix(1);
        while (!inner.empty() &&
               (inner.back() == '\n' || inner.back() == ' ' ||
                inner.back() == '\t' || inner.back() == '\r'))
            inner.remove_suffix(1);
        if (inner.empty()) continue;

        simdjson::dom::parser parser;
        simdjson::dom::element obj;
        simdjson::padded_string padded(inner);
        if (parser.parse(padded).get(obj) != simdjson::SUCCESS) continue;
        if (obj.type() != simdjson::dom::element_type::OBJECT) continue;

        ToolCall call;
        if (auto n = obj["name"].get_string(); !n.error())
            call.name.assign(n.value_unsafe());
        if (call.name.empty()) continue;

        simdjson::dom::element args;
        if (obj["arguments"].get(args) == simdjson::SUCCESS) {
            if (args.type() == simdjson::dom::element_type::STRING)
                call.arguments_json.assign(args.get_string().value_unsafe());
            else
                call.arguments_json = Serialize(args);
        }
        if (call.arguments_json.empty()) call.arguments_json = "{}";
        call.id = "call_" + std::to_string(calls.size());
        calls.push_back(std::move(call));
    }
    return calls;
}

std::vector<ToolCall> ParseBareToolCallJson(std::string_view text)
{
    std::vector<ToolCall> calls;
    for (size_t i = 0; i < text.size();) {
        if (text[i] != '{') {
            ++i;
            continue;
        }
        // Brace-match, honoring strings/escapes, to isolate one candidate object.
        int depth = 0;
        bool in_str = false, esc = false;
        size_t j = i;
        for (; j < text.size(); ++j) {
            const char c = text[j];
            if (in_str) {
                if (esc) esc = false;
                else if (c == '\\') esc = true;
                else if (c == '"') in_str = false;
            } else if (c == '"') {
                in_str = true;
            } else if (c == '{') {
                ++depth;
            } else if (c == '}') {
                if (--depth == 0) break;
            }
        }
        if (j >= text.size()) break;  // unterminated
        const std::string_view cand = text.substr(i, j - i + 1);
        i = j + 1;

        simdjson::dom::parser parser;
        simdjson::dom::element obj;
        simdjson::padded_string padded(cand);
        if (parser.parse(padded).get(obj) != simdjson::SUCCESS) continue;
        if (obj.type() != simdjson::dom::element_type::OBJECT) continue;

        std::string_view name;
        if (obj["name"].get_string().get(name) != simdjson::SUCCESS ||
            name.empty())
            continue;

        ToolCall call;
        call.name.assign(name);
        simdjson::dom::element args;
        if (obj["arguments"].get(args) == simdjson::SUCCESS) {
            if (args.type() == simdjson::dom::element_type::STRING)
                call.arguments_json.assign(args.get_string().value_unsafe());
            else
                call.arguments_json = Serialize(args);
        }
        if (call.arguments_json.empty()) call.arguments_json = "{}";
        call.id = "call_" + std::to_string(calls.size());
        calls.push_back(std::move(call));
    }
    return calls;
}

std::string NeutralizeControlMarkers(
    std::string_view text, const std::vector<std::string_view> &markers)
{
    // U+200B ZERO WIDTH SPACE, inserted after the first byte of a matched
    // marker so the marker no longer matches verbatim.
    static constexpr std::string_view kZwsp = "\xe2\x80\x8b";
    std::string out;
    out.reserve(text.size() + 16);
    size_t i = 0;
    while (i < text.size()) {
        bool matched = false;
        for (std::string_view m : markers) {
            if (m.size() >= 2 && text.compare(i, m.size(), m) == 0) {
                out += text[i];
                out += kZwsp;
                ++i;
                matched = true;
                break;
            }
        }
        if (!matched) {
            out += text[i];
            ++i;
        }
    }
    return out;
}

// ── Shared JSON/framing primitives (dedup phase 4) ──────────────────────
// Bodies moved verbatim from the byte-identical family copies (see the
// header for which family each shape belongs to).

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

std::string SerializeJson(simdjson::dom::element e)
{
    using simdjson::dom::element_type;
    switch (e.type()) {
        case element_type::OBJECT: {
            std::string out = "{";
            bool first = true;
            for (auto [k, v] : e.get_object()) {
                if (!first) out += ",";
                first = false;
                out += "\"" + JsonEscape(k) + "\":" + SerializeJson(v);
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
                out += SerializeJson(v);
            }
            out += "]";
            return out;
        }
        case element_type::STRING:
            return "\"" + JsonEscape(e.get_string().value_unsafe()) + "\"";
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

void AppendJinjaString(std::string &out, std::string_view s)
{
    out.push_back('"');
    for (char ch : s) {
        const unsigned char c = static_cast<unsigned char>(ch);
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\b': out += "\\b"; break;
        case '\f': out += "\\f"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (c < 0x20) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                out += buf;
            } else {
                out.push_back(ch);
            }
        }
    }
    out.push_back('"');
}

bool DecodeAndParseToolCalls(const ArrowTokenizerDb &tok,
                             const std::vector<int32_t> &gen_tokens,
                             std::vector<ToolCall> &out_calls,
                             std::vector<ToolCall> (*parse)(std::string_view))
{
    out_calls.clear();
    if (gen_tokens.empty()) return true;
    std::string text(4096, '\0');
    std::size_t cap = text.size();
    for (int attempt = 0; attempt < 8; ++attempt) {
        const int rc = arrow_tokenizer_decode(
            tok, gen_tokens.data(), gen_tokens.size(), text.data(), &cap);
        if (rc == 0) {
            text.resize(cap);
            out_calls = parse(text);
            return true;
        }
        if (rc == -2 && cap > text.size() && cap <= (1u << 20)) {
            text.resize(cap);
            continue;
        }
        break;
    }
    return false;
}

bool AppendEncoded(const ArrowTokenizerDb &tok, std::string_view text,
                   std::vector<int32_t> &ids)
{
    for (size_t cap = 256; cap <= 65536; cap *= 2) {
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

bool AppendEncodedLarge(const ArrowTokenizerDb &tok, std::string_view text,
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

bool AppendTextWithSentinels(const ArrowTokenizerDb &tok,
                             const std::string &text,
                             const std::vector<std::string_view> &sentinels,
                             std::vector<int32_t> &ids)
{
    size_t i = 0;
    size_t run = 0;
    auto flush = [&](size_t end) -> bool {
        if (end > run)
            return AppendEncodedLarge(
                tok, std::string_view(text).substr(run, end - run), ids);
        return true;
    };
    while (i < text.size()) {
        bool matched = false;
        for (std::string_view s : sentinels) {
            const size_t n = s.size();
            if (text.compare(i, n, s) != 0) continue;
            if (!flush(i)) return false;
            const int32_t id = arrow_tokenizer_lookup(tok, s.data(), n);
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

}  // namespace tools
}  // namespace lsxcommon
