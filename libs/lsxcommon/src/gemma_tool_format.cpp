#include "lsxcommon/gemma_tool_format.h"

#include <lsxnetwork/json_escape.h>
#include <simdjson.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "lsxcommon/tool_format.h"

namespace lsxcommon {
namespace {

// All Gemma control markers are special tokens (injected via lookup; the BPE
// encoder never resolves added tokens and the decoder drops them).
constexpr const char *kGemmaSentinels[] = {
    "<bos>",        "<eos>",           "<|turn>",         "<turn|>",
    "<|tool>",      "<tool|>",         "<|tool_call>",    "<tool_call|>",
    "<|tool_response>", "<tool_response|>", "<|channel>", "<channel|>",
    "<|\"|>",       "<|think|>",
};

std::string_view Trim(std::string_view s)
{
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t' ||
                          s.front() == '\r' || s.front() == '\n'))
        s.remove_prefix(1);
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t' ||
                          s.back() == '\r' || s.back() == '\n'))
        s.remove_suffix(1);
    return s;
}

using lsxnetwork::json::JsonEscape;

std::string SerializeJson(simdjson::dom::element e);

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

bool AppendGemmaEncoded(const ArrowTokenizerDb &tok, std::string_view text,
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

// --- rendering helpers -----------------------------------------------------

std::string Quote(const std::string &s) { return "<|\"|>" + s + "<|\"|>"; }

std::string FormatValue(simdjson::dom::element e, bool &ok);

std::string FormatObject(simdjson::dom::element obj, bool &ok)
{
    // Sort keys (the template uses dictsort) for stable output.
    std::vector<std::pair<std::string, simdjson::dom::element>> items;
    for (auto [k, v] : obj.get_object()) items.emplace_back(std::string(k), v);
    std::sort(items.begin(), items.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });
    std::string out = "{";
    bool first = true;
    for (auto &[k, v] : items) {
        if (!first) out += ",";
        first = false;
        out += k + ":" + FormatValue(v, ok);
    }
    out += "}";
    return out;
}

std::string FormatValue(simdjson::dom::element e, bool &ok)
{
    using simdjson::dom::element_type;
    switch (e.type()) {
        case element_type::STRING:
            return Quote(std::string(e.get_string().value_unsafe()));
        case element_type::BOOL:
            return e.get_bool().value_unsafe() ? "true" : "false";
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
        case element_type::OBJECT:
            return FormatObject(e, ok);
        case element_type::ARRAY: {
            std::string out = "[";
            bool first = true;
            for (auto v : e.get_array()) {
                if (!first) out += ",";
                first = false;
                out += FormatValue(v, ok);
            }
            out += "]";
            return out;
        }
        default:
            return "null";
    }
}

// `call:NAME{key:value,…}` argument body (keys unquoted, strings quoted).
std::string FormatArguments(std::string_view arguments_json, bool &ok)
{
    simdjson::dom::parser parser;
    simdjson::dom::element args;
    simdjson::padded_string padded(arguments_json);
    if (parser.parse(padded).get(args) != simdjson::SUCCESS ||
        args.type() != simdjson::dom::element_type::OBJECT)
        return "{}";
    return FormatObject(args, ok);
}

std::string Upper(std::string_view s)
{
    std::string out(s);
    for (char &c : out) c = static_cast<char>(std::toupper((unsigned char)c));
    return out;
}

// format_parameters(): `name:{description:…,type:…[,properties:{…}][,required:…]}`
std::string FormatProperties(simdjson::dom::element properties, bool &ok)
{
    std::vector<std::pair<std::string, simdjson::dom::element>> items;
    for (auto [k, v] : properties.get_object()) items.emplace_back(std::string(k), v);
    std::sort(items.begin(), items.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });

    std::string out;
    bool first = true;
    for (auto &[name, schema] : items) {
        if (!first) out += ",";
        first = false;
        out += name + ":{";
        bool wrote = false;
        simdjson::dom::element d;
        if (schema["description"].get(d) == simdjson::SUCCESS &&
            d.type() == simdjson::dom::element_type::STRING) {
            out += "description:" +
                   Quote(std::string(d.get_string().value_unsafe()));
            wrote = true;
        }
        std::string_view type;
        if (schema["type"].get_string().get(type) == simdjson::SUCCESS) {
            if (wrote) out += ",";
            out += "type:" + Quote(Upper(type));
            wrote = true;
        }
        simdjson::dom::element props;
        if (schema["properties"].get(props) == simdjson::SUCCESS &&
            props.type() == simdjson::dom::element_type::OBJECT) {
            if (wrote) out += ",";
            out += "properties:{" + FormatProperties(props, ok) + "}";
            wrote = true;
        }
        simdjson::dom::element req;
        if (schema["required"].get(req) == simdjson::SUCCESS &&
            req.type() == simdjson::dom::element_type::ARRAY) {
            if (wrote) out += ",";
            out += "required:[";
            bool rf = true;
            for (auto r : req.get_array()) {
                if (r.type() != simdjson::dom::element_type::STRING) continue;
                if (!rf) out += ",";
                rf = false;
                out += Quote(std::string(r.get_string().value_unsafe()));
            }
            out += "]";
            wrote = true;
        }
        out += "}";
    }
    return out;
}

std::string Declaration(const ToolSpec &t, bool &ok)
{
    std::string out = "declaration:" + t.name + "{description:" +
                      Quote(t.description);
    simdjson::dom::parser parser;
    simdjson::dom::element params;
    simdjson::padded_string padded(t.parameters_json);
    if (!t.parameters_json.empty() &&
        parser.parse(padded).get(params) == simdjson::SUCCESS &&
        params.type() == simdjson::dom::element_type::OBJECT) {
        out += ",parameters:{";
        bool wrote = false;
        simdjson::dom::element props;
        if (params["properties"].get(props) == simdjson::SUCCESS &&
            props.type() == simdjson::dom::element_type::OBJECT) {
            out += "properties:{" + FormatProperties(props, ok) + "}";
            wrote = true;
        }
        simdjson::dom::element req;
        if (params["required"].get(req) == simdjson::SUCCESS &&
            req.type() == simdjson::dom::element_type::ARRAY) {
            if (wrote) out += ",";
            out += "required:[";
            bool rf = true;
            for (auto r : req.get_array()) {
                if (r.type() != simdjson::dom::element_type::STRING) continue;
                if (!rf) out += ",";
                rf = false;
                out += Quote(std::string(r.get_string().value_unsafe()));
            }
            out += "]";
            wrote = true;
        }
        std::string_view type;
        if (params["type"].get_string().get(type) == simdjson::SUCCESS) {
            if (wrote) out += ",";
            out += "type:" + Quote(Upper(type));
        }
        out += "}";
    }
    out += "}";
    return out;
}

std::string ResolveToolName(const ChatMessage &assistant,
                            const std::string &tool_call_id)
{
    if (!tool_call_id.empty()) {
        for (const ToolCall &c : assistant.tool_calls)
            if (c.id == tool_call_id) return c.name;
    }
    return "unknown";
}

} // namespace

bool RenderGemmaConversation(const ChatConversation &conv, std::string &out)
{
    out.clear();
    static const std::vector<std::string_view> kMarkers = [] {
        std::vector<std::string_view> m;
        for (const char *s : kGemmaSentinels) m.emplace_back(s);
        return m;
    }();
    const auto esc = [](std::string_view s) {
        return tools::NeutralizeControlMarkers(s, kMarkers);
    };

    out += "<bos>";
    const bool first_system =
        !conv.messages.empty() && conv.messages[0].role == "system";
    if (!conv.tools.empty() || first_system) {
        out += "<|turn>system\n";
        if (first_system) out += esc(conv.messages[0].content);
        for (const ToolSpec &t : conv.tools) {
            bool ok = true;
            out += "<|tool>" + Declaration(t, ok) + "<tool|>";
        }
        out += "<turn|>\n";
    }

    for (size_t i = first_system ? 1 : 0; i < conv.messages.size(); ++i) {
        const ChatMessage &m = conv.messages[i];
        if (m.role == "system") {
            out += "<|turn>system\n" + esc(m.content) + "<turn|>\n";
        } else if (m.role == "user") {
            out += "<|turn>user\n" + esc(m.content) + "<turn|>\n";
        } else if (m.role == "assistant") {
            out += "<|turn>model\n" + esc(m.content);
            for (const ToolCall &c : m.tool_calls) {
                bool ok = true;
                // FormatArguments already returns a brace-delimited body
                // (`{k:v,…}`, or `{}` when the arguments do not parse).
                out += "<|tool_call>call:" + esc(c.name) +
                       FormatArguments(c.arguments_json, ok) + "<tool_call|>";
            }
            // Consecutive tool messages follow this model turn.
            size_t j = i + 1;
            while (j < conv.messages.size() &&
                   conv.messages[j].role == "tool") {
                const std::string name =
                    ResolveToolName(m, conv.messages[j].tool_call_id);
                out += "<|tool_response>response:" + name + "{value:" +
                       Quote(esc(conv.messages[j].content)) +
                       "}<tool_response|>";
                ++j;
            }
            out += "<turn|>\n";
            i = j - 1;
        } else if (m.role == "tool") {
            out += "<|tool_response>response:unknown{value:" +
                   Quote(esc(m.content)) + "}<tool_response|>";
        } else {
            return false;
        }
    }

    out += "<|turn>model\n<|channel>thought\n<channel|>";
    return true;
}

std::vector<ToolCall> ParseGemmaToolCallText(std::string_view text)
{
    std::vector<ToolCall> calls;
    size_t pos = 0;
    while (true) {
        const size_t cpos = text.find("call:", pos);
        if (cpos == std::string_view::npos) break;
        // Name runs to the opening brace.
        size_t b = cpos + 5;
        while (b < text.size() && text[b] != '{') ++b;
        if (b >= text.size()) break;
        const std::string name(std::string(Trim(text.substr(cpos + 5, b - (cpos + 5)))));
        if (name.empty()) {
            pos = b + 1;
            continue;
        }
        // Brace-match the argument body.
        int depth = 0;
        size_t e = b;
        for (; e < text.size(); ++e) {
            if (text[e] == '{') ++depth;
            else if (text[e] == '}' && --depth == 0) break;
        }
        if (e >= text.size()) break;
        const std::string_view body = text.substr(b + 1, e - (b + 1));
        pos = e + 1;

        std::string args = "{";
        bool first = true;
        size_t p = 0;
        while (p < body.size()) {
            // key up to top-level ':'
            int d = 0;
            size_t colon = std::string_view::npos, comma = std::string_view::npos;
            for (size_t q = p; q < body.size(); ++q) {
                const char ch = body[q];
                if (ch == '{' || ch == '[') ++d;
                else if (ch == '}' || ch == ']') --d;
                else if (d == 0 && ch == ':' && colon == std::string_view::npos)
                    colon = q;
                else if (d == 0 && ch == ',') { comma = q; break; }
            }
            if (colon == std::string_view::npos) break;
            const std::string key(std::string(Trim(body.substr(p, colon - p))));
            const size_t vstart = colon + 1;
            const size_t vend = (comma == std::string_view::npos) ? body.size()
                                                                 : comma;
            std::string raw(std::string(Trim(body.substr(vstart, vend - vstart))));
            // Drop any residual quote markers.
            while (raw.find("<|\"|>") != std::string::npos)
                raw.erase(raw.find("<|\"|>"), 5);
            p = (comma == std::string_view::npos) ? body.size() : comma + 1;

            if (!first) args += ",";
            first = false;
            args += "\"" + JsonEscape(key) + "\":";
            simdjson::dom::parser parser;
            simdjson::dom::element ev;
            simdjson::padded_string padded(raw);
            if (!raw.empty() &&
                parser.parse(padded).get(ev) == simdjson::SUCCESS)
                args += SerializeJson(ev);
            else
                args += "\"" + JsonEscape(raw) + "\"";
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

bool AppendGemmaChatIds(const ArrowTokenizerDb &tok, const std::string &text,
                        std::vector<int32_t> &out_ids)
{
    out_ids.clear();
    if (arrow_tokenizer_lookup(tok, "<|turn>", 7) < 0) return false;
    size_t i = 0;
    size_t run = 0;
    auto flush = [&](size_t end) -> bool {
        if (end > run &&
            !AppendGemmaEncoded(tok,
                                std::string_view(text).substr(run, end - run),
                                out_ids))
            return false;
        return true;
    };
    while (i < text.size()) {
        bool matched = false;
        for (const char *s : kGemmaSentinels) {
            const size_t n = std::strlen(s);
            if (text.compare(i, n, s) == 0) {
                if (!flush(i)) return false;
                const int32_t id = arrow_tokenizer_lookup(tok, s, n);
                if (id < 0) return false;
                out_ids.push_back(id);
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

bool ParseGemmaTokenCalls(const ArrowTokenizerDb &tok,
                          const std::vector<int32_t> &gen_tokens,
                          std::vector<ToolCall> &out_calls)
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
    out_calls = ParseGemmaToolCallText(text);
    if (out_calls.empty()) out_calls = tools::ParseToolCallsText(text);
    if (out_calls.empty()) out_calls = tools::ParseBareToolCallJson(text);
    return true;
}

} // namespace lsxcommon
