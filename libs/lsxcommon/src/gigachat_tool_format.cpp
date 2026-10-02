#include "lsxcommon/gigachat_tool_format.h"

#include <lsxnetwork/json_escape.h>
#include <simdjson.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "lsxcommon/tool_format.h"

namespace lsxcommon {
namespace {

constexpr const char *kRoleSep = "<|role_sep|>\n";
constexpr const char *kMsgSep = "<|message_sep|>\n\n";
constexpr const char *kThink = "<think>";
constexpr const char *kThinkEnd = "</think>";
constexpr const char *kGcml = "｜GCML｜";

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

std::string TsType(simdjson::dom::element schema)
{
    std::string_view type;
    if (schema["type"].get_string().get(type) != simdjson::SUCCESS)
        return "any";
    if (type == "string") return "string";
    if (type == "integer" || type == "number") return "number";
    if (type == "boolean") return "boolean";
    if (type == "array") {
        simdjson::dom::element items;
        if (schema["items"].get(items) == simdjson::SUCCESS)
            return TsType(items) + "[]";
        return "any[]";
    }
    return "any";
}

// json_schema_to_typescript for an object: `{\n  // desc\n  key: type,\n}`.
std::string TsObject(simdjson::dom::element schema)
{
    simdjson::dom::element props;
    if (schema["properties"].get(props) != simdjson::SUCCESS ||
        props.type() != simdjson::dom::element_type::OBJECT)
        return "{}";
    std::vector<std::pair<std::string, simdjson::dom::element>> items;
    for (auto [k, v] : props.get_object()) items.emplace_back(std::string(k), v);
    std::sort(items.begin(), items.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });

    std::string out = "{\n";
    for (auto &[name, ps] : items) {
        std::string_view desc;
        if (ps["description"].get_string().get(desc) == simdjson::SUCCESS)
            out += "  // " + std::string(desc) + "\n";
        out += "  " + name + ": " + TsType(ps) + ",\n";
    }
    out += "}";
    return out;
}

std::string RenderToolNamespace(const std::vector<ToolSpec> &tools)
{
    std::string out = "namespace functions {\n";
    for (const ToolSpec &t : tools) {
        if (!t.description.empty()) {
            // `// ` per non-empty line.
            std::string desc = t.description;
            std::string line;
            for (size_t i = 0; i <= desc.size(); ++i) {
                if (i == desc.size() || desc[i] == '\n') {
                    if (!line.empty()) out += "// " + line + "\n";
                    line.clear();
                } else {
                    line += desc[i];
                }
            }
        }
        simdjson::dom::parser parser;
        simdjson::dom::element params;
        simdjson::dom::element props;
        simdjson::padded_string padded(t.parameters_json);
        const bool has_params =
            !t.parameters_json.empty() &&
            parser.parse(padded).get(params) == simdjson::SUCCESS &&
            params.type() == simdjson::dom::element_type::OBJECT &&
            params["properties"].get(props) == simdjson::SUCCESS &&
            props.type() == simdjson::dom::element_type::OBJECT;
        if (has_params)
            out += "type " + t.name + " = (_: " + TsObject(params) + ") => any;\n";
        else
            out += "type " + t.name + " = () => any;\n";
    }
    out += "} // namespace functions";
    return out;
}

std::string RenderGcmlParameter(const std::string &key,
                                simdjson::dom::element v)
{
    std::string body;
    const bool is_str = v.type() == simdjson::dom::element_type::STRING;
    if (is_str)
        body = std::string(v.get_string().value_unsafe());
    else
        body = SerializeJson(v);
    return "<" + std::string(kGcml) + "parameter name=\"" + key +
           "\" string=\"" + (is_str ? "true" : "false") + "\">" + body + "</" +
           std::string(kGcml) + "parameter>";
}

std::string RenderGcmlToolCalls(const std::vector<ToolCall> &calls)
{
    std::string body = "<" + std::string(kGcml) + "tool_calls>\n";
    bool first_call = true;
    for (const ToolCall &c : calls) {
        if (!first_call) body += "\n";
        first_call = false;
        body += "<" + std::string(kGcml) + "invoke name=\"" + c.name + "\">\n";
        if (!c.arguments_json.empty()) {
            simdjson::dom::parser parser;
            simdjson::dom::element args;
            simdjson::padded_string padded(c.arguments_json);
            if (parser.parse(padded).get(args) == simdjson::SUCCESS &&
                args.type() == simdjson::dom::element_type::OBJECT) {
                bool first = true;
                for (auto [k, v] : args.get_object()) {
                    if (!first) body += "\n";
                    first = false;
                    body += RenderGcmlParameter(std::string(k), v);
                }
                body += "\n";
            }
        }
        body += "</" + std::string(kGcml) + "invoke>";
    }
    body += "\n</" + std::string(kGcml) + "tool_calls>";
    return body;
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

bool RenderGigaChatConversation(const ChatConversation &conv, std::string &out)
{
    out.clear();
    static const std::vector<std::string_view> kMarkers = {
        kRoleSep, kMsgSep, kThink, kThinkEnd, kGcml,
    };
    const auto esc = [](std::string_view s) {
        return tools::NeutralizeControlMarkers(s, kMarkers);
    };

    const bool first_system =
        !conv.messages.empty() && conv.messages[0].role == "system";
    if (first_system || !conv.tools.empty()) {
        out += "system";
        out += kRoleSep;
        if (first_system) out += esc(conv.messages[0].content);
        out += kMsgSep;
    }
    if (!conv.tools.empty()) {
        out += "function descriptions";
        out += kRoleSep;
        out += RenderToolNamespace(conv.tools);
        out += kMsgSep;
    }

    for (size_t i = first_system ? 1 : 0; i < conv.messages.size(); ++i) {
        const ChatMessage &m = conv.messages[i];
        if (m.role == "system") {
            out += "system" + std::string(kRoleSep) + esc(m.content) + kMsgSep;
        } else if (m.role == "user") {
            out += "user" + std::string(kRoleSep) + esc(m.content) + kMsgSep;
        } else if (m.role == "assistant") {
            out += "assistant";
            out += kRoleSep;
            out += kThink;
            out += kThinkEnd;
            if (!m.content.empty()) out += "\n\n" + esc(m.content);
            if (!m.tool_calls.empty())
                out += "\n\n" + RenderGcmlToolCalls(m.tool_calls);
            out += kMsgSep;
        } else if (m.role == "tool") {
            // Name comes from the nearest preceding assistant's tool_calls.
            std::string name = "unknown";
            for (size_t j = i; j-- > 0;) {
                if (conv.messages[j].role == "assistant") {
                    name = ResolveToolName(conv.messages[j], m.tool_call_id);
                    break;
                }
            }
            out += "function result";
            out += kRoleSep;
            out += "<function_response name=\"" + name + "\">" + esc(m.content) +
                   "</function_response>";
            out += kMsgSep;
        } else {
            return false;
        }
    }

    out += "assistant";
    out += kRoleSep;
    out += kThink;
    return true;
}

std::vector<ToolCall> ParseGigaChatToolCallText(std::string_view text)
{
    std::vector<ToolCall> calls;
    const std::string d(kGcml);
    const std::string block_open = d + "tool_calls>";
    const std::string block_close = "</" + d + "tool_calls>";
    const std::string call_prefix = d + "invoke name=\"";
    const std::string call_close = "</" + d + "invoke>";
    const std::string param_prefix = d + "parameter name=\"";
    const std::string param_close = "</" + d + "parameter>";

    size_t pos = 0;
    while (true) {
        const size_t b = text.find(block_open, pos);
        if (b == std::string_view::npos) break;
        const size_t e = text.find(block_close, b + block_open.size());
        if (e == std::string_view::npos) break;
        const std::string_view body =
            text.substr(b + block_open.size(), e - (b + block_open.size()));
        pos = e + block_close.size();

        size_t cp = 0;
        while (true) {
            const size_t cb = body.find(call_prefix, cp);
            if (cb == std::string_view::npos) break;
            const size_t name_start = cb + call_prefix.size();
            const size_t name_q = body.find('"', name_start);
            if (name_q == std::string_view::npos) break;
            const size_t ce = body.find(call_close, name_q);
            if (ce == std::string_view::npos) break;
            const std::string name(body.substr(name_start, name_q - name_start));
            const std::string_view region =
                body.substr(name_q + 1, ce - (name_q + 1));
            cp = ce + call_close.size();

            std::string args = "{";
            bool first = true;
            size_t pp = 0;
            while (true) {
                const size_t pb = region.find(param_prefix, pp);
                if (pb == std::string_view::npos) break;
                const size_t kstart = pb + param_prefix.size();
                const size_t kq = region.find('"', kstart);
                if (kq == std::string_view::npos) break;
                const std::string key(region.substr(kstart, kq - kstart));
                const size_t gt = region.find('>', kq);
                if (gt == std::string_view::npos) break;
                const std::string_view attrs =
                    region.substr(kq + 1, gt - (kq + 1));
                const bool is_string =
                    attrs.find("string=\"true\"") != std::string_view::npos;
                const size_t vend = region.find(param_close, gt + 1);
                if (vend == std::string_view::npos) break;
                const std::string_view raw =
                    region.substr(gt + 1, vend - (gt + 1));
                pp = vend + param_close.size();

                if (!first) args += ",";
                first = false;
                args += "\"" + JsonEscape(key) + "\":";
                if (is_string) {
                    args += "\"" + JsonEscape(raw) + "\"";
                    continue;
                }
                simdjson::dom::parser parser;
                simdjson::dom::element ev;
                simdjson::padded_string padded(raw);
                if (parser.parse(padded).get(ev) == simdjson::SUCCESS)
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
    }
    return calls;
}

bool AppendGigaChatIds(const ArrowTokenizerDb &tok, const std::string &text,
                       std::vector<int32_t> &out_ids)
{
    out_ids.clear();
    if (arrow_tokenizer_lookup(tok, kRoleSep, std::strlen(kRoleSep)) < 0)
        return false;
    const std::string_view sentinels[] = {kRoleSep, kMsgSep, kThink, kThinkEnd,
                                          kGcml};
    size_t i = 0;
    size_t run = 0;
    auto flush = [&](size_t end) -> bool {
        if (end <= run) return true;
        for (size_t cap = 256; cap <= 65536; cap *= 2) {
            std::vector<int32_t> buf(cap, 0);
            size_t enc_cap = cap;
            const int32_t n = arrow_tokenizer_encode(
                tok, text.data() + run, end - run, buf.data(), &enc_cap);
            if (n == -2) continue;
            if (n < 0) return false;
            const size_t begin = (tok.bos_id >= 0 && n > 0) ? 1 : 0;
            out_ids.insert(out_ids.end(), buf.begin() + begin, buf.begin() + n);
            return true;
        }
        return false;
    };
    while (i < text.size()) {
        bool matched = false;
        for (std::string_view s : sentinels) {
            if (text.compare(i, s.size(), s) == 0) {
                if (!flush(i)) return false;
                const int32_t id =
                    arrow_tokenizer_lookup(tok, s.data(), s.size());
                if (id < 0) return false;
                out_ids.push_back(id);
                i += s.size();
                run = i;
                matched = true;
                break;
            }
        }
        if (!matched) ++i;
    }
    return flush(text.size());
}

bool ParseGigaChatTokenCalls(const ArrowTokenizerDb &tok,
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
    out_calls = ParseGigaChatToolCallText(text);
    if (out_calls.empty()) out_calls = tools::ParseToolCallsText(text);
    if (out_calls.empty()) out_calls = tools::ParseBareToolCallJson(text);
    return true;
}

} // namespace lsxcommon
