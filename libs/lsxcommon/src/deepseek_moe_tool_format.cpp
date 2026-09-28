#include "lsxcommon/deepseek_moe_tool_format.h"

#include <simdjson.h>

#include <cstdio>
#include <cstring>
#include <string>

#include "lsxcommon/tool_format.h"

namespace lsxcommon {
namespace {

constexpr std::string_view kBos = "<｜begin▁of▁sentence｜>";
constexpr std::string_view kEos = "<｜end▁of▁sentence｜>";
constexpr std::string_view kSystem = "<｜System｜>";
constexpr std::string_view kUser = "<｜User｜>";
constexpr std::string_view kAssistant = "<｜Assistant｜>";
constexpr std::string_view kThink = "<think>";
constexpr std::string_view kThinkEnd = "</think>";
constexpr std::string_view kDsml = "｜DSML｜";

std::string JsonEscape(std::string_view in)
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

std::string ToolJson(const ToolSpec &t)
{
    return "{\"type\":\"function\",\"function\":{\"name\":\"" +
           JsonEscape(t.name) + "\",\"description\":\"" +
           JsonEscape(t.description) + "\",\"parameters\":" +
           (t.parameters_json.empty() ? std::string("{}")
                                      : t.parameters_json) +
           "}}";
}

std::string RenderToolsBlock(const std::vector<ToolSpec> &tools,
                             const DsmlTags &tags)
{
    const std::string d(kDsml);
    std::string out =
        "## Tools\n\n"
        "You have access to a set of tools to help answer the user's question. "
        "You can invoke tools by writing a \"" +
        d + std::string(tags.block) +
        "\" block like the following:\n\n" + d + std::string(tags.block) +
        "\n" + d + std::string(tags.call) +
        " name=\"$TOOL_NAME\">\n" + d + std::string(tags.param) +
        " name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</" + d +
        std::string(tags.param) + ">\n...\n</" + d + std::string(tags.call) +
        ">\n" + d + std::string(tags.call) +
        " name=\"$TOOL_NAME2\">\n...\n</" + d + std::string(tags.call) + ">\n</" +
        d + std::string(tags.block) +
        ">\n\n"
        "String parameters should be specified as is and set `string=\"true\"`. "
        "For all other types (numbers, booleans, arrays, objects), pass the "
        "value in JSON format and set `string=\"false\"`.\n\n"
        "If thinking_mode is enabled (triggered by <think>), you MUST output "
        "your complete reasoning inside <think>...</think> BEFORE any tool "
        "calls or final response.\n\n"
        "Otherwise, output directly after </think> with tool calls or final "
        "response.\n\n"
        "### Available Tool Schemas\n\n";
    for (size_t i = 0; i < tools.size(); ++i) {
        if (i) out += "\n";
        out += ToolJson(tools[i]);
    }
    out +=
        "\n\nYou MUST strictly follow the above defined tool name and parameter "
        "schemas to invoke tool calls.\n";
    return out;
}

std::string RenderToolCalls(const std::vector<ToolCall> &calls,
                            const DsmlTags &tags)
{
    const std::string d(kDsml);
    std::string body;
    for (const ToolCall &c : calls) {
        body += "<" + d + std::string(tags.call) + " name=\"" + c.name +
                "\">\n";
        if (!c.arguments_json.empty()) {
            simdjson::dom::parser parser;
            simdjson::dom::element args;
            simdjson::padded_string padded(c.arguments_json);
            if (parser.parse(padded).get(args) == simdjson::SUCCESS &&
                args.type() == simdjson::dom::element_type::OBJECT) {
                for (auto [k, v] : args.get_object()) {
                    const bool is_str =
                        v.type() == simdjson::dom::element_type::STRING;
                    body += "<" + d + std::string(tags.param) + " name=\"" +
                            std::string(k) + "\" string=\"" +
                            (is_str ? "true" : "false") + "\">";
                    body += is_str ? std::string(v.get_string().value_unsafe())
                                   : SerializeJson(v);
                    body += "</" + d + std::string(tags.param) + ">\n";
                }
            }
        }
        body += "</" + d + std::string(tags.call) + ">\n";
    }
    return "\n\n<" + d + std::string(tags.block) + ">\n" + body + "</" + d +
           std::string(tags.block) + ">";
}

} // namespace

bool RenderDsmlConversation(const ChatConversation &conv, const DsmlTags &tags,
                            std::string &out)
{
    out.clear();
    out += kBos;
    const bool first_system =
        !conv.messages.empty() && conv.messages[0].role == "system";
    if (first_system) {
        out += kSystem;
        out += conv.messages[0].content;
        if (!conv.tools.empty())
            out += "\n\n" + RenderToolsBlock(conv.tools, tags);
    }

    const auto is_last_turn = [&](size_t i) {
        return i + 1 >= conv.messages.size();
    };

    for (size_t i = 0; i < conv.messages.size(); ++i) {
        const ChatMessage &m = conv.messages[i];
        if (m.role == "system") {
            if (i == 0) continue;  // rendered above
            out += kSystem;
            out += m.content;
            if (is_last_turn(i)) { out += kAssistant; out += kThinkEnd; }
        } else if (m.role == "user") {
            out += kUser;
            out += m.content;
            if (is_last_turn(i)) { out += kAssistant; out += kThinkEnd; }
        } else if (m.role == "assistant") {
            out += kAssistant;
            out += kThinkEnd;
            out += m.content;
            if (!m.tool_calls.empty())
                out += RenderToolCalls(m.tool_calls, tags);
            out += kEos;
        } else if (m.role == "tool") {
            // Tool results ride a <｜User｜> turn with <tool_result> blocks;
            // consecutive tool messages share one turn.
            bool starts_group = i == 0 || conv.messages[i - 1].role != "tool";
            if (starts_group) out += kUser;
            else out += "\n\n";
            out += "<tool_result>" + m.content + "</tool_result>";
            if (is_last_turn(i)) { out += kAssistant; out += kThinkEnd; }
        } else {
            return false;
        }
    }
    return true;
}

std::vector<ToolCall> ParseDsmlToolCalls(std::string_view text,
                                         const DsmlTags &tags)
{
    std::vector<ToolCall> calls;
    const std::string d(kDsml);
    const std::string block_open = "<" + d + std::string(tags.block) + ">";
    const std::string block_close = "</" + d + std::string(tags.block) + ">";
    const std::string call_prefix = "<" + d + std::string(tags.call) + " name=\"";
    const std::string call_close = "</" + d + std::string(tags.call) + ">";
    const std::string param_prefix =
        "<" + d + std::string(tags.param) + " name=\"";
    const std::string param_close = "</" + d + std::string(tags.param) + ">";

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
            const std::string_view args_region =
                body.substr(name_q + 1, ce - (name_q + 1));
            cp = ce + call_close.size();

            std::string args = "{";
            bool first = true;
            size_t pp = 0;
            while (true) {
                const size_t pb = args_region.find(param_prefix, pp);
                if (pb == std::string_view::npos) break;
                const size_t kstart = pb + param_prefix.size();
                const size_t kq = args_region.find('"', kstart);
                if (kq == std::string_view::npos) break;
                const std::string key(
                    args_region.substr(kstart, kq - kstart));
                // ... string="true|false">VALUE</... parameter>
                const size_t gt = args_region.find('>', kq);
                if (gt == std::string_view::npos) break;
                const std::string_view attrs =
                    args_region.substr(kq + 1, gt - (kq + 1));
                const bool is_string =
                    attrs.find("string=\"true\"") != std::string_view::npos;
                const size_t vend = args_region.find(param_close, gt + 1);
                if (vend == std::string_view::npos) break;
                const std::string_view raw =
                    args_region.substr(gt + 1, vend - (gt + 1));
                pp = vend + param_close.size();

                if (!first) args += ",";
                first = false;
                args += "\"" + JsonEscape(key) + "\":";
                if (is_string) {
                    // The DSML `string` flag is authoritative: a string that
                    // happens to look like JSON must stay a string.
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

bool AppendDeepSeekIds(const ArrowTokenizerDb &tok, std::string_view text,
                       std::vector<int32_t> &out_ids)
{
    const std::string_view sentinels[] = {
        kBos, kEos, kSystem, kUser, kAssistant, kThink, kThinkEnd, kDsml,
    };
    size_t i = 0;
    size_t run = 0;
    auto append_text = [&](std::string_view piece) -> bool {
        for (size_t cap = 256; cap <= 65536; cap *= 2) {
            std::vector<int32_t> buf(cap, 0);
            size_t enc_cap = cap;
            const int32_t n = arrow_tokenizer_encode(tok, piece.data(),
                                                     piece.size(), buf.data(),
                                                     &enc_cap);
            if (n == -2) continue;
            if (n < 0) return false;
            const size_t begin = (tok.bos_id >= 0 && n > 0) ? 1 : 0;
            out_ids.insert(out_ids.end(), buf.begin() + begin, buf.begin() + n);
            return true;
        }
        return false;
    };
    auto flush = [&](size_t end) -> bool {
        if (end > run) return append_text(text.substr(run, end - run));
        return true;
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

bool ParseDeepSeekToolCalls(const ArrowTokenizerDb &tok,
                            const std::vector<int32_t> &gen_tokens,
                            const DsmlTags &tags,
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
    out_calls = ParseDsmlToolCalls(text, tags);
    if (out_calls.empty()) out_calls = tools::ParseToolCallsText(text);
    return true;
}

} // namespace lsxcommon
