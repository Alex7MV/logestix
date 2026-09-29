#include "lsxcommon/deepseek_moe_tool_format.h"

#include "json_escape.h"
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

using lsxcommon::json::JsonEscape;

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

std::string DsmlEsc(std::string_view s);

std::string ToolJson(const ToolSpec &t)
{
    return "{\"type\":\"function\",\"function\":{\"name\":\"" +
           JsonEscape(DsmlEsc(t.name)) + "\",\"description\":\"" +
           JsonEscape(DsmlEsc(t.description)) + "\",\"parameters\":" +
           (t.parameters_json.empty() ? std::string("{}")
                                      : DsmlEsc(t.parameters_json)) +
           "}}";
}

const std::vector<std::string_view> &DeepSeekMarkers()
{
    static const std::vector<std::string_view> m = {
        kBos, kEos, kSystem, kUser, kAssistant, kThink, kThinkEnd, kDsml,
    };
    return m;
}

std::string DsmlEsc(std::string_view s)
{
    return tools::NeutralizeControlMarkers(s, DeepSeekMarkers());
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
    simdjson::dom::parser parser;
    for (const ToolCall &c : calls) {
        body += "<" + d + std::string(tags.call) + " name=\"" +
                DsmlEsc(c.name) + "\">\n";
        if (!c.arguments_json.empty()) {
            simdjson::dom::element args;
            simdjson::padded_string padded(c.arguments_json);
            if (parser.parse(padded).get(args) == simdjson::SUCCESS &&
                args.type() == simdjson::dom::element_type::OBJECT) {
                for (auto [k, v] : args.get_object()) {
                    const bool is_str =
                        v.type() == simdjson::dom::element_type::STRING;
                    body += "<" + d + std::string(tags.param) + " name=\"" +
                            DsmlEsc(k) + "\" string=\"" +
                            (is_str ? "true" : "false") + "\">";
                    body += is_str
                                ? DsmlEsc(v.get_string().value_unsafe())
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
        out += tags.system;  // empty for V4 (no <｜System｜> token)
        out += DsmlEsc(conv.messages[0].content);
        if (!conv.tools.empty())
            out += "\n\n" + RenderToolsBlock(conv.tools, tags);
    }

    // Canonical merge (merge_tool_messages): consecutive user/tool messages
    // coalesce into one <｜User｜> turn whose parts are joined by "\n\n".
    enum class Kind { kUser, kSystem, kAssistant };
    struct Item {
        Kind kind;
        std::string content;             // system / assistant
        std::vector<ToolCall> calls;     // assistant
        std::vector<std::string> parts;  // user
    };
    std::vector<Item> items;
    for (size_t i = 0; i < conv.messages.size(); ++i) {
        const ChatMessage &m = conv.messages[i];
        if (m.role == "system") {
            if (i == 0) continue;  // rendered above
            Item it;
            it.kind = Kind::kSystem;
            it.content = DsmlEsc(m.content);
            items.push_back(std::move(it));
        } else if (m.role == "assistant") {
            Item it;
            it.kind = Kind::kAssistant;
            it.content = DsmlEsc(m.content);
            it.calls = m.tool_calls;
            items.push_back(std::move(it));
        } else if (m.role == "user" || m.role == "tool") {
            std::string part = m.role == "tool"
                                   ? "<tool_result>" + DsmlEsc(m.content) +
                                         "</tool_result>"
                                   : DsmlEsc(m.content);
            if (!items.empty() && items.back().kind == Kind::kUser) {
                items.back().parts.push_back(std::move(part));
            } else {
                Item it;
                it.kind = Kind::kUser;
                it.parts.push_back(std::move(part));
                items.push_back(std::move(it));
            }
        } else {
            return false;
        }
    }

    for (const Item &it : items) {
        if (it.kind == Kind::kSystem) {
            out += tags.system;  // empty for V4
            out += it.content;
        } else if (it.kind == Kind::kUser) {
            out += kUser;
            for (size_t p = 0; p < it.parts.size(); ++p) {
                if (p) out += "\n\n";
                out += it.parts[p];
            }
        } else {
            out += kAssistant;
            out += kThinkEnd;
            out += it.content;
            if (!it.calls.empty()) out += RenderToolCalls(it.calls, tags);
            out += kEos;
        }
    }

    // Generation prompt when the turn ends on a user observation (or a
    // mid-conversation system message, which counts as a user turn).
    if (!items.empty() &&
        (items.back().kind == Kind::kUser ||
         items.back().kind == Kind::kSystem)) {
        out += kAssistant;
        out += kThinkEnd;
    }
    return true;
}

std::vector<ToolCall> ParseDsmlToolCalls(std::string_view text,
                                         const DsmlTags &tags)
{
    std::vector<ToolCall> calls;
    simdjson::dom::parser parser;
    const std::string d(kDsml);
    // Key on the DSML token + tag name, not the surrounding angle brackets:
    // some checkpoints emit the opening tag as `｜DSML｜tool_calls` without the
    // leading `<` (closings keep `</…>`), while the template form is
    // `<｜DSML｜tool_calls>`. Searching after the bracket matches both.
    const std::string block_open = d + std::string(tags.block);
    const std::string block_close = "</" + d + std::string(tags.block) + ">";
    const std::string call_prefix =
        d + std::string(tags.call) + " name=\"";
    const std::string call_close = "</" + d + std::string(tags.call) + ">";
    const std::string param_prefix =
        d + std::string(tags.param) + " name=\"";
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
                       const DsmlTags &tags, std::vector<int32_t> &out_ids)
{
    std::vector<std::string_view> sentinels = {
        kBos, kEos, kUser, kAssistant, kThink, kThinkEnd, kDsml,
    };
    if (!tags.system.empty()) sentinels.push_back(tags.system);
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
    if (out_calls.empty()) out_calls = tools::ParseBareToolCallJson(text);
    return true;
}

} // namespace lsxcommon
