#include "lsxcommon/dory_tool_format.h"

#include <simdjson.h>

#include <cstdio>
#include <cstring>
#include <string>

#include <lsxnetwork/json_escape.h>

#include "lsxcommon/qwen_tool_format.h"
#include "lsxcommon/tool_format.h"

namespace lsxcommon {

namespace {

using lsxnetwork::json::JsonEscape;

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

// jinja `tojson` = json.dumps(obj, ensure_ascii=False): ", " / ": " separators.
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

void AppendJinjaJson(std::string &out, simdjson::dom::element e)
{
    using simdjson::dom::element_type;
    switch (e.type()) {
    case element_type::OBJECT: {
        out += "{";
        bool first = true;
        for (auto [k, v] : e.get_object()) {
            if (!first) out += ", ";
            first = false;
            AppendJinjaString(out, k);
            out += ": ";
            AppendJinjaJson(out, v);
        }
        out += "}";
        return;
    }
    case element_type::ARRAY: {
        out += "[";
        bool first = true;
        for (auto v : e.get_array()) {
            if (!first) out += ", ";
            first = false;
            AppendJinjaJson(out, v);
        }
        out += "]";
        return;
    }
    case element_type::STRING:
        AppendJinjaString(out, e.get_string().value_unsafe());
        return;
    case element_type::BOOL:
        out += e.get_bool().value_unsafe() ? "true" : "false";
        return;
    case element_type::NULL_VALUE:
        out += "null";
        return;
    case element_type::INT64:
        out += std::to_string(e.get_int64().value_unsafe());
        return;
    case element_type::UINT64:
        out += std::to_string(e.get_uint64().value_unsafe());
        return;
    case element_type::DOUBLE: {
        char buf[40];
        const double v = e.get_double().value_unsafe();
        std::snprintf(buf, sizeof(buf), "%.17g", v);
        out += buf;
        return;
    }
    default:
        out += "null";
        return;
    }
}

constexpr const char *kToolPreamble =
    "# Tools\n\nYou have access to the following functions:\n\n<tools>";
constexpr const char *kToolInstructions =
    "\n</tools>"
    "\n\nIf you choose to call a function ONLY reply in the following format "
    "with NO suffix:\n\n"
    "<tool_call>\n<function=example_function_name>\n"
    "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second "
    "parameter\nthat can span\nmultiple lines\n</parameter>\n"
    "</function>\n</tool_call>\n\n"
    "<IMPORTANT>\nReminder:\n"
    "- Function calls MUST follow the specified format: an inner "
    "<function=...></function> block must be nested within "
    "<tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural "
    "language BEFORE the function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal "
    "with your current knowledge and do not tell the user about function "
    "calls\n</IMPORTANT>";

// Appends the template's `<function>…</function>` block for one tool.
bool AppendToolDefinition(std::string &out, const ToolSpec &tool)
{
    out += "\n<function>\n<name>" + tool.name + "</name>";
    if (!tool.description.empty())
        out += "\n<description>" + std::string(TrimAscii(tool.description)) +
               "</description>";
    out += "\n<parameters>";
    simdjson::dom::parser parser;
    simdjson::dom::element params;
    if (!tool.parameters_json.empty()) {
        const simdjson::padded_string padded(tool.parameters_json);
        if (parser.parse(padded).get(params) != simdjson::SUCCESS ||
            params.type() != simdjson::dom::element_type::OBJECT)
            return false;
        simdjson::dom::element props;
        if (params["properties"].get(props) == simdjson::SUCCESS &&
            props.type() == simdjson::dom::element_type::OBJECT) {
            for (auto [pname, pfields] : props.get_object()) {
                out += "\n<parameter>\n<name>";
                out += pname;
                out += "</name>";
                if (pfields.type() != simdjson::dom::element_type::OBJECT)
                    continue;
                simdjson::dom::element sub;
                if (pfields["type"].get(sub) == simdjson::SUCCESS &&
                    sub.type() == simdjson::dom::element_type::STRING) {
                    out += "\n<type>";
                    out += sub.get_string().value_unsafe();
                    out += "</type>";
                }
                if (pfields["description"].get(sub) == simdjson::SUCCESS &&
                    sub.type() == simdjson::dom::element_type::STRING) {
                    out += "\n<description>";
                    out += std::string(
                        TrimAscii(sub.get_string().value_unsafe()));
                    out += "</description>";
                }
                if (pfields["enum"].get(sub) == simdjson::SUCCESS) {
                    out += "\n<enum>";
                    AppendJinjaJson(out, sub);
                    out += "</enum>";
                }
                out += "\n</parameter>";
            }
        }
        simdjson::dom::element req;
        if (params["required"].get(req) == simdjson::SUCCESS) {
            out += "\n<required>";
            AppendJinjaJson(out, req);
            out += "</required>";
        }
    }
    out += "\n</parameters>\n</function>";
    return true;
}

// Assistant turn body: the template wraps history content in a (possibly empty)
// think block and drops the reasoning of turns before the last user message.
std::string AssistantBody(const ChatMessage &m, bool truncate)
{
    const std::string_view content = TrimAscii(m.content);
    const size_t close = content.rfind("</think>");
    std::string body;
    if (close != std::string_view::npos) {
        const std::string_view tail = TrimAscii(content.substr(
            close + std::strlen("</think>")));
        body = "<think></think>" + std::string(tail);
    } else {
        const size_t open = content.find("<think>");
        const std::string_view tail =
            (open == std::string_view::npos) ? content
                                            : TrimAscii(content.substr(open));
        body = "<think></think>" + std::string(tail);
    }
    if (!truncate) return body;
    // The template's truncation only strips reasoning that precedes the
    // `</think>` marker; AssistantBody already reduced the trace to the closed
    // empty block, so the result is the same for both branches.
    return body;
}

} // namespace

bool RenderDoryConversation(const ChatConversation &conv, std::string &out,
                            const DoryRenderPolicy &policy)
{
    out.clear();
    static const std::vector<std::string_view> kMarkers = {
        "<|im_start|>",    "<|im_end|>",      "<tool_call>", "</tool_call>",
        "<tool_response>", "</tool_response>", "<think>",    "</think>",
    };
    const auto esc = [](std::string_view s) {
        return tools::NeutralizeControlMarkers(s, kMarkers);
    };

    size_t first = 0;
    std::string system_message;
    if (!conv.messages.empty() && conv.messages[0].role == "system") {
        system_message = esc(TrimAscii(conv.messages[0].content));
        first = 1;
    }

    // System turn: ALWAYS emitted, even when empty (this template has no
    // `{% if content %}` guard around it).
    out += "<|im_start|>system\n" + system_message;
    if (!conv.tools.empty()) {
        if (!system_message.empty()) out += "\n\n";
        out += kToolPreamble;
        for (const ToolSpec &t : conv.tools)
            if (!AppendToolDefinition(out, t)) return false;
        out += kToolInstructions;
    }
    out += "<|im_end|>\n";

    for (size_t i = first; i < conv.messages.size(); ++i) {
        const ChatMessage &m = conv.messages[i];
        if (m.role == "system") return false;   // only a leading system turn
        if (m.role == "user") {
            out += "<|im_start|>user\n" + esc(TrimAscii(m.content)) +
                   "<|im_end|>\n";
        } else if (m.role == "assistant") {
            const std::string body =
                AssistantBody(m, policy.truncate_history_thinking);
            out += "<|im_start|>assistant\n";
            if (m.tool_calls.empty()) {
                out += body + "<|im_end|>\n";
                continue;
            }
            // With tool calls the template appends a newline after the content
            // block and one after every </tool_call>.
            out += body + "\n";
            for (const ToolCall &c : m.tool_calls) {
                out += "<tool_call>\n<function=" + c.name + ">\n";
                simdjson::dom::parser parser;
                simdjson::dom::element args;
                if (!c.arguments_json.empty()) {
                    const simdjson::padded_string padded(c.arguments_json);
                    if (parser.parse(padded).get(args) == simdjson::SUCCESS &&
                        args.type() == simdjson::dom::element_type::OBJECT) {
                        for (auto [k, v] : args.get_object()) {
                            out += "<parameter=";
                            out += k;
                            out += ">\n";
                            if (v.type() == simdjson::dom::element_type::STRING)
                                out += v.get_string().value_unsafe();
                            else
                                AppendJinjaJson(out, v);
                            out += "\n</parameter>\n";
                        }
                    }
                }
                out += "</function>\n</tool_call>\n";
            }
            out += "<|im_end|>\n";
        } else if (m.role == "tool") {
            const bool first_of_run =
                i == first || conv.messages[i - 1].role != "tool";
            const bool last_of_run =
                i + 1 >= conv.messages.size() ||
                conv.messages[i + 1].role != "tool";
            if (first_of_run) out += "<|im_start|>user\n";
            out += "<tool_response>\n" + esc(TrimAscii(m.content)) +
                   "\n</tool_response>\n";
            if (last_of_run) out += "<|im_end|>\n";
        } else {
            return false;
        }
    }

    out += "<|im_start|>assistant\n<think>";
    out += policy.enable_thinking ? "\n" : "</think>";
    return true;
}

std::vector<ToolCall> ParseDoryToolCallText(std::string_view text)
{
    return ParseQwenToolCallText(text);
}

bool DoryChatSentinelsPresent(const ArrowTokenizerDb &tok)
{
    return QwenChatSentinelsPresent(tok);
}

bool AppendDoryChatIds(const ArrowTokenizerDb &tok, const std::string &text,
                       std::vector<int32_t> &out_ids)
{
    return AppendQwenChatIds(tok, text, out_ids);
}

bool ParseDoryTokenCalls(const ArrowTokenizerDb &tok,
                         const std::vector<int32_t> &gen_tokens,
                         std::vector<ToolCall> &out_calls)
{
    return ParseQwenTokenCalls(tok, gen_tokens, out_calls);
}

} // namespace lsxcommon
