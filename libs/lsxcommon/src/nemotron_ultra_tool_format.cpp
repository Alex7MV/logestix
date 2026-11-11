#include "lsxcommon/nemotron_ultra_tool_format.h"

#include <absl/strings/str_cat.h>

#include <cstdio>
#include <string>
#include <vector>

namespace lsxcommon::nemotron_ultra {
namespace {

// The tool preamble is quoted from the checkpoint's chat_template.jinja
// (lines 49-89). It is NOT the 30B sibling's text and not Qwen's: the wording
// references `example_function_name` and carries the four-line reminder block.
constexpr const char *kToolPreamble =
    "# Tools\n\nYou have access to the following functions:\n\n";
constexpr const char *kToolInstructions =
    "\n\nIf you choose to call a function ONLY reply in the following format with "
    "NO suffix:\n\n"
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

// jinja's `tojson`: JSON with ", " and ": " separators and ", " between array
// items, keys in insertion order. The tool schema strings arrive already
// serialised, so this only has to re-space a compact document.
std::string JsonRespace(const std::string &in)
{
    std::string out;
    out.reserve(in.size() + 8);
    bool in_string = false;
    bool escape = false;
    for (char c : in) {
        if (in_string) {
            out += c;
            if (escape) {
                escape = false;
            } else if (c == '\\') {
                escape = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        switch (c) {
        case '"':
            in_string = true;
            out += c;
            break;
        case ',':
            out += ", ";
            break;
        case ':':
            out += ": ";
            break;
        default:
            out += c;
            break;
        }
    }
    return out;
}

// The template's rule: a mapping or sequence is inlined as JSON, anything else
// goes through `string` — so a JSON string loses its quotes and a number keeps
// its spelling.
std::string Scalarize(const std::string &v)
{
    if (v.size() >= 2 && v.front() == '"' && v.back() == '"') {
        std::string out;
        out.reserve(v.size() - 2);
        bool escape = false;
        for (std::size_t i = 1; i + 1 < v.size(); ++i) {
            const char c = v[i];
            if (escape) {
                switch (c) {
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                case 'r': out += '\r'; break;
                default: out += c; break;
                }
                escape = false;
            } else if (c == '\\') {
                escape = true;
            } else {
                out += c;
            }
        }
        return out;
    }
    return v;
}

bool IsJsonObjectOrArray(const std::string &s)
{
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return false;
    return s[first] == '{' || s[first] == '[';
}

void AppendTools(std::string &out, const std::vector<ToolSpec> &tools)
{
    out += "<tools>";
    for (const auto &tool : tools) {
        out += "\n<function>\n<name>";
        out += tool.name;
        out += "</name>";
        if (!tool.description.empty()) {
            std::string desc = tool.description;
            // jinja `| trim`
            const auto b = desc.find_first_not_of(" \t\r\n");
            const auto e = desc.find_last_not_of(" \t\r\n");
            desc = b == std::string::npos ? std::string() : desc.substr(b, e - b + 1);
            out += "\n<description>";
            out += desc;
            out += "</description>";
        }
        out += "\n<parameters>";
        if (!tool.parameters_json.empty()) {
            out += JsonRespace(tool.parameters_json);
        }
        out += "</parameters>\n</function>";
    }
    out += "\n</tools>";
    out += kToolInstructions;
}

// Renders one `<tool_call>` block. Arguments are emitted in the order the JSON
// document lists them; nested values are inlined as JSON (the template's
// `tojson if mapping or sequence` rule).
void AppendToolCall(std::string &out, const ToolCall &call)
{
    out += "<tool_call>\n<function=";
    out += call.name;
    out += ">\n";
    // Split a flat JSON object into ordered key/value pairs without pulling in a
    // parser: the values are re-emitted verbatim, so only the top-level split
    // matters.
    const std::string &js = call.arguments_json;
    const auto open = js.find_first_of('{');
    const auto close = js.find_last_of('}');
    if (open != std::string::npos && close != std::string::npos && close > open) {
        std::size_t i = open + 1;
        bool in_string = false, escape = false;
        int depth = 0;
        std::string key, value;
        bool have_key = false;
        for (; i < close; ++i) {
            const char c = js[i];
            if (in_string) {
                (have_key ? value : key) += c;
                if (escape) escape = false;
                else if (c == '\\') escape = true;
                else if (c == '"') in_string = false;
                continue;
            }
            if (c == '"') {
                in_string = true;
                (have_key ? value : key) += c;
                continue;
            }
            if (c == '{' || c == '[') ++depth;
            if (c == '}' || c == ']') --depth;
            if (c == ':' && !have_key && depth == 0) {
                have_key = true;
                continue;
            }
            if (c == ',' && depth == 0) {
                // flush the pair
                if (!key.empty()) {
                    std::string k = key;
                    const auto kb = k.find_first_not_of(" \t\r\n");
                    const auto ke = k.find_last_not_of(" \t\r\n");
                    k = k.substr(kb, ke - kb + 1);
                    if (k.size() >= 2 && k.front() == '"' && k.back() == '"')
                        k = k.substr(1, k.size() - 2);
                    std::string v = value;
                    const auto vb = v.find_first_not_of(" \t\r\n");
                    const auto ve = v.find_last_not_of(" \t\r\n");
                    v = vb == std::string::npos ? std::string()
                                                : v.substr(vb, ve - vb + 1);
                    v = IsJsonObjectOrArray(v) ? JsonRespace(v) : Scalarize(v);
                    out += "<parameter=";
                    out += k;
                    out += ">\n";
                    out += v;
                    out += "\n</parameter>\n";
                }
                key.clear();
                value.clear();
                have_key = false;
                continue;
            }
            (have_key ? value : key) += c;
        }
        // last pair
        if (!key.empty()) {
            std::string k = key;
            const auto kb = k.find_first_not_of(" \t\r\n");
            const auto ke = k.find_last_not_of(" \t\r\n");
            k = k.substr(kb, ke - kb + 1);
            if (k.size() >= 2 && k.front() == '"' && k.back() == '"')
                k = k.substr(1, k.size() - 2);
            std::string v = value;
            const auto vb = v.find_first_not_of(" \t\r\n");
            const auto ve = v.find_last_not_of(" \t\r\n");
            v = vb == std::string::npos ? std::string() : v.substr(vb, ve - vb + 1);
            v = IsJsonObjectOrArray(v) ? JsonRespace(v) : Scalarize(v);
            out += "<parameter=";
            out += k;
            out += ">\n";
            out += v;
            out += "\n</parameter>\n";
        }
    }
    out += "</function>\n</tool_call>\n";
}

std::string Trim(const std::string &s)
{
    const auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return {};
    const auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

// The template's history rule: an assistant turn BEFORE the last user turn has
// its reasoning trace replaced by an empty closed block.
std::string TruncateThinking(const std::string &content)
{
    const std::string close = "</think>";
    const std::string open = "<think>";
    std::string c = content;
    const auto cpos = c.find(close);
    if (cpos != std::string::npos) {
        c = c.substr(cpos + close.size());
    } else {
        const auto opos = c.find(open);
        if (opos != std::string::npos) c = c.substr(0, opos);
    }
    return "<think></think>" + Trim(c);
}

} // namespace

bool RenderUltraConversation(const ChatConversation &conv, std::string &out,
                             const UltraRenderPolicy &policy)
{
    if (conv.messages.empty()) return false;
    out.clear();

    std::size_t first = 0;
    std::string system_message;
    bool have_system = false;
    if (conv.messages.front().role == "system") {
        system_message = conv.messages.front().content;
        have_system = true;
        first = 1;
    }
    const bool have_tools = !conv.tools.empty();

    // The template computes last_user_idx over the message list AFTER the system
    // turn is peeled off, which is what makes the index comparison meaningful.
    long last_user_idx = -1;
    for (std::size_t i = first; i < conv.messages.size(); ++i)
        if (conv.messages[i].role == "user")
            last_user_idx = static_cast<long>(i - first);

    if (have_system) {
        out += "<|im_start|>system\n";
        out += system_message;
    } else if (have_tools) {
        out += "<|im_start|>system\n";
    }
    if (have_tools) {
        if (have_system && !system_message.empty()) out += "\n\n";
        out += kToolPreamble;
        AppendTools(out, conv.tools);
    }
    if (have_system || have_tools) out += "<|im_end|>\n";

    for (std::size_t i = first; i < conv.messages.size(); ++i) {
        const ChatMessage &m = conv.messages[i];
        const long idx = static_cast<long>(i - first);
        const bool before_last_user = policy.truncate_history_thinking &&
                                      idx < last_user_idx;

        if (m.role == "assistant") {
            std::string content = m.content;
            if (!m.tool_calls.empty()) {
                out += "<|im_start|>assistant\n";
                if (!Trim(content).empty()) {
                    out += before_last_user ? TruncateThinking(content)
                                            : Trim(content);
                    out += "\n";
                } else {
                    out += "<think></think>";
                }
                for (const auto &c : m.tool_calls) AppendToolCall(out, c);
                out += "<|im_end|>\n";
            } else {
                if (!before_last_user) {
                    out += "<|im_start|>assistant\n";
                    out += Trim(content);
                    out += "<|im_end|>\n";
                } else {
                    std::string c = content;
                    if (c.find("<think>") != std::string::npos &&
                        c.find("</think>") != std::string::npos) {
                        c = "<think></think>" +
                            c.substr(c.rfind("</think>") + std::string("</think>").size());
                    }
                    c = Trim(c);
                    out += "<|im_start|>assistant\n";
                    out += c;
                    out += "<|im_end|>\n";
                }
            }
        } else if (m.role == "user" || m.role == "system") {
            out += "<|im_start|>";
            out += m.role;
            out += "\n";
            if (m.role == "user" && idx == last_user_idx && policy.medium_effort) {
                out += m.content;
                out += "\n\n{reasoning effort: efficient}";
            } else {
                out += m.content;
            }
            out += "<|im_end|>\n";
        } else if (m.role == "tool") {
            // Consecutive tool results share ONE user turn; the turn closes on
            // the last of them.
            const bool prev_is_tool =
                i > first && conv.messages[i - 1].role == "tool";
            if (!prev_is_tool) out += "<|im_start|>user\n";
            out += "<tool_response>\n";
            out += m.content;
            out += "\n</tool_response>\n";
            const bool next_is_tool =
                i + 1 < conv.messages.size() && conv.messages[i + 1].role == "tool";
            if (!next_is_tool) out += "<|im_end|>\n";
        } else {
            out += "<|im_start|>";
            out += m.role;
            out += "\n";
            out += m.content;
            out += "<|im_end|>\n";
        }
    }

    // Generation prompt. Ultra's template defaults enable_thinking to TRUE, so
    // the assistant turn opens an UNCLOSED think block.
    if (policy.enable_thinking) {
        out += "<|im_start|>assistant\n<think>\n";
    } else {
        out += "<|im_start|>assistant\n<think></think>";
    }
    return true;
}

} // namespace lsxcommon::nemotron_ultra
