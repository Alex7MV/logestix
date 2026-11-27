#include "lsxcommon/granite_tool_format.h"

#include <lsxnetwork/json_escape.h>
#include <simdjson.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "lsxcommon/tool_format.h"

namespace lsxcommon {
namespace {
// Dedup Ф4: BPE append (cap 65536) is the shared lsxcommon::tools body.
using lsxcommon::tools::AppendEncoded;

using lsxnetwork::json::JsonEscape;

// Added tokens in the checkpoint's tokenizer.json. The BPE encoder resolves
// none of them, so they must be looked up and injected as single ids; a text
// encoding of <think> (or the tool-call markers) changes the prompt enough
// that the model no longer recognizes the ChatML turn structure.
constexpr const char *kGraniteSentinels[] = {
    "<|im_start|>",   "<|im_end|>",      "<tool_call>",  "</tool_call>",
    "<tool_response>", "</tool_response>", "<think>",      "</think>",
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

std::string ToolJson(const ToolSpec &t)
{
    std::string out = "{\"name\": \"" + JsonEscape(t.name) +
                      "\", \"description\": \"" + JsonEscape(t.description) +
                      "\", \"parameters\": ";
    out += t.parameters_json.empty() ? "{}" : t.parameters_json;
    out += "}";
    return out;
}

// The tool-call arguments are plain text values in Granite's XML, not JSON
// scalars: always emit them as JSON strings.
std::string ArgsFromXml(std::string_view body)
{
    std::string out = "{";
    bool first = true;
    std::size_t pos = 0;
    while (true) {
        const std::size_t p = body.find("<parameter=", pos);
        if (p == std::string_view::npos) break;
        const std::size_t key_end = body.find('>', p + 11);
        if (key_end == std::string_view::npos) break;
        const std::string_view key =
            Trim(body.substr(p + 11, key_end - (p + 11)));
        const std::size_t value_end = body.find("</parameter>", key_end + 1);
        if (value_end == std::string_view::npos) break;
        const std::string_view value =
            Trim(body.substr(key_end + 1, value_end - (key_end + 1)));
        if (!first) out += ",";
        first = false;
        out += "\"" + JsonEscape(std::string(key)) + "\":\"" +
               JsonEscape(std::string(value)) + "\"";
        pos = value_end + 12;
    }
    out += "}";
    return out;
}

} // namespace

bool RenderGraniteConversation(const ChatConversation &conv, std::string &out)
{
    out.clear();
    const ChatMessage *system = nullptr;
    size_t first = 0;
    if (!conv.messages.empty() && conv.messages.front().role == "system") {
        system = &conv.messages.front();
        first = 1;
    }

    out += "<|im_start|>system\n";
    if (system != nullptr) out += system->content;
    if (!conv.tools.empty()) {
        if (system != nullptr && !system->content.empty()) out += "\n\n";
        out += "# Tools\n\nYou have access to the following functions:\n\n<tools>";
        for (const ToolSpec &t : conv.tools)
            out += "\n" + ToolJson(t);
        out += "\n</tools>";
        out +=
            "\n\nIf you choose to call a function ONLY reply in the following "
            "format with NO suffix:\n\n"
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
            "- You may provide optional reasoning for your function call in "
            "natural language BEFORE the function call, but NOT after\n"
            "- If there is no function call available, answer the question "
            "like normal with your current knowledge and do not tell the user "
            "about function calls\n</IMPORTANT>";
    }
    out += "<|im_end|>\n";

    for (size_t i = first; i < conv.messages.size(); ++i) {
        const ChatMessage &m = conv.messages[i];
        if (m.role == "assistant") {
            out += "<|im_start|>assistant\n";
            std::string content = m.content;
            if (content.find("<think>") == std::string::npos &&
                content.find("</think>") == std::string::npos)
                content = "<think></think>" + content;
            out += content;
            for (const ToolCall &c : m.tool_calls) {
                out += "<tool_call>\n<function=" + c.name + ">\n";
                // `arguments_json` is JSON; re-emit its members as Granite
                // parameter blocks.
                simdjson::dom::parser parser;
                simdjson::dom::element obj;
                if (parser.parse(simdjson::padded_string(c.arguments_json))
                        .get(obj) == simdjson::SUCCESS &&
                    obj.is_object()) {
                    for (auto [k, v] : obj.get_object()) {
                        std::string sv;
                        if (v.get_string().get(sv) != simdjson::SUCCESS)
                            sv = simdjson::to_string(v);
                        out += "<parameter=" + std::string(k) + ">\n" + sv +
                               "\n</parameter>\n";
                    }
                }
                out += "</function>\n</tool_call>\n";
            }
            out += "<|im_end|>\n";
        } else if (m.role == "tool") {
            out += "<|im_start|>user\n<tool_response>\n" + m.content +
                   "\n</tool_response>\n<|im_end|>\n";
        } else {
            out += "<|im_start|>" + m.role + "\n" + m.content + "<|im_end|>\n";
        }
    }

    // Tool round-trips are a direct-answer task; the checkpoint supports the
    // template's enable_thinking=false form. (The template default is true,
    // which spends the whole 256-token budget on reasoning before the call.)
    out += "<|im_start|>assistant\n<think></think>\n";
    return true;
}

std::vector<ToolCall> ParseGraniteToolCallText(std::string_view text)
{
    std::vector<ToolCall> calls;
    constexpr std::string_view kOpen = "<tool_call>";
    constexpr std::string_view kClose = "</tool_call>";
    std::size_t pos = 0;
    while (true) {
        const std::size_t b = text.find(kOpen, pos);
        if (b == std::string_view::npos) break;
        const std::size_t e = text.find(kClose, b + kOpen.size());
        if (e == std::string_view::npos) break;
        std::string_view inner = text.substr(b + kOpen.size(),
                                             e - (b + kOpen.size()));
        pos = e + kClose.size();
        const std::size_t fn = inner.find("<function=");
        if (fn == std::string_view::npos) continue;
        const std::size_t fn_end = inner.find('>', fn + 10);
        if (fn_end == std::string_view::npos) continue;
        ToolCall call;
        call.name = std::string(
            Trim(inner.substr(fn + 10, fn_end - (fn + 10))));
        if (call.name.empty()) continue;
        call.arguments_json = ArgsFromXml(inner.substr(fn_end + 1));
        call.id = "call_" + std::to_string(calls.size());
        calls.push_back(std::move(call));
    }
    return calls;
}

bool AppendGraniteChatIds(const ArrowTokenizerDb &tok, const std::string &text,
                          std::vector<int32_t> &out_ids)
{
    out_ids.clear();
    if (arrow_tokenizer_lookup(tok, "<|im_start|>", 12) < 0) return false;
    // The checkpoint's chat template starts directly with <|im_start|>;
    // HuggingFace's apply_chat_template does NOT prepend <s> here (verified
    // against the source tokenizer: ids[0] == 100256).
    size_t i = 0;
    size_t run = 0;
    auto flush = [&](size_t end) -> bool {
        if (end > run &&
            !AppendEncoded(tok, std::string_view(text).substr(run, end - run),
                           out_ids))
            return false;
        return true;
    };
    while (i < text.size()) {
        bool matched = false;
        for (const char *s : kGraniteSentinels) {
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

bool ParseGraniteTokenCalls(const ArrowTokenizerDb &tok,
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
    out_calls = ParseGraniteToolCallText(text);
    if (out_calls.empty()) out_calls = tools::ParseToolCallsText(text);
    if (out_calls.empty()) out_calls = tools::ParseBareToolCallJson(text);
    return true;
}

} // namespace lsxcommon
