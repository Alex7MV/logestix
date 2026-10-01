#include "lsxcommon/alice_tool_format.h"

#include "lsxcommon/tool_format.h"

#include <string>

namespace lsxcommon::alice {
namespace {

constexpr const char *kToolCallStart = "[TOOL_CALL_START]";
constexpr const char *kCotStart = "[COT_START]";
constexpr const char *kCotEnd = "[COT_END]";
constexpr const char *kToolsPrefix = "\u0422\u0435\u0431\u0435 \u0434\u043e\u0441\u0442\u0443\u043f\u043d\u044b "
                                     "\u0441\u043b\u0435\u0434\u0443\u044e\u0449\u0438\u0435 "
                                     "\u0444\u0443\u043d\u043a\u0446\u0438\u0438:";

std::string RenderToolDefinition(const ToolSpec &t)
{
    std::string s = "function {\"name\":\"" + t.name + "\",";
    if (!t.description.empty())
        s += "\"description\":\"" + t.description + "\",";
    s += "\"parameters\":" + t.parameters_json + "}";
    return s;
}

std::string RenderAssistantContent(const ChatMessage &m)
{
    std::string content;
    if (!m.tool_calls.empty()) {
        for (const auto &c : m.tool_calls)
            content += std::string("\n") + kToolCallStart + c.name + "\n" +
                       c.arguments_json;
    } else {
        content = m.content;
    }
    std::string out = "assistant:";
    if (!m.content.empty()) {
        // reasoning_content is not carried in ChatMessage; content is the body.
    }
    out += " " + content;
    return out;
}

std::string RenderMessage(const ChatMessage &m)
{
    if (m.role == "system") return "system: " + m.content;
    if (m.role == "user") return "user: " + m.content;
    if (m.role == "assistant") return RenderAssistantContent(m);
    if (m.role == "tool") return "Tool " + m.tool_call_id + ": " + m.content;
    if (m.role == "meta") return "meta: " + m.content;
    return {};
}

std::string Trim(std::string s)
{
    const auto b = s.find_first_not_of(" \t\n\r\f\v");
    if (b == std::string::npos) return {};
    const auto e = s.find_last_not_of(" \t\n\r\f\v");
    return s.substr(b, e - b + 1);
}

} // namespace

bool RenderConversation(const ArrowTokenizerDb &tok,
                        const ChatConversation &conv,
                        bool add_generation_prompt,
                        std::vector<int32_t> &out_ids)
{
    std::string body;
    const bool has_tools = !conv.tools.empty();
    if (has_tools) {
        body += kToolsPrefix;
        for (const auto &t : conv.tools)
            body += "\n" + RenderToolDefinition(t);
    }
    const int n = static_cast<int>(conv.messages.size());
    for (int i = 0; i < n; ++i) {
        const ChatMessage &m = conv.messages[static_cast<size_t>(i)];
        const bool last = (i == n - 1);
        std::string block = RenderMessage(m);
        if (last && !add_generation_prompt) block = Trim(block);
        if (block.empty()) continue;
        const std::string sep =
            (has_tools || i > 0) ? std::string("\n\n ") : std::string();
        body += sep + block;
    }
    if (add_generation_prompt) body += "\n\n assistant:";

    // arrow_tokenizer_encode prepends BOS (matching the template's bos_token).
    std::vector<int32_t> ids(static_cast<size_t>(body.size()) * 4 + 64);
    size_t cap = ids.size();
    const int32_t rc =
        arrow_tokenizer_encode(tok, body.data(), body.size(), ids.data(), &cap);
    if (rc < 0) return false;
    ids.resize(static_cast<size_t>(rc));
    out_ids = std::move(ids);
    return true;
}

bool ParseToolCallsAlice(const ArrowTokenizerDb &tok,
                         const std::vector<int32_t> &gen_tokens,
                         std::vector<ToolCall> &out_calls)
{
    out_calls.clear();
    if (gen_tokens.empty()) return true;
    std::size_t cap = gen_tokens.size() * 16 + 64;
    std::string text(cap, '\0');
    if (arrow_tokenizer_decode(tok, gen_tokens.data(), gen_tokens.size(),
                               text.data(), &cap) != 0)
        return false;
    text.resize(cap);

    const std::string marker = kToolCallStart;
    std::size_t pos = 0;
    while ((pos = text.find(marker, pos)) != std::string::npos) {
        std::size_t p = pos + marker.size();
        std::size_t nl = text.find('\n', p);
        if (nl == std::string::npos) break;
        ToolCall c;
        c.name = text.substr(p, nl - p);
        std::size_t end = text.find(marker, nl + 1);
        std::string args = text.substr(nl + 1, (end == std::string::npos
                                                    ? text.size()
                                                    : end) -
                                                   (nl + 1));
        // Trim trailing whitespace/newlines.
        const auto last = args.find_last_not_of(" \t\n\r");
        if (last != std::string::npos) args.resize(last + 1);
        c.arguments_json = args;
        if (!c.name.empty()) out_calls.push_back(std::move(c));
        pos = (end == std::string::npos) ? text.size() : end;
    }
    if (!out_calls.empty()) return true;

    out_calls = tools::ParseToolCallsText(text);
    if (out_calls.empty()) out_calls = tools::ParseBareToolCallJson(text);
    return true;
}

} // namespace lsxcommon::alice
