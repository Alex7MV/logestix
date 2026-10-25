#include "lsxcommon/kimi_moe_tool_format.h"

#include "lsxcommon/tool_format.h"

#include <lsxnetwork/json_escape.h>

#include <cstring>
#include <map>
#include <string>

namespace lsxcommon {
namespace {

using lsxnetwork::json::JsonEscape;

constexpr const char *kImSystem = "<|im_system|>";
constexpr const char *kImUser = "<|im_user|>";
constexpr const char *kImAssistant = "<|im_assistant|>";
constexpr const char *kImMiddle = "<|im_middle|>";
constexpr const char *kImEnd = "<|im_end|>";
constexpr const char *kSectionBegin = "<|tool_calls_section_begin|>";
constexpr const char *kSectionEnd = "<|tool_calls_section_end|>";
constexpr const char *kCallBegin = "<|tool_call_begin|>";
constexpr const char *kArgsBegin = "<|tool_call_argument_begin|>";
constexpr const char *kCallEnd = "<|tool_call_end|>";
constexpr const char *kThinkOpen = "<think>";
constexpr const char *kThinkClose = "</think>";

std::string WireId(const std::string &name, size_t index)
{
    return "functions." + name + ":" + std::to_string(index);
}

// "functions.x:3" -> "x", "x:3" -> "x", "x" -> "x".
std::string NameFromWireId(std::string_view id)
{
    constexpr std::string_view kPrefix = "functions.";
    if (id.rfind(kPrefix, 0) == 0) id.remove_prefix(kPrefix.size());
    const size_t colon = id.rfind(':');
    const std::string_view name =
        colon == std::string_view::npos ? id : id.substr(0, colon);
    return std::string(name);
}

void RenderTools(const std::vector<ToolSpec> &tools, std::string &out)
{
    out += kImSystem;
    out += "tool_declare";
    out += kImMiddle;
    out += '[';
    for (size_t i = 0; i < tools.size(); ++i) {
        if (i) out += ',';
        out += R"({"type":"function","function":{"name":")";
        out += JsonEscape(tools[i].name);
        out += R"(","description":")";
        out += JsonEscape(tools[i].description);
        out += R"(","parameters":)";
        out += tools[i].parameters_json.empty() ? "{}"
                                                : tools[i].parameters_json;
        out += "}}";
    }
    out += ']';
    out += kImEnd;
}

}  // namespace

bool RenderKimiConversation(const ChatConversation &conv, std::string &out)
{
    out.clear();
    if (!conv.tools.empty()) RenderTools(conv.tools, out);

    // Client tool_call_id -> wire id, so a following role=tool message can
    // echo the id the model actually saw.
    std::map<std::string, std::string> wire_ids;

    for (const ChatMessage &m : conv.messages) {
        if (m.role == "tool") {
            const auto it = wire_ids.find(m.tool_call_id);
            const std::string id =
                it == wire_ids.end() ? m.tool_call_id : it->second;
            out += kImSystem;
            out += "tool";
            out += kImMiddle;
            out += "## Return of " + id + "\n";
            out += m.content;
            out += kImEnd;
            continue;
        }
        if (m.role == "system") {
            out += kImSystem;
            out += "system";
            out += kImMiddle;
            out += m.content;
            out += kImEnd;
            continue;
        }
        if (m.role == "user") {
            out += kImUser;
            out += "user";
            out += kImMiddle;
            out += m.content;
            out += kImEnd;
            continue;
        }
        if (m.role != "assistant") return false;

        out += kImAssistant;
        out += "assistant";
        out += kImMiddle;
        out += kThinkOpen;
        out += kThinkClose;
        out += m.content;
        if (!m.tool_calls.empty()) {
            out += kSectionBegin;
            for (size_t i = 0; i < m.tool_calls.size(); ++i) {
                const ToolCall &c = m.tool_calls[i];
                const std::string wire = WireId(c.name, i);
                if (!c.id.empty()) wire_ids[c.id] = wire;
                out += kCallBegin;
                out += wire;
                out += kArgsBegin;
                out += c.arguments_json.empty() ? "{}" : c.arguments_json;
                out += kCallEnd;
            }
            out += kSectionEnd;
        }
        out += kImEnd;
    }

    out += kImAssistant;
    out += "assistant";
    out += kImMiddle;
    out += kThinkOpen;
    return true;
}

std::vector<ToolCall> ParseKimiToolCalls(std::string_view text)
{
    std::vector<ToolCall> calls;
    size_t pos = 0;
    while (true) {
        const size_t b = text.find(kCallBegin, pos);
        if (b == std::string_view::npos) break;
        const size_t id_begin = b + std::strlen(kCallBegin);
        const size_t a = text.find(kArgsBegin, id_begin);
        if (a == std::string_view::npos) break;
        const size_t args_begin = a + std::strlen(kArgsBegin);
        const size_t e = text.find(kCallEnd, args_begin);
        if (e == std::string_view::npos) break;
        ToolCall c;
        c.id = std::string(text.substr(id_begin, a - id_begin));
        c.name = NameFromWireId(c.id);
        c.arguments_json = std::string(text.substr(args_begin, e - args_begin));
        if (!c.name.empty()) calls.push_back(std::move(c));
        pos = e + std::strlen(kCallEnd);
    }
    if (calls.empty()) {
        calls = tools::ParseToolCallsText(text);
        if (calls.empty()) calls = tools::ParseBareToolCallJson(text);
    }
    return calls;
}

bool AppendKimiIds(const ArrowTokenizerDb &tok, std::string_view text,
                   std::vector<int32_t> &out_ids)
{
    static const char *const kSentinels[] = {
        kImSystem,   kImUser,      kImAssistant, kImMiddle,   kImEnd,
        kSectionBegin, kSectionEnd, kCallBegin,  kArgsBegin,  kCallEnd,
        kThinkOpen,  kThinkClose};
    out_ids.clear();

    size_t pos = 0;
    while (pos < text.size()) {
        size_t best = std::string_view::npos;
        const char *marker = nullptr;
        for (const char *s : kSentinels) {
            const size_t at = text.find(s, pos);
            if (at < best) {
                best = at;
                marker = s;
            }
        }
        const size_t end =
            best == std::string_view::npos ? text.size() : best;
        if (end > pos) {
            std::vector<int32_t> ids(text.size() - pos + 16);
            size_t cap = ids.size();
            const int32_t n = arrow_tokenizer_encode(
                tok, text.data() + pos, end - pos, ids.data(), &cap);
            if (n < 0) return false;
            // arrow_tokenizer_encode prepends bos_id when it is >= 0. Neither
            // the checkpoint nor this bridge wants it mid-transcript.
            const size_t skip =
                (tok.bos_id >= 0 && n > 0 && ids[0] == tok.bos_id) ? 1 : 0;
            out_ids.insert(out_ids.end(), ids.begin() + skip, ids.begin() + n);
        }
        if (marker == nullptr) break;
        const int32_t id =
            arrow_tokenizer_lookup(tok, marker, std::strlen(marker));
        if (id < 0) return false;
        out_ids.push_back(id);
        pos = best + std::strlen(marker);
    }
    return true;
}

}  // namespace lsxcommon
