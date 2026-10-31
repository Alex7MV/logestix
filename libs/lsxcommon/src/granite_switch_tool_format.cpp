#include "lsxcommon/granite_switch_tool_format.h"

#include <lsxnetwork/json_escape.h>
#include <simdjson.h>

#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "lsxcommon/tool_format.h"

namespace lsxcommon {
namespace {

using lsxnetwork::json::JsonEscape;

constexpr const char *kSentinels[] = {
    "<|start_of_role|>", "<|end_of_role|>",  "<|end_of_text|>",
    "<|guardian-core|>", "<|decision|>",     "<|audio|>",
    "<|pad|>",
};

struct AdapterSpec {
    const char *name;
    const char *token;
    const char *invocation;
    int drop;
};
constexpr AdapterSpec kAdapters[] = {
    {"guardian-core", "<|guardian-core|>", "<guardian>", 1},
    {"decision", "<|decision|>", "### DECISION", 3},
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
    return "{\"type\": \"function\", \"function\": {\"name\": \"" +
           JsonEscape(t.name) + "\", \"description\": \"" +
           JsonEscape(t.description) + "\", \"parameters\": " +
           (t.parameters_json.empty() ? "{}" : t.parameters_json) + "}}";
}

const AdapterSpec *FindAdapter(std::string_view name,
                              std::string_view invocation_text)
{
    for (const auto &a : kAdapters) {
        if (!name.empty() && name == a.name) return &a;
        if (name.empty() && invocation_text.find(a.invocation) !=
                                std::string_view::npos)
            return &a;
    }
    return nullptr;
}

bool ContainsControl(std::string_view text)
{
    for (const auto &a : kAdapters)
        if (text.find(a.token) != std::string_view::npos) return true;
    return false;
}

std::string InsertInvocation(std::string content, const AdapterSpec &a)
{
    const size_t pos = content.rfind(a.invocation);
    if (pos == std::string::npos) return content;
    std::string out = content.substr(0, pos);
    out += a.token;
    out += content.substr(pos + static_cast<size_t>(a.drop));
    return out;
}

std::string RenderToolCalls(const std::vector<ToolCall> &calls)
{
    std::string out;
    for (size_t i = 0; i < calls.size(); ++i) {
        const ToolCall &c = calls[i];
        if (i > 0) out += "\n";
        out += "<tool_call>\n{\"name\": \"" + JsonEscape(c.name) +
               "\", \"arguments\": ";
        out += c.arguments_json.empty() ? "{}" : c.arguments_json;
        out += "}\n</tool_call>";
    }
    return out;
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
        const size_t begin = (tok.bos_id >= 0 && n > 0) ? 1 : 0;
        ids.insert(ids.end(), buf.begin() + begin, buf.begin() + n);
        return true;
    }
    return false;
}

} // namespace

bool RenderGraniteSwitchConversation(const ChatConversation &conv,
                                     std::string &out)
{
    out.clear();
    const ChatMessage *system = nullptr;
    size_t first = 0;
    if (!conv.messages.empty() && conv.messages.front().role == "system") {
        system = &conv.messages.front();
        first = 1;
    }

    std::string system_text;
    if (system) system_text = system->content;
    if (!conv.tools.empty()) {
        std::string tools_block =
            "You are a helpful assistant with access to the following tools. "
            "You may call one or more tools to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> "
            "XML tags:\n<tools>";
        for (const ToolSpec &t : conv.tools)
            tools_block += "\n" + ToolJson(t);
        tools_block +=
            "\n</tools>\n\nFor each tool call, return a json object with "
            "function name and arguments within <tool_call></tool_call> XML "
            "tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": "
            "<args-json-object>}\n</tool_call>. If a tool does not exist in "
            "the provided list of tools, notify the user that you do not have "
            "the ability to fulfill the request.";
        if (!system_text.empty()) system_text += "\n\n";
        system_text += tools_block;
    }

    // Adapter selection. The native template keys off adapter_name +
    // invocation text; an explicit control token already present wins over
    // automatic invocation detection.
    std::string_view invocation;
    const AdapterSpec *adapter = FindAdapter(conv.adapter_name, {});
    bool any_control = false;
    for (const auto &m : conv.messages) any_control |= ContainsControl(m.content);
    size_t target_idx = static_cast<size_t>(-1);
    if (!any_control) {
        if (adapter) {
            for (size_t i = 0; i < conv.messages.size(); ++i)
                if (conv.messages[i].role == "user" &&
                    conv.messages[i].content.find(adapter->invocation) !=
                        std::string::npos)
                    target_idx = i;
        } else {
            for (size_t i = 0; i < conv.messages.size(); ++i) {
                if (conv.messages[i].role != "user") continue;
                const AdapterSpec *a =
                    FindAdapter({}, conv.messages[i].content);
                if (a) {
                    adapter = a;
                    target_idx = i;
                }
            }
        }
    }

    if (!system_text.empty())
        out += "<|start_of_role|>system<|end_of_role|>" + system_text +
               "<|end_of_text|>\n";

    bool tool_group_open = false;
    const size_t total = conv.messages.size();
    for (size_t i = first; i < total; ++i) {
        const ChatMessage &m = conv.messages[i];
        std::string content = m.content;
        if (i == target_idx && adapter) content = InsertInvocation(content, *adapter);

        if (m.role == "assistant") {
            tool_group_open = false;
            out += "<|start_of_role|>assistant<|end_of_role|>" + content;
            if (!m.tool_calls.empty()) {
                if (!content.empty()) out += "\n";
                out += RenderToolCalls(m.tool_calls);
            }
            out += "<|end_of_text|>\n";
        } else if (m.role == "tool") {
            if (!tool_group_open) {
                out += "<|start_of_role|>user<|end_of_role|>";
                tool_group_open = true;
            }
            out += "\n<tool_response>\n" + content + "\n</tool_response>";
            const bool last = (i + 1 == total) || conv.messages[i + 1].role != "tool";
            if (last) {
                out += "<|end_of_text|>\n";
                tool_group_open = false;
            }
        } else {
            tool_group_open = false;
            out += "<|start_of_role|>" + m.role + "<|end_of_role|>" + content +
                   "<|end_of_text|>\n";
        }
    }

    if (adapter && target_idx == static_cast<size_t>(-1) && !any_control)
        out += adapter->token;
    out += "<|start_of_role|>assistant<|end_of_role|>";
    return true;
}

std::vector<ToolCall> ParseGraniteSwitchToolCallText(std::string_view text)
{
    std::vector<ToolCall> calls;
    constexpr std::string_view kOpen = "<tool_call>";
    constexpr std::string_view kClose = "</tool_call>";
    size_t pos = 0;
    while (true) {
        const size_t b = text.find(kOpen, pos);
        if (b == std::string_view::npos) break;
        const size_t e = text.find(kClose, b + kOpen.size());
        if (e == std::string_view::npos) break;
        const std::string body(text.substr(b + kOpen.size(),
                                           e - (b + kOpen.size())));
        pos = e + kClose.size();
        simdjson::dom::parser parser;
        simdjson::dom::element obj;
        if (parser.parse(simdjson::padded_string(body)).get(obj) !=
                simdjson::SUCCESS ||
            !obj.is_object())
            continue;
        ToolCall call;
        std::string_view name;
        if (obj["name"].get_string().get(name) != simdjson::SUCCESS) continue;
        call.name.assign(name);
        simdjson::dom::element args;
        if (obj["arguments"].get(args) == simdjson::SUCCESS) {
            std::string_view sv;
            if (args.is_string() &&
                args.get_string().get(sv) == simdjson::SUCCESS)
                call.arguments_json.assign(sv);
            else
                call.arguments_json = simdjson::to_string(args);
        } else {
            call.arguments_json = "{}";
        }
        call.id = "call_" + std::to_string(calls.size());
        calls.push_back(std::move(call));
    }
    return calls;
}

bool AppendGraniteSwitchIds(const ArrowTokenizerDb &tok, const std::string &text,
                            std::vector<int32_t> &out_ids)
{
    out_ids.clear();
    if (arrow_tokenizer_lookup(tok, "<|start_of_role|>", std::strlen("<|start_of_role|>")) < 0) return false;
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
        for (const char *s : kSentinels) {
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

bool ParseGraniteSwitchTokenCalls(const ArrowTokenizerDb &tok,
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
    out_calls = ParseGraniteSwitchToolCallText(text);
    if (out_calls.empty()) out_calls = tools::ParseToolCallsText(text);
    if (out_calls.empty()) out_calls = tools::ParseBareToolCallJson(text);
    return true;
}

} // namespace lsxcommon
