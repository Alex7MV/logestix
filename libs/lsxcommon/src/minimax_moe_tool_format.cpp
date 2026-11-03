#include "lsxcommon/minimax_moe_tool_format.h"

#include "lsxcommon/tool_format.h"

#include <lsxnetwork/json_escape.h>

#include <simdjson.h>

#include <cstring>
#include <string>
#include <string_view>

namespace lsxcommon::minimax {
namespace {

using lsxnetwork::json::JsonEscape;

// Control markers (chat_template.jinja literals).
constexpr const char *kNs = "]<]minimax[>[";
constexpr const char *kBod = "]~!b[";
constexpr const char *kBos = "]~b]";
constexpr const char *kEos = "[e~[";
constexpr const char *kToolBegin = "]<]minimax[>[<tool_call>";
constexpr const char *kToolEnd = "]<]minimax[>[</tool_call>";
constexpr const char *kThinkBegin = "<mm:think>";
constexpr const char *kThinkEnd = "</mm:think>";

constexpr const char *kDefaultSystem =
    "Your model version is MiniMax-M3, developed by MiniMax. Knowledge cutoff: "
    "January 2026. Founded in early 2022, MiniMax is a global AI foundation "
    "model company committed to advancing the frontiers of AI towards AGI.";
constexpr const char *kDefaultDeveloper = "You are a helpful assistant.";

// Only REAL vocabulary pieces are injected as their own ids. `<tools>` /
// `<tool>` / `<response>` are ordinary text in the template (BPE-encoded), and
// the tool-call wrapper is the concatenation of two added tokens
// (`]<]minimax[>[` + `<tool_call>`), so it is matched as a prefix pair.
constexpr const char *kSentinels[] = {
    kBod,        kBos,          kEos,          kNs,
    "<tool_call>", "</tool_call>", kThinkBegin, kThinkEnd,
};

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

// Appends the BPE encoding of `text` (never a marker).
bool AppendEncoded(const ArrowTokenizerDb &tok, std::string_view text,
                   std::vector<int32_t> &ids)
{
    if (text.empty()) return true;
    for (size_t cap = 256; cap <= (1u << 20); cap *= 2) {
        std::vector<int32_t> buf(cap, 0);
        size_t enc_cap = cap;
        const int32_t n = arrow_tokenizer_encode(tok, text.data(), text.size(),
                                                 buf.data(), &enc_cap);
        if (n == -2) continue;
        if (n < 0) return false;
        // The engine never auto-prepends BOS for this checkpoint (the
        // template spells `]~b]` out); drop one defensively if a db carries it.
        // Strip the auto-prepended BOS only when it is actually there: the
        // single-stage tokenizer kinds (7..10) do not prepend one, and
        // an unconditional strip would eat the first real token.
        const size_t begin =
            (tok.bos_id >= 0 && n > 0 && buf[0] == tok.bos_id) ? 1 : 0;
        ids.insert(ids.end(), buf.begin() + begin, buf.begin() + n);
        return true;
    }
    return false;
}

// Splits `text` on the known markers and injects each marker as its own id.
bool AppendTextWithSentinels(const ArrowTokenizerDb &tok,
                             const std::string &text,
                             std::vector<int32_t> &ids)
{
    size_t i = 0;
    size_t run = 0;
    auto flush = [&](size_t end) -> bool {
        if (end > run)
            return AppendEncoded(
                tok, std::string_view(text).substr(run, end - run), ids);
        return true;
    };
    while (i < text.size()) {
        bool matched = false;
        for (const char *s : kSentinels) {
            const size_t n = std::strlen(s);
            if (text.compare(i, n, s) != 0) continue;
            if (!flush(i)) return false;
            const int32_t id = arrow_tokenizer_lookup(tok, s, n);
            if (id < 0) return false;
            ids.push_back(id);
            i += n;
            run = i;
            matched = true;
            break;
        }
        if (!matched) ++i;
    }
    return flush(text.size());
}

// Recursive XML -> JSON (the inverse of the template's `to_xml` macro).
std::string XmlToJson(std::string_view body);

// Extracts the text of the first `<tag ...>...</tag>` in [p, end); returns
// false when absent. `content_begin`/`content_end` delimit the inner body.

std::string XmlLeaf(std::string_view v)
{
    // The template renders scalars verbatim (booleans as true/false), so a
    // numeric or boolean leaf keeps its JSON type; everything else is a string.
    const std::string_view t = TrimAscii(v);
    if (t == "true" || t == "false" || t == "null") return std::string(t);
    bool numeric = !t.empty();
    for (size_t i = 0; i < t.size() && numeric; ++i) {
        const char c = t[i];
        if (!((c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.' ||
              c == 'e' || c == 'E'))
            numeric = false;
    }
    if (numeric) {
        // Only accept a value simdjson would parse the same way.
        simdjson::dom::parser p;
        simdjson::dom::element e;
        if (p.parse(std::string(t)).get(e) == simdjson::SUCCESS) return std::string(t);
    }
    return "\"" + JsonEscape(std::string(t)) + "\"";
}

// Parses the children of `body` into a JSON object (mapping) or, when the
// element repeats, into an array of `<item>` children.
std::string XmlChildrenToJson(std::string_view body)
{
    std::vector<std::string> names;
    std::vector<std::string> values;
    size_t i = 0;
    while (i < body.size()) {
        const size_t open = body.find(std::string(kNs) + "<", i);
        if (open == std::string_view::npos) break;
        const size_t name_begin = open + std::strlen(kNs) + 1;
        const size_t gt = body.find('>', name_begin);
        if (gt == std::string_view::npos) break;
        const std::string_view name = body.substr(name_begin, gt - name_begin);
        if (!name.empty() && name.front() == '/') {
            i = gt + 1;
            continue;
        }
        const std::string close = std::string(kNs) + "</" + std::string(name) + ">";
        const size_t c = body.find(close, gt + 1);
        if (c == std::string_view::npos) break;
        const std::string_view inner = body.substr(gt + 1, c - gt - 1);
        names.emplace_back(name);
        values.push_back(XmlToJson(inner));
        i = c + close.size();
    }
    // All-`item` children with >1 entry => array; a single `item` => array of 1.
    if (!names.empty()) {
        bool all_items = true;
        for (const auto &n : names)
            if (n != "item") all_items = false;
        if (all_items) {
            std::string out = "[";
            for (size_t k = 0; k < values.size(); ++k) {
                if (k) out += ",";
                out += values[k];
            }
            return out + "]";
        }
    }
    if (names.empty()) return XmlLeaf(body);
    std::string out = "{";
    for (size_t k = 0; k < names.size(); ++k) {
        if (k) out += ",";
        out += "\"" + JsonEscape(names[k]) + "\":" + values[k];
    }
    return out + "}";
}

std::string XmlToJson(std::string_view body)
{
    // A body with nested elements is a mapping; a plain body is a leaf.
    if (body.find(std::string(kNs) + "<") == std::string_view::npos)
        return XmlLeaf(body);
    return XmlChildrenToJson(body);
}

} // namespace

bool MinimaxChatSentinelsPresent(const ArrowTokenizerDb &tok)
{
    return arrow_tokenizer_lookup(tok, kBos, std::strlen(kBos)) >= 0 &&
           arrow_tokenizer_lookup(tok, kEos, std::strlen(kEos)) >= 0 &&
           arrow_tokenizer_lookup(tok, kNs, std::strlen(kNs)) >= 0 &&
           arrow_tokenizer_lookup(tok, "<tool_call>", 11) >= 0;
}

std::string RenderMinimaxInvoke(const ToolCall &call)
{
    std::string out = std::string(kNs) + "<invoke name=\"" + call.name + "\">";
    if (!call.arguments_json.empty()) {
        simdjson::dom::parser parser;
        simdjson::dom::element args;
        if (parser.parse(call.arguments_json).get(args) == simdjson::SUCCESS &&
            args.is_object()) {
            for (auto [k, v] : args.get_object()) {
                std::string sv;
                simdjson::error_code ec = simdjson::SUCCESS;
                if (v.is_string()) {
                    std::string_view s;
                    ec = v.get_string().get(s);
                    if (ec == simdjson::SUCCESS) sv = std::string(s);
                } else {
                    // Nested objects/arrays are recursively expanded as XML.
                    sv = simdjson::to_string(v);
                    if (sv.size() >= 2 && sv.front() == '"' &&
                        sv.back() == '"')
                        sv = sv.substr(1, sv.size() - 2);
                }
                if (ec != simdjson::SUCCESS) continue;
                out += std::string(kNs) + "<" + std::string(k) + ">" + sv +
                       std::string(kNs) + "</" + std::string(k) + ">";
            }
        }
    }
    out += std::string(kNs) + "</invoke>";
    return out;
}

std::vector<ToolCall> ParseMinimaxInvokeText(const std::string &text)
{
    std::vector<ToolCall> calls;
    size_t pos = 0;
    const std::string invoke_open = std::string(kNs) + "<invoke";
    while (true) {
        const size_t at = text.find(invoke_open, pos);
        if (at == std::string::npos) break;
        const size_t name_at = text.find("name=\"", at);
        if (name_at == std::string::npos) break;
        const size_t name_begin = name_at + 6;
        const size_t name_end = text.find('"', name_begin);
        if (name_end == std::string::npos) break;
        const size_t body_begin = text.find('>', name_end);
        if (body_begin == std::string::npos) break;
        const std::string invoke_close = std::string(kNs) + "</invoke>";
        const size_t body_end = text.find(invoke_close, body_begin);
        if (body_end == std::string::npos) break;
        ToolCall call;
        call.name = text.substr(name_begin, name_end - name_begin);
        call.arguments_json = XmlChildrenToJson(
            std::string_view(text).substr(body_begin + 1,
                                          body_end - body_begin - 1));
        if (call.arguments_json.empty() || call.arguments_json.front() != '{')
            call.arguments_json = "{}";
        calls.push_back(std::move(call));
        pos = body_end + invoke_close.size();
    }
    for (size_t i = 0; i < calls.size(); ++i)
        if (calls[i].id.empty()) calls[i].id = "call_" + std::to_string(i);
    return calls;
}

MinimaxRenderPolicy MinimaxFlashRenderPolicy()
{
    MinimaxRenderPolicy p;
    p.default_system =
        "Your model version is MiniMax-M3.1, developed by MiniMax. Knowledge "
        "cutoff: January 2026. Founded in early 2022, MiniMax is a global AI "
        "foundation model company committed to advancing the frontiers of AI "
        "towards AGI.";
    p.thinking_mode_line =
        "Current thinking mode: adaptive. You are encouraged to think for "
        "complex decision-making, multi-step reasoning, or when analyzing "
        "function/tool results.";
    p.system_suffix = "\n\n<model_id>8bb8149fe</model_id>";
    p.generation_think_end = false;
    p.developer_blocks = true;
    p.batch_tool_results = true;
    return p;
}

bool RenderMinimaxConversation(const ArrowTokenizerDb &tok,
                               const ChatConversation &conv,
                               std::vector<int32_t> &out_ids)
{
    return RenderMinimaxConversation(tok, conv, MinimaxRenderPolicy{}, out_ids);
}

bool RenderMinimaxConversation(const ArrowTokenizerDb &tok,
                               const ChatConversation &conv,
                               const MinimaxRenderPolicy &policy,
                               std::vector<int32_t> &out_ids)
{
    out_ids.clear();
    if (!MinimaxChatSentinelsPresent(tok)) return false;

    // Split system/developer out like the template: `root` = system, then a
    // system/developer message = developer; everything else is the transcript.
    const ChatMessage *system_msg = nullptr;
    const ChatMessage *dev_msg = nullptr;
    size_t first = 0;
    if (!conv.messages.empty() &&
        (conv.messages[0].role == "root" ||
         conv.messages[0].role == "system" ||
         conv.messages[0].role == "developer")) {
        system_msg = &conv.messages[0];
        first = 1;
        if (conv.messages[0].role != "root" && conv.messages.size() > 1 &&
            (conv.messages[1].role == "system" ||
             conv.messages[1].role == "developer")) {
            dev_msg = &conv.messages[1];
            first = 2;
        } else if (conv.messages[0].role == "root" &&
                   conv.messages.size() > 1 &&
                   (conv.messages[1].role == "system" ||
                    conv.messages[1].role == "developer")) {
            dev_msg = &conv.messages[1];
            first = 2;
        }
    }

    std::string text;
    text += std::string(kBod) + kBos + "system\n";
    text += system_msg && !system_msg->content.empty()
                ? system_msg->content
                : (policy.default_system ? policy.default_system
                                         : kDefaultSystem);
    text += std::string("\n\n<thinking_instructions>\n"
                        "You have a thinking capability that allows you to "
                        "reason step by step before responding. When thinking "
                        "is enabled, wrap your reasoning in ") +
            kThinkBegin + kThinkEnd +
            " tags before your response. When thinking is disabled, begin "
            "your response directly after the " + kThinkEnd +
            " prefix. When thinking is adaptive, decide on your own whether "
            "to think for the current turn.\n" +
            (policy.thinking_mode_line
                 ? policy.thinking_mode_line
                 : "Current thinking mode: disabled. Do not output any "
                   "thinking process.") +
            "\n</thinking_instructions>";
    if (policy.system_suffix) text += policy.system_suffix;
    text += kEos;
    text += "\n";

    text += std::string(kBos) + "developer\n";
    text += dev_msg && !dev_msg->content.empty() ? dev_msg->content
                                                 : kDefaultDeveloper;
    if (!conv.tools.empty()) {
        text += "\n\n# Tools\nYou may call one or more tools to assist with "
                "the user query.\nHere are the tools available in JSONSchema "
                "format:\n\n<tools>\n";
        for (const auto &t : conv.tools) {
            text += "<tool>{";
            text += "\"name\":\"" + JsonEscape(t.name) + "\"";
            if (!t.description.empty())
                text += ",\"description\":\"" + JsonEscape(t.description) + "\"";
            if (!t.parameters_json.empty())
                text += ",\"parameters\":" + t.parameters_json;
            text += "}</tool>\n";
        }
        text += "</tools>\n\n";
        text += std::string("To call tools, wrap all invocations in a single ") +
                kToolBegin + kToolEnd +
                " block. Parameter values containing nested objects or arrays "
                "are recursively expanded into XML elements. Example:\n\n" +
                kToolBegin + "\n" + kNs + "<invoke name=\"tool-name-1\">" + kNs +
                "<param-1>value-1" + kNs + "</param-1>" + kNs + "</invoke>\n" +
                kToolEnd;
    }
    text += kEos;
    text += "\n";

    for (size_t i = first; i < conv.messages.size(); ++i) {
        const ChatMessage &m = conv.messages[i];
        if (m.role == "assistant") {
            text += std::string(kBos) + "ai\n";
            // No reasoning_content is ever rendered for outbound turns; the
            // template then prefixes the think-END marker.
            text += kThinkEnd;
            text += m.content;
            if (!m.tool_calls.empty()) {
                text += kToolBegin;
                text += "\n";
                for (const auto &c : m.tool_calls)
                    text += RenderMinimaxInvoke(c) + "\n";
                text += kToolEnd;
            }
            text += kEos;
            text += "\n";
        } else if (policy.developer_blocks &&
                   (m.role == "system" || m.role == "developer")) {
            // Mid-conversation system/developer turns map to the lower-priority
            // `developer` role and keep their position (M3.1 template).
            text += std::string(kBos) + "developer\n" + m.content + kEos + "\n";
        } else if (m.role == "tool") {
            if (policy.batch_tool_results) {
                const bool first_of_run =
                    i == first || conv.messages[i - 1].role != "tool";
                const bool last_of_run =
                    i + 1 >= conv.messages.size() ||
                    conv.messages[i + 1].role != "tool";
                if (first_of_run) text += std::string(kBos) + "tool";
                text += "\n<response>" + m.content + "</response>";
                if (last_of_run) {
                    text += kEos;
                    text += "\n";
                }
            } else {
                text += std::string(kBos) + "tool\n<response>" + m.content +
                        "</response>" + kEos + "\n";
            }
        } else {  // user (and anything else)
            text += std::string(kBos) + "user\n" + m.content + kEos + "\n";
        }
    }

    // Generation prompt: M3 (thinking disabled) prefixes the think-END
    // marker; M3.1-Flash is fixed to ADAPTIVE and emits no prefix, letting the
    // model decide whether to open a `</mm:think>`/`<mm:think>` block.
    text += std::string(kBos) + "ai\n";
    if (policy.generation_think_end) text += kThinkEnd;
    return AppendTextWithSentinels(tok, text, out_ids);
}

bool ParseMinimaxToolCalls(const ArrowTokenizerDb &tok,
                           const std::vector<int32_t> &gen_tokens,
                           std::vector<ToolCall> &out_calls)
{
    out_calls.clear();
    if (gen_tokens.empty()) return true;
    // Assemble the text WITH special tokens: the MiniMax markers are
    // special:true, so `arrow_tokenizer_decode` would drop exactly the
    // structure we need.
    std::string text;
    text.reserve(gen_tokens.size() * 4);
    for (int32_t id : gen_tokens) {
        if (id < 0 || id >= tok.vocab_size) continue;
        const int64_t b = tok.token_offsets[id];
        const int64_t e = tok.token_offsets[id + 1];
        text.append(reinterpret_cast<const char *>(tok.token_bytes + b),
                    static_cast<size_t>(e - b));
    }

    const size_t begin = text.find(kToolBegin);
    const size_t end = text.find(kToolEnd);
    std::string body;
    if (begin != std::string::npos) {
        const size_t from = begin + std::strlen(kToolBegin);
        body = end != std::string::npos && end > from
                   ? text.substr(from, end - from)
                   : text.substr(from);
    } else {
        body = text;
    }
    out_calls = ParseMinimaxInvokeText(body);
    if (out_calls.empty()) out_calls = tools::ParseToolCallsText(text);
    if (out_calls.empty()) out_calls = tools::ParseBareToolCallJson(text);
    return true;
}

} // namespace lsxcommon::minimax
