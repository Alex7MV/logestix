#include "lsxcommon/qwen_tool_format.h"

#include <simdjson.h>

#include <cstdio>
#include <cstring>
#include <string>

#include "lsxcommon/tool_format.h"

namespace lsxcommon {
namespace {

// Appends the BPE encoding of `text` to `ids`. arrow_tokenizer_encode
// auto-prepends bos_id when the tokenizer defines one; ChatML frames turns with
// <|im_start|> instead, so the prepended BOS is dropped.
bool AppendQwenEncoded(const ArrowTokenizerDb &tok, std::string_view text,
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

// Control sentinels injected as their own vocab ids.
constexpr const char *kQwenSentinels[] = {
    "<|im_start|>",   "<|im_end|>",     "<tool_call>", "</tool_call>",
    "<tool_response>", "</tool_response>", "<think>",   "</think>",
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

std::string QwenJsonEscape(std::string_view in)
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

std::string QwenSerialize(simdjson::dom::element e);

std::string QwenSerialize(simdjson::dom::element e)
{
    using simdjson::dom::element_type;
    switch (e.type()) {
        case element_type::OBJECT: {
            std::string out = "{";
            bool first = true;
            for (auto [k, v] : e.get_object()) {
                if (!first) out += ",";
                first = false;
                out += "\"" + QwenJsonEscape(k) + "\":" + QwenSerialize(v);
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
                out += QwenSerialize(v);
            }
            out += "]";
            return out;
        }
        case element_type::STRING:
            return "\"" + QwenJsonEscape(e.get_string().value_unsafe()) + "\"";
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

} // namespace

bool RenderQwenConversation(const ChatConversation &conv, std::string &out,
                            bool enable_thinking)
{
    out.clear();
    const bool has_tools = !conv.tools.empty();
    const bool first_system =
        !conv.messages.empty() && conv.messages[0].role == "system";
    // Untrusted content must never forge a control marker (prompt injection):
    // break any marker occurrence with a zero-width space before assembly.
    static const std::vector<std::string_view> kMarkers = [] {
        std::vector<std::string_view> m;
        for (const char *s : kQwenSentinels) m.emplace_back(s);
        return m;
    }();
    const auto esc = [](std::string_view s) {
        return tools::NeutralizeControlMarkers(s, kMarkers);
    };

    if (has_tools) {
        out += "<|im_start|>system\n";
        out +=
            "# Tools\n\nYou have access to the following functions:\n\n"
            "<tools>";
        for (const ToolSpec &t : conv.tools) {
            out += "\n{\"type\":\"function\",\"function\":{\"name\":\"";
            out += QwenJsonEscape(esc(t.name));
            out += "\",\"description\":\"";
            out += QwenJsonEscape(esc(t.description));
            out += "\",\"parameters\":";
            out += t.parameters_json.empty() ? std::string("{}")
                                             : esc(t.parameters_json);
            out += "}}";
        }
        out +=
            "\n</tools>\n\nIf you choose to call a function ONLY reply in the "
            "following format with NO suffix:\n\n"
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
        if (first_system && !conv.messages[0].content.empty())
            out += "\n\n" + esc(conv.messages[0].content);
        out += "<|im_end|>\n";
    } else if (first_system) {
        out +=
            "<|im_start|>system\n" + esc(conv.messages[0].content) +
            "<|im_end|>\n";
    }

    simdjson::dom::parser parser;
    bool in_tool_turn = false;
    for (size_t mi = 0; mi < conv.messages.size(); ++mi) {
        const ChatMessage &m = conv.messages[mi];
        if (m.role == "system") {
            // The Qwen template only permits a leading system message.
            if (mi != 0) return false;
            continue;
        }
        if (m.role == "user") {
            in_tool_turn = false;
            out += "<|im_start|>user\n" + esc(m.content) + "<|im_end|>\n";
        } else if (m.role == "assistant") {
            in_tool_turn = false;
            out += "<|im_start|>assistant\n" + esc(m.content);
            bool first = true;
            for (const ToolCall &c : m.tool_calls) {
                if (first) {
                    out += m.content.empty()
                               ? "<tool_call>\n<function="
                               : "\n\n<tool_call>\n<function=";
                    first = false;
                } else {
                    out += "\n<tool_call>\n<function=";
                }
                out += esc(c.name) + ">\n";
                if (!c.arguments_json.empty()) {
                    simdjson::dom::element args;
                    simdjson::padded_string padded(c.arguments_json);
                    if (parser.parse(padded).get(args) == simdjson::SUCCESS &&
                        args.type() ==
                            simdjson::dom::element_type::OBJECT) {
                        for (auto [k, v] : args.get_object()) {
                            out += "<parameter=";
                            out += esc(k);
                            out += ">\n";
                            if (v.type() ==
                                simdjson::dom::element_type::STRING)
                                out += esc(v.get_string().value_unsafe());
                            else
                                out += QwenSerialize(v);
                            out += "\n</parameter>\n";
                        }
                    }
                }
                out += "</function>\n</tool_call>";
            }
            out += "<|im_end|>\n";
        } else if (m.role == "tool") {
            // A run of consecutive tool messages shares one user turn; only the
            // first opens <|im_start|>user and only the last closes <|im_end|>.
            if (!in_tool_turn) {
                out += "<|im_start|>user";
                in_tool_turn = true;
            }
            out += "\n<tool_response>\n" + esc(m.content) +
                   "\n</tool_response>";
            const bool next_is_tool =
                mi + 1 < conv.messages.size() &&
                conv.messages[mi + 1].role == "tool";
            if (!next_is_tool) {
                out += "<|im_end|>\n";
                in_tool_turn = false;
            }
        } else {
            return false;
        }
    }

    out += "<|im_start|>assistant\n<think>";
    out += enable_thinking ? "\n" : "\n\n</think>\n\n";
    return true;
}

std::vector<ToolCall> ParseQwenToolCallText(std::string_view text)
{
    std::vector<ToolCall> calls;
    simdjson::dom::parser parser;
    constexpr std::string_view kOpen = "<tool_call>";
    constexpr std::string_view kClose = "</tool_call>";
    constexpr std::string_view kFn = "<function=";
    constexpr std::string_view kFnEnd = "</function>";
    constexpr std::string_view kParam = "<parameter=";
    constexpr std::string_view kParamEnd = "</parameter>";
    size_t pos = 0;
    while (true) {
        const size_t b = text.find(kOpen, pos);
        if (b == std::string_view::npos) break;
        const size_t e = text.find(kClose, b + kOpen.size());
        if (e == std::string_view::npos) break;
        const std::string_view body =
            text.substr(b + kOpen.size(), e - (b + kOpen.size()));
        pos = e + kClose.size();

        // Canonical `<function=NAME>`; some checkpoints drop the `function=`
        // prefix and emit a bare `<NAME>` tag.
        size_t name_start, name_end;
        const size_t fb = body.find(kFn);
        if (fb != std::string_view::npos) {
            name_start = fb + kFn.size();
            name_end = body.find('>', name_start);
        } else {
            const size_t lt = body.find('<');
            if (lt == std::string_view::npos) continue;
            if (lt + 1 < body.size() && body[lt + 1] == '/') continue;
            name_start = lt + 1;
            name_end = body.find('>', name_start);
        }
        if (name_end == std::string_view::npos) continue;
        const std::string name(
            TrimAscii(body.substr(name_start, name_end - name_start)));
        if (name.empty()) continue;
        const size_t fe = body.find(kFnEnd, name_end);
        const size_t args_end = (fe == std::string_view::npos) ? body.size() : fe;
        const std::string_view region =
            body.substr(name_end + 1,
                        args_end > name_end + 1 ? args_end - (name_end + 1) : 0);

        std::string args = "{";
        bool first = true;
        size_t ap = 0;
        while (true) {
            const size_t pb = region.find(kParam, ap);
            if (pb == std::string_view::npos) break;
            const size_t kstart = pb + kParam.size();
            const size_t kend = region.find('>', kstart);
            if (kend == std::string_view::npos) break;
            const std::string key(
                TrimAscii(region.substr(kstart, kend - kstart)));
            const size_t vstart = kend + 1;
            const size_t vend = region.find(kParamEnd, vstart);
            if (vend == std::string_view::npos) break;
            const std::string_view raw =
                TrimAscii(region.substr(vstart, vend - vstart));
            ap = vend + kParamEnd.size();
            if (!first) args += ",";
            first = false;
            args += "\"" + QwenJsonEscape(key) + "\":";
            // Preserve JSON types (numbers/objects); otherwise a string.
            // The Qwen XML format carries no string flag, so a string value
            // that is itself valid JSON (e.g. "42") is inherently ambiguous and
            // round-trips as that JSON type.
            simdjson::dom::element ev;
            simdjson::padded_string padded(raw);
            if (parser.parse(padded).get(ev) == simdjson::SUCCESS)
                args += QwenSerialize(ev);
            else
                args += "\"" + QwenJsonEscape(raw) + "\"";
        }
        args += "}";
        ToolCall call;
        call.name = name;
        call.arguments_json = std::move(args);
        call.id = "call_" + std::to_string(calls.size());
        calls.push_back(std::move(call));
    }
    return calls;
}

bool QwenChatSentinelsPresent(const ArrowTokenizerDb &tok)
{
    return arrow_tokenizer_lookup(tok, "<|im_start|>", 12) >= 0 &&
           arrow_tokenizer_lookup(tok, "<|im_end|>", 10) >= 0;
}

bool AppendQwenChatIds(const ArrowTokenizerDb &tok, const std::string &text,
                       std::vector<int32_t> &out_ids)
{
    out_ids.clear();
    if (!QwenChatSentinelsPresent(tok)) return false;
    size_t i = 0;
    size_t run = 0;
    auto flush = [&](size_t end) -> bool {
        if (end > run &&
            !AppendQwenEncoded(tok,
                               std::string_view(text).substr(run, end - run),
                               out_ids))
            return false;
        return true;
    };
    while (i < text.size()) {
        bool matched = false;
        for (const char *s : kQwenSentinels) {
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

bool ParseQwenTokenCalls(const ArrowTokenizerDb &tok,
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
    out_calls = ParseQwenToolCallText(text);
    if (out_calls.empty()) out_calls = tools::ParseToolCallsText(text);
    if (out_calls.empty()) out_calls = tools::ParseBareToolCallJson(text);
    return true;
}

} // namespace lsxcommon
