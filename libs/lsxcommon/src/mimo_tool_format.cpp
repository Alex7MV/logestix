#include "lsxcommon/mimo_tool_format.h"
#include <lsxnetwork/json_escape.h>

#include <simdjson.h>

#include <cstdio>
#include <cstring>
#include <string>

#include "lsxcommon/tool_format.h"

namespace lsxcommon {
using lsxnetwork::json::JsonEscape;
namespace mimo {
namespace {

// Appends the BPE encoding of `text` to `ids`. arrow_tokenizer_encode
// auto-prepends bos_id when the tokenizer defines one; ChatML frames turns with
// <|im_start|> instead, so the prepended BOS is dropped.
bool AppendMimoEncoded(const ArrowTokenizerDb &tok, std::string_view text,
                       std::vector<int32_t> &ids)
{
    for (size_t cap = 256; cap <= 65536; cap *= 2) {
        std::vector<int32_t> buf(cap, 0);
        size_t enc_cap = cap;
        const int32_t n = arrow_tokenizer_encode(tok, text.data(), text.size(),
                                                 buf.data(), &enc_cap);
        if (n == -2) continue;
        if (n < 0) return false;
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

// Control sentinels injected as their own vocab ids.
constexpr const char *kMimoSentinels[] = {
    "<|im_start|>", "<|im_end|>",
};

// Markers neutralized inside untrusted content (id-injected sentinels plus the
// plain-text control tags).
constexpr const char *kMimoControlMarkers[] = {
    "<|im_start|>", "<|im_end|>", "<think>",      "</think>",
    "<tool_call>",  "</tool_call>", "<function=", "<parameter=",
    "<tools>",      "</tools>",
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

std::string MimoSerialize(simdjson::dom::element e);

std::string MimoSerialize(simdjson::dom::element e)
{
    using simdjson::dom::element_type;
    switch (e.type()) {
        case element_type::OBJECT: {
            std::string out = "{";
            bool first = true;
            for (auto [k, v] : e.get_object()) {
                if (!first) out += ",";
                first = false;
                out += "\"" + JsonEscape(k) + "\":" + MimoSerialize(v);
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
                out += MimoSerialize(v);
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
            std::snprintf(buf, sizeof(buf), "%.17g", e.get_double().value_unsafe());
            return std::string(buf);
        }
        case element_type::BOOL:
            return e.get_bool().value_unsafe() ? "true" : "false";
        default:
            return "null";
    }
}

std::string RenderTools(const std::vector<ToolSpec> &tools)
{
    std::string out = "You are provided with the following tools:\n\n<tools>";
    for (const ToolSpec &t : tools) {
        out += "\n{\"type\":\"function\",\"function\":{\"name\":\"";
        out += JsonEscape(t.name);
        out += "\",\"description\":\"";
        out += JsonEscape(t.description);
        out += "\",\"parameters\":";
        out += t.parameters_json.empty() ? std::string("{}") : t.parameters_json;
        out += "}}";
    }
    out += "\n</tools>";
    return out;
}

} // namespace

bool RenderMimoConversation(const ChatConversation &conv, std::string &out,
                            bool add_generation_prompt, bool enable_thinking)
{
    out.clear();
    static const std::vector<std::string_view> kMarkers = [] {
        std::vector<std::string_view> m;
        for (const char *s : kMimoControlMarkers) m.emplace_back(s);
        return m;
    }();
    const auto esc = [](std::string_view s) {
        return tools::NeutralizeControlMarkers(s, kMarkers);
    };

    // Top-level tools render as their own system turn.
    if (!conv.tools.empty())
        out += "<|im_start|>system\n" + RenderTools(conv.tools) + "<|im_end|>";

    simdjson::dom::parser parser;
    for (const ChatMessage &m : conv.messages) {
        if (m.role == "system" || m.role == "user" || m.role == "tool") {
            out += "<|im_start|>" + m.role + "\n" + esc(m.content) + "<|im_end|>";
        } else if (m.role == "assistant") {
            // reasoning_content is not carried in ChatMessage; history turns
            // render with an empty think block.
            out += "<|im_start|>assistant\n<think></think>" + esc(m.content);
            for (const ToolCall &c : m.tool_calls) {
                out += "<tool_call><function=" + esc(c.name) + ">";
                if (!c.arguments_json.empty()) {
                    simdjson::dom::element args;
                    simdjson::padded_string padded(c.arguments_json);
                    if (parser.parse(padded).get(args) == simdjson::SUCCESS &&
                        args.type() == simdjson::dom::element_type::OBJECT) {
                        for (auto [k, v] : args.get_object()) {
                            out += "<parameter=" + esc(k) + ">";
                            if (v.type() == simdjson::dom::element_type::STRING)
                                out += esc(v.get_string().value_unsafe());
                            else
                                out += MimoSerialize(v);
                            out += "</parameter>";
                        }
                    }
                }
                out += "</function></tool_call>";
            }
            out += "<|im_end|>";
        } else {
            return false;
        }
    }

    if (add_generation_prompt) {
        out += "<|im_start|>assistant\n";
        if (!enable_thinking) out += "<think></think>";
    }
    return true;
}

std::vector<ToolCall> ParseMimoToolCallText(std::string_view text)
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
            args += "\"" + JsonEscape(key) + "\":";
            // Preserve JSON types; otherwise a string.
            simdjson::dom::element ev;
            simdjson::padded_string padded(raw);
            if (parser.parse(padded).get(ev) == simdjson::SUCCESS)
                args += MimoSerialize(ev);
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
    return calls;
}

bool MimoChatSentinelsPresent(const ArrowTokenizerDb &tok)
{
    return arrow_tokenizer_lookup(tok, "<|im_start|>", 12) >= 0 &&
           arrow_tokenizer_lookup(tok, "<|im_end|>", 10) >= 0;
}

bool AppendMimoChatIds(const ArrowTokenizerDb &tok, const std::string &text,
                       std::vector<int32_t> &out_ids)
{
    out_ids.clear();
    if (!MimoChatSentinelsPresent(tok)) return false;
    size_t i = 0;
    size_t run = 0;
    auto flush = [&](size_t end) -> bool {
        if (end > run &&
            !AppendMimoEncoded(tok, std::string_view(text).substr(run, end - run),
                               out_ids))
            return false;
        return true;
    };
    while (i < text.size()) {
        bool matched = false;
        for (const char *s : kMimoSentinels) {
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

bool BuildChatInputIds(const ArrowTokenizerDb &tok, const ChatConversation &conv,
                       bool add_generation_prompt, bool enable_thinking,
                       std::vector<int32_t> &out_ids)
{
    std::string text;
    if (!RenderMimoConversation(conv, text, add_generation_prompt, enable_thinking))
        return false;
    return AppendMimoChatIds(tok, text, out_ids);
}

bool ParseChatToolCalls(const ArrowTokenizerDb &tok,
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
                                              gen_tokens.size(), text.data(), &cap);
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
    out_calls = ParseMimoToolCallText(text);
    if (out_calls.empty()) out_calls = tools::ParseToolCallsText(text);
    if (out_calls.empty()) out_calls = tools::ParseBareToolCallJson(text);
    return true;
}

} // namespace mimo
} // namespace lsxcommon
