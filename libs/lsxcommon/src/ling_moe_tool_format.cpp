#include "lsxcommon/ling_moe_tool_format.h"

#include "lsxcommon/tool_format.h"

#include <absl/strings/str_cat.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <string_view>

namespace lsxcommon::ling {
namespace {

// Longest first so a shorter marker never shadows a longer one.
constexpr std::array<const char *, 14> kMarkers = {
    "<|role_end|>", "<tool_response>", "</tool_response>", "<tool_call>",
    "</tool_call>", "<arg_value>",     "</arg_value>",     "<arg_key>",
    "</arg_key>",   "</role>",         "<role>",           "<think>",
    "</think>",     "<|fim_end|>"};

bool IsMarkerText(std::string_view s)
{
    for (const char *m : kMarkers)
        if (s == m) return true;
    return false;
}

bool EncodeText(const ArrowTokenizerDb &tok, std::string_view text,
                std::vector<int32_t> &out)
{
    if (text.empty()) return true;
    std::vector<int32_t> buf(text.size() + 16);
    size_t cap = buf.size();
    const int32_t n =
        arrow_tokenizer_encode(tok, text.data(), text.size(), buf.data(), &cap);
    if (n < 0) {
        // A single word above the encoder's cap: fall back to a byte-wise walk.
        if (n != -3) return false;
        for (size_t i = 0; i < text.size();) {
            size_t step = 1;
            while (step < text.size() - i && step < 8) {
                std::vector<int32_t> b(32);
                size_t c = b.size();
                if (arrow_tokenizer_encode(tok, text.data() + i, step + 1,
                                           b.data(), &c) >= 0) {
                    ++step;
                    continue;
                }
                break;
            }
            std::vector<int32_t> b(32);
            size_t c = b.size();
            if (arrow_tokenizer_encode(tok, text.data() + i, step, b.data(),
                                       &c) < 0)
                return false;
            out.insert(out.end(), b.begin(), b.begin() + c);
            i += step;
        }
        return true;
    }
    out.insert(out.end(), buf.begin(), buf.begin() + n);
    return true;
}

std::string JsonEscape(std::string_view s)
{
    std::string out;
    out.reserve(s.size() + 8);
    for (char ch : s) {
        switch (ch) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (static_cast<unsigned char>(ch) < 0x20)
                out += absl::StrCat("\\u", static_cast<int>(ch));
            else
                out += ch;
        }
    }
    return out;
}

std::string RenderToolSpec(const ToolSpec &t)
{
    return absl::StrCat(
        "{\"type\":\"function\",\"function\":{\"name\":\"",
        JsonEscape(t.name), "\",\"description\":\"",
        JsonEscape(t.description), "\",\"parameters\":",
        t.parameters_json.empty() ? "{}" : t.parameters_json, "}}");
}

std::string RenderArguments(const ToolCall &tc)
{
    // The template iterates the arguments object in order. Re-emit its key
    // order when it is a flat JSON object; otherwise pass the blob as one
    // argument so nothing is lost.
    const std::string &j = tc.arguments_json;
    std::string out;
    size_t i = 0;
    while (i < j.size() && j[i] != '{') ++i;
    if (i >= j.size()) {
        if (!j.empty())
            out += absl::StrCat("<arg_key>arguments</arg_key>\n<arg_value>",
                                j, "</arg_value>");
        return out;
    }
    ++i;
    while (i < j.size() && j[i] != '}') {
        while (i < j.size() && (j[i] == ' ' || j[i] == ',' || j[i] == '\n'))
            ++i;
        if (i >= j.size() || j[i] == '}') break;
        if (j[i] != '"') break;
        ++i;
        std::string key;
        while (i < j.size() && j[i] != '"') {
            if (j[i] == '\\' && i + 1 < j.size()) {
                key += j[i + 1];
                i += 2;
                continue;
            }
            key += j[i++];
        }
        if (i < j.size()) ++i;  // closing quote
        while (i < j.size() && (j[i] == ' ' || j[i] == ':')) ++i;
        std::string val;
        if (i < j.size() && (j[i] == '{' || j[i] == '[')) {
            const char open = j[i];
            const char close = open == '{' ? '}' : ']';
            int depth = 0;
            bool in_str = false;
            for (; i < j.size(); ++i) {
                const char ch = j[i];
                if (ch == '"' && (i == 0 || j[i - 1] != '\\')) in_str = !in_str;
                if (!in_str) {
                    if (ch == open) ++depth;
                    if (ch == close && --depth == 0) {
                        val += ch;
                        ++i;
                        break;
                    }
                }
                val += ch;
            }
        } else if (i < j.size() && j[i] == '"') {
            ++i;
            while (i < j.size() && j[i] != '"') {
                if (j[i] == '\\' && i + 1 < j.size()) {
                    val += j[i];
                    ++i;
                }
                val += j[i++];
            }
            if (i < j.size()) ++i;
        } else {
            while (i < j.size() && j[i] != ',' && j[i] != '}') val += j[i++];
        }
        if (val.size() >= 2 && val.front() == '"' && val.back() == '"')
            val = val.substr(1, val.size() - 2);
        out += absl::StrCat("<arg_key>", key, "</arg_key>\n<arg_value>", val,
                            "</arg_value>");
    }
    return out;
}

std::string Trim(std::string_view s)
{
    size_t b = 0, e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\n' || s[b] == '\t' ||
                     s[b] == '\r'))
        ++b;
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\n' || s[e - 1] == '\t' ||
                     s[e - 1] == '\r'))
        --e;
    return std::string(s.substr(b, e - b));
}

bool ParseArgs(const std::string &body, std::string &json)
{
    json = "{";
    bool first = true;
    size_t i = 0;
    while ((i = body.find("<arg_key>", i)) != std::string::npos) {
        const size_t kb = i + 9;
        const size_t ke = body.find("</arg_key>", kb);
        if (ke == std::string::npos) return false;
        const std::string key = Trim(std::string_view(body).substr(kb, ke - kb));
        size_t vb = body.find("<arg_value>", ke);
        if (vb == std::string::npos) return false;
        vb += 11;
        const size_t ve = body.find("</arg_value>", vb);
        if (ve == std::string::npos) return false;
        std::string val = Trim(std::string_view(body).substr(vb, ve - vb));
        if (val.empty() || (val.front() != '{' && val.front() != '[' &&
                            val.front() != '"' && val != "true" &&
                            val != "false" && val != "null" &&
                            !(val.front() == '-' ||
                              (val.front() >= '0' && val.front() <= '9')))) {
            val = "\"" + JsonEscape(val) + "\"";
        }
        if (!first) json += ",";
        first = false;
        json += "\"" + JsonEscape(key) + "\":" + val;
        i = ve + 12;
    }
    json += "}";
    return !first;
}

} // namespace

bool EncodeLingWithControls(const ArrowTokenizerDb &tok, std::string_view text,
                            std::vector<int32_t> &out_ids)
{
    size_t pos = 0;
    while (pos < text.size()) {
        size_t best = std::string::npos;
        const char *hit = nullptr;
        for (const char *m : kMarkers) {
            const size_t at = text.find(m, pos);
            if (at != std::string::npos &&
                (best == std::string::npos || at < best)) {
                best = at;
                hit = m;
            }
        }
        if (hit == nullptr) {
            if (!EncodeText(tok, text.substr(pos), out_ids)) return false;
            break;
        }
        if (best > pos && !EncodeText(tok, text.substr(pos, best - pos), out_ids))
            return false;
        const int32_t id = arrow_tokenizer_lookup(tok, hit, std::strlen(hit));
        if (id >= 0)
            out_ids.push_back(id);
        else if (!EncodeText(tok, hit, out_ids))
            return false;
        pos = best + std::strlen(hit);
    }
    return true;
}

bool RenderLingConversation(const ArrowTokenizerDb &tok,
                            const ChatConversation &conv,
                            std::vector<int32_t> &out_ids)
{
    if (conv.messages.empty()) return false;
    const bool has_tools = !conv.tools.empty();
    std::string s = "<role>SYSTEM</role>";
    const bool first_system =
        conv.messages.front().role == "system";
    if (has_tools) {
        if (first_system) s += conv.messages.front().content + "\n";
        s += "# Tools\n\nYou may call one or more functions to assist with the "
             "user query.\n\nYou are provided with function signatures within "
             "<tools></tools> XML tags:\n<tools>";
        for (const auto &t : conv.tools) s += "\n" + RenderToolSpec(t);
        s += "\n</tools>\n\nIf none of the functions can be used, point it out. "
             "If the given question lacks the parameters required by the "
             "function, also point it out.\nIf you need to use a function, for "
             "each function call, output the function name and arguments within "
             "the following XML format:\n<tool_call>{function-name}\n"
             "<arg_key>{arg-key-1}</arg_key>\n"
             "<arg_value>{arg-value-1}</arg_value>\n"
             "<arg_key>{arg-key-2}</arg_key>\n"
             "<arg_value>{arg-value-2}</arg_value>\n...\n</tool_call>\n";
        s += "detailed thinking on<|role_end|>";
    } else {
        if (first_system) s += conv.messages.front().content + "\n";
        s += "detailed thinking on<|role_end|>";
    }

    for (size_t mi = 0; mi < conv.messages.size(); ++mi) {
        const auto &m = conv.messages[mi];
        if (m.role == "user") {
            s += "<role>HUMAN</role>" + m.content + "<|role_end|>";
        } else if (m.role == "system" && mi != 0) {
            s += "<role>SYSTEM</role>" + m.content + "<|role_end|>";
        } else if (m.role == "assistant") {
            s += "<role>ASSISTANT</role>\n<think></think>" + m.content;
            for (const auto &tc : m.tool_calls) {
                s += "<tool_call>" + tc.name;
                s += RenderArguments(tc);
                s += "\n</tool_call>";
            }
            s += "<|role_end|>";
        } else if (m.role == "tool") {
            if (mi == 0 || conv.messages[mi - 1].role != "tool")
                s += "<role>OBSERVATION</role>";
            s += "\n<tool_response>\n" + m.content + "\n</tool_response>";
            if (mi + 1 == conv.messages.size() ||
                conv.messages[mi + 1].role != "tool")
                s += "<|role_end|>";
        }
    }
    s += "<role>ASSISTANT</role>\n<think>";
    out_ids.clear();
    return EncodeLingWithControls(tok, s, out_ids);
}

bool ParseLingToolCalls(const ArrowTokenizerDb &tok,
                        const std::vector<int32_t> &gen_tokens,
                        std::vector<ToolCall> &out_calls)
{
    out_calls.clear();
    if (gen_tokens.empty()) return false;
    std::string text(gen_tokens.size() * 8 + 64, '\0');
    size_t cap = text.size();
    if (arrow_tokenizer_decode(tok, gen_tokens.data(), gen_tokens.size(),
                               text.data(), &cap) != 0)
        text.clear();
    else
        text.resize(cap);

    size_t pos = 0;
    while ((pos = text.find("<tool_call>", pos)) != std::string::npos) {
        const size_t body_b = pos + 11;
        const size_t end = text.find("</tool_call>", body_b);
        if (end == std::string::npos) break;
        std::string body = text.substr(body_b, end - body_b);
        const size_t nl = body.find('\n');
        const size_t ak = body.find("<arg_key>");
        size_t cut = std::string::npos;
        if (nl != std::string::npos) cut = nl;
        if (ak != std::string::npos && (cut == std::string::npos || ak < cut))
            cut = ak;
        const std::string name =
            Trim(cut == std::string::npos ? std::string_view(body)
                                          : std::string_view(body).substr(
                                                0, cut));
        const std::string args_body =
            cut == std::string::npos ? std::string() : body.substr(cut);
        std::string args;
        if (!ParseArgs(args_body, args)) args = "{}";
        if (!name.empty()) {
            ToolCall tc;
            tc.name = name;
            tc.arguments_json = args;
            out_calls.push_back(std::move(tc));
        }
        pos = end + 12;
    }
    if (!out_calls.empty()) return true;
    out_calls = tools::ParseToolCallsText(text);
    if (!out_calls.empty()) return true;
    out_calls = tools::ParseBareToolCallJson(text);
    return !out_calls.empty();
}

} // namespace lsxcommon::ling
