#include "muse_tool_format.h"

#include "lsxcommon/tool_format.h"

#include <lsxnetwork/json_escape.h>

#include <simdjson.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>

namespace lsxcommon::muse {
namespace {

using lsxnetwork::json::JsonEscape;

// Channel markers (chat_template.jinja literals; ALL special:true vocab ids).
constexpr const char *kBos = "<|begin_of_text|>";
constexpr const char *kStart = "<|start|>";
constexpr const char *kMessage = "<|message|>";
constexpr const char *kEot = "<|eot|>";
constexpr const char *kEom = "<|eom|>";

constexpr const char *kSentinels[] = {kBos, kStart, kMessage, kEot, kEom};

// ATEM XML tags (plain text, BPE-encoded).
constexpr const char *kCallsBegin = "<atem:function_calls>";
constexpr const char *kCallsEnd = "</atem:function_calls>";
constexpr const char *kInvokeBegin = "<atem:invoke name=\"";
constexpr const char *kInvokeOpen = "\">";
constexpr const char *kInvokeEnd = "</atem:invoke>";
constexpr const char *kParamBegin = "<atem:parameter name=\"";
constexpr const char *kParamOpen = "\">";
constexpr const char *kParamEnd = "</atem:parameter>";

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

// Python json.dumps string escaping with ensure_ascii=True (what Jinja's
// `tojson` produces for the tool blobs; `<`, `>`, `&`, `'` are additionally
// escaped by Jinja's html-safe wrapper).
void PyJsonString(std::string_view s, std::string &out)
{
    out += '"';
    for (size_t i = 0; i < s.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        switch (c) {
        case '"': out += "\\\""; continue;
        case '\\': out += "\\\\"; continue;
        case '\n': out += "\\n"; continue;
        case '\r': out += "\\r"; continue;
        case '\t': out += "\\t"; continue;
        case '\b': out += "\\b"; continue;
        case '\f': out += "\\f"; continue;
        case '<': out += "\\u003c"; continue;
        case '>': out += "\\u003e"; continue;
        case '&': out += "\\u0026"; continue;
        case '\'': out += "\\u0027"; continue;
        default: break;
        }
        if (c < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\u%04x", c);
            out += buf;
        } else if (c < 0x80) {
            out += static_cast<char>(c);
        } else {
            int len = 1;
            unsigned cp = c;
            if ((c & 0xE0) == 0xC0) { len = 2; cp = c & 0x1F; }
            else if ((c & 0xF0) == 0xE0) { len = 3; cp = c & 0x0F; }
            else if ((c & 0xF8) == 0xF0) { len = 4; cp = c & 0x07; }
            for (int k = 1; k < len && i + k < s.size(); ++k)
                cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
            i += static_cast<size_t>(len - 1);
            char buf[16];
            if (cp >= 0x10000) {
                const unsigned v = cp - 0x10000;
                std::snprintf(buf, sizeof(buf), "\\u%04x\\u%04x",
                              static_cast<unsigned>(0xD800 + (v >> 10)),
                              static_cast<unsigned>(0xDC00 + (v & 0x3FF)));
            } else {
                std::snprintf(buf, sizeof(buf), "\\u%04x", cp);
            }
            out += buf;
        }
    }
    out += '"';
}

// Re-emits a JSON value in Python's json.dumps style (", " / ": " separators,
// insertion order preserved) so the rendered tool block is byte-identical to
// the template's `| tojson`.
void PyJsonValue(const simdjson::dom::element &v, std::string &out)
{
    switch (v.type()) {
    case simdjson::dom::element_type::OBJECT: {
        out += '{';
        bool first = true;
        for (auto [k, val] : v.get_object()) {
            if (!first) out += ", ";
            first = false;
            PyJsonString(k, out);
            out += ": ";
            PyJsonValue(val, out);
        }
        out += '}';
        break;
    }
    case simdjson::dom::element_type::ARRAY: {
        out += '[';
        bool first = true;
        for (auto e : v.get_array()) {
            if (!first) out += ", ";
            first = false;
            PyJsonValue(e, out);
        }
        out += ']';
        break;
    }
    case simdjson::dom::element_type::STRING: {
        std::string_view s;
        if (v.get_string().get(s) != simdjson::SUCCESS) s = {};
        PyJsonString(s, out);
        break;
    }
    case simdjson::dom::element_type::INT64: {
        int64_t x = 0;
        if (v.get_int64().get(x) != simdjson::SUCCESS) x = 0;
        out += std::to_string(x);
        break;
    }
    case simdjson::dom::element_type::UINT64: {
        uint64_t x = 0;
        if (v.get_uint64().get(x) != simdjson::SUCCESS) x = 0;
        out += std::to_string(x);
        break;
    }
    case simdjson::dom::element_type::DOUBLE: {
        double d = 0.0;
        if (v.get_double().get(d) != simdjson::SUCCESS) d = 0.0;
        char buf[40];
        std::snprintf(buf, sizeof(buf), "%g", d);
        out += buf;
        break;
    }
    case simdjson::dom::element_type::BOOL: {
        bool b = false;
        if (v.get_bool().get(b) != simdjson::SUCCESS) b = false;
        out += b ? "true" : "false";
        break;
    }
    case simdjson::dom::element_type::BIGINT:
        out += simdjson::to_string(v);
        break;
    case simdjson::dom::element_type::NULL_VALUE:
        out += "null";
        break;
    }
}

std::string PyJsonDumps(std::string_view json)
{
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(json).get(doc) != simdjson::SUCCESS) return {};
    std::string out;
    PyJsonValue(doc, out);
    return out;
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
        // Strip the auto-prepended BOS only when it is actually there: the
        // framing supplies <|begin_of_text|> as its own id.
        const size_t begin =
            (tok.bos_id >= 0 && n > 0 && buf[0] == tok.bos_id) ? 1 : 0;
        ids.insert(ids.end(), buf.begin() + begin, buf.begin() + n);
        return true;
    }
    return false;
}

// True when `c` is an ASCII word byte (letter/digit/underscore). The o200k
// pre-tokenizer splits text into words BEFORE matching added tokens, so a
// marker glued to a word ("assistant<|start|>") is NOT a marker — it BPEs as
// literal bytes. Sentinel injection must respect that word boundary.
inline bool IsWordByte(unsigned char c)
{
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c == '_';
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
        // A marker only resolves at a word boundary: the preceding byte must
        // not be a word byte (else the pre-tokenizer keeps it inside a word).
        const bool at_boundary =
            (i == 0) || !IsWordByte(static_cast<unsigned char>(text[i - 1]));
        if (at_boundary) {
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
        }
        if (!matched) ++i;
    }
    return flush(text.size());
}

// One ATEM parameter: strings/scalars raw, bool/null literals, lists/objects
// as Python-tojson JSON (the render_atem macro's type dispatch).
void AppendAtemParam(std::string_view key, const simdjson::dom::element &v,
                     std::string &out)
{
    out += kParamBegin;
    out += std::string(key);
    out += kParamOpen;
    if (v.is_string()) {
        std::string_view s;
        if (v.get_string().get(s) != simdjson::SUCCESS) s = {};
        out += std::string(s);
    } else if (v.is_bool()) {
        bool b = false;
        if (v.get_bool().get(b) != simdjson::SUCCESS) b = false;
        out += b ? "true" : "false";
    } else if (v.is_null()) {
        out += "null";
    } else if (v.is_object() || v.is_array()) {
        PyJsonValue(v, out);
    } else {
        // Numbers: the template renders them via {{ v }} (Python str()).
        out += simdjson::to_string(v);
    }
    out += kParamEnd;
    out += '\n';
}

void AppendToolCallText(const ToolCall &call, std::string &out)
{
    out += kCallsBegin;
    out += '\n';
    out += kInvokeBegin;
    out += call.name;
    out += kInvokeOpen;
    out += '\n';
    simdjson::dom::parser parser;
    simdjson::dom::element args;
    if (!call.arguments_json.empty() &&
        parser.parse(call.arguments_json).get(args) == simdjson::SUCCESS &&
        args.is_object()) {
        for (auto [k, v] : args.get_object()) AppendAtemParam(k, v, out);
    }
    out += kInvokeEnd;
    out += '\n';
    out += kCallsEnd;
}

// The template's render_tool_defs macro, byte-faithful.
void AppendToolDefs(const std::vector<ToolSpec> &tools, std::string &text)
{
    text += "In this environment you have access to a set of tools you can "
            "use to answer the user's question.\n\n";
    text += "You can invoke a function by writing a \"<atem:function_calls>\" "
            "block like the following:\n";
    text += "<atem:function_calls>\n<atem:invoke name=\"$FUNCTION_NAME\">\n"
            "<atem:parameter name=\"$PARAMETER_NAME\">$PARAMETER_VALUE"
            "</atem:parameter>\n...\n</atem:invoke>\n</atem:function_calls>\n\n";
    text += "String and scalar parameters should be specified as is, while "
            "lists and objects should use JSON format. Note that spaces for "
            "string values are not stripped. The output is not expected to be "
            "valid XML and is parsed with regular expressions.\n";
    text += "Here are the functions available in JSONSchema format:\n";
    text += "// Tool metadata\n";
    // Namespaces: the part before the first '.' of each tool name, in order
    // of first appearance.
    std::vector<std::string> namespaces;
    for (const auto &t : tools) {
        const std::string ns = t.name.substr(0, t.name.find('.'));
        bool seen = false;
        for (const auto &n : namespaces) if (n == ns) { seen = true; break; }
        if (!seen) namespaces.push_back(ns);
    }
    for (const auto &ns : namespaces) {
        text += "{\"name\": ";
        PyJsonString(ns, text);
        text += ", \"description\": \"\"}\n";
    }
    text += "// Function schemas";
    for (const auto &t : tools) {
        text += "\n{\"name\": ";
        PyJsonString(t.name, text);
        text += ", \"description\": ";
        PyJsonString(t.description, text);
        text += ", \"parameters\": ";
        std::string params = PyJsonDumps(t.parameters_json);
        if (params.empty()) params = "{}";
        text += params;
        text += "}";
    }
    text += "\n\nHere's an example of how to call a function in the tool set:\n";
    text += "(If the tool namespace is not specified, invoke the function "
            "directly as `example_function_name` rather than "
            "`example_tool_name.example_function_name`)\n\n";
    text += "to=example_tool_name.example_function_name\n\n";
    text += "<atem:function_calls>\n<atem:invoke "
            "name=\"example_tool_name.example_function_name\">\n";
    text += "<atem:parameter name=\"example_parameter_1\">value_1"
            "</atem:parameter>\n";
    text += "<atem:parameter name=\"example_parameter_2\">This is the value "
            "for the second parameter\nthat can span\n\"multiple\" lines\n"
            "</atem:parameter>\n";
    text += "</atem:invoke>\n</atem:function_calls>";
}

// The template's render_system_meta macro: the valid-recipient list.
void AppendSystemMeta(const std::vector<ToolSpec> &tools, std::string &text)
{
    text += "# Valid recipients: \"self\"";
    std::vector<std::string> namespaces;
    for (const auto &t : tools) {
        const std::string ns = t.name.substr(0, t.name.find('.'));
        bool seen = false;
        for (const auto &n : namespaces) if (n == ns) { seen = true; break; }
        if (!seen) namespaces.push_back(ns);
    }
    for (const auto &ns : namespaces) {
        text += ", \"";
        text += ns;
        text += ".*\"";
    }
    text += ", \"user\".";
}

// Case-insensitive substring search (the jinja's `'reasoning strength' not in
// (sys_text | lower)` guard).
bool ContainsLower(std::string_view hay, std::string_view needle)
{
    std::string h(hay);
    for (auto &c : h) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return h.find(needle) != std::string::npos;
}

// The jinja normalises "Reasoning effort" -> "Reasoning strength" (four
// realistic casings) in a caller-supplied system prompt.
std::string NormalizeReasoningEffort(std::string s)
{
    static const std::pair<const char *, const char *> kSubs[] = {
        {"Reasoning effort", "Reasoning strength"},
        {"Reasoning Effort", "Reasoning Strength"},
        {"reasoning effort", "reasoning strength"},
        {"REASONING EFFORT", "REASONING STRENGTH"},
    };
    for (const auto &[from, to] : kSubs) {
        size_t pos = 0;
        const size_t flen = std::strlen(from);
        while ((pos = s.find(from, pos)) != std::string::npos) {
            s.replace(pos, flen, to);
            pos += std::strlen(to);
        }
    }
    return s;
}

} // namespace

bool MuseChatSentinelsPresent(const ArrowTokenizerDb &tok)
{
    return arrow_tokenizer_lookup(tok, kBos, std::strlen(kBos)) >= 0 &&
           arrow_tokenizer_lookup(tok, kStart, std::strlen(kStart)) >= 0 &&
           arrow_tokenizer_lookup(tok, kMessage, std::strlen(kMessage)) >= 0 &&
           arrow_tokenizer_lookup(tok, kEot, std::strlen(kEot)) >= 0 &&
           arrow_tokenizer_lookup(tok, kEom, std::strlen(kEom)) >= 0;
}

std::string RenderMuseToolCallText(const ToolCall &call)
{
    std::string out;
    AppendToolCallText(call, out);
    return out;
}

std::vector<ToolCall> ParseMuseAtemText(const std::string &text)
{
    std::vector<ToolCall> calls;
    size_t pos = 0;
    while (true) {
        const size_t cb = text.find(kCallsBegin, pos);
        if (cb == std::string::npos) break;
        const size_t ce = text.find(kCallsEnd, cb);
        if (ce == std::string::npos) break;
        // One block may hold several <atem:invoke> entries.
        size_t p = cb + std::strlen(kCallsBegin);
        while (true) {
            const size_t ib = text.find(kInvokeBegin, p);
            if (ib == std::string::npos || ib >= ce) break;
            const size_t name_begin = ib + std::strlen(kInvokeBegin);
            const size_t name_end = text.find(kInvokeOpen, name_begin);
            if (name_end == std::string::npos || name_end >= ce) break;
            const size_t ie = text.find(kInvokeEnd, name_end);
            if (ie == std::string::npos || ie > ce) break;
            ToolCall call;
            call.name = std::string(std::string_view(text).substr(
                name_begin, name_end - name_begin));
            std::string args = "{";
            bool first = true;
            size_t q = name_end + std::strlen(kInvokeOpen);
            while (true) {
                const size_t pb = text.find(kParamBegin, q);
                if (pb == std::string::npos || pb >= ie) break;
                const size_t kb = pb + std::strlen(kParamBegin);
                const size_t ke = text.find(kParamOpen, kb);
                if (ke == std::string::npos || ke >= ie) break;
                const size_t vb = ke + std::strlen(kParamOpen);
                const size_t ve = text.find(kParamEnd, vb);
                if (ve == std::string::npos || ve >= ie) break;
                const std::string key(text.data() + kb, ke - kb);
                const std::string_view raw(text.data() + vb, ve - vb);
                if (!first) args += ",";
                first = false;
                args += "\"" + JsonEscape(key) + "\":";
                // A value that is already valid JSON (number/bool/null/object/
                // array) keeps its type; anything else is a string. The
                // template renders strings raw, so a bare word is a string.
                const std::string_view t = TrimAscii(raw);
                simdjson::dom::parser vp;
                simdjson::dom::element vdoc;
                const bool json_ok =
                    !t.empty() && t.front() != '"' &&
                    vp.parse(t).get(vdoc) == simdjson::SUCCESS &&
                    !vdoc.is_string();
                if (json_ok)
                    args += std::string(t);
                else
                    args += "\"" + JsonEscape(std::string(raw)) + "\"";
                q = ve + std::strlen(kParamEnd);
            }
            args += "}";
            call.arguments_json = std::move(args);
            calls.push_back(std::move(call));
            p = ie + std::strlen(kInvokeEnd);
        }
        pos = ce + std::strlen(kCallsEnd);
    }
    for (size_t i = 0; i < calls.size(); ++i)
        if (calls[i].id.empty()) calls[i].id = "call_" + std::to_string(i);
    return calls;
}

bool RenderMuseConversation(const ArrowTokenizerDb &tok,
                            const ChatConversation &conv,
                            std::vector<int32_t> &out_ids)
{
    return RenderMuseConversation(tok, conv, out_ids, MuseRenderPolicy{});
}

bool RenderMuseConversation(const ArrowTokenizerDb &tok,
                            const ChatConversation &conv,
                            std::vector<int32_t> &out_ids,
                            const MuseRenderPolicy &policy)
{
    out_ids.clear();
    if (!MuseChatSentinelsPresent(tok)) return false;

    const bool has_tools = !conv.tools.empty();
    bool has_system = false;
    for (const auto &m : conv.messages)
        if (m.role == "system") { has_system = true; break; }

    std::string text;
    text += kBos;

    // Default system turn when the caller supplies none.
    if (!has_system) {
        text += kStart;
        text += "system";
        text += kMessage;
        text += "You are a helpful AI assistant.\nKnowledge cutoff: ";
        text += policy.knowledge_cutoff;
        text += ".";
        if (!policy.current_date.empty()) {
            text += "\nCurrent date: ";
            text += policy.current_date;
            text += ".";
        }
        text += "\n\nReasoning strength: ";
        text += policy.reasoning_strength.empty() ? "high"
                                                  : policy.reasoning_strength;
        text += ".";
        if (has_tools) {
            text += "\n\n";
            AppendToolDefs(conv.tools, text);
        }
        text += "\n\n";
        AppendSystemMeta(conv.tools, text);
        text += kEot;
    }

    for (size_t mi = 0; mi < conv.messages.size(); ++mi) {
        const ChatMessage &m = conv.messages[mi];
        const bool next_same_role =
            (mi + 1 < conv.messages.size() &&
             conv.messages[mi + 1].role == m.role);
        const std::string end_token = next_same_role ? kEom : kEot;

        if (m.role == "system") {
            std::string sys = NormalizeReasoningEffort(m.content);
            text += kStart;
            text += "system";
            text += kMessage;
            text += sys;
            if (!ContainsLower(sys, "reasoning strength")) {
                text += "\n\nReasoning strength: ";
                text += policy.reasoning_strength.empty()
                            ? "high"
                            : policy.reasoning_strength;
                text += ".";
            }
            if (has_tools) {
                text += "\n\n";
                AppendToolDefs(conv.tools, text);
            }
            text += "\n\n";
            AppendSystemMeta(conv.tools, text);
            text += kEot;
        } else if (m.role == "user") {
            text += kStart;
            text += "user";
            text += kMessage;
            text += m.content;
            text += kEot;
        } else if (m.role == "tool") {
            // Resolve the tool name: explicit `name` field, else the matching
            // tool_call_id in a previous assistant turn.
            std::string tname = m.tool_call_id;
            for (const auto &prev : conv.messages) {
                for (const auto &tc : prev.tool_calls) {
                    if (!m.tool_call_id.empty() && tc.id == m.tool_call_id)
                        tname = tc.name;
                }
            }
            // ChatMessage carries no `name` field; the tool_call_id resolution
            // above is the template's fallback path.
            text += kStart;
            text += "tool ";
            text += tname;
            text += kMessage;
            text += "<tool_output name=\"";
            text += tname;
            text += "\">\n";
            text += m.content;
            text += "\n</tool_output>";
            text += kEot;
        } else if (m.role == "assistant") {
            if (!m.tool_calls.empty()) {
                for (size_t ci = 0; ci < m.tool_calls.size(); ++ci) {
                    const auto &tc = m.tool_calls[ci];
                    text += kStart;
                    text += "assistant to=";
                    text += tc.name;
                    text += kMessage;
                    AppendToolCallText(tc, text);
                    text += (ci + 1 == m.tool_calls.size()) ? end_token : kEom;
                }
            } else {
                text += kStart;
                text += "assistant to=user";
                text += kMessage;
                text += m.content;
                text += end_token;
            }
        }
    }

    // Generation prompt: <|start|>assistant (no recipient, no message marker).
    text += kStart;
    text += "assistant";
    return AppendTextWithSentinels(tok, text, out_ids);
}

bool MusePromptInputIds(const ArrowTokenizerDb &tok, std::string_view raw_prompt,
                        std::vector<int32_t> &out_ids)
{
    ChatConversation conv;
    conv.messages.push_back({"user", std::string(raw_prompt), "", {}});
    return RenderMuseConversation(tok, conv, out_ids);
}

bool ParseMuseToolCalls(const ArrowTokenizerDb &tok,
                        const std::vector<int32_t> &gen_tokens,
                        std::vector<ToolCall> &out_calls)
{
    out_calls.clear();
    if (gen_tokens.empty()) return true;

    const int32_t start_id = arrow_tokenizer_lookup(tok, kStart, std::strlen(kStart));
    const int32_t message_id =
        arrow_tokenizer_lookup(tok, kMessage, std::strlen(kMessage));
    const int32_t eot_id = arrow_tokenizer_lookup(tok, kEot, std::strlen(kEot));
    const int32_t eom_id = arrow_tokenizer_lookup(tok, kEom, std::strlen(kEom));

    // The channel markers are special:true (decode drops them), so scan ids:
    // a tool channel is <|start|> "assistant to=<fn>" <|message|> … where <fn>
    // is neither "user" nor "self". Collect each channel's body text.
    std::string native_text;
    bool found_channel = false;
    if (start_id >= 0 && message_id >= 0) {
        size_t i = 0;
        while (i < gen_tokens.size()) {
            if (gen_tokens[i] != start_id) { ++i; continue; }
            // Read the header text up to <|message|>.
            size_t j = i + 1;
            std::vector<int32_t> header;
            while (j < gen_tokens.size() && gen_tokens[j] != message_id &&
                   gen_tokens[j] != start_id) {
                header.push_back(gen_tokens[j]);
                ++j;
            }
            if (j >= gen_tokens.size() || gen_tokens[j] != message_id) {
                i = j;
                continue;
            }
            std::string htext;
            if (!header.empty()) {
                std::vector<char> buf(header.size() * 16 + 64);
                size_t cap = buf.size();
                if (arrow_tokenizer_decode(tok, header.data(), header.size(),
                                           buf.data(), &cap) == 0)
                    htext.assign(buf.data(), cap);
            }
            const std::string_view hv(htext);
            const std::string_view prefix = "assistant to=";
            const bool is_tool_channel =
                hv.rfind(prefix, 0) == 0 &&
                hv.substr(prefix.size()) != "user" &&
                hv.substr(prefix.size()) != "self";
            // Body runs to the next channel marker / turn end.
            size_t body_begin = j + 1;
            size_t body_end = body_begin;
            while (body_end < gen_tokens.size() &&
                   gen_tokens[body_end] != start_id &&
                   gen_tokens[body_end] != eot_id &&
                   gen_tokens[body_end] != eom_id)
                ++body_end;
            if (is_tool_channel && body_end > body_begin) {
                std::vector<char> buf((body_end - body_begin) * 16 + 64);
                size_t cap = buf.size();
                if (arrow_tokenizer_decode(tok, gen_tokens.data() + body_begin,
                                           body_end - body_begin, buf.data(),
                                           &cap) == 0) {
                    native_text.append(buf.data(), cap);
                    native_text += "\n";
                    found_channel = true;
                }
            }
            i = body_end;
        }
    }
    if (found_channel) {
        auto calls = ParseMuseAtemText(native_text);
        if (!calls.empty()) {
            out_calls = std::move(calls);
            return true;
        }
    }

    // Fallback: decode the whole tail and run the generic parsers.
    std::string decoded;
    {
        std::vector<char> buf(gen_tokens.size() * 16 + 64);
        size_t cap = buf.size();
        if (arrow_tokenizer_decode(tok, gen_tokens.data(), gen_tokens.size(),
                                   buf.data(), &cap) != 0)
            return false;
        decoded.assign(buf.data(), cap);
    }
    auto calls = ParseMuseAtemText(decoded);
    if (!calls.empty()) {
        out_calls = std::move(calls);
        return true;
    }
    out_calls = tools::ParseToolCallsText(decoded);
    if (!out_calls.empty()) return true;
    out_calls = tools::ParseBareToolCallJson(decoded);
    // A plain answer (no tool call) is a SUCCESSFUL parse with zero calls —
    // the round-trip helper and the serve path both treat false as an error.
    return true;
}

} // namespace lsxcommon::muse