#include "lsxcommon/kimi_moe_k3_tool_format.h"

#include "lsxcommon/tool_format.h"

#include <lsxnetwork/json_escape.h>

#include <simdjson.h>

#include <algorithm>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace lsxcommon {
namespace {

using lsxnetwork::json::JsonEscape;

constexpr const char *kOpen = "<|open|>";
constexpr const char *kClose = "<|close|>";
constexpr const char *kSep = "<|sep|>";
constexpr const char *kEndOfMsg = "<|end_of_msg|>";

// encoding_k3.py `_escape_attr_value`: & first, then ".
std::string EscapeAttr(std::string_view v)
{
    std::string out;
    out.reserve(v.size());
    for (char c : v) {
        if (c == '&') out += "&amp;";
        else if (c == '"') out += "&quot;";
        else out += c;
    }
    return out;
}

std::string UnescapeAttr(std::string_view v)
{
    std::string out;
    out.reserve(v.size());
    for (size_t i = 0; i < v.size();) {
        if (v.compare(i, 6, "&quot;") == 0) {
            out += '"';
            i += 6;
        } else if (v.compare(i, 5, "&amp;") == 0) {
            out += '&';
            i += 5;
        } else {
            out += v[i++];
        }
    }
    return out;
}

void AddText(std::vector<KimiK3Segment> &segs, std::string text)
{
    if (text.empty()) return;
    segs.push_back({std::move(text), false});
}

void AddControl(std::vector<KimiK3Segment> &segs, std::string text)
{
    segs.push_back({std::move(text), true});
}

// _open_tag(tag, attrs)
void OpenTag(std::vector<KimiK3Segment> &segs, const std::string &tag,
             const std::vector<std::pair<std::string, std::string>> &attrs = {})
{
    AddControl(segs, kOpen);
    AddText(segs, tag);
    for (const auto &[k, v] : attrs) {
        AddText(segs, " " + k);
        AddText(segs, "=\"");
        AddText(segs, EscapeAttr(v));
        AddText(segs, "\"");
    }
    AddControl(segs, kSep);
}

void CloseTag(std::vector<KimiK3Segment> &segs, const std::string &tag)
{
    AddControl(segs, kClose);
    AddText(segs, tag);
    AddControl(segs, kSep);
}

void EndOfMsg(std::vector<KimiK3Segment> &segs) { AddControl(segs, kEndOfMsg); }

std::string XtmlType(std::string_view type)
{
    if (type == "string" || type == "number" || type == "boolean" ||
        type == "null" || type == "object" || type == "array")
        return std::string(type);
    return "string";
}

// Minimal JSON argument splitter for tool-call rendering: one level deep so
// non-string values keep their original literal bytes (encoding_k3.py's
// _parse_arguments_object). Returns false for anything that is not a JSON
// object; the caller then emits the raw <json> block.
struct XtmlArg {
    std::string key;
    std::string type;
    std::string text;
};

bool ParseArgumentsObject(std::string_view s, std::vector<XtmlArg> &out)
{
    // Faithful port of encoding_k3.py `_parse_arguments_object`: one level
    // deep, keys decoded, non-string values keep their original literal bytes,
    // anything after the closing brace is ignored.
    size_t i = 0;
    auto skip_ws = [&] {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' ||
                                s[i] == '\r'))
            ++i;
    };
    // Scans one JSON value starting at i; returns its [begin, end) or false.
    auto scan_value = [&](size_t &end) -> int {
        const size_t b = i;
        if (i >= s.size()) return -1;  // -1 = error, else the type code
        const char c = s[i];
        if (c == '"') {
            ++i;
            bool esc = false;
            while (i < s.size()) {
                if (esc) { esc = false; ++i; continue; }
                if (s[i] == '\\') { esc = true; ++i; continue; }
                if (s[i] == '"') { ++i; end = i; return 0; }
                ++i;
            }
            return -1;
        }
        if (c == '{' || c == '[') {
            int depth = 0;
            bool in_str = false, esc = false;
            while (i < s.size()) {
                const char d = s[i];
                if (esc) { esc = false; ++i; continue; }
                if (in_str) {
                    if (d == '\\') esc = true;
                    else if (d == '"') in_str = false;
                    ++i;
                    continue;
                }
                if (d == '"') { in_str = true; ++i; continue; }
                if (d == '{' || d == '[') ++depth;
                else if (d == '}' || d == ']') {
                    --depth;
                    if (depth == 0) { ++i; end = i; return c == '{' ? 2 : 3; }
                }
                ++i;
            }
            return -1;
        }
        // number / true / false / null
        while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']' &&
               s[i] != ' ' && s[i] != '\t' && s[i] != '\n' && s[i] != '\r')
            ++i;
        end = i;
        if (end == b) return -1;
        const std::string_view lit = s.substr(b, end - b);
        if (lit == "true" || lit == "false") return 4;
        if (lit == "null") return 5;
        return 1;  // number
    };

    skip_ws();
    if (i >= s.size() || s[i] != '{') return false;
    ++i;
    skip_ws();
    if (i < s.size() && s[i] == '}') return true;

    for (;;) {
        skip_ws();
        if (i >= s.size() || s[i] != '"') return false;
        const size_t key_begin = i;
        size_t key_end = 0;
        const size_t save = i;
        i = key_begin;
        int t = scan_value(key_end);
        if (t != 0) return false;
        std::string key_json(s.substr(key_begin, key_end - key_begin));
        std::string key;
        {
            simdjson::dom::parser parser;
            simdjson::dom::element el;
            if (parser.parse(simdjson::padded_string(key_json)).get(el) !=
                simdjson::SUCCESS)
                return false;
            std::string_view sv;
            if (el.get_string().get(sv) != simdjson::SUCCESS) return false;
            key = std::string(sv);
        }
        (void)save;
        skip_ws();
        if (i >= s.size() || s[i] != ':') return false;
        ++i;
        skip_ws();
        const size_t value_begin = i;
        size_t value_end = 0;
        const int vt = scan_value(value_end);
        if (vt < 0) return false;
        XtmlArg a;
        a.key = key;
        const std::string_view lit =
            s.substr(value_begin, value_end - value_begin);
        switch (vt) {
        case 0: {
            std::string lit_json(lit);
            simdjson::dom::parser parser;
            simdjson::dom::element el;
            if (parser.parse(simdjson::padded_string(lit_json)).get(el) !=
                simdjson::SUCCESS)
                return false;
            std::string_view sv;
            if (el.get_string().get(sv) != simdjson::SUCCESS) return false;
            a.type = "string";
            a.text = std::string(sv);
            break;
        }
        case 1:
            a.type = "number";
            a.text = std::string(lit);
            break;
        case 2:
            a.type = "object";
            a.text = std::string(lit);
            break;
        case 3:
            a.type = "array";
            a.text = std::string(lit);
            break;
        case 4:
            a.type = "boolean";
            a.text = std::string(lit);
            break;
        default:
            a.type = "null";
            a.text = "null";
            break;
        }
        out.push_back(std::move(a));

        skip_ws();
        if (i >= s.size()) return false;
        const char sep = s[i++];
        skip_ws();
        if (sep == '}') break;
        if (sep != ',') return false;
    }
    return true;
}

void RenderToolsPreamble(const std::vector<ToolSpec> &tools,
                         std::vector<KimiK3Segment> &segs)
{
    // OpenAI tools array with deep-sorted keys, compact separators — the same
    // bytes encoding_k3.py renders for the same input.
    std::string json = "[";
    for (size_t i = 0; i < tools.size(); ++i) {
        if (i) json += ',';
        std::string params;
        if (!tools[i].parameters_json.empty() &&
            KimiK3CanonicalJson(tools[i].parameters_json, params)) {
            // canonicalized above
        } else {
            params = tools[i].parameters_json.empty() ? "{}"
                                                      : tools[i].parameters_json;
        }
        json += "{\"function\":{\"description\":\"";
        json += JsonEscape(tools[i].description);
        json += "\",\"name\":\"";
        json += JsonEscape(tools[i].name);
        json += "\",\"parameters\":";
        json += params;
        json += "},\"type\":\"function\"}";
    }
    json += ']';

    OpenTag(segs, "message", {{"role", "system"}, {"type", "tool-declare"}});
    AddText(segs,
            "# Tools\nHere are the available tools, described in JSONSchema."
            "\n\n```json\n" + json + "\n```");
    CloseTag(segs, "message");
    EndOfMsg(segs);
}

void RenderContent(std::vector<KimiK3Segment> &segs, const std::string &text)
{
    AddText(segs, text);
}

void RenderAssistant(const ChatMessage &msg,
                     std::vector<KimiK3Segment> &segs)
{
    // Thinking mode is the checkpoint default: the channel is structural and
    // always present (empty when the caller supplied no reasoning).
    OpenTag(segs, "think");
    CloseTag(segs, "think");
    OpenTag(segs, "response");
    RenderContent(segs, msg.content);
    CloseTag(segs, "response");

    if (!msg.tool_calls.empty()) {
        OpenTag(segs, "tools");
        int index = 1;
        for (const ToolCall &call : msg.tool_calls) {
            OpenTag(segs, "call",
                    {{"tool", call.name}, {"index", std::to_string(index)}});
            std::vector<XtmlArg> args;
            if (!call.arguments_json.empty() &&
                ParseArgumentsObject(call.arguments_json, args)) {
                for (const XtmlArg &a : args) {
                    OpenTag(segs, "argument",
                            {{"key", a.key}, {"type", a.type}});
                    AddText(segs, a.text);
                    CloseTag(segs, "argument");
                }
            } else if (!call.arguments_json.empty()) {
                OpenTag(segs, "json", {{"type", "object"}});
                AddText(segs, call.arguments_json);
                CloseTag(segs, "json");
            }
            CloseTag(segs, "call");
            ++index;
        }
        CloseTag(segs, "tools");
    }
}

// Resolves the tool name for a role=tool message: the explicit name, else the
// name of the matching call by position (encoding_k3.py fallback).
std::string ToolNameFor(const ChatMessage &msg,
                        const std::vector<ToolCall> *prev_calls, int index)
{
    if (!msg.tool_call_id.empty() && prev_calls) {
        for (const ToolCall &c : *prev_calls)
            if (!c.id.empty() && c.id == msg.tool_call_id) return c.name;
    }
    if (prev_calls && index >= 1 &&
        static_cast<size_t>(index) <= prev_calls->size())
        return (*prev_calls)[index - 1].name;
    return {};
}

}  // namespace

bool KimiK3CanonicalJson(std::string_view json, std::string &out)
{
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(simdjson::padded_string(json)).get(doc) != simdjson::SUCCESS)
        return false;

    std::string s;
    // Explicit recursion via a small stack lambda.
    struct Emit {
        static bool Run(simdjson::dom::element v, std::string &s)
        {
            switch (v.type()) {
            case simdjson::dom::element_type::OBJECT: {
                simdjson::dom::object obj;
                if (v.get_object().get(obj) != simdjson::SUCCESS) return false;
                std::map<std::string, simdjson::dom::element> sorted;
                for (auto field : obj)
                    sorted.emplace(std::string(field.key), field.value);
                s += '{';
                bool first = true;
                for (auto &[k, val] : sorted) {
                    if (!first) s += ',';
                    first = false;
                    s += '"';
                    s += JsonEscape(k);
                    s += "\":";
                    if (!Run(val, s)) return false;
                }
                s += '}';
                return true;
            }
            case simdjson::dom::element_type::ARRAY: {
                simdjson::dom::array arr;
                if (v.get_array().get(arr) != simdjson::SUCCESS) return false;
                s += '[';
                bool first = true;
                for (auto item : arr) {
                    if (!first) s += ',';
                    first = false;
                    if (!Run(item, s)) return false;
                }
                s += ']';
                return true;
            }
            case simdjson::dom::element_type::STRING: {
                std::string_view sv;
                if (v.get_string().get(sv) != simdjson::SUCCESS) return false;
                s += '"';
                s += JsonEscape(sv);
                s += '"';
                return true;
            }
            case simdjson::dom::element_type::INT64: {
                int64_t x;
                const auto i64_err = v.get_int64().get(x);
                (void)i64_err;
                s += std::to_string(x);
                return true;
            }
            case simdjson::dom::element_type::UINT64: {
                uint64_t x;
                const auto u64_err = v.get_uint64().get(x);
                (void)u64_err;
                s += std::to_string(x);
                return true;
            }
            case simdjson::dom::element_type::DOUBLE: {
                double x;
                const auto f64_err = v.get_double().get(x);
                (void)f64_err;
                char buf[64];
                std::snprintf(buf, sizeof(buf), "%.17g", x);
                s += buf;
                return true;
            }
            case simdjson::dom::element_type::BOOL: {
                bool b = false;
                const auto bool_err = v.get_bool().get(b);
                (void)bool_err;
                s += b ? "true" : "false";
                return true;
            }
            case simdjson::dom::element_type::NULL_VALUE:
                s += "null";
                return true;
            default:
                return false;
            }
        }
    };
    if (!Emit::Run(doc, s)) return false;
    out = std::move(s);
    return true;
}

bool RenderKimiK3Conversation(const ChatConversation &conv,
                              std::vector<KimiK3Segment> &out)
{
    out.clear();
    if (!conv.tools.empty()) RenderToolsPreamble(conv.tools, out);

    const std::vector<ToolCall> *prev_calls = nullptr;
    int tool_index = 0;
    for (const ChatMessage &m : conv.messages) {
        if (m.role == "user") {
            OpenTag(out, "message", {{"role", "user"}});
            RenderContent(out, m.content);
            CloseTag(out, "message");
            EndOfMsg(out);
            tool_index = 0;
        } else if (m.role == "system") {
            OpenTag(out, "message", {{"role", "system"}});
            RenderContent(out, m.content);
            CloseTag(out, "message");
            EndOfMsg(out);
        } else if (m.role == "tool") {
            ++tool_index;
            const std::string name = ToolNameFor(m, prev_calls, tool_index);
            if (name.empty()) return false;
            OpenTag(out, "message",
                    {{"role", "tool"},
                     {"tool", name},
                     {"index", std::to_string(tool_index)}});
            RenderContent(out, m.content);
            CloseTag(out, "message");
            EndOfMsg(out);
        } else if (m.role == "assistant") {
            OpenTag(out, "message", {{"role", "assistant"}});
            RenderAssistant(m, out);
            CloseTag(out, "message");
            EndOfMsg(out);
            prev_calls = &m.tool_calls;
            tool_index = 0;
        } else {
            return false;
        }
    }

    OpenTag(out, "message", {{"role", "assistant"}});
    OpenTag(out, "think");
    return true;
}

std::string KimiK3SegmentsText(const std::vector<KimiK3Segment> &segments)
{
    std::string out;
    for (const auto &s : segments) out += s.text;
    return out;
}

bool AppendKimiK3Ids(const ArrowTokenizerDb &tok,
                     const std::vector<KimiK3Segment> &segments,
                     std::vector<int32_t> &out_ids)
{
    for (const auto &seg : segments) {
        if (seg.text.empty()) continue;
        if (seg.is_control) {
            const int32_t id =
                arrow_tokenizer_lookup(tok, seg.text.data(), seg.text.size());
            if (id < 0) return false;
            out_ids.push_back(id);
            continue;
        }
        std::vector<int32_t> ids(seg.text.size() * 4 + 64, 0);
        size_t cap = ids.size();
        const int32_t n = arrow_tokenizer_encode(tok, seg.text.data(),
                                                 seg.text.size(), ids.data(),
                                                 &cap);
        if (n < 0) return false;
        ids.resize(static_cast<size_t>(n));
        out_ids.insert(out_ids.end(), ids.begin(), ids.end());
    }
    return true;
}

std::vector<ToolCall> ParseKimiK3ToolCalls(std::string_view text)
{
    std::vector<ToolCall> calls;
    const std::string open = kOpen;
    const std::string close = kClose;
    const std::string sep = kSep;

    // Locate "<|open|>tools<|sep|>".
    const std::string tools_open = open + "tools" + sep;
    size_t pos = text.find(tools_open);
    if (pos == std::string_view::npos) {
        auto generic = tools::ParseToolCallsText(std::string(text));
        if (!generic.empty()) return generic;
        return tools::ParseBareToolCallJson(std::string(text));
    }
    pos += tools_open.size();
    const std::string tools_close = close + "tools" + sep;
    const std::string call_open = open + "call";
    const std::string call_close = close + "call" + sep;

    auto read_attr = [](std::string_view head, std::string_view name,
                        std::string &out) {
        const std::string key = " " + std::string(name) + "=\"";
        const size_t p = head.find(key);
        if (p == std::string_view::npos) return false;
        const size_t q = head.find('"', p + key.size());
        if (q == std::string_view::npos) return false;
        out = UnescapeAttr(head.substr(p + key.size(), q - p - key.size()));
        return true;
    };

    while (pos < text.size()) {
        if (text.compare(pos, tools_close.size(), tools_close) == 0) break;
        if (text.compare(pos, call_open.size(), call_open) != 0) {
            ++pos;  // tolerate stray text between calls
            continue;
        }
        size_t head_end = text.find(sep, pos);
        if (head_end == std::string_view::npos) break;
        const std::string_view head = text.substr(pos, head_end - pos);
        ToolCall call;
        if (!read_attr(head, "tool", call.name)) return {};
        pos = head_end + sep.size();

        std::string args;
        bool first_arg = true;
        bool raw_json_args = false;
        const std::string arg_open = open + "argument";
        const std::string arg_close = close + "argument" + sep;
        const std::string json_open = open + "json";
        const std::string json_close = close + "json" + sep;
        while (pos < text.size()) {
            if (text.compare(pos, call_close.size(), call_close) == 0) {
                pos += call_close.size();
                break;
            }
            if (text.compare(pos, arg_open.size(), arg_open) == 0) {
                const size_t he = text.find(sep, pos);
                if (he == std::string_view::npos) return {};
                const std::string_view ah = text.substr(pos, he - pos);
                std::string key, type;
                if (!read_attr(ah, "key", key)) return {};
                read_attr(ah, "type", type);
                const size_t vbeg = he + sep.size();
                const size_t vend = text.find(arg_close, vbeg);
                if (vend == std::string_view::npos) return {};
                const std::string_view value = text.substr(vbeg, vend - vbeg);
                if (!first_arg) args += ',';
                first_arg = false;
                args += '"';
                args += JsonEscape(key);
                args += "\":";
                if (XtmlType(type) == "string") {
                    args += '"';
                    args += JsonEscape(value);
                    args += '"';
                } else {
                    args.append(value.data(), value.size());
                }
                pos = vend + arg_close.size();
                continue;
            }
            if (text.compare(pos, json_open.size(), json_open) == 0) {
                const size_t he = text.find(sep, pos);
                if (he == std::string_view::npos) return {};
                const size_t vbeg = he + sep.size();
                const size_t vend = text.find(json_close, vbeg);
                if (vend == std::string_view::npos) return {};
                args.assign(text.data() + vbeg, vend - vbeg);
                first_arg = false;
                raw_json_args = true;
                pos = vend + json_close.size();
                continue;
            }
            ++pos;
        }
        if (raw_json_args)
            call.arguments_json = args.empty() ? "{}" : args;
        else
            call.arguments_json = args.empty() ? "{}" : "{" + args + "}";
        calls.push_back(std::move(call));
    }
    return calls;
}

}  // namespace lsxcommon
