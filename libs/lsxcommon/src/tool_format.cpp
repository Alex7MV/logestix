#include "lsxcommon/tool_format.h"

#include <simdjson.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace lsxcommon {
namespace tools {
namespace {

std::string JsonEscape(const std::string& in) {
    std::string out;
    out.reserve(in.size() + 8);
    static const char* kHex = "0123456789abcdef";
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

// Minimal recursive serializer for a simdjson DOM element (compact JSON).
std::string Serialize(simdjson::dom::element e);

std::string Serialize(simdjson::dom::element e) {
    using simdjson::dom::element_type;
    switch (e.type()) {
        case element_type::OBJECT: {
            std::string out = "{";
            bool first = true;
            for (auto [k, v] : e.get_object()) {
                if (!first) out += ",";
                first = false;
                out += "\"" + JsonEscape(std::string(k)) + "\":" + Serialize(v);
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
                out += Serialize(v);
            }
            out += "]";
            return out;
        }
        case element_type::STRING:
            return "\"" + JsonEscape(std::string(e.get_string().value_unsafe())) + "\"";
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

}  // namespace

std::string RenderToolsBlock(const std::vector<ToolSpec>& tools) {
    if (tools.empty()) return {};
    std::string out = "<tools>\n[";
    for (std::size_t i = 0; i < tools.size(); ++i) {
        const ToolSpec& t = tools[i];
        if (i) out += ",";
        out += "{\"type\":\"function\",\"function\":{\"name\":\"" +
               JsonEscape(t.name) + "\",\"description\":\"" +
               JsonEscape(t.description) + "\",\"parameters\":" +
               (t.parameters_json.empty() ? std::string("{}")
                                          : t.parameters_json) +
               "}}";
    }
    out += "]\n</tools>";
    return out;
}

std::string RenderConversation(const ChatConversation& conv, bool include_tools) {
    std::string out;
    const std::string tools_block =
        include_tools ? RenderToolsBlock(conv.tools) : std::string{};
    if (!tools_block.empty()) out += tools_block + "\n";
    for (const ChatMessage& m : conv.messages) {
        if (!out.empty()) out += "\n";
        out += m.role + ": " + m.content;
        for (const ToolCall& c : m.tool_calls)
            out += "\n<tool_call>{\"name\":\"" + JsonEscape(c.name) +
                   "\",\"arguments\":" +
                   (c.arguments_json.empty() ? std::string("{}")
                                             : c.arguments_json) +
                   "}</tool_call>";
    }
    return out;
}

std::vector<ToolCall> ParseToolCallsText(std::string_view text) {
    std::vector<ToolCall> calls;
    constexpr std::string_view kOpen = "<tool_call>";
    constexpr std::string_view kClose = "</tool_call>";
    std::size_t pos = 0;
    while (true) {
        const std::size_t b = text.find(kOpen, pos);
        if (b == std::string_view::npos) break;
        const std::size_t e = text.find(kClose, b + kOpen.size());
        if (e == std::string_view::npos) break;
        std::string_view inner =
            text.substr(b + kOpen.size(), e - (b + kOpen.size()));
        pos = e + kClose.size();
        while (!inner.empty() &&
               (inner.front() == '\n' || inner.front() == ' ' ||
                inner.front() == '\t' || inner.front() == '\r'))
            inner.remove_prefix(1);
        while (!inner.empty() &&
               (inner.back() == '\n' || inner.back() == ' ' ||
                inner.back() == '\t' || inner.back() == '\r'))
            inner.remove_suffix(1);
        if (inner.empty()) continue;

        simdjson::dom::parser parser;
        simdjson::dom::element obj;
        simdjson::padded_string padded(inner);
        if (parser.parse(padded).get(obj) != simdjson::SUCCESS) continue;
        if (obj.type() != simdjson::dom::element_type::OBJECT) continue;

        ToolCall call;
        if (auto n = obj["name"].get_string(); !n.error())
            call.name.assign(n.value_unsafe());
        if (call.name.empty()) continue;

        simdjson::dom::element args;
        if (obj["arguments"].get(args) == simdjson::SUCCESS) {
            if (args.type() == simdjson::dom::element_type::STRING)
                call.arguments_json.assign(args.get_string().value_unsafe());
            else
                call.arguments_json = Serialize(args);
        }
        if (call.arguments_json.empty()) call.arguments_json = "{}";
        call.id = "call_" + std::to_string(calls.size());
        calls.push_back(std::move(call));
    }
    return calls;
}

}  // namespace tools
}  // namespace lsxcommon
