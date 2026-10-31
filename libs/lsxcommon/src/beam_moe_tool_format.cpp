#include "lsxcommon/beam_moe_tool_format.h"

#include "beam_moe_constants.h"

#include "lsxcommon/log.h"
#include "lsxcommon/tokenizer.h"

#include <absl/strings/str_cat.h>
#include <simdjson.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace lsxcommon::beam {
namespace {

const char kIdentity[] = "You are Beam, an AI assistant trained by Reflection.";
const char kValidChannels[] =
    "\n\n# Valid channels: analysis, final. Channel must be included for every "
    "message.";

bool EncodeText(const ArrowTokenizerDb &tok, std::string_view text,
                std::vector<int32_t> &out_ids)
{
    std::vector<int32_t> ids(text.size() * 4 + 16);
    size_t cap = ids.size();
    const int32_t n = arrow_tokenizer_encode(tok, text.data(), text.size(),
                                             ids.data(), &cap);
    if (n < 0) return false;
    ids.resize(static_cast<size_t>(n));
    out_ids.insert(out_ids.end(), ids.begin(), ids.end());
    return true;
}

std::string Join(const std::vector<std::string> &v, const char *sep)
{
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) s += sep;
        s += v[i];
    }
    return s;
}

void AppendEscapedJsonString(std::string &out, std::string_view s)
{
    out.push_back('"');
    for (char c : s) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default: out.push_back(c); break;
        }
    }
    out.push_back('"');
}

// A parameter body is raw text for strings and JSON for everything else (the
// template's render_parameter_value). On the way back in, a body that parses as
// JSON is kept verbatim; anything else is a JSON string.
std::string ParameterToJson(std::string_view text)
{
    simdjson::dom::parser parser;
    simdjson::dom::element e;
    if (parser.parse(text).get(e) == simdjson::SUCCESS) {
        simdjson::dom::element_type t = e.type();
        if (t == simdjson::dom::element_type::OBJECT ||
            t == simdjson::dom::element_type::ARRAY ||
            t == simdjson::dom::element_type::INT64 ||
            t == simdjson::dom::element_type::UINT64 ||
            t == simdjson::dom::element_type::DOUBLE ||
            t == simdjson::dom::element_type::BOOL) {
            return std::string(text);
        }
    }
    std::string out;
    AppendEscapedJsonString(out, text);
    return out;
}

std::string SchemaToTs(simdjson::dom::element schema, int indent);

std::string PropertiesToTs(simdjson::dom::element schema,
                           simdjson::dom::object props, int indent)
{
    const std::string pad(static_cast<size_t>(indent), ' ');
    std::string out = "{\n";
    bool first = true;
    for (auto [key, val] : props) {
        if (!first) out += "\n";
        first = false;
        std::string comment;
        std::string_view desc;
        if (val["description"].get_string().get(desc) == simdjson::SUCCESS &&
            !desc.empty()) {
            std::string_view rest = desc;
            while (!rest.empty()) {
                const size_t nl = rest.find('\n');
                const std::string_view line =
                    nl == std::string_view::npos ? rest : rest.substr(0, nl);
                comment += pad + "    // " + std::string(line) + "\n";
                if (nl == std::string_view::npos) break;
                rest = rest.substr(nl + 1);
            }
        }
        out += comment;
        bool required = false;
        if (auto req = schema["required"].get_array(); req.error() == simdjson::SUCCESS)
            for (auto r : req.value()) {
                std::string_view rs;
                if (r.get_string().get(rs) == simdjson::SUCCESS && rs == key)
                    required = true;
            }
        out += pad + "    " + std::string(key) +
               (required ? ": " : "?: ") + SchemaToTs(val, indent + 4);
    }
    out += "\n" + pad + "}";
    return out;
}

std::string SchemaToTs(simdjson::dom::element schema, int indent)
{
    if (schema.is_null()) return "any";
    std::string_view type;
    if (schema["type"].get_string().get(type) != simdjson::SUCCESS) {
        if (auto one = schema["oneOf"].get_array();
            one.error() == simdjson::SUCCESS) {
            std::string out;
            bool first = true;
            for (auto v : one.value()) {
                if (!first) out += " | ";
                first = false;
                out += SchemaToTs(v, indent);
            }
            return out.empty() ? "any" : out;
        }
        return "any";
    }
    if (type == "string") {
        if (auto en = schema["enum"].get_array(); en.error() == simdjson::SUCCESS) {
            std::string out;
            bool first = true;
            for (auto v : en.value()) {
                std::string_view s;
                if (v.get_string().get(s) != simdjson::SUCCESS) continue;
                if (!first) out += " | ";
                first = false;
                out += "\"" + std::string(s) + "\"";
            }
            if (!first) return out;
        }
        return "string";
    }
    if (type == "integer") return "number";
    if (type == "number") return "number";
    if (type == "boolean") return "boolean";
    if (type == "object") {
        simdjson::dom::object props;
        if (schema["properties"].get_object().get(props) != simdjson::SUCCESS)
            return "any";
        return PropertiesToTs(schema, props, indent);
    }
    if (type == "array") {
        simdjson::dom::element items;
        if (schema["items"].get(items) != simdjson::SUCCESS) return "any[]";
        return "Array<" + SchemaToTs(items, indent) + ">";
    }
    return "any";
}

std::string RenderToolDeclaration(const ToolSpec &tool)
{
    simdjson::dom::parser parser;
    simdjson::dom::element schema;
    const bool parsed =
        parser.parse(tool.parameters_json).get(schema) == simdjson::SUCCESS;
    std::string_view type;
    simdjson::dom::object props;
    const bool is_string_tool =
        parsed &&
        schema["type"].get_string().get(type) == simdjson::SUCCESS &&
        type == "string" &&
        schema["properties"].get_object().get(props) != simdjson::SUCCESS;
    if (is_string_tool) {
        return "## " + tool.name + "\n\n" + tool.description;
    }
    std::string out;
    if (!tool.description.empty()) {
        std::string_view rest(tool.description);
        while (true) {
            const size_t nl = rest.find('\n');
            const std::string_view line =
                nl == std::string_view::npos ? rest : rest.substr(0, nl);
            out += "// " + std::string(line) + "\n";
            if (nl == std::string_view::npos) break;
            rest = rest.substr(nl + 1);
        }
    }
    const std::string ts = parsed ? SchemaToTs(schema, 0) : "any";
    out += "type " + tool.name + " = (_: " + ts + ") => any;";
    return out;
}

std::string RenderToolsSection(const std::vector<ToolSpec> &tools)
{
    std::vector<std::string> decls;
    for (const auto &t : tools) decls.push_back(RenderToolDeclaration(t));
    return "# Tools\n\n" + Join(decls, "\n\n");
}

std::string RenderParameters(const ToolCall &call)
{
    std::string out;
    simdjson::dom::parser parser;
    simdjson::dom::object args;
    if (parser.parse(call.arguments_json).get(args) != simdjson::SUCCESS)
        return out;
    bool first = true;
    for (auto [key, val] : args) {
        std::string value;
        switch (val.type()) {
        case simdjson::dom::element_type::STRING: {
            std::string_view s;
            val.get_string().get(s);
            value.assign(s);
            break;
        }
        default:
            value = simdjson::to_string(val);
            break;
        }
        if (!first) out += "\n";
        first = false;
        out += "<rfl:parameter name=\"" + std::string(key) + "\">\n" + value +
               "\n</rfl:parameter>";
    }
    return out;
}

std::string ToolResultName(const ChatConversation &conv,
                           const ChatMessage &msg)
{
    if (!msg.tool_call_id.empty()) {
        for (const auto &m : conv.messages)
            for (const auto &c : m.tool_calls)
                if (c.id == msg.tool_call_id) return c.name;
    }
    return "tool";
}

} // namespace

bool RenderBeamConversation(const ArrowTokenizerDb &tok,
                            const ChatConversation &conv,
                            std::vector<int32_t> &out_ids)
{
    if (conv.messages.empty()) return false;
    out_ids.clear();
    std::string text;

    // System block.
    text += "<|start|>system<|message|>";
    text += kIdentity;
    text += "\n\nReasoning effort: 512";
    text += kValidChannels;
    text += "<|end|>";

    // Leading system/developer run joins into the developer instructions.
    size_t first = 0;
    std::vector<std::string> instructions;
    while (first < conv.messages.size() &&
           (conv.messages[first].role == "system" ||
            conv.messages[first].role == "developer")) {
        if (!conv.messages[first].content.empty())
            instructions.push_back(conv.messages[first].content);
        ++first;
    }
    if (!instructions.empty() || !conv.tools.empty()) {
        text += "<|start|>developer<|message|>";
        if (!instructions.empty()) text += "# Instructions\n\n" + Join(instructions, "\n\n");
        if (!conv.tools.empty()) {
            if (!instructions.empty()) text += "\n\n";
            text += RenderToolsSection(conv.tools);
        }
        text += "<|end|>";
    }

    for (size_t i = first; i < conv.messages.size(); ++i) {
        const ChatMessage &m = conv.messages[i];
        if (m.role == "assistant") {
            text += "<|start|>assistant<|channel|>analysis<|message|><|end|>";
            if (!m.tool_calls.empty()) {
                for (size_t c = 0; c < m.tool_calls.size(); ++c) {
                    const ToolCall &call = m.tool_calls[c];
                    text += "<|start|>assistant to=" + call.name +
                            "<|channel|>analysis <|constrain|>xml<|message|>" +
                            RenderParameters(call);
                    text += (c + 1 == m.tool_calls.size()) ? "<|call|>"
                                                          : "<|end|>";
                }
            } else {
                text += "<|start|>assistant<|channel|>final<|message|>" +
                        m.content + "<|return|>";
            }
        } else if (m.role == "tool") {
            text += "<|start|>" + ToolResultName(conv, m) +
                    " to=assistant<|channel|>analysis<|message|>" + m.content +
                    "<|end|>";
        } else {
            text += "<|start|>user<|message|>" + m.content + "<|end|>";
        }
    }
    text += "<|start|>assistant";
    return EncodeText(tok, text, out_ids);
}

bool ParseBeamToolCalls(const ArrowTokenizerDb &tok,
                        const std::vector<int32_t> &gen_tokens,
                        std::vector<ToolCall> &out_calls)
{
    out_calls.clear();
    std::vector<char> buf(gen_tokens.size() * 8 + 64);
    size_t cap = buf.size();
    const int32_t rc = arrow_tokenizer_decode(tok, gen_tokens.data(),
                                              gen_tokens.size(), buf.data(),
                                              &cap);
    if (rc < 0) return false;
    const std::string text(buf.data(), cap);

    const std::string kHead = "<|start|>assistant to=";
    const std::string kChan = "<|channel|>analysis <|constrain|>xml<|message|>";
    const std::string kParam = "<rfl:parameter name=\"";
    size_t pos = 0;
    while (true) {
        const size_t at = text.find(kHead, pos);
        if (at == std::string::npos) break;
        const size_t name_at = at + kHead.size();
        const size_t chan = text.find(kChan, name_at);
        if (chan == std::string::npos) break;
        ToolCall call;
        call.name = text.substr(name_at, chan - name_at);
        const size_t body_at = chan + kChan.size();
        size_t end = text.find("<|call|>", body_at);
        const size_t end_tag = text.find("<|end|>", body_at);
        if (end == std::string::npos || (end_tag != std::string::npos && end_tag < end))
            end = end_tag;
        if (end == std::string::npos) break;
        const std::string body = text.substr(body_at, end - body_at);
        pos = end + 1;

        std::string args = "{";
        bool first = true;
        size_t p = 0;
        while (true) {
            const size_t k = body.find(kParam, p);
            if (k == std::string::npos) break;
            const size_t key_at = k + kParam.size();
            const size_t key_end = body.find('"', key_at);
            if (key_end == std::string::npos) break;
            const std::string key = body.substr(key_at, key_end - key_at);
            const size_t val_at = body.find('>', key_end);
            if (val_at == std::string::npos) break;
            size_t val_begin = val_at + 1;
            if (val_begin < body.size() && body[val_begin] == '\n') ++val_begin;
            size_t val_end = body.find("\n</rfl:parameter>", val_begin);
            if (val_end == std::string::npos)
                val_end = body.find("</rfl:parameter>", val_begin);
            if (val_end == std::string::npos) break;
            size_t value_stop = val_end;
            if (value_stop > val_begin && body[value_stop - 1] == '\n') --value_stop;
            const std::string value =
                body.substr(val_begin, value_stop - val_begin);
            if (!first) args += ",";
            first = false;
            AppendEscapedJsonString(args, key);
            args += ":";
            args += ParameterToJson(value);
            p = val_end + 1;
        }
        args += "}";
        call.arguments_json = args;
        out_calls.push_back(std::move(call));
    }
    return !out_calls.empty();
}

} // namespace lsxcommon::beam
