// Inkling-2-Small native conversation + tool-call codec (Model Ingestion
// Protocol step 4b). Host-only.
//
// The grammar is transcribed from the checkpoint's `chat_template.jinja`; the
// expected frames in tests/test_inkling_moe_tool_framing.cpp are the template's
// own output, not a paraphrase.
#include "lsxcommon/inkling_moe_tool_format.h"
#include "lsxcommon/tool_format.h"

#include <algorithm>

namespace lsxcommon {
namespace {


// ── markers (added_tokens, all special:true) ──────────────────────────────
constexpr char kUser[] = "<|message_user|>";
constexpr char kModel[] = "<|message_model|>";
constexpr char kSystem[] = "<|message_system|>";
constexpr char kTool[] = "<|message_tool|>";
constexpr char kText[] = "<|content_text|>";
constexpr char kThinking[] = "<|content_thinking|>";
constexpr char kEndMessage[] = "<|end_message|>";
constexpr char kEndSampling[] = "<|content_model_end_sampling|>";
constexpr char kXml[] = "<|content_xml|>";
constexpr char kInvokeJson[] = "<|content_invoke_tool_json|>";
// The template's default reasoning effort (`reasoning_effort` unset -> 0.9).
constexpr char kEffortPrefix[] = "Thinking effort level: ";
constexpr char kEffortDefault[] = "Thinking effort level: 0.9";

void Control(std::vector<InklingSegment> &segs, const char *marker)
{
    segs.push_back({marker, true});
}

// Free text that must stay distinguishable from a marker of the same spelling:
// an empty text segment is dropped (nothing to encode), but a non-empty one is
// always its own segment so the BPE encoder never sees a marker.
void Text(std::vector<InklingSegment> &segs, std::string text)
{
    if (text.empty()) return;
    segs.push_back({std::move(text), false});
}

const char *RoleMarker(const std::string &role)
{
    if (role == "user") return kUser;
    if (role == "assistant") return kModel;
    if (role == "system") return kSystem;
    if (role == "tool") return kTool;
    return nullptr;
}

// `<role><|content_text|>…<|end_message|>`
void EmitTextTurn(std::vector<InklingSegment> &segs, const std::string &role,
                  const std::string &content)
{
    Control(segs, RoleMarker(role));
    Control(segs, kText);
    Text(segs, content);
    Control(segs, kEndMessage);
}

// Compact, key-sorted JSON of one tool spec, byte-compatible with Jinja's
// `tojson(sort_keys=true, separators=(",",":"))` over
// `{description,name,parameters,type}`.
std::string ToolSpecJson(const ToolSpec &spec)
{
    std::string out = "{\"description\":";
    out += spec.description.empty() ? "\"\"" : "\"" + spec.description + "\"";
    out += ",\"name\":\"" + spec.name + "\"";
    out += ",\"parameters\":";
    out += spec.parameters_json.empty() ? "{}" : spec.parameters_json;
    out += ",\"type\":\"function\"}";
    return out;
}

}  // namespace

std::string InklingSegmentsText(const std::vector<InklingSegment> &segs)
{
    std::string out;
    for (const auto &s : segs) out += s.text;
    return out;
}

bool RenderInklingConversation(const ChatConversation &conv,
                              std::vector<InklingSegment> &out)
{
    out.clear();
    for (const auto &m : conv.messages)
        if (RoleMarker(m.role) == nullptr) return false;

    // Tool declarations come FIRST (the template renders them before the
    // message loop).
    if (!conv.tools.empty()) {
        Control(out, kSystem);
        Text(out, "tool_declare");
        Control(out, kXml);
        std::string json = "[";
        for (std::size_t i = 0; i < conv.tools.size(); ++i) {
            if (i) json += ",";
            json += ToolSpecJson(conv.tools[i]);
        }
        json += "]";
        Text(out, json);
        Control(out, kEndMessage);
    }

    // The effort line goes before the first NON-SYSTEM message, once.
    bool effort_emitted = false;
    for (const auto &m : conv.messages) {
        if (!effort_emitted && m.role != "system") {
            Control(out, kSystem);
            Control(out, kText);
            Text(out, kEffortDefault);
            Control(out, kEndMessage);
            effort_emitted = true;
        }
        if (m.role == "assistant") {
            // Reasoning channel first, then the visible content, then the tool
            // calls, then the end-of-sampling marker. The template skips the
            // text wrapper only for an assistant turn that has NOTHING but
            // tool calls / reasoning.
            if (!m.content.empty()) EmitTextTurn(out, m.role, m.content);
            for (const auto &call : m.tool_calls) {
                if (call.name.empty()) return false;
                Control(out, kModel);
                Text(out, call.name);
                Control(out, kInvokeJson);
                std::string payload = "{\"name\":\"" + call.name + "\",\"args\":";
                payload += call.arguments_json.empty() ? "{}" : call.arguments_json;
                payload += "}";
                Text(out, payload);
                Control(out, kEndMessage);
            }
            if (m.content.empty() && m.tool_calls.empty()) {
                // An empty assistant turn still needs its frame.
                EmitTextTurn(out, m.role, std::string());
            }
            Control(out, kEndSampling);
        } else if (m.role == "tool") {
            // The tool-frame header carries the tool name when the client
            // supplied one (the template appends `message.name` to the role
            // token).
            EmitTextTurn(out, m.role, m.content);
        } else {
            EmitTextTurn(out, m.role, m.content);
        }
    }
    if (!effort_emitted) {
        Control(out, kSystem);
        Control(out, kText);
        Text(out, kEffortDefault);
        Control(out, kEndMessage);
    }
    return true;
}

std::size_t ParseInklingToolCalls(std::string_view text,
                                 std::vector<ToolCall> &out_calls)
{
    out_calls.clear();
    std::size_t pos = 0;
    while (true) {
        const std::size_t at = text.find(kInvokeJson, pos);
        if (at == std::string_view::npos) break;
        std::size_t begin = at + std::string_view(kInvokeJson).size();
        // The payload is a JSON object; scan it to its matching brace so a
        // nested "args" object is not truncated.
        while (begin < text.size() &&
               (text[begin] == ' ' || text[begin] == '\n' || text[begin] == '\t'))
            ++begin;
        if (begin >= text.size() || text[begin] != '{') {
            pos = begin;
            continue;
        }
        int depth = 0;
        bool in_str = false;
        bool esc = false;
        std::size_t end = begin;
        for (; end < text.size(); ++end) {
            const char c = text[end];
            if (in_str) {
                if (esc) {
                    esc = false;
                } else if (c == '\\') {
                    esc = true;
                } else if (c == '"') {
                    in_str = false;
                }
                continue;
            }
            if (c == '"') {
                in_str = true;
            } else if (c == '{') {
                ++depth;
            } else if (c == '}') {
                --depth;
                if (depth == 0) {
                    ++end;
                    break;
                }
            }
        }
        const std::string_view payload = text.substr(begin, end - begin);
        // The header spelling (the model token before the marker) is the
        // function name; the payload repeats it and carries the arguments.
        std::string name;
        const std::size_t nm = payload.find("\"name\"");
        if (nm != std::string_view::npos) {
            const std::size_t q = payload.find('"', nm + 6);
            const std::size_t q2 = q == std::string_view::npos
                                       ? q
                                       : payload.find('"', q + 1);
            if (q != std::string_view::npos && q2 != std::string_view::npos)
                name = std::string(payload.substr(q + 1, q2 - q - 1));
        }
        std::string args = "{}";
        const std::size_t am = payload.find("\"args\"");
        if (am != std::string_view::npos) {
            const std::size_t brace = payload.find('{', am);
            if (brace != std::string_view::npos) {
                // Stop at the args object's OWN closing brace: the payload's
                // final brace belongs to the envelope, not to the arguments.
                int d = 0;
                bool str = false;
                bool escape = false;
                std::size_t e = brace;
                for (; e < payload.size(); ++e) {
                    const char c = payload[e];
                    if (str) {
                        if (escape) escape = false;
                        else if (c == '\\') escape = true;
                        else if (c == '"') str = false;
                        continue;
                    }
                    if (c == '"') str = true;
                    else if (c == '{') ++d;
                    else if (c == '}') {
                        --d;
                        if (d == 0) { ++e; break; }
                    }
                }
                args = std::string(payload.substr(brace, e - brace));
            }
        }
        if (!name.empty()) {
            ToolCall call;
            call.name = name;
            call.arguments_json = args;
            out_calls.push_back(std::move(call));
        }
        pos = end;
    }
    return out_calls.size();
}

int32_t InklingMarkerId(const ArrowTokenizerDb &tok, std::string_view spelling)
{
    return arrow_tokenizer_lookup(tok, spelling.data(), spelling.size());
}

bool AppendInklingIds(const ArrowTokenizerDb &tok,
                     const std::vector<InklingSegment> &segs,
                     std::vector<int32_t> &out_ids)
{
    return lsxcommon::tools::AppendSegmentIds<InklingSegment>(tok, segs, out_ids);
}

}  // namespace lsxcommon
