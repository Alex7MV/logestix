// Inkling-2-Mini native conversation + tool-call codec (Model Ingestion
// Protocol step 4b). Host-only.
//
// The grammar is transcribed from the checkpoint's `chat_template.jinja`; the
// expected frames in tests/test_inkling_moe_loop_tool_framing.cpp are that
// template's own output rendered with jinja2, not a paraphrase.
#include "lsxcommon/inkling_moe_loop_tool_format.h"

#include "lsxcommon/prompt_framing.h"
#include "lsxcommon/tool_format.h"

namespace lsxcommon::inkling::loop::tools {
namespace {

// ── markers (added_tokens, all special:true) ──────────────────────────────
constexpr char kUser[] = "<|message_user|>";
constexpr char kModel[] = "<|message_model|>";
constexpr char kSystem[] = "<|message_system|>";
constexpr char kTool[] = "<|message_tool|>";
constexpr char kText[] = "<|content_text|>";
constexpr char kEndMessage[] = "<|end_message|>";
constexpr char kEndSampling[] = "<|content_model_end_sampling|>";
constexpr char kXml[] = "<|content_xml|>";
constexpr char kInvokeJson[] = "<|content_invoke_tool_json|>";
// The template's default reasoning effort (`reasoning_effort` unset -> 0.9).
constexpr char kEffortDefault[] = "Thinking effort level: 0.9";

void Control(std::vector<Segment> &segs, const char *marker)
{
    segs.push_back({marker, true});
}

// Free text that must stay distinguishable from a marker of the same spelling:
// an empty text segment is dropped (nothing to encode), but a non-empty one is
// always its own segment so the BPE encoder never sees a marker.
void Text(std::vector<Segment> &segs, std::string text)
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

// `<role>[{name}]<|content_text|>…<|end_message|>`. The tool role carries the
// tool name between the role token and the content marker.
void EmitTextTurn(std::vector<Segment> &segs, const std::string &role,
                  const std::string &content, const std::string &name = {})
{
    Control(segs, RoleMarker(role));
    if (!name.empty()) Text(segs, name);
    Control(segs, kText);
    Text(segs, content);
    Control(segs, kEndMessage);
}

// The template renders `message.name` on a tool result; ChatMessage carries no
// name, so resolve it through the tool_call_id of an earlier assistant turn.
std::string ToolResultName(const ChatConversation &conv, const ChatMessage &m)
{
    if (m.tool_call_id.empty()) return {};
    for (const auto &prev : conv.messages) {
        if (prev.role != "assistant") continue;
        for (const auto &tc : prev.tool_calls)
            if (tc.id == m.tool_call_id && !tc.name.empty()) return tc.name;
    }
    return {};
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

std::string SegmentsText(const std::vector<Segment> &segs)
{
    std::string out;
    for (const auto &s : segs) out += s.text;
    return out;
}

bool RenderConversation(const ChatConversation &conv,
                        std::vector<Segment> &out)
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
            // Visible content first, then the tool calls, then the
            // end-of-sampling marker. The template skips the text wrapper only
            // for an assistant turn that has NOTHING but tool calls.
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
            EmitTextTurn(out, m.role, m.content, ToolResultName(conv, m));
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

std::size_t ParseToolCalls(std::string_view text,
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

int32_t MarkerId(const ArrowTokenizerDb &tok, std::string_view spelling)
{
    return arrow_tokenizer_lookup(tok, spelling.data(), spelling.size());
}

bool AppendIds(const ArrowTokenizerDb &tok, const std::vector<Segment> &segs,
               std::vector<int32_t> &out_ids)
{
    return lsxcommon::tools::AppendSegmentIds<Segment>(tok, segs, out_ids);
}

bool EncodePromptIds(const ArrowTokenizerDb &tok, std::string_view raw_prompt,
                     std::vector<int32_t> &out_ids)
{
    ChatConversation conv;
    conv.messages.push_back(
        {"user", std::string(raw_prompt), std::string(), {}});
    std::vector<Segment> segs;
    if (!RenderConversation(conv, segs)) return false;
    // Without the template markers (a synthetic fixture) fall back to the raw
    // encode rather than emitting an unframed ChatML string.
    if (MarkerId(tok, kUser) < 0)
        return lsxcommon::prompt::EncodeRawPrompt(tok, raw_prompt, out_ids);
    return AppendIds(tok, segs, out_ids);
}

bool EncodeConversationIds(const ArrowTokenizerDb &tok,
                           const ChatConversation &conv,
                           std::vector<int32_t> &out_ids)
{
    std::vector<Segment> segs;
    if (!RenderConversation(conv, segs)) return false;
    if (MarkerId(tok, kUser) < 0)
        return EncodePromptIds(tok, SegmentsText(segs), out_ids);
    return AppendIds(tok, segs, out_ids);
}

bool ParseGeneratedCalls(const ArrowTokenizerDb &tok,
                         const std::vector<int32_t> &gen_tokens,
                         std::vector<ToolCall> &out_calls)
{
    out_calls.clear();
    if (gen_tokens.empty()) return true;
    // Every marker is special:true, so decode drops it. A tool call is therefore
    // found by SCANNING THE IDS for the invoke marker and decoding only the span
    // after it.
    const int32_t invoke = MarkerId(tok, kInvokeJson);
    const int32_t end_msg = MarkerId(tok, kEndMessage);
    if (invoke < 0) {
        // No native markers: fall back to the generic parser over the text.
        std::string text(4096, '\0');
        std::size_t cap = text.size();
        for (int attempt = 0; attempt < 8; ++attempt) {
            const int rc = arrow_tokenizer_decode(tok, gen_tokens.data(),
                                                 gen_tokens.size(), text.data(),
                                                 &cap);
            if (rc == 0) {
                text.resize(cap);
                out_calls = lsxcommon::tools::ParseToolCallsText(text);
                return true;
            }
            if (rc == -2 && cap > text.size() && cap <= (1u << 20)) {
                text.resize(cap);
                continue;
            }
            break;
        }
        return false;
    }

    std::size_t i = 0;
    while (i < gen_tokens.size()) {
        if (gen_tokens[i] != invoke) {
            ++i;
            continue;
        }
        std::size_t j = i + 1;
        while (j < gen_tokens.size() && gen_tokens[j] != end_msg &&
               gen_tokens[j] != invoke)
            ++j;
        if (j <= i + 1) {
            i = j;
            continue;
        }
        std::string text((j - i) * 8 + 16, '\0');
        std::size_t cap = text.size();
        if (arrow_tokenizer_decode(tok, gen_tokens.data() + i + 1, j - i - 1,
                                   text.data(), &cap) == 0) {
            text.resize(cap);
            // Re-attach the marker spelling so the shared parser sees the native
            // payload shape it expects.
            ParseToolCalls(std::string(kInvokeJson) + text, out_calls);
        }
        i = j;
    }
    return true;
}

}  // namespace lsxcommon::inkling::loop::tools
