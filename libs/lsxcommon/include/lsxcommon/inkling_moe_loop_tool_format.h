#pragma once

// Inkling-2-Mini native conversation + tool-call codec (host-only).
//
// Grammar transcribed from the checkpoint's own `chat_template.jinja`
// (`/mnt/fast_storage/huggingface/Inkling-2-Mini/chat_template.jinja`), which is
// the source of truth for this family. The template differs from the
// Inkling-2-Small one only by `{% generation %}` span markers, which render
// nothing, so the rendered text is identical; every frame in
// tests/test_inkling_moe_loop_tool_framing.cpp is that template's own output
// over a fixed conversation, rendered once with jinja2.
//
// Every marker below is an ADDED token with `special: true`, so it is injected
// as its own vocabulary id and never BPE-encoded; free text goes through the
// ordinary BPE. Inside a frame the tokenizer's auto-BOS is dropped (this
// checkpoint has no BOS).
//
//   role header      <|message_user|> | <|message_model|>
//                    <|message_system|> | <|message_tool|>{tool name}
//   content wrappers <|content_text|>{text}<|end_message|>
//                    <|content_thinking|>{reasoning}<|end_message|>
//   tool declaration <|message_system|>tool_declare<|content_xml|>{compact json}
//                    <|end_message|>
//   tool call        <|message_model|>{name}<|content_invoke_tool_json|>
//                    {"name":"{name}","args":{...}}<|end_message|>
//   assistant end    <|content_model_end_sampling|>          (this IS the EOS)
//   effort line      <|message_system|><|content_text|>Thinking effort level:
//                    0.9<|end_message|>
//
// Structural rules pinned by the test, all of them template behaviour:
//   * the tool declarations are a system turn emitted BEFORE the message loop;
//   * the effort line is emitted before the FIRST NON-SYSTEM message (not
//     before a leading system message), exactly once — and at the very end when
//     the conversation is all-system;
//   * an assistant turn with neither text nor tool calls still carries its
//     frame, and every assistant turn ends with the end-of-sampling marker (so
//     a rendered history must not add a second one);
//   * the tool-result header repeats the tool NAME
//     (`<|message_tool|>get_weather<|content_text|>`). ChatMessage carries no
//     `name` field, so the name is resolved from the `tool_call_id` of an
//     earlier assistant turn (the inkling-base / muse / beam precedent); an
//     unresolvable id degrades to the bare role token.
//
// The tool-call payload repeats the function name (`{name}` in the header AND
// `"name"` inside the JSON); the parser reads the JSON after the invoke marker
// and ignores the header spelling.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/tokenizer.h"

namespace lsxcommon::inkling::loop::tools {

// One render segment: `is_control` segments are template markers injected
// verbatim as vocabulary ids; the rest is free text and is BPE-encoded.
struct Segment {
    std::string text;
    bool is_control = false;
};

// Renders the conversation (tool declarations, effort line, turns, tool calls
// and the assistant generation prompt). Returns false for an unknown role or a
// tool call without a name.
bool RenderConversation(const ChatConversation &conv,
                        std::vector<Segment> &out);

// Concatenates the segment texts (framing only; used by tests and by the
// fallback path when the tokenizer lacks the markers).
std::string SegmentsText(const std::vector<Segment> &segs);

// Appends the encoded ids: control markers through `arrow_tokenizer_lookup`,
// free text through `arrow_tokenizer_encode`.
bool AppendIds(const ArrowTokenizerDb &tok, const std::vector<Segment> &segs,
               std::vector<int32_t> &out_ids);

// Parses `<|content_invoke_tool_json|>{"name":…,"args":{…}}<|end_message|>`
// payloads out of DECODED text. Returns the number of calls found.
std::size_t ParseToolCalls(std::string_view text,
                           std::vector<ToolCall> &out_calls);

// The vocabulary id of a template marker, or -1 when the tokenizer does not
// carry it (a synthetic fixture without the markers).
int32_t MarkerId(const ArrowTokenizerDb &tok, std::string_view spelling);

}  // namespace lsxcommon::inkling::loop::tools
