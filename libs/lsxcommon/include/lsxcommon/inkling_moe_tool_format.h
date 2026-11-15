#pragma once

// Inkling-2-Small native conversation + tool-call codec (host-only).
//
// Grammar transcribed from the checkpoint's own `chat_template.jinja`, which is
// the source of truth for this family (there is no separate encoder module).
// Every marker below is an ADDED token with `special: true`, so it is injected
// as its own vocabulary id and never BPE-encoded; free text goes through the
// ordinary BPE. Inside a ChatML frame the tokenizer's auto-BOS is dropped (this
// checkpoint has no BOS at all).
//
//   role header      <|message_user|> | <|message_model|>
//                    <|message_system|> | <|message_tool|>
//   content wrappers <|content_text|>{text}<|end_message|>
//                    <|content_thinking|>{reasoning}<|end_message|>
//                    <|content_image|><|unused_200054|><|end_message|>
//   tool declaration <|message_system|>tool_declare<|content_xml|>{compact json}
//                    <|end_message|>
//   tool call        <|message_model|>{name}<|content_invoke_tool_json|>
//                    {"name":"{name}","args":{...}}<|end_message|>
//   assistant end    <|content_model_end_sampling|>          (this IS the EOS)
//   effort line      <|message_system|><|content_text|>Thinking effort level:
//                    0.9<|end_message|>
//
// Two structural rules that are easy to miss and both pinned by the test:
//   * the effort line is emitted BEFORE the first non-system message (not
//     before a leading system message, and not at the end);
//   * every assistant turn ends with <|content_model_end_sampling|>, which is
//     also the configured stop token — so a rendered history must not append it
//     after the LAST assistant turn twice.
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

namespace lsxcommon {

// One render segment: `is_control` segments are template markers injected
// verbatim as vocabulary ids; the rest is free text and is BPE-encoded.
struct InklingSegment {
    std::string text;
    bool is_control = false;
};

// Renders the conversation (tool declarations, effort line, turns, tool calls
// and the assistant generation prompt). Returns false for an unknown role or a
// malformed tool spec.
bool RenderInklingConversation(const ChatConversation &conv,
                              std::vector<InklingSegment> &out);

// Concatenates the segment texts (framing only; used by tests and by the
// fallback path when the tokenizer lacks the markers).
std::string InklingSegmentsText(const std::vector<InklingSegment> &segs);

// Appends the encoded ids: control markers through `arrow_tokenizer_lookup`,
// free text through `arrow_tokenizer_encode`.
bool AppendInklingIds(const ArrowTokenizerDb &tok,
                     const std::vector<InklingSegment> &segs,
                     std::vector<int32_t> &out_ids);

// Parses `<|content_invoke_tool_json|>{"name":…,"args":{…}}<|end_message|>`
// payloads out of DECODED text. Returns the number of calls found.
std::size_t ParseInklingToolCalls(std::string_view text,
                                 std::vector<ToolCall> &out_calls);

// The vocabulary id of a template marker, or -1 when the tokenizer does not
// carry it (a synthetic fixture without the markers).
int32_t InklingMarkerId(const ArrowTokenizerDb &tok, std::string_view spelling);

}  // namespace lsxcommon
