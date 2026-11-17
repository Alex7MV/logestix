#pragma once

// Inkling (base) native conversation + tool-call codec (host-only, Model
// Ingestion Protocol step 4b).
//
// Grammar transcribed from the BASE checkpoint's own `chat_template.jinja`
// (/mnt/fast_storage/huggingface/Inkling, verified by rendering it on
// 2026-11-17). It is the same marker family as Inkling-2-Small's with two
// verified differences:
//   * the tool-result header carries the tool name:
//     `<|message_tool|>{name}<|content_text|>…<|end_message|>` (the name is
//     resolved from message.name or the matching assistant tool_call id);
//   * the effort map has no "xhigh" key (irrelevant to the default 0.9 line).
// Every marker is an ADDED token with special:true, injected as its own
// vocabulary id; free text goes through ordinary BPE. No BOS exists.
//
//   role header      <|message_user|> | <|message_model|>
//                    <|message_system|> | <|message_tool|>
//   content wrappers <|content_text|>{text}<|end_message|>
//                    <|content_thinking|>{reasoning}<|end_message|>
//   tool declaration <|message_system|>tool_declare<|content_xml|>{compact json}
//                    <|end_message|>
//   tool call        <|message_model|>{name}<|content_invoke_tool_json|>
//                    {"name":"{name}","args":{...}}<|end_message|>
//   assistant end    <|content_model_end_sampling|>          (this IS the EOS)
//   effort line      <|message_system|><|content_text|>Thinking effort level:
//                    0.9<|end_message|>   (before the first non-system turn)
//   generation prompt <|message_model|>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/tokenizer.h"

namespace lsxcommon::inkling::base {

// One render segment: `is_control` segments are template markers injected
// verbatim as vocabulary ids; the rest is free text and is BPE-encoded.
struct Segment {
    std::string text;
    bool is_control = false;
};

// Renders the conversation. Returns false for an unknown role or a malformed
// tool call. `generation_prompt` appends the trailing <|message_model|>.
bool RenderConversation(const ChatConversation &conv,
                        std::vector<Segment> &out,
                        bool generation_prompt = false);

std::string SegmentsText(const std::vector<Segment> &segs);

bool AppendIds(const ArrowTokenizerDb &tok, const std::vector<Segment> &segs,
               std::vector<int32_t> &out_ids);

// Parses `<|content_invoke_tool_json|>{"name":…,"args":{…}}` payloads out of
// DECODED text. Returns the number of calls found.
std::size_t ParseToolCalls(std::string_view text,
                           std::vector<ToolCall> &out_calls);

// The vocabulary id of a template marker, or -1 when absent.
int32_t MarkerId(const ArrowTokenizerDb &tok, std::string_view spelling);

}  // namespace lsxcommon::inkling::base