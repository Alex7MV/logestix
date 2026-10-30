#pragma once

#include <string>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/tokenizer.h"

namespace lsxcommon::flint {

// Native Recursive Flint chat codec (Model Ingestion Protocol step 4b), derived
// from the checkpoint's own chat_template.jinja:
//
//   <|system|>S<|user|>U<|assistant|>            roles are INLINE (no separator)
//   <|assistant|><think>rc</think>text<tool_call>name<arg_key>k</arg_key>
//   <arg_value>v</arg_value></tool_call>
//   <|observation|><tool_response>...</tool_response>
//
// The tools preamble is its own `<|system|>` turn emitted before the first
// non-system message. Free text goes through the template's `flint_text`
// sanitation (drop NUL, strip literal <think>/</think>, HTML-escape the role
// markers).

// Renders the template's text (no tokenization). Exposed so the framing test
// can compare byte-for-byte with the checkpoint's own Jinja output.
std::string RenderFlintConversationText(const ChatConversation &conv,
                                        bool add_generation_prompt);

// Renders the conversation to token ids, injecting every control marker as its
// own vocab id (added_tokens are never BPE'd). Returns false when the
// tokenizer lacks the required markers.
bool RenderFlintConversation(const ArrowTokenizerDb &tok,
                             const ChatConversation &conv,
                             std::vector<int32_t> &out_ids);

bool FlintChatSentinelsPresent(const ArrowTokenizerDb &tok);

// Parses `<tool_call>` blocks out of decoded generation text. Values that are
// valid JSON keep their type; everything else is a string.
std::vector<ToolCall> ParseFlintToolCallText(const std::string &text);

bool ParseFlintToolCalls(const ArrowTokenizerDb &tok,
                         const std::vector<int32_t> &gen_tokens,
                         std::vector<ToolCall> &out_calls);

} // namespace lsxcommon::flint
