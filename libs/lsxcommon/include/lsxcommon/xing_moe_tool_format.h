#pragma once

// Xing4.0-29B-A4B native chat + tool codec (host-only, Model Ingestion
// Protocol step 4b).
//
// Verified against the checkpoint's own chat_template.jinja and
// tokenizer_config.json added_tokens_decoder — NOT guessed:
//
//   system      <_system>{text}
//   user        <_user>{text}\n
//   assistant   <_bot>\n<think>\n{reasoning}\n</think>{content}<_end>\n
//   tools block "\n# Tools\n\nYou may call one or more functions ...\n<tools>"
//               {json(tool)}* "</tools>\n\nFor each function call, output the
//               function name and arguments within the following XML format:
//               <tool_call>{function-name}<param_key>…</param_key>
//               <param_value>…</param_value>…</tool_call>"
//   observation <_observation><tool_response>{content}</tool_response>
//   generation  <_bot><think>\n
//
// All eleven control markers are `special: true` added tokens with FIXED ids
// (1..6, 9..14). The tool-call markers carry a ZERO-WIDTH SPACE (U+200B)
// after the opening '<' — the same trick MiMo uses, and the reason a naive
// "<tool_call>" string never matches.
//
// Control markers are injected as their own vocab ids: arrow_tokenizer_encode
// is pure BPE and never resolves added_tokens.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/tokenizer.h"

namespace lsxcommon {
namespace xing {
namespace tool_format {

// Renders the conversation exactly as chat_template.jinja would, using the
// markers resolved from the vocabulary (see the design note in the .cpp).
bool Render(const ArrowTokenizerDb &tok, const ChatConversation &conv,
            std::vector<int32_t> &out_ids, bool add_generation_prompt);

// Same transcript as TEXT with the vocabulary's marker bytes. Exposed for
// tests that pin the frame shape without a tokenizer.
bool RenderText(const ChatConversation &conv, std::string &out,
                bool add_generation_prompt = true);

// True when every control marker this codec needs exists in the vocabulary.
bool SentinelsPresent(const ArrowTokenizerDb &tok);

// Parses every `<tool_call>name<param_key>k</param_key><param_value>v…
// </tool_call>` block in @p text.
std::vector<ToolCall> ParseToolCallText(std::string_view text);

// Render + id injection in one step (the model's BuildConversationInputIds).
bool Render(const ArrowTokenizerDb &tok, const ChatConversation &conv,
            std::vector<int32_t> &out_ids,
            bool add_generation_prompt = true);

// Decoded ids → native XML parse → generic Hermes → bare-JSON fallback.
bool Parse(const ArrowTokenizerDb &tok, const std::vector<int32_t> &gen_tokens,
           std::vector<ToolCall> &out_calls);

}  // namespace tool_format
}  // namespace xing
}  // namespace lsxcommon