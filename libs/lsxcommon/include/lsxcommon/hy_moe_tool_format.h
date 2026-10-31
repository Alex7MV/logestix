#pragma once

#include <string>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/tokenizer.h"

namespace lsxcommon::hy {

// Hy-Memory-30B-A3B native chat/tool codec (checkpoint chat_template.jinja).
//
// Framing (every marker is a literal vocabulary piece injected by id):
//   <bos>[<system>]<reasoning_mode>reasoning_effort:<effort>   (no tools)
//   <bos>[<system>][# Tools ... </tools> ... </tool_calls><reasoning_mode>...
//                                                             (tools present)
//   <|User|><text>
//   <|Assistant|><think></think><text><eos:6124c78e>          (answer)
//   <|Assistant|><think>[reasoning]</think><text><tool_calls>\n
//       <tool_call>{name}<tool_sep>\n
//       <arg_key>{k}</arg_key>\n<arg_value>{v}</arg_value>\n ... </tool_call>\n
//     </tool_calls><eos:6124c78e>                             (tool call)
//   <tool_responses>\n<tool_response>\n{content}\n</tool_response>\n</tool_responses>
//
// The generation prompt carries the template's `no_think` default, i.e.
// `<|Assistant|><think></think>`, so the answer starts directly.
bool HyChatSentinelsPresent(const ArrowTokenizerDb &tok);

// Per-checkpoint template deltas. Hy-Memory's tool instructions close the
// example call with `</tool_call>`; Hy-SimulMT's template closes it with
// `</tool_calls>` (the checkpoint's own literal). Everything else is shared.
struct HyRenderPolicy {
    bool example_call_close_is_calls_end = false;
};

// Renders a conversation (system prompt, optional tools, tool calls and tool
// responses) into the checkpoint's exact token framing.
bool RenderHyConversation(const ArrowTokenizerDb &tok,
                          const ChatConversation &conv,
                          std::vector<int32_t> &out_ids);
bool RenderHyConversation(const ArrowTokenizerDb &tok,
                          const ChatConversation &conv,
                          std::vector<int32_t> &out_ids,
                          const HyRenderPolicy &policy);

// Parses the native `<tool_calls>` block out of the generated tokens; falls
// back to the generic Hermes/bare-JSON parsers when the native markers are
// absent.
bool ParseHyToolCalls(const ArrowTokenizerDb &tok,
                      const std::vector<int32_t> &gen_tokens,
                      std::vector<ToolCall> &out_calls);

// Renders/parses one `<tool_call>` block (exposed for unit tests).
std::string RenderHyToolCallText(const ToolCall &call);
std::vector<ToolCall> ParseHyToolCallText(const std::string &text);

} // namespace lsxcommon::hy
