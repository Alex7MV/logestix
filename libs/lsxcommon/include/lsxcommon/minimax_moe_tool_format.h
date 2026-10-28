#pragma once

#include <string>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/tokenizer.h"

namespace lsxcommon::minimax {

// MiniMax-M3 native chat/tool codec (checkpoint chat_template.jinja).
//
// Framing (all markers are literal vocabulary pieces):
//   ]~!b[]~b]system\n<system>[e~[\n
//   ]~b]developer\n<developer>[\n\n# Tools\n...<tools>...</tools>...][e~[\n
//   ]~b]user\n<text>[e~[\n
//   ]~b]ai\n[<mm:think>reasoning</mm:think>|</mm:think>]<text>[
//     ]<]minimax[>[<tool_call>]<]minimax[>[<invoke name="f">]<]minimax[>[<k>v
//     ]<]minimax[>[</k>]<]minimax[>[</invoke>]<]minimax[>[</tool_call>][e~[\n
//   ]~b]tool\n<response>...</response>[e~[\n
//
// Thinking mode is "disabled" by default in this engine: the generation prompt
// is prefixed with the think-END marker, which is the template's own
// "answer directly" switch (the same prefix an assistant turn without
// reasoning_content carries).
bool MinimaxChatSentinelsPresent(const ArrowTokenizerDb &tok);

bool RenderMinimaxConversation(const ArrowTokenizerDb &tok,
                               const ChatConversation &conv,
                               std::vector<int32_t> &out_ids);

// Parses `<invoke name="f"><k>v</k>...</invoke>` blocks out of the decoded
// generated text; falls back to the generic Hermes/bare-JSON parsers.
bool ParseMinimaxToolCalls(const ArrowTokenizerDb &tok,
                           const std::vector<int32_t> &gen_tokens,
                           std::vector<ToolCall> &out_calls);

// Renders one assistant tool call in the native XML shape (exposed for tests).
std::string RenderMinimaxInvoke(const ToolCall &call);

// Parses an invoke list from plain text (exposed for tests).
std::vector<ToolCall> ParseMinimaxInvokeText(const std::string &text);

} // namespace lsxcommon::minimax
