#pragma once

// Kimi-K2.7-Code native conversation + tool-call codec (host-only).
//
// Grammar transcribed from the checkpoint's chat_template.jinja:
//   system       <|im_system|>system<|im_middle|>{text}<|im_end|>
//   user         <|im_user|>user<|im_middle|>{text}<|im_end|>
//   assistant    <|im_assistant|>assistant<|im_middle|><think></think>{text}
//                {tool-calls}<|im_end|>
//   tool result  <|im_system|>tool<|im_middle|>## Return of {id}\n{text}<|im_end|>
//   tools        <|im_system|>tool_declare<|im_middle|>[{...}]<|im_end|>
//   tool calls   <|tool_calls_section_begin|><|tool_call_begin|>{id}
//                <|tool_call_argument_begin|>{json}<|tool_call_end|>
//                <|tool_calls_section_end|>
//   generation   <|im_assistant|>assistant<|im_middle|><think>
//
// Call ids on the wire are "functions.{name}:{index}" — the only place the
// function name appears (the deployment guide names the vendor parser
// `kimi_k2`; the template renders `tool_call['id']` verbatim). A following
// role=tool message echoes the same id, so the renderer keeps a client-id ->
// wire-id map while it walks the conversation.
//
// Kimi-K2.7-Code is thinking-only: every assistant turn carries a <think>
// block (empty when the caller supplied no reasoning) and the generation
// prompt ends with <think>. Vision content is out of scope (the tower is
// stripped at conversion).

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/tokenizer.h"

namespace lsxcommon {

// Renders the whole conversation (tools preamble, turns, tool results and the
// generation prompt) into Kimi's wire text. Returns false for an unknown role.
bool RenderKimiConversation(const ChatConversation &conv, std::string &out);

// Parses every native tool-call block in @p text; falls back to the generic
// Hermes text parser and then the bare-object parser when no native block is
// present.
std::vector<ToolCall> ParseKimiToolCalls(std::string_view text);

// Appends @p text to @p out_ids, injecting the `<|im_*|>`, tool and `<think>`
// sentinels as their own vocabulary ids and BPE-encoding the rest. Returns
// false when a sentinel is missing from the vocabulary or the encoder fails.
bool AppendKimiIds(const ArrowTokenizerDb &tok, std::string_view text,
                   std::vector<int32_t> &out_ids);

}  // namespace lsxcommon
