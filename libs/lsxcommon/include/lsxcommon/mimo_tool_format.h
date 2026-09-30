#pragma once

// MiMo-V2.6-Flash-RL ChatML + native tool codec (host-only).
//
// Verified against the checkpoint chat_template.jinja:
//   (top-level tools) <|im_start|>system\n{render_tools}<|im_end|>
//   <|im_start|>{role}\n{body}<|im_end|>
//   assistant: <|im_start|>assistant\n<think>{reasoning}</think>{content}
//              <tool_call><function=NAME>{args}</function></tool_call><|im_end|>
//   generation: <|im_start|>assistant\n[<think></think>]   (enable_thinking=false)
// render_tools: "You are provided with the following tools:\n\n<tools>"
//               {"\n" + tojson(tool)}* + "\n</tools>"
//
// <|im_start|>/<|im_end|> are added tokens (injected as their own vocab ids);
// <think>, <tool_call>, <function=, <parameter= are plain BPE text.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/tokenizer.h"

namespace lsxcommon {
namespace mimo {

// Renders a conversation as the MiMo ChatML transcript text.
bool RenderMimoConversation(const ChatConversation &conv, std::string &out,
                            bool add_generation_prompt = true,
                            bool enable_thinking = false);

// Parses every `<tool_call><function=NAME>…<parameter=K>V</parameter>…
// </function></tool_call>` call in @p text.
std::vector<ToolCall> ParseMimoToolCallText(std::string_view text);

// True when the ChatML control sentinels are present in the vocabulary.
bool MimoChatSentinelsPresent(const ArrowTokenizerDb &tok);

// Appends @p text to @p out_ids, injecting the ChatML sentinels as their own
// vocab ids (arrow_tokenizer_encode is pure BPE and never resolves added
// tokens). Returns false (and clears out_ids) when a sentinel is missing.
bool AppendMimoChatIds(const ArrowTokenizerDb &tok, const std::string &text,
                       std::vector<int32_t> &out_ids);

// Render + AppendMimoChatIds in one step (the model's BuildConversationInputIds).
bool BuildChatInputIds(const ArrowTokenizerDb &tok, const ChatConversation &conv,
                       bool add_generation_prompt, bool enable_thinking,
                       std::vector<int32_t> &out_ids);

// Decode generated ids → native XML parse → generic Hermes → bare-JSON
// fallback. Returns false only when decoding fails.
bool ParseChatToolCalls(const ArrowTokenizerDb &tok,
                        const std::vector<int32_t> &gen_tokens,
                        std::vector<ToolCall> &out_calls);

} // namespace mimo
} // namespace lsxcommon
