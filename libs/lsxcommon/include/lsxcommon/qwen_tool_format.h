#pragma once

// Qwen3.5 / 3.6 ChatML + native tool codec (host-only), shared by the dense
// QwenBaseModel and the MoE QwenMoeModelBase (they are unrelated classes).
//
// Verified against the checkpoint chat_template.jinja:
//   <|im_start|>system\n# Tools\n\n…<tools>…</tools>…<IMPORTANT>…</IMPORTANT>
//   <|im_start|>user\n…<|im_end|>\n
//   <|im_start|>assistant\n{content}<tool_call><function=NAME><parameter=K>V…
//   <|im_start|>user\n<tool_response>…</tool_response><|im_end|>\n
//   <|im_start|>assistant\n<think>\n\n</think>\n\n   (enable_thinking=false)

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/tokenizer.h"

namespace lsxcommon {

// Renders a conversation as the Qwen ChatML/XML transcript text. The
// generation prompt is the checkpoint's enable_thinking=false prefill
// (pre-closed empty think block) unless @p enable_thinking is true, in which
// case the model is left to reason inside an open <think>.
bool RenderQwenConversation(const ChatConversation &conv, std::string &out,
                            bool enable_thinking = false);

// Parses every `<tool_call>…<function=NAME>…<parameter=K>V</parameter>…`
// (and `<parameter=K>V</parameter>`) XML call in @p text.
std::vector<ToolCall> ParseQwenToolCallText(std::string_view text);

// True when the ChatML control sentinels are present in the vocabulary.
bool QwenChatSentinelsPresent(const ArrowTokenizerDb &tok);

// Appends @p text to @p out_ids, injecting the ChatML sentinels as their own
// vocab ids (arrow_tokenizer_encode is pure BPE and never resolves added
// tokens). Returns false (and clears out_ids) when a sentinel is missing.
bool AppendQwenChatIds(const ArrowTokenizerDb &tok, const std::string &text,
                       std::vector<int32_t> &out_ids);

// Decode generated ids → native XML parse → generic Hermes → bare-JSON
// fallback. Returns false only when decoding fails.
bool ParseQwenTokenCalls(const ArrowTokenizerDb &tok,
                         const std::vector<int32_t> &gen_tokens,
                         std::vector<ToolCall> &out_calls);

} // namespace lsxcommon
