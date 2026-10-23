#pragma once

// Granite-4.2 native tool codec (host-only), shared by the Granite family.
//
// Verified against the checkpoint chat_template.jinja:
//   <|im_start|>system\n{system}
//   # Tools\n\nYou have access to the following functions:\n\n
//   <tools>\n{"name": ..., "description": ..., "parameters": ...}\n</tools>
//   ...<IMPORTANT>...</IMPORTANT><|im_end|>\n
//   <|im_start|>user\n{content}<|im_end|>\n
//   <|im_start|>assistant\n<think></think><tool_call>
//       <function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>
//       </tool_call><|im_end|>\n
// The control markers <|im_start|>/<|im_end|> are added special tokens; the
// tool XML is plain text. The parser therefore works on decoded text.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/tokenizer.h"

namespace lsxcommon {

bool RenderGraniteConversation(const ChatConversation &conv, std::string &out);

std::vector<ToolCall> ParseGraniteToolCallText(std::string_view text);

bool AppendGraniteChatIds(const ArrowTokenizerDb &tok, const std::string &text,
                          std::vector<int32_t> &out_ids);

bool ParseGraniteTokenCalls(const ArrowTokenizerDb &tok,
                            const std::vector<int32_t> &gen_tokens,
                            std::vector<ToolCall> &out_calls);

} // namespace lsxcommon
