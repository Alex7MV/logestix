#pragma once

// GigaChat3.5 native tool codec (host-only). chatml-v5 layout with GCML calls:
//   <bos>system<|role_sep|>\n{system}<|message_sep|>\n\n
//   function descriptions<|role_sep|>\n{TypeScript namespace}<|message_sep|>\n\n
//   user<|role_sep|>\n{content}<|message_sep|>\n\n
//   assistant<|role_sep|>\n<think>{reasoning}</think>\n\n{content}\n\n
//       <｜GCML｜tool_calls>\n<｜GCML｜invoke name="NAME">\n
//       <｜GCML｜parameter name="K" string="true|false">V</｜GCML｜parameter>\n
//       </｜GCML｜invoke>\n</｜GCML｜tool_calls><|message_sep|>\n\n
//   function result<|role_sep|>\n<function_response name="NAME">body
//       </function_response><|message_sep|>\n\n
//   generation: assistant<|role_sep|>\n<think>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/tokenizer.h"

namespace lsxcommon {

bool RenderGigaChatConversation(const ChatConversation &conv, std::string &out);

std::vector<ToolCall> ParseGigaChatToolCallText(std::string_view text);

bool AppendGigaChatIds(const ArrowTokenizerDb &tok, const std::string &text,
                       std::vector<int32_t> &out_ids);

bool ParseGigaChatTokenCalls(const ArrowTokenizerDb &tok,
                             const std::vector<int32_t> &gen_tokens,
                             std::vector<ToolCall> &out_calls);

} // namespace lsxcommon
