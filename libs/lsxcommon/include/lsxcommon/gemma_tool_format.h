#pragma once

// Gemma-4 native tool codec (host-only), shared by the Gemma family.
//
// Verified against the checkpoint chat_template.jinja:
//   <bos><|turn>system\n{content}<|tool>declaration:NAME{…}<tool|><turn|>\n
//   <|turn>user\n{content}<turn|>\n
//   <|turn>model\n{content}<|tool_call>call:NAME{k:v,…}<tool_call|>
//       <|tool_response>response:NAME{value:…}<tool_response|><turn|>\n
//   generation prompt (enable_thinking=false): <|turn>model\n<|channel>thought
//       \n<channel|>
// All control markers are `special:true` (decode drops them), so the parser
// works from decoded text where they are already absent (e.g. `call:NAME{…}`).

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/tokenizer.h"

namespace lsxcommon {

bool RenderGemmaConversation(const ChatConversation &conv, std::string &out);

std::vector<ToolCall> ParseGemmaToolCallText(std::string_view text);

bool AppendGemmaChatIds(const ArrowTokenizerDb &tok, const std::string &text,
                        std::vector<int32_t> &out_ids);

bool ParseGemmaTokenCalls(const ArrowTokenizerDb &tok,
                          const std::vector<int32_t> &gen_tokens,
                          std::vector<ToolCall> &out_calls);

} // namespace lsxcommon
