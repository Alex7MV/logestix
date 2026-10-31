#pragma once

// Granite-Switch 4.1 native chat / aLoRA / tool codec (host-only).
//
// Verified against the checkpoint's chat_template.jinja:
//   <|start_of_role|>role<|end_of_role|>content<|end_of_text|>\n
// with the template's <tools> system block and bare-JSON tool calls:
//   <tool_call>\n{"name": ..., "arguments": ...}\n</tool_call>
// aLoRA activation inserts the control token in place of the invocation
// prefix (`<guardian>` -> `<|guardian-core|>guardian>`,
// `### DECISION` -> `<|decision|> DECISION`) or before the assistant marker.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/tokenizer.h"

namespace lsxcommon {

bool RenderGraniteSwitchConversation(const ChatConversation &conv,
                                     std::string &out);
std::vector<ToolCall> ParseGraniteSwitchToolCallText(std::string_view text);
bool AppendGraniteSwitchIds(const ArrowTokenizerDb &tok, const std::string &text,
                            std::vector<int32_t> &out_ids);
bool ParseGraniteSwitchTokenCalls(const ArrowTokenizerDb &tok,
                                  const std::vector<int32_t> &gen_tokens,
                                  std::vector<ToolCall> &out_calls);

} // namespace lsxcommon
