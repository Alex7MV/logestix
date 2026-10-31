#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/tokenizer.h"

namespace lsxcommon::step {

// True when the tokenizer resolves the Step-5 native control markers
// (`<|im_start|>`, `<|im_end|>`, `<think>`, `<tool_call>`).
bool StepChatSentinelsPresent(const ArrowTokenizerDb &tok);

// Renders the checkpoint's chat template (chat_template.jinja) into input ids:
// BOS, `<|im_start|>system` block (with the tool schema when tools are given),
// per-message `<|im_start|>role` framing, and the `<|im_start|>assistant\n
// <think>` generation prefix.
bool RenderStepConversation(const ArrowTokenizerDb &tok,
                            const ChatConversation &conv,
                            std::vector<int32_t> &out_ids);

// Parses the native tool-call XML
//   <tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n
//   </function>\n</tool_call>
// from decoded generation tokens. A bare Hermes JSON object falls back to the
// shared parser.
std::vector<ToolCall> ParseStepToolCalls(const ArrowTokenizerDb &tok,
                                         const std::vector<int32_t> &gen_tokens);

} // namespace lsxcommon::step
