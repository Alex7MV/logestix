#pragma once

#include <cstdint>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/tokenizer.h"

namespace lsxcommon::alice {

// Native AliceAI chat/tool codec (chat_template.jinja): roles
// `system:`/`user:`/`assistant:`/`meta:`, a `` tools header followed by
// `function {...}` lines, `[TOOL_CALL_START]`-prefixed assistant tool calls and
// `[COT_START]...[COT_END]` reasoning. Free text is BPE-encoded; the control
// markers are ordinary vocabulary pieces.
bool RenderConversation(const ArrowTokenizerDb &tok,
                        const ChatConversation &conv,
                        bool add_generation_prompt,
                        std::vector<int32_t> &out_ids);

// Parses `[TOOL_CALL_START]<name>\n<args-json>` from the decoded generation,
// falling back to the generic text/bare-JSON parsers.
bool ParseToolCallsAlice(const ArrowTokenizerDb &tok,
                         const std::vector<int32_t> &gen_tokens,
                         std::vector<ToolCall> &out_calls);

} // namespace lsxcommon::alice
