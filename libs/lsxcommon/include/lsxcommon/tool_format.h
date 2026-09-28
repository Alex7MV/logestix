#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "lsxcommon/chat_conversation.h"

namespace lsxcommon {
namespace tools {

/// Renders the tool schemas as a Hermes-style JSON block:
///   <tools>
///   [{"type":"function","function":{"name","description","parameters"}}]
///   </tools>
std::string RenderToolsBlock(const std::vector<ToolSpec>& tools);

/// Renders a conversation as a role-marked transcript. When @p include_tools
/// and tools are present, the block is prepended as a system section.
std::string RenderConversation(const ChatConversation& conv, bool include_tools);

/// Extracts every `<tool_call>…</tool_call>` JSON object from decoded text.
/// `arguments` may be a JSON object/array (normalized to compact JSON) or an
/// already-encoded string (kept verbatim). Ids are assigned `call_0`, `call_1`, …
std::vector<ToolCall> ParseToolCallsText(std::string_view text);

}  // namespace tools
}  // namespace lsxcommon
