#pragma once

#include <string>
#include <vector>

namespace lsxcommon {

/// One tool call produced by an assistant turn (OpenAI shape; `arguments_json`
/// is the raw JSON string, not an object).
struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

/// One tool definition supplied by the client.
struct ToolSpec {
    std::string name;
    std::string description;
    std::string parameters_json;  // JSON Schema object
};

/// One conversation message. `tool_calls` is populated for assistant turns that
/// request tools; `tool_call_id` is populated for role=="tool" results.
struct ChatMessage {
    std::string role;  // system|user|assistant|tool
    std::string content;
    std::string tool_call_id;
    std::vector<ToolCall> tool_calls;
};

/// A full conversation plus the available tools.
struct ChatConversation {
    std::vector<ChatMessage> messages;
    std::vector<ToolSpec> tools;
};

}  // namespace lsxcommon
