#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <simdjson.h>

#include "lsxnetwork/openai_endpoint.h"  // ChatInputMessage / ChatToolDef / ChatToolCall

namespace lsxnetwork {

/// Parsed Anthropic Messages request. `system` is flattened to text; content
/// blocks are folded into the shared chat-message shape so the same
/// model-side framing/tool machinery is reused.
struct AnthropicRequest {
    std::string model;
    int max_tokens = 0;
    bool stream = false;
    std::string system;
    std::vector<ChatInputMessage> messages;  // roles user/assistant/tool
    std::vector<ChatToolDef> tools;
    std::string tool_choice;
};

simdjson::error_code ParseAnthropicRequest(const char* data, std::size_t size,
                                           std::size_t capacity,
                                           AnthropicRequest& out);

/// Frames one SSE event: `event: <name>\ndata: <json>\n\n`.
std::string AnthropicEvent(const std::string& event, const std::string& data);

std::string AnthropicMessageStart(const std::string& id, const std::string& model,
                                  int input_tokens);
std::string AnthropicContentBlockStartText(int index);
std::string AnthropicContentBlockDeltaText(int index, const std::string& text);
std::string AnthropicContentBlockStop(int index);
std::string AnthropicToolUseBlockStart(int index, const std::string& id,
                                       const std::string& name);
std::string AnthropicToolUseDelta(int index, const std::string& partial_json);
std::string AnthropicMessageDelta(const std::string& stop_reason,
                                  int output_tokens);
std::string AnthropicMessageStop();

std::string BuildAnthropicMessage(const std::string& id, const std::string& model,
                                  const std::string& text,
                                  const std::string& stop_reason,
                                  int input_tokens, int output_tokens);
std::string BuildAnthropicToolMessage(const std::string& id,
                                      const std::string& model,
                                      const std::vector<ChatToolCall>& calls,
                                      const std::string& stop_reason,
                                      int input_tokens, int output_tokens);
std::string BuildAnthropicError(const std::string& message,
                                const std::string& type);

/// Anthropic model-discovery shape: `{"data":[{id,display_name,description,type}]}`.
std::string BuildAnthropicModelsList(const std::vector<std::string>& ids);

}  // namespace lsxnetwork
