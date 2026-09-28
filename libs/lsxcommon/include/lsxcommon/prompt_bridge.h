#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/tokenizer.h"

namespace lsxcommon {

/// Host-only: frame a raw prompt into model input ids using the model family's
/// own chat template (`IModel::BuildPromptInputIds`). The model instance is
/// created through the factory but NOT Load()ed: no weights are read and no GPU
/// work happens. Framing ownership stays with each model (no central registry).
bool FramePromptForModel(const std::string& model_path,
                         std::string_view raw_prompt,
                         std::vector<std::int32_t>& out_ids);

/// Host-only: render a full conversation (with optional tool schemas) into
/// model input ids via the family's `IModel::BuildConversationInputIds`. The
/// throwaway model instance is not Load()ed (no GPU work).
bool FrameConversationForModel(const std::string& model_path,
                               const ArrowTokenizerDb& tok,
                               const ChatConversation& conv,
                               std::vector<std::int32_t>& out_ids);

/// Host-only: parse generated tokens into tool calls via the family's
/// `IModel::ParseToolCalls`. @p tok is the same tokenizer used for framing.
bool ParseToolCallsForModel(const std::string& model_path,
                            const ArrowTokenizerDb& tok,
                            const std::vector<std::int32_t>& gen_tokens,
                            std::vector<ToolCall>& out_calls);

}  // namespace lsxcommon
