#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <lsxcommon/chat_conversation.h>

namespace taskmosaic {

/// Per-request streaming token decoder (UTF-8-safe).
class StreamingDecoder {
public:
    virtual ~StreamingDecoder() = default;
    virtual std::string Push(std::int32_t id) = 0;
    virtual std::string Flush() = 0;
};

/// Host-only model framing/decoding boundary, injectable for tests.
class PromptFramer {
public:
    virtual ~PromptFramer() = default;
    virtual bool Encode(std::string_view model_path, std::string_view raw_prompt,
                        std::vector<std::int32_t>& out_ids) = 0;
    virtual std::string Decode(std::string_view model_path,
                               const std::vector<std::int32_t>& ids) = 0;
    virtual std::unique_ptr<StreamingDecoder> MakeDecoder(
        std::string_view model_path) = 0;

    /// Renders a full conversation (optionally carrying tool schemas) into
    /// model input ids.
    virtual bool EncodeConversation(std::string_view model_path,
                                    const lsxcommon::ChatConversation& conv,
                                    std::vector<std::int32_t>& out_ids) = 0;

    /// Parses generated tokens into tool calls.
    virtual bool ParseToolCalls(std::string_view model_path,
                                const std::vector<std::int32_t>& ids,
                                std::vector<lsxcommon::ToolCall>& out) = 0;
};

}  // namespace taskmosaic
