#pragma once

// Ling-3.1-flash chat/tool codec (Model Ingestion Protocol step 4b).
// Native framing: '<role>SYSTEM</role>...<|role_end|>' with
// '<role>HUMAN</role>' / '<role>ASSISTANT</role>' / '<role>OBSERVATION</role>'
// and '<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>'.
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/tokenizer.h"

namespace lsxcommon::ling {

// Renders the checkpoint's chat template and encodes it, injecting every
// control marker as its own vocabulary id (BPE-encoding free text only).
bool RenderLingConversation(const ArrowTokenizerDb &tok,
                            const ChatConversation &conv,
                            std::vector<int32_t> &out_ids);

// Parses the native '<tool_call>' XML from the generated text, falling back to
// the shared XML/bare-JSON parsers. Returns false when nothing parses.
bool ParseLingToolCalls(const ArrowTokenizerDb &tok,
                        const std::vector<int32_t> &gen_tokens,
                        std::vector<ToolCall> &out_calls);

// Encodes text with the control markers resolved to their vocabulary ids.
bool EncodeLingWithControls(const ArrowTokenizerDb &tok, std::string_view text,
                            std::vector<int32_t> &out_ids);

} // namespace lsxcommon::ling
