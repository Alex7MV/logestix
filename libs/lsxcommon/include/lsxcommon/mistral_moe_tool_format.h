#pragma once

// Mistral-Small-4-119B native chat + tool codec (host-only, Model Ingestion
// Protocol step 4b).
//
// Verified against the checkpoint's own chat_template.jinja and tokenizer.json
// added_tokens — NOT guessed:
//
//   bos         <s>
//   system      [SYSTEM_PROMPT]{text}[/SYSTEM_PROMPT]
//   user        [INST]{text}[/INST]
//   assistant   {text}[TOOL_CALLS]{name}[ARGS]{json}…</s>
//   tool        [TOOL_RESULTS]{text}[/TOOL_RESULTS]
//   tools       [AVAILABLE_TOOLS]{tools json}[/AVAILABLE_TOOLS]
//               [MODEL_SETTINGS]{"reasoning_effort":"none"}[/MODEL_SETTINGS]
//   thinking    [THINK]{reasoning}[/THINK]
//
// Every control marker is a `special: true` added token with a FIXED id
// ([INST]=3, [/INST]=4, [TOOL_CALLS]=9, [ARGS]=32, …). arrow_tokenizer_decode
// drops special tokens, so parsing scans token ids and splices the markers back
// from the frozen ids; rendering injects the ids directly and BPE-encodes only
// free text.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/tokenizer.h"

namespace lsxcommon {
namespace mistral_moe {
namespace tool_format {

// Per-family render policy. The two checkpoints share the Mistral native frame
// but differ in the `[MODEL_SETTINGS]` block:
//   small4: emitted only when tools are present, always "none";
//   large4: always emitted before the first user turn, reasoning_effort
//           defaults to "high" (chat_template.jinja) and may be low/medium/high.
enum class MistralFormat { kSmall4 = 0, kLarge4 = 1 };

struct RenderOptions {
    MistralFormat format = MistralFormat::kSmall4;
    const char *reasoning_effort = "high";
};

// Renders the conversation exactly as chat_template.jinja would.
bool Render(const ArrowTokenizerDb &tok, const ChatConversation &conv,
            std::vector<int32_t> &out_ids, bool add_generation_prompt = true,
            const RenderOptions &opts = {});

// Same transcript as TEXT. Exposed for tests that pin the frame shape without a
// tokenizer.
bool RenderText(const ChatConversation &conv, std::string &out,
                bool add_generation_prompt = true, const RenderOptions &opts = {});

// True when every control marker this codec needs exists in the vocabulary.
bool SentinelsPresent(const ArrowTokenizerDb &tok);

// Parses every `[TOOL_CALLS]name[ARGS]{json}` block in @p text.
std::vector<ToolCall> ParseToolCallText(std::string_view text);

// Decoded ids -> native parse -> generic Hermes -> bare-JSON fallback.
bool Parse(const ArrowTokenizerDb &tok, const std::vector<int32_t> &gen_tokens,
           std::vector<ToolCall> &out_calls);

} // namespace tool_format
} // namespace mistral_moe
} // namespace lsxcommon
