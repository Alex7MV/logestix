#pragma once

// DeepSeek-V4 / V4.1 DSML conversation + tool-call codec (host-only).
//
// Both checkpoints share one encoder (encoding/encoding_dsv4.py and
// encoding/encoding.py) and differ only in the DSML tag names:
//   V4   : block "tool_calls",   call "invoke",    param "parameter"
//   V4.1 : block " calls",       call " invoke",   param " parameter"
// Tags are rendered with the "｜DSML｜" token, e.g.
//   <｜DSML｜ calls>\n<｜DSML｜ invoke name="T">\n
//   <｜DSML｜ parameter name="K" string="true">V</｜DSML｜ parameter>\n
//   </｜DSML｜ invoke>\n</｜DSML｜ calls>
//
// Scope: the bridge's chat-mode path (no thinking mode, no context window, no
// multimodal, no task/output-id handling). Tool results are rendered as
// <tool_result>…</tool_result> inside a <｜User｜> turn.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/tokenizer.h"

namespace lsxcommon {

struct DsmlTags {
    std::string_view block;  // "tool_calls" (V4) / " calls" (V4.1)
    std::string_view call;   // "invoke" (V4) / " invoke" (V4.1)
    std::string_view param;  // "parameter" (V4) / " parameter" (V4.1)
};

// Renders a conversation as the DeepSeek ChatML/DSML transcript text.
bool RenderDsmlConversation(const ChatConversation &conv, const DsmlTags &tags,
                            std::string &out);

// Parses every DSML tool-call block in @p text into OpenAI-shaped ToolCalls.
std::vector<ToolCall> ParseDsmlToolCalls(std::string_view text,
                                         const DsmlTags &tags);

// Appends @p text to @p out_ids, injecting the DeepSeek control sentinels
// (bos/eos/role/think/DSML) via arrow_tokenizer_lookup and BPE-encoding the
// rest. Returns false when a sentinel is missing or the encoder fails.
bool AppendDeepSeekIds(const ArrowTokenizerDb &tok, std::string_view text,
                       std::vector<int32_t> &out_ids);

// Shared override body: decode generated ids and parse DSML, falling back to
// the generic Hermes text parser when no DSML block is present.
bool ParseDeepSeekToolCalls(const ArrowTokenizerDb &tok,
                            const std::vector<int32_t> &gen_tokens,
                            const DsmlTags &tags,
                            std::vector<ToolCall> &out_calls);

} // namespace lsxcommon
