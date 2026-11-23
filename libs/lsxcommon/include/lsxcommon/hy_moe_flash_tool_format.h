// hy_moe_flash_tool_format.h — the native Hunyuan conversation/tool bridge
// (Model Ingestion Protocol step 4b).
//
// The format is transcribed from the checkpoint's own chat_template.jinja, not
// guessed: role turns are `<｜hy_start:opensource｜>{role}<｜hy_middle:opensource｜>
// {body}<｜hy_end:opensource｜>`, reasoning sits in `<think:opensource>…</think:opensource>`
// and tool calls are
//   `<tool_calls:opensource><tool_call:opensource>{name}`
//   `<arg_key:opensource>{k}</arg_key:opensource>`
//   `<arg_value:opensource>{v}</arg_value:opensource>` …
//   `</tool_call:opensource></tool_calls:opensource>`
// Tool results come back inside `<tool_response:opensource>…</tool_response:opensource>`
// in a `tool` turn, and the template's `reasoning_effort` switch
// (`no_think` / `high`) selects whether the generation prompt opens a thinking
// block.
#pragma once

#include <string>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/tokenizer.h"

namespace lsxcommon {
namespace hy4 {

// Renders `conv` to text exactly as the checkpoint's template does for the
// text-only path. `reasoning_effort` is "no_think" or "high".
std::string RenderHy4ConversationText(const ChatConversation &conv,
                                      bool add_generation_prompt,
                                      std::string_view reasoning_effort);

// True when the tokenizer carries the seven control markers the native frame
// needs (a synthetic fixture must add them to its added_tokens).
bool Hy4ChatSentinelsPresent(const ArrowTokenizerDb &tok);

// Frames `conv` into input ids, injecting every control marker as its own vocab
// id and BPE-encoding only free text.
bool RenderHy4Conversation(const ArrowTokenizerDb &tok,
                           const ChatConversation &conv,
                           std::vector<int32_t> &out_ids);

// Parses `<tool_call:opensource>` blocks out of generated text.
std::vector<ToolCall> ParseHy4ToolCallText(const std::string &text);

// Decodes generated ids and parses; falls back to the shared Hermes parser and
// then to a bare JSON object when the native markers are absent.
bool ParseHy4ToolCalls(const ArrowTokenizerDb &tok,
                       const std::vector<int32_t> &gen_tokens,
                       std::vector<ToolCall> &out_calls);

} // namespace hy4
} // namespace lsxcommon
