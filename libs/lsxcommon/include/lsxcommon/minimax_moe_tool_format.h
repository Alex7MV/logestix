#pragma once

#include <string>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/tokenizer.h"

namespace lsxcommon::minimax {

// MiniMax-M3 native chat/tool codec (checkpoint chat_template.jinja).
//
// Framing (all markers are literal vocabulary pieces):
//   ]~!b[]~b]system\n<system>[e~[\n
//   ]~b]developer\n<developer>[\n\n# Tools\n...<tools>...</tools>...][e~[\n
//   ]~b]user\n<text>[e~[\n
//   ]~b]ai\n[<mm:think>reasoning</mm:think>|</mm:think>]<text>[
//     ]<]minimax[>[<tool_call>]<]minimax[>[<invoke name="f">]<]minimax[>[<k>v
//     ]<]minimax[>[</k>]<]minimax[>[</invoke>]<]minimax[>[</tool_call>][e~[\n
//   ]~b]tool\n<response>...</response>[e~[\n
//
// Thinking mode is "disabled" by default in this engine: the generation prompt
// is prefixed with the think-END marker, which is the template's own
// "answer directly" switch (the same prefix an assistant turn without
// reasoning_content carries).
bool MinimaxChatSentinelsPresent(const ArrowTokenizerDb &tok);

// Per-checkpoint deltas of the shared MiniMax chat template. The default
// policy reproduces MiniMax-M3 (thinking disabled + think-END generation
// prefix, no developer blocks, one `]~b]tool` block per tool message).
struct MinimaxRenderPolicy {
    const char *default_system = nullptr;      // nullptr => the M3 text
    const char *thinking_mode_line = nullptr;  // nullptr => the M3 disabled line
    const char *system_suffix = nullptr;       // e.g. "\n\n<model_id>..."
    bool generation_think_end = true;          // M3 prefixes </mm:think>
    bool developer_blocks = false;             // mid-conversation system turns
    bool batch_tool_results = false;           // consecutive tool msgs share a block
};

// The MiniMax-M3.1-Flash checkpoint's own template (chat_template.jinja):
// model version M3.1 + `<model_id>8bb8149fe</model_id>`, fixed ADAPTIVE
// thinking (no generation prefix), `developer` blocks for mid-conversation
// system/developer turns, and ONE `]~b]tool` block for a run of tool results.
MinimaxRenderPolicy MinimaxFlashRenderPolicy();

bool RenderMinimaxConversation(const ArrowTokenizerDb &tok,
                               const ChatConversation &conv,
                               std::vector<int32_t> &out_ids);

bool RenderMinimaxConversation(const ArrowTokenizerDb &tok,
                               const ChatConversation &conv,
                               const MinimaxRenderPolicy &policy,
                               std::vector<int32_t> &out_ids);

// Parses `<invoke name="f"><k>v</k>...</invoke>` blocks out of the decoded
// generated text; falls back to the generic Hermes/bare-JSON parsers.
bool ParseMinimaxToolCalls(const ArrowTokenizerDb &tok,
                           const std::vector<int32_t> &gen_tokens,
                           std::vector<ToolCall> &out_calls);

// Renders one assistant tool call in the native XML shape (exposed for tests).
std::string RenderMinimaxInvoke(const ToolCall &call);

// Parses an invoke list from plain text (exposed for tests).
std::vector<ToolCall> ParseMinimaxInvokeText(const std::string &text);

} // namespace lsxcommon::minimax
