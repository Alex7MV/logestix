#pragma once

// Dory1.0-4B conversation / tool codec (Model Ingestion Protocol step 4b).
//
// Verified against the checkpoint's own `chat_template.jinja` (Nemotron-v3
// ChatML with a `<tools>` XML preamble and Qwen3-Coder XML tool calls):
//
//   <|im_start|>system\n[# Tools\n\n…<tools>\n<function>…</function>\n</tools>
//                   …<IMPORTANT>…</IMPORTANT>]<|im_end|>\n
//   <|im_start|>user\n…<|im_end|>\n
//   <|im_start|>assistant\n<think></think>…<tool_call>\n<function=NAME>\n
//                   <parameter=K>\nV\n</parameter>\n</function>\n</tool_call>\n
//                   <|im_end|>\n
//   <|im_start|>user\n<tool_response>\n…\n</tool_response>\n<|im_end|>\n
//   <|im_start|>assistant\n<think></think>      (enable_thinking=false prefill)
//   <|im_start|>assistant\n<think>\n            (enable_thinking=true prefill)
//
// Two properties of THIS template differ from the Qwen one and are why the
// renderer is family-local: the (possibly empty) system turn is always emitted,
// and tool definitions are serialised as the template's own `<function>` /
// `<parameter>` XML instead of a JSON array. The tool-call WIRE it emits is
// byte-identical to the Qwen3-Coder format, so the parser is shared
// (`ParseQwenToolCallText`); only rendering is Dory's.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/tokenizer.h"

namespace lsxcommon {

struct DoryRenderPolicy {
    // Leave the generation prompt inside an OPEN `<think>` block. TRUE is the
    // checkpoint's served mode (the model card enables reasoning on vLLM and
    // llama.cpp, and it is the template default); the FSI/CODE/DET goldens are
    // captured in it. Thinking off changes DET-TEST-02's answer (CRITICAL
    // instead of CATASTROPHIC), so this is a semantic switch, not a style one.
    bool enable_thinking = true;
    // Drop the reasoning block of assistant turns that precede the last user
    // turn (the template's truncate_history_thinking default).
    bool truncate_history_thinking = true;
};

// Renders a conversation as the Dory ChatML/XML transcript. Returns false on an
// unsupported role or a malformed tool schema.
bool RenderDoryConversation(const ChatConversation &conv, std::string &out,
                            const DoryRenderPolicy &policy = {});

// Parses every `<tool_call>…<function=NAME>…<parameter=K>V</parameter>` call in
// decoded text (shared Qwen3-Coder wire).
std::vector<ToolCall> ParseDoryToolCallText(std::string_view text);

// True when the ChatML control sentinels are present in the vocabulary.
bool DoryChatSentinelsPresent(const ArrowTokenizerDb &tok);

// Appends `text` to `out_ids`, injecting the eight ChatML/think/tool sentinels
// as their own vocab ids (arrow_tokenizer_encode is pure BPE and never resolves
// added tokens).
bool AppendDoryChatIds(const ArrowTokenizerDb &tok, const std::string &text,
                       std::vector<int32_t> &out_ids);

// Decode generated ids -> native XML parse -> generic Hermes -> bare-JSON.
bool ParseDoryTokenCalls(const ArrowTokenizerDb &tok,
                         const std::vector<int32_t> &gen_tokens,
                         std::vector<ToolCall> &out_calls);

} // namespace lsxcommon
