#pragma once

// Qwen3.5 / 3.6 / 3.8 ChatML + native tool codec (host-only), shared by the
// dense QwenBaseModel, the MoE QwenMoeModelBase and the qwen4_exp
// (Qwen3.8-Flash-Next) model — they are unrelated classes.
//
// Verified against the checkpoints' own chat_template.jinja:
//   <|im_start|>system\n# Tools\n\n…<tools>…</tools>…<IMPORTANT>…</IMPORTANT>
//   <|im_start|>user\n…<|im_end|>\n
//   <|im_start|>assistant\n{content}<tool_call><function=NAME><parameter=K>V…
//   <|im_start|>user\n<tool_response>…</tool_response><|im_end|>\n
//   <|im_start|>assistant\n<think>\n\n</think>\n\n   (enable_thinking=false)
//
// The 3.8 (`qwen4_exp`) template adds two deltas over the 3.5/3.6 one, carried
// by QwenRenderPolicy so the older families keep their own wire:
//   * every assistant *history* turn is prefixed with the closed think block
//     (`<think>\n{reasoning}\n</think>\n\n`), which is also what the
//     preserve_thinking=true default of the 3.8 template renders for an empty
//     reasoning trace;
//   * the tool definitions are serialised like jinja's `tojson`
//     (json.dumps(obj, ensure_ascii=False): ", " and ": " separators) instead of
//     the compact form.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/tokenizer.h"

namespace lsxcommon {

/// Per-family switch for the deltas between the Qwen3.5/3.6 template and the
/// Qwen3.8 (`qwen4_exp`) one. The defaults reproduce the 3.5/3.6 wire exactly.
struct QwenRenderPolicy {
    /// Leave the generation prompt inside an OPEN <think> block. False (the
    /// default, and what the qwen3.8-flash-next goldens were captured with)
    /// closes it: `<|im_start|>assistant\n<think>\n\n</think>\n\n`.
    bool enable_thinking = false;
    /// Prefix every assistant history turn with the closed think block
    /// (`<think>\n\n</think>\n\n`) the 3.8 template always renders.
    bool history_think_block = false;
    /// Serialise tool definitions like jinja `tojson`
    /// (json.dumps(..., ensure_ascii=False) separators) instead of compact.
    bool jinja_tool_json = false;
    /// Strip leading/trailing ASCII whitespace from message content, matching
    /// the template's `render_content(...)|trim`.
    bool trim_content = false;
};

/// The policy the qwen4_exp (Qwen3.8-Flash-Next) model frames with: the 3.5/3.6
/// wire plus the three deltas its own `chat_template.jinja` carries (each one is
/// exercised in test_qwen_tool_framing against the template rendered by
/// transformers 5.15.1). Named and exported rather than file-local so the
/// choice is pinnable by a test -- a flag that is covered by a renderer test but
/// never switched on for the model frames bytes the reference never produces.
inline constexpr QwenRenderPolicy kQwen38RenderPolicy{
    .enable_thinking = false,  // the state the FSI/CODE/DET goldens and the
                               // native tool round-trip are captured in
    .history_think_block = true,
    .jinja_tool_json = true,
    .trim_content = true,
};

// Renders a conversation as the Qwen ChatML/XML transcript text. The
// generation prompt is the checkpoint's enable_thinking=false prefill
// (pre-closed empty think block) unless the policy keeps thinking open.
bool RenderQwenConversation(const ChatConversation &conv, std::string &out,
                            const QwenRenderPolicy &policy);
bool RenderQwenConversation(const ChatConversation &conv, std::string &out,
                            bool enable_thinking = false);

// Parses every `<tool_call>…<function=NAME>…<parameter=K>V</parameter>…`
// (and `<parameter=K>V</parameter>`) XML call in @p text.
std::vector<ToolCall> ParseQwenToolCallText(std::string_view text);

// True when the ChatML control sentinels are present in the vocabulary.
bool QwenChatSentinelsPresent(const ArrowTokenizerDb &tok);

// Appends @p text to @p out_ids, injecting the ChatML sentinels as their own
// vocab ids (arrow_tokenizer_encode is pure BPE and never resolves added
// tokens). Returns false (and clears out_ids) when a sentinel is missing.
bool AppendQwenChatIds(const ArrowTokenizerDb &tok, const std::string &text,
                       std::vector<int32_t> &out_ids);

// Decode generated ids → native XML parse → generic Hermes → bare-JSON
// fallback. Returns false only when decoding fails.
bool ParseQwenTokenCalls(const ArrowTokenizerDb &tok,
                         const std::vector<int32_t> &gen_tokens,
                         std::vector<ToolCall> &out_calls);

} // namespace lsxcommon
