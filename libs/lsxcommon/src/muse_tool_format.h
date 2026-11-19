#pragma once

#include <string>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/tokenizer.h"

namespace lsxcommon::muse {

// Muse-Glimmer-30B native ATEM chat/tool codec (checkpoint chat_template.jinja).
//
// Framing (the <|…|> markers are SPECIAL vocabulary ids injected literally;
// the ATEM XML tags are plain text):
//   <|begin_of_text|>
//   <|start|>system<|message|>{system}<|eot|>        (default system turn when
//       the caller supplies none: "You are a helpful AI assistant.\nKnowledge
//       cutoff: 2026-01-04.[\nCurrent date: …]\n\nReasoning strength: high.
//       [\n\n{tool defs}]\n\n# Valid recipients: …")
//   <|start|>user<|message|>{content}<|eot|>
//   <|start|>assistant to=self<|message|>{reasoning}<|eom|>   (reasoning turn)
//   <|start|>assistant to={fn}<|message|><atem:function_calls>\n
//       <atem:invoke name="{fn}">\n
//       <atem:parameter name="{k}">{v}</atem:parameter>\n…
//       </atem:invoke>\n</atem:function_calls><|eot|>          (tool call)
//   <|start|>assistant to=user<|message|>{content}<|eot|>      (answer)
//   <|start|>tool {fn}<|message|><tool_output name="{fn}">\n{content}\n
//       </tool_output><|eot|>                                  (tool result)
//   <|start|>assistant                                         (generation prompt)
//
// Between two same-role messages the template closes with <|eom|> instead of
// <|eot|>. Tool parameter values: strings/scalars raw, bool/null literals,
// lists/objects as Python-tojson JSON.
bool MuseChatSentinelsPresent(const ArrowTokenizerDb &tok);

// Per-render template knobs. `current_date` mirrors the jinja's optional
// `current_date` kwarg (empty = omit the line, deterministic renders);
// `reasoning_strength` mirrors `reasoning_strength` (default "high").
struct MuseRenderPolicy {
    std::string current_date;
    std::string reasoning_strength = "high";
    std::string knowledge_cutoff = "2026-01-04";
};

// Renders a conversation into the checkpoint's exact token framing, with the
// generation prompt `<|start|>assistant` appended.
bool RenderMuseConversation(const ArrowTokenizerDb &tok,
                            const ChatConversation &conv,
                            std::vector<int32_t> &out_ids);
bool RenderMuseConversation(const ArrowTokenizerDb &tok,
                            const ChatConversation &conv,
                            std::vector<int32_t> &out_ids,
                            const MuseRenderPolicy &policy);

// Frames a single raw prompt as one user turn (default system + user +
// generation prompt) — the model's BuildPromptInputIds path.
bool MusePromptInputIds(const ArrowTokenizerDb &tok, std::string_view raw_prompt,
                        std::vector<int32_t> &out_ids);

// Parses native ATEM tool calls out of the generated tokens. The channel
// markers are special:true (decode drops them), so the parse scans token ids
// for `<|start|>assistant to=<fn><|message|>` channels and reads the ATEM XML
// between them; falls back to the generic Hermes/bare-JSON parsers when no
// native channel is present.
bool ParseMuseToolCalls(const ArrowTokenizerDb &tok,
                        const std::vector<int32_t> &gen_tokens,
                        std::vector<ToolCall> &out_calls);

// Renders/parses one ATEM function_calls block (exposed for unit tests).
std::string RenderMuseToolCallText(const ToolCall &call);
std::vector<ToolCall> ParseMuseAtemText(const std::string &text);

} // namespace lsxcommon::muse