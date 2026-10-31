#pragma once

#include <string>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/tokenizer.h"

namespace lsxcommon::beam {

// Native Beam (modified Harmony / `harmony_xml`) conversation renderer. The
// wire format is pinned by the checkpoint's chat_template.jinja:
//
//   <|start|>system<|message|>{identity}\n\nReasoning effort: 512
//   \n\n# Valid channels: analysis, final. Channel must be included for every
//   message.<|end|>
//   <|start|>developer<|message|># Instructions\n\n{...}<|end|>
//   <|start|>user<|message|>{...}<|end|>
//   <|start|>assistant<|channel|>analysis<|message|>{...}<|end|>
//   <|start|>assistant to={name}<|channel|>analysis <|constrain|>xml<|message|>
//     <rfl:parameter name="k">\nv\n</rfl:parameter><|call|>
//   <|start|>assistant<|channel|>final<|message|>{...}<|return|>
//   <|start|>{tool} to=assistant<|channel|>analysis<|message|>{...}<|end|>
//   <|start|>assistant            (generation prompt)
//
// The Harmony markers are PLAIN (special:false) added tokens, so the byte-level
// encoder resolves them literally and no id injection is needed.
bool RenderBeamConversation(const ArrowTokenizerDb &tok,
                            const ChatConversation &conv,
                            std::vector<int32_t> &out_ids);

// Parses the assistant `to={name}` / `<rfl:parameter>` tool-call bodies out of
// a decoded token stream. Returns false when no call was found.
bool ParseBeamToolCalls(const ArrowTokenizerDb &tok,
                        const std::vector<int32_t> &gen_tokens,
                        std::vector<ToolCall> &out_calls);

} // namespace lsxcommon::beam
