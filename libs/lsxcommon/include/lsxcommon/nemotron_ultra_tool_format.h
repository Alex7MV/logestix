#pragma once

// Nemotron-3-Ultra-550B conversation codec (host-only).
//
// Ultra's chat template is NOT the 30B sibling's: `chat_template.jinja` differs
// in sha (fa104c9a… vs a2445a9f…) and the difference is semantic, not cosmetic:
//   * a `medium_effort` switch appends `\n\n{reasoning effort: efficient}` to the
//     LAST user turn when enabled (default off, so the wire is unchanged);
//   * everything else — the `<|im_start|>` ChatML frame, the `# Tools` preamble
//     with a `<tools>` XML block and the `<IMPORTANT>` reminder, the
//     `<tool_call><function=…><parameter=…>` call form, the aggregated
//     `<tool_response>` turn, and the `<think>` generation prompt — is byte
//     compatible with the sibling, which is why the CALL PARSER is shared
//     (ParseQwenToolCallText) while rendering is Ultra's own.
//
// Rendering is verified against the checkpoint's own template in
// test_nemotron_ultra_tool_framing, not by reading it.

#include <string>
#include <vector>

#include "lsxcommon/chat_conversation.h"

namespace lsxcommon::nemotron_ultra {

struct UltraRenderPolicy {
    /// Leave the generation prompt inside an OPEN `<think>` block. This
    /// checkpoint's template defaults `enable_thinking` to TRUE, so the default
    /// here is true (the opposite of the qwen families' default).
    bool enable_thinking = true;
    /// Append `\n\n{reasoning effort: efficient}` to the last user turn. Off by
    /// default, as the template's own default.
    bool medium_effort = false;
    /// Replace the reasoning trace of assistant history turns that precede the
    /// last user turn with an empty closed block (template default true).
    bool truncate_history_thinking = true;
};

/// Renders `conv` into the exact text the checkpoint's template produces.
/// Returns false only when the conversation has no messages.
bool RenderUltraConversation(const ChatConversation &conv, std::string &out,
                             const UltraRenderPolicy &policy = {});

} // namespace lsxcommon::nemotron_ultra
