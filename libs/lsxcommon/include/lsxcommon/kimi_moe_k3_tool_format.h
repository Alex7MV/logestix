#pragma once

// Kimi-K3 native XTML conversation + tool-call codec (host-only).
//
// Grammar transcribed from the checkpoint's encoding_k3.py (build_chat_segments
// / _open_tag / _close_tag), which is also what kimi_k3_processor.py feeds the
// model. There is no chat_template.jinja in this checkpoint.
//
//   open tag    <|open|>{tag}{ attrs}<|sep|>          attrs = ' k="v"'
//   close tag   <|close|>{tag}<|sep|>
//   message end <|end_of_msg|>
//
//   tools       <|open|>message role="system" type="tool-declare"<|sep|>
//               # Tools\nHere are the available tools, described in JSONSchema.
//               \n\n```json\n{compact tools}\n```<|close|>message<|sep|><|end_of_msg|>
//   system      <|open|>message role="system"<|sep|>{text}<|close|>message<|sep|><|end_of_msg|>
//   user        <|open|>message role="user"<|sep|>{text}<|close|>message<|sep|><|end_of_msg|>
//   assistant   <|open|>message role="assistant"<|sep|>
//               <|open|>think<|sep|>{reasoning}<|close|>think<|sep|>
//               <|open|>response<|sep|>{content}<|close|>response<|sep|>
//               [<|open|>tools<|sep|>{calls}<|close|>tools<|sep|>]
//               <|close|>message<|sep|><|end_of_msg|>
//   call        <|open|>call tool="{name}" index="{i}"<|sep|>{args}
//               <|close|>call<|sep|>
//   argument    <|open|>argument key="{k}" type="{t}"<|sep|>{value}
//               <|close|>argument<|sep|>
//   json args   <|open|>json type="object"<|sep|>{raw json}<|close|>json<|sep|>
//   tool result <|open|>message role="tool" tool="{name}" index="{i}"<|sep|>
//               {text}<|close|>message<|sep|><|end_of_msg|>
//   generation  <|open|>message role="assistant"<|sep|><|open|>think<|sep|>
//
// <|open|>/<|close|>/<|sep|> are tiktoken-special (matched literally) but
// survive decode (HF marks them special:false), so tool calls parse from the
// decoded text; <|end_of_msg|> is the EOS and is dropped by decode.
//
// The reference encodes each EncodeSegment separately: control markers as their
// own ids and free text as ordinary BPE (never resolving marker spellings that
// appear inside user text). That is why the renderer emits segments instead of
// a flat string.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/tokenizer.h"

namespace lsxcommon {

// One render segment: `is_control` segments are XTML markers and are injected
// verbatim as vocabulary ids; the rest is free text and is BPE-encoded.
struct KimiK3Segment {
    std::string text;
    bool is_control = false;
};

// Renders the conversation into XTML segments (tools preamble, turns, tool
// results and the thinking generation prompt). Returns false for an unknown
// role.
bool RenderKimiK3Conversation(const ChatConversation &conv,
                              std::vector<KimiK3Segment> &out);

// Concatenates the segments (markers included) — the wire text the model sees.
std::string KimiK3SegmentsText(const std::vector<KimiK3Segment> &segments);

// Appends the segments to @p out_ids: control markers become their own ids,
// free text is BPE-encoded. Returns false when a marker is missing from the
// vocabulary or the encoder fails.
bool AppendKimiK3Ids(const ArrowTokenizerDb &tok,
                     const std::vector<KimiK3Segment> &segments,
                     std::vector<int32_t> &out_ids);

// Parses every native XTML tool-call block in the decoded text; falls back to
// the generic Hermes parser and then the bare-object parser when no native
// block is present.
std::vector<ToolCall> ParseKimiK3ToolCalls(std::string_view text);

// Sorted-key compact JSON (encoding_k3.py's deep_sort + separators=(",",":")).
// Returns false on malformed input (the caller then passes the raw text).
bool KimiK3CanonicalJson(std::string_view json, std::string &out);

}  // namespace lsxcommon
