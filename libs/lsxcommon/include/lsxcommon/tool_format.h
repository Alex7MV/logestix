#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <simdjson.h>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/tokenizer.h"

namespace lsxcommon {
namespace tools {

/// Renders the tool schemas as a Hermes-style JSON block:
///   <tools>
///   [{"type":"function","function":{"name","description","parameters"}}]
///   </tools>
std::string RenderToolsBlock(const std::vector<ToolSpec>& tools);

/// Renders a conversation as a role-marked transcript. When @p include_tools
/// and tools are present, the block is prepended as a system section.
std::string RenderConversation(const ChatConversation& conv, bool include_tools);

/// Extracts every `<tool_call>…</tool_call>` JSON object from decoded text.
/// `arguments` may be a JSON object/array (normalized to compact JSON) or an
/// already-encoded string (kept verbatim). Ids are assigned `call_0`, `call_1`, …
std::vector<ToolCall> ParseToolCallsText(std::string_view text);

/// Last-resort parser for a bare (unwrapped) Hermes-style call object found in
/// decoded text, e.g. {"name":"f","arguments":{…}} with no <tool_call> wrapper.
/// Some checkpoints emit the object directly after a reasoning block.
std::vector<ToolCall> ParseBareToolCallJson(std::string_view text);

/// Inserts a zero-width space (U+200B) after the first byte of every occurrence
/// of each control marker in @p text, so untrusted message content can never
/// forge a chat template's control tokens when the transcript is assembled.
/// Pass the same marker set the family's id-builder scans for.
std::string NeutralizeControlMarkers(
    std::string_view text, const std::vector<std::string_view> &markers);

// ── Shared JSON/framing primitives (dedup phase 4) ──────────────────────
// Moved verbatim from the byte-identical per-family copies. The three escape
// semantics are DISTINCT and must not be merged into one another:
//   * PyJson*  — Python json.dumps(ensure_ascii=True) shape (flint/hy/muse).
//   * SerializeJson — compact JSON via lsxnetwork JsonEscape, %.17g doubles
//     (deepseek/gemma/gigachat).
//   * AppendJinjaString — Jinja tojson (dory/qwen).
// Goldens: tests/test_tool_format_json.cpp.

/// Escapes one string as a Python json.dumps(ensure_ascii=True) literal.
void PyJsonString(std::string_view s, std::string &out);
/// Serializes a simdjson DOM element in the Python tojson shape.
void PyJsonValue(const simdjson::dom::element &v, std::string &out);
/// Parses @p json and re-emits it in the Python tojson shape; {} on parse error.
std::string PyJsonDumps(std::string_view json);

/// Compact JSON serializer (JsonEscape, %.17g doubles) — the deepseek/gemma/
/// gigachat tool-format shape. NOT the Python shape above.
std::string SerializeJson(simdjson::dom::element e);

/// Jinja `tojson` string escaping: control chars \u00XX, everything else
/// (including UTF-8 and HTML-significant bytes) verbatim.
void AppendJinjaString(std::string &out, std::string_view s);

/// Decodes generated token ids to text with the shared retry-doubling buffer
/// (4096 start, 8 attempts, cap 1<<20) and hands the text to @p parse.
/// Returns false on decode failure; empty gen_tokens is a success with no
/// calls. The kimi-k3/k3.1-pro ParseToolCalls copy (dedup Ф5 residual).
bool DecodeAndParseToolCalls(const ArrowTokenizerDb &tok,
                             const std::vector<int32_t> &gen_tokens,
                             std::vector<ToolCall> &out_calls,
                             std::vector<ToolCall> (*parse)(std::string_view));

/// BPE-encodes @p text and appends the ids, stripping an auto-prepended BOS
/// only when actually present. Retry-doubling buffer up to 65536 ids
/// (the glm/granite/granite-switch/mimo/qwen copy).
bool AppendEncoded(const ArrowTokenizerDb &tok, std::string_view text,
                   std::vector<int32_t> &ids);
/// Same, empty-text fast path and buffer up to 1<<20 ids
/// (the hy/minimax/muse/step copy — a distinct body, kept distinct).
bool AppendEncodedLarge(const ArrowTokenizerDb &tok, std::string_view text,
                        std::vector<int32_t> &ids);

/// Splits @p text on @p sentinels and injects each marker as its own token id;
/// the runs between markers go through AppendEncodedLarge. The sentinel table
/// stays family-owned (passed in); the body is the shared flint/hy/minimax/
/// step copy.
bool AppendTextWithSentinels(const ArrowTokenizerDb &tok,
                             const std::string &text,
                             const std::vector<std::string_view> &sentinels,
                             std::vector<int32_t> &ids);

/// Appends ids for a segment list: control segments become single token ids,
/// text segments are BPE-encoded with a per-segment sized buffer. Templated on
/// the family's segment struct ({std::string text; bool is_control;}) — the
/// inkling pair's bodies were identical modulo the struct name.
template <class Segment>
bool AppendSegmentIds(const ArrowTokenizerDb &tok,
                      const std::vector<Segment> &segs,
                      std::vector<int32_t> &out_ids)
{
    for (const auto &s : segs) {
        if (s.text.empty()) continue;
        if (s.is_control) {
            const int32_t id = arrow_tokenizer_lookup(tok, s.text.data(),
                                                      s.text.size());
            if (id < 0) return false;
            out_ids.push_back(id);
            continue;
        }
        std::vector<int32_t> tmp(s.text.size() * 4 + 64, 0);
        std::size_t cap = tmp.size();
        const int32_t n = arrow_tokenizer_encode(tok, s.text.data(),
                                                 s.text.size(), tmp.data(), &cap);
        if (n < 0) return false;
        out_ids.insert(out_ids.end(), tmp.begin(),
                       tmp.begin() + static_cast<std::size_t>(n));
    }
    return true;
}

}  // namespace tools
}  // namespace lsxcommon
