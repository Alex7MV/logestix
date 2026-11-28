// selfcheck_common.h — helpers shared by the FSI/CODE/DET/MCP parity
// harnesses (dedup phase 5). Unfence was a byte-identical copy in
// fsi_selfcheck.cpp, code_selfcheck.cpp and det_selfcheck.cpp.
#pragma once

#include <functional>
#include <string>
#include <string_view>

namespace lsxcommon {
namespace selfcheck {

// Strips surrounding whitespace and one markdown code fence (```lang ... ```),
// keeping the inner content. An unterminated fence keeps everything after the
// opener; a first line that is not a bare language tag stays in the content.
std::string Unfence(std::string_view raw);

// Fence-strips and then drops everything before the LAST reasoning close, so a
// chain of thought cannot shadow the final answer. Two closers are recognised:
// the `</think>` tag and the `[COT_END]` literal (alice). A payload with
// neither is returned unfenced, unchanged — which is what keeps this a no-op
// for the models that never emit a reasoning block.
std::string StripBeforeFinalAnswer(std::string_view raw);

// Cuts a prompt ECHO after the answer. A fragile reasoning model can fail to
// emit a turn-end token and repeat the prompt; when the golden carries the
// prompt body, everything from the body's first 32 characters onwards is
// dropped (after the reasoning close). An empty body cuts nothing.
std::string CutPromptEcho(std::string_view text, std::string_view body);

// The first COMPLETE balanced `{...}` object in `text` that contains `anchor`,
// returned minified (all whitespace outside string literals removed). Empty
// when no complete object carries the anchor.
//
// `accept`, when set, has the final say on a candidate: the FIRST anchored
// object it accepts wins, and a rejected candidate only lets the scan continue.
// That is how a caller asks for a shape ("this one must be a request") without
// re-implementing the scan.
//
// Scanning for the anchor *across* candidate objects, rather than taking the
// first object and then requiring the anchor, is what makes a model that
// prefixes its payload with a stray `{}` or a one-key example still passable:
// the old rule extracted the stray object and then failed on the anchor check,
// so a correct payload behind it was reported as an empty extraction.
std::string ExtractMinifiedJson(
    std::string_view text, std::string_view anchor,
    const std::function<bool(std::string_view)> &accept = {});

// Escapes a decoded model output so one verdict stays ONE log record. A bare
// newline inside a spdlog message silently truncates it for every
// line-oriented reader (`grep 'raw=.*'` returns the first line only), which is
// how a real `BIT_ERO` answer once read as a clean `BIT_ZERO`. Bytes >= 0x80
// pass through: UTF-8 output is legitimate and escaping it would make every
// non-English model unreadable.
std::string EscapeForLog(std::string_view in);

} // namespace selfcheck
} // namespace lsxcommon
