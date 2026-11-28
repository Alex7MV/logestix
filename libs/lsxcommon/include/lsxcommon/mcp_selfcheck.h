#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "lsxcommon/model.h"

namespace lsxcommon {
namespace mcp {

// The three protocol shapes the prompt set exercises.
enum class McpKind { kToolCall, kResourceRead, kErrorResponse };

const char *McpKindName(McpKind kind);

struct McpGoldenCase {
    std::string id;
    McpKind kind = McpKind::kToolCall;
    // A substring the payload must contain (e.g. "tools/call"). It is what
    // selects the payload object out of surrounding prose; empty disables the
    // filter and the first complete object wins.
    std::string anchor;
    std::vector<int32_t> input_ids;
    int max_new_tokens = 512;   // reasoning models burn tokens before the payload
    bool ignore_eos = false;
    // Reasoning models: id of the marker that ENDS the thinking block, dropped
    // by the decoder. When >= 0 only the tokens after its LAST occurrence are
    // decoded, so the final payload — not a line from the reasoning chain — is
    // what gets extracted. -1 keeps the historical behaviour.
    int32_t answer_start_token_id = -1;
    // The canonical JSON-RPC 2.0 payload. This one is PROTOCOL-defined, not
    // model-defined (see the design doc §3): a request has jsonrpc/method/
    // params/id and JSON declares key order insignificant, so the gate compares
    // documents, not bytes.
    std::string expected;
    // Diagnostic only: empty for the tokenizer-only capture this gate ships
    // with, which test_model_suite_guard cross-checks against `source`.
    std::string raw;
    // Optional prompt body: a model that echoes the prompt after its answer is
    // cut at the body's prefix before extraction.
    std::string body;
    // ATEM-style channel framing (Muse-Glimmer): see CodeGoldenCase.
    std::string answer_channel;
    std::string reasoning_channel;
};

struct McpGolden {
    std::string model_type;
    std::vector<McpGoldenCase> cases;
};

struct McpCaseResult {
    std::string id;
    std::string expected;    // canonical form of the golden payload
    std::string extracted;   // canonical form of what the model emitted ("" if none)
    // Whether the extracted TEXT equals the golden template byte for byte, i.e.
    // whether the model reproduced the template's key order and formatting.
    // Recorded, logged, and deliberately NOT part of `passed`: the protocol
    // does not require it, and failing on it would measure layout-luck.
    bool exact = false;
    std::string raw;         // decoded generated text (diagnostic)
    std::string diff;        // first document difference, "" when equal
    bool passed = false;
    std::string error;
};

struct McpResult {
    std::vector<McpCaseResult> cases;
    bool all_passed = false;
    std::string error;
};

// Extracts the JSON-RPC payload from a decoded answer: fence-strip, drop the
// reasoning chain, cut a prompt echo, then take the first complete `{...}` that
// carries `anchor` and minify it. Empty when there is no such object.
std::string ExtractMcpAnswer(McpKind kind, std::string_view anchor,
                             std::string_view raw);

// Canonical JSON: object keys sorted, no insignificant whitespace, strings
// escaped the same way on both sides. False (with a reason) when `json` is not
// a JSON document.
bool CanonicalizeJson(std::string_view json, std::string *canonical,
                      std::string *error);

// The first difference between two JSON documents, as a short one-line
// sentence ("error.code: -32601 != -32602", "params: missing key arguments").
// Empty string means the documents are EQUAL under canonicalization — which is
// the gate's criterion.
std::string JsonDocumentDiff(std::string_view expected, std::string_view got);

bool ParseMcpGolden(std::string_view json, McpGolden &out, std::string *error);
bool LoadMcpGoldenFile(const std::string &path, McpGolden &out, std::string *error);

using McpDecodeFn = std::function<std::string(const std::vector<int32_t> &)>;

McpResult RunMcpSelfCheck(IModel &model, const McpGolden &golden);
McpResult RunMcpSelfCheck(IModel &model, const McpGolden &golden,
                          const McpDecodeFn &decode);

} // namespace mcp
} // namespace lsxcommon
