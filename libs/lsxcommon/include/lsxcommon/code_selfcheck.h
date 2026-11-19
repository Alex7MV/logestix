#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "lsxcommon/model.h"

namespace lsxcommon {
namespace code {

enum class CodeKind { kJson, kAssignment, kRegex };

struct CodeGoldenCase {
    std::string id;
    CodeKind kind = CodeKind::kJson;
    std::string anchor;
    std::vector<int32_t> input_ids;
    int max_new_tokens = 128;
    bool ignore_eos = false;   // run the full token budget (no EOS stop)
    std::string expected;
    // Reasoning models: the id of the marker that ENDS the thinking block
    // (e.g. Mistral-Large-4's [THINK]==34 / [/THINK]==35, which are special
    // tokens and therefore dropped by the decoder). When >= 0 the harness
    // decodes only the tokens AFTER its LAST occurrence, so the final answer —
    // not a line from the reasoning chain — is what gets extracted. -1 keeps
    // the historical behaviour.
    int32_t answer_start_token_id = -1;
    std::string raw;
    // Optional prompt body: if the model echoes the prompt after its answer
    // (degeneration), the harness cuts the echo at the body's prefix before
    // extraction. Empty -> no cut.
    std::string body;
    // ATEM-style channel framing (Muse-Glimmer): the answer lives in a channel
    // whose decoded header is `answer_channel` (e.g. "assistant to=user"), and
    // reasoning in `reasoning_channel` ("assistant to=self"). The channel
    // markers are special tokens the decoder drops, so the headers survive as
    // literal text. When set, the harness selects the FIRST complete answer
    // channel (the last can be truncated by the token budget) instead of the
    // raw tail. Empty -> historical behaviour.
    std::string answer_channel;
    std::string reasoning_channel;
};

struct CodeGolden {
    std::string model_type;
    std::vector<CodeGoldenCase> cases;
};

struct CodeCaseResult {
    std::string id;
    std::string expected;
    std::string extracted;
    std::string raw;   // decoded generated text (diagnostic)
    bool passed = false;
    std::string error;
};

struct CodeResult {
    std::vector<CodeCaseResult> cases;
    bool all_passed = false;
    std::string error;
};

std::string ExtractCodeAnswer(CodeKind kind, std::string_view anchor,
                              std::string_view raw);

bool ParseCodeGolden(std::string_view json, CodeGolden &out, std::string *error);
bool LoadCodeGoldenFile(const std::string &path, CodeGolden &out,
                        std::string *error);

using CodeDecodeFn = std::function<std::string(const std::vector<int32_t> &)>;

CodeResult RunCodeSelfCheck(IModel &model, const CodeGolden &golden);
CodeResult RunCodeSelfCheck(IModel &model, const CodeGolden &golden,
                            const CodeDecodeFn &decode);

}  // namespace code
}  // namespace lsxcommon
