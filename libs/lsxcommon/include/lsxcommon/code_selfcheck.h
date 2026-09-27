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
    std::string raw;
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
