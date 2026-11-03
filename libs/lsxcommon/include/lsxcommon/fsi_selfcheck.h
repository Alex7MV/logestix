#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "lsxcommon/model.h"

namespace lsxcommon {
namespace fsi {

enum class FsiKind { kArithmetic, kComparison };

struct FsiGoldenCase {
    std::string id;
    FsiKind kind = FsiKind::kArithmetic;
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
    // A model that reasons INLINE (no [THINK] markers, so nothing can be cut
    // by id) states the answer last. When true the extractor scans the lines
    // from the end, so a line from the reasoning chain cannot shadow it.
    bool prefer_last_answer = false;
    std::string raw;
};

struct FsiGolden {
    std::string model_type;
    std::vector<FsiGoldenCase> cases;
};

struct FsiCaseResult {
    std::string id;
    std::string expected;
    std::string extracted;
    std::string raw;   // decoded generated text (diagnostic)
    bool passed = false;
    std::string error;
};

struct FsiResult {
    std::vector<FsiCaseResult> cases;
    bool all_passed = false;
    std::string error;
};

// `prefer_last` keeps the LAST matching answer segment: a reasoning model's
// chain of thought can contain competing formulas before the final answer.
std::string ExtractFsiFormula(FsiKind kind, std::string_view anchor,
                              std::string_view raw, bool prefer_last = false);

bool ParseFsiGolden(std::string_view json, FsiGolden &out, std::string *error);
bool LoadFsiGoldenFile(const std::string &path, FsiGolden &out,
                       std::string *error);

using FsiDecodeFn = std::function<std::string(const std::vector<int32_t> &)>;

FsiResult RunFsiSelfCheck(IModel &model, const FsiGolden &golden);
FsiResult RunFsiSelfCheck(IModel &model, const FsiGolden &golden,
                          const FsiDecodeFn &decode);

}  // namespace fsi
}  // namespace lsxcommon
