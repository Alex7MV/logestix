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

std::string ExtractFsiFormula(FsiKind kind, std::string_view anchor,
                              std::string_view raw);

bool ParseFsiGolden(std::string_view json, FsiGolden &out, std::string *error);
bool LoadFsiGoldenFile(const std::string &path, FsiGolden &out,
                       std::string *error);

using FsiDecodeFn = std::function<std::string(const std::vector<int32_t> &)>;

FsiResult RunFsiSelfCheck(IModel &model, const FsiGolden &golden);
FsiResult RunFsiSelfCheck(IModel &model, const FsiGolden &golden,
                          const FsiDecodeFn &decode);

}  // namespace fsi
}  // namespace lsxcommon
