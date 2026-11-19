#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "lsxcommon/model.h"

namespace lsxcommon {
namespace det {

enum class DetKind { kToken };

struct DetGoldenCase {
    std::string id;
    DetKind kind = DetKind::kToken;
    std::vector<int32_t> input_ids;
    int max_new_tokens = 64;
    bool ignore_eos = false;   // run the full token budget (no EOS stop)
    int stress_runs = 3;       // determinism repetition count (floored at 3)
    std::string expected;
    // Reasoning models: the id of the marker that ENDS the thinking block
    // (e.g. Mistral-Large-4's [THINK]==34 / [/THINK]==35, which are special
    // tokens and therefore dropped by the decoder). When >= 0 the harness
    // decodes only the tokens AFTER its LAST occurrence, so the final answer —
    // not a line from the reasoning chain — is what gets extracted. -1 keeps
    // the historical behaviour.
    int32_t answer_start_token_id = -1;
    std::string raw;           // diagnostic
    std::string body;          // optional prompt body for the echo cut
    // ATEM-style channel framing (Muse-Glimmer): see CodeGoldenCase. When set,
    // the harness selects the FIRST complete answer channel.
    std::string answer_channel;
    std::string reasoning_channel;
};

struct DetGolden {
    std::string model_type;
    std::vector<DetGoldenCase> cases;
};

struct DetCaseResult {
    std::string id;
    std::string expected;
    std::string extracted;
    std::string raw;             // decoded run-0 text (diagnostic)
    int runs = 0;                // effective R
    bool deterministic = false;  // all runs' gen_tokens identical
    bool passed = false;
    std::string error;
};

struct DetResult {
    std::vector<DetCaseResult> cases;
    bool all_passed = false;
    std::string error;
};

std::string ExtractDetAnswer(DetKind kind, std::string_view raw);

bool ParseDetGolden(std::string_view json, DetGolden &out, std::string *error);
bool LoadDetGoldenFile(const std::string &path, DetGolden &out,
                       std::string *error);

using DetDecodeFn = std::function<std::string(const std::vector<int32_t> &)>;

DetResult RunDetSelfCheck(IModel &model, const DetGolden &golden);
DetResult RunDetSelfCheck(IModel &model, const DetGolden &golden,
                          const DetDecodeFn &decode,
                          int stress_runs_override = 0);

}  // namespace det
}  // namespace lsxcommon
