#pragma once

#include <cstdint>
#include <string>

#include <absl/strings/ascii.h>

namespace lsxcommon {
namespace gpu {

struct InferenceConfig {
    std::string model_path;
    int max_gen_tokens = 64;
};

struct InferenceResult {
    double ttft_ms = 0.0;
    double tokens_per_sec = 0.0;
    double total_time_ms = 0.0;
    int num_tokens = 0;
    int num_layers = 0;
    float final_loss = 0.0f;
    std::string output_text;
    std::string error_msg;
    bool ok = false;

    // Single-token self-check diagnostics (filled on the first generated step).
    int   first_gen_tok_id = -1;
    std::string first_gen_text;
    float first_gen_maxv = 0.0f;   // argmax logit of the first token, pre-tempclamp
    bool  selfcheck_passed = false;
};

} // namespace gpu
} // namespace lsxcommon
