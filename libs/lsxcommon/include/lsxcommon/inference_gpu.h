#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <absl/strings/ascii.h>

namespace lsxcommon {
namespace gpu {

struct InferenceConfig {
    std::string model_path;
    int max_gen_tokens = 64;
};

struct InferenceResult {
    // Timings (engine-recorded cudaEvent deltas; 0 when the run failed early).
    double ttft_ms = 0.0;            // stream start -> first generated token
    double total_time_ms = 0.0;      // stream start -> final token
    double tokens_per_sec = 0.0;     // num_tokens / (total_time_ms/1000)
    // Residency + occupancy (Item 10b).
    std::uint64_t weight_resident_bytes = 0;  // peak resident weight set over the run
    int32_t kv_slots_used = 0;       // KV occupancy at run end
    // Resident routed-expert cache capacity the run decoded with (0 = the
    // family has no streamed expert cache). For a streamed-expert family the
    // decode rate is a measured function of this number, so a harness that
    // records throughput has to record it too -- otherwise a run made with less
    // VRAM headroom (hence fewer slots) reads as a code regression.
    int32_t expert_cache_slots = 0;
    // EOS routing (Item 10c): true when the run stopped on the tokenizer EOS.
    bool eos_terminated = false;
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

    // Full generated token-id stream and the raw [vocab] logits row of the
    // final decode step (published by GLM-5.2 inference). Empty when the run
    // produced no output. The logits row lets self-check stress harnesses
    // verify bit-exact determinism across repeated runs.
    std::vector<int32_t> gen_tokens;
    std::vector<float> final_logits;
};

} // namespace gpu
} // namespace lsxcommon
