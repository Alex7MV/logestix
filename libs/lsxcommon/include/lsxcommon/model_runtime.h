#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/inference_gpu.h"
#include "lsxcommon/model.h"
#include "lsxcommon/tokenizer.h"

namespace lsxcommon {
namespace runtime {

// Item 2 — unified decode capacity-retry. Decodes a generated token stream
// into text. Grows the output buffer on the tokenizer's -2 (capacity too
// small) return; returns "" (with an spdlog warning) on persistent failure.
// Host-only: safe to call from .cu TUs.
std::string DecodeGeneratedText(const ArrowTokenizerDb &tok_db,
                                const std::vector<int32_t> &ids);

// Item 3 — unified SelfCheck. Forces max(2, req.max_gen_tokens), delegates to
// model.Infer, then matches the decoded 1->2-token prefix against "Paris".
// Returns the InferenceResult with selfcheck_passed filled.
gpu::InferenceResult RunParisSelfCheck(IModel &model, const ModelRequest &req);

// Non-owning view over the five same-shaped per-family XxxGenerationResult
// structs (tokens, final_logits, ttft_ms, total_ms, error_msg,
// eos_terminated, kv occupancy, first-token logit, resident bytes). Lets a
// single mapping helper serve every model family.
struct GenResultView {
    const std::vector<int32_t> *tokens;
    const std::vector<float> *final_logits;  // may be null (family lacks it)
    double ttft_ms;
    double total_ms;
    const std::string *error_msg;            // may be null
    bool eos_terminated;
    int kv_slots_used;                       // -1 => caller computes fallback
    float first_maxv;                        // 0 when the family lacks it
    std::uint64_t resident_bytes;
};

// Maps a family generation result into InferenceResult (ok, num_tokens,
// gen_tokens/final_logits copies, timings, t/s, eos, occupancy, residency).
// Does NOT decode output_text and does NOT touch per-model session state —
// callers own both. `num_layers` stamps r.num_layers.
void FillInferenceResult(gpu::InferenceResult &r, const GenResultView &g,
                         int num_layers);

// Shared Benchmark result tail: ok + tokens/timing/occupancy mapping.
void FillBenchmarkResult(gpu::BenchmarkResult &r, double total_ms,
                         int num_tokens, int num_layers,
                         std::uint64_t resident_bytes);

struct GenerationCliArgs {
    std::string model_path;
    std::string prompt;
    int max_gen_tokens = 64;
};

// Item 6 — unified CLI lifecycle. Each helper wraps
// Create -> Load -> frame/encode -> run -> print -> Unload and returns a
// process exit code. The caller owns Core::Initialize/Shutdown.
int RunCliGeneration(const GenerationCliArgs &args);
int RunCliSelfCheck(const GenerationCliArgs &args);
int RunCliInspect(const std::string &model_path, bool full);
void PrintGenerationResult(const std::string &prompt, const gpu::InferenceResult &r);

} // namespace runtime
} // namespace lsxcommon