#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/inference_gpu.h"
#include "lsxcommon/model.h"
#include "lsxcommon/tokenizer.h"

namespace lsxcommon {
namespace runtime {

// Single error format for the Infer/Benchmark tails that receive both a
// cudaError_t status and a family generation-error string: family error wins
// on a clean CUDA status, the CUDA string alone otherwise, and both present
// join as "<cuda-string> / <family-error>" (lifecycle-unification 2026-08).
std::string CombineGenError(cudaError_t err, const std::string &gen_error);

// Item 2 — unified decode capacity-retry. Decodes a generated token stream
// into text. Grows the output buffer on the tokenizer's -2 (capacity too
// small) return; returns "" (with an spdlog warning) on persistent failure.
// Host-only: safe to call from .cu TUs.
std::string DecodeGeneratedText(const ArrowTokenizerDb &tok_db,
                                const std::vector<int32_t> &ids);

// Options for the unified Paris SelfCheck harness (model-unification
// 2026-08). Defaults reproduce the original strict contract exactly:
// force max(2, max_gen_tokens), decode the first generated token(s) and pass
// iff the stripped 1->2-token prefix equals "Paris".
struct ParisSelfCheckOptions {
    bool frame_prompt = false;   // rebuild input_ids from "The capital of
                                 // France is" via model.BuildPromptInputIds
    int min_tokens = 2;          // forced lower bound on max_gen_tokens
    int answer_window = 2;       // decode stripped prefixes of length
                                 // 1..min(answer_window, len) and require
                                 // equality; with substring_match, decode the
                                 // whole window instead
    bool substring_match = false;// pass iff decoded window CONTAINS "Paris"
                                 // instead of stripped equality
};

// Item 3 — unified SelfCheck. Forces max(opts.min_tokens, req.max_gen_tokens),
// delegates to model.Infer, then matches decoded prefixes against "Paris".
// Returns the InferenceResult with selfcheck_passed filled.
gpu::InferenceResult RunParisSelfCheck(IModel &model, const ModelRequest &req,
                                       const ParisSelfCheckOptions &opts = {});

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
    // Routes NVFP4 GEMMs through the native tensor-core path (sm_120a).
    // Tolerance-class numerics; off keeps the bit-exact scalar path.
    bool fast_math = false;
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