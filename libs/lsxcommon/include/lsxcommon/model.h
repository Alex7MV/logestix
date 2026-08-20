#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "lsxcommon/inference_gpu.h"   // gpu::InferenceResult
#include "lsxcommon/safetensors.h"     // SafetensorsModel
#include "lsxcommon/tokenizer.h"       // ArrowTokenizerDb
#include "lsxcommon/benchmark_gpu.h"   // gpu::BenchmarkResult

namespace lsxcommon {

// Flat, host-only streaming token packet. Transmitted by const reference so no
// allocation or copy occurs on the model->caller hot path. Consumed by
// lsxcommon::InferenceEngine::Run between decode steps.
struct TokenChunk {
    uint32_t token_id = 0;      // generated token id
    uint32_t step_t = 0;        // decode step index (0-based)
    float    logit_maxv = 0.0f; // argmax logit of this step, pre-softmax
    bool     is_special = false; // true when the tokenizer flags the token special
};

// Zero-allocation streaming callback (no heap pressure per invocation).
using TokenCallback = std::function<void(const TokenChunk &)>;

struct ModelInitConfig {
    std::string model_path;   // required
    std::string model_type;   // empty = autodetect from logestix/model.index.json
    int max_gen_tokens = 64;
};

struct ModelRequest {
    std::vector<int32_t> input_ids;   // pre-tokenized prompt
    int max_gen_tokens = 64;
    // Streaming hook: invoked with each generated TokenChunk. Passed by const
    // reference (no store/copy). nullptr = streaming disabled.
    TokenCallback token_callback = nullptr;
    // Cooperative interruption intercept. When *abort_flag turns true during
    // generation, the engine loop stops publishing new chunks and the decode
    // run aborts. Never dereferenced when null.
    const std::atomic<bool> *abort_flag = nullptr;
};

class IModel {
public:
    virtual ~IModel() = default;

    virtual bool Load() = 0;
    virtual gpu::InferenceResult Infer(const ModelRequest &req) = 0;
    virtual void Unload() = 0;
    virtual SafetensorsModel Inspect() = 0;
    virtual gpu::InferenceResult SelfCheck(const ModelRequest &req) = 0;

    // Runs the model's own engine loop on synthetic tokens. Additive default:
    // ok=false keeps test fakes compiling without an override.
    virtual gpu::BenchmarkResult Benchmark(int num_tokens = 2048) { return {}; }

    // Peak resident weight set (pinned + arena + staged experts), in bytes.
    // Default 0 keeps the interface additive for test fakes.
    virtual std::uint64_t ResidentWeightBytes() const { return 0; }

    virtual const std::string &Type() const = 0;
    virtual ArrowTokenizerDb &Tokenizer() = 0;
};

} // namespace lsxcommon