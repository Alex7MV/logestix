#pragma once

#include <atomic>
#include <memory>
#include <string>

#include "lsxcommon/model.h"

namespace lsxcommon {

// Strict, thread-safe Meyers singleton facade over the unified inference
// runtime. Wraps model creation (LsxModelFactory) and execution (Run) under
// one deterministic RAII lifecycle boundary.
//
// Lifecycle: InitializeEngine() -> [Run/CreateModel]* -> ShutdownEngine().
// The nested Session guard drives that lifecycle in a scope.
//
// HOST-STERILIZED: no CUDA/driver types leak through this header; safe to
// include from both plain C++ TUs and heavy .cu/.cuh units.
class InferenceEngine {
public:
    static InferenceEngine &Instance();

    InferenceEngine(const InferenceEngine &) = delete;
    InferenceEngine &operator=(const InferenceEngine &) = delete;
    InferenceEngine(InferenceEngine &&) = delete;
    InferenceEngine &operator=(InferenceEngine &&) = delete;

    // RAII lifecycle guard: InitializeEngine() on entry, ShutdownEngine() on
    // scope exit. Mirrors Core::Initialize/Core::Shutdown for embedded use.
    class Session {
    public:
        Session() { InferenceEngine::Instance().InitializeEngine(); }
        ~Session() { InferenceEngine::Instance().ShutdownEngine(); }
        Session(const Session &) = delete;
        Session &operator=(const Session &) = delete;
    };

    bool InitializeEngine();
    void ShutdownEngine();
    bool IsInitialized() const noexcept { return initialized_.load(std::memory_order_acquire); }

    // Open a model through the unified factory (nullptr when the type is
    // unknown or the engine is not initialized).
    std::unique_ptr<IModel> CreateModel(const ModelInitConfig &cfg);

    // Unified inference boundary. Validates the engine lifecycle, checks the
    // request's abort_flag before dispatch, performs the decode through
    // model.Infer, and streams the generated tokens out via
    // req.token_callback as TokenChunk events. Abort and the caller's
    // callback are read/called on the calling thread (no internal threads).
    gpu::InferenceResult Run(IModel &model, ModelRequest &req);

private:
    InferenceEngine() = default;
    ~InferenceEngine() = default;

    std::atomic<bool> initialized_{false};
};

} // namespace lsxcommon