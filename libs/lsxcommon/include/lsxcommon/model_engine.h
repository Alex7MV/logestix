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
    // Sessions are reference-counted: nested guards share one engine lifetime,
    // and the engine shuts down only when the outermost guard exits.
    class Session {
    public:
        Session() { InferenceEngine::Instance().AcquireSession(); }
        ~Session() { InferenceEngine::Instance().ReleaseSession(); }
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
    // request's abort_flag before dispatch, then delegates to model.Infer.
    // The model owns streaming: it invokes req.token_callback once per decoded
    // token (and honors req.abort_flag between steps) on the calling thread.
    // The engine does not replay gen_tokens, so a token is never published
    // twice.
    gpu::InferenceResult Run(IModel &model, ModelRequest &req);

private:
    InferenceEngine() = default;
    ~InferenceEngine() = default;

    // Session lifetime counter: a Session increments on entry, decrements on
    // exit, and the engine shuts down on the last decrement. Kept separate
    // from initialized_ so a direct InitializeEngine()/ShutdownEngine() pair
    // keeps its unconditional toggle semantics.
    //
    // Mixing the two entry points is safe: a direct InitializeEngine() that
    // actually brings the engine up sets direct_owned_, and a Session release
    // then skips the shutdown so it cannot tear the engine down under the
    // direct caller. A repeated InitializeEngine() is a no-op and takes no
    // second claim, and ShutdownEngine() clears the claim unconditionally.
    void AcquireSession();
    void ReleaseSession();
    // Idempotent bring-up shared by InitializeEngine() and AcquireSession()
    // (governor init + perf policy + the ready flag). Deliberately does NOT
    // touch direct_owned_: only the public InitializeEngine() records a claim.
    void EnsureEngineUp();

    std::atomic<bool> initialized_{false};
    std::atomic<bool> direct_owned_{false};
    std::atomic<int> session_depth_{0};
};

} // namespace lsxcommon