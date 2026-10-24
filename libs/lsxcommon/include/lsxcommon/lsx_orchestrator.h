#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <absl/container/flat_hash_map.h>

#include "lsxcommon/model.h"          // IModel
#include "lsxnetwork/frame_queue.h"   // lsxnetwork::FrameQueue
#include "lsxnetwork/openai_endpoint.h" // lsxnetwork::ChatCompletion

namespace lsxcommon {

/// Per-frame dispatch verdict returned by ServeTaskMosaic.
enum class DispatchSignal {
    kGlm52,          ///< routed to the "glm-5.2" branch
    kDeepSeekFlash,  ///< routed to the "deepseek-v4-flash" branch
    kUnknown,        ///< model string did not match a registered branch
};

/// Shared, per-branch counters recorded by the serve loop. The integration
/// harness uses these to assert zero token-drift (taken == retired per branch).
/// Counters are atomic because the serve loop runs on a dedicated consumer
/// thread while the harness observes them from the producer/asserting thread.
struct TaskMosaicStats {
    std::atomic<std::uint64_t> taken_frames  {0};  ///< removed from ring via TryTake
    std::atomic<std::uint64_t> glm_routed    {0};  ///< dispatched to GLM-5.2 branch
    std::atomic<std::uint64_t> deepseek_routed{0}; ///< dispatched to DeepSeek branch
    std::atomic<std::uint64_t> unknown_routed{0};  ///< unmapped model string
    std::atomic<std::uint64_t> retired_slots {0};  ///< slots released to producer
    std::atomic<std::uint64_t> dispatch_exceptions{0}; ///< IModel::Infer/Parse threw
    std::atomic<std::uint64_t> infer_failures{0};  ///< IModel::Infer returned !ok
    std::atomic<std::uint64_t> responses_ok{0};           ///< responses published to ring
    std::atomic<std::uint64_t> responses_failed{0};       ///< ring full / frame too large
    std::atomic<std::uint64_t> response_serialize_failures{0}; ///< WriteCompletion OOM
};

/// Multi-model core orchestration layer for the asynchronous TaskMosaic queue
/// pipeline.
///
/// `ServeTaskMosaic` is a headless consumer loop that polls the lock-free SPSC
/// `FrameQueue` tail watermark (via `FrameQueue::TryTake`, never a blocking host
/// sync fence) and dispatches each frame to exactly one model branch. The
/// zero-copy retirement chain guarantees a ring slot is released (and its
/// `tail` advanced) only after the target model's CUDA execution step has fully
/// returned.
class LsxOrchestrator {
public:
    /// Model dispatch table. Keys are the model type strings (e.g.
    /// "glm-5.2", "deepseek-v4-flash"); values are caller-owned model
    /// instances. Ownership is NOT transferred: the orchestrator only reads /
    /// invokes them and releases ring slots after each call returns.
    using ModelTable = absl::flat_hash_map<std::string, lsxcommon::IModel*>;

    explicit LsxOrchestrator(const ModelTable& models);

    /// Selects the target branch from a parsed request's `model` field.
    static DispatchSignal Classify(std::string_view model) noexcept;

    /// Serves frames until @p stop becomes true.
    ///
    /// @param queue SPSC frame ring to poll (thread-safe single consumer).
    /// @param responses SPSC frame ring that receives the serialized reply for
    ///                  every routed branch. Full / oversized replies are
    ///                  dropped (responses_failed) — never back-pressure the
    ///                  request consumer.
    /// @param stats accumulates the loop's dispatch/retirement totals.
    void ServeTaskMosaic(lsxnetwork::FrameQueue& queue,
                         lsxnetwork::FrameQueue& responses,
                         TaskMosaicStats& stats,
                         std::atomic<bool>& stop);

    /// Drains @p responses and sends each frame verbatim over a ZMQ REP socket
    /// bound on @p endpoint (blocking until @p stop). Intended to run on a
    /// second pinned IoWorker when the caller wants network delivery; tests
    /// drain the ring directly instead.
    void ServeResponses(lsxnetwork::FrameQueue& responses,
                        std::atomic<bool>& stop,
                        const std::string& endpoint);

    /// Welds the streaming ingestion loop into a pinned IoWorker (cores 0..3).
    void ServeTaskMosaicPinned(lsxnetwork::FrameQueue& queue,
                               lsxnetwork::FrameQueue& responses,
                               TaskMosaicStats& stats,
                               std::atomic<bool>& stop);

private:
    void DispatchOne(const lsxnetwork::FrameQueue::Frame& frame,
                     lsxnetwork::FrameQueue& responses,
                     TaskMosaicStats& stats);

    // Serializes @p res for @p cc and publishes it onto @p responses. Never
    // throws; failures bump the response_* counters.
    void EmitResponse(lsxnetwork::FrameQueue& responses,
                      const lsxnetwork::ChatCompletion& cc,
                      const gpu::InferenceResult& res,
                      TaskMosaicStats& stats);

    // Model targets resolved once from the dispatch table (keyed by type), so
    // the hot dispatch path never constructs a std::string or runs a hash.
    lsxcommon::IModel* glm_     = nullptr;  // "glm-5.2"
    lsxcommon::IModel* deepseek_ = nullptr; // "deepseek-v4-flash"
};

}  // namespace lsxcommon
