#include "lsxcommon/lsx_orchestrator.h"

#include <simdjson.h>
#include <spdlog/spdlog.h>
#include <exception>
#include <thread>

#include "lsxnetwork/openai_endpoint.h"
#include "lsxnetwork/io_worker.h"

#include "lsxcommon/path_registry.h"

namespace lsxcommon {

namespace {
constexpr std::string_view kModelGlm52 = "glm-5.2";
constexpr std::string_view kModelDeepSeekFlash = "deepseek-v4-flash";

/// RAII zero-copy retirement guard. Releases the ring slot (and records the
/// retirement) on EVERY exit path — including exceptions escaping Parse() or
/// IModel::Infer() — so a taken frame is never left permanently occupying the
/// ring. Without the guard a thrown exception would skip Release(), leaving
/// that slot unreachable forever (the SPSC producer would stall).
class SlotRetire {
public:
    SlotRetire(lsxnetwork::FrameQueue& queue, std::size_t index,
               TaskMosaicStats& stats) noexcept
        : queue_(queue), index_(index), stats_(stats) {}

    SlotRetire(const SlotRetire&) = delete;
    SlotRetire& operator=(const SlotRetire&) = delete;

    ~SlotRetire() noexcept {
        queue_.Release(index_);
        ++stats_.retired_slots;
    }

private:
    lsxnetwork::FrameQueue& queue_;
    std::size_t index_;
    TaskMosaicStats& stats_;
};
}  // namespace

LsxOrchestrator::LsxOrchestrator(const ModelTable& models) {
    // Resolve the two dispatch targets once. Registration is done inside the
    // constructor body (site-init order is fixed there rather than deferred to
    // on-demand static init), so the model table must already be fully built.
    if (auto it = models.find(std::string(kModelGlm52)); it != models.end())
        glm_ = it->second;
    if (auto it = models.find(std::string(kModelDeepSeekFlash)); it != models.end())
        deepseek_ = it->second;
}

DispatchSignal LsxOrchestrator::Classify(std::string_view model) noexcept {
    if (model == kModelGlm52) return DispatchSignal::kGlm52;
    if (model == kModelDeepSeekFlash) return DispatchSignal::kDeepSeekFlash;
    return DispatchSignal::kUnknown;
}

void LsxOrchestrator::ServeTaskMosaic(lsxnetwork::FrameQueue& queue,
                                      TaskMosaicStats& stats,
                                      std::atomic<bool>& stop) {
    // Headless consumer loop. Polling the lock-free SPSC tail watermark via
    // TryTake() is non-blocking: an empty ring yields the CPU slice and retries
    // — no cudaDeviceSynchronize() / blocking host sync fence lives on the poll
    // path. The single slow consumer bounds the producer (Acquire() -> nullptr
    // -> 429/503) without ever overwriting a live slot.
    while (!stop.load(std::memory_order_acquire)) {
        lsxnetwork::FrameQueue::Frame frame;
        std::size_t index = 0;
        if (!queue.TryTake(frame, index)) {
            std::this_thread::yield();
            continue;
        }
        ++stats.taken_frames;

        // Zero-copy: the frame's bytes stay queue-owned while the branch runs;
        // the slot is retired only after the target model's CUDA execution step
        // has fully returned (SlotRetire dtor below — exception-safe).
        SlotRetire retire(queue, index, stats);
        try {
            DispatchOne(frame, stats);
        } catch (const std::exception& ex) {
            // SlotRetire guarantees the ring slot itself is never leaked, but
            // without this catch an exception escaping DispatchOne()/Infer()
            // (e.g. std::bad_alloc, a filesystem error from a missing weight
            // file) would keep unwinding straight through this while loop and
            // permanently kill the consumer thread — turning one bad request
            // into a permanent TaskMosaic outage instead of a single dropped
            // frame.
            ++stats.dispatch_exceptions;
            spdlog::error("[orchestrator] DispatchOne threw: {}", ex.what());
        } catch (...) {
            ++stats.dispatch_exceptions;
            spdlog::error("[orchestrator] DispatchOne threw a non-std::exception");
        }
    }
}

void LsxOrchestrator::ServeTaskMosaicPinned(
    lsxnetwork::FrameQueue& queue, TaskMosaicStats& stats,
    std::atomic<bool>& stop, const std::string& passport_path) {
    // The entire consumer/ingestion loop runs inside a single pinned worker.
    // That worker threads a privately-owned LsxTokenTranslator, loaded from the
    // unified model-metadata vault during thread entry (after affinity pin).
    // The per-thread translator is fully isolated — no shared singleton — and
    // its mapping views stay alive only for the lifetime of this worker.
    lsxnetwork::IoWorker worker(
        [this, &queue, &stats, &stop] { ServeTaskMosaic(queue, stats, stop); },
        passport_path);
    if (worker.Start() != 0) {
        spdlog::error("[orchestrator] pinned serve worker failed to start");
        return;
    }
    // Joins implicitly via the worker dtor once ServeTaskMosaic returns on stop.
}

void LsxOrchestrator::DispatchOne(const lsxnetwork::FrameQueue::Frame& frame,
                                  TaskMosaicStats& stats) {
    // Zero-alloc extraction of the dispatch target (top-level "model").
    lsxnetwork::ChatCompletion cc;
    const simdjson::error_code rc =
        lsxnetwork::OpenAiEndpoint().Parse(frame.data, frame.size, frame.capacity, cc);
    if (rc != simdjson::SUCCESS) {
        // Slot still retired by the SlotRetire guard in ServeTaskMosaic.
        spdlog::warn("[orchestrator] frame parse failed: {}", simdjson::error_message(rc));
        return;
    }

    // Fast conditional switch over the model string. Hot path is pointer
    // dispatch only (targets pre-resolved at construction) — no allocation.
    const DispatchSignal sig = Classify(cc.model);
    switch (sig) {
        case DispatchSignal::kGlm52: {
            ++stats.glm_routed;
            ModelRequest req;
            if (glm_ != nullptr) {
                // GLM-5.2 execution branch. GlmV5Model::Infer is currently an
                // honest stub (returns "GLM inference pipeline not implemented")
                // until the GLM runtime lands; routing + retirement are real.
                glm_->Infer(req);
            } else {
                spdlog::warn("[orchestrator] glm-5.2 routed but no model registered");
            }
            break;
        }
        case DispatchSignal::kDeepSeekFlash: {
            ++stats.deepseek_routed;
            ModelRequest req;
            if (deepseek_ != nullptr) {
                // DeepSeek-V4-Flash execution branch -> gpu::RunInferenceModel.
                deepseek_->Infer(req);
            } else {
                spdlog::warn("[orchestrator] deepseek-v4-flash routed but no model registered");
            }
            break;
        }
        case DispatchSignal::kUnknown:
            ++stats.unknown_routed;
            spdlog::warn("[orchestrator] unknown model '{}'", cc.model);
            break;
    }
}

}  // namespace lsxcommon