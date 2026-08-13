#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <future>
#include <string>
#include <thread>

#include "lsxnetwork/lsx_token_translator.h"

namespace lsxnetwork {

/// Runs a network I/O loop body on a worker thread hard-pinned to physical
/// cores 0..3.
///
/// Pinning happens inside the newly-created thread (CPU affinity is
/// per-thread on Linux). The loop is expected to cooperatively poll
/// `Stopping()` between iterations so `Stop()` can join promptly.
///
/// Phase 4.5 — per-thread translator isolation. Each IoWorker owns exactly one
/// private `LsxTokenTranslator` that is `Load()`ed on thread entry, immediately
/// after affinity pinning, from the vault-resolved passport path supplied by
/// the caller (the orchestrator in lsxcommon resolves the unified
/// model-metadata vault; lsxnetwork cannot link lsxcommon without a cycle). No
/// shared/static singleton is used: distinct workers each carry an independent,
/// fully isolated mapping over the same read-only passport backing file.
class IoWorker {
public:
    explicit IoWorker(std::function<void()> loop);

    /// @param loop          loop body to run on the pinned thread.
    /// @param passport_path vault-resolved "model-metadata/passport/passport.bin"
    ///                      path this worker's private translator maps on entry.
    IoWorker(std::function<void()> loop, std::string passport_path);

    ~IoWorker() noexcept;
    IoWorker(const IoWorker&) = delete;
    IoWorker& operator=(const IoWorker&) = delete;

    /// Spawns the worker and waits until it has established affinity to cores
    /// 0..3. Returns 0 on success, otherwise a non-zero OS error code.
    int Start();

    /// Requests stop and joins the worker thread.
    void Stop() noexcept;

    /// True once the worker reported successful affinity pinning.
    bool Pinned() const noexcept { return pinned_.load(std::memory_order_acquire); }

    /// Cooperative stop probe for the I/O loop body.
    bool Stopping() const noexcept { return !running_.load(std::memory_order_acquire); }

    /// The worker's privately-owned, thread-pinned translator. Loaded on thread
    /// entry (after affinity pin) from the constructor-supplied passport path.
    /// Move-only object; access is restricted to the owning worker's context.
    LsxTokenTranslator& Translator() noexcept { return translator_; }
    const LsxTokenTranslator& Translator() const noexcept { return translator_; }

private:
    void Entry();

    std::function<void()> loop_;
    std::string passport_path_;
    alignas(64) LsxTokenTranslator translator_;
    std::thread thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> pinned_{false};
    std::promise<int> pin_result_;
};

}  // namespace lsxnetwork
