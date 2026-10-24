#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <future>
#include <string>
#include <thread>

namespace lsxnetwork {

/// Runs a network I/O loop body on a worker thread hard-pinned to physical
/// cores 0..3.
///
/// Pinning happens inside the newly-created thread (CPU affinity is
/// per-thread on Linux). The loop is expected to cooperatively poll
/// `Stopping()` between iterations so `Stop()` can join promptly.
class IoWorker {
public:
    explicit IoWorker(std::function<void()> loop);

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

private:
    void Entry();

    std::function<void()> loop_;
    std::thread thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> pinned_{false};
    std::promise<int> pin_result_;
};

}  // namespace lsxnetwork
