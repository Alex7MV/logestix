#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>

#include "config.h"
#include "lsxnetwork/fabric_stream_client.h"

namespace taskmosaic {

/// One lsxfabric backend connection with bounded admission.
///
/// Admission counts *in-flight + queued* requests: a caller that cannot admit
/// maps to HTTP 429. The connection mutex serializes actual use (one request in
/// flight per backend).
class Backend {
public:
    Backend(const BackendConfig& cfg, int queue_depth);

    int Connect();
    bool Ready() const noexcept { return ready_.load(std::memory_order_acquire); }
    void MarkDown() noexcept { ready_.store(false, std::memory_order_release); }

    /// Active health check (ping/pong). Updates the cached ready flag and
    /// returns whether the backend answered. Safe to call while a request is in
    /// flight (uses the dedicated probe channel).
    bool Probe(int timeout_ms);

    bool TryAdmit() noexcept;
    void Release() noexcept;

    std::uint32_t NextRequestId() noexcept { return ++last_id_; }
    lsxnetwork::FabricStreamClient& Client() noexcept { return client_; }
    std::mutex& Mutex() noexcept { return mtx_; }

private:
    BackendConfig cfg_;
    int queue_depth_;
    std::atomic<int> admitted_{0};
    std::atomic<bool> ready_{false};
    std::atomic<std::uint32_t> last_id_{0};
    lsxnetwork::FabricStreamClient client_;
    std::mutex mtx_;
};

}  // namespace taskmosaic
