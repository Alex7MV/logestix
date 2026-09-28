#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>

namespace lsxcommon {
class IModel;
}

namespace lsxfabric {

/// Streaming ZeroMQ server for the v2 fabric protocol.
///
/// Main thread owns a ROUTER socket: it receives requests and streams token
/// batches (from the model's per-token callback) as separate messages. A second
/// thread owns a PULL socket and sets the in-flight request's abort flag when a
/// matching cancel arrives.
class FabricServer {
public:
    FabricServer(lsxcommon::IModel& model, std::uint32_t default_max_gen);

    /// Binds @p endpoint (ROUTER) and @p cancel_endpoint (PULL) and serves until
    /// @p stop becomes true.
    /// @param ready set true after both binds succeed (test synchronization).
    /// @param resolved_endpoint / resolved_cancel_endpoint receive the concrete
    ///        bound endpoints (resolve a "tcp://127.0.0.1:*" wildcard).
    /// @return 0 on clean stop, non-zero when a socket could not bind.
    int Run(const std::string& endpoint, const std::string& cancel_endpoint,
            std::atomic<bool>& stop, std::atomic<bool>* ready = nullptr,
            std::string* resolved_endpoint = nullptr,
            std::string* resolved_cancel_endpoint = nullptr);

private:
    lsxcommon::IModel& model_;
    std::uint32_t default_max_gen_;

    std::mutex cancel_mtx_;
    std::atomic<bool>* current_abort_ = nullptr;  // guarded by cancel_mtx_
    std::uint32_t current_id_ = 0;                // guarded by cancel_mtx_
};

}  // namespace lsxfabric
