#pragma once

#include <atomic>
#include <cstdint>
#include <string>

namespace lsxcommon {
class IModel;
}

namespace lsxfabric {

/// Single-threaded ZeroMQ REP server for the lsxfabric binary-token protocol.
///
/// One request in flight: the server blocks on recv, runs the injected model,
/// and sends one binary response frame. It owns no threads; Run() is expected
/// to be called on a dedicated thread with a stop flag.
class FabricServer {
public:
    FabricServer(lsxcommon::IModel& model, std::uint32_t default_max_gen);

    /// Binds @p endpoint and serves until @p stop becomes true.
    ///
    /// @param ready set true after a successful bind (for test synchronization).
    /// @param resolved_endpoint when non-null, receives the concrete bound
    ///        endpoint (resolves a "tcp://127.0.0.1:*" wildcard).
    /// @return 0 on clean stop, non-zero when the socket could not bind.
    int Run(const std::string& endpoint, std::atomic<bool>& stop,
            std::atomic<bool>* ready = nullptr,
            std::string* resolved_endpoint = nullptr);

private:
    lsxcommon::IModel& model_;
    std::uint32_t default_max_gen_;
    std::atomic<bool> abort_{false};
};

}  // namespace lsxfabric
