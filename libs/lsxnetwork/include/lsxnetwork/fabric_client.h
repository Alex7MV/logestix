#pragma once

#include <cstdint>
#include <memory>
#include <span>

#include "lsxnetwork/fabric_wire.h"
#include "lsxnetwork/zmq_context.h"

namespace lsxnetwork {

/// Blocking ZeroMQ REQ client for the lsxfabric binary-token server.
///
/// Strictly one request in flight: REQ must alternate send/recv. Not
/// thread-safe; use one instance per thread.
class FabricClient {
public:
    FabricClient() = default;
    ~FabricClient() = default;
    FabricClient(const FabricClient&) = delete;
    FabricClient& operator=(const FabricClient&) = delete;

    /// Creates the REQ socket and connects. Returns 0 on success, -1 if the
    /// socket could not be created or connected.
    int Connect(const char* endpoint);

    bool Connected() const noexcept { return sock_ != nullptr; }

    /// Sends @p input_ids and waits for the reply. Returns false on transport
    /// error or timeout, with `out.status = -1`. On success returns true and
    /// `out` holds the decoded response (check `out.status`).
    bool Infer(std::span<const std::int32_t> input_ids, std::uint32_t max_gen,
               FabricResponse& out, int timeout_ms = 30000);

private:
    ZmqContext ctx_;
    std::unique_ptr<ZmqSocket> sock_;
};

}  // namespace lsxnetwork
