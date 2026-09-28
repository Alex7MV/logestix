#pragma once

#include <cstdint>
#include <memory>
#include <span>

#include "lsxnetwork/fabric_wire.h"
#include "lsxnetwork/zmq_context.h"

namespace lsxnetwork {

/// Blocking DEALER client for the v2 streaming fabric server, with a companion
/// PUSH socket for per-request cancellation.
///
/// One request in flight (DEALER must not interleave requests). Not thread-safe;
/// use one instance per backend.
class FabricStreamClient {
public:
    enum class RecvStatus { kOk, kTimeout, kTransport, kMalformed };

    FabricStreamClient() = default;
    ~FabricStreamClient() = default;
    FabricStreamClient(const FabricStreamClient&) = delete;
    FabricStreamClient& operator=(const FabricStreamClient&) = delete;

    /// Creates the DEALER + cancel PUSH sockets and connects both. Returns 0 on
    /// success, -1 on socket/connect failure.
    int Connect(const char* endpoint, const char* cancel_endpoint);

    bool Connected() const noexcept { return dealer_ && dealer_->Valid(); }

    /// Sends the request frame. Returns false on transport error.
    bool Submit(std::uint32_t request_id,
                std::span<const std::int32_t> input_ids, std::uint32_t max_gen);

    /// Receives the next stream frame. kTimeout means no frame within
    /// @p timeout_ms; kMalformed means the frame failed to decode.
    RecvStatus Recv(FabricStreamFrame& out, int timeout_ms);

    /// Best-effort cancel of @p request_id on the PUSH channel.
    void Cancel(std::uint32_t request_id);

private:
    ZmqContext ctx_;
    std::unique_ptr<ZmqSocket> dealer_;
    std::unique_ptr<ZmqSocket> cancel_;
};

}  // namespace lsxnetwork
