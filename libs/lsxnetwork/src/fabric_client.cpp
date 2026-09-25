#include "lsxnetwork/fabric_client.h"

#include <zmq.h>

namespace lsxnetwork {

int FabricClient::Connect(const char* endpoint) {
    sock_ = std::make_unique<ZmqSocket>(ctx_, ZMQ_REQ);
    if (!sock_->Valid()) {
        sock_.reset();
        return -1;
    }
    if (sock_->Connect(endpoint) != 0) {
        sock_.reset();
        return -1;
    }
    return 0;
}

bool FabricClient::Infer(std::span<const std::int32_t> input_ids,
                         std::uint32_t max_gen, FabricResponse& out,
                         int timeout_ms) {
    out = FabricResponse{};
    if (!sock_ || !sock_->Valid()) {
        out.status = -1;
        return false;
    }
    if (sock_->SetSockOpt(ZMQ_RCVTIMEO, &timeout_ms, sizeof(timeout_ms)) != 0) {
        out.status = -1;
        return false;
    }

    const std::vector<std::byte> frame = EncodeFabricRequest(input_ids, max_gen);
    if (sock_->Send(frame.data(), frame.size(), 0) < 0) {
        out.status = -1;
        return false;
    }

    zmq_msg_t msg;
    zmq_msg_init(&msg);
    const int n = zmq_msg_recv(&msg, sock_->Get(), 0);
    if (n < 0) {
        zmq_msg_close(&msg);
        out.status = -1;
        return false;
    }
    const std::span<const std::byte> reply(
        static_cast<const std::byte*>(zmq_msg_data(&msg)),
        static_cast<std::size_t>(n));
    const WireStatus ws = DecodeFabricResponse(reply, out);
    zmq_msg_close(&msg);
    if (ws != WireStatus::kOk) {
        out = FabricResponse{};
        out.status = -1;
        return false;
    }
    return true;
}

}  // namespace lsxnetwork
