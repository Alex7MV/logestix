#include "lsxnetwork/fabric_stream_client.h"

#include <zmq.h>

#include <cstring>

namespace lsxnetwork {

int FabricStreamClient::Connect(const char* endpoint,
                                const char* cancel_endpoint) {
    dealer_ = std::make_unique<ZmqSocket>(ctx_, ZMQ_DEALER);
    cancel_ = std::make_unique<ZmqSocket>(ctx_, ZMQ_PUSH);
    if (!dealer_->Valid() || !cancel_->Valid()) {
        dealer_.reset();
        cancel_.reset();
        return -1;
    }
    if (dealer_->Connect(endpoint) != 0) {
        dealer_.reset();
        cancel_.reset();
        return -1;
    }
    // Cancel is fire-and-forget: never block the caller if the server is gone.
    const int sndtimeo = 0;
    cancel_->SetSockOpt(ZMQ_SNDTIMEO, &sndtimeo, sizeof(sndtimeo));
    cancel_->SetSockOpt(ZMQ_LINGER, &sndtimeo, sizeof(sndtimeo));
    if (cancel_endpoint != nullptr && *cancel_endpoint != '\0')
        cancel_->Connect(cancel_endpoint);
    return 0;
}

bool FabricStreamClient::Submit(std::uint32_t request_id,
                                std::span<const std::int32_t> input_ids,
                                std::uint32_t max_gen) {
    if (!Connected()) return false;
    const std::vector<std::byte> frame =
        EncodeFabricRequest(request_id, input_ids, max_gen);
    return dealer_->Send(frame.data(), frame.size(), 0) ==
           static_cast<int>(frame.size());
}

FabricStreamClient::RecvStatus FabricStreamClient::Recv(
    FabricStreamFrame& out, int timeout_ms) {
    if (!Connected()) return RecvStatus::kTransport;
    if (dealer_->SetSockOpt(ZMQ_RCVTIMEO, &timeout_ms, sizeof(timeout_ms)) != 0)
        return RecvStatus::kTransport;

    zmq_msg_t msg;
    zmq_msg_init(&msg);
    const int n = zmq_msg_recv(&msg, dealer_->Get(), 0);
    if (n < 0) {
        const bool timed_out = (zmq_errno() == EAGAIN);
        zmq_msg_close(&msg);
        return timed_out ? RecvStatus::kTimeout : RecvStatus::kTransport;
    }
    const std::span<const std::byte> frame(
        static_cast<const std::byte*>(zmq_msg_data(&msg)),
        static_cast<std::size_t>(n));
    const WireStatus ws = DecodeFabricStreamFrame(frame, out);
    zmq_msg_close(&msg);
    return ws == WireStatus::kOk ? RecvStatus::kOk : RecvStatus::kMalformed;
}

void FabricStreamClient::Cancel(std::uint32_t request_id) {
    if (!cancel_ || !cancel_->Valid()) return;
    const std::vector<std::byte> frame = EncodeFabricCancel(request_id);
    cancel_->Send(frame.data(), frame.size(), 0);
}

}  // namespace lsxnetwork
