#include "lsxnetwork/fabric_stream_client.h"

#include <zmq.h>

#include <chrono>
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
    const int sndtimeo = 2000;
    dealer_->SetSockOpt(ZMQ_SNDTIMEO, &sndtimeo, sizeof(sndtimeo));
    // Cancel is fire-and-forget: never block the caller if the server is gone.
    const int nosend = 0;
    cancel_->SetSockOpt(ZMQ_SNDTIMEO, &nosend, sizeof(nosend));
    cancel_->SetSockOpt(ZMQ_LINGER, &nosend, sizeof(nosend));
    if (cancel_endpoint != nullptr && *cancel_endpoint != '\0')
        cancel_->Connect(cancel_endpoint);

    // Dedicated health-probe channel (best effort; absence only disables Probe).
    probe_ = std::make_unique<ZmqSocket>(ctx_, ZMQ_DEALER);
    if (!probe_->Valid() || probe_->Connect(endpoint) != 0)
        probe_.reset();
    else
        probe_->SetSockOpt(ZMQ_SNDTIMEO, &sndtimeo, sizeof(sndtimeo));
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

bool FabricStreamClient::Probe(int timeout_ms) {
    if (!probe_ || !probe_->Valid()) return false;
    std::lock_guard<std::mutex> lock(probe_mtx_);
    const std::uint32_t nonce = ++probe_nonce_;
    const std::vector<std::byte> ping = EncodeFabricPing(nonce);
    if (probe_->Send(ping.data(), ping.size(), 0) !=
        static_cast<int>(ping.size()))
        return false;

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (true) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) return false;
        int remaining = static_cast<int>(
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now)
                .count());
        if (remaining < 1) remaining = 1;
        probe_->SetSockOpt(ZMQ_RCVTIMEO, &remaining, sizeof(remaining));

        zmq_msg_t msg;
        zmq_msg_init(&msg);
        const int n = zmq_msg_recv(&msg, probe_->Get(), 0);
        if (n < 0) {
            zmq_msg_close(&msg);
            return false;  // timeout / transport
        }
        const std::span<const std::byte> frame(
            static_cast<const std::byte*>(zmq_msg_data(&msg)),
            static_cast<std::size_t>(n));
        std::uint32_t got = 0;
        const bool ok = DecodeFabricPong(frame, got) == WireStatus::kOk &&
                        got == nonce;
        zmq_msg_close(&msg);
        if (ok) return true;
        // Nonce mismatch: a stale pong from an earlier probe. Keep waiting
        // for the matching one until the deadline.
    }
}

}  // namespace lsxnetwork
