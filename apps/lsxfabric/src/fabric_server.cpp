#include "fabric_server.h"

#include "lsxcommon/log.h"
#include "lsxcommon/model.h"
#include "lsxnetwork/fabric_wire.h"
#include "lsxnetwork/zmq_context.h"

#include <zmq.h>

#include <chrono>
#include <cstring>
#include <exception>
#include <span>
#include <thread>
#include <utility>
#include <vector>

namespace lsxfabric {
namespace {

// ROUTER recv: first frame is the peer identity, second is the payload.
bool RecvRouted(void* sock, std::vector<std::byte>& ident,
                std::vector<std::byte>& payload) {
    zmq_msg_t id;
    zmq_msg_init(&id);
    if (zmq_msg_recv(&id, sock, 0) < 0) {
        zmq_msg_close(&id);
        return false;
    }
    ident.assign(static_cast<const std::byte*>(zmq_msg_data(&id)),
                 static_cast<const std::byte*>(zmq_msg_data(&id)) +
                     zmq_msg_size(&id));
    const bool more = zmq_msg_more(&id) != 0;
    zmq_msg_close(&id);
    if (!more) return false;

    zmq_msg_t body;
    zmq_msg_init(&body);
    if (zmq_msg_recv(&body, sock, 0) < 0) {
        zmq_msg_close(&body);
        return false;
    }
    payload.assign(static_cast<const std::byte*>(zmq_msg_data(&body)),
                   static_cast<const std::byte*>(zmq_msg_data(&body)) +
                       zmq_msg_size(&body));
    zmq_msg_close(&body);
    return true;
}

void SendRouted(void* sock, const std::vector<std::byte>& ident,
                const std::vector<std::byte>& body) {
    zmq_send(sock, ident.data(), ident.size(), ZMQ_SNDMORE);
    zmq_send(sock, body.data(), body.size(), 0);
}

// Returns false when either part could not be handed to libzmq (e.g. the peer
// stalled past ZMQ_SNDTIMEO); callers then drop the stream instead of blocking
// inference on a slow consumer.
bool TrySendRouted(void* sock, const std::vector<std::byte>& ident,
                   const std::vector<std::byte>& body) {
    if (zmq_send(sock, ident.data(), ident.size(), ZMQ_SNDMORE) < 0)
        return false;
    if (zmq_send(sock, body.data(), body.size(), 0) >= 0)
        return true;
    // A multipart cannot be abandoned half-sent: the identity frame is already
    // queued, and the next TrySendRouted would append its identity to this same
    // (incomplete) message and corrupt every later stream. Close the message
    // with an empty terminal frame (best effort) so the peer decodes a
    // droppable truncated frame instead of a desynced socket.
    (void)zmq_send(sock, "", 0, 0);
    return false;
}

}  // namespace

FabricServer::FabricServer(lsxcommon::IModel& model,
                           std::uint32_t default_max_gen)
    : model_(model), default_max_gen_(default_max_gen) {}

int FabricServer::Run(const std::string& endpoint,
                      const std::string& cancel_endpoint,
                      std::atomic<bool>& stop, std::atomic<bool>* ready,
                      std::string* resolved_endpoint,
                      std::string* resolved_cancel_endpoint) {
    lsxnetwork::ZmqContext ctx;
    lsxnetwork::ZmqSocket router(ctx, ZMQ_ROUTER);
    if (!router.Valid() || router.Bind(endpoint.c_str()) != 0) {
        lsxcommon::log::error("[fabric] cannot bind " + endpoint);
        return 1;
    }
    auto resolve = [](lsxnetwork::ZmqSocket& s, std::string* out) {
        if (out == nullptr) return;
        char ep[256];
        std::size_t len = sizeof(ep);
        if (s.GetSockOpt(ZMQ_LAST_ENDPOINT, ep, &len) == 0) *out = std::string(ep);
    };
    resolve(router, resolved_endpoint);

    // Bound the outbound queue and fail fast instead of blocking the inference
    // callback (which holds the residency lock) on a stalled consumer.
    const int sndhwm = 1024;
    const int sndtimeo = 2000;
    router.SetSockOpt(ZMQ_SNDHWM, &sndhwm, sizeof(sndhwm));
    router.SetSockOpt(ZMQ_SNDTIMEO, &sndtimeo, sizeof(sndtimeo));

    const int rcvtimeo = 200;

    // Cancel listener: owns (creates + binds) the PULL socket in its own thread
    // — ZMQ sockets must not be migrated between threads.
    std::atomic<bool> listener_stop{false};
    std::atomic<bool> cancel_ready{false};
    std::thread listener([&] {
        lsxnetwork::ZmqSocket pull(ctx, ZMQ_PULL);
        if (!pull.Valid() || pull.Bind(cancel_endpoint.c_str()) != 0) {
            lsxcommon::log::error("[fabric] cannot bind cancel endpoint " +
                                  cancel_endpoint);
            cancel_ready.store(true, std::memory_order_release);
            return;
        }
        if (resolved_cancel_endpoint != nullptr) {
            char ep[256];
            std::size_t len = sizeof(ep);
            if (pull.GetSockOpt(ZMQ_LAST_ENDPOINT, ep, &len) == 0)
                *resolved_cancel_endpoint = std::string(ep);
        }
        pull.SetSockOpt(ZMQ_RCVTIMEO, &rcvtimeo, sizeof(rcvtimeo));
        cancel_ready.store(true, std::memory_order_release);
        while (!listener_stop.load(std::memory_order_acquire) &&
               !stop.load(std::memory_order_acquire)) {
            zmq_msg_t msg;
            zmq_msg_init(&msg);
            const int n = zmq_msg_recv(&msg, pull.Get(), 0);
            if (n < 0) {
                zmq_msg_close(&msg);
                continue;  // EAGAIN
            }
            const std::span<const std::byte> f(
                static_cast<const std::byte*>(zmq_msg_data(&msg)),
                static_cast<std::size_t>(n));
            std::uint32_t rid = 0;
            const bool ok = lsxnetwork::DecodeFabricCancel(f, rid) ==
                            lsxnetwork::WireStatus::kOk;
            zmq_msg_close(&msg);
            if (!ok) continue;
            std::lock_guard<std::mutex> lock(cancel_mtx_);
            if (current_abort_ != nullptr && rid == current_id_)
                current_abort_->store(true, std::memory_order_release);
        }
    });

    while (!cancel_ready.load(std::memory_order_acquire))
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (ready != nullptr) ready->store(true);

    lsxcommon::log::info("[fabric] serving on " + endpoint + " (cancel " +
                         cancel_endpoint + ")");
    router.SetSockOpt(ZMQ_RCVTIMEO, &rcvtimeo, sizeof(rcvtimeo));
    while (!stop.load(std::memory_order_acquire)) {
        std::vector<std::byte> ident;
        std::vector<std::byte> payload;
        if (!RecvRouted(router.Get(), ident, payload)) {
            if (zmq_errno() == EAGAIN) continue;  // recv timeout: re-check stop
            break;
        }

        lsxnetwork::FabricRequestView req;
        if (lsxnetwork::DecodeFabricRequest(payload, req) !=
            lsxnetwork::WireStatus::kOk) {
            // A health probe is answered here, off the inference path.
            std::uint32_t nonce = 0;
            if (lsxnetwork::DecodeFabricPing(payload, nonce) ==
                lsxnetwork::WireStatus::kOk) {
                SendRouted(router.Get(), ident,
                           lsxnetwork::EncodeFabricPong(nonce));
                continue;
            }
            SendRouted(router.Get(), ident,
                       lsxnetwork::EncodeFabricError(0u, "malformed request"));
            continue;
        }

        std::atomic<bool> abort{false};
        {
            std::lock_guard<std::mutex> lock(cancel_mtx_);
            current_abort_ = &abort;
            current_id_ = req.request_id;
        }

        lsxcommon::ModelRequest mreq;
        mreq.input_ids.assign(req.tokens.begin(), req.tokens.end());
        mreq.max_gen_tokens = req.max_gen != 0u
                                  ? static_cast<int>(req.max_gen)
                                  : static_cast<int>(default_max_gen_);
        mreq.abort_flag = &abort;
        mreq.sampling.temperature = req.sampling.temperature;
        mreq.sampling.top_p = req.sampling.top_p;
        mreq.sampling.top_k = req.sampling.top_k;
        mreq.sampling.seed = req.sampling.seed;
        // Reuse one frame buffer across tokens (no per-token allocation) and
        // drop the stream if the consumer stalls.
        std::vector<std::byte> frame;
        bool send_failed = false;
        mreq.token_callback = [&](const lsxcommon::TokenChunk& chunk) {
            if (send_failed) return;
            const std::int32_t tok = static_cast<std::int32_t>(chunk.token_id);
            lsxnetwork::EncodeFabricTokenBatch(
                frame, req.request_id, std::span<const std::int32_t>(&tok, 1));
            if (!TrySendRouted(router.Get(), ident, frame)) {
                send_failed = true;
                abort.store(true, std::memory_order_release);
            }
        };

        // Infer() acquires the shared residency lock itself (via
        // ModelBase::RunUnderResidencyLock); taking it here as well would be a
        // recursive read-lock on a std::shared_mutex -- undefined behavior that
        // can deadlock against a concurrent Unload()'s unique lock.
        lsxcommon::gpu::InferenceResult res;
        try {
            res = model_.Infer(mreq);
        } catch (const std::exception& e) {
            SendRouted(router.Get(), ident,
                       lsxnetwork::EncodeFabricError(req.request_id, e.what()));
            std::lock_guard<std::mutex> lock(cancel_mtx_);
            current_abort_ = nullptr;
            continue;
        }

        if (!send_failed) {
            if (!res.ok) {
                TrySendRouted(
                    router.Get(), ident,
                    lsxnetwork::EncodeFabricError(req.request_id, res.error_msg));
            } else {
                TrySendRouted(
                    router.Get(), ident,
                    lsxnetwork::EncodeFabricDone(
                        req.request_id, res.eos_terminated,
                        static_cast<std::uint32_t>(res.gen_tokens.size()),
                        res.ttft_ms, res.total_time_ms, res.tokens_per_sec));
            }
        }
        std::lock_guard<std::mutex> lock(cancel_mtx_);
        current_abort_ = nullptr;
    }

    listener_stop.store(true, std::memory_order_release);
    if (listener.joinable()) listener.join();
    return 0;
}

}  // namespace lsxfabric
