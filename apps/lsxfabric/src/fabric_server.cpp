#include "fabric_server.h"

#include "lsxcommon/log.h"
#include "lsxcommon/model.h"
#include "lsxnetwork/fabric_wire.h"
#include "lsxnetwork/zmq_context.h"

#include <zmq.h>

#include <exception>
#include <shared_mutex>
#include <span>
#include <utility>
#include <vector>

namespace lsxfabric {
namespace {

using lsxnetwork::FabricRequestView;
using lsxnetwork::FabricResponse;
using lsxnetwork::FabricStatus;
using lsxnetwork::WireStatus;

std::vector<std::byte> BadRequest(std::string msg) {
    FabricResponse r;
    r.status = static_cast<std::int32_t>(FabricStatus::kBadRequest);
    r.error = std::move(msg);
    return lsxnetwork::EncodeFabricResponse(r);
}

std::vector<std::byte> HandleFrame(lsxcommon::IModel& model,
                                   std::uint32_t default_max_gen,
                                   std::atomic<bool>& abort,
                                   std::span<const std::byte> frame) {
    FabricRequestView req;
    if (lsxnetwork::DecodeFabricRequest(frame, req) != WireStatus::kOk)
        return BadRequest("malformed request");

    lsxcommon::ModelRequest mreq;
    mreq.input_ids.assign(req.tokens.begin(), req.tokens.end());
    mreq.max_gen_tokens = req.max_gen != 0u
                              ? static_cast<int>(req.max_gen)
                              : static_cast<int>(default_max_gen);
    mreq.abort_flag = &abort;

    lsxcommon::gpu::InferenceResult res;
    try {
        std::shared_lock<std::shared_mutex> lock(model.ResidencyMutex());
        res = model.Infer(mreq);
    } catch (const std::exception& e) {
        FabricResponse r;
        r.status = static_cast<std::int32_t>(FabricStatus::kInternal);
        r.error = e.what();
        return lsxnetwork::EncodeFabricResponse(r);
    }

    FabricResponse out;
    if (!res.ok) {
        out.status = static_cast<std::int32_t>(FabricStatus::kInferFailed);
        out.error = res.error_msg;
    } else {
        out.status = static_cast<std::int32_t>(FabricStatus::kOk);
        out.tokens = res.gen_tokens;
        out.eos_terminated = res.eos_terminated;
        out.ttft_ms = res.ttft_ms;
        out.total_ms = res.total_time_ms;
        out.tokens_per_sec = res.tokens_per_sec;
    }
    return lsxnetwork::EncodeFabricResponse(out);
}

}  // namespace

FabricServer::FabricServer(lsxcommon::IModel& model,
                           std::uint32_t default_max_gen)
    : model_(model), default_max_gen_(default_max_gen) {}

int FabricServer::Run(const std::string& endpoint, std::atomic<bool>& stop,
                      std::atomic<bool>* ready,
                      std::string* resolved_endpoint) {
    lsxnetwork::ZmqContext ctx;
    lsxnetwork::ZmqSocket sock(ctx, ZMQ_REP);
    if (!sock.Valid() || sock.Bind(endpoint.c_str()) != 0) {
        lsxcommon::log::error("[fabric] cannot bind endpoint: " + endpoint);
        return 1;
    }
    if (resolved_endpoint != nullptr) {
        char ep[256];
        std::size_t len = sizeof(ep);
        if (sock.GetSockOpt(ZMQ_LAST_ENDPOINT, ep, &len) == 0)
            *resolved_endpoint = std::string(ep);
    }
    if (ready != nullptr) ready->store(true);

    const int rcvtimeo = 200;
    sock.SetSockOpt(ZMQ_RCVTIMEO, &rcvtimeo, sizeof(rcvtimeo));

    lsxcommon::log::info("[fabric] serving on " + endpoint);
    while (!stop.load(std::memory_order_acquire)) {
        zmq_msg_t msg;
        zmq_msg_init(&msg);
        const int n = zmq_msg_recv(&msg, sock.Get(), 0);
        if (n < 0) {
            const int err = zmq_errno();
            zmq_msg_close(&msg);
            if (err == EAGAIN) continue;  // recv timeout: re-check stop
            break;                        // ETERM or fatal
        }
        const std::span<const std::byte> frame(
            static_cast<const std::byte*>(zmq_msg_data(&msg)),
            static_cast<std::size_t>(n));
        const std::vector<std::byte> out =
            HandleFrame(model_, default_max_gen_, abort_, frame);
        zmq_msg_close(&msg);
        sock.Send(out.data(), out.size(), 0);
    }
    return 0;
}

}  // namespace lsxfabric
