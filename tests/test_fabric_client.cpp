// FabricClient round-trip against an in-process fake REP server. Host-only.
#include <catch2/catch_test_macros.hpp>

#include "lsxnetwork/fabric_client.h"
#include "lsxnetwork/fabric_wire.h"
#include "lsxnetwork/zmq_context.h"

#include <zmq.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <string>
#include <thread>
#include <vector>

using namespace lsxnetwork;

namespace {

// Minimal REP server: binds tcp://127.0.0.1:* and replies with a fixed payload.
class FakeRepServer {
public:
    // `mode`: 0 = OK with tokens {7,8,9}; 1 = infer-failed "boom"; 2 = no reply.
    void Start(int mode) {
        std::promise<std::string> ready;
        std::future<std::string> fut = ready.get_future();
        thread_ = std::thread([this, mode, ready = std::move(ready)]() mutable {
            ZmqContext ctx;
            ZmqSocket sock(ctx, ZMQ_REP);
            if (sock.Bind("tcp://127.0.0.1:*") != 0) {
                ready.set_value(std::string{});
                return;
            }
            char ep[256];
            std::size_t len = sizeof(ep);
            sock.GetSockOpt(ZMQ_LAST_ENDPOINT, ep, &len);
            ready.set_value(std::string(ep));

            const int rcvtimeo = 100;
            sock.SetSockOpt(ZMQ_RCVTIMEO, &rcvtimeo, sizeof(rcvtimeo));
            while (!stop_.load(std::memory_order_acquire)) {
                zmq_msg_t msg;
                zmq_msg_init(&msg);
                const int n = zmq_msg_recv(&msg, sock.Get(), 0);
                if (n < 0) {
                    zmq_msg_close(&msg);
                    if (zmq_errno() == EAGAIN) continue;
                    break;
                }
                if (mode == 2) {  // swallow, never reply
                    zmq_msg_close(&msg);
                    continue;
                }
                FabricResponse r;
                if (mode == 0) {
                    r.status = static_cast<std::int32_t>(FabricStatus::kOk);
                    r.eos_terminated = true;
                    r.ttft_ms = 1.0;
                    r.total_ms = 2.0;
                    r.tokens_per_sec = 3.0;
                    r.tokens = {7, 8, 9};
                } else {
                    r.status =
                        static_cast<std::int32_t>(FabricStatus::kInferFailed);
                    r.error = "boom";
                }
                const std::vector<std::byte> out = EncodeFabricResponse(r);
                sock.Send(out.data(), out.size(), 0);
                zmq_msg_close(&msg);
            }
        });
        endpoint_ = fut.get();
    }

    void Stop() {
        stop_.store(true, std::memory_order_release);
        if (thread_.joinable()) thread_.join();
    }

    const std::string& Endpoint() const { return endpoint_; }

private:
    std::thread thread_;
    std::atomic<bool> stop_{false};
    std::string endpoint_;
};

}  // namespace

TEST_CASE("fabric_client: OK round-trip", "[fabric][client]") {
    FakeRepServer server;
    server.Start(0);
    REQUIRE_FALSE(server.Endpoint().empty());

    FabricClient client;
    REQUIRE(client.Connect(server.Endpoint().c_str()) == 0);
    FabricResponse resp;
    const std::vector<std::int32_t> ids{1, 2, 3};
    REQUIRE(client.Infer(ids, 4u, resp, 2000));
    CHECK(resp.status == static_cast<std::int32_t>(FabricStatus::kOk));
    CHECK(resp.eos_terminated);
    CHECK(resp.ttft_ms == 1.0);
    REQUIRE(resp.tokens.size() == 3);
    CHECK(resp.tokens[2] == 9);
    server.Stop();
}

TEST_CASE("fabric_client: infer-failed response surfaces error",
          "[fabric][client]") {
    FakeRepServer server;
    server.Start(1);
    FabricClient client;
    REQUIRE(client.Connect(server.Endpoint().c_str()) == 0);
    FabricResponse resp;
    const std::vector<std::int32_t> ids{1};
    REQUIRE(client.Infer(ids, 1u, resp, 2000));
    CHECK(resp.status == static_cast<std::int32_t>(FabricStatus::kInferFailed));
    CHECK(resp.error == "boom");
    server.Stop();
}

TEST_CASE("fabric_client: timeout reports transport failure",
          "[fabric][client]") {
    FakeRepServer server;
    server.Start(2);
    FabricClient client;
    REQUIRE(client.Connect(server.Endpoint().c_str()) == 0);
    FabricResponse resp;
    const std::vector<std::int32_t> ids{1};
    REQUIRE_FALSE(client.Infer(ids, 1u, resp, 150));
    CHECK(resp.status == -1);
    server.Stop();
}
