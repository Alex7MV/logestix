// FabricStreamClient <-> a minimal inproc ROUTER/PULL fake server.
#include <catch2/catch_test_macros.hpp>

#include <lsxnetwork/fabric_stream_client.h>
#include <lsxnetwork/fabric_wire.h>
#include <lsxnetwork/zmq_context.h>

#include <zmq.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <thread>
#include <vector>

using namespace lsxnetwork;

namespace {

// A tiny server: ROUTER streams a token batch + done per request; PULL records
// received cancel ids.
struct FakeServer {
    ZmqContext ctx;
    ZmqSocket router{ctx, ZMQ_ROUTER};
    ZmqSocket pull{ctx, ZMQ_PULL};
    std::string endpoint;
    std::string cancel_endpoint;
    std::atomic<bool> stop{false};
    std::thread thread;
    std::atomic<std::uint32_t> last_cancel{0};

    bool Start() {
        if (!router.Valid() || !pull.Valid()) return false;
        if (router.Bind("tcp://127.0.0.1:*") != 0) return false;
        if (pull.Bind("tcp://127.0.0.1:*") != 0) return false;
        char ep[256];
        std::size_t len = sizeof(ep);
        router.GetSockOpt(ZMQ_LAST_ENDPOINT, ep, &len);
        endpoint = ep;
        len = sizeof(ep);
        pull.GetSockOpt(ZMQ_LAST_ENDPOINT, ep, &len);
        cancel_endpoint = ep;
        thread = std::thread([this] { Loop(); });
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        return true;
    }

    void Loop() {
        while (!stop.load()) {
            zmq_pollitem_t items[2] = {
                {router.Get(), 0, ZMQ_POLLIN, 0},
                {pull.Get(), 0, ZMQ_POLLIN, 0},
            };
            const int n = zmq_poll(items, 2, 20);
            if (n <= 0) continue;
            if (items[1].revents & ZMQ_POLLIN) {
                zmq_msg_t m;
                zmq_msg_init(&m);
                if (zmq_msg_recv(&m, pull.Get(), 0) >= 0) {
                    std::uint32_t id = 0;
                    const std::span<const std::byte> f(
                        static_cast<const std::byte*>(zmq_msg_data(&m)),
                        zmq_msg_size(&m));
                    if (DecodeFabricCancel(f, id) == WireStatus::kOk)
                        last_cancel.store(id);
                }
                zmq_msg_close(&m);
            }
            if (items[0].revents & ZMQ_POLLIN) {
                zmq_msg_t ident;
                zmq_msg_init(&ident);
                if (zmq_msg_recv(&ident, router.Get(), 0) < 0) {
                    zmq_msg_close(&ident);
                    continue;
                }
                zmq_msg_t payload;
                zmq_msg_init(&payload);
                zmq_msg_recv(&payload, router.Get(), 0);
                std::uint32_t rid = 0;
                {
                    const std::span<const std::byte> f(
                        static_cast<const std::byte*>(zmq_msg_data(&payload)),
                        zmq_msg_size(&payload));
                    std::uint32_t nonce = 0;
                    if (DecodeFabricPing(f, nonce) == WireStatus::kOk) {
                        Send(ident, EncodeFabricPong(nonce));
                        zmq_msg_close(&payload);
                        zmq_msg_close(&ident);
                        continue;
                    }
                    FabricRequestView v;
                    if (DecodeFabricRequest(f, v) == WireStatus::kOk)
                        rid = v.request_id;
                }
                zmq_msg_close(&payload);
                Send(ident, EncodeFabricTokenBatch(
                                rid, std::vector<std::int32_t>{5, 6}));
                Send(ident, EncodeFabricDone(rid, true, 2u, 1.0, 2.0, 3.0));
                zmq_msg_close(&ident);
            }
        }
    }

    void Send(zmq_msg_t& ident, const std::vector<std::byte>& body) {
        zmq_msg_t part;
        zmq_msg_init_size(&part, zmq_msg_size(&ident));
        std::memcpy(zmq_msg_data(&part), zmq_msg_data(&ident), zmq_msg_size(&ident));
        zmq_msg_send(&part, router.Get(), ZMQ_SNDMORE);
        zmq_msg_t b;
        zmq_msg_init_size(&b, body.size());
        std::memcpy(zmq_msg_data(&b), body.data(), body.size());
        zmq_msg_send(&b, router.Get(), 0);
    }

    void Stop() {
        stop.store(true);
        if (thread.joinable()) thread.join();
    }
};

}  // namespace

TEST_CASE("fabric_stream_client: streams batch then done",
          "[fabric][client][stream]") {
    FakeServer srv;
    REQUIRE(srv.Start());
    FabricStreamClient client;
    REQUIRE(client.Connect(srv.endpoint.c_str(), srv.cancel_endpoint.c_str()) == 0);
    REQUIRE(client.Submit(11u, std::vector<std::int32_t>{1, 2, 3}, 0u));

    FabricStreamFrame f;
    REQUIRE(client.Recv(f, 2000) == FabricStreamClient::RecvStatus::kOk);
    CHECK(f.type == FabricFrameType::kTokenBatch);
    CHECK(f.request_id == 11u);
    REQUIRE(f.tokens.size() == 2u);
    CHECK(f.tokens[1] == 6);

    REQUIRE(client.Recv(f, 2000) == FabricStreamClient::RecvStatus::kOk);
    CHECK(f.type == FabricFrameType::kDone);
    CHECK(f.eos_terminated);
    CHECK(f.total_tokens == 2u);
    srv.Stop();
}

TEST_CASE("fabric_stream_client: cancel reaches the PULL channel",
          "[fabric][client][cancel]") {
    FakeServer srv;
    REQUIRE(srv.Start());
    FabricStreamClient client;
    REQUIRE(client.Connect(srv.endpoint.c_str(), srv.cancel_endpoint.c_str()) == 0);
    client.Cancel(99u);
    for (int i = 0; i < 200 && srv.last_cancel.load() != 99u; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    CHECK(srv.last_cancel.load() == 99u);
    srv.Stop();
}

TEST_CASE("fabric_stream_client: recv times out when idle",
          "[fabric][client][timeout]") {
    FakeServer srv;
    REQUIRE(srv.Start());
    FabricStreamClient client;
    REQUIRE(client.Connect(srv.endpoint.c_str(), srv.cancel_endpoint.c_str()) == 0);
    FabricStreamFrame f;
    CHECK(client.Recv(f, 50) == FabricStreamClient::RecvStatus::kTimeout);
    srv.Stop();
}

TEST_CASE("fabric_stream_client: probe pings a live backend",
          "[fabric][client][probe]") {
    // Unconnected / no probe channel -> false.
    FabricStreamClient dead;
    CHECK_FALSE(dead.Probe(50));

    FakeServer srv;
    REQUIRE(srv.Start());
    FabricStreamClient client;
    REQUIRE(client.Connect(srv.endpoint.c_str(), srv.cancel_endpoint.c_str()) == 0);
    CHECK(client.Probe(2000));
    srv.Stop();
}
