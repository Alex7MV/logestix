// FabricServer <-> FabricClient integration over TCP, with a fake IModel.
// No CUDA, no real weights: label fast.
#include <catch2/catch_test_macros.hpp>

#include "fabric_server.h"

#include "lsxcommon/model.h"
#include "lsxnetwork/fabric_client.h"
#include "lsxnetwork/fabric_wire.h"
#include "lsxnetwork/zmq_context.h"

#include <zmq.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <shared_mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace {

class FakeModel : public lsxcommon::IModel {
public:
    bool Load() override { return true; }
    void Unload() override {}
    lsxcommon::SafetensorsModel Inspect() override { return {}; }

    lsxcommon::gpu::InferenceResult Infer(
        const lsxcommon::ModelRequest& req) override {
        last_input = req.input_ids;
        last_max_gen = req.max_gen_tokens;
        ++calls;
        lsxcommon::gpu::InferenceResult r;
        if (fail) {
            r.ok = false;
            r.error_msg = "boom";
            return r;
        }
        r.ok = true;
        for (int i = 0; i < req.max_gen_tokens; ++i)
            r.gen_tokens.push_back(1000 + i);
        r.num_tokens = static_cast<int>(r.gen_tokens.size());
        r.eos_terminated = req.max_gen_tokens >= 3;
        r.ttft_ms = 1.5;
        r.total_time_ms = 2.5;
        r.tokens_per_sec = 3.5;
        return r;
    }

    lsxcommon::gpu::InferenceResult SelfCheck(
        const lsxcommon::ModelRequest& req) override {
        return Infer(req);
    }

    const std::string& Type() const override {
        static const std::string kType = "fake";
        return kType;
    }
    lsxcommon::ArrowTokenizerDb& Tokenizer() override { return tok_; }
    std::shared_mutex& ResidencyMutex() override { return mtx_; }

    std::vector<std::int32_t> last_input;
    int last_max_gen = -1;
    int calls = 0;
    bool fail = false;

private:
    lsxcommon::ArrowTokenizerDb tok_;
    std::shared_mutex mtx_;
};

struct ServerFixture {
    FakeModel model;
    std::atomic<bool> stop{false};
    std::atomic<bool> ready{false};
    std::string endpoint;
    lsxfabric::FabricServer server{model, 4u};
    std::thread thread;

    void Start() {
        thread = std::thread([this] {
            server.Run("tcp://127.0.0.1:*", stop, &ready, &endpoint);
        });
        while (!ready.load())
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    void Stop() {
        stop.store(true);
        if (thread.joinable()) thread.join();
    }
};

}  // namespace

TEST_CASE("fabric_server: serves a request and returns stats",
          "[fabric][server]") {
    ServerFixture fx;
    fx.Start();
    REQUIRE_FALSE(fx.endpoint.empty());

    lsxnetwork::FabricClient client;
    REQUIRE(client.Connect(fx.endpoint.c_str()) == 0);
    lsxnetwork::FabricResponse resp;
    const std::vector<std::int32_t> ids{4, 5, 6};
    REQUIRE(client.Infer(ids, 0u, resp, 2000));
    CHECK(resp.status ==
          static_cast<std::int32_t>(lsxnetwork::FabricStatus::kOk));
    CHECK(fx.model.last_input == ids);
    CHECK(fx.model.last_max_gen == 4);  // 0 => server default
    REQUIRE(resp.tokens.size() == 4);
    CHECK(resp.tokens[0] == 1000);
    CHECK(resp.eos_terminated);
    CHECK(resp.ttft_ms == 1.5);
    fx.Stop();
}

TEST_CASE("fabric_server: per-request max_gen overrides default",
          "[fabric][server]") {
    ServerFixture fx;
    fx.Start();
    lsxnetwork::FabricClient client;
    REQUIRE(client.Connect(fx.endpoint.c_str()) == 0);
    lsxnetwork::FabricResponse resp;
    const std::vector<std::int32_t> ids{1};
    REQUIRE(client.Infer(ids, 7u, resp, 2000));
    CHECK(fx.model.last_max_gen == 7);
    REQUIRE(resp.tokens.size() == 7);
    fx.Stop();
}

TEST_CASE("fabric_server: infer failure maps to kInferFailed",
          "[fabric][server]") {
    ServerFixture fx;
    fx.model.fail = true;
    fx.Start();
    lsxnetwork::FabricClient client;
    REQUIRE(client.Connect(fx.endpoint.c_str()) == 0);
    lsxnetwork::FabricResponse resp;
    const std::vector<std::int32_t> ids{1};
    REQUIRE(client.Infer(ids, 2u, resp, 2000));
    CHECK(resp.status ==
          static_cast<std::int32_t>(lsxnetwork::FabricStatus::kInferFailed));
    CHECK(resp.error == "boom");
    fx.Stop();
}

TEST_CASE("fabric_server: malformed frame maps to kBadRequest",
          "[fabric][server]") {
    ServerFixture fx;
    fx.Start();

    lsxnetwork::ZmqContext ctx;
    lsxnetwork::ZmqSocket sock(ctx, ZMQ_REQ);
    REQUIRE(sock.Valid());
    REQUIRE(sock.Connect(fx.endpoint.c_str()) == 0);
    const char garbage[] = "not-a-fabric-frame";
    REQUIRE(sock.Send(garbage, sizeof(garbage) - 1, 0) >= 0);

    zmq_msg_t msg;
    zmq_msg_init(&msg);
    REQUIRE(zmq_msg_recv(&msg, sock.Get(), 0) >= 0);
    const std::span<const std::byte> reply(
        static_cast<const std::byte*>(zmq_msg_data(&msg)),
        static_cast<std::size_t>(zmq_msg_size(&msg)));
    lsxnetwork::FabricResponse resp;
    REQUIRE(lsxnetwork::DecodeFabricResponse(reply, resp) ==
            lsxnetwork::WireStatus::kOk);
    CHECK(resp.status ==
          static_cast<std::int32_t>(lsxnetwork::FabricStatus::kBadRequest));
    zmq_msg_close(&msg);
    CHECK(fx.model.calls == 0);
    fx.Stop();
}
