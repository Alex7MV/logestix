// FabricServer (v2 ROUTER/cancel) <-> FabricStreamClient with a fake IModel.
#include <catch2/catch_test_macros.hpp>

#include "fabric_server.h"

#include "lsxcommon/model.h"
#include "lsxnetwork/fabric_stream_client.h"
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
        for (int i = 0; i < req.max_gen_tokens; ++i) {
            if (req.abort_flag && req.abort_flag->load()) break;
            r.gen_tokens.push_back(1000 + i);
            if (req.token_callback) {
                lsxcommon::TokenChunk c;
                c.token_id = static_cast<std::uint32_t>(1000 + i);
                c.step_t = static_cast<std::uint32_t>(i);
                req.token_callback(c);
            }
            if (delay_ms > 0)
                std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        }
        r.num_tokens = static_cast<int>(r.gen_tokens.size());
        r.eos_terminated = !r.gen_tokens.empty();
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
    int delay_ms = 0;
    bool fail = false;

private:
    lsxcommon::ArrowTokenizerDb tok_;
    std::shared_mutex mtx_;
};

struct Fixture {
    FakeModel model;
    std::atomic<bool> stop{false};
    std::atomic<bool> ready{false};
    std::string router_ep;
    std::string cancel_ep;
    lsxfabric::FabricServer server{model, 4u};
    std::thread thread;

    void Start() {
        thread = std::thread([this] {
            server.Run("tcp://127.0.0.1:*", "tcp://127.0.0.1:*", stop, &ready,
                       &router_ep, &cancel_ep);
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

TEST_CASE("fabric_server v2: streams batches then done", "[fabric][server][v2]") {
    Fixture fx;
    fx.Start();
    REQUIRE_FALSE(fx.router_ep.empty());
    lsxnetwork::FabricStreamClient client;
    REQUIRE(client.Connect(fx.router_ep.c_str(), fx.cancel_ep.c_str()) == 0);
    REQUIRE(client.Submit(7u, std::vector<std::int32_t>{1, 2, 3}, 0u));

    std::vector<std::int32_t> all;
    lsxnetwork::FabricStreamFrame f;
    bool done = false;
    for (int i = 0; i < 100 && !done; ++i) {
        REQUIRE(client.Recv(f, 2000) ==
                lsxnetwork::FabricStreamClient::RecvStatus::kOk);
        CHECK(f.request_id == 7u);
        if (f.type == lsxnetwork::FabricFrameType::kTokenBatch)
            all.insert(all.end(), f.tokens.begin(), f.tokens.end());
        else if (f.type == lsxnetwork::FabricFrameType::kDone)
            done = true;
    }
    CHECK(done);
    REQUIRE(all.size() == 4u);  // server default max_gen
    CHECK(all[0] == 1000);
    CHECK(fx.model.last_input == std::vector<std::int32_t>{1, 2, 3});
    CHECK(fx.model.last_max_gen == 4);
    CHECK(f.total_tokens == 4u);
    CHECK(f.ttft_ms == 1.5);
    fx.Stop();
}

TEST_CASE("fabric_server v2: per-request max_gen overrides default",
          "[fabric][server][v2]") {
    Fixture fx;
    fx.Start();
    lsxnetwork::FabricStreamClient client;
    REQUIRE(client.Connect(fx.router_ep.c_str(), fx.cancel_ep.c_str()) == 0);
    REQUIRE(client.Submit(1u, std::vector<std::int32_t>{1}, 7u));
    lsxnetwork::FabricStreamFrame f;
    std::size_t count = 0;
    for (;;) {
        REQUIRE(client.Recv(f, 2000) ==
                lsxnetwork::FabricStreamClient::RecvStatus::kOk);
        if (f.type == lsxnetwork::FabricFrameType::kDone) break;
        if (f.type == lsxnetwork::FabricFrameType::kTokenBatch)
            count += f.tokens.size();
    }
    CHECK(fx.model.last_max_gen == 7);
    CHECK(count == 7u);
    fx.Stop();
}

TEST_CASE("fabric_server v2: infer failure maps to an error frame",
          "[fabric][server][v2]") {
    Fixture fx;
    fx.model.fail = true;
    fx.Start();
    lsxnetwork::FabricStreamClient client;
    REQUIRE(client.Connect(fx.router_ep.c_str(), fx.cancel_ep.c_str()) == 0);
    REQUIRE(client.Submit(1u, std::vector<std::int32_t>{1}, 2u));
    lsxnetwork::FabricStreamFrame f;
    REQUIRE(client.Recv(f, 2000) ==
            lsxnetwork::FabricStreamClient::RecvStatus::kOk);
    CHECK(f.type == lsxnetwork::FabricFrameType::kError);
    CHECK(f.error == "boom");
    fx.Stop();
}

TEST_CASE("fabric_server v2: cancel aborts an in-flight generation",
          "[fabric][server][v2][cancel]") {
    Fixture fx;
    fx.model.delay_ms = 5;
    fx.Start();
    lsxnetwork::FabricStreamClient client;
    REQUIRE(client.Connect(fx.router_ep.c_str(), fx.cancel_ep.c_str()) == 0);
    REQUIRE(client.Submit(55u, std::vector<std::int32_t>{1}, 1000u));

    lsxnetwork::FabricStreamFrame f;
    REQUIRE(client.Recv(f, 2000) ==
            lsxnetwork::FabricStreamClient::RecvStatus::kOk);
    REQUIRE(f.type == lsxnetwork::FabricFrameType::kTokenBatch);
    client.Cancel(55u);

    std::size_t count = f.tokens.size();
    for (;;) {
        REQUIRE(client.Recv(f, 5000) ==
                lsxnetwork::FabricStreamClient::RecvStatus::kOk);
        if (f.type == lsxnetwork::FabricFrameType::kDone) break;
        if (f.type == lsxnetwork::FabricFrameType::kError) break;
        count += f.tokens.size();
    }
    CHECK(count < 1000u);  // aborted well before max_gen
    fx.Stop();
}

TEST_CASE("fabric_server v2: answers health probes off the inference path",
          "[fabric][server][v2][probe]") {
    Fixture fx;
    fx.Start();
    lsxnetwork::FabricStreamClient client;
    REQUIRE(client.Connect(fx.router_ep.c_str(), fx.cancel_ep.c_str()) == 0);
    CHECK(client.Probe(2000));
    CHECK(fx.model.calls == 0);  // a probe must never invoke inference
    fx.Stop();
}

TEST_CASE("fabric_server v2: malformed request yields an error frame",
          "[fabric][server][v2]") {
    Fixture fx;
    fx.Start();
    lsxnetwork::ZmqContext ctx;
    lsxnetwork::ZmqSocket dealer(ctx, ZMQ_DEALER);
    REQUIRE(dealer.Valid());
    REQUIRE(dealer.Connect(fx.router_ep.c_str()) == 0);
    const char garbage[] = "nope";
    REQUIRE(dealer.Send(garbage, sizeof(garbage) - 1, 0) >= 0);
    zmq_msg_t msg;
    zmq_msg_init(&msg);
    REQUIRE(zmq_msg_recv(&msg, dealer.Get(), 0) >= 0);
    const std::span<const std::byte> f(
        static_cast<const std::byte*>(zmq_msg_data(&msg)), zmq_msg_size(&msg));
    lsxnetwork::FabricStreamFrame frame;
    REQUIRE(lsxnetwork::DecodeFabricStreamFrame(f, frame) ==
            lsxnetwork::WireStatus::kOk);
    CHECK(frame.type == lsxnetwork::FabricFrameType::kError);
    CHECK(fx.model.calls == 0);
    zmq_msg_close(&msg);
    fx.Stop();
}
