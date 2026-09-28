#include <catch2/catch_test_macros.hpp>

#include "chat_bridge.h"
#include "framer.h"

#include <httplib.h>
#include <zmq.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include <lsxnetwork/fabric_wire.h>
#include <lsxnetwork/zmq_context.h>

using namespace taskmosaic;

namespace {

// Minimal ROUTER fake backend: emits token ids 10, 11 then a done frame.
struct FakeBackend {
    lsxnetwork::ZmqContext ctx;
    lsxnetwork::ZmqSocket router{ctx, ZMQ_ROUTER};
    lsxnetwork::ZmqSocket pull{ctx, ZMQ_PULL};
    std::string router_ep;
    std::string cancel_ep;
    std::atomic<bool> stop{false};
    std::thread thread;

    bool Start() {
        if (router.Bind("tcp://127.0.0.1:*") != 0) return false;
        if (pull.Bind("tcp://127.0.0.1:*") != 0) return false;
        char ep[256];
        std::size_t len = sizeof(ep);
        router.GetSockOpt(ZMQ_LAST_ENDPOINT, ep, &len);
        router_ep = ep;
        len = sizeof(ep);
        pull.GetSockOpt(ZMQ_LAST_ENDPOINT, ep, &len);
        cancel_ep = ep;
        const int rcvtimeo = 50;
        router.SetSockOpt(ZMQ_RCVTIMEO, &rcvtimeo, sizeof(rcvtimeo));
        thread = std::thread([this] { Loop(); });
        return true;
    }

    void Send(zmq_msg_t& ident, const std::vector<std::byte>& body) {
        zmq_msg_t part;
        zmq_msg_init_size(&part, zmq_msg_size(&ident));
        std::memcpy(zmq_msg_data(&part), zmq_msg_data(&ident),
                    zmq_msg_size(&ident));
        zmq_msg_send(&part, router.Get(), ZMQ_SNDMORE);
        zmq_msg_t b;
        zmq_msg_init_size(&b, body.size());
        std::memcpy(zmq_msg_data(&b), body.data(), body.size());
        zmq_msg_send(&b, router.Get(), 0);
    }

    void Loop() {
        while (!stop.load()) {
            zmq_msg_t ident;
            zmq_msg_init(&ident);
            const int n = zmq_msg_recv(&ident, router.Get(), 0);
            if (n < 0) {
                zmq_msg_close(&ident);
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }
            std::uint32_t rid = 0;
            {
                zmq_msg_t payload;
                zmq_msg_init(&payload);
                zmq_msg_recv(&payload, router.Get(), 0);
                const std::span<const std::byte> f(
                    static_cast<const std::byte*>(zmq_msg_data(&payload)),
                    zmq_msg_size(&payload));
                lsxnetwork::FabricRequestView v;
                if (lsxnetwork::DecodeFabricRequest(f, v) ==
                    lsxnetwork::WireStatus::kOk)
                    rid = v.request_id;
                zmq_msg_close(&payload);
            }
            Send(ident, lsxnetwork::EncodeFabricTokenBatch(
                            rid, std::vector<std::int32_t>{10}));
            Send(ident, lsxnetwork::EncodeFabricTokenBatch(
                            rid, std::vector<std::int32_t>{11}));
            Send(ident, lsxnetwork::EncodeFabricDone(rid, true, 2u, 1.0, 2.0,
                                                     3.0));
            zmq_msg_close(&ident);
        }
    }

    void Stop() {
        stop.store(true);
        /* unblock recv with a harmless empty request is unnecessary: the ROUTER
           recv times out via the sleep loop above. */
        if (thread.joinable()) thread.join();
    }
};

class FakeFramer : public PromptFramer {
public:
    bool Encode(std::string_view, std::string_view raw,
                std::vector<std::int32_t>& ids) override {
        prompts.emplace_back(raw);
        ids = {1, 2};
        return true;
    }
    bool EncodeConversation(std::string_view,
                            const lsxcommon::ChatConversation& conv,
                            std::vector<std::int32_t>& ids) override {
        has_tools = !conv.tools.empty();
        ids = {1, 2};
        return true;
    }
    bool ParseToolCalls(std::string_view, const std::vector<std::int32_t>&,
                        std::vector<lsxcommon::ToolCall>& out) override {
        out.clear();
        if (has_tools)
            out.push_back({"call_0", "get_weather", R"({"city":"Paris"})"});
        return true;
    }
    std::string Decode(std::string_view,
                       const std::vector<std::int32_t>&) override {
        return "hello";
    }
    std::unique_ptr<StreamingDecoder> MakeDecoder(std::string_view) override {
        struct D : StreamingDecoder {
            std::string Push(std::int32_t id) override {
                if (id == 10) return "he";
                if (id == 11) return "llo";
                return {};
            }
            std::string Flush() override { return {}; }
        };
        return std::make_unique<D>();
    }
    std::vector<std::string> prompts;
    bool has_tools = false;
};

struct BridgeFixture {
    FakeBackend backend;
    BridgeConfig cfg;
    FakeFramer framer;
    std::unique_ptr<ChatBridge> bridge;
    std::thread thread;

    bool Start() {
        if (!backend.Start()) return false;
        cfg.host = "127.0.0.1";
        cfg.port = 0;
        cfg.default_max_tokens = 16;
        cfg.request_timeout_ms = 5000;
        ModelConfig mc;
        mc.id = "m1";
        mc.model_path = "/fake";
        mc.backend.endpoint = backend.router_ep;
        mc.backend.cancel_endpoint = backend.cancel_ep;
        cfg.models.push_back(mc);
        bridge = std::make_unique<ChatBridge>(cfg, framer);
        std::string err;
        if (!bridge->Initialize(err)) return false;
        thread = std::thread([this] { bridge->Listen(); });
        while (bridge->BoundPort() == 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return true;
    }
    void Stop() {
        bridge->Stop();
        if (thread.joinable()) thread.join();
        backend.Stop();
    }
};

}  // namespace

TEST_CASE("bridge: non-streaming chat completion", "[taskmosaic][bridge][http]") {
    BridgeFixture fx;
    REQUIRE(fx.Start());
    httplib::Client cli("127.0.0.1", fx.bridge->BoundPort());
    auto res = cli.Post("/v1/chat/completions",
                        R"({"model":"m1","messages":[{"role":"user","content":"hi"}]})",
                        "application/json");
    REQUIRE(res);
    CHECK(res->status == 200);
    CHECK(res->body.find("\"content\":\"hello\"") != std::string::npos);
    CHECK(res->body.find("\"completion_tokens\":2") != std::string::npos);
    fx.Stop();
}

TEST_CASE("bridge: streaming chat completion emits SSE",
          "[taskmosaic][bridge][http]") {
    BridgeFixture fx;
    REQUIRE(fx.Start());
    httplib::Client cli("127.0.0.1", fx.bridge->BoundPort());
    auto res = cli.Post("/v1/chat/completions",
                        R"({"model":"m1","stream":true,"messages":[{"role":"user","content":"hi"}]})",
                        "application/json");
    REQUIRE(res);
    CHECK(res->status == 200);
    CHECK(res->body.find("data: ") != std::string::npos);
    CHECK(res->body.find("he") != std::string::npos);
    CHECK(res->body.find("[DONE]") != std::string::npos);
    fx.Stop();
}

TEST_CASE("bridge: models list, retrieve and 404",
          "[taskmosaic][bridge][http]") {
    BridgeFixture fx;
    REQUIRE(fx.Start());
    httplib::Client cli("127.0.0.1", fx.bridge->BoundPort());
    auto list = cli.Get("/v1/models");
    REQUIRE(list);
    CHECK(list->status == 200);
    CHECK(list->body.find("\"id\":\"m1\"") != std::string::npos);
    CHECK(list->body.find("\"object\":\"list\"") != std::string::npos);

    auto one = cli.Get("/v1/models/m1");
    REQUIRE(one);
    CHECK(one->status == 200);

    auto missing = cli.Get("/v1/models/nope");
    REQUIRE(missing);
    CHECK(missing->status == 404);
    fx.Stop();
}

TEST_CASE("bridge: health and ready", "[taskmosaic][bridge][http]") {
    BridgeFixture fx;
    REQUIRE(fx.Start());
    httplib::Client cli("127.0.0.1", fx.bridge->BoundPort());
    auto h = cli.Get("/health");
    REQUIRE(h);
    CHECK(h->status == 200);
    auto r = cli.Get("/ready");
    REQUIRE(r);
    CHECK(r->status == 200);
    fx.Stop();
}

TEST_CASE("bridge: tool calling returns tool_calls", "[taskmosaic][bridge][tools]") {
    BridgeFixture fx;
    REQUIRE(fx.Start());
    httplib::Client cli("127.0.0.1", fx.bridge->BoundPort());
    auto res = cli.Post(
        "/v1/chat/completions",
        R"({"model":"m1","messages":[{"role":"user","content":"weather?"}],
            "tools":[{"type":"function","function":{"name":"get_weather","parameters":{}}}]})",
        "application/json");
    REQUIRE(res);
    CHECK(res->status == 200);
    CHECK(res->body.find("\"tool_calls\"") != std::string::npos);
    CHECK(res->body.find("\"finish_reason\":\"tool_calls\"") != std::string::npos);
    fx.Stop();
}

TEST_CASE("bridge: streaming tool calling", "[taskmosaic][bridge][tools]") {
    BridgeFixture fx;
    REQUIRE(fx.Start());
    httplib::Client cli("127.0.0.1", fx.bridge->BoundPort());
    auto res = cli.Post(
        "/v1/chat/completions",
        R"({"model":"m1","stream":true,"messages":[{"role":"user","content":"weather?"}],
            "tools":[{"type":"function","function":{"name":"get_weather","parameters":{}}}]})",
        "application/json");
    REQUIRE(res);
    CHECK(res->status == 200);
    CHECK(res->body.find("\"tool_calls\"") != std::string::npos);
    CHECK(res->body.find("[DONE]") != std::string::npos);
    fx.Stop();
}

TEST_CASE("bridge: anthropic messages non-stream and count_tokens",
          "[taskmosaic][bridge][anthropic]") {
    BridgeFixture fx;
    REQUIRE(fx.Start());
    httplib::Client cli("127.0.0.1", fx.bridge->BoundPort());
    auto msg = cli.Post(
        "/v1/messages",
        R"({"model":"m1","max_tokens":16,"messages":[{"role":"user","content":"hi"}]})",
        "application/json");
    REQUIRE(msg);
    CHECK(msg->status == 200);
    CHECK(msg->body.find("\"type\":\"message\"") != std::string::npos);
    CHECK(msg->body.find("hello") != std::string::npos);
    CHECK(msg->body.find("\"stop_reason\":\"end_turn\"") != std::string::npos);

    auto ct = cli.Post(
        "/v1/messages/count_tokens",
        R"({"model":"m1","messages":[{"role":"user","content":"hi"}]})",
        "application/json");
    REQUIRE(ct);
    CHECK(ct->status == 200);
    CHECK(ct->body.find("\"input_tokens\":2") != std::string::npos);

    auto hello = cli.Get("/api/hello");
    REQUIRE(hello);
    CHECK(hello->status == 200);

    auto models = cli.Get("/v1/models",
                          httplib::Headers{{"anthropic-version", "2023-06-01"}});
    REQUIRE(models);
    CHECK(models->body.find("\"display_name\":\"m1\"") != std::string::npos);
    fx.Stop();
}

TEST_CASE("bridge: anthropic messages streaming event sequence",
          "[taskmosaic][bridge][anthropic]") {
    BridgeFixture fx;
    REQUIRE(fx.Start());
    httplib::Client cli("127.0.0.1", fx.bridge->BoundPort());
    auto res = cli.Post(
        "/v1/messages",
        R"({"model":"m1","stream":true,"max_tokens":16,
            "messages":[{"role":"user","content":"hi"}]})",
        "application/json");
    REQUIRE(res);
    CHECK(res->status == 200);
    CHECK(res->body.find("event: message_start") != std::string::npos);
    CHECK(res->body.find("content_block_delta") != std::string::npos);
    CHECK(res->body.find("event: message_stop") != std::string::npos);
    fx.Stop();
}
