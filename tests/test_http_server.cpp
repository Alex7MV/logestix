#include <catch2/catch_test_macros.hpp>

#include <lsxnetwork/http_server.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>

#include <httplib.h>

#include <string_view>

using lsxnetwork::HttpRequest;
using lsxnetwork::HttpResponse;
using lsxnetwork::HttpServer;
using lsxnetwork::StreamSource;

TEST_CASE("http_server: pinned cpp-httplib version",
          "[http][dependencies][versions]") {
    STATIC_REQUIRE(std::string_view(CPPHTTPLIB_VERSION) == "0.58.0");
}

namespace {
struct Fixture {
    HttpServer server;
    std::thread thread;
    void Start() {
        server.Get("/health", [](const HttpRequest&, HttpResponse& r) {
            r.status = 200;
            r.body = "{\"status\":\"ok\"}";
        });
        thread = std::thread([this] { server.Listen("127.0.0.1", 0); });
        while (server.BoundPort() == 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    void Stop() {
        server.Stop();
        if (thread.joinable()) thread.join();
    }
    int BoundPort() const { return server.BoundPort(); }
};
}  // namespace

TEST_CASE("http_server: serves a JSON route", "[http][server]") {
    Fixture fx;
    fx.Start();
    httplib::Client cli("127.0.0.1", fx.BoundPort());
    auto res = cli.Get("/health");
    REQUIRE(res);
    CHECK(res->status == 200);
    CHECK(res->body == "{\"status\":\"ok\"}");
    fx.Stop();
}

TEST_CASE("http_server: streams SSE chunks", "[http][server][sse]") {
    HttpServer server;
    server.PostStream("/sse", [](const HttpRequest&, HttpResponse& head)
                                 -> std::unique_ptr<StreamSource> {
        head.content_type = "text/event-stream";
        struct Src : StreamSource {
            int i = 0;
            bool Next(std::string& out) override {
                if (i >= 2) return false;
                out = "data: " + std::to_string(i++) + "\n\n";
                return true;
            }
            void Cancel() override {}
        };
        return std::make_unique<Src>();
    });
    std::thread t([&] { server.Listen("127.0.0.1", 0); });
    while (server.BoundPort() == 0)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

    httplib::Client cli("127.0.0.1", server.BoundPort());
    auto res = cli.Post("/sse", "", "application/json");
    const int err = res ? 0 : static_cast<int>(res.error());
    std::string body = res ? res->body : std::string{};
    const int status = res ? res->status : -1;
    server.Stop();
    t.join();
    REQUIRE(err == 0);
    CHECK(status == 200);
    CHECK(body.find("data: 0") != std::string::npos);
    CHECK(body.find("data: 1") != std::string::npos);
}
