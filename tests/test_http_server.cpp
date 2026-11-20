#include <catch2/catch_test_macros.hpp>

#include <lsxnetwork/http_server.h>

#include <atomic>
#include <filesystem>
#include <iterator>
#include <chrono>
#include <memory>
#include <string>
#include <thread>

#include <httplib.h>

using lsxnetwork::HttpRequest;
using lsxnetwork::HttpResponse;
using lsxnetwork::HttpServer;
using lsxnetwork::StreamSource;

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

#if !defined(_WIN32)
namespace {
// Open descriptors of THIS process. A listening socket or an httplib worker
// that survives Stop() shows up here; RSS would not, because the pool is
// small and the allocator reuses the same pages.
int OpenFdCount() { return static_cast<int>(std::distance(
    std::filesystem::directory_iterator("/proc/self/fd"),
    std::filesystem::directory_iterator())); }
}  // namespace

TEST_CASE("http_server: start/stop cycles leak no descriptors (A3.10)",
          "[http][server][leak]") {
    // Every cycle builds a fresh HttpServer, serves one request and stops.
    // A leaked listening socket, a detached worker or an unclosed client
    // connection ratchets the fd count up by at least one per cycle.
    const auto cycle = [](int port) {
        Fixture fx;
        fx.Start();
        if (port != 0) REQUIRE(fx.BoundPort() > 0);
        httplib::Client cli("127.0.0.1", fx.BoundPort());
        auto res = cli.Get("/health");
        REQUIRE(res);
        CHECK(res->status == 200);
        fx.Stop();
    };
    cycle(1);   // warm-up: settles one-time allocations (logger, TLS, pool)
    const int baseline = OpenFdCount();
    for (int i = 0; i < 5; ++i) {
        cycle(1);
        INFO("cycle " << i << ": fds " << OpenFdCount() << " vs baseline "
                      << baseline);
        REQUIRE(OpenFdCount() <= baseline);
    }
}

TEST_CASE("http_server: the concrete port is reusable after Stop (A3.10)",
          "[http][server][leak]") {
    // Stop() must release the listening socket: binding the SAME concrete
    // port again has to succeed. A leaked listener fails the second bind.
    int port = 0;
    {
        Fixture fx;
        fx.Start();
        port = fx.BoundPort();
        REQUIRE(port > 0);
        fx.Stop();
    }
    for (int i = 0; i < 3; ++i) {
        Fixture fx;
        fx.server.Get("/health", [](const HttpRequest&, HttpResponse& r) {
            r.status = 200;
        });
        std::thread t([&] { fx.server.Listen("127.0.0.1", port); });
        while (fx.server.BoundPort() == 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        INFO("rebind cycle " << i << " on port " << port);
        REQUIRE(fx.server.BoundPort() == port);
        fx.server.Stop();
        if (t.joinable()) t.join();
    }
}
#endif
