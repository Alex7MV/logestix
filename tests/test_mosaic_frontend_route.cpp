// TDD Phase 5: TaskMosaic CE static SPA frontend — in-memory router harness.
//
// Verifies that the lsxnetwork static router serves the compiled SPA bundle
// bytes directly from memory: (1) every embedded asset is 64-byte cache-line
// aligned, (2) GET / yields exactly 200 OK with byte-for-byte asset content,
// (3) a simultaneous multi-threaded hit-rate surge returns 200 on every hit
// with zero content drift and zero alignment loss, and (4) the SSE
// /v1/chat/completions stream frames a 200 event-stream. RED/GREEN: any
// misaligned pointer, wrong status, or corrupted byte marks the case RED.

#include <catch2/catch_test_macros.hpp>

#include <lsx_embedded_www.h>        // generated: embedded www/ bundle
#include <lsxnetwork/http_router.h>
#include <lsxnetwork/spa_assets.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

using lsxnetwork::HttpReply;
using lsxnetwork::HttpRouter;
using lsxnetwork::spa::Asset;
using lsxnetwork::spa::FindSpaAsset;
using lsxnetwork::spa::GetSpaAssetCount;
using lsxnetwork::spa::GetSpaAssets;

namespace {

constexpr std::size_t kCacheLine = 64;

// Normalize a URL path the same way the router does (strip query/fragment).
std::string_view StripQuery(std::string_view path) {
    const std::size_t q = path.find_first_of("?#");
    return q == std::string_view::npos ? path : path.substr(0, q);
}

}  // namespace

TEST_CASE("spa: every crafted asset is 64-byte cache-line aligned (GREEN)",
          "[frontend][spa][align]")
{
    const std::size_t n = GetSpaAssetCount();
    REQUIRE(n >= 6);  // index.html + styles.css + 4 JS modules
    const Asset* a = GetSpaAssets();
    for (std::size_t i = 0; i < n; ++i) {
        CAPTURE(a[i].path);
        REQUIRE(a[i].path != nullptr);
        REQUIRE(a[i].data != nullptr);
        CHECK(reinterpret_cast<std::uintptr_t>(a[i].data) % kCacheLine == 0);  // RED if misaligned
        CHECK(a[i].size > 0);
        CHECK(a[i].mime != nullptr);
    }
}

TEST_CASE("spa: default document and every asset are indexable by path",
          "[frontend][spa][lookup]")
{
    // The index route (path "/") must be present for the SPA entry point.
    const Asset* root = FindSpaAsset("/");
    REQUIRE(root != nullptr);
    CHECK(std::string_view(root->mime).find("text/html") != std::string_view::npos);
    CHECK(root->size > 64);  // a real HTML document, not a stub

    CHECK(FindSpaAsset("/js/app.js") != nullptr);
    CHECK(FindSpaAsset("/css/styles.css") != nullptr);
    CHECK(FindSpaAsset("/js/modules/mosaic.js") != nullptr);
    // Unknown paths resolve to nothing (router then 404s).
    CHECK(FindSpaAsset("/no/such/tile") == nullptr);
}

TEST_CASE("router: GET / returns 200 with byte-for-byte embedded HTML (GREEN)",
          "[frontend][router][static]")
{
    HttpRouter router;

    const Asset* root = FindSpaAsset("/");
    REQUIRE(root != nullptr);

    HttpReply reply;
    const int status = router.Handle("GET", "/", "", reply);

    CHECK(status == 200);
    CHECK(reply.status == 200);
    CHECK(reply.reason == "OK");
    CHECK(std::string_view(reply.content_type).find("text/html") != std::string_view::npos);
    CHECK(reply.chunked == false);
    CHECK(reply.body_size == root->size);
    REQUIRE(reply.body != nullptr);
    // Body aliases the aligned embedded bytes — no corruption, no copy.
    CHECK(std::memcmp(reply.body, root->data, root->size) == 0);
}

TEST_CASE("router: every embedded asset is served 200 with exact bytes (GREEN)",
          "[frontend][router][static][assets]")
{
    HttpRouter router;
    const std::size_t n = GetSpaAssetCount();
    const Asset* a = GetSpaAssets();
    for (std::size_t i = 0; i < n; ++i) {
        HttpReply reply;
        const int s = router.Handle("GET", a[i].path, "", reply);
        CAPTURE(a[i].path);
        REQUIRE(s == 200);
        REQUIRE(reply.body_size == a[i].size);
        REQUIRE(std::memcmp(reply.body, a[i].data, a[i].size) == 0);
    }
}

TEST_CASE("router: unknown static path 404s without crashing", "[frontend][router][404]")
{
    HttpRouter router;
    HttpReply reply;
    CHECK(router.Handle("GET", "/missing.css", "", reply) == 404);
    CHECK(router.Handle("GET", "/", "?fragment", reply) == 200);  // query stripped
}

TEST_CASE("router: multi-threaded hit-rate surge stays 200, aligned, un-corrupted (GREEN)",
          "[frontend][router][concurrency][surge]")
{
    HttpRouter router;
    const Asset* root = FindSpaAsset("/");
    REQUIRE(root != nullptr);

    constexpr int kThreads = 8;
    constexpr int kHitsPerThread = 6000;
    constexpr std::int64_t kTotal = kThreads * kHitsPerThread;

    std::atomic<std::int64_t> ok{0};
    std::atomic<std::int64_t> non200{0};
    std::atomic<std::int64_t> misaligned{0};
    std::atomic<std::int64_t> corrupted{0};

    std::vector<std::thread> workers;
    workers.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        workers.emplace_back([&] {
            for (int h = 0; h < kHitsPerThread; ++h) {
                HttpReply reply;
                const int s = router.Handle("GET", "/", "", reply);
                if (s == 200) {
                    ok.fetch_add(1, std::memory_order_relaxed);
                } else {
                    non200.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                // Perfect memory alignment indicator must hold under load.
                if (reinterpret_cast<std::uintptr_t>(reply.body) % kCacheLine != 0) {
                    misaligned.fetch_add(1, std::memory_order_relaxed);
                }
                if (reply.body_size != root->size ||
                    std::memcmp(reply.body, root->data, root->size) != 0) {
                    corrupted.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    for (auto& w : workers) w.join();

    CHECK(ok.load() == kTotal);                      // every hit 200 OK
    CHECK(non200.load() == 0);                       // zero failure responses
    CHECK(misaligned.load() == 0);                   // alignment never slips
    CHECK(corrupted.load() == 0);                    // bytes never drift
}

TEST_CASE("router: SSE /v1/chat/completions frames 200 event-stream (GREEN)",
          "[frontend][router][stream][sse]")
{
    HttpRouter router;
    const std::string body = R"({"id":"chatcmpl-abc","model":"mosaic","stream":true})";

    std::string frames;
    HttpReply head;
    const int status = router.HandleStream(
        "/v1/chat/completions", body, head,
        [&](std::string_view frame) {
            frames.append(frame.data(), frame.size());
            return true;
        });

    REQUIRE(status == 200);
    CHECK(head.status == 200);
    CHECK(head.chunked == true);
    CHECK(std::string_view(head.content_type).find("text/event-stream") != std::string_view::npos);

    // Each frame is an SSE `data:` payload; the stream closes with [DONE].
    CHECK(frames.find("data: ") == 0);
    CHECK(frames.find("\n\n", frames.find("data: ") + 6) != std::string_view::npos);
    CHECK(frames.find("chatcmpl-abc") != std::string_view::npos);
    CHECK(frames.find("data: [DONE]") != std::string_view::npos);
    CHECK(frames.find("finish_reason") != std::string_view::npos);
}

TEST_CASE("router: stream aborts cleanly when the client disconnects (GREEN)",
          "[frontend][router][stream][abort]")
{
    HttpRouter router;
    std::size_t yielded = 0;
    HttpReply head;
    // Emit returns false from the very first frame -> the router must stop and
    // return 200 (client-closed is not an HTTP error).
    const int status = router.HandleStream(
        "/v1/chat/completions", "{}", head,
        [&](std::string_view) {
            ++yielded;
            return false;
        });
    CHECK(status == 200);
    CHECK(head.status == 200);
    CHECK(yielded == 1);
}

TEST_CASE("router: stream on a non-chat route 404s", "[frontend][router][stream][404]")
{
    HttpRouter router;
    HttpReply head;
    CHECK(router.HandleStream("/", "{}", head, [](std::string_view) { return false; }) == 404);
    CHECK(head.status == 404);
    CHECK(std::string_view(head.content_type).find("text/plain") != std::string_view::npos);
}
