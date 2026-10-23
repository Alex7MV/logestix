#include <catch2/catch_test_macros.hpp>

#include <lsxnetwork/fabric_wire.h>

#include <cstdint>
#include <span>
#include <string>
#include <vector>

using namespace lsxnetwork;

namespace {
std::span<const std::byte> AsBytes(const std::vector<std::byte>& v) {
    return std::span<const std::byte>(v.data(), v.size());
}
}  // namespace

TEST_CASE("fabric_wire v3: request round-trip with sampling",
          "[fabric][wire][v3]") {
    const std::vector<std::int32_t> ids{7, 8, 9, 10};
    RequestSampling sampling;
    sampling.temperature = 0.7f;
    sampling.top_p = 0.9f;
    sampling.top_k = 40;
    sampling.seed = 0x1234'5678'9ABCull;
    const auto enc = EncodeFabricRequest(42u, ids, 16u, sampling);
    CHECK(enc.size() == kFabricRequestHeaderBytes + 16u);
    FabricRequestView view;
    REQUIRE(DecodeFabricRequest(AsBytes(enc), view) == WireStatus::kOk);
    CHECK(view.request_id == 42u);
    CHECK(view.max_gen == 16u);
    CHECK(view.sampling.temperature == 0.7f);
    CHECK(view.sampling.top_p == 0.9f);
    CHECK(view.sampling.top_k == 40);
    CHECK(view.sampling.seed == 0x1234'5678'9ABCull);
    REQUIRE(view.tokens.size() == 4u);
    CHECK(view.tokens[3] == 10);
}

TEST_CASE("fabric_wire v3: default request is greedy", "[fabric][wire][v3]") {
    const auto enc = EncodeFabricRequest(1u, std::vector<std::int32_t>{5}, 1u);
    FabricRequestView view;
    REQUIRE(DecodeFabricRequest(AsBytes(enc), view) == WireStatus::kOk);
    CHECK(view.sampling.greedy());
    CHECK(view.sampling.temperature == 0.0f);
    CHECK(view.sampling.top_p == 1.0f);
    CHECK(view.sampling.top_k == 0);
}

TEST_CASE("fabric_wire: a v2 request still decodes as greedy",
          "[fabric][wire][v2]") {
    // Hand-built v2 frame: 24-byte header, magic/version 2/reqid/maxgen/ntok.
    const std::vector<std::int32_t> ids{11, 12};
    std::vector<std::byte> frame(kFabricRequestHeaderBytesV2 + 8u,
                                 std::byte{0});
    auto put = [&](std::size_t at, std::uint32_t v) {
        for (int i = 0; i < 4; ++i)
            frame[at + static_cast<std::size_t>(i)] =
                static_cast<std::byte>((v >> (8 * i)) & 0xFFu);
    };
    put(0, kFabricMagic);
    frame[4] = std::byte{static_cast<unsigned char>(kFabricVersion)};
    frame[5] = std::byte{0};
    put(8, 77u);
    put(12, 3u);
    put(16, 2u);
    put(kFabricRequestHeaderBytesV2, 11u);
    put(kFabricRequestHeaderBytesV2 + 4u, 12u);
    FabricRequestView view;
    REQUIRE(DecodeFabricRequest(AsBytes(frame), view) == WireStatus::kOk);
    CHECK(view.request_id == 77u);
    CHECK(view.sampling.greedy());
    REQUIRE(view.tokens.size() == 2u);
    CHECK(view.tokens[1] == 12);
}

TEST_CASE("fabric_wire v2: request rejects bad magic/version/trailing",
          "[fabric][wire][v2]") {
    auto enc = EncodeFabricRequest(1u, std::vector<std::int32_t>{1}, 0u);
    FabricRequestView view;
    auto bad = enc;
    bad[0] = std::byte{0};
    CHECK(DecodeFabricRequest(AsBytes(bad), view) == WireStatus::kBadMagic);

    bad = enc;
    bad[4] = std::byte{9};
    CHECK(DecodeFabricRequest(AsBytes(bad), view) == WireStatus::kBadVersion);

    bad = enc;
    bad.push_back(std::byte{0});
    CHECK(DecodeFabricRequest(AsBytes(bad), view) == WireStatus::kTrailing);

    std::vector<std::byte> trunc(enc.begin(), enc.begin() + 10);
    CHECK(DecodeFabricRequest(AsBytes(trunc), view) == WireStatus::kTruncated);
}

TEST_CASE("fabric_wire v2: token batch round-trip", "[fabric][wire][v2]") {
    const std::vector<std::int32_t> t{1000, 1001, 1002};
    FabricStreamFrame f;
    REQUIRE(DecodeFabricStreamFrame(AsBytes(EncodeFabricTokenBatch(5u, t)), f) ==
            WireStatus::kOk);
    CHECK(f.type == FabricFrameType::kTokenBatch);
    CHECK(f.request_id == 5u);
    CHECK(f.tokens == t);
}

TEST_CASE("fabric_wire v2: done round-trip", "[fabric][wire][v2]") {
    FabricStreamFrame f;
    REQUIRE(DecodeFabricStreamFrame(
                AsBytes(EncodeFabricDone(9u, true, 12u, 1.5, 2.5, 3.5)), f) ==
            WireStatus::kOk);
    CHECK(f.type == FabricFrameType::kDone);
    CHECK(f.request_id == 9u);
    CHECK(f.eos_terminated);
    CHECK(f.total_tokens == 12u);
    CHECK(f.ttft_ms == 1.5);
    CHECK(f.total_ms == 2.5);
    CHECK(f.tokens_per_sec == 3.5);
}

TEST_CASE("fabric_wire v2: error frame carries UTF-8", "[fabric][wire][v2]") {
    FabricStreamFrame f;
    REQUIRE(DecodeFabricStreamFrame(
                AsBytes(EncodeFabricError(3u, "boom")), f) == WireStatus::kOk);
    CHECK(f.type == FabricFrameType::kError);
    CHECK(f.request_id == 3u);
    CHECK(f.error == "boom");
}

TEST_CASE("fabric_wire v2: unknown frame type rejected", "[fabric][wire][v2]") {
    FabricStreamFrame f;
    auto enc = EncodeFabricTokenBatch(1u, std::vector<std::int32_t>{1});
    enc[6] = std::byte{0x7f};  // type lo byte
    CHECK(DecodeFabricStreamFrame(AsBytes(enc), f) == WireStatus::kBadType);
}

TEST_CASE("fabric_wire v2: cancel round-trip", "[fabric][wire][v2]") {
    std::uint32_t id = 0;
    REQUIRE(DecodeFabricCancel(AsBytes(EncodeFabricCancel(77u)), id) ==
            WireStatus::kOk);
    CHECK(id == 77u);
}

TEST_CASE("fabric_wire v2: empty token batch is valid", "[fabric][wire][v2]") {
    FabricStreamFrame f;
    REQUIRE(DecodeFabricStreamFrame(
                AsBytes(EncodeFabricTokenBatch(1u, {})), f) == WireStatus::kOk);
    CHECK(f.tokens.empty());
}

TEST_CASE("fabric_wire v2: out-param token batch matches the owning encoder",
          "[fabric][wire][v2]") {
    const std::vector<std::int32_t> t{4, 5, 6};
    std::vector<std::byte> out;
    // Reused across calls: stale longer content must be overwritten exactly.
    EncodeFabricTokenBatch(out, 1u, std::vector<std::int32_t>{1, 2, 3, 4, 5});
    EncodeFabricTokenBatch(out, 8u, t);
    CHECK(out == EncodeFabricTokenBatch(8u, t));
}

TEST_CASE("fabric_wire v2: ping/pong round-trip and separation",
          "[fabric][wire][v2]") {
    std::uint32_t nonce = 0;
    REQUIRE(DecodeFabricPing(AsBytes(EncodeFabricPing(123u)), nonce) ==
            WireStatus::kOk);
    CHECK(nonce == 123u);
    REQUIRE(DecodeFabricPong(AsBytes(EncodeFabricPong(456u)), nonce) ==
            WireStatus::kOk);
    CHECK(nonce == 456u);

    // A ping must not decode as a pong (distinct magic).
    CHECK(DecodeFabricPong(AsBytes(EncodeFabricPing(1u)), nonce) ==
          WireStatus::kBadMagic);
    // Nor as a request (too short for the v2 request header) or a cancel.
    FabricRequestView rv;
    CHECK(DecodeFabricRequest(AsBytes(EncodeFabricPing(1u)), rv) ==
          WireStatus::kTruncated);
    std::uint32_t cid = 0;
    CHECK(DecodeFabricCancel(AsBytes(EncodeFabricPing(1u)), cid) ==
          WireStatus::kBadMagic);
}
