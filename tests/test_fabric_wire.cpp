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

TEST_CASE("fabric_wire v2: request round-trip", "[fabric][wire][v2]") {
    const std::vector<std::int32_t> ids{7, 8, 9, 10};
    const auto enc = EncodeFabricRequest(42u, ids, 16u);
    CHECK(enc.size() == kFabricRequestHeaderBytes + 16u);
    FabricRequestView view;
    REQUIRE(DecodeFabricRequest(AsBytes(enc), view) == WireStatus::kOk);
    CHECK(view.request_id == 42u);
    CHECK(view.max_gen == 16u);
    REQUIRE(view.tokens.size() == 4u);
    CHECK(view.tokens[3] == 10);
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
