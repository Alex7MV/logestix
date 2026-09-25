// Binary wire codec tests: pure functions, no ZMQ, no CUDA, no Arrow.
#include <catch2/catch_test_macros.hpp>

#include "lsxnetwork/fabric_wire.h"

#include <cstdint>
#include <span>
#include <vector>

using namespace lsxnetwork;

namespace {

void PatchU16(std::vector<std::byte>& f, std::size_t off, std::uint16_t v) {
    f[off] = static_cast<std::byte>(v & 0xFFu);
    f[off + 1] = static_cast<std::byte>((v >> 8) & 0xFFu);
}
void PatchU32(std::vector<std::byte>& f, std::size_t off, std::uint32_t v) {
    for (int i = 0; i < 4; ++i)
        f[off + i] = static_cast<std::byte>((v >> (8 * i)) & 0xFFu);
}

}  // namespace

TEST_CASE("fabric_wire: request round-trip", "[fabric][wire]") {
    const std::vector<std::int32_t> ids{1, -2, 300000, 0};
    const std::vector<std::byte> frame = EncodeFabricRequest(ids, 16u);

    REQUIRE(frame.size() == kFabricRequestHeaderBytes + 4u * ids.size());
    FabricRequestView v;
    REQUIRE(DecodeFabricRequest(frame, v) == WireStatus::kOk);
    CHECK(v.max_gen == 16u);
    REQUIRE(v.tokens.size() == ids.size());
    CHECK(v.tokens[0] == 1);
    CHECK(v.tokens[1] == -2);
    CHECK(v.tokens[2] == 300000);
    CHECK(v.tokens[3] == 0);
}

TEST_CASE("fabric_wire: empty request round-trip", "[fabric][wire]") {
    const std::vector<std::int32_t> ids;
    const std::vector<std::byte> frame = EncodeFabricRequest(ids, 0u);
    FabricRequestView v;
    REQUIRE(DecodeFabricRequest(frame, v) == WireStatus::kOk);
    CHECK(v.max_gen == 0u);
    CHECK(v.tokens.empty());
}

TEST_CASE("fabric_wire: malformed request rejected", "[fabric][wire]") {
    const std::vector<std::int32_t> ids{5, 6};
    const std::vector<std::byte> good = EncodeFabricRequest(ids, 1u);

    SECTION("bad magic") {
        std::vector<std::byte> f = good;
        PatchU32(f, 0, 0xDEADBEEFu);
        FabricRequestView v;
        CHECK(DecodeFabricRequest(f, v) == WireStatus::kBadMagic);
    }
    SECTION("bad version") {
        std::vector<std::byte> f = good;
        PatchU16(f, 4, 99u);
        FabricRequestView v;
        CHECK(DecodeFabricRequest(f, v) == WireStatus::kBadVersion);
    }
    SECTION("truncated header") {
        std::vector<std::byte> f(good.begin(), good.begin() + 10);
        FabricRequestView v;
        CHECK(DecodeFabricRequest(f, v) == WireStatus::kTruncated);
    }
    SECTION("truncated body") {
        std::vector<std::byte> f(good.begin(), good.end() - 1);
        FabricRequestView v;
        CHECK(DecodeFabricRequest(f, v) == WireStatus::kTruncated);
    }
    SECTION("trailing garbage") {
        std::vector<std::byte> f = good;
        f.push_back(std::byte{0});
        FabricRequestView v;
        CHECK(DecodeFabricRequest(f, v) == WireStatus::kTrailing);
    }
    SECTION("num_tokens overflow") {
        std::vector<std::byte> f = good;
        PatchU32(f, 12, 0xFFFFFFFFu);
        FabricRequestView v;
        CHECK(DecodeFabricRequest(f, v) == WireStatus::kTruncated);
    }
}

TEST_CASE("fabric_wire: response round-trip carries timings and eos",
          "[fabric][wire]") {
    FabricResponse r;
    r.status = static_cast<std::int32_t>(FabricStatus::kOk);
    r.eos_terminated = true;
    r.ttft_ms = 12.5;
    r.total_ms = 34.25;
    r.tokens_per_sec = 78.75;
    r.tokens = {11, 22, 33};

    const std::vector<std::byte> frame = EncodeFabricResponse(r);
    REQUIRE(frame.size() == kFabricResponseHeaderBytes + 12u);
    FabricResponse d;
    REQUIRE(DecodeFabricResponse(frame, d) == WireStatus::kOk);
    CHECK(d.status == r.status);
    CHECK(d.eos_terminated);
    CHECK(d.ttft_ms == 12.5);
    CHECK(d.total_ms == 34.25);
    CHECK(d.tokens_per_sec == 78.75);
    REQUIRE(d.tokens.size() == 3);
    CHECK(d.tokens[2] == 33);
    CHECK(d.error.empty());
}

TEST_CASE("fabric_wire: error response carries message and no tokens",
          "[fabric][wire]") {
    FabricResponse r;
    r.status = static_cast<std::int32_t>(FabricStatus::kInferFailed);
    r.error = "boom";
    const std::vector<std::byte> frame = EncodeFabricResponse(r);
    FabricResponse d;
    REQUIRE(DecodeFabricResponse(frame, d) == WireStatus::kOk);
    CHECK(d.status == static_cast<std::int32_t>(FabricStatus::kInferFailed));
    CHECK(d.tokens.empty());
    CHECK(d.error == "boom");
}

TEST_CASE("fabric_wire: empty error message round-trips", "[fabric][wire]") {
    FabricResponse r;
    r.status = static_cast<std::int32_t>(FabricStatus::kBadRequest);
    const std::vector<std::byte> frame = EncodeFabricResponse(r);
    FabricResponse d;
    REQUIRE(DecodeFabricResponse(frame, d) == WireStatus::kOk);
    CHECK(d.status == static_cast<std::int32_t>(FabricStatus::kBadRequest));
    CHECK(d.error.empty());
}
