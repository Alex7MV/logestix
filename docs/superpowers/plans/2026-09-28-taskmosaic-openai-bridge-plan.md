# TaskMosaic OpenAI Bridge — Plan 1: Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Repo constraint:** AGENTS.md forbids Task subagents. Execute **inline** in the
> main session with superpowers:executing-plans (batch + checkpoints), not via
> dispatched subagents.

**Goal:** Stand up an OpenAI-compatible HTTP server in `taskmosaic` that bridges
host-side tokenization/framing to a streaming `lsxfabric` GPU backend over the v2
DEALER/ROUTER + PUSH/PULL protocol, with per-request cancellation.

**Architecture:** `taskmosaic` hosts a cpp-httplib server + a per-model backend
pool (one `FabricStreamClient` each, bounded admission queue). Each backend
streams v2 token-batch frames from a ROUTER socket while `model.Infer` runs;
a PULL cancel listener sets the model's `abort_flag`. `taskmosaic` detokenizes
incrementally and emits SSE for `stream:true`. Scope of this plan: wire v2, the
stream client, the ROUTER/cancel server, the incremental detokenizer, httplib +
the generic `HttpServer` wrapper, bridge config, the backend pool, the extended
OpenAI codec, `/v1/chat/completions` (+ `/v1/completions`, `/v1/models`,
`/health`, `/ready`), the taskmosaic server entry point and the end-to-end fake
backend test. Tool calling, the Anthropic adapter, all-family streaming and the
compatibility extras are later plans.

**Tech Stack:** C++20, CMake 3.25+, catch2 v3.16.0, libzmq (submodule),
simdjson, abseil, spdlog, cpp-httplib (new FetchContent dep), Arrow tokenizer,
CUDA backend behind `lsxcommon::IModel`.

## Global Constraints

- Target arch Blackwell CC 12.0/12.1 only; forbidden sm_90/sm_80. No backward compat.
- Build: `cmake --preset linux-debug && cmake --build --preset linux-debug`; `ctest --preset linux-debug`.
- Logging only via spdlog (`#include <spdlog/spdlog.h>`); never `std::cout`/`printf`.
- Host-only C++ prefers `absl::` (`flat_hash_map`, `StrCat`, `StrFormat` printf-style: `%s/%d`; `{}` prints literally).
- Headers reachable from `.cu`/`.cuh` MUST NOT include `<absl/container/*>`.
- Every new CTest target gets a label (`fast`) in the same commit and is added to
  the matching `set_tests_properties(... LABELS ...)` list in `tests/CMakeLists.txt`
  (guard `test_ctest_labels_guard`).
- New model-family files would need naming/suite/independence guards — this plan
  adds none; do not introduce family-prefixed stems.
- Do not touch `.cu` numerics in this plan (plan 2 does).
- Commit prefixes: `build:`/`feat:`/`fix:`/`refactor:`/`test:`/`docs:`.

## File Structure

**Create**
- `libs/lsxnetwork/include/lsxnetwork/fabric_stream_client.h`
- `libs/lsxnetwork/src/fabric_stream_client.cpp`
- `libs/lsxnetwork/include/lsxnetwork/http_server.h`
- `libs/lsxnetwork/src/http_server.cpp`
- `libs/lsxcommon/include/lsxcommon/incremental_detokenizer.h`
- `libs/lsxcommon/src/incremental_detokenizer.cpp`
- `apps/taskmosaic/src/config.h` / `config.cpp`
- `apps/taskmosaic/src/backend.h` / `backend.cpp`
- `apps/taskmosaic/src/model_context.h` / `model_context.cpp`
- `apps/taskmosaic/src/chat_bridge.h` / `chat_bridge.cpp`
- `apps/taskmosaic/src/http_routes.h` / `http_routes.cpp`
- `apps/taskmosaic/tests/test_taskmosaic_config.cpp`
- `apps/taskmosaic/tests/test_taskmosaic_backend.cpp`
- `apps/taskmosaic/tests/test_taskmosaic_bridge.cpp`
- `tests/test_incremental_detokenizer.cpp`
- `tests/test_http_server.cpp`

**Modify**
- `libs/lsxnetwork/include/lsxnetwork/fabric_wire.h` + `src/fabric_wire.cpp` (v2)
- `libs/lsxnetwork/CMakeLists.txt` (sources, httplib)
- `libs/lsxcommon/CMakeLists.txt` (detokenizer source)
- `apps/lsxfabric/src/fabric_server.h` + `fabric_server.cpp` (ROUTER + cancel)
- `apps/lsxfabric/src/cli.h` + `cli.cpp` + `main.cpp` (`--listen-cancel`)
- `apps/lsxfabric/tests/test_fabric_server.cpp`, `test_fabric_cli_edges.cpp`
- `apps/taskmosaic/src/main.cpp`, `cli.h`, `cli.cpp` (server mode)
- `apps/taskmosaic/CMakeLists.txt`, `apps/taskmosaic/tests/CMakeLists.txt`
- `apps/taskmosaic/tests/test_taskmosaic.cpp`, `test_taskmosaic_cli.cpp`
- `tests/test_fabric_wire.cpp`, `tests/test_fabric_client.cpp`
- `tests/CMakeLists.txt` (new targets + labels)
- `CMakeLists.txt` (cpp-httplib FetchContent)

**Delete**
- `libs/lsxnetwork/include/lsxnetwork/fabric_client.h`
- `libs/lsxnetwork/src/fabric_client.cpp`

---

### Task 1: Wire protocol v2 codec

**Files:**
- Modify: `libs/lsxnetwork/include/lsxnetwork/fabric_wire.h`
- Modify: `libs/lsxnetwork/src/fabric_wire.cpp`
- Test: `tests/test_fabric_wire.cpp` (rewrite)

**Interfaces:**
- Consumes: nothing new.
- Produces: `kFabricMagic`, `kFabricCancelMagic`, `kFabricVersion=2`,
  `kFabricRequestHeaderBytes=24`, `kFabricStreamHeaderBytes=16`,
  `kFabricDoneTailBytes=32`, `kFabricCancelBytes=12`, `WireStatus`,
  `FabricFrameType`, `FabricRequestView{request_id,max_gen,tokens}`,
  `FabricStreamFrame{type,request_id,tokens,eos_terminated,total_tokens,ttft_ms,total_ms,tokens_per_sec,error}`,
  `EncodeFabricRequest(request_id, tokens, max_gen)`,
  `DecodeFabricRequest(frame, out)`,
  `EncodeFabricTokenBatch(request_id, tokens)`,
  `EncodeFabricDone(request_id, eos, total_tokens, ttft, total, tps)`,
  `EncodeFabricError(request_id, message)`,
  `DecodeFabricStreamFrame(frame, out)`,
  `EncodeFabricCancel(request_id)`, `DecodeFabricCancel(frame, request_id)`.

- [ ] **Step 1: Replace `fabric_wire.h` with the v2 declarations**

```cpp
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace lsxnetwork {

/// v2 streaming wire protocol (little-endian, explicit byte-wise encoding):
///   request    = 24-byte header + 4 * num_tokens bytes  (client -> server)
///   stream     = one message per frame, each: 16-byte header + body
///                token_batch: 16 + 4 * count bytes
///                done:        16 + 32 bytes
///                error:       16 + len bytes (UTF-8)
///   cancel     = 12 bytes (client -> server, separate PUSH/PULL channel)
inline constexpr std::uint32_t kFabricMagic = 0x4246584Cu;        // "LXFB" LE
inline constexpr std::uint32_t kFabricCancelMagic = 0x4346584Cu;  // "LXFC" LE
inline constexpr std::uint16_t kFabricVersion = 2;

inline constexpr std::size_t kFabricRequestHeaderBytes = 24;
inline constexpr std::size_t kFabricStreamHeaderBytes = 16;
inline constexpr std::size_t kFabricDoneTailBytes = 32;  // flags4+total4+3*f64
inline constexpr std::size_t kFabricCancelBytes = 12;

enum class WireStatus {
    kOk,
    kBadMagic,
    kBadVersion,
    kTruncated,
    kTrailing,
    kBadType,
};

enum class FabricFrameType : std::uint16_t {
    kTokenBatch = 1,
    kDone = 2,
    kError = 3,
};

/// Non-owning decoded request. `tokens` aliases the caller's frame buffer.
struct FabricRequestView {
    std::uint32_t request_id = 0;
    std::uint32_t max_gen = 0;  // 0 => server default
    std::span<const std::int32_t> tokens;
};

/// Owned decoded stream frame. Exactly one of the payload members is populated
/// according to `type`.
struct FabricStreamFrame {
    FabricFrameType type = FabricFrameType::kTokenBatch;
    std::uint32_t request_id = 0;
    std::vector<std::int32_t> tokens;  // kTokenBatch
    bool eos_terminated = false;       // kDone
    std::uint32_t total_tokens = 0;    // kDone
    double ttft_ms = 0.0;              // kDone
    double total_ms = 0.0;             // kDone
    double tokens_per_sec = 0.0;       // kDone
    std::string error;                 // kError
};

std::vector<std::byte> EncodeFabricRequest(std::uint32_t request_id,
                                           std::span<const std::int32_t> tokens,
                                           std::uint32_t max_gen);
WireStatus DecodeFabricRequest(std::span<const std::byte> frame,
                               FabricRequestView& out);

std::vector<std::byte> EncodeFabricTokenBatch(
    std::uint32_t request_id, std::span<const std::int32_t> tokens);
std::vector<std::byte> EncodeFabricDone(std::uint32_t request_id,
                                        bool eos_terminated,
                                        std::uint32_t total_tokens,
                                        double ttft_ms, double total_ms,
                                        double tokens_per_sec);
std::vector<std::byte> EncodeFabricError(std::uint32_t request_id,
                                         std::string_view message);
WireStatus DecodeFabricStreamFrame(std::span<const std::byte> frame,
                                   FabricStreamFrame& out);

std::vector<std::byte> EncodeFabricCancel(std::uint32_t request_id);
WireStatus DecodeFabricCancel(std::span<const std::byte> frame,
                              std::uint32_t& request_id);

}  // namespace lsxnetwork
```

- [ ] **Step 2: Write the failing codec test**

Replace `tests/test_fabric_wire.cpp` with:

```cpp
#include <catch2/catch_test_macros.hpp>

#include <lsxnetwork/fabric_wire.h>

#include <cstdint>
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
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cmake --build --preset linux-debug --target test_fabric_wire && ctest --preset linux-debug -R test_fabric_wire --output-on-failure`
Expected: build FAILS (v2 functions undefined) — good.

- [ ] **Step 4: Implement `fabric_wire.cpp`**

```cpp
#include "lsxnetwork/fabric_wire.h"

#include <cstring>

namespace lsxnetwork {
namespace {

void PutU16(std::byte* p, std::uint16_t v) {
    p[0] = static_cast<std::byte>(v & 0xFFu);
    p[1] = static_cast<std::byte>((v >> 8) & 0xFFu);
}
void PutU32(std::byte* p, std::uint32_t v) {
    for (int i = 0; i < 4; ++i)
        p[i] = static_cast<std::byte>((v >> (8 * i)) & 0xFFu);
}
void PutI32(std::byte* p, std::int32_t v) { PutU32(p, static_cast<std::uint32_t>(v)); }
void PutF64(std::byte* p, double v) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    for (int i = 0; i < 8; ++i)
        p[i] = static_cast<std::byte>((bits >> (8 * i)) & 0xFFull);
}
std::uint16_t GetU16(const std::byte* p) {
    return static_cast<std::uint16_t>(std::to_integer<unsigned>(p[0]) |
                                      (std::to_integer<unsigned>(p[1]) << 8));
}
std::uint32_t GetU32(const std::byte* p) {
    std::uint32_t v = 0;
    for (int i = 0; i < 4; ++i)
        v |= static_cast<std::uint32_t>(std::to_integer<unsigned>(p[i])) << (8 * i);
    return v;
}
double GetF64(const std::byte* p) {
    std::uint64_t bits = 0;
    for (int i = 0; i < 8; ++i)
        bits |= static_cast<std::uint64_t>(std::to_integer<unsigned>(p[i])) << (8 * i);
    double v = 0.0;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

// Shared v2 stream header: magic, version, type, request_id.
void PutStreamHeader(std::byte* p, FabricFrameType type, std::uint32_t rid) {
    PutU32(p + 0, kFabricMagic);
    PutU16(p + 4, kFabricVersion);
    PutU16(p + 6, static_cast<std::uint16_t>(type));
    PutU32(p + 8, rid);
}
WireStatus CheckStreamHeader(const std::byte* p, std::size_t size,
                             FabricFrameType& type, std::uint32_t& rid) {
    if (size < kFabricStreamHeaderBytes) return WireStatus::kTruncated;
    if (GetU32(p) != kFabricMagic) return WireStatus::kBadMagic;
    if (GetU16(p + 4) != kFabricVersion) return WireStatus::kBadVersion;
    const std::uint16_t t = GetU16(p + 6);
    if (t < 1 || t > 3) return WireStatus::kBadType;
    type = static_cast<FabricFrameType>(t);
    rid = GetU32(p + 8);
    return WireStatus::kOk;
}

}  // namespace

std::vector<std::byte> EncodeFabricRequest(std::uint32_t request_id,
                                           std::span<const std::int32_t> tokens,
                                           std::uint32_t max_gen) {
    std::vector<std::byte> out(kFabricRequestHeaderBytes + tokens.size() * 4u);
    std::byte* p = out.data();
    PutU32(p + 0, kFabricMagic);
    PutU16(p + 4, kFabricVersion);
    PutU16(p + 6, 0);
    PutU32(p + 8, request_id);
    PutU32(p + 12, max_gen);
    PutU32(p + 16, static_cast<std::uint32_t>(tokens.size()));
    for (std::size_t i = 0; i < tokens.size(); ++i)
        PutI32(p + kFabricRequestHeaderBytes + i * 4u, tokens[i]);
    return out;
}

WireStatus DecodeFabricRequest(std::span<const std::byte> frame,
                               FabricRequestView& out) {
    if (frame.size() < kFabricRequestHeaderBytes) return WireStatus::kTruncated;
    const std::byte* p = frame.data();
    if (GetU32(p) != kFabricMagic) return WireStatus::kBadMagic;
    if (GetU16(p + 4) != kFabricVersion) return WireStatus::kBadVersion;
    const std::uint32_t n = GetU32(p + 16);
    const std::size_t avail = frame.size() - kFabricRequestHeaderBytes;
    if (static_cast<std::size_t>(n) > avail / 4u) return WireStatus::kTruncated;
    if (static_cast<std::size_t>(n) * 4u != avail) return WireStatus::kTrailing;
    out.request_id = GetU32(p + 8);
    out.max_gen = GetU32(p + 12);
    out.tokens = std::span<const std::int32_t>(
        reinterpret_cast<const std::int32_t*>(p + kFabricRequestHeaderBytes), n);
    return WireStatus::kOk;
}

std::vector<std::byte> EncodeFabricTokenBatch(
    std::uint32_t request_id, std::span<const std::int32_t> tokens) {
    std::vector<std::byte> out(kFabricStreamHeaderBytes + tokens.size() * 4u);
    std::byte* p = out.data();
    PutStreamHeader(p, FabricFrameType::kTokenBatch, request_id);
    PutU32(p + 12, static_cast<std::uint32_t>(tokens.size()));
    for (std::size_t i = 0; i < tokens.size(); ++i)
        PutI32(p + kFabricStreamHeaderBytes + i * 4u, tokens[i]);
    return out;
}

std::vector<std::byte> EncodeFabricDone(std::uint32_t request_id,
                                        bool eos_terminated,
                                        std::uint32_t total_tokens,
                                        double ttft_ms, double total_ms,
                                        double tokens_per_sec) {
    std::vector<std::byte> out(kFabricStreamHeaderBytes + kFabricDoneTailBytes);
    std::byte* p = out.data();
    PutStreamHeader(p, FabricFrameType::kDone, request_id);
    PutU32(p + 12, 0);  // reserved/count
    PutU32(p + 16, eos_terminated ? 1u : 0u);
    PutU32(p + 20, total_tokens);
    PutF64(p + 24, ttft_ms);
    PutF64(p + 32, total_ms);
    PutF64(p + 40, tokens_per_sec);
    return out;
}

std::vector<std::byte> EncodeFabricError(std::uint32_t request_id,
                                         std::string_view message) {
    std::vector<std::byte> out(kFabricStreamHeaderBytes + message.size());
    std::byte* p = out.data();
    PutStreamHeader(p, FabricFrameType::kError, request_id);
    PutU32(p + 12, static_cast<std::uint32_t>(message.size()));
    if (!message.empty())
        std::memcpy(p + kFabricStreamHeaderBytes, message.data(), message.size());
    return out;
}

WireStatus DecodeFabricStreamFrame(std::span<const std::byte> frame,
                                   FabricStreamFrame& out) {
    out = FabricStreamFrame{};
    FabricFrameType type{};
    std::uint32_t rid = 0;
    const WireStatus hs =
        CheckStreamHeader(frame.data(), frame.size(), type, rid);
    if (hs != WireStatus::kOk) return hs;
    out.type = type;
    out.request_id = rid;
    const std::byte* p = frame.data();
    const std::size_t avail = frame.size() - kFabricStreamHeaderBytes;

    if (type == FabricFrameType::kTokenBatch) {
        const std::uint32_t n = GetU32(p + 12);
        if (static_cast<std::size_t>(n) > avail / 4u) return WireStatus::kTruncated;
        if (static_cast<std::size_t>(n) * 4u != avail) return WireStatus::kTrailing;
        const auto* first =
            reinterpret_cast<const std::int32_t*>(p + kFabricStreamHeaderBytes);
        out.tokens.assign(first, first + n);
        return WireStatus::kOk;
    }
    if (type == FabricFrameType::kDone) {
        if (avail != kFabricDoneTailBytes) return WireStatus::kTrailing;
        out.eos_terminated = (GetU32(p + 16) & 1u) != 0u;
        out.total_tokens = GetU32(p + 20);
        out.ttft_ms = GetF64(p + 24);
        out.total_ms = GetF64(p + 32);
        out.tokens_per_sec = GetF64(p + 40);
        return WireStatus::kOk;
    }
    // kError
    const std::uint32_t len = GetU32(p + 12);
    if (static_cast<std::size_t>(len) > avail) return WireStatus::kTruncated;
    if (static_cast<std::size_t>(len) != avail) return WireStatus::kTrailing;
    out.error.assign(
        reinterpret_cast<const char*>(p + kFabricStreamHeaderBytes), len);
    return WireStatus::kOk;
}

std::vector<std::byte> EncodeFabricCancel(std::uint32_t request_id) {
    std::vector<std::byte> out(kFabricCancelBytes);
    std::byte* p = out.data();
    PutU32(p + 0, kFabricCancelMagic);
    PutU16(p + 4, kFabricVersion);
    PutU16(p + 6, 0);
    PutU32(p + 8, request_id);
    return out;
}

WireStatus DecodeFabricCancel(std::span<const std::byte> frame,
                              std::uint32_t& request_id) {
    if (frame.size() < kFabricCancelBytes) return WireStatus::kTruncated;
    if (frame.size() != kFabricCancelBytes) return WireStatus::kTrailing;
    const std::byte* p = frame.data();
    if (GetU32(p) != kFabricCancelMagic) return WireStatus::kBadMagic;
    if (GetU16(p + 4) != kFabricVersion) return WireStatus::kBadVersion;
    request_id = GetU32(p + 8);
    return WireStatus::kOk;
}

}  // namespace lsxnetwork
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cmake --build --preset linux-debug --target test_fabric_wire && ctest --preset linux-debug -R test_fabric_wire --output-on-failure`
Expected: PASS, all v2 cases.

- [ ] **Step 6: Commit**

```bash
git add libs/lsxnetwork/include/lsxnetwork/fabric_wire.h libs/lsxnetwork/src/fabric_wire.cpp tests/test_fabric_wire.cpp
git commit -m "feat: v2 streaming fabric wire codec"
```

---

### Task 2: `FabricStreamClient` (DEALER + cancel PUSH)

**Files:**
- Create: `libs/lsxnetwork/include/lsxnetwork/fabric_stream_client.h`
- Create: `libs/lsxnetwork/src/fabric_stream_client.cpp`
- Modify: `libs/lsxnetwork/CMakeLists.txt`
- Test: `tests/test_fabric_client.cpp` (rewrite)

**Interfaces:**
- Consumes: Task 1 codec, `ZmqContext`/`ZmqSocket`.
- Produces: `FabricStreamClient` with `Connect(endpoint, cancel_endpoint)`,
  `Connected()`, `Submit(request_id, input_ids, max_gen)`,
  `RecvStatus { kOk, kTimeout, kTransport, kMalformed }`,
  `Recv(FabricStreamFrame&, timeout_ms)`, `Cancel(request_id)`.

- [ ] **Step 1: Write the header**

```cpp
#pragma once

#include <cstdint>
#include <memory>
#include <span>

#include "lsxnetwork/fabric_wire.h"
#include "lsxnetwork/zmq_context.h"

namespace lsxnetwork {

/// Blocking DEALER client for the v2 streaming fabric server, with a companion
/// PUSH socket for per-request cancellation.
///
/// One request in flight (DEALER must not interleave requests). Not thread-safe;
/// use one instance per backend.
class FabricStreamClient {
public:
    enum class RecvStatus { kOk, kTimeout, kTransport, kMalformed };

    FabricStreamClient() = default;
    ~FabricStreamClient() = default;
    FabricStreamClient(const FabricStreamClient&) = delete;
    FabricStreamClient& operator=(const FabricStreamClient&) = delete;

    /// Creates the DEALER + cancel PUSH sockets and connects both. Returns 0 on
    /// success, -1 on socket/connect failure.
    int Connect(const char* endpoint, const char* cancel_endpoint);

    bool Connected() const noexcept { return dealer_ && dealer_->Valid(); }

    /// Sends the request frame. Returns false on transport error.
    bool Submit(std::uint32_t request_id,
                std::span<const std::int32_t> input_ids, std::uint32_t max_gen);

    /// Receives the next stream frame. kTimeout means no frame within
    /// @p timeout_ms; kMalformed means the frame failed to decode.
    RecvStatus Recv(FabricStreamFrame& out, int timeout_ms);

    /// Best-effort cancel of @p request_id on the PUSH channel.
    void Cancel(std::uint32_t request_id);

private:
    ZmqContext ctx_;
    std::unique_ptr<ZmqSocket> dealer_;
    std::unique_ptr<ZmqSocket> cancel_;
};

}  // namespace lsxnetwork
```

- [ ] **Step 2: Write the failing test**

Replace `tests/test_fabric_client.cpp`:

```cpp
// FabricStreamClient <-> a minimal inproc ROUTER/PULL fake server.
#include <catch2/catch_test_macros.hpp>

#include <lsxnetwork/fabric_stream_client.h>
#include <lsxnetwork/fabric_wire.h>
#include <lsxnetwork/zmq_context.h>

#include <zmq.h>

#include <atomic>
#include <chrono>
#include <string>
#include <thread>

using namespace lsxnetwork;

namespace {

// A tiny server: ROUTER that streams a token batch + done for each request, and
// a PULL that records received cancel ids.
struct FakeServer {
    ZmqContext ctx;
    ZmqSocket router{ctx, ZMQ_ROUTER};
    ZmqSocket pull{ctx, ZMQ_PULL};
    std::string endpoint;
    std::atomic<bool> stop{false};
    std::thread thread;
    std::atomic<std::uint32_t> last_cancel{0};

    bool Start() {
        if (!router.Valid() || !pull.Valid()) return false;
        if (router.Bind("inproc://lsx_sc_router") != 0) return false;
        if (pull.Bind("inproc://lsx_sc_cancel") != 0) return false;
        endpoint = "inproc://lsx_sc_router";
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
                    FabricRequestView v;
                    if (DecodeFabricRequest(f, v) == WireStatus::kOk)
                        rid = v.request_id;
                }
                zmq_msg_close(&payload);
                Send(ident, EncodeFabricTokenBatch(rid, {5, 6}));
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
    REQUIRE(client.Connect(srv.endpoint.c_str(), "inproc://lsx_sc_cancel") == 0);
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
    REQUIRE(client.Connect(srv.endpoint.c_str(), "inproc://lsx_sc_cancel") == 0);
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
    REQUIRE(client.Connect(srv.endpoint.c_str(), "inproc://lsx_sc_cancel") == 0);
    FabricStreamFrame f;
    CHECK(client.Recv(f, 50) == FabricStreamClient::RecvStatus::kTimeout);
    srv.Stop();
}
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cmake --build --preset linux-debug --target test_fabric_client`
Expected: build FAILS (header/source missing).

- [ ] **Step 4: Implement the client**

```cpp
#include "lsxnetwork/fabric_stream_client.h"

#include <zmq.h>

#include <cstring>

namespace lsxnetwork {

int FabricStreamClient::Connect(const char* endpoint,
                                const char* cancel_endpoint) {
    dealer_ = std::make_unique<ZmqSocket>(ctx_, ZMQ_DEALER);
    cancel_ = std::make_unique<ZmqSocket>(ctx_, ZMQ_PUSH);
    if (!dealer_->Valid() || !cancel_->Valid()) {
        dealer_.reset();
        cancel_.reset();
        return -1;
    }
    if (dealer_->Connect(endpoint) != 0) {
        dealer_.reset();
        cancel_.reset();
        return -1;
    }
    // Cancel is fire-and-forget: never block the caller if the server is gone.
    const int sndtimeo = 0;
    cancel_->SetSockOpt(ZMQ_SNDTIMEO, &sndtimeo, sizeof(sndtimeo));
    cancel_->SetSockOpt(ZMQ_LINGER, &sndtimeo, sizeof(sndtimeo));
    if (cancel_endpoint != nullptr && *cancel_endpoint != '\0') {
        // PUSH connect is asynchronous; a connect failure surfaces on send.
        cancel_->Connect(cancel_endpoint);
    }
    return 0;
}

bool FabricStreamClient::Submit(std::uint32_t request_id,
                                std::span<const std::int32_t> input_ids,
                                std::uint32_t max_gen) {
    if (!Connected()) return false;
    const std::vector<std::byte> frame =
        EncodeFabricRequest(request_id, input_ids, max_gen);
    return dealer_->Send(frame.data(), frame.size(), 0) ==
           static_cast<int>(frame.size());
}

FabricStreamClient::RecvStatus FabricStreamClient::Recv(
    FabricStreamFrame& out, int timeout_ms) {
    if (!Connected()) return RecvStatus::kTransport;
    if (dealer_->SetSockOpt(ZMQ_RCVTIMEO, &timeout_ms, sizeof(timeout_ms)) != 0)
        return RecvStatus::kTransport;

    zmq_msg_t msg;
    zmq_msg_init(&msg);
    const int n = zmq_msg_recv(&msg, dealer_->Get(), 0);
    if (n < 0) {
        const bool timed_out = (zmq_errno() == EAGAIN);
        zmq_msg_close(&msg);
        return timed_out ? RecvStatus::kTimeout : RecvStatus::kTransport;
    }
    const std::span<const std::byte> frame(
        static_cast<const std::byte*>(zmq_msg_data(&msg)),
        static_cast<std::size_t>(n));
    const WireStatus ws = DecodeFabricStreamFrame(frame, out);
    zmq_msg_close(&msg);
    return ws == WireStatus::kOk ? RecvStatus::kOk : RecvStatus::kMalformed;
}

void FabricStreamClient::Cancel(std::uint32_t request_id) {
    if (!cancel_ || !cancel_->Valid()) return;
    const std::vector<std::byte> frame = EncodeFabricCancel(request_id);
    cancel_->Send(frame.data(), frame.size(), 0);
}

}  // namespace lsxnetwork
```

- [ ] **Step 5: Wire the build (add source, drop the old client)**

In `libs/lsxnetwork/CMakeLists.txt`: replace `src/fabric_client.cpp` with
`src/fabric_stream_client.cpp` in `add_library(lsxnetwork STATIC ...)`, and add
the new header (headers are not listed). Delete the old files:

```bash
git rm libs/lsxnetwork/include/lsxnetwork/fabric_client.h libs/lsxnetwork/src/fabric_client.cpp
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `cmake --preset linux-debug && cmake --build --preset linux-debug --target test_fabric_client && ctest --preset linux-debug -R test_fabric_client --output-on-failure`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add -A libs/lsxnetwork tests/test_fabric_client.cpp
git commit -m "feat: FabricStreamClient over DEALER with cancel PUSH"
```

---

### Task 3: `FabricServer` ROUTER + cancel PULL

**Files:**
- Modify: `apps/lsxfabric/src/fabric_server.h`
- Modify: `apps/lsxfabric/src/fabric_server.cpp`
- Test: `apps/lsxfabric/tests/test_fabric_server.cpp` (rewrite)

**Interfaces:**
- Consumes: Task 1 codec, `lsxcommon::IModel`, `ModelRequest`.
- Produces: `FabricServer(model, default_max_gen)` with
  `Run(endpoint, cancel_endpoint, stop, ready, resolved_endpoint,
  resolved_cancel_endpoint)`; streams token batches from `req.token_callback`
  via ROUTER, terminates with done/error, honors cancel on PULL.

- [ ] **Step 1: Rewrite the header**

```cpp
#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>

namespace lsxcommon {
class IModel;
}

namespace lsxfabric {

/// Streaming ZeroMQ server for the v2 fabric protocol.
///
/// Main thread owns a ROUTER socket: it receives requests and streams token
/// batches (from the model's per-token callback) as separate messages. A second
/// thread owns a PULL socket and sets the in-flight request's abort flag when a
/// matching cancel arrives.
class FabricServer {
public:
    FabricServer(lsxcommon::IModel& model, std::uint32_t default_max_gen);

    /// Binds @p endpoint (ROUTER) and @p cancel_endpoint (PULL) and serves until
    /// @p stop becomes true.
    /// @param ready set true after both binds succeed (test synchronization).
    /// @param resolved_endpoint / resolved_cancel_endpoint receive the concrete
    ///        bound endpoints (resolve a "tcp://127.0.0.1:*" wildcard).
    /// @return 0 on clean stop, non-zero when a socket could not bind.
    int Run(const std::string& endpoint, const std::string& cancel_endpoint,
            std::atomic<bool>& stop, std::atomic<bool>* ready = nullptr,
            std::string* resolved_endpoint = nullptr,
            std::string* resolved_cancel_endpoint = nullptr);

private:
    lsxcommon::IModel& model_;
    std::uint32_t default_max_gen_;

    std::mutex cancel_mtx_;
    std::atomic<bool>* current_abort_ = nullptr;  // guarded by cancel_mtx_
    std::uint32_t current_id_ = 0;                // guarded by cancel_mtx_
};

}  // namespace lsxfabric
```

- [ ] **Step 2: Write the failing server test**

Replace `apps/lsxfabric/tests/test_fabric_server.cpp`:

```cpp
// FabricServer (v2 ROUTER/cancel) <-> FabricStreamClient with a fake IModel.
#include <catch2/catch_test_macros.hpp>

#include "fabric_server.h"

#include "lsxcommon/model.h"
#include "lsxnetwork/fabric_stream_client.h"
#include "lsxnetwork/fabric_wire.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr const char* kRouter = "inproc://lsx_srv_router";
constexpr const char* kCancel = "inproc://lsx_srv_cancel";

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
    lsxfabric::FabricServer server{model, 4u};
    std::thread thread;

    void Start() {
        thread = std::thread([this] {
            server.Run(kRouter, kCancel, stop, &ready);
        });
        while (!ready.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
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
    lsxnetwork::FabricStreamClient client;
    REQUIRE(client.Connect(kRouter, kCancel) == 0);
    REQUIRE(client.Submit(7u, std::vector<std::int32_t>{1, 2, 3}, 0u));

    std::vector<std::int32_t> all;
    lsxnetwork::FabricStreamFrame f;
    bool done = false;
    for (int i = 0; i < 100 && !done; ++i) {
        REQUIRE(client.Recv(f, 2000) == lsxnetwork::FabricStreamClient::RecvStatus::kOk);
        CHECK(f.request_id == 7u);
        if (f.type == lsxnetwork::FabricFrameType::kTokenBatch)
            all.insert(all.end(), f.tokens.begin(), f.tokens.end());
        else if (f.type == lsxnetwork::FabricFrameType::kDone)
            done = true;
    }
    CHECK(done);
    REQUIRE(all.size() == 4u);          // server default max_gen
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
    REQUIRE(client.Connect(kRouter, kCancel) == 0);
    REQUIRE(client.Submit(1u, std::vector<std::int32_t>{1}, 7u));
    lsxnetwork::FabricStreamFrame f;
    std::size_t count = 0;
    for (;;) {
        REQUIRE(client.Recv(f, 2000) == lsxnetwork::FabricStreamClient::RecvStatus::kOk);
        if (f.type == lsxnetwork::FabricFrameType::kDone) break;
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
    REQUIRE(client.Connect(kRouter, kCancel) == 0);
    REQUIRE(client.Submit(1u, std::vector<std::int32_t>{1}, 2u));
    lsxnetwork::FabricStreamFrame f;
    REQUIRE(client.Recv(f, 2000) == lsxnetwork::FabricStreamClient::RecvStatus::kOk);
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
    REQUIRE(client.Connect(kRouter, kCancel) == 0);
    REQUIRE(client.Submit(55u, std::vector<std::int32_t>{1}, 1000u));

    lsxnetwork::FabricStreamFrame f;
    // Consume the first batch, then cancel.
    REQUIRE(client.Recv(f, 2000) == lsxnetwork::FabricStreamClient::RecvStatus::kOk);
    REQUIRE(f.type == lsxnetwork::FabricFrameType::kTokenBatch);
    client.Cancel(55u);

    std::size_t count = f.tokens.size();
    for (;;) {
        REQUIRE(client.Recv(f, 5000) == lsxnetwork::FabricStreamClient::RecvStatus::kOk);
        if (f.type == lsxnetwork::FabricFrameType::kDone) break;
        if (f.type == lsxnetwork::FabricFrameType::kError) break;
        count += f.tokens.size();
    }
    CHECK(count < 1000u);  // aborted well before max_gen
    fx.Stop();
}

TEST_CASE("fabric_server v2: malformed request yields an error frame",
          "[fabric][server][v2]") {
    Fixture fx;
    fx.Start();
    lsxnetwork::ZmqContext ctx;
    lsxnetwork::ZmqSocket dealer(ctx, ZMQ_DEALER);
    REQUIRE(dealer.Valid());
    REQUIRE(dealer.Connect(kRouter) == 0);
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
```

Add `#include <zmq.h>` to the test includes.

- [ ] **Step 3: Run the test to verify it fails**

Run: `cmake --build --preset linux-debug --target test_fabric_server`
Expected: build FAILS (old `FabricServer::Run` signature).

- [ ] **Step 4: Implement `fabric_server.cpp`**

```cpp
#include "fabric_server.h"

#include "lsxcommon/log.h"
#include "lsxcommon/model.h"
#include "lsxnetwork/fabric_wire.h"
#include "lsxnetwork/zmq_context.h"

#include <zmq.h>

#include <cstring>
#include <exception>
#include <shared_mutex>
#include <span>
#include <utility>
#include <vector>

namespace lsxfabric {
namespace {

// ROUTER recv: first frame is the peer identity, second is the payload.
bool RecvRouted(void* sock, std::vector<std::byte>& ident,
                std::vector<std::byte>& payload) {
    zmq_msg_t id;
    zmq_msg_init(&id);
    if (zmq_msg_recv(&id, sock, 0) < 0) {
        zmq_msg_close(&id);
        return false;
    }
    ident.assign(static_cast<const std::byte*>(zmq_msg_data(&id)),
                 static_cast<const std::byte*>(zmq_msg_data(&id)) +
                     zmq_msg_size(&id));
    bool more = zmq_msg_more(&id) != 0;
    zmq_msg_close(&id);
    if (!more) return false;

    zmq_msg_t body;
    zmq_msg_init(&body);
    if (zmq_msg_recv(&body, sock, 0) < 0) {
        zmq_msg_close(&body);
        return false;
    }
    payload.assign(static_cast<const std::byte*>(zmq_msg_data(&body)),
                   static_cast<const std::byte*>(zmq_msg_data(&body)) +
                       zmq_msg_size(&body));
    zmq_msg_close(&body);
    return true;
}

void SendRouted(void* sock, const std::vector<std::byte>& ident,
                const std::vector<std::byte>& body) {
    zmq_send(sock, ident.data(), ident.size(), ZMQ_SNDMORE);
    zmq_send(sock, body.data(), body.size(), 0);
}

}  // namespace

FabricServer::FabricServer(lsxcommon::IModel& model,
                           std::uint32_t default_max_gen)
    : model_(model), default_max_gen_(default_max_gen) {}

int FabricServer::Run(const std::string& endpoint,
                      const std::string& cancel_endpoint,
                      std::atomic<bool>& stop, std::atomic<bool>* ready,
                      std::string* resolved_endpoint,
                      std::string* resolved_cancel_endpoint) {
    lsxnetwork::ZmqContext ctx;
    lsxnetwork::ZmqSocket router(ctx, ZMQ_ROUTER);
    lsxnetwork::ZmqSocket pull(ctx, ZMQ_PULL);
    if (!router.Valid() || !pull.Valid() || router.Bind(endpoint.c_str()) != 0 ||
        pull.Bind(cancel_endpoint.c_str()) != 0) {
        lsxcommon::log::error("[fabric] cannot bind " + endpoint + " / " +
                              cancel_endpoint);
        return 1;
    }
    auto resolve = [](lsxnetwork::ZmqSocket& s, std::string* out) {
        if (out == nullptr) return;
        char ep[256];
        std::size_t len = sizeof(ep);
        if (s.GetSockOpt(ZMQ_LAST_ENDPOINT, ep, &len) == 0) *out = std::string(ep);
    };
    resolve(router, resolved_endpoint);
    resolve(pull, resolved_cancel_endpoint);

    const int rcvtimeo = 200;
    router.SetSockOpt(ZMQ_RCVTIMEO, &rcvtimeo, sizeof(rcvtimeo));
    if (ready != nullptr) ready->store(true);

    // Cancel listener: owns the PULL socket; sets the current request's abort.
    std::atomic<bool> listener_stop{false};
    std::thread listener([&] {
        lsxnetwork::ZmqSocket& p = pull;
        p.SetSockOpt(ZMQ_RCVTIMEO, &rcvtimeo, sizeof(rcvtimeo));
        while (!listener_stop.load(std::memory_order_acquire) &&
               !stop.load(std::memory_order_acquire)) {
            zmq_msg_t msg;
            zmq_msg_init(&msg);
            const int n = zmq_msg_recv(&msg, p.Get(), 0);
            if (n < 0) {
                zmq_msg_close(&msg);
                continue;  // EAGAIN
            }
            const std::span<const std::byte> f(
                static_cast<const std::byte*>(zmq_msg_data(&msg)),
                static_cast<std::size_t>(n));
            std::uint32_t rid = 0;
            const bool ok =
                lsxnetwork::DecodeFabricCancel(f, rid) ==
                lsxnetwork::WireStatus::kOk;
            zmq_msg_close(&msg);
            if (!ok) continue;
            std::lock_guard<std::mutex> lock(cancel_mtx_);
            if (current_abort_ != nullptr && rid == current_id_)
                current_abort_->store(true, std::memory_order_release);
        }
    });

    lsxcommon::log::info("[fabric] serving on " + endpoint + " (cancel " +
                         cancel_endpoint + ")");
    while (!stop.load(std::memory_order_acquire)) {
        std::vector<std::byte> ident;
        std::vector<std::byte> payload;
        if (!RecvRouted(router.Get(), ident, payload)) {
            if (zmq_errno() == EAGAIN) continue;  // recv timeout: re-check stop
            break;
        }

        lsxnetwork::FabricRequestView req;
        if (lsxnetwork::DecodeFabricRequest(payload, req) !=
            lsxnetwork::WireStatus::kOk) {
            SendRouted(router.Get(), ident,
                       lsxnetwork::EncodeFabricError(0u, "malformed request"));
            continue;
        }

        std::atomic<bool> abort{false};
        {
            std::lock_guard<std::mutex> lock(cancel_mtx_);
            current_abort_ = &abort;
            current_id_ = req.request_id;
        }

        lsxcommon::ModelRequest mreq;
        mreq.input_ids.assign(req.tokens.begin(), req.tokens.end());
        mreq.max_gen_tokens = req.max_gen != 0u
                                  ? static_cast<int>(req.max_gen)
                                  : static_cast<int>(default_max_gen_);
        mreq.abort_flag = &abort;
        mreq.token_callback = [&](const lsxcommon::TokenChunk& chunk) {
            const std::int32_t tok = static_cast<std::int32_t>(chunk.token_id);
            const std::vector<std::byte> frame =
                lsxnetwork::EncodeFabricTokenBatch(
                    req.request_id, std::span<const std::int32_t>(&tok, 1));
            SendRouted(router.Get(), ident, frame);
        };

        lsxcommon::gpu::InferenceResult res;
        try {
            std::shared_lock<std::shared_mutex> lock(model_.ResidencyMutex());
            res = model_.Infer(mreq);
        } catch (const std::exception& e) {
            SendRouted(router.Get(), ident,
                       lsxnetwork::EncodeFabricError(req.request_id, e.what()));
            std::lock_guard<std::mutex> lock(cancel_mtx_);
            current_abort_ = nullptr;
            continue;
        }

        if (!res.ok) {
            SendRouted(router.Get(), ident,
                       lsxnetwork::EncodeFabricError(req.request_id, res.error_msg));
        } else {
            SendRouted(router.Get(), ident,
                       lsxnetwork::EncodeFabricDone(
                           req.request_id, res.eos_terminated,
                           static_cast<std::uint32_t>(res.gen_tokens.size()),
                           res.ttft_ms, res.total_time_ms, res.tokens_per_sec));
        }
        std::lock_guard<std::mutex> lock(cancel_mtx_);
        current_abort_ = nullptr;
    }

    listener_stop.store(true, std::memory_order_release);
    if (listener.joinable()) listener.join();
    return 0;
}

}  // namespace lsxfabric
```

Add to `fabric_server.h` includes: `<thread>`.

- [ ] **Step 5: Run the test to verify it passes**

Run: `cmake --build --preset linux-debug --target test_fabric_server && ctest --preset linux-debug -R test_fabric_server --output-on-failure`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add apps/lsxfabric/src/fabric_server.h apps/lsxfabric/src/fabric_server.cpp apps/lsxfabric/tests/test_fabric_server.cpp
git commit -m "feat: streaming ROUTER FabricServer with per-request cancel"
```

---

### Task 4: lsxfabric CLI `--listen-cancel` + main wiring

**Files:**
- Modify: `apps/lsxfabric/src/cli.h`, `cli.cpp`, `main.cpp`
- Test: `apps/lsxfabric/tests/test_fabric_cli_edges.cpp`

**Interfaces:**
- Consumes: Task 3 `FabricServer::Run`.
- Produces: `FabricCliArgs::listen_cancel` (default `tcp://*:5556`).

- [ ] **Step 1: Add a failing CLI test**

Append to `apps/lsxfabric/tests/test_fabric_cli_edges.cpp`:

```cpp
TEST_CASE("fabric cli: --listen-cancel parsed", "[fabric][cli]") {
    const char* argv[] = {"lsxfabric", "--serve", "--listen",
                          "tcp://*:6001", "--listen-cancel", "tcp://*:6002"};
    lsxfabric::FabricCliArgs a =
        lsxfabric::ParseFabricCliArgs(8, const_cast<char**>(argv));
    CHECK(a.serve);
    CHECK(a.listen == "tcp://*:6001");
    CHECK(a.listen_cancel == "tcp://*:6002");
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cmake --build --preset linux-debug --target test_fabric_cli`
Expected: FAIL (no `listen_cancel` member).

- [ ] **Step 3: Update `cli.h` / `cli.cpp`**

In `cli.h` add after `listen`:

```cpp
    std::string listen_cancel = "tcp://*:5556";
```

In `cli.cpp` add `--listen-cancel` to `IsKnownFlag` and the parse loop:

```cpp
        } else if (std::strcmp(argv[i], "--listen-cancel") == 0) {
            if (has_value) a.listen_cancel = argv[++i];
```

- [ ] **Step 4: Update `main.cpp` serve path**

Replace the `FabricServer server(...); rc = server.Run(cli.listen, stop);`
lines with:

```cpp
                lsxfabric::FabricServer server(
                    *model, static_cast<std::uint32_t>(cli.max_gen));
                rc = server.Run(cli.listen, cli.listen_cancel, stop);
```

- [ ] **Step 5: Run tests + build lsxfabric**

Run: `cmake --build --preset linux-debug --target test_fabric_cli lsxfabric && ctest --preset linux-debug -R test_fabric_cli --output-on-failure`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add apps/lsxfabric/src/cli.h apps/lsxfabric/src/cli.cpp apps/lsxfabric/src/main.cpp apps/lsxfabric/tests/test_fabric_cli_edges.cpp
git commit -m "feat: lsxfabric --listen-cancel for the streaming server"
```

---

### Task 5: Incremental detokenizer

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/incremental_detokenizer.h`
- Create: `libs/lsxcommon/src/incremental_detokenizer.cpp`
- Modify: `libs/lsxcommon/CMakeLists.txt`
- Test: `tests/test_incremental_detokenizer.cpp`

**Interfaces:**
- Consumes: `ArrowTokenizerDb`, `arrow_tokenizer_decode`.
- Produces: `IncrementalDetokenizer(DecodeFn)`, `static DecodeFn ArrowDecoder(const ArrowTokenizerDb&)`,
  `std::string Push(std::int32_t id)`, `std::string Flush()`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_incremental_detokenizer.cpp`:

```cpp
#include <catch2/catch_test_macros.hpp>

#include <lsxcommon/incremental_detokenizer.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

using lsxcommon::IncrementalDetokenizer;

namespace {
// A fake decoder mapping token ids to raw byte segments (may split a codepoint).
IncrementalDetokenizer MakeFake(const std::map<std::int32_t, std::string>& t) {
    return IncrementalDetokenizer(
        [t](const std::vector<std::int32_t>& ids) {
            std::string s;
            for (const auto id : ids) {
                auto it = t.find(id);
                if (it != t.end()) s += it->second;
            }
            return s;
        });
}
}  // namespace

TEST_CASE("detokenizer: ascii deltas stream immediately",
          "[detokenizer][fast]") {
    auto d = MakeFake({{1, "Hel"}, {2, "lo"}});
    CHECK(d.Push(1) == "Hel");
    CHECK(d.Push(2) == "lo");
    CHECK(d.Flush().empty());
}

TEST_CASE("detokenizer: holds back an incomplete UTF-8 sequence",
          "[detokenizer][fast]") {
    // "é" = 0xC3 0xA9 split across two tokens.
    auto d = MakeFake({{1, std::string("\xC3", 1)},
                       {2, std::string("\xA9", 1)},
                       {3, "!"}});
    CHECK(d.Push(1).empty());         // incomplete, withheld
    CHECK(d.Push(2) == "\xC3\xA9");   // now complete
    CHECK(d.Push(3) == "!");
}

TEST_CASE("detokenizer: flush emits any held bytes",
          "[detokenizer][fast]") {
    auto d = MakeFake({{1, std::string("\xC3", 1)}});
    CHECK(d.Push(1).empty());
    CHECK(d.Flush() == std::string("\xC3", 1));
}

TEST_CASE("detokenizer: multi-byte codepoint intact",
          "[detokenizer][fast]") {
    auto d = MakeFake({{1, "\xE2\x82"}, {2, "\xAC"}});  // €
    CHECK(d.Push(1).empty());
    CHECK(d.Push(2) == "\xE2\x82\xAC");
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cmake --build --preset linux-debug --target test_incremental_detokenizer`
Expected: build FAILS (target not registered yet; register in Step 5 first if needed).

- [ ] **Step 3: Write the header**

```cpp
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "lsxcommon/tokenizer.h"

namespace lsxcommon {

/// Converts a token-id stream into UTF-8 text deltas without ever splitting a
/// multibyte codepoint across emitted chunks.
///
/// The decoder is injected (default `ArrowDecoder`) so the boundary logic is
/// unit-testable without a real vocabulary. Each Push re-decodes the full id
/// list (matching `runtime::DecodeGeneratedText`) and emits only the newly
/// completed, UTF-8-valid suffix; incomplete trailing bytes are withheld until
/// later tokens complete them (or Flush releases them).
class IncrementalDetokenizer {
public:
    using DecodeFn = std::function<std::string(const std::vector<std::int32_t>&)>;

    explicit IncrementalDetokenizer(DecodeFn decode);

    /// Decoder backed by the real Arrow tokenizer (specials skipped, raw bytes
    /// appended) with capacity retry — same semantics as DecodeGeneratedText.
    static DecodeFn ArrowDecoder(const ArrowTokenizerDb& db);

    /// Appends @p id and returns the new UTF-8-safe text suffix (possibly empty).
    std::string Push(std::int32_t id);

    /// Releases any withheld bytes at end of stream.
    std::string Flush();

private:
    DecodeFn decode_;
    std::vector<std::int32_t> ids_;
    std::string emitted_;
};

}  // namespace lsxcommon
```

- [ ] **Step 4: Implement it**

```cpp
#include "lsxcommon/incremental_detokenizer.h"

#include <cstddef>

namespace lsxcommon {
namespace {

// Length of the longest prefix of `s` that is a sequence of complete UTF-8
// codepoints. Returns s.size() when the whole string is valid.
std::size_t ValidUtf8Prefix(const std::string& s) {
    std::size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        std::size_t need = 0;
        if (c < 0x80u) {
            need = 1;
        } else if ((c & 0xE0u) == 0xC0u) {
            need = 2;
        } else if ((c & 0xF0u) == 0xE0u) {
            need = 3;
        } else if ((c & 0xF8u) == 0xF0u) {
            need = 4;
        } else {
            return i;  // invalid lead byte: stop here
        }
        if (i + need > s.size()) return i;  // incomplete: withhold
        for (std::size_t k = 1; k < need; ++k) {
            if ((static_cast<unsigned char>(s[i + k]) & 0xC0u) != 0x80u)
                return i;  // malformed continuation
        }
        i += need;
    }
    return i;
}

}  // namespace

IncrementalDetokenizer::IncrementalDetokenizer(DecodeFn decode)
    : decode_(std::move(decode)) {}

IncrementalDetokenizer::DecodeFn IncrementalDetokenizer::ArrowDecoder(
    const ArrowTokenizerDb& db) {
    return [&db](const std::vector<std::int32_t>& ids) -> std::string {
        if (ids.empty()) return {};
        std::size_t cap = 4096;
        std::string out(cap, '\0');
        for (int attempt = 0; attempt < 8; ++attempt) {
            const int rc =
                arrow_tokenizer_decode(db, ids.data(), ids.size(), out.data(), &cap);
            if (rc == 0) {
                out.resize(cap);
                return out;
            }
            if (rc == -2 && cap > out.size() && cap <= (1u << 20)) {
                out.resize(cap);
                continue;
            }
            break;
        }
        return {};
    };
}

std::string IncrementalDetokenizer::Push(std::int32_t id) {
    ids_.push_back(id);
    const std::string full = decode_ ? decode_(ids_) : std::string{};
    const std::size_t valid_len = ValidUtf8Prefix(full);
    if (valid_len <= emitted_.size()) return {};
    // Only emit when the previously-emitted prefix is still a prefix (decoder
    // output is monotonic for append-only id lists).
    if (full.compare(0, emitted_.size(), emitted_) != 0) {
        emitted_ = full.substr(0, valid_len);
        return emitted_;
    }
    std::string delta = full.substr(emitted_.size(), valid_len - emitted_.size());
    emitted_ = full.substr(0, valid_len);
    return delta;
}

std::string IncrementalDetokenizer::Flush() {
    const std::string full = decode_ ? decode_(ids_) : std::string{};
    if (full.size() <= emitted_.size()) return {};
    std::string delta = full.substr(emitted_.size());
    emitted_ = full;
    return delta;
}

}  // namespace lsxcommon
```

- [ ] **Step 5: Register the target and source**

Add `src/incremental_detokenizer.cpp` to the `lsxcommon` source list in
`libs/lsxcommon/CMakeLists.txt`.

In `tests/CMakeLists.txt`, after the `test_fabric_wire` block add:

```cmake
add_executable(test_incremental_detokenizer test_incremental_detokenizer.cpp)
target_link_libraries(test_incremental_detokenizer PRIVATE lsxcommon Catch2::Catch2WithMain)
add_test(NAME test_incremental_detokenizer COMMAND test_incremental_detokenizer)
```

Append `test_incremental_detokenizer` to the big `LABELS fast` list (line ~4749).

- [ ] **Step 6: Run the test to verify it passes**

Run: `cmake --preset linux-debug && cmake --build --preset linux-debug --target test_incremental_detokenizer && ctest --preset linux-debug -R test_incremental_detokenizer --output-on-failure`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/incremental_detokenizer.h libs/lsxcommon/src/incremental_detokenizer.cpp libs/lsxcommon/CMakeLists.txt tests/test_incremental_detokenizer.cpp tests/CMakeLists.txt
git commit -m "feat: incremental UTF-8-safe detokenizer"
```

---

### Task 6: cpp-httplib dependency + `HttpServer` wrapper

**Files:**
- Modify: `CMakeLists.txt` (FetchContent)
- Create: `libs/lsxnetwork/include/lsxnetwork/http_server.h`
- Create: `libs/lsxnetwork/src/http_server.cpp`
- Modify: `libs/lsxnetwork/CMakeLists.txt`
- Test: `tests/test_http_server.cpp`

**Interfaces:**
- Consumes: `HttpRouter` (static SPA), httplib (private).
- Produces: `HttpRequest{method,path,body}`, `HttpResponse{status,content_type,body,headers}`,
  `StreamSource{Next(out),Cancel()}`, `HttpServer` with `Get`, `Post`,
  `PostStream(path, StreamFactory)`, `EnableStaticSpa()`,
  `Listen(host,port)`, `BoundPort()`, `Stop()`.

- [ ] **Step 1: Add the dependency**

In `CMakeLists.txt`, after the Catch2 block (line ~286) add:

```cmake
# ── DEPENDENCIES: cpp-httplib via FetchContent (host-only HTTP server) ────────
# Header-only. OpenSSL/Zlib integrations are OFF so no extra system deps leak in.
set(HTTPLIB_REQUIRE_OPENSSL OFF CACHE INTERNAL "")
set(HTTPLIB_USE_OPENSSL_IF_AVAILABLE OFF CACHE INTERNAL "")
set(HTTPLIB_USE_ZLIB_IF_AVAILABLE OFF CACHE INTERNAL "")
set(HTTPLIB_USE_BROTLI_IF_AVAILABLE OFF CACHE INTERNAL "")
FetchContent_Declare(
    httplib
    GIT_REPOSITORY "https://github.com/yhirose/cpp-httplib.git"
    GIT_TAG        "v0.18.3"
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(httplib)
```

- [ ] **Step 2: Write the wrapper header**

```cpp
#pragma once

#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <string_view>

namespace lsxnetwork {

struct HttpRequest {
    std::string method;
    std::string path;
    std::string body;
};

struct HttpResponse {
    int status = 200;
    std::string content_type = "application/json; charset=utf-8";
    std::string body;
    std::map<std::string, std::string> headers;
};

/// Pull-based SSE body producer. `Next` blocks until the next chunk is ready
/// (or the request terminates) and returns false when the stream is finished.
/// `Cancel` is invoked by the transport when the client disconnects and must
/// unblock `Next`.
class StreamSource {
public:
    virtual ~StreamSource() = default;
    virtual bool Next(std::string& out) = 0;
    virtual void Cancel() = 0;
};

using StreamFactory =
    std::function<std::unique_ptr<StreamSource>(const HttpRequest&, HttpResponse&)>;

/// Minimal HTTP/1.1 server facade. cpp-httplib types never appear in this
/// header; taskmosaic registers callbacks and never includes httplib directly.
class HttpServer {
public:
    using Handler = std::function<void(const HttpRequest&, HttpResponse&)>;

    HttpServer();
    ~HttpServer();
    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    void Get(const std::string& path, Handler handler);
    void Post(const std::string& path, Handler handler);
    /// Registers a chunked `text/event-stream` POST route.
    void PostStream(const std::string& path, StreamFactory factory);
    /// Installs the embedded SPA as a GET/HEAD fallback.
    void EnableStaticSpa();

    /// Binds and serves. A value <= 0 picks an ephemeral port. Blocks until
    /// Stop(); returns false if the bind failed.
    bool Listen(const std::string& host, int port);
    /// Concrete bound port (valid after Listen has been entered).
    int BoundPort() const;
    void Stop();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace lsxnetwork
```

- [ ] **Step 3: Write the failing test**

Create `tests/test_http_server.cpp`:

```cpp
#include <catch2/catch_test_macros.hpp>

#include <lsxnetwork/http_server.h>

#include <atomic>
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
    server.PostStream("/sse", [](const HttpRequest&,
                                 HttpResponse& head) -> std::unique_ptr<StreamSource> {
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
    while (server.BoundPort() == 0) std::this_thread::sleep_for(std::chrono::milliseconds(1));

    httplib::Client cli("127.0.0.1", server.BoundPort());
    std::string body;
    auto res = cli.Post("/sse", [&](const char* data, std::size_t len) {
        body.append(data, len);
        return true;
    });
    REQUIRE(res);
    CHECK(res->status == 200);
    CHECK(body.find("data: 0") != std::string::npos);
    CHECK(body.find("data: 1") != std::string::npos);
    server.Stop();
    t.join();
}
```

- [ ] **Step 4: Run to verify it fails**

Run: `cmake --preset linux-debug && cmake --build --preset linux-debug --target test_http_server`
Expected: build FAILS.

- [ ] **Step 5: Implement `http_server.cpp`**

```cpp
#include "lsxnetwork/http_server.h"

#include "lsxnetwork/http_router.h"

#include <httplib.h>

#include <atomic>
#include <memory>
#include <utility>

namespace lsxnetwork {

struct HttpServer::Impl {
    httplib::Server server;
    std::atomic<int> port{0};
    std::unique_ptr<HttpRouter> router;
};

HttpServer::HttpServer() : impl_(std::make_unique<Impl>()) {}
HttpServer::~HttpServer() = default;

namespace {
lsxnetwork::HttpResponse ToReply(const httplib::Response& res) {
    HttpResponse out;
    out.status = res.status;
    out.body = res.body;
    auto it = res.headers.find("Content-Type");
    if (it != res.headers.end()) out.content_type = it->second;
    return out;
}
}  // namespace

void HttpServer::Get(const std::string& path, Handler handler) {
    impl_->server.Get(path, [handler](const httplib::Request& req,
                                      httplib::Response& res) {
        HttpRequest rq{req.method, req.path, req.body};
        HttpResponse rr;
        handler(rq, rr);
        res.status = rr.status;
        res.set_content(rr.body, rr.content_type.c_str());
        for (const auto& [k, v] : rr.headers) res.set_header(k, v);
    });
}

void HttpServer::Post(const std::string& path, Handler handler) {
    impl_->server.Post(path, [handler](const httplib::Request& req,
                                       httplib::Response& res) {
        HttpRequest rq{req.method, req.path, req.body};
        HttpResponse rr;
        handler(rq, rr);
        res.status = rr.status;
        res.set_content(rr.body, rr.content_type.c_str());
        for (const auto& [k, v] : rr.headers) res.set_header(k, v);
    });
}

void HttpServer::PostStream(const std::string& path, StreamFactory factory) {
    impl_->server.Post(path, [factory](const httplib::Request& req,
                                       httplib::Response& res) {
        HttpRequest rq{req.method, req.path, req.body};
        HttpResponse head;
        std::unique_ptr<StreamSource> source = factory(rq, head);
        if (!source) {
            res.status = head.status;
            res.set_content(head.body, head.content_type.c_str());
            return;
        }
        res.status = 200;
        auto shared = std::shared_ptr<StreamSource>(std::move(source));
        res.set_chunked_content_provider(
            head.content_type.c_str(),
            [shared](std::size_t, httplib::DataSink& sink) {
                if (!sink.is_writable()) {
                    shared->Cancel();
                    return false;
                }
                std::string out;
                if (!shared->Next(out)) return false;
                return sink.write(out.data(), out.size());
            });
    });
}

void HttpServer::EnableStaticSpa() {
    impl_->router = std::make_unique<HttpRouter>();
    impl_->server.Get(".*", [this](const httplib::Request& req,
                                   httplib::Response& res) {
        HttpReply reply;
        impl_->router->Handle(req.method, req.path, req.body, reply);
        res.status = reply.status;
        res.set_content(static_cast<const char*>(reply.body), reply.body_size,
                        std::string(reply.content_type).c_str());
    });
}

bool HttpServer::Listen(const std::string& host, int port) {
    if (port <= 0) {
        if (!impl_->server.bind_to_any_port(host)) return false;
        impl_->port.store(impl_->server.port());
    } else {
        if (!impl_->server.bind_to_port(host, port)) return false;
        impl_->port.store(port);
    }
    return impl_->server.listen_after_bind();
}

int HttpServer::BoundPort() const { return impl_->port.load(); }
void HttpServer::Stop() { impl_->server.stop(); }

}  // namespace lsxnetwork
```

- [ ] **Step 6: Wire the build**

In `libs/lsxnetwork/CMakeLists.txt` add `src/http_server.cpp` to the library
sources and link httplib PRIVATE (headers do not expose httplib):

```cmake
target_link_libraries(lsxnetwork PRIVATE httplib::httplib)
```

In `tests/CMakeLists.txt` add and register:

```cmake
add_executable(test_http_server test_http_server.cpp)
target_link_libraries(test_http_server PRIVATE lsxnetwork Catch2::Catch2WithMain)
add_test(NAME test_http_server COMMAND test_http_server)
```

Append `test_http_server` to the `LABELS fast` list.

- [ ] **Step 7: Run the test to verify it passes**

Run: `cmake --preset linux-debug && cmake --build --preset linux-debug --target test_http_server && ctest --preset linux-debug -R test_http_server --output-on-failure`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add CMakeLists.txt libs/lsxnetwork/include/lsxnetwork/http_server.h libs/lsxnetwork/src/http_server.cpp libs/lsxnetwork/CMakeLists.txt tests/test_http_server.cpp tests/CMakeLists.txt
git commit -m "feat: cpp-httplib HttpServer wrapper with SSE support"
```

---

### Task 7: taskmosaic config parser

**Files:**
- Create: `apps/taskmosaic/src/config.h` / `config.cpp`
- Test: `apps/taskmosaic/tests/test_taskmosaic_config.cpp`
- Modify: `apps/taskmosaic/tests/CMakeLists.txt`

**Interfaces:**
- Consumes: simdjson.
- Produces: `taskmosaic::BackendConfig{endpoint,cancel_endpoint}`,
  `taskmosaic::ModelConfig{id,model_path,backend}`,
  `taskmosaic::BridgeConfig{host,port,default_model,queue_depth,request_timeout_ms,default_max_tokens,models}`,
  `bool ParseBridgeConfig(std::string_view json, BridgeConfig& out, std::string& err)`,
  `bool LoadBridgeConfigFile(const std::string& path, BridgeConfig& out, std::string& err)`.

- [ ] **Step 1: Write the failing test**

Create `apps/taskmosaic/tests/test_taskmosaic_config.cpp`:

```cpp
#include <catch2/catch_test_macros.hpp>

#include "config.h"

#include <filesystem>
#include <fstream>
#include <string>

using namespace taskmosaic;

TEST_CASE("config: parses a full document", "[taskmosaic][config]") {
    const std::string json = R"({
      "listen": { "host": "0.0.0.0", "port": 9090 },
      "default_model": "m1",
      "queue_depth": 3,
      "request_timeout_ms": 12345,
      "default_max_tokens": 128,
      "models": [
        { "id": "m1", "model_path": "/vault/m1",
          "backend": { "endpoint": "tcp://127.0.0.1:5555",
                       "cancel_endpoint": "tcp://127.0.0.1:5556" } }
      ]
    })";
    BridgeConfig cfg;
    std::string err;
    REQUIRE(ParseBridgeConfig(json, cfg, err));
    CHECK(cfg.host == "0.0.0.0");
    CHECK(cfg.port == 9090);
    CHECK(cfg.default_model == "m1");
    CHECK(cfg.queue_depth == 3);
    CHECK(cfg.request_timeout_ms == 12345);
    CHECK(cfg.default_max_tokens == 128);
    REQUIRE(cfg.models.size() == 1);
    CHECK(cfg.models[0].id == "m1");
    CHECK(cfg.models[0].model_path == "/vault/m1");
    CHECK(cfg.models[0].backend.endpoint == "tcp://127.0.0.1:5555");
    CHECK(cfg.models[0].backend.cancel_endpoint == "tcp://127.0.0.1:5556");
}

TEST_CASE("config: defaults when optional fields are absent",
          "[taskmosaic][config]") {
    const std::string json =
        R"({"models":[{"id":"m","model_path":"/m","backend":{"endpoint":"tcp://x:1"}}]})";
    BridgeConfig cfg;
    std::string err;
    REQUIRE(ParseBridgeConfig(json, cfg, err));
    CHECK(cfg.host == "127.0.0.1");
    CHECK(cfg.port == 8080);
    CHECK(cfg.queue_depth == 4);
    CHECK(cfg.default_max_tokens == 64);
    CHECK(cfg.models[0].backend.cancel_endpoint.empty());
}

TEST_CASE("config: rejects missing models and bad port", "[taskmosaic][config]") {
    BridgeConfig cfg;
    std::string err;
    CHECK_FALSE(ParseBridgeConfig(R"({"models":[]})", cfg, err));
    CHECK_FALSE(err.empty());
    CHECK_FALSE(ParseBridgeConfig(
        R"({"listen":{"port":70000},"models":[{"id":"m","model_path":"/m","backend":{"endpoint":"tcp://x:1"}}]})",
        cfg, err));
}

TEST_CASE("config: loads from file", "[taskmosaic][config]") {
    const auto path = std::filesystem::temp_directory_path() / "lsx_bridge_cfg.json";
    std::ofstream(path) << R"({"models":[{"id":"m","model_path":"/m","backend":{"endpoint":"tcp://x:1"}}]})";
    BridgeConfig cfg;
    std::string err;
    REQUIRE(LoadBridgeConfigFile(path.string(), cfg, err));
    CHECK(cfg.models.size() == 1);
    std::filesystem::remove(path);
}
```

- [ ] **Step 2: Run it to verify it fails**

First register the target (Step 5), then:
Run: `cmake --build --preset linux-debug --target test_taskmosaic_config`
Expected: build FAILS.

- [ ] **Step 3: Write `config.h` / `config.cpp`**

`config.h`:

```cpp
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace taskmosaic {

struct BackendConfig {
    std::string endpoint;
    std::string cancel_endpoint;
};

struct ModelConfig {
    std::string id;
    std::string model_path;
    BackendConfig backend;
};

struct BridgeConfig {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string default_model;
    int queue_depth = 4;
    int request_timeout_ms = 300000;
    int default_max_tokens = 64;
    std::vector<ModelConfig> models;
};

/// Parses a bridge JSON document. On failure returns false and fills @p err.
bool ParseBridgeConfig(std::string_view json, BridgeConfig& out, std::string& err);

/// Reads and parses @p path. On failure returns false and fills @p err.
bool LoadBridgeConfigFile(const std::string& path, BridgeConfig& out,
                          std::string& err);

}  // namespace taskmosaic
```

`config.cpp`:

```cpp
#include "config.h"

#include <simdjson.h>

#include <fstream>
#include <sstream>

namespace taskmosaic {
namespace {

bool GetString(simdjson::dom::element obj, std::string_view key, std::string& out) {
    auto v = obj[key].get_string();
    if (v.error()) return false;
    out.assign(v.value_unsafe());
    return true;
}

bool GetInt(simdjson::dom::element obj, std::string_view key, int& out) {
    int64_t v = 0;
    if (obj[key].get_int64().get(v)) return false;
    out = static_cast<int>(v);
    return true;
}

}  // namespace

bool ParseBridgeConfig(std::string_view json, BridgeConfig& out, std::string& err) {
    simdjson::dom::parser parser;
    simdjson::dom::element root;
    if (parser.parse(json).get(root)) {
        err = "invalid JSON";
        return false;
    }
    BridgeConfig cfg;

    simdjson::dom::element listen;
    if (!root["listen"].get(listen)) {
        GetString(listen, "host", cfg.host);
        int port = 0;
        if (!GetInt(listen, "port", port) || port <= 0 || port > 65535) {
            err = "listen.port must be 1..65535";
            return false;
        }
        cfg.port = port;
    }
    GetString(root, "default_model", cfg.default_model);
    GetInt(root, "queue_depth", cfg.queue_depth);
    GetInt(root, "request_timeout_ms", cfg.request_timeout_ms);
    GetInt(root, "default_max_tokens", cfg.default_max_tokens);
    if (cfg.queue_depth < 1) cfg.queue_depth = 1;

    simdjson::dom::array models;
    if (root["models"].get(models)) {
        err = "models must be a non-empty array";
        return false;
    }
    for (simdjson::dom::element m : models) {
        ModelConfig mc;
        if (!GetString(m, "id", mc.id) || mc.id.empty()) {
            err = "each model needs a non-empty id";
            return false;
        }
        if (!GetString(m, "model_path", mc.model_path) || mc.model_path.empty()) {
            err = "model '" + mc.id + "' needs model_path";
            return false;
        }
        simdjson::dom::element be;
        if (!m["backend"].get(be) ||
            !GetString(be, "endpoint", mc.backend.endpoint) ||
            mc.backend.endpoint.empty()) {
            err = "model '" + mc.id + "' needs backend.endpoint";
            return false;
        }
        GetString(be, "cancel_endpoint", mc.backend.cancel_endpoint);
        cfg.models.push_back(std::move(mc));
    }
    if (cfg.models.empty()) {
        err = "models must be a non-empty array";
        return false;
    }
    out = std::move(cfg);
    return true;
}

bool LoadBridgeConfigFile(const std::string& path, BridgeConfig& out,
                          std::string& err) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        err = "cannot open config file: " + path;
        return false;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ParseBridgeConfig(ss.str(), out, err);
}

}  // namespace taskmosaic
```

- [ ] **Step 4: Register the test target**

In `apps/taskmosaic/tests/CMakeLists.txt` add:

```cmake
add_executable(test_taskmosaic_config test_taskmosaic_config.cpp ../src/config.cpp)
target_include_directories(test_taskmosaic_config PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(test_taskmosaic_config PRIVATE lsxcommon simdjson::simdjson Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_taskmosaic_config POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>" "$<TARGET_FILE_DIR:test_taskmosaic_config>")
    add_custom_command(TARGET test_taskmosaic_config POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>" "$<TARGET_FILE_DIR:test_taskmosaic_config>")
endif()

add_test(NAME test_taskmosaic_config COMMAND test_taskmosaic_config)
set_tests_properties(test_taskmosaic_config PROPERTIES LABELS fast)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cmake --preset linux-debug && cmake --build --preset linux-debug --target test_taskmosaic_config && ctest --preset linux-debug -R test_taskmosaic_config --output-on-failure`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add apps/taskmosaic/src/config.h apps/taskmosaic/src/config.cpp apps/taskmosaic/tests/test_taskmosaic_config.cpp apps/taskmosaic/tests/CMakeLists.txt
git commit -m "feat: taskmosaic bridge config parser"
```

---

### Task 8: Per-backend client + bounded admission

**Files:**
- Create: `apps/taskmosaic/src/backend.h` / `backend.cpp`
- Test: `apps/taskmosaic/tests/test_taskmosaic_backend.cpp`
- Modify: `apps/taskmosaic/tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `BackendConfig` (Task 7), `FabricStreamClient` (Task 2).
- Produces: `Backend` with `Connect()`, `Ready()`, `TryAdmit()`, `Release()`,
  `NextRequestId()`, `Client()`, `Mutex()`, `Cancel(id)`.

- [ ] **Step 1: Write the failing test**

Create `apps/taskmosaic/tests/test_taskmosaic_backend.cpp`:

```cpp
#include <catch2/catch_test_macros.hpp>

#include "backend.h"

using taskmosaic::Backend;
using taskmosaic::BackendConfig;

TEST_CASE("backend: admission is bounded by queue_depth", "[taskmosaic][backend]") {
    BackendConfig cfg;
    cfg.endpoint = "inproc://nobody";
    Backend b(cfg, 2);
    CHECK(b.TryAdmit());
    CHECK(b.TryAdmit());
    CHECK_FALSE(b.TryAdmit());  // third exceeds depth
    b.Release();
    CHECK(b.TryAdmit());
}

TEST_CASE("backend: request ids are unique and increasing",
          "[taskmosaic][backend]") {
    BackendConfig cfg;
    Backend b(cfg, 1);
    const auto a = b.NextRequestId();
    const auto c = b.NextRequestId();
    CHECK(c > a);
}
```

- [ ] **Step 2: Run it to verify it fails**

Register the target first (Step 4), then build — expect FAIL.

- [ ] **Step 3: Implement**

`backend.h`:

```cpp
#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>

#include "config.h"
#include "lsxnetwork/fabric_stream_client.h"

namespace taskmosaic {

/// One lsxfabric backend connection with bounded admission.
///
/// Admission counts *in-flight + queued* requests: a caller that cannot admit
/// maps to HTTP 429. The connection mutex serializes actual use (one request in
/// flight per backend).
class Backend {
public:
    Backend(const BackendConfig& cfg, int queue_depth);

    int Connect();
    bool Ready() const noexcept { return ready_.load(std::memory_order_acquire); }
    void Cancel() noexcept { ready_.store(false, std::memory_order_release); }

    bool TryAdmit() noexcept;
    void Release() noexcept;

    std::uint32_t NextRequestId() noexcept {
        return ++last_id_;
    }
    lsxnetwork::FabricStreamClient& Client() noexcept { return client_; }
    std::mutex& Mutex() noexcept { return mtx_; }

private:
    BackendConfig cfg_;
    int queue_depth_;
    std::atomic<int> admitted_{0};
    std::atomic<bool> ready_{false};
    std::atomic<std::uint32_t> last_id_{0};
    lsxnetwork::FabricStreamClient client_;
    std::mutex mtx_;
};

}  // namespace taskmosaic
```

`backend.cpp`:

```cpp
#include "backend.h"

namespace taskmosaic {

Backend::Backend(const BackendConfig& cfg, int queue_depth)
    : cfg_(cfg), queue_depth_(queue_depth < 1 ? 1 : queue_depth) {}

int Backend::Connect() {
    const int rc = client_.Connect(
        cfg_.endpoint.c_str(),
        cfg_.cancel_endpoint.empty() ? nullptr : cfg_.cancel_endpoint.c_str());
    ready_.store(rc == 0, std::memory_order_release);
    return rc;
}

bool Backend::TryAdmit() noexcept {
    int cur = admitted_.load(std::memory_order_relaxed);
    while (cur < queue_depth_) {
        if (admitted_.compare_exchange_weak(cur, cur + 1,
                                            std::memory_order_acq_rel))
            return true;
    }
    return false;
}

void Backend::Release() noexcept {
    admitted_.fetch_sub(1, std::memory_order_acq_rel);
}

}  // namespace taskmosaic
```

- [ ] **Step 4: Register the test**

In `apps/taskmosaic/tests/CMakeLists.txt` add a `test_taskmosaic_backend`
target compiling `test_taskmosaic_backend.cpp ../src/backend.cpp`, linking
`lsxcommon lsxnetwork ...`, with the standard Windows DLL-copy block and
`set_tests_properties(... LABELS fast)`.

- [ ] **Step 5: Run the test to verify it passes**

Run: `cmake --preset linux-debug && cmake --build --preset linux-debug --target test_taskmosaic_backend && ctest --preset linux-debug -R test_taskmosaic_backend --output-on-failure`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add apps/taskmosaic/src/backend.h apps/taskmosaic/src/backend.cpp apps/taskmosaic/tests/test_taskmosaic_backend.cpp apps/taskmosaic/tests/CMakeLists.txt
git commit -m "feat: taskmosaic backend admission"
```

---

### Task 9: OpenAI codec — request parse + response builders

**Files:**
- Modify: `libs/lsxnetwork/include/lsxnetwork/openai_endpoint.h`
- Modify: `libs/lsxnetwork/src/openai_endpoint.cpp`
- Test: `tests/test_mosaic_network.cpp` (append)

**Interfaces:**
- Consumes: simdjson.
- Produces (additive — the existing `ChatCompletion`/`CompletionResponse`/
  `WriteCompletion` stay for the legacy orchestrator): `ChatInputMessage`,
  `ChatRequest`, `LegacyCompletionRequest`, `ParseChatRequest`,
  `ParseLegacyCompletionRequest`, `JsonEscape`, `BuildChatCompletion`,
  `BuildChatChunk`, `BuildTextCompletion`, `BuildTextChunk`, `BuildModelsList`,
  `BuildModel`, `BuildError`, `SseFrame`.

- [ ] **Step 1: Append tests to `tests/test_mosaic_network.cpp`**

```cpp
TEST_CASE("openai endpoint: parses chat request with messages and tools ignored",
          "[mosaic][openai][parse]") {
    const std::string body = R"({
      "model":"m1","stream":true,"max_tokens":32,
      "stop":["<|end|>"],
      "stream_options":{"include_usage":true},
      "messages":[{"role":"system","content":"sys"},
                  {"role":"user","content":"hi"}]})";
    lsxnetwork::ChatRequest req;
    REQUIRE(lsxnetwork::OpenAiEndpoint().ParseChatRequest(
                body.data(), body.size(), body.size(), req) == simdjson::SUCCESS);
    CHECK(req.model == "m1");
    CHECK(req.stream);
    CHECK(req.max_tokens == 32);
    CHECK(req.include_usage);
    REQUIRE(req.stop.size() == 1);
    CHECK(req.stop[0] == "<|end|>");
    REQUIRE(req.messages.size() == 2);
    CHECK(req.messages[0].role == "system");
    CHECK(req.messages[1].content == "hi");
}

TEST_CASE("openai endpoint: builds completion and error JSON",
          "[mosaic][openai][build]") {
    const std::string done = lsxnetwork::BuildChatCompletion(
        "chatcmpl-1", "m1", "hello", "stop", 3, 5);
    CHECK(done.find("\"object\":\"chat.completion\"") != std::string::npos);
    CHECK(done.find("\"content\":\"hello\"") != std::string::npos);
    CHECK(done.find("\"completion_tokens\":5") != std::string::npos);

    const std::string chunk = lsxnetwork::BuildChatChunk(
        "chatcmpl-1", "m1", "he", true, "", 0, 0, false);
    CHECK(chunk.find("chat.completion.chunk") != std::string::npos);
    CHECK(chunk.find("\"role\":\"assistant\"") != std::string::npos);

    const std::string err =
        lsxnetwork::BuildError("nope", "invalid_request_error");
    CHECK(err.find("\"type\":\"invalid_request_error\"") != std::string::npos);

    CHECK(lsxnetwork::JsonEscape("a\"b\n") == "a\\\"b\\n");
    CHECK(lsxnetwork::SseFrame("data") == "data: data\n\n");
}
```

Add `#include <lsxnetwork/openai_endpoint.h>` (already present) and
`#include <simdjson.h>` (already present).

- [ ] **Step 2: Run to verify it fails**

Run: `cmake --build --preset linux-debug --target test_mosaic_network`
Expected: build FAILS.

- [ ] **Step 3: Extend `openai_endpoint.h`**

Append inside `namespace lsxnetwork`:

```cpp
struct ChatInputMessage {
    std::string role;
    std::string content;
};

struct ChatRequest {
    std::string model;
    std::vector<ChatInputMessage> messages;
    bool stream = false;
    int max_tokens = 0;
    std::vector<std::string> stop;
    bool include_usage = false;
};

struct LegacyCompletionRequest {
    std::string model;
    std::string prompt;
    bool stream = false;
    int max_tokens = 0;
    std::vector<std::string> stop;
    bool include_usage = false;
};

// --- new methods to add to class OpenAiEndpoint ---
simdjson::error_code ParseChatRequest(const char* data, std::size_t size,
                                      std::size_t capacity,
                                      ChatRequest& out) const;
simdjson::error_code ParseLegacyCompletionRequest(const char* data,
                                                  std::size_t size,
                                                  std::size_t capacity,
                                                  LegacyCompletionRequest& out) const;

std::string JsonEscape(const std::string& in);
std::string SseFrame(const std::string& data);

std::string BuildChatCompletion(const std::string& id, const std::string& model,
                                const std::string& text,
                                const std::string& finish_reason,
                                int prompt_tokens, int completion_tokens);
std::string BuildChatChunk(const std::string& id, const std::string& model,
                           const std::string& delta_content, bool first_delta,
                           const std::string& finish_reason, int prompt_tokens,
                           int completion_tokens, bool include_usage);
std::string BuildTextCompletion(const std::string& id, const std::string& model,
                                const std::string& text,
                                const std::string& finish_reason,
                                int prompt_tokens, int completion_tokens);
std::string BuildTextChunk(const std::string& id, const std::string& model,
                           const std::string& delta_text,
                           const std::string& finish_reason);
std::string BuildModelsList(const std::vector<std::string>& ids);
std::string BuildModel(const std::string& id);
std::string BuildError(const std::string& message, const std::string& type);
```

- [ ] **Step 4: Implement in `openai_endpoint.cpp`**

Add the methods (the existing v1 `Parse`/`WriteCompletion` stay unchanged).
Key implementations:

```cpp
simdjson::error_code OpenAiEndpoint::ParseChatRequest(const char* data,
                                                      std::size_t size,
                                                      std::size_t capacity,
                                                      ChatRequest& out) const {
    simdjson::ondemand::parser parser;
    simdjson::ondemand::document doc;
    simdjson::padded_string_view v(std::string_view(data, size), capacity);
    if (auto e = parser.iterate(v).get(doc)) return e;
    simdjson::ondemand::object root;
    if (auto e = doc.get_object().get(root)) return e;

    if (auto e = root["model"].get_string(out.model)) {
        if (e != simdjson::NO_SUCH_FIELD) return e;
    }
    if (auto s = root["stream"].get_bool(); !s.error()) out.stream = s.value_unsafe();
    if (auto mt = root["max_tokens"].get_int64(); !mt.error())
        out.max_tokens = static_cast<int>(mt.value_unsafe());
    if (auto so = root["stream_options"].get_object(); !so.error()) {
        bool iu = false;
        if (auto e = so.value()["include_usage"].get_bool(); !e.error()) iu = true;
        out.include_usage = iu;
    }
    if (auto stop = root["stop"].get_array(); !stop.error()) {
        for (auto s : stop.value()) {
            std::string_view sv;
            if (!s.get_string().get(sv)) out.stop.emplace_back(sv);
        }
    } else if (auto stop_str = root["stop"].get_string(); !stop_str.error()) {
        out.stop.emplace_back(stop_str.value_unsafe());
    }
    simdjson::ondemand::array messages;
    if (auto e = root["messages"].get_array().get(messages)) return e;
    for (auto m : messages) {
        simdjson::ondemand::object mo;
        if (auto e = m.get_object().get(mo)) return e;
        ChatInputMessage msg;
        std::string_view s;
        if (auto e = mo["role"].get_string().get(s); !e) msg.role.assign(s);
        if (auto c = mo["content"].get_string(); !c.error())
            msg.content.assign(c.value_unsafe());
        out.messages.push_back(std::move(msg));
    }
    return simdjson::SUCCESS;
}
```

`ParseLegacyCompletionRequest` is the same shape with `prompt` (string or
string array — concatenate array elements).

JSON escaping + builders:

```cpp
std::string JsonEscape(const std::string& in) {
    std::string out;
    out.reserve(in.size() + 8);
    static const char* kHex = "0123456789abcdef";
    for (char c : in) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    out += "\\u00";
                    out += kHex[(c >> 4) & 0xF];
                    out += kHex[c & 0xF];
                } else {
                    out += c;
                }
        }
    }
    return out;
}

std::string SseFrame(const std::string& data) { return "data: " + data + "\n\n"; }

std::string BuildChatCompletion(const std::string& id, const std::string& model,
                                const std::string& text,
                                const std::string& finish_reason,
                                int prompt_tokens, int completion_tokens) {
    return std::string("{\"id\":\"") + id + "\",\"object\":\"chat.completion\",\"model\":\"" +
           model + "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"" +
           JsonEscape(text) + "\"},\"finish_reason\":\"" + finish_reason +
           "\"}],\"usage\":{\"prompt_tokens\":" + std::to_string(prompt_tokens) +
           ",\"completion_tokens\":" + std::to_string(completion_tokens) +
           ",\"total_tokens\":" + std::to_string(prompt_tokens + completion_tokens) + "}}";
}

std::string BuildChatChunk(const std::string& id, const std::string& model,
                           const std::string& delta_content, bool first_delta,
                           const std::string& finish_reason, int prompt_tokens,
                           int completion_tokens, bool include_usage) {
    std::string out = "{\"id\":\"" + id + "\",\"object\":\"chat.completion.chunk\",\"model\":\"" +
                      model + "\",\"choices\":[{\"index\":0,\"delta\":{";
    if (first_delta) out += "\"role\":\"assistant\",";
    out += "\"content\":\"" + JsonEscape(delta_content) + "\"},\"finish_reason\":";
    out += finish_reason.empty() ? "null" : ("\"" + finish_reason + "\"");
    out += "}]";
    if (include_usage && !finish_reason.empty()) {
        out += ",\"usage\":{\"prompt_tokens\":" + std::to_string(prompt_tokens) +
               ",\"completion_tokens\":" + std::to_string(completion_tokens) +
               ",\"total_tokens\":" + std::to_string(prompt_tokens + completion_tokens) + "}";
    }
    out += "}";
    return out;
}
```

Implement `BuildTextCompletion`/`BuildTextChunk` analogously
(`"object":"text_completion"`, `choices[0].text`). Implement `BuildModelsList`,
`BuildModel`, `BuildError`:

```cpp
std::string BuildModelsList(const std::vector<std::string>& ids) {
    std::string data = "[";
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i) data += ",";
        data += BuildModel(ids[i]);
    }
    data += "]";
    return "{\"object\":\"list\",\"data\":" + data + "}";
}
std::string BuildModel(const std::string& id) {
    return "{\"id\":\"" + JsonEscape(id) +
           "\",\"object\":\"model\",\"created\":0,\"owned_by\":\"logestix\"}";
}
std::string BuildError(const std::string& message, const std::string& type) {
    return "{\"error\":{\"message\":\"" + JsonEscape(message) + "\",\"type\":\"" +
           JsonEscape(type) + "\",\"param\":null,\"code\":null}}";
}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cmake --build --preset linux-debug --target test_mosaic_network && ctest --preset linux-debug -R test_mosaic_network --output-on-failure`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add libs/lsxnetwork/include/lsxnetwork/openai_endpoint.h libs/lsxnetwork/src/openai_endpoint.cpp tests/test_mosaic_network.cpp
git commit -m "feat: OpenAI chat request parse and response builders"
```

---

### Task 10: Prompt framer + chat bridge + HTTP routes

**Files:**
- Create: `apps/taskmosaic/src/framer.h`, `lsx_framer.h` / `lsx_framer.cpp`
- Create: `apps/taskmosaic/src/chat_bridge.h` / `chat_bridge.cpp`
- Test: `apps/taskmosaic/tests/test_taskmosaic_bridge.cpp`
- Modify: `apps/taskmosaic/tests/CMakeLists.txt`

**Interfaces:**
- Consumes: Tasks 6–9.
- Produces: `PromptFramer` (interface), `StreamingDecoder` (interface),
  `LsxPromptFramer(PromptFramer)` (real), `ChatBridge(config, framer)` with
  `Initialize(err)`, `Listen()`, `BoundPort()`, `Stop()`.

- [ ] **Step 1: Write the failing integration test**

The test uses a `FakeFramer` (text "hi" → ids {1,2}; ids → text "<n>") and a
minimal fake ROUTER/PULL backend, then drives the HTTP server.

```cpp
#include <catch2/catch_test_macros.hpp>

#include "chat_bridge.h"
#include "framer.h"

#include <httplib.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <zmq.h>
// Fake ROUTER server emits token ids 10,11 then done for every request.
// (Loop body mirrors the FakeServer in tests/test_fabric_client.cpp, but sends
//  EncodeFabricTokenBatch(rid, {10}) and then {11}, then EncodeFabricDone.)
struct FakeBackend { /* ... ROUTER + PULL, thread, Start(router,cancel) ... */ };

struct FakeFramer : taskmosaic::PromptFramer {
    bool Encode(std::string_view, std::string_view raw,
                std::vector<std::int32_t>& ids) override {
        prompts.emplace_back(raw);
        ids = {1, 2};
        return true;
    }
    std::string Decode(std::string_view, const std::vector<std::int32_t>&) override {
        return "hello";
    }
    std::unique_ptr<taskmosaic::StreamingDecoder> MakeDecoder(std::string_view) override;
    std::vector<std::string> prompts;
};

TEST_CASE("bridge: non-streaming chat completion", "[taskmosaic][bridge][http]") {
    // start FakeBackend at tcp://127.0.0.1:<ephemeral>
    // cfg.models = {{"m1","/fake","tcp://...router","tcp://...cancel"}}
    // FakeFramer framer; ChatBridge bridge(cfg, framer);
    // REQUIRE(bridge.Initialize(err));
    // thread t([&]{ bridge.Listen(); });
    // httplib::Client cli("127.0.0.1", bridge.BoundPort());
    // auto res = cli.Post("/v1/chat/completions",
    //     R"({"model":"m1","messages":[{"role":"user","content":"hi"}]})",
    //     "application/json");
    // REQUIRE(res); CHECK(res->status == 200);
    // CHECK(res->body.find("\"content\":\"hello\"") != std::string::npos);
    // brige.Stop(); t.join();
}

TEST_CASE("bridge: unknown model 404 and /v1/models lists ids",
          "[taskmosaic][bridge][http]") { /* analogous */ }
```

(For brevity in the test file, implement the concrete FakeBackend by copying
the `FakeServer` helper from `tests/test_fabric_client.cpp` and adjusting the
emitted frames. The bridge tests must bind tcp endpoints; have the fake backend
bind `tcp://127.0.0.1:*` and report its resolved endpoints via `ZMQ_LAST_ENDPOINT`.)

- [ ] **Step 2: Run to verify it fails**

Register the target (Step 6) and build — expect FAIL.

- [ ] **Step 3: Implement `framer.h`, `lsx_framer.*`**

```cpp
// framer.h
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace taskmosaic {

/// Per-request streaming token decoder (UTF-8-safe).
class StreamingDecoder {
public:
    virtual ~StreamingDecoder() = default;
    virtual std::string Push(std::int32_t id) = 0;
    virtual std::string Flush() = 0;
};

/// Host-only model framing/decoding boundary, injectable for tests.
class PromptFramer {
public:
    virtual ~PromptFramer() = default;
    virtual bool Encode(std::string_view model_path, std::string_view raw_prompt,
                        std::vector<std::int32_t>& out_ids) = 0;
    virtual std::string Decode(std::string_view model_path,
                               const std::vector<std::int32_t>& ids) = 0;
    virtual std::unique_ptr<StreamingDecoder> MakeDecoder(
        std::string_view model_path) = 0;
};

}  // namespace taskmosaic
```

`lsx_framer.h`:

```cpp
#pragma once
#include "framer.h"
#include <absl/container/flat_hash_map.h>
#include <memory>
#include <mutex>
#include <string>

namespace taskmosaic {

/// Real framer: per-family chat framing via lsxcommon::FramePromptForModel,
/// full decode via runtime::DecodeGeneratedText, and the Arrow tokenizer-backed
/// incremental decoder (tokenizer DB cached per model path).
class LsxPromptFramer : public PromptFramer {
public:
    bool Encode(std::string_view model_path, std::string_view raw_prompt,
                std::vector<std::int32_t>& out_ids) override;
    std::string Decode(std::string_view model_path,
                       const std::vector<std::int32_t>& ids) override;
    std::unique_ptr<StreamingDecoder> MakeDecoder(std::string_view model_path) override;

private:
    lsxcommon::ArrowTokenizerDb* TokenizerFor(std::string_view model_path);
    std::mutex mtx_;
    absl::flat_hash_map<std::string, std::unique_ptr<lsxcommon::ArrowTokenizerDb>> toks_;
};

}  // namespace taskmosaic
```

`lsx_framer.cpp`: `Encode` calls `lsxcommon::FramePromptForModel(std::string(model_path), raw_prompt, out_ids)`;
`Decode` uses the cached tokenizer + `lsxcommon::runtime::DecodeGeneratedText`;
`MakeDecoder` wraps `lsxcommon::IncrementalDetokenizer::ArrowDecoder(*tok)` in a
small adapter class. `TokenizerFor` lazily calls
`lsxcommon::LoadTokenizerArrows(std::string(model_path), *db)`.

- [ ] **Step 4: Implement `chat_bridge.*`**

```cpp
// chat_bridge.h
#pragma once
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "backend.h"
#include "config.h"
#include "framer.h"
#include "lsxnetwork/http_server.h"

namespace taskmosaic {

/// OpenAI-compatible HTTP bridge: routes requests to per-model lsxfabric
/// backends, streams SSE, and cancels on client disconnect.
class ChatBridge {
public:
    ChatBridge(BridgeConfig cfg, PromptFramer& framer);
    ~ChatBridge();

    bool Initialize(std::string& err);
    bool Listen();          // blocks
    int BoundPort() const;
    void Stop();

    /// Stop watcher hook: Listen() polls this flag and calls Stop() when it
    /// flips (used for SIGINT/SIGTERM in main.cpp).
    void SetStopFlag(std::atomic<bool>* flag) noexcept { stop_flag_ = flag; }

    const BridgeConfig& Config() const noexcept { return cfg_; }

private:
    struct ModelRuntime {
        ModelConfig cfg;
        std::unique_ptr<Backend> backend;
    };
    void RegisterRoutes();
    ModelRuntime* Find(const std::string& id);

    BridgeConfig cfg_;
    PromptFramer& framer_;
    std::vector<ModelRuntime> models_;
    std::atomic<bool>* stop_flag_ = nullptr;
    lsxnetwork::HttpServer server_;
};

}  // namespace taskmosaic
```

`chat_bridge.cpp` (core logic):

```cpp
// RegisterRoutes: /health, /ready, /v1/models, /v1/models/{id},
//                 /v1/chat/completions (non-stream + stream), /v1/completions.
// Non-stream chat handler:
//   parse ChatRequest; find model (404 error JSON); flatten messages to a raw
//   prompt; framer_.Encode -> ids; backend.TryAdmit() else 429;
//   lock backend.Mutex(); Submit(NextRequestId(), ids, max_tokens?default);
//   loop Recv until done/error accumulating ids; framer_.Decode -> text;
//   BuildChatCompletion(...); backend.Release().
// Stream chat handler: PostStream(path, factory) where the factory captures the
//   parsed request, finds the model, flattens, encodes, admits, and returns a
//   ChatStreamSource that holds a unique_lock on backend.Mutex() for its life.
// /v1/models/{id} must be registered before /v1/models (httplib exact match).

class ChatStreamSource : public lsxnetwork::StreamSource {
public:
    ChatStreamSource(Backend& b, PromptFramer& f, std::string model_path,
                     std::string id, std::string model, std::string finish_id_prefix)
        : backend_(b), framer_(f), model_path_(std::move(model_path)),
          id_(std::move(id)), model_(std::move(model)) {
        lock_ = std::make_unique<std::unique_lock<std::mutex>>(b.Mutex());
        decoder_ = f.MakeDecoder(model_path_);
    }
    ~ChatStreamSource() override { backend_.Release(); }
    bool Next(std::string& out) override;
    void Cancel() override { cancelled_.store(true); if (request_id_) backend_.Client().Cancel(request_id_); }
private:
    Backend& backend_;
    PromptFramer& framer_;
    std::string model_path_, id_, model_;
    std::unique_ptr<std::unique_lock<std::mutex>> lock_;
    std::unique_ptr<StreamingDecoder> decoder_;
    bool submitted_ = false;
    bool done_ = false;
    std::uint32_t request_id_ = 0;
    std::atomic<bool> cancelled_{false};
};
```

`Next` semantics: on first call, sends `[DONE]`-not-yet: if not submitted, submit
and return the first delta (possibly empty leading chunk). Each subsequent call
`Recv` a frame; on token batch, push each id through the decoder and append SSE
frames; on done, append the final chunk (finish_reason) + `data: [DONE]\n\n`,
return false. On error, append an error SSE frame + `[DONE]`, return false. On
`Recv` timeout, append nothing and return true (keep waiting) unless the request
timeout elapsed, then emit an error and stop.

`build raw prompt`: helper `FlattenMessages(const std::vector<ChatInputMessage>&)`
joining `role: content` lines with `\n`.

- [ ] **Step 5: Run the test to verify it passes**

Run: `cmake --preset linux-debug && cmake --build --preset linux-debug --target test_taskmosaic_bridge && ctest --preset linux-debug -R test_taskmosaic_bridge --output-on-failure`
Expected: PASS (non-stream chat, models list, 404).

- [ ] **Step 6: Register the test target**

In `apps/taskmosaic/tests/CMakeLists.txt` add
`test_taskmosaic_bridge` compiling `test_taskmosaic_bridge.cpp` plus
`../src/config.cpp ../src/backend.cpp ../src/chat_bridge.cpp ../src/lsx_framer.cpp`,
linking `lsxcommon lsxnetwork simdjson::simdjson arrow_shared arrow_cuda_shared`,
label `fast` (+ Windows DLL copies).

- [ ] **Step 7: Commit**

```bash
git add apps/taskmosaic/src/framer.h apps/taskmosaic/src/lsx_framer.h apps/taskmosaic/src/lsx_framer.cpp apps/taskmosaic/src/chat_bridge.h apps/taskmosaic/src/chat_bridge.cpp apps/taskmosaic/tests/test_taskmosaic_bridge.cpp apps/taskmosaic/tests/CMakeLists.txt
git commit -m "feat: taskmosaic chat bridge and OpenAI HTTP routes"
```

---

### Task 11: taskmosaic server entry point + CMake; remove legacy client

**Files:**
- Rewrite: `apps/taskmosaic/src/main.cpp`
- Modify: `apps/taskmosaic/src/cli.h` / `cli.cpp`
- Modify: `apps/taskmosaic/CMakeLists.txt`
- Modify/Delete: `apps/taskmosaic/tests/test_taskmosaic.cpp`,
  `apps/taskmosaic/tests/test_taskmosaic_cli.cpp`

**Interfaces:**
- Consumes: Task 10 `ChatBridge`, Task 7 `LoadBridgeConfigFile`.
- Produces: `taskmosaic --config <file> [--host H] [--port P]` server mode.

- [ ] **Step 1: Update CLI parsing test**

Rewrite `apps/taskmosaic/tests/test_taskmosaic_cli.cpp`:

```cpp
#include <catch2/catch_test_macros.hpp>

#include "cli.h"

TEST_CASE("taskmosaic cli: server flags", "[taskmosaic][cli]") {
    const char* argv[] = {"taskmosaic", "--config", "/tmp/c.json",
                          "--host", "0.0.0.0", "--port", "9000"};
    auto a = taskmosaic::ParseTaskMosaicCliArgs(7, const_cast<char**>(argv));
    CHECK(a.config == "/tmp/c.json");
    CHECK(a.host == "0.0.0.0");
    CHECK(a.port == 9000);
}
```

- [ ] **Step 2: Rewrite `cli.h`**

```cpp
#pragma once

#include <string>

namespace taskmosaic {

/// taskmosaic is the OpenAI/Anthropic HTTP bridge to a running lsxfabric.
struct TaskMosaicCliArgs {
    std::string config;               // required: bridge JSON config path
    std::string host;                 // optional override
    int port = 0;                     // optional override (0 = keep config)
};

TaskMosaicCliArgs ParseTaskMosaicCliArgs(int argc, char** argv);

}  // namespace taskmosaic
```

`cli.cpp`: parse `--config`, `--host`, `--port` (reuse the existing positive-int
helper); unknown tokens ignored (project convention).

- [ ] **Step 3: Rewrite `main.cpp`**

```cpp
#include "lsxcommon/core.h"
#include "lsxcommon/log.h"

#include <absl/strings/str_format.h>

#include <atomic>
#include <csignal>
#include <string>

#include "chat_bridge.h"
#include "cli.h"
#include "config.h"
#include "lsx_framer.h"

namespace {
std::atomic<bool>* g_stop = nullptr;
void OnSignal(int) {
    if (g_stop != nullptr) g_stop->store(true);
}
}  // namespace

int main(int argc, char* argv[]) {
    const taskmosaic::TaskMosaicCliArgs cli =
        taskmosaic::ParseTaskMosaicCliArgs(argc, argv);

    lsxcommon::Core::Initialize("taskmosaic_runtime.log");
    lsxcommon::version();

    int rc = 0;
    {
        if (cli.config.empty()) {
            lsxcommon::log::error("taskmosaic: pass --config <bridge.json>");
            rc = 2;
        } else {
            taskmosaic::BridgeConfig cfg;
            std::string err;
            if (!taskmosaic::LoadBridgeConfigFile(cli.config, cfg, err)) {
                lsxcommon::log::error("config error: " + err);
                rc = 2;
            } else {
                if (!cli.host.empty()) cfg.host = cli.host;
                if (cli.port > 0) cfg.port = cli.port;
                taskmosaic::LsxPromptFramer framer;
                taskmosaic::ChatBridge bridge(cfg, framer);
                if (!bridge.Initialize(err)) {
                    lsxcommon::log::error("bridge init failed: " + err);
                    rc = 1;
                } else {
                    std::atomic<bool> stop{false};
                    g_stop = &stop;
                    std::signal(SIGINT, OnSignal);
                    std::signal(SIGTERM, OnSignal);
                    lsxcommon::log::info(absl::StrFormat(
                        "taskmosaic bridge on %s:%d (%u models)", cfg.host,
                        cfg.port, static_cast<unsigned>(cfg.models.size())));
                    bridge.SetStopFlag(&stop);
                    // Bridge::Listen blocks; the stop flag watcher inside
                    // ChatBridge::Listen calls HttpServer::Stop on signal.
                    bridge.Listen();
                    g_stop = nullptr;
                }
            }
        }
    }
    lsxcommon::Core::Shutdown();
    return rc;
}
```

`ChatBridge` must expose a `Stop()` and arrange for a signal watcher: add to
`ChatBridge` a `std::atomic<bool>* stop_flag_` settable via
`SetStopFlag(std::atomic<bool>*)`; `Listen()` starts a watcher thread that calls
`server_.Stop()` when the flag flips, then joins it after `server_.Listen()`
returns. Update `main.cpp` to call `bridge.SetStopFlag(&stop)` before `Listen()`.

- [ ] **Step 4: Update `apps/taskmosaic/CMakeLists.txt`**

```cmake
add_executable(taskmosaic
    src/main.cpp src/cli.cpp src/config.cpp src/backend.cpp
    src/chat_bridge.cpp src/lsx_framer.cpp)
target_link_libraries(taskmosaic PRIVATE lsxcommon lsxnetwork simdjson::simdjson
    arrow_shared arrow_cuda_shared ${CMAKE_DL_LIBS})
```

(Retain the existing `lsx_copy_third_party_notices`, Windows DLL-copy block and
`install()`.)

- [ ] **Step 5: Delete/repurpose the old tests**

Delete `apps/taskmosaic/tests/test_taskmosaic.cpp` and its target (the old
client MVP); keep `test_taskmosaic_cli`. Remove stale references in
`apps/taskmosaic/tests/CMakeLists.txt`.

- [ ] **Step 6: Build and run**

Run: `cmake --preset linux-debug && cmake --build --preset linux-debug --target taskmosaic test_taskmosaic_cli`
Run: `ctest --preset linux-debug -R "test_taskmosaic" --output-on-failure`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add -A apps/taskmosaic
git commit -m "feat: taskmosaic OpenAI HTTP bridge server entry point"
```

---

### Task 12: End-to-end smoke, docs, guard check

**Files:**
- Modify: `AGENTS.md` (taskmosaic description + cpp-httplib dependency row)
- Modify: `README.md` (dependency table if present)
- Test: optional `inference`-labeled smoke (documented, not run in CI)

**Interfaces:**
- Consumes: everything above.
- Produces: docs; a manual real-model verification recipe.

- [ ] **Step 1: Verify the whole fast suite is green**

Run: `cmake --build --preset linux-debug && ctest --preset linux-debug --output-on-failure`
Expected: PASS; `test_ctest_labels_guard` passes (new targets labeled).

- [ ] **Step 2: Verify guard tests**

Run: `ctest --preset linux-debug -R "labels|naming|independence|suite" --output-on-failure`
Expected: PASS.

- [ ] **Step 3: Document**

- In `AGENTS.md` §Dependencies add a `cpp-httplib | v0.18.3 | FetchContent`
  row; in §Structure update the `apps/taskmosaic/` description to
  "OpenAI/Anthropic HTTP bridge to lsxfabric".
- Add a short "Running the bridge" recipe: start `lsxfabric --serve
  --localmodel <model> --listen tcp://*:5555 --listen-cancel tcp://*:5556`,
  write a bridge config, start `taskmosaic --config bridge.json`, then
  `curl http://127.0.0.1:8080/v1/models` and a streaming
  `curl -N -X POST .../v1/chat/completions -d '{"model":"<id>","stream":true,"messages":[{"role":"user","content":"hi"}]}'`.

- [ ] **Step 4: Commit**

```bash
git add AGENTS.md README.md
git commit -m "docs: document the taskmosaic OpenAI bridge"
```

---

## Self-Review (spec coverage)

| Spec section | Plan 1 task |
|---|---|
| §4 wire v2 | Task 1 |
| §5 lsxfabric server + cancel + CLI | Tasks 3, 4 |
| §6.1 family streaming | deferred to Plan 2 (foundation uses the existing callback; gigachat/nemotron stream now, others replay via the model) |
| §6.2 tool calling | deferred to Plan 3 |
| §7.1 config | Task 7 |
| §7.2 model context | Task 10 (`LsxPromptFramer` cache) |
| §7.3 backend + queue | Task 8 |
| §7.4 incremental detokenizer | Task 5 |
| §7.5 format adapters (OpenAI) | Task 9 |
| §7.6 server entry + legacy removal | Task 11 |
| §7.7 HttpServer wrapper | Task 6 |
| §8 OpenAI endpoints | Tasks 9, 10, 11 |
| §8 Anthropic | deferred to Plan 4 |
| §9 compatibility extras | deferred to Plan 5 (partial: `include_usage` plumbed in Task 9) |
| §10 concurrency/cancel | Tasks 3, 8, 10 |
| §11 errors | Tasks 9, 10 |
| §12 tests | Tasks 1–12 |
| §13 CMake/deps | Task 6 |

**Known deferrals (by design):** real per-token TTFT for non-gigachat/nemotron
families (Plan 2); tool calling (Plan 3); Anthropic (Plan 4); `stop`, tolerant
params, full finish-reason mapping (Plan 5).

**Placeholder scan:** the only intentionally abbreviated test bodies are the
`FakeBackend` scaffolding in Task 10 Step 1 (the concrete ROUTER loop is
specified by reference to the Task 2 `FakeServer`, which is complete code) and
the `ChatStreamSource::Next` narrative in Task 10 Step 4 (its state fields and
branch behavior are fully specified; the implementer fills the SSE assembly
using the Task 9 builders). Everything else is complete code.

**Type consistency:** `FabricStreamClient::RecvStatus`/`FabricFrameType`/
`EncodeFabric*`/`DecodeFabric*` names are identical across Tasks 1–3 and the
tests. `Backend`/`BackendConfig`/`BridgeConfig`/`ModelConfig` names match across
Tasks 7–11. `PromptFramer`/`StreamingDecoder`/`ChatBridge` match between Task 10
and Task 11. `ChatRequest`/`BuildChatCompletion`/`BuildChatChunk`/`BuildModelsList`/
`BuildError`/`SseFrame` match between Tasks 9 and 10.

## Execution Handoff

Plan 1 complete. Execute **inline** (AGENTS.md forbids subagents) via
superpowers:executing-plans with checkpoints between tasks. After Plan 1 lands
and passes the fast suite, proceed to Plan 2 (all-family streaming), then
Plan 3 (tool calling), Plan 4 (Anthropic adapter), Plan 5 (compatibility).
