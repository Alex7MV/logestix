# lsxfabric Binary-Token Server + taskmosaic Client — Implementation Plan

> **For agentic workers:** Per `AGENTS.md` ("No parallel/subagents"), do NOT use
> subagent-driven-development or dispatch subagents. Execute this plan inline in
> the main session, using the `superpowers:executing-plans` skill. Steps use
> `- [ ]` checkboxes.

**Goal:** Add a ZeroMQ binary-token inference server to `lsxfabric` and a matching client used by a client-only `taskmosaic` MVP.

**Architecture:** A pure codec (`fabric_wire`) and blocking REQ client (`FabricClient`) live in `libs/lsxnetwork`. A single-threaded REP `FabricServer` in `apps/lsxfabric` wraps an injected `lsxcommon::IModel`. `taskmosaic` loads the tokenizer/framing host-only, calls the client, and decodes the result.

**Tech Stack:** C++20, CMake 3.25+, libzmq (C API, via `lsxnetwork`), Arrow, Catch2 v3.16, abseil (host-only).

## Global Constraints

- Target Blackwell CC 12.0/12.1 only; no pre-Blackwell code.
- Host code C++20; logging only via spdlog / `lsxcommon::log` — never `std::cout`/`printf` in lib/app code.
- Host-only container policy prefers `absl::`, but these new headers are not reachable from `.cu`/`.cuh`; `std::vector`/`std::span` are fine here.
- Wire format is little-endian, explicit byte-wise, magic `"LXFB"`, version `1`.
- `REQ`/`REP` is strictly one request in flight; no internal threads in the server.
- New tests: label `fast` in the SAME commit as the target (`test-lifecycle`, enforced by `tests/test_ctest_labels_guard.cpp`).
- New `tests/` files must match `test_*.cpp`; new app test targets must be registered in their app `tests/CMakeLists.txt` label call.
- Build: `cmake --preset linux-debug` then `cmake --build --preset linux-debug --target <t>`; test: `ctest --preset linux-debug -R <t>`.
- Commit prefixes: `test:`/`feat:`/`build:`/`docs:`. Commit only as the plan instructs.
- Branch: `feat/lsxfabric-binary-server` (already created; design doc committed at `a6e1f93`).

---

### Task 1: Binary wire codec (`fabric_wire`)

**Files:**
- Create: `libs/lsxnetwork/include/lsxnetwork/fabric_wire.h`
- Create: `libs/lsxnetwork/src/fabric_wire.cpp`
- Create: `tests/test_fabric_wire.cpp`
- Modify: `libs/lsxnetwork/CMakeLists.txt` (add `src/fabric_wire.cpp`)
- Modify: `tests/CMakeLists.txt` (add executable + test + add to `LABELS fast` list)

**Interfaces:**
- Produces (later tasks rely on these exact names/types):
  - `lsxnetwork::kFabricMagic` (`std::uint32_t`), `kFabricVersion` (`std::uint16_t`)
  - `kFabricRequestHeaderBytes = 16`, `kFabricResponseHeaderBytes = 40`
  - `enum class FabricStatus : std::uint16_t { kOk=0, kBadRequest=1, kInferFailed=2, kInternal=3 }`
  - `enum class WireStatus { kOk, kBadMagic, kBadVersion, kTruncated, kTrailing }`
  - `struct FabricRequestView { std::uint32_t max_gen; std::span<const std::int32_t> tokens; }`
  - `struct FabricResponse { std::int32_t status; bool eos_terminated; double ttft_ms, total_ms, tokens_per_sec; std::vector<std::int32_t> tokens; std::string error; }`
  - `std::vector<std::byte> EncodeFabricRequest(std::span<const std::int32_t> tokens, std::uint32_t max_gen)`
  - `WireStatus DecodeFabricRequest(std::span<const std::byte> frame, FabricRequestView& out)`
  - `std::vector<std::byte> EncodeFabricResponse(const FabricResponse& r)`
  - `WireStatus DecodeFabricResponse(std::span<const std::byte> frame, FabricResponse& out)`

- [ ] **Step 1: Write the failing codec test**

Create `tests/test_fabric_wire.cpp`:

```cpp
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
```

- [ ] **Step 2: Wire the target into CMake, run the test, verify it fails**

In `libs/lsxnetwork/CMakeLists.txt`, add `src/fabric_wire.cpp` to the `add_library(lsxnetwork STATIC ...)` source list (after `src/frame_queue.cpp`).

Create `libs/lsxnetwork/include/lsxnetwork/fabric_wire.h` as an EMPTY placeholder first (so the target still configures), then build:

Run:
```bash
cmake --preset linux-debug
cmake --build --preset linux-debug --target test_fabric_wire
```
Expected: configure error until the test target exists. So first add the test block and header stub:

Append to `tests/CMakeLists.txt` (after the `test_integrated_pipeline` block, before the CUDA kernel section):

```cmake
# ── lsxfabric binary-token wire codec ────────────────────────────────────────
add_executable(test_fabric_wire test_fabric_wire.cpp)
target_link_libraries(test_fabric_wire PRIVATE lsxnetwork Catch2::Catch2WithMain)
add_test(NAME test_fabric_wire COMMAND test_fabric_wire)
```

Add both `test_fabric_wire` (and the later `test_fabric_client`) to the `set_tests_properties(... PROPERTIES LABELS fast)` list near the end of `tests/CMakeLists.txt`, right after `test_integrated_pipeline`:

```cmake
    test_fabric_wire
    test_fabric_client
```

Create the header with the full declarations from Step 3 (a stub is unnecessary — just write the real header now; the test still fails because `fabric_wire.cpp` is not yet written). Then:

Run:
```bash
cmake --preset linux-debug
cmake --build --preset linux-debug --target test_fabric_wire
```
Expected: LINK error / undefined reference to `lsxnetwork::EncodeFabricRequest` (and friends) — the RED state.

- [ ] **Step 3: Write the header**

Create `libs/lsxnetwork/include/lsxnetwork/fabric_wire.h`:

```cpp
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace lsxnetwork {

/// Binary token protocol (little-endian, explicit byte-wise encoding):
///   request  = 16-byte header + 4 * num_tokens bytes
///   response = 40-byte header + (tokens | utf-8 error) bytes
inline constexpr std::uint32_t kFabricMagic = 0x4246584Cu;  // "LXFB" LE
inline constexpr std::uint16_t kFabricVersion = 1;

inline constexpr std::size_t kFabricRequestHeaderBytes = 16;
inline constexpr std::size_t kFabricResponseHeaderBytes = 40;

enum class FabricStatus : std::uint16_t {
    kOk = 0,
    kBadRequest = 1,
    kInferFailed = 2,
    kInternal = 3,
};

/// Codec verdict. kOk means the frame was accepted; every other value maps to
/// a `kBadRequest` reply on the server side.
enum class WireStatus {
    kOk,
    kBadMagic,
    kBadVersion,
    kTruncated,
    kTrailing,
};

/// Non-owning view into a decoded request frame. `tokens` aliases the caller's
/// frame buffer, which must outlive the view.
struct FabricRequestView {
    std::uint32_t max_gen = 0;  // 0 => server default
    std::span<const std::int32_t> tokens;
};

/// Owned response. On status != kOk, `tokens` is empty and `error` carries a
/// UTF-8 message (possibly empty).
struct FabricResponse {
    std::int32_t status = static_cast<std::int32_t>(FabricStatus::kOk);
    bool eos_terminated = false;
    double ttft_ms = 0.0;
    double total_ms = 0.0;
    double tokens_per_sec = 0.0;
    std::vector<std::int32_t> tokens;
    std::string error;
};

std::vector<std::byte> EncodeFabricRequest(std::span<const std::int32_t> tokens,
                                           std::uint32_t max_gen);

WireStatus DecodeFabricRequest(std::span<const std::byte> frame,
                               FabricRequestView& out);

std::vector<std::byte> EncodeFabricResponse(const FabricResponse& r);

WireStatus DecodeFabricResponse(std::span<const std::byte> frame,
                                FabricResponse& out);

}  // namespace lsxnetwork
```

- [ ] **Step 4: Run the test to verify it fails at link**

Run:
```bash
cmake --build --preset linux-debug --target test_fabric_wire
```
Expected: FAIL to link — undefined references to the four codec functions.

- [ ] **Step 5: Write the minimal implementation**

Create `libs/lsxnetwork/src/fabric_wire.cpp`:

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

void PutI32(std::byte* p, std::int32_t v) {
    PutU32(p, static_cast<std::uint32_t>(v));
}

void PutF64(std::byte* p, double v) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    for (int i = 0; i < 8; ++i)
        p[i] = static_cast<std::byte>((bits >> (8 * i)) & 0xFFull);
}

std::uint16_t GetU16(const std::byte* p) {
    return static_cast<std::uint16_t>(
        std::to_integer<unsigned>(p[0]) |
        (std::to_integer<unsigned>(p[1]) << 8));
}

std::uint32_t GetU32(const std::byte* p) {
    std::uint32_t v = 0;
    for (int i = 0; i < 4; ++i)
        v |= static_cast<std::uint32_t>(std::to_integer<unsigned>(p[i]))
             << (8 * i);
    return v;
}

std::int32_t GetI32(const std::byte* p) {
    return static_cast<std::int32_t>(GetU32(p));
}

double GetF64(const std::byte* p) {
    std::uint64_t bits = 0;
    for (int i = 0; i < 8; ++i)
        bits |= static_cast<std::uint64_t>(std::to_integer<unsigned>(p[i]))
                << (8 * i);
    double v = 0.0;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

}  // namespace

std::vector<std::byte> EncodeFabricRequest(
    std::span<const std::int32_t> tokens, std::uint32_t max_gen) {
    std::vector<std::byte> out(kFabricRequestHeaderBytes + tokens.size() * 4u);
    std::byte* p = out.data();
    PutU32(p + 0, kFabricMagic);
    PutU16(p + 4, kFabricVersion);
    PutU16(p + 6, 0);
    PutU32(p + 8, max_gen);
    PutU32(p + 12, static_cast<std::uint32_t>(tokens.size()));
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

    const std::uint32_t n = GetU32(p + 12);
    const std::size_t avail = frame.size() - kFabricRequestHeaderBytes;
    if (static_cast<std::size_t>(n) > avail / 4u) return WireStatus::kTruncated;
    if (static_cast<std::size_t>(n) * 4u != avail) return WireStatus::kTrailing;

    out.max_gen = GetU32(p + 8);
    // The 16-byte header keeps the token region 4-byte aligned for every backing
    // buffer (std::vector and libzmq messages are >= 8-byte aligned).
    out.tokens = std::span<const std::int32_t>(
        reinterpret_cast<const std::int32_t*>(p + kFabricRequestHeaderBytes), n);
    return WireStatus::kOk;
}

std::vector<std::byte> EncodeFabricResponse(const FabricResponse& r) {
    const bool ok =
        r.status == static_cast<std::int32_t>(FabricStatus::kOk);
    const std::size_t body = ok ? r.tokens.size() * 4u : r.error.size();
    std::vector<std::byte> out(kFabricResponseHeaderBytes + body);
    std::byte* p = out.data();
    PutU32(p + 0, kFabricMagic);
    PutU16(p + 4, kFabricVersion);
    PutU16(p + 6, static_cast<std::uint16_t>(r.status));
    PutU32(p + 8, ok ? static_cast<std::uint32_t>(r.tokens.size()) : 0u);
    PutU32(p + 12, r.eos_terminated ? 1u : 0u);
    PutF64(p + 16, r.ttft_ms);
    PutF64(p + 24, r.total_ms);
    PutF64(p + 32, r.tokens_per_sec);
    if (ok) {
        for (std::size_t i = 0; i < r.tokens.size(); ++i)
            PutI32(p + kFabricResponseHeaderBytes + i * 4u, r.tokens[i]);
    } else if (!r.error.empty()) {
        std::memcpy(p + kFabricResponseHeaderBytes, r.error.data(),
                    r.error.size());
    }
    return out;
}

WireStatus DecodeFabricResponse(std::span<const std::byte> frame,
                                FabricResponse& out) {
    out = FabricResponse{};
    if (frame.size() < kFabricResponseHeaderBytes)
        return WireStatus::kTruncated;
    const std::byte* p = frame.data();
    if (GetU32(p) != kFabricMagic) return WireStatus::kBadMagic;
    if (GetU16(p + 4) != kFabricVersion) return WireStatus::kBadVersion;

    out.status = static_cast<std::int32_t>(GetU16(p + 6));
    out.eos_terminated = (GetU32(p + 12) & 1u) != 0u;
    out.ttft_ms = GetF64(p + 16);
    out.total_ms = GetF64(p + 24);
    out.tokens_per_sec = GetF64(p + 32);

    const std::size_t avail = frame.size() - kFabricResponseHeaderBytes;
    if (out.status == static_cast<std::int32_t>(FabricStatus::kOk)) {
        const std::uint32_t n = GetU32(p + 8);
        if (static_cast<std::size_t>(n) > avail / 4u)
            return WireStatus::kTruncated;
        if (static_cast<std::size_t>(n) * 4u != avail)
            return WireStatus::kTrailing;
        const auto* first = reinterpret_cast<const std::int32_t*>(
            p + kFabricResponseHeaderBytes);
        out.tokens.assign(first, first + n);
    } else {
        out.error.assign(reinterpret_cast<const char*>(p + kFabricResponseHeaderBytes),
                         avail);
    }
    return WireStatus::kOk;
}

}  // namespace lsxnetwork
```

- [ ] **Step 6: Build and run the test to verify it passes**

Run:
```bash
cmake --build --preset linux-debug --target test_fabric_wire
ctest --preset linux-debug -R test_fabric_wire --output-on-failure
```
Expected: PASS (all `fabric_wire` sections green).

- [ ] **Step 7: Commit**

```bash
git add libs/lsxnetwork/include/lsxnetwork/fabric_wire.h \
        libs/lsxnetwork/src/fabric_wire.cpp \
        libs/lsxnetwork/CMakeLists.txt \
        tests/test_fabric_wire.cpp tests/CMakeLists.txt
git commit -m "feat: add lsxfabric binary-token wire codec"
```

---

### Task 2: `FabricClient` (ZMQ REQ)

**Files:**
- Create: `libs/lsxnetwork/include/lsxnetwork/fabric_client.h`
- Create: `libs/lsxnetwork/src/fabric_client.cpp`
- Create: `tests/test_fabric_client.cpp`
- Modify: `libs/lsxnetwork/CMakeLists.txt` (add `src/fabric_client.cpp`)
- Modify: `tests/CMakeLists.txt` (add executable + test; label already added in Task 1)

**Interfaces:**
- Consumes: `fabric_wire` (Task 1) and `lsxnetwork::ZmqContext`/`ZmqSocket` (`zmq_context.h`).
- Produces: `lsxnetwork::FabricClient` with
  - `int Connect(const char* endpoint)` (0 on success)
  - `bool Infer(std::span<const std::int32_t> input_ids, std::uint32_t max_gen, FabricResponse& out, int timeout_ms = 30000)`
  - On transport failure/timeout: returns `false`, sets `out.status = -1`.

- [ ] **Step 1: Write the failing client test**

Create `tests/test_fabric_client.cpp`:

```cpp
// FabricClient round-trip against an in-process fake REP server. Host-only.
#include <catch2/catch_test_macros.hpp>

#include "lsxnetwork/fabric_client.h"
#include "lsxnetwork/fabric_wire.h"
#include "lsxnetwork/zmq_context.h"

#include <zmq.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <string>
#include <thread>
#include <vector>

using namespace lsxnetwork;

namespace {

// Minimal REP server: binds tcp://127.0.0.1:* and replies with a fixed payload.
class FakeRepServer {
public:
    // `mode`: 0 = OK with tokens {7,8,9}; 1 = infer-failed "boom"; 2 = no reply.
    void Start(int mode) {
        std::promise<std::string> ready;
        std::future<std::string> fut = ready.get_future();
        thread_ = std::thread([this, mode, ready = std::move(ready)]() mutable {
            ZmqContext ctx;
            ZmqSocket sock(ctx, ZMQ_REP);
            if (sock.Bind("tcp://127.0.0.1:*") != 0) {
                ready.set_value(std::string{});
                return;
            }
            char ep[256];
            std::size_t len = sizeof(ep);
            sock.GetSockOpt(ZMQ_LAST_ENDPOINT, ep, &len);
            ready.set_value(std::string(ep));

            const int rcvtimeo = 100;
            sock.SetSockOpt(ZMQ_RCVTIMEO, &rcvtimeo, sizeof(rcvtimeo));
            while (!stop_.load(std::memory_order_acquire)) {
                zmq_msg_t msg;
                zmq_msg_init(&msg);
                const int n = zmq_msg_recv(&msg, sock.Get(), 0);
                if (n < 0) {
                    zmq_msg_close(&msg);
                    if (zmq_errno() == EAGAIN) continue;
                    break;
                }
                if (mode == 2) {          // swallow, never reply
                    zmq_msg_close(&msg);
                    continue;
                }
                FabricResponse r;
                if (mode == 0) {
                    r.status = static_cast<std::int32_t>(FabricStatus::kOk);
                    r.eos_terminated = true;
                    r.ttft_ms = 1.0;
                    r.total_ms = 2.0;
                    r.tokens_per_sec = 3.0;
                    r.tokens = {7, 8, 9};
                } else {
                    r.status = static_cast<std::int32_t>(FabricStatus::kInferFailed);
                    r.error = "boom";
                }
                const std::vector<std::byte> out = EncodeFabricResponse(r);
                sock.Send(out.data(), out.size(), 0);
                zmq_msg_close(&msg);
            }
        });
        endpoint_ = fut.get();
    }

    void Stop() {
        stop_.store(true, std::memory_order_release);
        if (thread_.joinable()) thread_.join();
    }

    const std::string& Endpoint() const { return endpoint_; }

private:
    std::thread thread_;
    std::atomic<bool> stop_{false};
    std::string endpoint_;
};

}  // namespace

TEST_CASE("fabric_client: OK round-trip", "[fabric][client]") {
    FakeRepServer server;
    server.Start(0);
    REQUIRE_FALSE(server.Endpoint().empty());

    FabricClient client;
    REQUIRE(client.Connect(server.Endpoint().c_str()) == 0);
    FabricResponse resp;
    const std::vector<std::int32_t> ids{1, 2, 3};
    REQUIRE(client.Infer(ids, 4u, resp, 2000));
    CHECK(resp.status == static_cast<std::int32_t>(FabricStatus::kOk));
    CHECK(resp.eos_terminated);
    CHECK(resp.ttft_ms == 1.0);
    REQUIRE(resp.tokens.size() == 3);
    CHECK(resp.tokens[2] == 9);
    server.Stop();
}

TEST_CASE("fabric_client: infer-failed response surfaces error",
          "[fabric][client]") {
    FakeRepServer server;
    server.Start(1);
    FabricClient client;
    REQUIRE(client.Connect(server.Endpoint().c_str()) == 0);
    FabricResponse resp;
    const std::vector<std::int32_t> ids{1};
    REQUIRE(client.Infer(ids, 1u, resp, 2000));
    CHECK(resp.status == static_cast<std::int32_t>(FabricStatus::kInferFailed));
    CHECK(resp.error == "boom");
    server.Stop();
}

TEST_CASE("fabric_client: timeout reports transport failure",
          "[fabric][client]") {
    FakeRepServer server;
    server.Start(2);
    FabricClient client;
    REQUIRE(client.Connect(server.Endpoint().c_str()) == 0);
    FabricResponse resp;
    const std::vector<std::int32_t> ids{1};
    REQUIRE_FALSE(client.Infer(ids, 1u, resp, 150));
    CHECK(resp.status == -1);
    server.Stop();
}
```

- [ ] **Step 2: Add the test target, run, verify RED**

Append to `tests/CMakeLists.txt` after the `test_fabric_wire` block:

```cmake
add_executable(test_fabric_client test_fabric_client.cpp)
target_link_libraries(test_fabric_client PRIVATE lsxnetwork Catch2::Catch2WithMain)
add_test(NAME test_fabric_client COMMAND test_fabric_client)
```

Run:
```bash
cmake --preset linux-debug
cmake --build --preset linux-debug --target test_fabric_client
```
Expected: FAIL — `fabric_client.h` not found.

- [ ] **Step 3: Write the header**

Create `libs/lsxnetwork/include/lsxnetwork/fabric_client.h`:

```cpp
#pragma once

#include <cstdint>
#include <memory>
#include <span>

#include "lsxnetwork/fabric_wire.h"
#include "lsxnetwork/zmq_context.h"

namespace lsxnetwork {

/// Blocking ZeroMQ REQ client for the lsxfabric binary-token server.
///
/// Strictly one request in flight: REQ must alternate send/recv. Not
/// thread-safe; use one instance per thread.
class FabricClient {
public:
    FabricClient() = default;
    ~FabricClient() = default;
    FabricClient(const FabricClient&) = delete;
    FabricClient& operator=(const FabricClient&) = delete;

    /// Creates the REQ socket and connects. Returns 0 on success, -1 if the
    /// socket could not be created.
    int Connect(const char* endpoint);

    bool Connected() const noexcept { return sock_ != nullptr; }

    /// Sends @p input_ids and waits for the reply. Returns false on transport
    /// error or timeout, with `out.status = -1`. On success returns true and
    /// `out` holds the decoded response (check `out.status`).
    bool Infer(std::span<const std::int32_t> input_ids, std::uint32_t max_gen,
               FabricResponse& out, int timeout_ms = 30000);

private:
    ZmqContext ctx_;
    std::unique_ptr<ZmqSocket> sock_;
};

}  // namespace lsxnetwork
```

- [ ] **Step 4: Write the implementation**

Create `libs/lsxnetwork/src/fabric_client.cpp`:

```cpp
#include "lsxnetwork/fabric_client.h"

#include <zmq.h>

#include <cstring>

namespace lsxnetwork {

int FabricClient::Connect(const char* endpoint) {
    sock_ = std::make_unique<ZmqSocket>(ctx_, ZMQ_REQ);
    if (!sock_->Valid()) {
        sock_.reset();
        return -1;
    }
    if (sock_->Connect(endpoint) != 0) {
        sock_.reset();
        return -1;
    }
    return 0;
}

bool FabricClient::Infer(std::span<const std::int32_t> input_ids,
                         std::uint32_t max_gen, FabricResponse& out,
                         int timeout_ms) {
    out = FabricResponse{};
    if (!sock_ || !sock_->Valid()) {
        out.status = -1;
        return false;
    }
    if (sock_->SetSockOpt(ZMQ_RCVTIMEO, &timeout_ms, sizeof(timeout_ms)) != 0) {
        out.status = -1;
        return false;
    }

    const std::vector<std::byte> frame =
        EncodeFabricRequest(input_ids, max_gen);
    if (sock_->Send(frame.data(), frame.size(), 0) < 0) {
        out.status = -1;
        return false;
    }

    zmq_msg_t msg;
    zmq_msg_init(&msg);
    const int n = zmq_msg_recv(&msg, sock_->Get(), 0);
    if (n < 0) {
        zmq_msg_close(&msg);
        out.status = -1;
        return false;
    }
    const std::span<const std::byte> reply(
        static_cast<const std::byte*>(zmq_msg_data(&msg)),
        static_cast<std::size_t>(n));
    const WireStatus ws = DecodeFabricResponse(reply, out);
    zmq_msg_close(&msg);
    if (ws != WireStatus::kOk) {
        out = FabricResponse{};
        out.status = -1;
        return false;
    }
    return true;
}

}  // namespace lsxnetwork
```

Add `src/fabric_client.cpp` to the `add_library(lsxnetwork STATIC ...)` source list.

- [ ] **Step 5: Build and run the test to verify it passes**

Run:
```bash
cmake --build --preset linux-debug --target test_fabric_client
ctest --preset linux-debug -R test_fabric_client --output-on-failure
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add libs/lsxnetwork/include/lsxnetwork/fabric_client.h \
        libs/lsxnetwork/src/fabric_client.cpp \
        libs/lsxnetwork/CMakeLists.txt \
        tests/test_fabric_client.cpp tests/CMakeLists.txt
git commit -m "feat: add lsxfabric binary-token FabricClient"
```

---

### Task 3: `FabricServer` + `--serve` CLI

**Files:**
- Create: `apps/lsxfabric/src/fabric_server.h`
- Create: `apps/lsxfabric/src/fabric_server.cpp`
- Create: `apps/lsxfabric/tests/test_fabric_server.cpp`
- Modify: `apps/lsxfabric/src/cli.h`, `apps/lsxfabric/src/cli.cpp`
- Modify: `apps/lsxfabric/src/main.cpp`
- Modify: `apps/lsxfabric/CMakeLists.txt` (add source, link `lsxnetwork`)
- Modify: `apps/lsxfabric/tests/CMakeLists.txt` (add server test + label)
- Modify: `apps/lsxfabric/tests/test_fabric_cli_edges.cpp` (add `--serve`/`--listen` cases)

**Interfaces:**
- Consumes: `lsxnetwork::FabricClient`/`FabricResponse` (Task 2), `lsxcommon::IModel`, `lsxcommon::LsxModelFactory`.
- Produces:
  - `lsxfabric::FabricServer(lsxcommon::IModel& model, std::uint32_t default_max_gen)`
  - `int Run(const std::string& endpoint, std::atomic<bool>& stop, std::atomic<bool>* ready = nullptr, std::string* resolved_endpoint = nullptr)`
  - `FabricCliArgs` gains `bool serve` and `std::string listen = "tcp://*:5555"`.

- [ ] **Step 1: Extend the CLI test (RED)**

Read `apps/lsxfabric/tests/test_fabric_cli_edges.cpp` and append these cases (adjust `Parse` helper name to match the file's existing helper):

```cpp
TEST_CASE("fabric cli: --serve and --listen parse", "[lsxfabric][cli]") {
    const lsxfabric::FabricCliArgs a =
        Parse({"lsxfabric", "--serve", "--listen", "tcp://127.0.0.1:6000"});
    REQUIRE(a.serve);
    REQUIRE(a.listen == "tcp://127.0.0.1:6000");
    REQUIRE_FALSE(a.validate);
    REQUIRE_FALSE(a.testprompt);
}

TEST_CASE("fabric cli: --serve default endpoint", "[lsxfabric][cli]") {
    const lsxfabric::FabricCliArgs a = Parse({"lsxfabric", "--serve"});
    REQUIRE(a.serve);
    REQUIRE(a.listen == "tcp://*:5555");
}
```

Run:
```bash
cmake --preset linux-debug
cmake --build --preset linux-debug --target test_fabric_cli
ctest --preset linux-debug -R test_fabric_cli --output-on-failure
```
Expected: FAIL — `FabricCliArgs` has no `serve`/`listen`.

- [ ] **Step 2: Extend the CLI**

In `apps/lsxfabric/src/cli.h`, add to `struct FabricCliArgs` (after `bool testprompt = false;`):

```cpp
    bool serve = false;
    std::string listen = "tcp://*:5555";
```

In `apps/lsxfabric/src/cli.cpp`, add `--serve`/`--listen` to `IsKnownFlag`:

```cpp
           std::strcmp(s, "--serve") == 0 ||
           std::strcmp(s, "--listen") == 0 ||
```
and parse them in `ParseFabricCliArgs`:

```cpp
        } else if (std::strcmp(argv[i], "--serve") == 0) {
            a.serve = true;
        } else if (std::strcmp(argv[i], "--listen") == 0) {
            if (has_value) a.listen = argv[++i];
        }
```

Run the CLI test again — expected PASS.

- [ ] **Step 3: Write the failing server test**

Create `apps/lsxfabric/tests/test_fabric_server.cpp`:

```cpp
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
        while (!ready.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
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
    CHECK(fx.model.last_max_gen == 4);   // 0 => server default
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
```

- [ ] **Step 4: Add the server test target, verify RED**

In `apps/lsxfabric/tests/CMakeLists.txt`, append:

```cmake
add_executable(test_fabric_server test_fabric_server.cpp ../src/fabric_server.cpp)
target_include_directories(test_fabric_server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(test_fabric_server PRIVATE lsxcommon lsxnetwork arrow_shared arrow_cuda_shared Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_fabric_server POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_fabric_server>"
    )
    add_custom_command(TARGET test_fabric_server POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_fabric_server>"
    )
endif()

add_test(NAME test_fabric_server COMMAND test_fabric_server)
set_tests_properties(test_fabric_server PROPERTIES LABELS fast)
set_tests_properties(test_fabric_server PROPERTIES TIMEOUT 120)
```

Run:
```bash
cmake --preset linux-debug
cmake --build --preset linux-debug --target test_fabric_server
```
Expected: FAIL — `fabric_server.h` not found.

- [ ] **Step 5: Write the server header**

Create `apps/lsxfabric/src/fabric_server.h`:

```cpp
#pragma once

#include <atomic>
#include <cstdint>
#include <string>

namespace lsxcommon {
class IModel;
}

namespace lsxfabric {

/// Single-threaded ZeroMQ REP server for the lsxfabric binary-token protocol.
///
/// One request in flight: the server blocks on recv, runs the injected model,
/// and sends one binary response frame. It owns no threads; Run() is expected
/// to be called on a dedicated thread with a stop flag.
class FabricServer {
public:
    FabricServer(lsxcommon::IModel& model, std::uint32_t default_max_gen);

    /// Binds @p endpoint and serves until @p stop becomes true.
    ///
    /// @param ready set true after a successful bind (for test synchronization).
    /// @param resolved_endpoint when non-null, receives the concrete bound
    ///        endpoint (resolves a "tcp://127.0.0.1:*" wildcard).
    /// @return 0 on clean stop, non-zero when the socket could not bind.
    int Run(const std::string& endpoint, std::atomic<bool>& stop,
            std::atomic<bool>* ready = nullptr,
            std::string* resolved_endpoint = nullptr);

private:
    lsxcommon::IModel& model_;
    std::uint32_t default_max_gen_;
    std::atomic<bool> abort_{false};
};

}  // namespace lsxfabric
```

- [ ] **Step 6: Write the server implementation**

Create `apps/lsxfabric/src/fabric_server.cpp`:

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
#include <vector>

namespace lsxfabric {
namespace {

using lsxnetwork::FabricRequestView;
using lsxnetwork::FabricResponse;
using lsxnetwork::FabricStatus;
using lsxnetwork::WireStatus;

std::vector<std::byte> BadRequest(std::string msg) {
    FabricResponse r;
    r.status = static_cast<std::int32_t>(FabricStatus::kBadRequest);
    r.error = std::move(msg);
    return lsxnetwork::EncodeFabricResponse(r);
}

std::vector<std::byte> HandleFrame(lsxcommon::IModel& model,
                                   std::uint32_t default_max_gen,
                                   std::atomic<bool>& abort,
                                   std::span<const std::byte> frame) {
    FabricRequestView req;
    if (lsxnetwork::DecodeFabricRequest(frame, req) != WireStatus::kOk)
        return BadRequest("malformed request");

    lsxcommon::ModelRequest mreq;
    mreq.input_ids.assign(req.tokens.begin(), req.tokens.end());
    mreq.max_gen_tokens =
        req.max_gen != 0u ? static_cast<int>(req.max_gen)
                          : static_cast<int>(default_max_gen);
    mreq.abort_flag = &abort;

    lsxcommon::gpu::InferenceResult res;
    try {
        std::shared_lock<std::shared_mutex> lock(model.ResidencyMutex());
        res = model.Infer(mreq);
    } catch (const std::exception& e) {
        FabricResponse r;
        r.status = static_cast<std::int32_t>(FabricStatus::kInternal);
        r.error = e.what();
        return lsxnetwork::EncodeFabricResponse(r);
    }

    FabricResponse out;
    if (!res.ok) {
        out.status = static_cast<std::int32_t>(FabricStatus::kInferFailed);
        out.error = res.error_msg;
    } else {
        out.status = static_cast<std::int32_t>(FabricStatus::kOk);
        out.tokens = res.gen_tokens;
        out.eos_terminated = res.eos_terminated;
        out.ttft_ms = res.ttft_ms;
        out.total_ms = res.total_time_ms;
        out.tokens_per_sec = res.tokens_per_sec;
    }
    return lsxnetwork::EncodeFabricResponse(out);
}

}  // namespace

FabricServer::FabricServer(lsxcommon::IModel& model,
                           std::uint32_t default_max_gen)
    : model_(model), default_max_gen_(default_max_gen) {}

int FabricServer::Run(const std::string& endpoint, std::atomic<bool>& stop,
                      std::atomic<bool>* ready,
                      std::string* resolved_endpoint) {
    lsxnetwork::ZmqContext ctx;
    lsxnetwork::ZmqSocket sock(ctx, ZMQ_REP);
    if (!sock.Valid() || sock.Bind(endpoint.c_str()) != 0) {
        lsxcommon::log::error("[fabric] cannot bind endpoint: " + endpoint);
        return 1;
    }
    if (resolved_endpoint != nullptr) {
        char ep[256];
        std::size_t len = sizeof(ep);
        if (sock.GetSockOpt(ZMQ_LAST_ENDPOINT, ep, &len) == 0)
            *resolved_endpoint = std::string(ep);
    }
    if (ready != nullptr) ready->store(true);

    const int rcvtimeo = 200;
    sock.SetSockOpt(ZMQ_RCVTIMEO, &rcvtimeo, sizeof(rcvtimeo));

    lsxcommon::log::info("[fabric] serving on " + endpoint);
    while (!stop.load(std::memory_order_acquire)) {
        zmq_msg_t msg;
        zmq_msg_init(&msg);
        const int n = zmq_msg_recv(&msg, sock.Get(), 0);
        if (n < 0) {
            const int err = zmq_errno();
            zmq_msg_close(&msg);
            if (err == EAGAIN) continue;  // recv timeout: re-check stop
            break;                        // ETERM or fatal
        }
        const std::span<const std::byte> frame(
            static_cast<const std::byte*>(zmq_msg_data(&msg)),
            static_cast<std::size_t>(n));
        const std::vector<std::byte> out =
            HandleFrame(model_, default_max_gen_, abort_, frame);
        zmq_msg_close(&msg);
        sock.Send(out.data(), out.size(), 0);
    }
    return 0;
}

}  // namespace lsxfabric
```

Add `src/fabric_server.cpp` to the `add_executable(lsxfabric ...)` source list and `lsxnetwork` to its `target_link_libraries` in `apps/lsxfabric/CMakeLists.txt`:

```cmake
add_executable(lsxfabric src/main.cpp src/cli.cpp src/validator.cpp src/fabric_server.cpp)
target_link_libraries(lsxfabric PRIVATE lsxcommon lsxnetwork arrow_shared arrow_cuda_shared simdjson::simdjson ${CMAKE_DL_LIBS})
```

- [ ] **Step 7: Wire `--serve` into `main.cpp`**

Replace the body of `apps/lsxfabric/src/main.cpp` with:

```cpp
#include "lsxcommon/core.h"
#include <absl/strings/str_format.h>
#include "lsxcommon/common.h"
#include "lsxcommon/log.h"
#include "lsxcommon/model_engine.h"
#include "lsxcommon/model_factory.h"
#include "lsxcommon/model_runtime.h"
#include "cli.h"
#include "fabric_server.h"
#include "validator.h"

#include <atomic>
#include <csignal>
#include <memory>

namespace {

std::atomic<bool>* g_serve_stop = nullptr;

void OnServeSignal(int) {
    if (g_serve_stop != nullptr) g_serve_stop->store(true);
}

std::unique_ptr<lsxcommon::IModel> LoadServeModel(const std::string& path) {
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = path;
    cfg.fast_math = false;
    cfg.fast_math_mode = lsxcommon::FastMathMode::kAuto;
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    if (!model || !model->Load()) return nullptr;
    return model;
}

}  // namespace

int main(int argc, char* argv[]) {
    const lsxfabric::FabricCliArgs cli =
        lsxfabric::ParseFabricCliArgs(argc, argv);

    lsxcommon::Core::Initialize("logestix_runtime.log");
    lsxcommon::version();
    lsxcommon::log_gpu_info();

    int rc = 0;
    {
        lsxcommon::InferenceEngine::Session engine_session;
        lsxcommon::log::info(absl::StrFormat("Model path: %s", cli.model_path));
        if (cli.serve) {
            auto model = LoadServeModel(cli.model_path);
            if (!model) {
                lsxcommon::log::error("Failed to load model for --serve");
                rc = 1;
            } else {
                std::atomic<bool> stop{false};
                g_serve_stop = &stop;
                std::signal(SIGINT, OnServeSignal);
                std::signal(SIGTERM, OnServeSignal);
                lsxfabric::FabricServer server(
                    *model, static_cast<std::uint32_t>(cli.max_gen));
                rc = server.Run(cli.listen, stop);
                g_serve_stop = nullptr;
                model->Unload();
            }
        } else if (cli.testprompt) {
            lsxcommon::runtime::GenerationCliArgs args;
            args.model_path = cli.model_path;
            args.prompt = cli.prompt;
            args.max_gen_tokens = cli.max_gen;
            args.fast_math_mode = lsxcommon::FastMathMode::kAuto;
            rc = lsxcommon::runtime::RunCliGeneration(args);
        } else if (cli.validate) {
            lsxcommon::log::info("Running validation...");
            rc = lsxfabric::RunModelValidation(cli.model_path) ? 0 : 1;
        } else {
            lsxcommon::log::error(
                "No run mode selected: pass --serve, --validate or --testprompt");
            rc = 1;
        }
    }
    lsxcommon::Core::Shutdown();
    return rc;
}
```

- [ ] **Step 8: Build and run the server test to verify it passes**

Run:
```bash
cmake --preset linux-debug
cmake --build --preset linux-debug --target test_fabric_server test_fabric_cli
ctest --preset linux-debug -R "test_fabric_server|test_fabric_cli" --output-on-failure
```
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add apps/lsxfabric/src/fabric_server.h apps/lsxfabric/src/fabric_server.cpp \
        apps/lsxfabric/src/cli.h apps/lsxfabric/src/cli.cpp apps/lsxfabric/src/main.cpp \
        apps/lsxfabric/CMakeLists.txt apps/lsxfabric/tests/CMakeLists.txt \
        apps/lsxfabric/tests/test_fabric_server.cpp \
        apps/lsxfabric/tests/test_fabric_cli_edges.cpp
git commit -m "feat: add lsxfabric --serve binary-token server"
```

---

### Task 4: Host-only tokenizer/framing bridge (`lsxcommon`)

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/tokenizer_arrows.h`
- Create: `libs/lsxcommon/src/tokenizer_arrows.cpp`
- Create: `libs/lsxcommon/include/lsxcommon/prompt_bridge.h`
- Create: `libs/lsxcommon/src/prompt_bridge.cpp`
- Modify: `libs/lsxcommon/src/model_base.cpp` (delegate `LoadTokenizerDb`)
- Modify: `libs/lsxcommon/CMakeLists.txt` (add the two .cpp files)

**Interfaces:**
- Produces:
  - `bool lsxcommon::LoadTokenizerArrows(const std::string& model_path, ArrowTokenizerDb& out)`
  - `bool lsxcommon::FramePromptForModel(const std::string& model_path, std::string_view raw_prompt, std::vector<std::int32_t>& out_ids)`

- [ ] **Step 1: Write the tokenizer-arrows header + implementation**

Create `libs/lsxcommon/include/lsxcommon/tokenizer_arrows.h`:

```cpp
#pragma once

#include <string>

#include "lsxcommon/tokenizer.h"

namespace lsxcommon {

/// Host-only: mmap and load a converted model's tokenizer arrows
/// (`<model_path>/logestix/pinned/tokenizer_{vocab,merges}.arrow`, with the
/// MoE alternate roots). No GPU, no weights. Returns false with `out.error_msg`
/// set on missing files or format error.
bool LoadTokenizerArrows(const std::string& model_path, ArrowTokenizerDb& out);

}  // namespace lsxcommon
```

Create `libs/lsxcommon/src/tokenizer_arrows.cpp`:

```cpp
#include "lsxcommon/tokenizer_arrows.h"

#include <arrow/io/file.h>

#include <filesystem>

namespace lsxcommon {
namespace {

std::filesystem::path ResolvePinnedDir(const std::string& model_path) {
    auto pinned = std::filesystem::path(model_path) / "logestix" / "pinned";
    if (!std::filesystem::exists(pinned)) {
        for (const char* alt : {"logestix-qwen35moe", "logestix-glm",
                                "logestix-kimi", "logestix-gemma"}) {
            auto cand = std::filesystem::path(model_path) / alt / "pinned";
            if (std::filesystem::exists(cand)) return cand;
        }
    }
    return pinned;
}

}  // namespace

bool LoadTokenizerArrows(const std::string& model_path, ArrowTokenizerDb& out) {
    const std::filesystem::path pinned = ResolvePinnedDir(model_path);
    auto vocab = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_vocab.arrow").string(), arrow::io::FileMode::READ);
    auto merges = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_merges.arrow").string(), arrow::io::FileMode::READ);
    if (!vocab.ok() || !merges.ok()) {
        out.error_msg = "tokenizer arrows not found under " + pinned.string();
        return false;
    }
    auto st = load_tokenizer_metadata(*vocab, *merges, out);
    if (!st.ok()) {
        out.error_msg = st.ToString();
        return false;
    }
    return true;
}

}  // namespace lsxcommon
```

- [ ] **Step 2: Write the framing bridge**

Create `libs/lsxcommon/include/lsxcommon/prompt_bridge.h`:

```cpp
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace lsxcommon {

/// Host-only: frame a raw prompt into model input ids using the model family's
/// own chat template (`IModel::BuildPromptInputIds`). The model instance is
/// created through the factory but NOT Load()ed: no weights are read and no GPU
/// work happens. Framing ownership stays with each model (no central registry).
bool FramePromptForModel(const std::string& model_path,
                         std::string_view raw_prompt,
                         std::vector<std::int32_t>& out_ids);

}  // namespace lsxcommon
```

Create `libs/lsxcommon/src/prompt_bridge.cpp`:

```cpp
#include "lsxcommon/prompt_bridge.h"

#include "lsxcommon/model_factory.h"
#include "lsxcommon/tokenizer_arrows.h"

namespace lsxcommon {

bool FramePromptForModel(const std::string& model_path,
                         std::string_view raw_prompt,
                         std::vector<std::int32_t>& out_ids) {
    ModelInitConfig cfg;
    cfg.model_path = model_path;
    auto model = LsxModelFactory::Create(cfg);
    if (!model) return false;

    ArrowTokenizerDb tok;
    if (!LoadTokenizerArrows(model_path, tok)) return false;

    return model->BuildPromptInputIds(tok, raw_prompt, out_ids);
}

}  // namespace lsxcommon
```

- [ ] **Step 3: Refactor `ModelBase::LoadTokenizerDb` to use the helper**

In `libs/lsxcommon/src/model_base.cpp`, replace the body of `ModelBase::LoadTokenizerDb` from the `auto pinned = ...` block through the end of the function with:

```cpp
    tok_db_ = std::make_unique<ArrowTokenizerDb>();
    if (!LoadTokenizerArrows(model_path, *tok_db_)) {
        if (tokenizer_optional_) {
            log::warn("[model] tokenizer arrows unavailable (OK for synthetic test)");
            return true;
        }
        log::error(
            absl::StrFormat("[model] cannot load tokenizer arrows for %s",
                            model_path));
        tok_db_.reset();
        return false;
    }
    return true;
```

Add `#include "lsxcommon/tokenizer_arrows.h"` to the includes of `model_base.cpp`, and remove the now-unused `<arrow/io/file.h>` only if it is not used elsewhere in the file (grep first; leave it if used).

- [ ] **Step 4: Add the sources to CMake and build**

In `libs/lsxcommon/CMakeLists.txt`, add `src/tokenizer_arrows.cpp` and `src/prompt_bridge.cpp` to the `add_library(lsxcommon STATIC ...)` source list (next to `src/tokenizer.cpp` / `src/model_factory.cpp`).

Run:
```bash
cmake --preset linux-debug
cmake --build --preset linux-debug --target lsxcommon
```
Expected: builds cleanly.

- [ ] **Step 5: Regression-check the tokenizer/model tests**

Run:
```bash
cmake --build --preset linux-debug --target test_tokenizer test_model_factory
ctest --preset linux-debug -R "test_tokenizer|test_model_factory" --output-on-failure
```
Expected: PASS (the refactor preserves behavior).

- [ ] **Step 6: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/tokenizer_arrows.h \
        libs/lsxcommon/src/tokenizer_arrows.cpp \
        libs/lsxcommon/include/lsxcommon/prompt_bridge.h \
        libs/lsxcommon/src/prompt_bridge.cpp \
        libs/lsxcommon/src/model_base.cpp libs/lsxcommon/CMakeLists.txt
git commit -m "feat: add host-only tokenizer and prompt-framing bridge"
```

---

### Task 5: taskmosaic client-only MVP

**Files:**
- Create: `apps/taskmosaic/src/cli.h`
- Create: `apps/taskmosaic/src/cli.cpp`
- Create: `apps/taskmosaic/tests/test_taskmosaic_cli.cpp`
- Modify: `apps/taskmosaic/src/main.cpp`
- Modify: `apps/taskmosaic/CMakeLists.txt` (add cli.cpp, link lsxnetwork)
- Modify: `apps/taskmosaic/tests/CMakeLists.txt` (add CLI test + label)

**Interfaces:**
- Consumes: `lsxcommon::LoadTokenizerArrows`, `lsxcommon::FramePromptForModel` (Task 4), `lsxcommon::runtime::DecodeGeneratedText`, `lsxnetwork::FabricClient` (Task 2).
- Produces: `apps/taskmosaic/src/cli.h` with
  `struct TaskMosaicCliArgs { std::string server; std::string model; std::string prompt; int max_gen = 64; }`
  `TaskMosaicCliArgs ParseTaskMosaicCliArgs(int argc, char** argv)`

- [ ] **Step 1: Write the failing CLI parse test**

Create `apps/taskmosaic/tests/test_taskmosaic_cli.cpp`:

```cpp
#include <catch2/catch_test_macros.hpp>

#include "cli.h"

#include <string>
#include <vector>

namespace {

std::vector<char*> MakeArgv(const std::vector<std::string>& storage) {
    std::vector<char*> argv;
    argv.reserve(storage.size());
    for (const std::string& s : storage)
        argv.push_back(const_cast<char*>(s.c_str()));
    return argv;
}

taskmosaic::TaskMosaicCliArgs Parse(const std::vector<std::string>& args) {
    std::vector<char*> argv = MakeArgv(args);
    return taskmosaic::ParseTaskMosaicCliArgs(static_cast<int>(argv.size()),
                                              argv.data());
}

}  // namespace

TEST_CASE("taskmosaic cli: parses server/model/prompt/max-gen",
          "[taskmosaic][cli]") {
    const taskmosaic::TaskMosaicCliArgs a =
        Parse({"taskmosaic", "--server", "tcp://127.0.0.1:5555", "--model",
               "/models/x", "--prompt", "hello", "--max-gen", "9"});
    REQUIRE(a.server == "tcp://127.0.0.1:5555");
    REQUIRE(a.model == "/models/x");
    REQUIRE(a.prompt == "hello");
    REQUIRE(a.max_gen == 9);
}

TEST_CASE("taskmosaic cli: prompt defaults empty (stdin)",
          "[taskmosaic][cli]") {
    const taskmosaic::TaskMosaicCliArgs a =
        Parse({"taskmosaic", "--server", "tcp://x:1", "--model", "/m"});
    REQUIRE(a.prompt.empty());
    REQUIRE(a.max_gen == 64);
}

TEST_CASE("taskmosaic cli: missing required args stay empty",
          "[taskmosaic][cli]") {
    const taskmosaic::TaskMosaicCliArgs a = Parse({"taskmosaic"});
    REQUIRE(a.server.empty());
    REQUIRE(a.model.empty());
}
```

- [ ] **Step 2: Add the test target, verify RED**

In `apps/taskmosaic/tests/CMakeLists.txt`, append:

```cmake
add_executable(test_taskmosaic_cli test_taskmosaic_cli.cpp ../src/cli.cpp)
target_include_directories(test_taskmosaic_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(test_taskmosaic_cli PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_taskmosaic_cli POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_taskmosaic_cli>"
    )
    add_custom_command(TARGET test_taskmosaic_cli POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_taskmosaic_cli>"
    )
endif()

add_test(NAME test_taskmosaic_cli COMMAND test_taskmosaic_cli)
set_tests_properties(test_taskmosaic_cli PROPERTIES LABELS fast)
```

Run:
```bash
cmake --preset linux-debug
cmake --build --preset linux-debug --target test_taskmosaic_cli
```
Expected: FAIL — `cli.h` not found.

- [ ] **Step 3: Write the CLI**

Create `apps/taskmosaic/src/cli.h`:

```cpp
#pragma once

#include <string>

namespace taskmosaic {

/// taskmosaic is a client-only CLI: it talks to a running lsxfabric server.
struct TaskMosaicCliArgs {
    std::string server;  // ZMQ endpoint, required
    std::string model;   // converted model path (tokenizer + framing), required
    std::string prompt;  // empty => read stdin
    int max_gen = 64;    // 0 => server default is NOT used here; sent verbatim
};

TaskMosaicCliArgs ParseTaskMosaicCliArgs(int argc, char** argv);

}  // namespace taskmosaic
```

Create `apps/taskmosaic/src/cli.cpp`:

```cpp
#include "cli.h"

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>

namespace taskmosaic {
namespace {

bool HasValue(int argc, char** argv, int i) {
    return (i + 1 < argc) && (argv[i + 1][0] != '-');
}

bool ParsePositiveInt(const char* s, int& out) {
    errno = 0;
    char* end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v < 1 || v > INT_MAX)
        return false;
    out = static_cast<int>(v);
    return true;
}

}  // namespace

TaskMosaicCliArgs ParseTaskMosaicCliArgs(int argc, char** argv) {
    TaskMosaicCliArgs a;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--server") == 0) {
            if (HasValue(argc, argv, i)) a.server = argv[++i];
        } else if (std::strcmp(argv[i], "--model") == 0) {
            if (HasValue(argc, argv, i)) a.model = argv[++i];
        } else if (std::strcmp(argv[i], "--prompt") == 0) {
            if (HasValue(argc, argv, i)) a.prompt = argv[++i];
        } else if (std::strcmp(argv[i], "--max-gen") == 0) {
            int v = 0;
            if (HasValue(argc, argv, i) && ParsePositiveInt(argv[i + 1], v)) {
                a.max_gen = v;
                ++i;
            }
        }
    }
    return a;
}

}  // namespace taskmosaic
```

- [ ] **Step 4: Run the CLI test to verify it passes**

Run:
```bash
cmake --build --preset linux-debug --target test_taskmosaic_cli
ctest --preset linux-debug -R test_taskmosaic_cli --output-on-failure
```
Expected: PASS.

- [ ] **Step 5: Write the client-only `main.cpp`**

Replace `apps/taskmosaic/src/main.cpp` with:

```cpp
#include "lsxcommon/common.h"
#include "lsxcommon/core.h"
#include "lsxcommon/log.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/prompt_bridge.h"
#include "lsxcommon/tokenizer.h"
#include "lsxcommon/tokenizer_arrows.h"
#include "lsxnetwork/fabric_client.h"

#include <absl/strings/str_format.h>

#include <cstdint>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "cli.h"

namespace {

std::string ReadAllStdin() {
    return std::string(std::istreambuf_iterator<char>(std::cin),
                       std::istreambuf_iterator<char>());
}

}  // namespace

int main(int argc, char* argv[]) {
    const taskmosaic::TaskMosaicCliArgs cli =
        taskmosaic::ParseTaskMosaicCliArgs(argc, argv);

    lsxcommon::Core::Initialize("taskmosaic_runtime.log");
    lsxcommon::version();

    int rc = 0;
    {
        if (cli.server.empty() || cli.model.empty()) {
            lsxcommon::log::error(
                "taskmosaic is a client: pass --server <endpoint> and "
                "--model <converted-model-path> [--prompt <text>|stdin]");
            rc = 2;
        } else {
            const std::string prompt =
                cli.prompt.empty() ? ReadAllStdin() : cli.prompt;

            lsxcommon::ArrowTokenizerDb tok;
            if (!lsxcommon::LoadTokenizerArrows(cli.model, tok)) {
                lsxcommon::log::error("cannot load tokenizer for " + cli.model);
                rc = 1;
            } else {
                std::vector<std::int32_t> ids;
                if (!lsxcommon::FramePromptForModel(cli.model, prompt, ids)) {
                    lsxcommon::log::error("prompt framing failed");
                    rc = 1;
                } else {
                    lsxnetwork::FabricClient client;
                    if (client.Connect(cli.server.c_str()) != 0) {
                        lsxcommon::log::error("cannot connect to " + cli.server);
                        rc = 1;
                    } else {
                        lsxnetwork::FabricResponse resp;
                        if (!client.Infer(ids, static_cast<std::uint32_t>(cli.max_gen),
                                          resp)) {
                            lsxcommon::log::error(
                                absl::StrFormat("inference transport failed on %s",
                                                cli.server));
                            rc = 1;
                        } else if (resp.status !=
                                   static_cast<std::int32_t>(
                                       lsxnetwork::FabricStatus::kOk)) {
                            lsxcommon::log::error(absl::StrFormat(
                                "server error (status %d): %s", resp.status,
                                resp.error));
                            rc = 1;
                        } else {
                            const std::string text =
                                lsxcommon::runtime::DecodeGeneratedText(tok,
                                                                        resp.tokens);
                            lsxcommon::log::info(text);
                            lsxcommon::log::info(absl::StrFormat(
                                "tokens=%u ttft=%.1fms total=%.1fms %.1f t/s",
                                static_cast<unsigned>(resp.tokens.size()),
                                resp.ttft_ms, resp.total_ms,
                                resp.tokens_per_sec));
                        }
                    }
                }
            }
        }
    }
    lsxcommon::Core::Shutdown();
    return rc;
}
```

Note: the include block above already provides `<iostream>` for `std::cin`; do not add `<istream>`.

- [ ] **Step 6: Update the taskmosaic CMake and build**

In `apps/taskmosaic/CMakeLists.txt`, change:

```cmake
add_executable(taskmosaic src/main.cpp src/cli.cpp)
target_link_libraries(taskmosaic PRIVATE lsxcommon lsxnetwork arrow_shared arrow_cuda_shared ${CMAKE_DL_LIBS})
```

Run:
```bash
cmake --preset linux-debug
cmake --build --preset linux-debug --target taskmosaic test_taskmosaic_cli
ctest --preset linux-debug -R "test_taskmosaic" --output-on-failure
```
Expected: builds; taskmosaic tests PASS.

- [ ] **Step 7: Commit**

```bash
git add apps/taskmosaic/src/cli.h apps/taskmosaic/src/cli.cpp \
        apps/taskmosaic/src/main.cpp apps/taskmosaic/CMakeLists.txt \
        apps/taskmosaic/tests/CMakeLists.txt apps/taskmosaic/tests/test_taskmosaic_cli.cpp
git commit -m "feat: make taskmosaic a binary-token client of lsxfabric"
```

---

### Task 6: Full verification and guard sweep

**Files:** none created; this task only builds, runs, and (if needed) fixes.

- [ ] **Step 1: Configure and build the whole tree**

Run:
```bash
cmake --preset linux-debug
cmake --build --preset linux-debug
```
Expected: clean build (no warnings promoted to errors).

- [ ] **Step 2: Run the fast suite**

Run:
```bash
ctest --preset linux-debug --output-on-failure
```
Expected: all `fast` tests PASS, including:
- `test_fabric_wire`
- `test_fabric_client`
- `test_fabric_server`
- `test_fabric_cli`
- `test_taskmosaic`, `test_taskmosaic_cli`
- `test_ctest_labels_guard` (every new test labeled exactly once)
- `test_code_naming_guard`, `test_model_suite_guard`, `test_model_independence_guard`

- [ ] **Step 3: Fix any guard violations inline**

Likely failure modes and fixes:
- `unlabeled test: test_fabric_server` → ensure it is in the `LABELS fast` call in `apps/lsxfabric/tests/CMakeLists.txt`.
- `unlabeled test: test_taskmosaic_cli` / `test_fabric_wire` / `test_fabric_client` → ensure entries in the respective CMakeLists label calls.
- duplicate label → remove the test from any second label call.
- `tests/ naming violation` → ensure new test files match `test_*.cpp`.

Re-run Step 2 until green.

- [ ] **Step 4: Commit the final state (only if Step 3 changed anything)**

```bash
git add -A
git commit -m "build: register fabric server/client test labels"
```

---

### Task 7: Update the spec/notes with as-built deltas

**Files:**
- Modify: `docs/superpowers/specs/2026-09-25-lsxfabric-binary-server-design.md`

- [ ] **Step 1: Record the as-built API deltas**

Append a short "As-built deltas" section to the spec noting:
- `FabricServer::Run` additionally takes optional `ready` / `resolved_endpoint`
  out-params for tests (wildcard bind resolution).
- The host-only bridge files are `tokenizer_arrows.{h,cpp}` and
  `prompt_bridge.{h,cpp}` under `libs/lsxcommon`.

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-09-25-lsxfabric-binary-server-design.md
git commit -m "docs: record as-built fabric server deltas"
```

---

## Self-Review

**1. Spec coverage**
- Transport ZMQ REP/REQ → Tasks 2, 3. ✔
- One request in flight → server loop + REQ client. ✔
- Whole-stream response → single frame codec. ✔
- Ready `input_ids`, no framing on server → `FabricRequestView.tokens` → `ModelRequest.input_ids`. ✔
- Per-request `max_gen`, 0 = CLI default → Task 3 test + `HandleFrame`. ✔
- `--serve` / `--listen` (default `tcp://*:5555`) → Task 3. ✔
- Tokens + status + eos + timings → codec response, Task 1/2/3. ✔
- taskmosaic client-only, tokenizer/framing host-only → Tasks 4, 5. ✔
- Codec/client in `lsxnetwork`; server app-local → Tasks 1–3. ✔
- Error handling (bad request/malformed, infer failure, exception, transport) → codec + server/client. ✔
- TDD + labels (`fast`) + branch + commit prefixes → every task. ✔
- Out-of-scope items (streaming, ROUTER, tokenize endpoints, TLS) → not implemented. ✔

**2. Placeholder scan** — no TBD/TODO; all steps carry complete code or exact edits.

**3. Type consistency**
- `FabricRequestView.tokens` is `std::span<const std::int32_t>` throughout (server assigns to `ModelRequest.input_ids`).
- `FabricResponse.status` is `std::int32_t`; encode takes `std::uint16_t` cast, decode widens.
- `FabricClient::Infer(span, uint32_t, FabricResponse&, int)` matches all call sites (tests, taskmosaic, server tests).
- `FabricServer::Run(const std::string&, std::atomic<bool>&, std::atomic<bool>*, std::string*)` matches server-test call site.
- `LoadTokenizerArrows` / `FramePromptForModel` signatures match taskmosaic usage.
- Header size constants `kFabricRequestHeaderBytes=16` / `kFabricResponseHeaderBytes=40` match every offset used in the codec.
