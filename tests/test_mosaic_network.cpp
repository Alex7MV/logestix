#include <catch2/catch_test_macros.hpp>

#include <lsxnetwork/affinity.h>
#include <lsxnetwork/frame_queue.h>
#include <lsxnetwork/openai_endpoint.h>
#include <lsxnetwork/zmq_context.h>

#include <simdjson.h>

#include <zmq.h>

#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string_view>
#include <thread>

using namespace std::string_view_literals;

namespace {

constexpr std::string_view kOpenAIPayload =
    R"({
  "id": "chatcmpl-7QyqpwdfhqwajicIEznoc6Q47XAy0D",
  "object": "chat.completion",
  "created": 1687886491,
  "model": "gpt-45",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 11,
    "completion_tokens": 8,
    "total_tokens": 19
  }
})"sv;

// True when the host permits the calling thread to be pinned to cores 0..3
// (process/container affinity may be narrower in CI or sandboxes).
bool AffinityToCores0To3Feasible() {
    const std::uint32_t before = lsxnetwork::CpuAffinity::CurrentAllowedCores();
    return (before & lsxnetwork::CpuAffinity::kAllowedMask) ==
           lsxnetwork::CpuAffinity::kAllowedMask;
}

}  // namespace

TEST_CASE("lsxnetwork: capture -> affinity-lock -> simdjson parse -> queue",
          "[mosaic][network]")
{
    using lsxnetwork::CpuAffinity;
    using lsxnetwork::ChatCompletion;
    using lsxnetwork::FrameQueue;
    using lsxnetwork::OpenAiEndpoint;

    // 1) Hard-lock the capturing thread to physical cores 0..3 when the host
    //    allows it; otherwise the pin is impossible and must not fail the suite.
    if (AffinityToCores0To3Feasible()) {
        REQUIRE(CpuAffinity::PinToCores0To3() == 0);
        CHECK(CpuAffinity::CurrentAllowedCores() == CpuAffinity::kAllowedMask);
    }

    // 2) Queue ownership model: the frame queue owns the raw chunk, so the
    //    parsed views never dangle and no heap churn occurs on the I/O path.
    FrameQueue queue(4u, 4096u);

    std::size_t cap = 0;
    char* chunk = queue.Acquire(&cap);
    REQUIRE(chunk != nullptr);
    REQUIRE(cap >= kOpenAIPayload.size());
    std::memcpy(chunk, kOpenAIPayload.data(), kOpenAIPayload.size());
    REQUIRE(queue.Publish(kOpenAIPayload.size()) == 0);

    // 3) Capture + parse straight from the queue-owned buffer (no corruption).
    FrameQueue::Frame frame;
    REQUIRE(queue.Peek(frame));
    CHECK(frame.size == kOpenAIPayload.size());
    CHECK(std::string_view(frame.data, frame.size) == kOpenAIPayload);
    CHECK(frame.capacity >= frame.size + FrameQueue::kSimdjsonPadding);

    ChatCompletion cc;
    const simdjson::error_code rc =
        OpenAiEndpoint().Parse(frame.data, frame.size, frame.capacity, cc);
    REQUIRE(rc == simdjson::SUCCESS);
    CHECK(cc.object == "chat.completion");
    CHECK(cc.role == "assistant");
    CHECK(cc.content == "The capital of France is Paris.");
    CHECK(cc.finish_reason == "stop");
    CHECK(cc.prompt_tokens == 11);
    CHECK(cc.completion_tokens == 8);
    CHECK(cc.total_tokens == 19);

    queue.Consume();
    CHECK(queue.Empty());
    CHECK_FALSE(queue.Full());
}

TEST_CASE("lsxnetwork: SPSC handoff across threads preserves frame integrity",
          "[mosaic][network][spsc]")
{
    using lsxnetwork::CpuAffinity;
    using lsxnetwork::FrameQueue;

    constexpr int kFrames = 2000;
    FrameQueue queue(32u, 4096u);

    // Producer fills fixed-size frames; each frame i carries byte (i & 0xff)
    // and length (i % 3000) + 1. The consumer verifies FIFO order and content.
    std::thread producer([&queue] {
        CpuAffinity::PinToCores0To3();  // best-effort within the worker
        for (int i = 0; i < kFrames; ++i) {
            const std::size_t len = static_cast<std::size_t>(i % 3000) + 1;
            std::size_t cap = 0;
            char* chunk = nullptr;
            while ((chunk = queue.Acquire(&cap)) == nullptr) {
                std::this_thread::yield();
            }
            std::memset(chunk, static_cast<int>(i & 0xff), len);
            while (queue.Publish(len) != 0) {
                std::this_thread::yield();
            }
        }
    });

    int received = 0;
    while (received < kFrames) {
        FrameQueue::Frame frame;
        if (!queue.Peek(frame)) {
            std::this_thread::yield();
            continue;
        }
        const int expected = received;
        const std::size_t expected_len = static_cast<std::size_t>(expected % 3000) + 1;
        CHECK(frame.size == expected_len);
        if (frame.size == expected_len) {
            const char fill = static_cast<char>(expected & 0xff);
            for (std::size_t i = 0; i < frame.size; ++i) {
                CHECK(frame.data[i] == fill);
            }
        }
        queue.Consume();
        ++received;
    }
    producer.join();
    CHECK(received == kFrames);
}

TEST_CASE("lsxnetwork: minimal two-slot queue wraps without corruption",
          "[mosaic][network][spsc]")
{
    using lsxnetwork::FrameQueue;

    FrameQueue queue(2u, 1024u);

    std::size_t cap = 0;
    char* chunk = queue.Acquire(&cap);
    REQUIRE(chunk != nullptr);
    chunk[0] = 'x';
    REQUIRE(queue.Publish(1u) == 0);

    chunk = queue.Acquire(&cap);
    REQUIRE(chunk != nullptr);
    chunk[0] = 'y';
    REQUIRE(queue.Publish(1u) == 0);
    CHECK(queue.Full());  // both slots in flight

    FrameQueue::Frame frame;
    REQUIRE(queue.Peek(frame));
    CHECK(frame.data[0] == 'x');
    queue.Consume();  // frees slot 0, producer can wrap

    chunk = queue.Acquire(&cap);
    REQUIRE(chunk != nullptr);
    chunk[0] = 'z';
    REQUIRE(queue.Publish(1u) == 0);

    REQUIRE(queue.Peek(frame));
    CHECK(frame.data[0] == 'y');
    queue.Consume();
    REQUIRE(queue.Peek(frame));
    CHECK(frame.data[0] == 'z');
    queue.Consume();

    CHECK(queue.Empty());
    CHECK_FALSE(queue.Full());
}

TEST_CASE("lsxnetwork: zero-alloc C-API ZeroMQ context is constructible",
          "[mosaic][network][zmq]")
{
    lsxnetwork::ZmqContext ctx;
    CHECK(ctx.Valid());
}

TEST_CASE("lsxnetwork: full queue rejects writes without corrupting live frames",
          "[mosaic][network][backpressure]")
{
    using lsxnetwork::FrameQueue;

    FrameQueue queue(4u, 4096u);
    std::size_t cap = 0;

    // Fill every slot.
    for (int i = 0; i < 4; ++i) {
        char* c = queue.Acquire(&cap);
        REQUIRE(c != nullptr);
        std::memset(c, static_cast<int>('A' + i), 8u);
        REQUIRE(queue.Publish(8u) == 0);
    }
    CHECK(queue.Full());

    // Slow-consumer tax: the 5th acquire must deterministically reject
    // (nullptr -> 429/503), never silently overwrite the ring head.
    CHECK(queue.Acquire(&cap) == nullptr);

    // Backlog remains intact — no dangling pointers, no corruption.
    for (int i = 0; i < 4; ++i) {
        FrameQueue::Frame f;
        REQUIRE(queue.Peek(f));
        CHECK(f.size == 8u);
        for (int b = 0; b < 8; ++b) {
            CHECK(f.data[b] == static_cast<char>('A' + i));
        }
        queue.Consume();
    }
    CHECK(queue.Empty());
}

TEST_CASE("lsxnetwork: zero-copy TryTake/Release frees a slot only on Release",
          "[mosaic][network][zerocopy]")
{
    using lsxnetwork::FrameQueue;

    FrameQueue queue(2u, 4096u);
    std::size_t cap = 0;

    char* c0 = queue.Acquire(&cap);
    REQUIRE(c0 != nullptr);
    c0[0] = 'a';
    REQUIRE(queue.Publish(1u) == 0);
    char* c1 = queue.Acquire(&cap);
    REQUIRE(c1 != nullptr);
    c1[0] = 'b';
    REQUIRE(queue.Publish(1u) == 0);
    CHECK(queue.Full());

    FrameQueue::Frame f;
    std::size_t idx0 = 0, idx1 = 0;
    REQUIRE(queue.TryTake(f, idx0));
    CHECK(idx0 == 0u);
    CHECK(f.data[0] == 'a');
    REQUIRE(queue.TryTake(f, idx1));
    CHECK(idx1 == 1u);
    CHECK(f.data[0] == 'b');

    // Neither frame has reached its transport yet: the producer is still
    // throttled (both slots in flight), so it must not be able to reuse them.
    CHECK(queue.Full());
    CHECK(queue.Acquire(&cap) == nullptr);

    // Releasing one slot unthrottles the producer by exactly one slot.
    queue.Release(0);
    char* c = queue.Acquire(&cap);
    REQUIRE(c != nullptr);
    c[0] = 'r';
    REQUIRE(queue.Publish(1u) == 0);
    // Slot 1 (frame 'b') is still in flight -> producer is throttled again.
    CHECK(queue.Acquire(&cap) == nullptr);

    queue.Release(1);
    CHECK(queue.Acquire(&cap) != nullptr);
}

TEST_CASE("lsxnetwork: zero-copy ZMQ send releases the slot only after transit",
          "[mosaic][network][zerocopy][zmq]")
{
    using lsxnetwork::FrameQueue;
    using lsxnetwork::ZmqContext;
    using lsxnetwork::ZmqSocket;

    ZmqContext ctx;
    REQUIRE(ctx.Valid());
    ZmqSocket sender(ctx, ZMQ_PAIR);
    ZmqSocket receiver(ctx, ZMQ_PAIR);
    REQUIRE(sender.Bind("inproc://lsx_zc_test") == 0);
    REQUIRE(receiver.Connect("inproc://lsx_zc_test") == 0);

    FrameQueue queue(2u, 4096u);
    std::size_t cap = 0;
    char* chunk = queue.Acquire(&cap);
    REQUIRE(chunk != nullptr);
    const std::string_view payload = "zero-copy prompt must not be distorted";
    std::memcpy(chunk, payload.data(), payload.size());
    REQUIRE(queue.Publish(payload.size()) == 0);

    FrameQueue::Frame f;
    std::size_t idx = 0;
    REQUIRE(queue.TryTake(f, idx));

    // Hand the queue chunk straight to ZMQ. The slot is released exclusively by
    // libzmq's free callback — never by the caller. (For same-thread inproc the
    // transit is synchronous, so the callback may fire before SendZeroCopy
    // returns; the payload integrity check below is what proves the buffer was
    // not overwritten mid-transit.)
    REQUIRE(sender.SendZeroCopy(queue, f, idx) == static_cast<int>(payload.size()));

    // Peer consumes the exact bytes…
    char rbuf[256];
    const int n = receiver.Recv(rbuf, sizeof(rbuf));
    REQUIRE(n == static_cast<int>(payload.size()));
    CHECK(std::memcmp(rbuf, payload.data(), payload.size()) == 0);

    // …and only now (after the free callback fires) the slot is reusable.
    bool released = false;
    for (int i = 0; i < 10000 && !released; ++i) {
        if (queue.Acquire(&cap) != nullptr) {
            released = true;
        }
    }
    CHECK(released);
}

#if !defined(_WIN32)
TEST_CASE("lsxnetwork: bind is idempotent across a stale ipc:// socket file",
          "[mosaic][network][zmq][ipc]")
{
    using lsxnetwork::ZmqContext;
    using lsxnetwork::ZmqSocket;

    const std::string path = "/tmp/lsx_mosaic_test.ipc";
    const std::string endpoint = "ipc://" + path;

    // Simulate a stale socket file left behind by an aborted/crashed process.
    std::ofstream(path).close();
    REQUIRE(std::filesystem::exists(path));

    ZmqContext ctx;
    REQUIRE(ctx.Valid());
    ZmqSocket sock(ctx, ZMQ_REP);
    REQUIRE(sock.Bind(endpoint.c_str()) == 0);

    std::filesystem::remove(path);
}
#endif

TEST_CASE("openai endpoint: parses chat request",
          "[mosaic][openai][parse]") {
    const std::string body = R"({
      "model":"m1","stream":true,"max_tokens":32,
      "stop":["<|end|>"],
      "stream_options":{"include_usage":true},
      "messages":[{"role":"system","content":"sys"},
                  {"role":"user","content":"hi"}]})";
    lsxnetwork::ChatRequest req;
    simdjson::padded_string padded(body);
    REQUIRE(lsxnetwork::OpenAiEndpoint().ParseChatRequest(
                padded.data(), padded.size(), padded.size() + simdjson::SIMDJSON_PADDING, req) ==
            simdjson::SUCCESS);
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

TEST_CASE("openai endpoint: parses legacy completion prompt",
          "[mosaic][openai][parse]") {
    const std::string body = R"({"model":"m1","prompt":"hello"})";
    lsxnetwork::LegacyCompletionRequest req;
    simdjson::padded_string padded(body);
    REQUIRE(lsxnetwork::OpenAiEndpoint().ParseLegacyCompletionRequest(
                padded.data(), padded.size(), padded.size() + simdjson::SIMDJSON_PADDING, req) ==
            simdjson::SUCCESS);
    CHECK(req.prompt == "hello");
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
