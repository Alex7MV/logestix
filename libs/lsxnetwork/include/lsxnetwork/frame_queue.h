#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace lsxnetwork {

/// Lock-free single-producer / single-consumer queue of raw network frames.
///
/// ZERO-ALLOC: the queue owns a fixed ring of chunk buffers allocated once at
/// construction. A published frame is a `string_view` into that queue-owned
/// stable memory, so simdjson can parse it zero-copy without heap allocations
/// and without dangling references. The SPSC handoff uses the Vyukov sequence
/// protocol on `std::atomic`.
class FrameQueue {
public:
    /// A captured frame: raw payload bytes residing in queue-owned storage.
    struct Frame {
        const char* data = nullptr;
        std::size_t size = 0;
        /// Total readable bytes at @p data (payload + simdjson trailing
        /// padding). Hand this to OpenAiEndpoint::Parse as `capacity`.
        std::size_t capacity = 0;
    };

    /// Padding bytes simdjson requires readable past the document end.
    static constexpr std::size_t kSimdjsonPadding = 64;

    /// @param capacity      number of frame slots; min 2 (the sequence protocol
    ///                      cannot distinguish free from in-flight at N=1),
    ///                      rounded up to a power of 2.
    /// @param chunk_payload  payload bytes reserved per frame slot.
    FrameQueue(std::size_t capacity = 8, std::size_t chunk_payload = 65536);
    ~FrameQueue() noexcept;
    FrameQueue(FrameQueue&&) = delete;
    FrameQueue(const FrameQueue&) = delete;
    FrameQueue& operator=(const FrameQueue&) = delete;

    /// Producer: returns a writable chunk and its payload capacity, or nullptr
    /// if the queue is full. The caller fills the buffer then calls Publish().
    ///
    /// A nullptr return is the deterministic backpressure signal: the queue is
    /// full and the caller MUST NOT write to the head. Map it to an HTTP 429
    /// (Too Many Requests) or 503 (Service Unavailable) at the endpoint layer.
    /// The queue never silently overwrites a live slot, so no producer can
    /// corrupt a buffer a consumer is still reading.
    char* Acquire(std::size_t* out_capacity) noexcept;

    /// Producer: seals @p payload_size bytes as the next published frame.
    /// Returns 0 on success, -1 if @p payload_size exceeds the chunk payload.
    int Publish(std::size_t payload_size) noexcept;

    /// Consumer: fetches the oldest frame. Returns false when empty. The
    /// pointed-to bytes remain valid until the next Consume().
    bool Peek(Frame& out) noexcept;

    /// Consumer: releases the current oldest frame slot back to the producer.
    void Consume() noexcept;

    /// Consumer (zero-copy handoff): takes the next published frame WITHOUT
    /// freeing its slot. @p index is the frame's global slot token; hand it to
    /// a zero-copy transport (e.g. ZmqSocket::SendZeroCopy) which will call
    /// Release() exactly when the buffer is no longer in flight.
    ///
    /// Precondition: a queue is consumed by exactly one mode — either
    /// Peek/Consume or TryTake/Release — never both.
    bool TryTake(Frame& out, std::size_t& index) noexcept;

    /// Frees the slot identified by @p index (returned by TryTake). Safe to
    /// call from any thread (the ZMQ I/O thread invokes it via the message
    /// free callback). Idempotent: the slot is released exactly once.
    void Release(std::size_t index) noexcept;

    bool Empty() const noexcept;
    bool Full() const noexcept;

private:
    std::size_t CapacityMask() const noexcept { return capacity_ - 1; }

    std::size_t capacity_ = 0;         // power of two
    std::size_t payload_cap_ = 0;      // payload bytes per chunk
    std::size_t stride_ = 0;           // chunk_payload + padding, cache-line aligned
    std::unique_ptr<std::byte[]> storage_;
    std::unique_ptr<std::uint32_t[]> len_;      // per-slot payload length
    std::unique_ptr<std::atomic<std::uint64_t>[]> seq_;  // per-slot sequence
    std::atomic<std::size_t> head_{0};   // producer (write) index
    std::atomic<std::size_t> tail_{0};   // consumer (read) index
    std::size_t read_ = 0;               // zero-copy consumer watermark (single consumer)
};

}  // namespace lsxnetwork