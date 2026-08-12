#include "lsxnetwork/frame_queue.h"

#include <algorithm>
#include <cstring>

namespace lsxnetwork {

namespace {

std::size_t NextPow2(std::size_t v) {
    --v;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    return v + 1;
}

}  // namespace

FrameQueue::FrameQueue(std::size_t capacity, std::size_t chunk_payload)
    : capacity_(NextPow2(std::max<std::size_t>(2, capacity))),
      payload_cap_(chunk_payload),
      stride_((chunk_payload + kSimdjsonPadding + 63u) & ~std::size_t(63u)) {
    const std::size_t bytes = capacity_ * stride_;
    storage_ = std::unique_ptr<std::byte[]>(new std::byte[bytes]());
    len_ = std::unique_ptr<std::uint32_t[]>(new std::uint32_t[capacity_]());
    seq_ = std::unique_ptr<std::atomic<std::uint64_t>[]>(
        new std::atomic<std::uint64_t>[capacity_]);
    for (std::size_t i = 0; i < capacity_; ++i) {
        seq_[i].store(static_cast<std::uint64_t>(i), std::memory_order_relaxed);
    }
    head_.store(0, std::memory_order_relaxed);
    tail_.store(0, std::memory_order_relaxed);
}

FrameQueue::~FrameQueue() noexcept = default;

char* FrameQueue::Acquire(std::size_t* out_capacity) noexcept {
    const std::size_t n = head_.load(std::memory_order_relaxed);
    const std::size_t slot = n & CapacityMask();
    if (seq_[slot].load(std::memory_order_acquire) < n) {
        return nullptr;  // producer has caught up with the consumer
    }
    *out_capacity = payload_cap_;
    return reinterpret_cast<char*>(storage_.get()) + slot * stride_;
}

int FrameQueue::Publish(std::size_t payload_size) noexcept {
    const std::size_t n = head_.load(std::memory_order_relaxed);
    const std::size_t slot = n & CapacityMask();
    if (payload_size > payload_cap_) {
        return -1;
    }
    len_[slot] = static_cast<std::uint32_t>(payload_size);
    seq_[slot].store(n + 1, std::memory_order_release);
    head_.store(n + 1, std::memory_order_relaxed);  // single producer
    return 0;
}

bool FrameQueue::Peek(Frame& out) noexcept {
    const std::size_t n = tail_.load(std::memory_order_relaxed);
    const std::size_t slot = n & CapacityMask();
    if (seq_[slot].load(std::memory_order_acquire) != n + 1) {
        return false;  // empty
    }
    out.data = reinterpret_cast<const char*>(storage_.get()) + slot * stride_;
    out.size = static_cast<std::size_t>(len_[slot]);
    out.capacity = stride_;
    return true;
}

void FrameQueue::Consume() noexcept {
    const std::size_t n = tail_.load(std::memory_order_relaxed);
    const std::size_t slot = n & CapacityMask();
    seq_[slot].store(n + capacity_, std::memory_order_release);
    tail_.store(n + 1, std::memory_order_relaxed);  // single consumer
}

bool FrameQueue::TryTake(Frame& out, std::size_t& index) noexcept {
    const std::size_t i = read_;
    const std::size_t slot = i & CapacityMask();
    if (seq_[slot].load(std::memory_order_acquire) != i + 1) {
        return false;  // nothing published at the read watermark yet
    }
    out.data = reinterpret_cast<const char*>(storage_.get()) + slot * stride_;
    out.size = static_cast<std::size_t>(len_[slot]);
    out.capacity = stride_;
    index = i;
    read_ = i + 1;  // single consumer
    return true;
}

void FrameQueue::Release(std::size_t index) noexcept {
    // Advances the slot's sequence so the producer can reuse it. Only touches
    // this slot's atomic, so it is safe to invoke from any thread (e.g. ZMQ's
    // I/O thread inside the message free callback). seq[slot] is monotonic:
    // indices assigned to a slot are `capacity` apart, so a double release is
    // idempotent and out-of-order releases across slots are benign.
    const std::size_t slot = index & CapacityMask();
    seq_[slot].store(index + capacity_, std::memory_order_release);
}

bool FrameQueue::Empty() const noexcept {
    const std::size_t n = tail_.load(std::memory_order_relaxed);
    const std::size_t slot = n & CapacityMask();
    return seq_[slot].load(std::memory_order_acquire) != n + 1;
}

bool FrameQueue::Full() const noexcept {
    const std::size_t n = head_.load(std::memory_order_relaxed);
    const std::size_t slot = n & CapacityMask();
    return seq_[slot].load(std::memory_order_acquire) < n;
}

}  // namespace lsxnetwork