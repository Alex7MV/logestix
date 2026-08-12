#pragma once

#include <cstddef>

#include "lsxnetwork/frame_queue.h"

namespace lsxnetwork {

/// Opaque ZeroMQ handle.
using ZmqHandle = void*;

/// Thin RAII wrapper over the ZeroMQ C API context (`zmq_ctx_*`).
///
/// Kept deliberately small: the network layer is a native-C wrapper on top of
/// `libzmq-static` (the C API), avoiding an external cppzmq dependency. The
/// wrapper owns the context lifetime and never allocates on the I/O path.
class ZmqContext {
public:
    ZmqContext();
    ~ZmqContext() noexcept;
    ZmqContext(ZmqContext&& other) noexcept;
    ZmqContext& operator=(ZmqContext&& other) noexcept;
    ZmqContext(const ZmqContext&) = delete;
    ZmqContext& operator=(const ZmqContext&) = delete;

    bool Valid() const noexcept { return ctx_ != nullptr; }
    ZmqHandle Get() const noexcept { return ctx_; }

private:
    ZmqHandle ctx_ = nullptr;
};

/// RAII wrapper over a ZeroMQ socket (`zmq_socket`/`zmq_close`).
///
/// Socket type is an int taken directly from `<zmq.h>` (ZMQ_REP, ZMQ_ROUTER,
/// ZMQ_PAIR, ...). All receive/send operations are non-allocating.
class ZmqSocket {
public:
    explicit ZmqSocket(ZmqContext& ctx, int type);
    ~ZmqSocket() noexcept;
    ZmqSocket(ZmqSocket&& other) noexcept;
    ZmqSocket& operator=(ZmqSocket&& other) noexcept;
    ZmqSocket(const ZmqSocket&) = delete;
    ZmqSocket& operator=(const ZmqSocket&) = delete;

    bool Valid() const noexcept { return sock_ != nullptr; }
    ZmqHandle Get() const noexcept { return sock_; }

    /// Binds the socket to an endpoint string (e.g. "tcp://*:5555").
    /// Idempotent for "ipc://" endpoints: a Unix-domain socket file left by a
    /// crashed process is detected (EADDRINUSE) and removed before retrying.
    int Bind(const char* endpoint) noexcept;
    /// Connects the socket to an endpoint string (e.g. "tcp://server:5555").
    int Connect(const char* endpoint) noexcept;
    /// Receives up to @p len bytes; returns bytes received, or -1 on error
    /// (inspect the error via `zmq_errno()`).
    int Recv(void* buf, std::size_t len, int flags = 0) noexcept;
    /// Sends @p len bytes; returns bytes sent, or -1 on error
    /// (inspect the error via `zmq_errno()`).
    int Send(const void* buf, std::size_t len, int flags = 0) noexcept;

    /// Zero-copy send of a queue-owned frame. The payload is NOT copied into a
    /// libzmq buffer: `zmq_msg_init_data` wraps the queue's chunk, so the
    /// network worker can never overwrite the bytes mid-transit.
    ///
    /// Ownership of the queue slot is transferred to libzmq. The slot is
    /// released (reusable by the producer) only when libzmq's message free
    /// callback runs after transit completes. After a successful handoff DO NOT
    /// call FrameQueue::Release for @p index yourself.
    int SendZeroCopy(FrameQueue& queue, const FrameQueue::Frame& frame,
                     std::size_t index, int flags = 0) noexcept;
    int SetSockOpt(int option, const void* value, std::size_t value_len) noexcept;
    int GetSockOpt(int option, void* value, std::size_t* value_len) noexcept;

private:
    ZmqHandle sock_ = nullptr;
};

}  // namespace lsxnetwork