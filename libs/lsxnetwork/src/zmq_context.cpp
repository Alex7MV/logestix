#include "lsxnetwork/zmq_context.h"

#include <zmq.h>

#include <spdlog/spdlog.h>

#include <cerrno>
#include <filesystem>
#include <new>
#include <string>
#include <string_view>

namespace lsxnetwork {

namespace {

// Control token passed as the `hint` of zmq_msg_init_data. The message free
// callback runs on whatever thread libzmq frees the message from (its I/O
// thread), so the queue's slot must be released there, not before.
struct FrameReleaseToken {
    FrameQueue* queue = nullptr;
    std::size_t index = 0;
};

void FrameReleaseCb(void*, void* hint) {
    auto* token = static_cast<FrameReleaseToken*>(hint);
    if (token->queue != nullptr) {
        token->queue->Release(token->index);
    }
    delete token;
}

}  // namespace

ZmqContext::ZmqContext() : ctx_(zmq_ctx_new()) {
    if (ctx_ == nullptr) {
        spdlog::error("lsxnetwork: zmq_ctx_new failed");
    }
}

ZmqContext::~ZmqContext() noexcept {
    if (ctx_ != nullptr) {
        zmq_ctx_term(ctx_);
    }
}

ZmqContext::ZmqContext(ZmqContext&& other) noexcept : ctx_(other.ctx_) {
    other.ctx_ = nullptr;
}

ZmqContext& ZmqContext::operator=(ZmqContext&& other) noexcept {
    if (this != &other) {
        if (ctx_ != nullptr) {
            zmq_ctx_term(ctx_);
        }
        ctx_ = other.ctx_;
        other.ctx_ = nullptr;
    }
    return *this;
}

ZmqSocket::ZmqSocket(ZmqContext& ctx, int type) : sock_(zmq_socket(ctx.Get(), type)) {
    if (sock_ == nullptr) {
        spdlog::error("lsxnetwork: zmq_socket (type {}) failed: {}", type,
                      zmq_strerror(zmq_errno()));
    }
}

ZmqSocket::~ZmqSocket() noexcept {
    if (sock_ != nullptr) {
        zmq_close(sock_);
    }
}

ZmqSocket::ZmqSocket(ZmqSocket&& other) noexcept : sock_(other.sock_) {
    other.sock_ = nullptr;
}

ZmqSocket& ZmqSocket::operator=(ZmqSocket&& other) noexcept {
    if (this != &other) {
        if (sock_ != nullptr) {
            zmq_close(sock_);
        }
        sock_ = other.sock_;
        other.sock_ = nullptr;
    }
    return *this;
}

int ZmqSocket::Bind(const char* endpoint) noexcept {
    int rc = zmq_bind(sock_, endpoint);
#if !defined(_WIN32)
    // A crashed process can leave a stale Unix-domain socket file (ipc://)
    // behind; zmq_bind then fails with EADDRINUSE. Remove it and retry exactly
    // once so startup stays idempotent across power/abort failures. The retry
    // is gated on EADDRINUSE only, so genuine bind errors are not masked.
    if (rc != 0 && zmq_errno() == EADDRINUSE &&
        std::string_view(endpoint).starts_with("ipc://")) {
        const std::string path(std::string_view(endpoint).substr(6));  // drop "ipc://"
        std::error_code ec;
        std::filesystem::remove(path, ec);
        rc = zmq_bind(sock_, endpoint);
    }
#endif
    return rc;
}

int ZmqSocket::Connect(const char* endpoint) noexcept {
    return zmq_connect(sock_, endpoint);
}

int ZmqSocket::Recv(void* buf, std::size_t len, int flags) noexcept {
    return static_cast<int>(zmq_recv(sock_, buf, len, flags));
}

int ZmqSocket::Send(const void* buf, std::size_t len, int flags) noexcept {
    return static_cast<int>(zmq_send(sock_, buf, len, flags));
}

int ZmqSocket::SendZeroCopy(FrameQueue& queue, const FrameQueue::Frame& frame,
                            std::size_t index, int flags) noexcept {
    auto* token = new (std::nothrow) FrameReleaseToken{&queue, index};
    if (token == nullptr) {
        queue.Release(index);  // handoff failed: free the slot ourselves
        return -1;
    }
    zmq_msg_t msg;
    if (zmq_msg_init_data(&msg, const_cast<char*>(frame.data), frame.size,
                          &FrameReleaseCb, token) != 0) {
        queue.Release(index);
        delete token;
        return -1;
    }
    const int rc = static_cast<int>(zmq_msg_send(&msg, sock_, flags));
    if (rc < 0) {
        // zmq_msg_close destroys the message and invokes FrameReleaseCb, which
        // releases the slot and frees the token.
        zmq_msg_close(&msg);
        return -1;
    }
    return rc;
}

int ZmqSocket::SetSockOpt(int option, const void* value, std::size_t value_len) noexcept {
    return zmq_setsockopt(sock_, option, value, value_len);
}

int ZmqSocket::GetSockOpt(int option, void* value, std::size_t* value_len) noexcept {
    return zmq_getsockopt(sock_, option, value, value_len);
}

}  // namespace lsxnetwork