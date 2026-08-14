#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string_view>

#include "lsxnetwork/spa_assets.h"

namespace lsxnetwork {

/// A fully-materialized HTTP/1.1 response head + body reference.
///
/// `body` ALWAYS points into stable, immutable memory: a mounted static asset
/// (`spa::Asset::data`) or a caller-owned byte range. The transport therefore
/// zero-copy writes it — no request-scoped allocation on the serving path.
struct HttpReply {
    int status = 0;               ///< 200, 404, …
    std::string_view reason;      ///< "OK", "Not Found", …
    std::string_view content_type;///< "text/html; charset=utf-8", …
    std::string_view cache_control;  ///< "public, max-age=3600" for assets.
    const void* body = nullptr;   ///< Immutable body bytes (may be null).
    std::size_t body_size = 0;    ///< Body byte length.
    bool chunked = false;         ///< true for SSE (no Content-Length precomputed).
    int keep_alive = 1;
};

/// Static router for the TaskMosaic CE SPA.
///
/// Two routing modes, both lock-free for concurrent readers:
///  * static assets are served from an immutable compiled-in table (the CMake
///    embedded `www/` bundle) — `body` aliases the aligned asset bytes;
///  * `POST /v1/chat/completions` yields a streaming `text/event-stream`
///    response through a caller-supplied frame sink.
class HttpRouter {
public:
    HttpRouter();

    /// Replace the static asset table (default: the compiled-in SPA bundle).
    /// @p assets must outlive the router. Called at most once, before serving.
    void MountStatic(const spa::Asset* assets, std::size_t count);

    /// Handle a non-streaming request (GET/HEAD static, POST non-stream).
    /// @p path may carry a query string; it is stripped before matching.
    /// Returns the HTTP status and fills @p out.
    int Handle(std::string_view method, std::string_view path,
               std::string_view body, HttpReply& out) const;

    /// Handle `POST /v1/chat/completions` with `stream:true` (SSE).
    ///
    /// @p head receives 200 + `text/event-stream` + chunked framing. Each
    /// produced SSE frame (already `data: {...}\n\n` framed) is pushed through
    /// @p emit, which returns true to keep streaming and false to abort
    /// (e.g. client disconnect). Returns 200 on clean completion.
    int HandleStream(std::string_view path, std::string_view body,
                     HttpReply& head,
                     const std::function<bool(std::string_view frame)>& emit) const;

private:
    const spa::Asset* assets_ = nullptr;
    std::size_t asset_count_ = 0;
};

}  // namespace lsxnetwork
