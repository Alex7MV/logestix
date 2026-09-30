#pragma once

#include <absl/container/btree_map.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <string>

namespace lsxnetwork {

struct HttpRequest {
    std::string method;
    std::string path;
    std::string body;
    absl::btree_map<std::string, std::string> headers;  // lower-cased keys
};

struct HttpResponse {
    int status = 200;
    std::string content_type = "application/json; charset=utf-8";
    std::string body;
    absl::btree_map<std::string, std::string> headers;
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
