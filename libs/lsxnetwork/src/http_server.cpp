#include "lsxnetwork/http_server.h"

#include "lsxnetwork/http_router.h"

#include <httplib.h>

#include <atomic>
#include <memory>
#include <string>
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
void WriteReply(const HttpResponse& rr, httplib::Response& res) {
    res.status = rr.status;
    res.set_content(rr.body, rr.content_type.c_str());
    for (const auto& [k, v] : rr.headers) res.set_header(k, v);
}
}  // namespace

void HttpServer::Get(const std::string& path, Handler handler) {
    impl_->server.Get(path, [handler](const httplib::Request& req,
                                      httplib::Response& res) {
        HttpRequest rq{req.method, req.path, req.body};
        HttpResponse rr;
        handler(rq, rr);
        WriteReply(rr, res);
    });
}

void HttpServer::Post(const std::string& path, Handler handler) {
    impl_->server.Post(path, [handler](const httplib::Request& req,
                                       httplib::Response& res) {
        HttpRequest rq{req.method, req.path, req.body};
        HttpResponse rr;
        handler(rq, rr);
        WriteReply(rr, res);
    });
}

void HttpServer::PostStream(const std::string& path, StreamFactory factory) {
    impl_->server.Post(path, [factory](const httplib::Request& req,
                                       httplib::Response& res) {
        HttpRequest rq{req.method, req.path, req.body};
        HttpResponse head;
        std::unique_ptr<StreamSource> source = factory(rq, head);
        if (!source) {
            WriteReply(head, res);
            return;
        }
        res.status = 200;
        auto shared = std::shared_ptr<StreamSource>(std::move(source));
        res.set_chunked_content_provider(
            head.content_type.c_str(),
            [shared](std::size_t, httplib::DataSink& sink) {
                if (!sink.is_writable()) {
                    shared->Cancel();
                    sink.done();
                    return true;
                }
                std::string out;
                if (!shared->Next(out)) {
                    sink.done();  // graceful end of the chunked body
                    return true;
                }
                if (out.empty()) return true;  // nothing ready yet; keep polling
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
        const int bound = impl_->server.bind_to_any_port(host);
        if (bound <= 0) return false;
        impl_->port.store(bound);
    } else {
        if (!impl_->server.bind_to_port(host, port)) return false;
        impl_->port.store(port);
    }
    return impl_->server.listen_after_bind();
}

int HttpServer::BoundPort() const { return impl_->port.load(); }
void HttpServer::Stop() { impl_->server.stop(); }

}  // namespace lsxnetwork
