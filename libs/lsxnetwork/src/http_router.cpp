#include "lsxnetwork/http_router.h"

#include <lsx_embedded_www.h>  // generated: embedded www/ bundle (see CMakeLists.txt)

#include <cstring>
#include <string>

namespace lsxnetwork {

namespace {

constexpr std::string_view kOkReason = "OK";
constexpr std::string_view kNotFoundReason = "Not Found";
constexpr std::string_view kBadReqReason = "Bad Request";
constexpr std::string_view kTextPlain = "text/plain; charset=utf-8";
constexpr std::string_view kJson = "application/json; charset=utf-8";
constexpr std::string_view kEventStream = "text/event-stream; charset=utf-8";
constexpr std::string_view kAssetCache = "public, max-age=3600";

const char kNotFoundBody[] = "404 Not Found — no such mosaic tile.";
const char kBadRequestBody[] = "400 Bad Request — unrecognized route.";

// Strip a trailing query string / fragment from a URL path.
std::string_view StripQuery(std::string_view path) {
    const std::size_t q = path.find_first_of("?#");
    if (q == std::string_view::npos) return path;
    return path.substr(0, q);
}

// Best-effort extraction of the OpenAI `"id"` and `"model"` string fields from
// the request body. The router only needs these for framing the stream; an
// absent/odd payload falls back to canned values rather than failing the route.
void ExtractIdModel(std::string_view body, std::string_view& id,
                    std::string_view& model) {
    constexpr std::string_view kIdKey = "\"id\":\"";
    constexpr std::string_view kModelKey = "\"model\":\"";
    auto take = [&](std::string_view key, std::string_view& dst) {
        const std::size_t k = body.find(key);
        if (k == std::string_view::npos) return;
        const std::size_t s = k + key.size();
        const std::size_t e = body.find('"', s);
        if (e == std::string_view::npos || e == s) return;
        dst = body.substr(s, e - s);
    };
    take(kIdKey, id);
    take(kModelKey, model);
}

}  // namespace

HttpRouter::HttpRouter() {
    // Compile-time bundle: immutable, aligned, lock-free to read.
    assets_ = spa::GetSpaAssets();
    asset_count_ = spa::GetSpaAssetCount();
}

void HttpRouter::MountStatic(const spa::Asset* assets, std::size_t count) {
    assets_ = assets;
    asset_count_ = count;
}

int HttpRouter::Handle(std::string_view method, std::string_view path,
                       std::string_view /*body*/, HttpReply& out) const {
    out = HttpReply{};
    std::string_view p = StripQuery(path);
    if (p.empty()) p = "/";

    // Static assets.
    if (method == "GET" || method == "HEAD") {
        const spa::Asset* asset = nullptr;
        for (std::size_t i = 0; i < asset_count_; ++i) {
            if (assets_[i].path == p) {
                asset = &assets_[i];
                break;
            }
        }
        if (asset != nullptr) {
            out.status = 200;
            out.reason = kOkReason;
            out.content_type = asset->mime;
            out.cache_control = kAssetCache;
            out.body = asset->data;
            out.body_size = (method == "HEAD") ? 0u : asset->size;
            out.keep_alive = 1;
            return 200;
        }
        out.status = 404;
        out.reason = kNotFoundReason;
        out.content_type = kTextPlain;
        out.body = kNotFoundBody;
        out.body_size = sizeof(kNotFoundBody) - 1;
        return 404;
    }

    // Chat completions (non-streaming JSON).
    if (method == "POST" && p == "/v1/chat/completions") {
        out.status = 200;
        out.reason = kOkReason;
        out.content_type = kJson;
        static const char kCompletion[] =
            "{\"id\":\"chatcmpl-mosaic\",\"object\":\"chat.completion\","
            "\"model\":\"mosaic\",\"choices\":[{\"index\":0,\"message\":{"
            "\"role\":\"assistant\",\"content\":\"mosaic\"},\"finish_reason\":"
            "\"stop\"}]}";
        out.body = kCompletion;
        out.body_size = sizeof(kCompletion) - 1;
        return 200;
    }

    out.status = 400;
    out.reason = kBadReqReason;
    out.content_type = kTextPlain;
    out.body = kBadRequestBody;
    out.body_size = sizeof(kBadRequestBody) - 1;
    return 400;
}

int HttpRouter::HandleStream(
    std::string_view path, std::string_view body, HttpReply& head,
    const std::function<bool(std::string_view frame)>& emit) const {
    head = HttpReply{};
    if (StripQuery(path) != "/v1/chat/completions") {
        head.status = 404;
        head.reason = kNotFoundReason;
        head.content_type = kTextPlain;
        head.body = kNotFoundBody;
        head.body_size = sizeof(kNotFoundBody) - 1;
        return 404;
    }

    head.status = 200;
    head.reason = kOkReason;
    head.content_type = kEventStream;
    head.chunked = true;

    std::string_view id = "chatcmpl-mosaic";
    std::string_view model = "mosaic";
    ExtractIdModel(body, id, model);

    // Canned streaming body for the Phase-5 plumbing. The real backend feeds
    // this endpoint off the orchestrated token stream; the router only frames
    // the SSE chunk payloads. Frames are short-lived, caller-scoped strings.
    static constexpr std::string_view kTokens[] = {
        "Task", "Mosaic", " breaks", " a", " prompt", " into", " routed",
        " tiles", " —", " each", " streamed", " live", ".",
    };
    for (const std::string_view tok : kTokens) {
        std::string frame;
        frame.reserve(96 + tok.size());
        frame.append("data: {\"id\":\"");
        frame.append(id);
        frame.append("\",\"object\":\"chat.completion.chunk\",\"model\":\"");
        frame.append(model);
        frame.append("\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"");
        frame.append(tok);
        frame.append("\"},\"finish_reason\":null}]}\n\n");
        if (!emit(frame)) {
            return 200;  // client closed early; not an HTTP error
        }
    }
    static constexpr std::string_view kDone = "data: [DONE]\n\n";
    emit(kDone);
    return 200;
}

}  // namespace lsxnetwork
