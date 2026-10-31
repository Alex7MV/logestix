#include "chat_bridge.h"

#include "lsxcommon/log.h"
#include "lsxnetwork/anthropic_endpoint.h"
#include "lsxnetwork/openai_endpoint.h"

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace taskmosaic {
namespace {

constexpr int kPollMs = 100;
constexpr const char* kJson = "application/json; charset=utf-8";

lsxcommon::ChatConversation BuildConversation(
    const std::vector<lsxnetwork::ChatInputMessage>& messages,
    const std::vector<lsxnetwork::ChatToolDef>& tools) {
    lsxcommon::ChatConversation conv;
    for (const auto& m : messages) {
        lsxcommon::ChatMessage cm;
        cm.role = m.role;
        cm.content = m.content;
        cm.tool_call_id = m.tool_call_id;
        for (const auto& tc : m.tool_calls)
            cm.tool_calls.push_back({tc.id, tc.name, tc.arguments_json});
        conv.messages.push_back(std::move(cm));
    }
    for (const auto& t : tools)
        conv.tools.push_back({t.name, t.description, t.parameters_json});
    return conv;
}

lsxcommon::ChatConversation BuildConversation(const lsxnetwork::ChatRequest& req) {
    lsxcommon::ChatConversation conv =
        BuildConversation(req.messages, req.tools);
    conv.adapter_name = req.adapter_name;
    return conv;
}

bool ToolsRequested(const lsxnetwork::ChatRequest& req) {
    return !req.tools.empty() && req.tool_choice != "none";
}

// RAII admission slot: releases the backend counter on scope exit.
struct Admission {
    Backend& backend;
    bool held;
    ~Admission() {
        if (held) backend.Release();
    }
};

// Blocking (non-streaming) generation: returns true and fills `ids`/`eos`.
bool RunBlocking(Backend& backend, const std::vector<std::int32_t>& input_ids,
                 int max_tokens, const lsxnetwork::RequestSampling& sampling,
                 int timeout_ms,
                 std::vector<std::int32_t>& out_ids, bool& eos,
                 std::string& error) {
    std::unique_lock<std::mutex> lock(backend.Mutex());
    const std::uint32_t rid = backend.NextRequestId();
    if (!backend.Client().Submit(rid, input_ids,
                                 static_cast<std::uint32_t>(max_tokens),
                                 sampling)) {
        backend.MarkDown();
        error = "backend submit failed";
        return false;
    }
    int elapsed = 0;
    while (elapsed < timeout_ms) {
        lsxnetwork::FabricStreamFrame f;
        const auto st = backend.Client().Recv(f, kPollMs);
        if (st == lsxnetwork::FabricStreamClient::RecvStatus::kTimeout) {
            elapsed += kPollMs;
            continue;
        }
        if (st == lsxnetwork::FabricStreamClient::RecvStatus::kTransport) {
            backend.MarkDown();
            error = "backend transport error";
            return false;
        }
        if (st != lsxnetwork::FabricStreamClient::RecvStatus::kOk) {
            error = "backend protocol error";
            return false;
        }
        if (f.type == lsxnetwork::FabricFrameType::kError) {
            error = f.error.empty() ? "inference failed" : f.error;
            return false;
        }
        if (f.type == lsxnetwork::FabricFrameType::kTokenBatch) {
            out_ids.insert(out_ids.end(), f.tokens.begin(), f.tokens.end());
            continue;
        }
        eos = f.eos_terminated;  // kDone
        return true;
    }
    error = "backend timeout";
    backend.Client().Cancel(rid);
    int drain = 0;
    while (drain < 3000) {
        lsxnetwork::FabricStreamFrame f;
        const auto st = backend.Client().Recv(f, kPollMs);
        if (st == lsxnetwork::FabricStreamClient::RecvStatus::kTimeout) {
            drain += kPollMs;
            continue;
        }
        if (st != lsxnetwork::FabricStreamClient::RecvStatus::kOk) break;
        if (f.type == lsxnetwork::FabricFrameType::kDone ||
            f.type == lsxnetwork::FabricFrameType::kError)
            break;
    }
    return false;
}

std::string ToolCallsWire(const std::vector<lsxcommon::ToolCall>& calls) {
    std::vector<lsxnetwork::ChatToolCall> wire;
    wire.reserve(calls.size());
    for (const auto& c : calls) wire.push_back({c.id, c.name, c.arguments_json});
    return lsxnetwork::ToolCallsToJson(wire);
}

// If any stop sequence occurs, returns the text truncated before the earliest
// occurrence and sets @p hit; otherwise returns @p text unchanged.
std::string TruncateAtStop(std::string text, const std::vector<std::string>& stops,
                           bool& hit) {
    hit = false;
    std::size_t best = std::string::npos;
    for (const auto& s : stops) {
        if (s.empty()) continue;
        const std::size_t p = text.find(s);
        if (p != std::string::npos && p < best) best = p;
    }
    if (best != std::string::npos) {
        hit = true;
        text.resize(best);
    }
    return text;
}

// After an early-terminated request the shared DEALER may still hold unread
// token/done frames from the backend; cancel the request and drain them so the
// NEXT request on this backend does not read stale frames.
void DrainToTerminal(Backend& backend, std::uint32_t request_id, int grace_ms) {
    if (request_id != 0) backend.Client().Cancel(request_id);
    int waited = 0;
    while (waited < grace_ms) {
        lsxnetwork::FabricStreamFrame f;
        const auto st = backend.Client().Recv(f, kPollMs);
        if (st == lsxnetwork::FabricStreamClient::RecvStatus::kTimeout) {
            waited += kPollMs;
            continue;
        }
        if (st != lsxnetwork::FabricStreamClient::RecvStatus::kOk) return;
        if (f.type == lsxnetwork::FabricFrameType::kDone ||
            f.type == lsxnetwork::FabricFrameType::kError)
            return;
    }
}

/// SSE source for one streaming chat completion. Holds the backend lock and
/// admission slot for its lifetime.
class ChatStreamSource : public lsxnetwork::StreamSource {
public:
    ChatStreamSource(Backend& backend, PromptFramer& framer,
                     std::string model_path, std::string id, std::string model,
                     std::vector<std::int32_t> input_ids, int max_tokens,
                     lsxnetwork::RequestSampling sampling, int timeout_ms,
                     bool include_usage, bool buffer_tools,
                     std::vector<std::string> stops)
        : backend_(backend),
          framer_(framer),
          model_path_(std::move(model_path)),
          id_(std::move(id)),
          model_(std::move(model)),
          input_ids_(std::move(input_ids)),
          max_tokens_(max_tokens),
          sampling_(sampling),
          timeout_ms_(timeout_ms),
          include_usage_(include_usage),
          buffer_tools_(buffer_tools),
          buffered_(buffer_tools || !stops.empty()),
          stops_(std::move(stops)) {}

    ~ChatStreamSource() override {
        if (submitted_ && !terminal_)
            DrainToTerminal(backend_, request_id_, 3000);
        backend_.Release();
    }

    bool Next(std::string& out) override {
        if (!lock_)
            lock_ = std::make_unique<std::unique_lock<std::mutex>>(backend_.Mutex());

        if (!submitted_) {
            if (!buffered_) decoder_ = framer_.MakeDecoder(model_path_);
            request_id_ = backend_.NextRequestId();
            if (!backend_.Client().Submit(request_id_, input_ids_,
                                          static_cast<std::uint32_t>(max_tokens_),
                                          sampling_)) {
                backend_.MarkDown();
                out = Finish("backend submit failed");
                return true;
            }
            submitted_ = true;
            out = lsxnetwork::SseFrame(lsxnetwork::BuildChatChunk(
                id_, model_, "", true, "", 0, 0, false));
            return true;
        }
        if (done_) return false;

        lsxnetwork::FabricStreamFrame f;
        const auto st = backend_.Client().Recv(f, kPollMs);
        if (st == lsxnetwork::FabricStreamClient::RecvStatus::kTimeout) {
            elapsed_ += kPollMs;
            if (elapsed_ >= timeout_ms_) {
                out = Finish("backend timeout");
                return true;
            }
            out.clear();
            return true;
        }
        if (st == lsxnetwork::FabricStreamClient::RecvStatus::kTransport) {
            backend_.MarkDown();
            out = Finish("backend transport error");
            return true;
        }
        if (st != lsxnetwork::FabricStreamClient::RecvStatus::kOk) {
            out = Finish("backend protocol error");
            return true;
        }
        if (f.type == lsxnetwork::FabricFrameType::kError) {
            out = Finish(f.error.empty() ? "inference failed" : f.error);
            terminal_ = true;
            return true;
        }
        if (f.type == lsxnetwork::FabricFrameType::kTokenBatch) {
            out.clear();
            for (const auto tok : f.tokens) {
                out_ids_.push_back(tok);
                if (!buffered_ && decoder_) {
                    const std::string delta = decoder_->Push(tok);
                    if (!delta.empty())
                        out += lsxnetwork::SseFrame(lsxnetwork::BuildChatChunk(
                            id_, model_, delta, false, "", 0, 0, false));
                }
            }
            return true;
        }
        // kDone
        out.clear();
        const int pt = static_cast<int>(input_ids_.size());
        const int ct = static_cast<int>(out_ids_.size());
        if (buffered_) {
            std::vector<lsxcommon::ToolCall> calls;
            if (buffer_tools_) framer_.ParseToolCalls(model_path_, out_ids_, calls);
            if (!calls.empty()) {
                out = lsxnetwork::SseFrame(lsxnetwork::BuildChatChunkTools(
                    id_, model_, ToolCallsWire(calls), true, pt, ct,
                    include_usage_));
            } else {
                bool stop_hit = false;
                const std::string text =
                    TruncateAtStop(framer_.Decode(model_path_, out_ids_), stops_,
                                   stop_hit);
                const std::string fin = stop_hit ? "stop"
                                       : (f.eos_terminated ? "stop" : "length");
                out = lsxnetwork::SseFrame(lsxnetwork::BuildChatChunk(
                    id_, model_, text, true, fin, pt, ct, include_usage_));
            }
        } else {
            if (decoder_) {
                const std::string tail = decoder_->Flush();
                if (!tail.empty())
                    out += lsxnetwork::SseFrame(lsxnetwork::BuildChatChunk(
                        id_, model_, tail, false, "", 0, 0, false));
            }
            out += lsxnetwork::SseFrame(lsxnetwork::BuildChatChunk(
                id_, model_, "", false, f.eos_terminated ? "stop" : "length",
                pt, ct, include_usage_));
        }
        out += lsxnetwork::SseFrame("[DONE]");
        done_ = true;
        terminal_ = true;
        return true;
    }

    void Cancel() override {
        cancelled_ = true;
        if (submitted_ && request_id_ != 0)
            backend_.Client().Cancel(request_id_);
    }

private:
    std::string Finish(const std::string& message) {
        done_ = true;
        return lsxnetwork::SseFrame(
                   lsxnetwork::BuildError(message, "server_error")) +
               lsxnetwork::SseFrame("[DONE]");
    }

    Backend& backend_;
    PromptFramer& framer_;
    std::string model_path_;
    std::string id_;
    std::string model_;
    std::vector<std::int32_t> input_ids_;
    int max_tokens_;
    lsxnetwork::RequestSampling sampling_;
    int timeout_ms_;
    bool include_usage_;
    bool buffer_tools_;
    bool buffered_ = false;
    std::vector<std::string> stops_;
    std::unique_ptr<std::unique_lock<std::mutex>> lock_;
    std::unique_ptr<StreamingDecoder> decoder_;
    std::vector<std::int32_t> out_ids_;
    std::uint32_t request_id_ = 0;
    int elapsed_ = 0;
    bool submitted_ = false;
    bool done_ = false;
    bool terminal_ = false;
    bool cancelled_ = false;
};

/// Anthropic Messages SSE source.
class AnthropicStreamSource : public lsxnetwork::StreamSource {
public:
    AnthropicStreamSource(Backend& backend, PromptFramer& framer,
                          std::string model_path, std::string id,
                          std::string model,
                          std::vector<std::int32_t> input_ids, int max_tokens,
                          lsxnetwork::RequestSampling sampling, int timeout_ms,
                          bool buffer_tools,
                          std::vector<std::string> stops)
        : backend_(backend),
          framer_(framer),
          model_path_(std::move(model_path)),
          id_(std::move(id)),
          model_(std::move(model)),
          input_ids_(std::move(input_ids)),
          max_tokens_(max_tokens),
          sampling_(sampling),
          timeout_ms_(timeout_ms),
          buffer_tools_(buffer_tools),
          buffered_(buffer_tools || !stops.empty()),
          stops_(std::move(stops)) {}

    ~AnthropicStreamSource() override {
        if (submitted_ && !terminal_)
            DrainToTerminal(backend_, request_id_, 3000);
        backend_.Release();
    }

    bool Next(std::string& out) override {
        if (!lock_)
            lock_ = std::make_unique<std::unique_lock<std::mutex>>(backend_.Mutex());
        if (!submitted_) {
            if (!buffered_) decoder_ = framer_.MakeDecoder(model_path_);
            request_id_ = backend_.NextRequestId();
            if (!backend_.Client().Submit(request_id_, input_ids_,
                                          static_cast<std::uint32_t>(max_tokens_),
                                          sampling_)) {
                out = FinishError("backend submit failed");
                return true;
            }
            submitted_ = true;
            out = lsxnetwork::AnthropicMessageStart(
                id_, model_, static_cast<int>(input_ids_.size()));
            return true;
        }
        if (done_) return false;

        lsxnetwork::FabricStreamFrame f;
        const auto st = backend_.Client().Recv(f, kPollMs);
        if (st == lsxnetwork::FabricStreamClient::RecvStatus::kTimeout) {
            elapsed_ += kPollMs;
            if (elapsed_ >= timeout_ms_) {
                out = FinishError("backend timeout");
                return true;
            }
            out.clear();
            return true;
        }
        if (st != lsxnetwork::FabricStreamClient::RecvStatus::kOk) {
            out = FinishError("backend transport error");
            return true;
        }
        if (f.type == lsxnetwork::FabricFrameType::kError) {
            out = FinishError(f.error.empty() ? "inference failed" : f.error);
            terminal_ = true;
            return true;
        }
        if (f.type == lsxnetwork::FabricFrameType::kTokenBatch) {
            out.clear();
            for (const auto tok : f.tokens) {
                out_ids_.push_back(tok);
                if (buffered_ || !decoder_) continue;
                const std::string delta = decoder_->Push(tok);
                if (delta.empty()) continue;
                if (!text_block_open_) {
                    out += lsxnetwork::AnthropicContentBlockStartText(0);
                    text_block_open_ = true;
                }
                out += lsxnetwork::AnthropicContentBlockDeltaText(0, delta);
            }
            return true;
        }
        // kDone
        out.clear();
        const int ct = static_cast<int>(out_ids_.size());
        if (buffered_) {
            std::vector<lsxcommon::ToolCall> calls;
            if (buffer_tools_) framer_.ParseToolCalls(model_path_, out_ids_, calls);
            if (!calls.empty()) {
                for (std::size_t i = 0; i < calls.size(); ++i) {
                    out += lsxnetwork::AnthropicToolUseBlockStart(
                        static_cast<int>(i), calls[i].id, calls[i].name);
                    out += lsxnetwork::AnthropicToolUseDelta(
                        static_cast<int>(i), calls[i].arguments_json.empty()
                                                ? std::string("{}")
                                                : calls[i].arguments_json);
                    out += lsxnetwork::AnthropicContentBlockStop(
                        static_cast<int>(i));
                }
                out += lsxnetwork::AnthropicMessageDelta("tool_use", ct);
            } else {
                bool stop_hit = false;
                const std::string text = TruncateAtStop(
                    framer_.Decode(model_path_, out_ids_), stops_, stop_hit);
                out += lsxnetwork::AnthropicContentBlockStartText(0);
                out += lsxnetwork::AnthropicContentBlockDeltaText(0, text);
                out += lsxnetwork::AnthropicContentBlockStop(0);
                out += lsxnetwork::AnthropicMessageDelta(
                    stop_hit ? "stop_sequence"
                             : (f.eos_terminated ? "end_turn" : "max_tokens"),
                    ct);
            }
        } else {
            if (decoder_) {
                const std::string tail = decoder_->Flush();
                if (!tail.empty()) {
                    if (!text_block_open_) {
                        out += lsxnetwork::AnthropicContentBlockStartText(0);
                        text_block_open_ = true;
                    }
                    out += lsxnetwork::AnthropicContentBlockDeltaText(0, tail);
                }
            }
            if (text_block_open_)
                out += lsxnetwork::AnthropicContentBlockStop(0);
            out += lsxnetwork::AnthropicMessageDelta(
                f.eos_terminated ? "end_turn" : "max_tokens", ct);
        }
        out += lsxnetwork::AnthropicMessageStop();
        done_ = true;
        terminal_ = true;
        return true;
    }

    void Cancel() override {
        cancelled_ = true;
        if (submitted_ && request_id_ != 0)
            backend_.Client().Cancel(request_id_);
    }

private:
    std::string FinishError(const std::string& message) {
        done_ = true;
        return lsxnetwork::AnthropicEvent(
                   "error", lsxnetwork::BuildAnthropicError(message, "api_error")) +
               lsxnetwork::AnthropicMessageStop();
    }

    Backend& backend_;
    PromptFramer& framer_;
    std::string model_path_;
    std::string id_;
    std::string model_;
    std::vector<std::int32_t> input_ids_;
    int max_tokens_;
    lsxnetwork::RequestSampling sampling_;
    int timeout_ms_;
    bool buffer_tools_;
    bool buffered_ = false;
    std::vector<std::string> stops_;
    std::unique_ptr<std::unique_lock<std::mutex>> lock_;
    std::unique_ptr<StreamingDecoder> decoder_;
    std::vector<std::int32_t> out_ids_;
    std::uint32_t request_id_ = 0;
    int elapsed_ = 0;
    bool submitted_ = false;
    bool done_ = false;
    bool terminal_ = false;
    bool text_block_open_ = false;
    bool cancelled_ = false;
};

}  // namespace

ChatBridge::ChatBridge(BridgeConfig cfg, PromptFramer& framer)
    : cfg_(std::move(cfg)), framer_(framer) {}

ChatBridge::~ChatBridge() = default;

bool ChatBridge::Initialize(std::string& err) {
    models_.reserve(cfg_.models.size());
    for (const auto& mc : cfg_.models) {
        ModelRuntime rt;
        rt.cfg = mc;
        rt.backend = std::make_unique<Backend>(mc.backend, cfg_.queue_depth);
        if (rt.backend->Connect() != 0) {
            err = "cannot connect backend for model '" + mc.id + "' at " +
                  mc.backend.endpoint;
            return false;
        }
        models_.push_back(std::move(rt));
    }
    RegisterRoutes();
    return true;
}

ChatBridge::ModelRuntime* ChatBridge::Find(const std::string& id) {
    for (auto& m : models_)
        if (m.cfg.id == id) return &m;
    return nullptr;
}

ChatBridge::ModelRuntime* ChatBridge::ResolveModel(
    const std::string& requested, lsxnetwork::HttpResponse& err_out) {
    std::string id = requested;
    if (id.empty()) id = cfg_.default_model;
    if (id.empty() && !models_.empty()) id = models_.front().cfg.id;
    ModelRuntime* rt = Find(id);
    if (rt == nullptr) {
        err_out.status = 404;
        err_out.body = lsxnetwork::BuildError(
            "model '" + requested + "' not found", "model_not_found");
    }
    return rt;
}

void ChatBridge::RegisterRoutes() {
    using lsxnetwork::HttpRequest;
    using lsxnetwork::HttpResponse;

    server_.Get("/health", [](const HttpRequest&, HttpResponse& r) {
        r.status = 200;
        r.body = "{\"status\":\"ok\"}";
    });

    server_.Get("/ready", [this](const HttpRequest&, HttpResponse& r) {
        for (const auto& m : models_) {
            // Cheap when already healthy; actively re-probes after a failure or
            // before the first request, so a down backend reports 503 instead
            // of a sticky optimistic 200.
            if (!m.backend->Ready() && !m.backend->Probe(250)) {
                r.status = 503;
                r.body = "{\"status\":\"not_ready\"}";
                return;
            }
        }
        r.status = 200;
        r.body = "{\"status\":\"ready\"}";
    });

    server_.Get("/v1/models", [this](const HttpRequest& rq, HttpResponse& r) {
        std::vector<std::string> ids;
        ids.reserve(models_.size());
        for (const auto& m : models_) ids.push_back(m.cfg.id);
        r.status = 200;
        r.body = rq.headers.count("anthropic-version") != 0
                     ? lsxnetwork::BuildAnthropicModelsList(ids)
                     : lsxnetwork::BuildModelsList(ids);
    });

    // Claude Code connection-warming probe; rejectable without breaking.
    server_.Get("/api/hello", [](const HttpRequest&, HttpResponse& r) {
        r.status = 200;
        r.body = "{}";
    });

    server_.Get("/v1/models/.*", [this](const HttpRequest& rq, HttpResponse& r) {
        const std::string prefix = "/v1/models/";
        const std::string id =
            rq.path.size() > prefix.size() ? rq.path.substr(prefix.size())
                                           : std::string{};
        if (Find(id) == nullptr) {
            r.status = 404;
            r.body = lsxnetwork::BuildError("model '" + id + "' not found",
                                            "model_not_found");
            return;
        }
        r.status = 200;
        r.body = lsxnetwork::BuildModel(id);
    });

    server_.PostStream(
        "/v1/chat/completions",
        [this](const HttpRequest& rq,
               HttpResponse& head) -> std::unique_ptr<lsxnetwork::StreamSource> {
            head.content_type = "text/event-stream; charset=utf-8";
            lsxnetwork::ChatRequest req;
            simdjson::padded_string padded(rq.body);
            if (lsxnetwork::OpenAiEndpoint().ParseChatRequest(
                    padded.data(), padded.size(),
                    padded.size() + simdjson::SIMDJSON_PADDING, req) !=
                simdjson::SUCCESS) {
                head.status = 400;
                head.content_type = kJson;
                head.body = lsxnetwork::BuildError("invalid request body",
                                                   "invalid_request_error");
                return nullptr;
            }
            HttpResponse err;
            ModelRuntime* rt = ResolveModel(req.model, err);
            if (rt == nullptr) {
                head.status = err.status;
                head.content_type = kJson;
                head.body = err.body;
                return nullptr;
            }
            const lsxcommon::ChatConversation conv = BuildConversation(req);
            std::vector<std::int32_t> input_ids;
            if (!framer_.EncodeConversation(rt->cfg.model_path, conv, input_ids)) {
                head.status = 500;
                head.content_type = kJson;
                head.body = lsxnetwork::BuildError("prompt framing failed",
                                                   "server_error");
                return nullptr;
            }
            if (!rt->backend->TryAdmit()) {
                head.status = 429;
                head.content_type = kJson;
                head.body = lsxnetwork::BuildError("queue full",
                                                   "rate_limit_error");
                return nullptr;
            }
            const int max_tokens =
                req.max_tokens > 0 ? req.max_tokens : cfg_.default_max_tokens;
            if (req.stream) {
                return std::make_unique<ChatStreamSource>(
                    *rt->backend, framer_, rt->cfg.model_path, "chatcmpl-bridge",
                    rt->cfg.id, std::move(input_ids), max_tokens,
                    req.sampling, cfg_.request_timeout_ms, req.include_usage,
                    ToolsRequested(req), req.stop);
            }
            Admission admission{*rt->backend, true};
            std::vector<std::int32_t> out_ids;
            bool eos = false;
            std::string error;
            if (!RunBlocking(*rt->backend, input_ids, max_tokens,
                             req.sampling, cfg_.request_timeout_ms, out_ids,
                             eos, error)) {
                head.status = 502;
                head.content_type = kJson;
                head.body = lsxnetwork::BuildError(error, "server_error");
                return nullptr;
            }
            std::vector<lsxcommon::ToolCall> calls;
            framer_.ParseToolCalls(rt->cfg.model_path, out_ids, calls);
            head.status = 200;
            head.content_type = kJson;
            if (!calls.empty()) {
                head.body = lsxnetwork::BuildChatCompletionTools(
                    "chatcmpl-bridge", rt->cfg.id, ToolCallsWire(calls),
                    static_cast<int>(input_ids.size()),
                    static_cast<int>(out_ids.size()));
            } else {
                bool stop_hit = false;
                const std::string text = TruncateAtStop(
                    framer_.Decode(rt->cfg.model_path, out_ids), req.stop, stop_hit);
                head.body = lsxnetwork::BuildChatCompletion(
                    "chatcmpl-bridge", rt->cfg.id, text,
                    (stop_hit || eos) ? "stop" : "length",
                    static_cast<int>(input_ids.size()),
                    static_cast<int>(out_ids.size()));
            }
            return nullptr;
        });

    // ── Anthropic Messages ──────────────────────────────────────────────
    server_.Post("/v1/messages/count_tokens",
                 [this](const HttpRequest& rq, HttpResponse& r) {
        lsxnetwork::AnthropicRequest req;
        simdjson::padded_string padded(rq.body);
        if (lsxnetwork::ParseAnthropicRequest(
                padded.data(), padded.size(),
                padded.size() + simdjson::SIMDJSON_PADDING, req) !=
            simdjson::SUCCESS) {
            r.status = 400;
            r.body = lsxnetwork::BuildAnthropicError("invalid request body",
                                                     "invalid_request_error");
            return;
        }
        HttpResponse err;
        ModelRuntime* rt = ResolveModel(req.model, err);
        if (rt == nullptr) {
            r.status = err.status;
            r.body = lsxnetwork::BuildAnthropicError("model not found",
                                                     "not_found_error");
            return;
        }
        const lsxcommon::ChatConversation conv =
            BuildConversation(req.messages, req.tools);
        std::vector<std::int32_t> input_ids;
        if (!framer_.EncodeConversation(rt->cfg.model_path, conv, input_ids)) {
            r.status = 500;
            r.body = lsxnetwork::BuildAnthropicError("prompt framing failed",
                                                     "api_error");
            return;
        }
        r.status = 200;
        r.body = "{\"input_tokens\":" + std::to_string(input_ids.size()) + "}";
    });

    server_.PostStream(
        "/v1/messages",
        [this](const HttpRequest& rq,
               HttpResponse& head) -> std::unique_ptr<lsxnetwork::StreamSource> {
            head.content_type = "text/event-stream; charset=utf-8";
            lsxnetwork::AnthropicRequest req;
            simdjson::padded_string padded(rq.body);
            if (lsxnetwork::ParseAnthropicRequest(
                    padded.data(), padded.size(),
                    padded.size() + simdjson::SIMDJSON_PADDING, req) !=
                simdjson::SUCCESS) {
                head.status = 400;
                head.content_type = kJson;
                head.body = lsxnetwork::BuildAnthropicError(
                    "invalid request body", "invalid_request_error");
                return nullptr;
            }
            HttpResponse err;
            ModelRuntime* rt = ResolveModel(req.model, err);
            if (rt == nullptr) {
                head.status = 404;
                head.content_type = kJson;
                head.body = lsxnetwork::BuildAnthropicError("model not found",
                                                            "not_found_error");
                return nullptr;
            }
            const lsxcommon::ChatConversation conv =
                BuildConversation(req.messages, req.tools);
            std::vector<std::int32_t> input_ids;
            if (!framer_.EncodeConversation(rt->cfg.model_path, conv, input_ids)) {
                head.status = 500;
                head.content_type = kJson;
                head.body = lsxnetwork::BuildAnthropicError(
                    "prompt framing failed", "api_error");
                return nullptr;
            }
            if (!rt->backend->TryAdmit()) {
                head.status = 429;
                head.content_type = kJson;
                head.headers["retry-after"] = "1";
                head.body = lsxnetwork::BuildAnthropicError("queue full",
                                                            "overloaded_error");
                return nullptr;
            }
            const int max_tokens =
                req.max_tokens > 0 ? req.max_tokens : cfg_.default_max_tokens;
            const bool buffer_tools = !req.tools.empty() &&
                                      req.tool_choice != "none";
            if (req.stream) {
                return std::make_unique<AnthropicStreamSource>(
                    *rt->backend, framer_, rt->cfg.model_path, "msg-bridge",
                    rt->cfg.id, std::move(input_ids), max_tokens,
                    req.sampling, cfg_.request_timeout_ms, buffer_tools,
                    req.stop_sequences);
            }
            Admission admission{*rt->backend, true};
            std::vector<std::int32_t> out_ids;
            bool eos = false;
            std::string error;
            if (!RunBlocking(*rt->backend, input_ids, max_tokens,
                             req.sampling, cfg_.request_timeout_ms, out_ids,
                             eos, error)) {
                head.status = 502;
                head.content_type = kJson;
                head.body = lsxnetwork::BuildAnthropicError(error, "api_error");
                return nullptr;
            }
            std::vector<lsxcommon::ToolCall> calls;
            framer_.ParseToolCalls(rt->cfg.model_path, out_ids, calls);
            head.status = 200;
            head.content_type = kJson;
            if (!calls.empty()) {
                std::vector<lsxnetwork::ChatToolCall> wire;
                for (const auto& c : calls)
                    wire.push_back({c.id, c.name, c.arguments_json});
                head.body = lsxnetwork::BuildAnthropicToolMessage(
                    "msg-bridge", rt->cfg.id, wire, "tool_use",
                    static_cast<int>(input_ids.size()),
                    static_cast<int>(out_ids.size()));
            } else {
                bool stop_hit = false;
                const std::string text = TruncateAtStop(
                    framer_.Decode(rt->cfg.model_path, out_ids),
                    req.stop_sequences, stop_hit);
                head.body = lsxnetwork::BuildAnthropicMessage(
                    "msg-bridge", rt->cfg.id, text,
                    stop_hit ? "stop_sequence"
                             : (eos ? "end_turn" : "max_tokens"),
                    static_cast<int>(input_ids.size()),
                    static_cast<int>(out_ids.size()));
            }
            return nullptr;
        });

    server_.Post("/v1/completions", [this](const HttpRequest& rq,
                                           HttpResponse& r) {
        lsxnetwork::LegacyCompletionRequest req;
        simdjson::padded_string padded(rq.body);
        if (lsxnetwork::OpenAiEndpoint().ParseLegacyCompletionRequest(
                padded.data(), padded.size(),
                padded.size() + simdjson::SIMDJSON_PADDING, req) !=
            simdjson::SUCCESS) {
            r.status = 400;
            r.body = lsxnetwork::BuildError("invalid request body",
                                            "invalid_request_error");
            return;
        }
        ModelRuntime* rt = ResolveModel(req.model, r);
        if (rt == nullptr) return;
        std::vector<std::int32_t> input_ids;
        if (!framer_.Encode(rt->cfg.model_path, req.prompt, input_ids)) {
            r.status = 500;
            r.body = lsxnetwork::BuildError("prompt framing failed",
                                            "server_error");
            return;
        }
        if (!rt->backend->TryAdmit()) {
            r.status = 429;
            r.body = lsxnetwork::BuildError("queue full", "rate_limit_error");
            return;
        }
        Admission admission{*rt->backend, true};
        std::vector<std::int32_t> out_ids;
        bool eos = false;
        std::string error;
        const int max_tokens =
            req.max_tokens > 0 ? req.max_tokens : cfg_.default_max_tokens;
        if (!RunBlocking(*rt->backend, input_ids, max_tokens,
                         req.sampling, cfg_.request_timeout_ms, out_ids, eos,
                         error)) {
            r.status = 502;
            r.body = lsxnetwork::BuildError(error, "server_error");
            return;
        }
        const std::string text = framer_.Decode(rt->cfg.model_path, out_ids);
        r.status = 200;
        r.body = lsxnetwork::BuildTextCompletion(
            "cmpl-bridge", rt->cfg.id, text, eos ? "stop" : "length",
            static_cast<int>(input_ids.size()),
            static_cast<int>(out_ids.size()));
    });
}

bool ChatBridge::Listen() {
    std::atomic<bool> done{false};
    std::thread watcher;
    if (stop_flag_ != nullptr) {
        watcher = std::thread([this, &done] {
            while (!done.load()) {
                if (stop_flag_->load()) {
                    server_.Stop();
                    return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        });
    }
    const bool ok = server_.Listen(cfg_.host, cfg_.port);
    done.store(true);
    if (watcher.joinable()) watcher.join();
    return ok;
}

int ChatBridge::BoundPort() const { return server_.BoundPort(); }
void ChatBridge::Stop() { server_.Stop(); }

}  // namespace taskmosaic
