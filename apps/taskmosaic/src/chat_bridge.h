#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "backend.h"
#include "config.h"
#include "framer.h"
#include "lsxnetwork/http_server.h"

namespace taskmosaic {

/// OpenAI-compatible HTTP bridge: routes requests to per-model lsxfabric
/// backends, streams SSE, and cancels on client disconnect.
class ChatBridge {
public:
    ChatBridge(BridgeConfig cfg, PromptFramer& framer);
    ~ChatBridge();

    bool Initialize(std::string& err);
    bool Listen();  // blocks
    int BoundPort() const;
    void Stop();

    void SetStopFlag(std::atomic<bool>* flag) noexcept { stop_flag_ = flag; }

    const BridgeConfig& Config() const noexcept { return cfg_; }

private:
    struct ModelRuntime {
        ModelConfig cfg;
        std::unique_ptr<Backend> backend;
    };

    void RegisterRoutes();
    ModelRuntime* Find(const std::string& id);
    ModelRuntime* ResolveModel(const std::string& requested,
                               lsxnetwork::HttpResponse& err_out);

    BridgeConfig cfg_;
    PromptFramer& framer_;
    std::vector<ModelRuntime> models_;
    std::atomic<bool>* stop_flag_ = nullptr;
    lsxnetwork::HttpServer server_;
};

}  // namespace taskmosaic
