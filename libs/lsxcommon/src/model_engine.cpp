#include "lsxcommon/model_engine.h"

#include "lsxcommon/log.h"
#include "lsxcommon/model_factory.h"
#include "lsxcommon/resource_governor.h"

namespace lsxcommon {

InferenceEngine &InferenceEngine::Instance()
{
    static InferenceEngine instance;  // Meyers singleton
    return instance;
}

bool InferenceEngine::InitializeEngine()
{
    ResourceGovernor::Instance().Initialize(GovernorOptions{});
    initialized_.store(true, std::memory_order_release);
    log::debug("[model_engine] initialized");
    return true;
}

void InferenceEngine::ShutdownEngine()
{
    initialized_.store(false, std::memory_order_release);
    ResourceGovernor::Instance().Shutdown();
    log::debug("[model_engine] shutdown");
}

std::unique_ptr<IModel> InferenceEngine::CreateModel(const ModelInitConfig &cfg)
{
    if (!IsInitialized()) return nullptr;
    return LsxModelFactory::Create(cfg);
}

gpu::InferenceResult InferenceEngine::Run(IModel &model, ModelRequest &req)
{
    gpu::InferenceResult r;
    if (!IsInitialized()) {
        r.error_msg = "engine not initialized";
        return r;
    }
    if (req.abort_flag && req.abort_flag->load(std::memory_order_acquire)) {
        r.error_msg = "aborted by caller (abort_flag)";
        return r;
    }

    r = model.Infer(req);
    if (!r.ok) return r;

    // Replay the generated token stream into the caller's callback as
    // TokenChunk events (step-fanout is synchronous, on the calling thread).
    // The cooperative interruption intercept is honored between chunks so an
    // abort mid-decode truncates the published stream cleanly.
    TokenCallback &cb = req.token_callback;
    if (cb && !r.gen_tokens.empty()) {
        TokenChunk chunk;
        for (chunk.step_t = 0; chunk.step_t < r.gen_tokens.size(); ++chunk.step_t) {
            if (req.abort_flag &&
                req.abort_flag->load(std::memory_order_acquire))
                break;
            chunk.token_id = static_cast<uint32_t>(r.gen_tokens[chunk.step_t]);
            chunk.logit_maxv = 0.0f;  // per-step argmax is published by the
                                      // model's own streaming hooks when present
            chunk.is_special = false;
            cb(chunk);
        }
    }
    return r;
}

} // namespace lsxcommon